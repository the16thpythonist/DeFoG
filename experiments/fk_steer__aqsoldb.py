"""
FK-SMC logP steering on AqSolDB -- a SINGLE (BETA, WARMUP_FRAC) configuration.

This experiment is ONE run of one config. It evaluates that config across the 3
target levels (low/med/high = dataset p10/p50/p90) using a FIXED compute budget of
RUNS independent FK-SMC ensembles (guided proposal + reward resampling, late
resampling, per-run dedupe), and reports per target:

  * mean-to-target bias  (mean(logP) - target)   -- how well the mean matches
  * logP MAE             (mean |logP - target|)  -- overall tightness
  * n_unique             (distinct canonical SMILES over the fixed budget)
    -- diversity: with a fixed budget, a collapsing config yields few unique.

The SWEEP is NOT done here: it is produced by SUBMITTING MANY of these runs to
SLURM, one per grid point, e.g.
  beta   in {1.0, 1.5, 2.0, 3.0, 4.0}
  warmup in {0.6, 0.7, 0.8, 0.9}
then aggregating the per-run archives afterwards (Pareto over bias / MAE / diversity).

Usage (one config):
    python experiments/fk_steer__aqsoldb.py --BETA 2.0 --WARMUP_FRAC 0.8
    python experiments/fk_steer__aqsoldb.py --__TESTING__ True
"""
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from rdkit.Chem import Crippen
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog.core import (
    DeFoGModel, ExactGuidance, FeynmanKacSampler,
    MoleculePropertyEnergy, ConditionalSizeDistribution,
)
from defog.domains import MoleculeDomain
from defog.domains.molecule import build_encoders, pyg_data_to_mol, mol_to_smiles
from experiments.guided_logp_demo import (
    derive_atom_types, build_dataset, save_grid, plot_distributions,
    plot_size_distributions, save_generated_json, BOND_TYPES, LEVELS,
)

RDLogger.DisableLog("rdApp.*")

# == configuration ==========================================================
CKPT_PATH: str = os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt")
GUIDANCE_PATH: str = "experiments/_guided_logp_out/guided_logp_amortized.ckpt"
CSV_PATH: str = "data/aqsoldb_conditional.csv"
SEED: int = 0

# --- swept knobs ---
# :param BETA: FK reward tilt strength r ∝ exp(-BETA * energy).
BETA: float = 1.5
# :param WARMUP_FRAC: fraction of steps before the first resample (resample LATE
#     to avoid premature culling of large molecules / size collapse).
WARMUP_FRAC: float = 0.8

# --- fixed compute budget (so n_unique is a fair diversity signal) ---
# :param PARTICLES: K particles per SMC ensemble.
PARTICLES: int = 128
# :param RUNS: independent ensembles per target (fixed budget = RUNS*PARTICLES).
RUNS: int = 4
# :param SAMPLE_STEPS / ETA / OMEGA / WEIGHT / RESAMPLE_INTERVAL / SIZE_BANDWIDTH:
SAMPLE_STEPS: int = 200
ETA: float = 2.0
OMEGA: float = 0.3
WEIGHT: float = 2.0
RESAMPLE_INTERVAL: int = 15
SIZE_BANDWIDTH: float = 0.5

TARGET_PERCENTILES: list = [10, 50, 90]
LEVEL_NAMES: list = ["low", "med", "high"]

# NOTE: __DEBUG__ must be False when submitting the sweep -- in debug mode pycomex
# writes to a single overwriteable 'debug' folder, so parallel/serial runs would
# clobber each other's results. False -> unique timestamped archive per run.
__DEBUG__: bool = False
__TESTING__: bool = False


@torch.no_grad()
def collect_fixed(fk, runs, K, size_dist, tgt, device, ad, bd):
    """RUNS ensembles, per-run dedupe (keep clones out), pool. Returns
    (mols, logps, smiles, sizes) -- cross-run repeats kept (density),
    n_unique computed as distinct SMILES over the pool."""
    mols, logps, smis, sizes = [], [], [], []
    for _ in range(runs):
        cond = torch.full((K, 1), float(tgt))
        seen = set()
        for s in fk.sample(K, size_dist=size_dist, condition=cond, device=device, show_progress=False):
            mol = pyg_data_to_mol(s, ad, bd)
            if mol is None:
                continue
            smi = mol_to_smiles(mol)
            if smi is None or smi in seen:
                continue
            seen.add(smi)
            try:
                logps.append(float(Crippen.MolLogP(mol)))
                mols.append(mol)
                smis.append(smi)
                sizes.append(int(mol.GetNumHeavyAtoms()))
            except Exception:
                pass
    return mols, np.array(logps), smis, np.array(sizes)


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"FK-SMC sweep config: BETA={e.BETA} WARMUP_FRAC={e.WARMUP_FRAC} "
          f"RUNS={e.RUNS} K={e.PARTICLES} steps={e.SAMPLE_STEPS}")
    e["config/BETA"] = e.BETA
    e["config/WARMUP_FRAC"] = e.WARMUP_FRAC
    from pytorch_lightning import seed_everything
    seed_everything(e.SEED, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(e.CSV_PATH)
    atom_types = derive_atom_types(df["smiles"])
    ae, ad, be, bd = build_encoders(atom_types, BOND_TYPES)
    logp_all = df["logp"].values.astype(float)
    prop_mean, prop_std = float(logp_all.mean()), float(logp_all.std())
    targets = dict(zip(e.LEVEL_NAMES, np.percentile(logp_all, e.TARGET_PERCENTILES)))

    graphs, _, _ = build_dataset(df, ae, be)
    conds = torch.tensor([[float(g.prop_val)] for g in graphs])
    sizes = torch.tensor([int(g.x.size(0)) for g in graphs])
    size_dist = ConditionalSizeDistribution(conds, sizes, method="kernel", bandwidth=e.SIZE_BANDWIDTH)

    base = DeFoGModel.load(e.CKPT_PATH, device="cpu").to(device).eval()
    h = DeFoGModel.load(e.GUIDANCE_PATH, device="cpu").to(device).eval()
    domain = MoleculeDomain(ad, bd)

    abs_bias, maes, uniques, generated = [], [], [], {}
    per_target_logps, per_target_mols = {}, {}
    for lvl in e.LEVEL_NAMES:
        tgt = float(targets[lvl])
        energy_fn = MoleculePropertyEnergy(domain, Crippen.MolLogP, tgt)
        guidance = ExactGuidance(h, prop_mean=prop_mean, prop_std=prop_std,
                                 weight=e.WEIGHT).set_target(tgt)
        fk = FeynmanKacSampler(
            base, energy_fn, proposal_transform=guidance.reweight, beta=e.BETA,
            resample_interval=e.RESAMPLE_INTERVAL, warmup_frac=e.WARMUP_FRAC,
            eta=e.ETA, omega=e.OMEGA, sample_steps=e.SAMPLE_STEPS, time_distortion="polydec",
        )
        mols, logps, smis, szs = collect_fixed(fk, e.RUNS, e.PARTICLES, size_dist, tgt, device, ad, bd)
        per_target_logps[lvl] = logps
        per_target_mols[lvl] = mols
        n_pool = len(logps)
        n_unique = len(set(smis))
        mean = float(np.mean(logps)) if n_pool else float("nan")
        bias = float(mean - tgt) if n_pool else float("nan")
        mae = float(np.mean(np.abs(logps - tgt))) if n_pool else float("nan")
        res = {
            "target": tgt, "n_pool": n_pool, "n_unique": n_unique,
            "mean": mean, "median": float(np.median(logps)) if n_pool else float("nan"),
            "bias": bias, "mae": mae,
            "size_mean": float(szs.mean()) if n_pool else float("nan"),
            "size_std": float(szs.std()) if n_pool else float("nan"),
        }
        e[f"results/{lvl}"] = res
        generated[lvl] = [{"smiles": s, "logp": float(l), "n_atoms": int(z)}
                          for s, l, z in zip(smis, logps, szs)]
        abs_bias.append(abs(bias)); maes.append(mae); uniques.append(n_unique)
        e.log(f"  {lvl}: target={tgt:.2f} mean={mean:.2f} bias={bias:+.2f} MAE={mae:.2f} "
              f"n_unique={n_unique}/{n_pool} size={res['size_mean']:.1f}")
        if mols:
            save_grid(mols, logps, tgt, os.path.join(e.path, f"grid_{lvl}.png"))

    # ---- artifacts: distribution plot, size plot (vs prior), SMILES ----
    plot_distributions(logp_all, per_target_logps, targets,
                       os.path.join(e.path, "logp_distributions.png"))
    proposed = {lvl: size_dist.sample(4000, condition=torch.full((4000, 1), float(targets[lvl]))).cpu().numpy()
                for lvl in e.LEVEL_NAMES}
    plot_size_distributions(per_target_mols, targets,
                            os.path.join(e.path, "size_distributions.png"), proposed=proposed)

    summary = {
        "BETA": e.BETA, "WARMUP_FRAC": e.WARMUP_FRAC,
        "avg_abs_bias": float(np.mean(abs_bias)),
        "avg_mae": float(np.mean(maes)),
        "total_unique": int(np.sum(uniques)),
        "min_unique": int(np.min(uniques)),
    }
    e["results/summary"] = summary
    e.commit_json("summary.json", summary)
    e.commit_json("generated_smiles.json", generated)
    e.log(f"SUMMARY BETA={e.BETA} WARMUP={e.WARMUP_FRAC} | avg|bias|={summary['avg_abs_bias']:.3f} "
          f"avg_MAE={summary['avg_mae']:.3f} total_unique={summary['total_unique']} "
          f"min_unique={summary['min_unique']}")


@experiment.testing
def testing(e: Experiment) -> None:
    e.PARTICLES = 16
    e.RUNS = 1
    e.SAMPLE_STEPS = 40


experiment.run_if_main()
