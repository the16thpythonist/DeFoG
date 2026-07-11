"""
FK-SMC with restart-based diversity: sample_until(target).

Run independent FK-SMC ensembles (K particles each). After each ensemble, keep
only the molecules that are UNIQUE WITHIN THAT RUN (drops the resampling clones),
and add them to the pool. Do NOT dedupe across runs -- a molecule recurring in an
independent run is legitimate density, not a duplicate. Repeat until the pool
reaches `target` (or `max_runs` is hit).

This gives genuinely diverse samples (fresh noise per run) at the cost of running
K particles to harvest only the ~unique-survivors per run.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from rdkit.Chem import Crippen

from defog.core import (
    DeFoGModel, ExactGuidance, FeynmanKacSampler,
    MoleculePropertyEnergy, ConditionalSizeDistribution,
)
from defog.domains import MoleculeDomain
from defog.domains.molecule import build_encoders, pyg_data_to_mol, mol_to_smiles
from experiments.guided_logp_demo import (
    derive_atom_types, build_dataset, save_grid, plot_distributions, BOND_TYPES, LEVELS,
)

RDLogger.DisableLog("rdApp.*")


@torch.no_grad()
def sample_until(fk, target, K, size_dist, tgt, device, ad, bd, max_runs, label):
    """Keep unique-within-run molecules from independent ensembles until `target`."""
    mols, logps = [], []
    runs = 0
    while len(mols) < target and runs < max_runs:
        runs += 1
        cond = torch.full((K, 1), float(tgt))
        seen = set()  # per-RUN dedupe only
        kept = 0
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
                kept += 1
            except Exception:
                pass
        print(f"[restart] {label} run {runs:2d}: {kept:3d} unique-in-run -> pool {len(mols)}/{target}", flush=True)
    return mols, np.array(logps), runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--guidance", default="experiments/_guided_logp_out/guided_logp_amortized.ckpt")
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--outdir", default="experiments/_guided_logp_smc_restart")
    ap.add_argument("--target", type=int, default=500, help="unique samples to collect per target level")
    ap.add_argument("--particles", type=int, default=128)
    ap.add_argument("--max-runs", type=int, default=30, help="safety cap on ensembles per target")
    ap.add_argument("--sample-steps", type=int, default=200)
    ap.add_argument("--eta", type=float, default=2.0)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--resample-interval", type=int, default=25)
    ap.add_argument("--ess-frac", type=float, default=0.5)
    ap.add_argument("--size-bandwidth", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    atom_types = derive_atom_types(df["smiles"])
    ae, ad, be, bd = build_encoders(atom_types, BOND_TYPES)
    logp_all = df["logp"].values.astype(float)
    prop_mean, prop_std = float(logp_all.mean()), float(logp_all.std())
    targets = dict(zip(LEVELS, np.percentile(logp_all, [10, 50, 90])))

    graphs, _, _ = build_dataset(df, ae, be)
    conds = torch.tensor([[float(g.prop_val)] for g in graphs])
    sizes = torch.tensor([int(g.x.size(0)) for g in graphs])
    size_dist = ConditionalSizeDistribution(conds, sizes, method="kernel", bandwidth=args.size_bandwidth)

    base = DeFoGModel.load(args.ckpt, device="cpu").to(device).eval()
    h = DeFoGModel.load(args.guidance, device="cpu").to(device).eval()
    domain = MoleculeDomain(ad, bd)
    print(f"[restart] target={args.target}/level K={args.particles} beta={args.beta} "
          f"ess_frac={args.ess_frac} steps={args.sample_steps} max_runs={args.max_runs}", flush=True)

    metrics, per_target = {}, {}
    for lvl in LEVELS:
        tgt = float(targets[lvl])
        energy_fn = MoleculePropertyEnergy(domain, Crippen.MolLogP, tgt)
        guidance = ExactGuidance(h, prop_mean=prop_mean, prop_std=prop_std,
                                 weight=args.weight).set_target(tgt)
        fk = FeynmanKacSampler(
            base, energy_fn, proposal_transform=guidance.reweight, beta=args.beta,
            resample_interval=args.resample_interval, ess_frac=args.ess_frac,
            eta=args.eta, omega=args.omega, sample_steps=args.sample_steps,
            time_distortion="polydec",
        )
        mols, logps, runs = sample_until(fk, args.target, args.particles, size_dist,
                                         tgt, device, ad, bd, args.max_runs, lvl)
        per_target[lvl] = logps
        n = len(logps)
        n_atoms = np.array([m.GetNumHeavyAtoms() for m in mols]) if mols else np.array([])
        metrics[lvl] = {
            "target": tgt, "n_unique": n, "runs": runs,
            "achieved_mean_logp": float(np.mean(logps)) if n else float("nan"),
            "median_logp": float(np.median(logps)) if n else float("nan"),
            "logp_mae": float(np.mean(np.abs(logps - tgt))) if n else float("nan"),
            "size_std": float(n_atoms.std()) if n else float("nan"),
            "n_distinct_sizes": int(len(set(n_atoms.tolist()))) if n else 0,
        }
        m = metrics[lvl]
        print(f"[restart] {lvl}: {n} unique in {runs} runs | mean={m['achieved_mean_logp']:.2f} "
              f"median={m['median_logp']:.2f} MAE={m['logp_mae']:.2f} size_std={m['size_std']:.1f} "
              f"distinct_sizes={m['n_distinct_sizes']}", flush=True)
        if mols:
            save_grid(mols, logps, tgt, os.path.join(args.outdir, f"guided_logp_grid_{lvl}.png"))

    plot_distributions(logp_all, per_target, targets,
                       os.path.join(args.outdir, "guided_logp_distributions.png"))
    with open(os.path.join(args.outdir, "guided_logp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[restart] DONE -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
