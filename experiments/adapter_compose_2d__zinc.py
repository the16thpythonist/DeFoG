"""
Validate frozen-base CFG-adapter STACKING: compose the independently-trained
logP adapter and TPSA adapter (product-of-experts on the rate matrices) and steer
to all 4 high/low combinations. Plot the generated molecules over the 2D logP x
TPSA density of the ZINC dataset -- success = the 4 clusters land in the correct
quadrants.

Sampling-only: loads the frozen base + two adapters (no retraining).

Usage:
    python experiments/adapter_compose_2d__zinc.py --__TESTING__ True
    python experiments/adapter_compose_2d__zinc.py \
        --LOGP_CKPT "'.../logp_adapter.ckpt'" --TPSA_CKPT "'.../tpsa_adapter.ckpt'"
"""
import os
import json
import itertools

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Draw
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import build_encoders, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, AdaLNAdapter, AdapterComposition, ConditionBranch, AdaptedSampler,
)

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
ATOM_TYPES: list = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")
LOGP_CKPT: str = ""             # path to logp_adapter.ckpt ("" -> build fresh, smoke only)
TPSA_CKPT: str = ""             # path to tpsa_adapter.ckpt

COMPOSE_MODE: str = "mean"      # recommended for N>1
WEIGHT: float = 2.0             # per-branch guidance weight
N_PER_COMBO: int = 200
EVAL_STEPS: int = 250
ETA: float = 5.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
EVAL_CHUNK: int = 40
TARGET_PERCENTILES: list = [5, 95]
REF_SUBSAMPLE: int = 20000      # dataset molecules for the 2D density background
GRID_N: int = 12

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


def compute_2props(smiles_iter):
    lp, tp = [], []
    for smi in smiles_iter:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi
        if m is None:
            continue
        try:
            lp.append(float(Crippen.MolLogP(m)))
            tp.append(float(Descriptors.TPSA(m)))
        except Exception:
            pass
    return np.asarray(lp), np.asarray(tp)


def decode_props(samples, atom_decoder, bond_decoder):
    mols, lp, tp = [], [], []
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi is not None and Chem.MolFromSmiles(smi) is not None:
            try:
                lp.append(float(Crippen.MolLogP(mol)))
                tp.append(float(Descriptors.TPSA(mol)))
                mols.append(mol)
            except Exception:
                pass
    return mols, np.asarray(lp), np.asarray(tp)


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log("ZINC adapter STACKING: compose logP + TPSA adapters, 2D high/low validation")
    import pytorch_lightning as pl
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(e.ATOM_TYPES, e.BOND_TYPES)

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0

    def _load_or_fresh(ckpt, prop):
        if ckpt:
            a = AdaLNAdapter.load(ckpt, device=device)
            a.check_compatible(base)
            e.log(f"loaded {prop} adapter from {ckpt}")
        else:
            a = AdaLNAdapter.for_base(base, cond_dim=1, hidden=32, cond_type=prop, name=f"{prop}_adapter").to(device)
            e.log(f"[fresh/untrained] {prop} adapter (smoke)")
        return a.eval()

    adapters = {"logp": _load_or_fresh(e.LOGP_CKPT, "logp"),
                "tpsa": _load_or_fresh(e.TPSA_CKPT, "tpsa")}

    # dataset 2D density + high/low targets
    df = pd.read_csv(e.CSV_PATH)
    ref_smiles = df[e.SMILES_COLUMN].sample(min(e.REF_SUBSAMPLE, len(df)), random_state=e.SEED).tolist()
    ds_lp, ds_tp = compute_2props(ref_smiles)
    tgt = {
        "logp": dict(zip(["low", "high"], [float(x) for x in np.percentile(ds_lp, e.TARGET_PERCENTILES)])),
        "tpsa": dict(zip(["low", "high"], [float(x) for x in np.percentile(ds_tp, e.TARGET_PERCENTILES)])),
    }
    e["eval/targets"] = tgt
    e.log(f"targets: {tgt}")

    colors = {("low", "low"): "#2c7fb8", ("low", "high"): "#31a354",
              ("high", "low"): "#d95f0e", ("high", "high"): "#756bb1"}
    results, gen = {"mode": e.COMPOSE_MODE, "weight": e.WEIGHT, "targets": tgt, "combos": {}}, {}

    for lp_lvl, tp_lvl in itertools.product(["low", "high"], ["low", "high"]):
        lp_t, tp_t = tgt["logp"][lp_lvl], tgt["tpsa"][tp_lvl]
        comp = AdapterComposition([
            ConditionBranch(adapters["logp"], torch.tensor([lp_t]), e.WEIGHT),
            ConditionBranch(adapters["tpsa"], torch.tensor([tp_t]), e.WEIGHT),
        ], base=base, mode=e.COMPOSE_MODE)
        samp = AdaptedSampler(base, comp, eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS,
                              time_distortion=e.TIME_DISTORTION)
        samples, rem = [], e.N_PER_COMBO
        while rem > 0:
            cur = min(e.EVAL_CHUNK, rem)
            samples += samp.sample(cur, device=device, show_progress=False)
            rem -= cur
        mols, glp, gtp = decode_props(samples, atom_decoder, bond_decoder)
        gen[(lp_lvl, tp_lvl)] = (glp, gtp)
        results["combos"][f"logp-{lp_lvl}_tpsa-{tp_lvl}"] = {
            "target_logp": lp_t, "target_tpsa": tp_t, "n_valid": len(mols),
            "mean_logp": float(glp.mean()) if glp.size else None,
            "mean_tpsa": float(gtp.mean()) if gtp.size else None,
            "mae_logp": float(np.mean(np.abs(glp - lp_t))) if glp.size else None,
            "mae_tpsa": float(np.mean(np.abs(gtp - tp_t))) if gtp.size else None,
        }
        e.log(f"[logp-{lp_lvl} tpsa-{tp_lvl}] target=({lp_t:.1f},{tp_t:.1f}) n={len(mols)} "
              f"mean=({glp.mean() if glp.size else float('nan'):.1f},{gtp.mean() if gtp.size else float('nan'):.1f})")
        if len(mols) > 0:
            Draw.MolsToGridImage(mols[:e.GRID_N], molsPerRow=4, subImgSize=(200, 200),
                                 legends=[f"lP{a:.1f} T{b:.0f}" for a, b in zip(glp[:e.GRID_N], gtp[:e.GRID_N])]
                                 ).save(os.path.join(e.path, f"grid_logp-{lp_lvl}_tpsa-{tp_lvl}.png"))
    e.commit_json("compose_2d_metrics.json", results)

    # -- the 2D plot: dataset density + 4 generated clusters + target crosshairs
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.hexbin(ds_lp, ds_tp, gridsize=45, cmap="Greys", bins="log", mincnt=1, zorder=1, alpha=0.9)
    for (lp_lvl, tp_lvl), (glp, gtp) in gen.items():
        c = colors[(lp_lvl, tp_lvl)]
        lp_t, tp_t = tgt["logp"][lp_lvl], tgt["tpsa"][tp_lvl]
        if glp.size:
            ax.scatter(glp, gtp, s=14, c=c, alpha=0.5, edgecolors="none", zorder=2,
                       label=f"logP-{lp_lvl}, TPSA-{tp_lvl}")
        ax.scatter([lp_t], [tp_t], marker="X", s=220, c=c, edgecolors="black", linewidths=1.5, zorder=4)
    ax.set_xlabel("logP (Crippen)"); ax.set_ylabel("TPSA")
    ax.set_title(f"Composed adapters (logP x TPSA) over ZINC density\n"
                 f"mode={e.COMPOSE_MODE}, w={e.WEIGHT}  (X = target, dots = generated)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(np.percentile(ds_lp, 0.5), np.percentile(ds_lp, 99.5))
    ax.set_ylim(max(0, np.percentile(ds_tp, 0.5)), np.percentile(ds_tp, 99.5))
    fig.tight_layout()
    e.commit_fig("compose_2d.png", fig)

    e.log("=" * 60)
    for k, v in results["combos"].items():
        e.log(f"{k}: target=({v['target_logp']:.1f},{v['target_tpsa']:.1f}) "
              f"mean=({v['mean_logp']},{v['mean_tpsa']}) n={v['n_valid']}")
    e.log("Done.")


@experiment.testing
def testing(e: Experiment):
    e.N_PER_COMBO = 8
    e.EVAL_CHUNK = 8
    e.EVAL_STEPS = 5
    e.REF_SUBSAMPLE = 400
    e.GRID_N = 4
    # LOGP_CKPT/TPSA_CKPT default "" -> fresh untrained adapters (mechanism smoke)


experiment.run_if_main()
