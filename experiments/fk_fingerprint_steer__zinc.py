"""
Feynman-Kac (SMC) steering toward a target Morgan fingerprint on ZINC 250k,
combined WITH the trained latent guidance adapter.

Motivation
----------
The per-coordinate exact-guidance adapter (``fingerprint_guidance__zinc.py``)
barely steered toward a target fingerprint (holistic reward -> nearly-flat
per-coordinate guidance field). Feynman-Kac steering instead scores WHOLE
predicted-clean molecules by a global reward and resamples trajectories, which
does not require the reward to factorize. Per prior experience FK *alone* tends
to under-steer / reward-hack, so here we test the combination the framework is
built for: FK resampling on top of the guided proposal
(``FeynmanKacSampler(proposal_transform=guidance.reweight)``).

This is a SAMPLING-ONLY experiment: it LOADS the already-trained adapter (no
retraining), so it starts immediately.

Ablation, per held-out target fingerprint (mean Tanimoto of generated -> target):
    baseline      : unconditional base
    guidance      : per-coordinate adapter only (GuidedSampler)
    fk            : FK-SMC on the bare base proposal (energy = 1 - Tanimoto)
    fk+guidance   : FK-SMC on the GUIDED proposal  <- the hypothesis

Usage:
    python experiments/fk_fingerprint_steer__zinc.py --__TESTING__ True
    python experiments/fk_fingerprint_steer__zinc.py --BETA 10 --GUIDE_W 3
"""
import os
import sys
import json
import random

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Draw
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import build_encoders, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, ExactGuidance, GuidedSampler, Sampler, FeynmanKacSampler,
)
from defog.core.data import dense_to_pyg

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")
ADAPTER_CKPT: str = os.path.join(_PROJECT_DIR, "ckpts", "fp_guidance_26407.ckpt")
ADAPTER_STATS: str = os.path.join(_PROJECT_DIR, "ckpts", "fp_guidance_26407_stats.json")

# --- What to compare ---
METHODS: list = ["guidance", "fk", "fk+guidance"]  # baseline always computed
N_TARGETS: int = 6
N_PER_TARGET: int = 48       # for FK this is the particle count K per population
N_BASELINE: int = 128
EVAL_CHUNK: int = 48         # == N_PER_TARGET so FK runs one population per target

# --- Sampling ---
STEPS: int = 200
ETA: float = 5.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"

# --- Guidance (proposal) ---
GUIDE_W: float = 3.0         # adapter guidance weight in the proposal

# --- Feynman-Kac ---
BETA: float = 10.0           # reward tilt; energy = 1 - Tanimoto in [0,1]
WARMUP_FRAC: float = 0.7     # resample only late, once clean predictions are shaped
RESAMPLE_INTERVAL: int = 0   # 0 -> None -> steps // 8
ESS_FRAC: float = 0.5        # adaptive resampling threshold

GRID_N: int = 24
GRID_METHOD: str = "fk+guidance"
SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ============================================================================
# Helpers
# ============================================================================
def mol_morgan_bits(mol, radius, n_bits) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def tanimoto_to_target(fp_mat: np.ndarray, target: np.ndarray) -> np.ndarray:
    if fp_mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    inter = fp_mat @ target
    union = fp_mat.sum(1) + target.sum() - inter
    return inter / np.clip(union, 1e-8, None)


def decode_and_fp(samples, atom_decoder, bond_decoder, radius, n_bits):
    mols, smis = [], []
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi is not None and Chem.MolFromSmiles(smi) is not None:
            mols.append(mol)
            smis.append(smi)
    fp = np.stack([mol_morgan_bits(m, radius, n_bits) for m in mols]) if mols else \
        np.zeros((0, n_bits), dtype=np.float32)
    return mols, smis, fp


class FingerprintTanimotoEnergy:
    """FK energy = ``1 - Tanimoto(fp(x1), target)`` on the predicted CLEAN graph.

    Decodes each dense one-hot graph to an RDKit mol, computes its Morgan FP and
    the Tanimoto to a fixed target FP; invalid/undecodable graphs get ``invalid``
    (default 1.0 = worst) so FK culls them. Lower energy = closer to target."""

    def __init__(self, atom_decoder, bond_decoder, target_fp, radius, n_bits, invalid=1.0):
        self.atom_decoder, self.bond_decoder = atom_decoder, bond_decoder
        self.target = np.asarray(target_fp, dtype=np.float32)
        self.tsum = float(self.target.sum())
        self.radius, self.n_bits, self.invalid = radius, n_bits, invalid

    def __call__(self, X1, E1, node_mask):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), float(self.invalid))
        for i, d in enumerate(datas):
            mol = pyg_data_to_mol(d, self.atom_decoder, self.bond_decoder)
            if mol is None:
                continue
            try:
                fp = mol_morgan_bits(mol, self.radius, self.n_bits)
                inter = float(fp @ self.target)
                union = float(fp.sum()) + self.tsum - inter
                out[i] = 1.0 - (inter / union if union > 0 else 0.0)
            except Exception:
                pass
        return out


def chunked_sample(sampler, n, chunk, device):
    out, remaining = [], n
    while remaining > 0:
        cur = min(chunk, remaining)
        out += sampler.sample(cur, device=device, show_progress=False)
        remaining -= cur
    return out


def stats_block(fp_mat, smis, target_fp):
    sims = tanimoto_to_target(fp_mat, target_fp)
    n_valid = fp_mat.shape[0]
    n_unique = len(set(smis))
    return {
        "n_valid": int(n_valid), "n_unique": int(n_unique),
        "uniqueness": (n_unique / n_valid) if n_valid else 0.0,
        "mean_tanimoto": float(sims.mean()) if sims.size else None,
        "median_tanimoto": float(np.median(sims)) if sims.size else None,
        "max_tanimoto": float(sims.max()) if sims.size else None,
    }, sims


# ============================================================================
# Experiment
# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log("FK-SMC + adapter guidance: fingerprint steering on ZINC")
    import pytorch_lightning as pl
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = json.load(open(e.ADAPTER_STATS))
    fp_bits, fp_radius = stats["fp_bits"], stats["fp_radius"]
    atom_types = stats["atom_types"]
    cond_mean = np.asarray(stats["cond_mean"], dtype=np.float32)
    cond_std = np.asarray(stats["cond_std"], dtype=np.float32)
    e.log(f"adapter stats: {fp_bits}-bit r{fp_radius}, atoms={atom_types}")
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, e.BOND_TYPES)

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    h = DeFoGModel.load(e.ADAPTER_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0 and h.cond_dim == fp_bits, (base.cond_dim, h.cond_dim)
    guidance = ExactGuidance(h, prop_mean=cond_mean, prop_std=cond_std, weight=e.GUIDE_W)
    e.log(f"loaded base ({sum(p.numel() for p in base.parameters()):,}) + adapter "
          f"({sum(p.numel() for p in h.parameters()):,}, cond_dim={h.cond_dim})")

    # -- target molecules + their fingerprints ------------------------------
    df = pd.read_csv(e.CSV_PATH)
    pool = random.sample(df[e.SMILES_COLUMN].tolist(), min(len(df), e.N_TARGETS * 5))
    tgt_smiles, tgt_mols, tgt_fps = [], [], []
    for s in pool:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        tgt_smiles.append(s); tgt_mols.append(m)
        tgt_fps.append(mol_morgan_bits(m, fp_radius, fp_bits))
        if len(tgt_smiles) >= e.N_TARGETS:
            break
    e.log(f"{e.N_TARGETS} target molecules: {tgt_smiles}")

    ri = e.RESAMPLE_INTERVAL if e.RESAMPLE_INTERVAL > 0 else None

    def build_sampler(method, target_fp):
        energy = FingerprintTanimotoEnergy(atom_decoder, bond_decoder, target_fp, fp_radius, fp_bits)
        if method == "guidance":
            guidance.set_weight(e.GUIDE_W).set_target(target_fp)
            return GuidedSampler(base, guidance, eta=e.ETA, omega=e.OMEGA,
                                 sample_steps=e.STEPS, time_distortion=e.TIME_DISTORTION)
        if method == "fk":
            return FeynmanKacSampler(base, energy, beta=e.BETA, warmup_frac=e.WARMUP_FRAC,
                                     resample_interval=ri, ess_frac=e.ESS_FRAC, proposal_transform=None,
                                     eta=e.ETA, omega=e.OMEGA, sample_steps=e.STEPS,
                                     time_distortion=e.TIME_DISTORTION)
        if method == "fk+guidance":
            guidance.set_weight(e.GUIDE_W).set_target(target_fp)
            return FeynmanKacSampler(base, energy, beta=e.BETA, warmup_frac=e.WARMUP_FRAC,
                                     resample_interval=ri, ess_frac=e.ESS_FRAC,
                                     proposal_transform=guidance.reweight,
                                     eta=e.ETA, omega=e.OMEGA, sample_steps=e.STEPS,
                                     time_distortion=e.TIME_DISTORTION)
        raise ValueError(method)

    # -- unconditional baseline pool (shared) -------------------------------
    e.log(f"sampling {e.N_BASELINE} unconditional baseline molecules ...")
    base_sampler = Sampler(base, eta=e.ETA, omega=e.OMEGA, sample_steps=e.STEPS,
                           time_distortion=e.TIME_DISTORTION)
    _, base_smis, base_fp = decode_and_fp(
        chunked_sample(base_sampler, e.N_BASELINE, e.EVAL_CHUNK, device),
        atom_decoder, bond_decoder, fp_radius, fp_bits)
    e.log(f"baseline valid: {base_fp.shape[0]}/{e.N_BASELINE}")

    # -- run every method per target ----------------------------------------
    results = {m: [] for m in ["baseline"] + e.METHODS}
    agg_sims = {m: [] for m in ["baseline"] + e.METHODS}
    grids = {}
    for ti, (tsmi, tfp, tmol) in enumerate(zip(tgt_smiles, tgt_fps, tgt_mols)):
        base_blk, base_sims = stats_block(base_fp, base_smis, tfp)
        results["baseline"].append({"target": tsmi, **base_blk})
        agg_sims["baseline"].extend(base_sims.tolist())
        e.log(f"[target {ti+1}/{e.N_TARGETS}] {tsmi}  baseline<T>={base_blk['mean_tanimoto']}")
        for method in e.METHODS:
            sampler = build_sampler(method, tfp)
            mols, smis, gfp = decode_and_fp(
                chunked_sample(sampler, e.N_PER_TARGET, e.EVAL_CHUNK, device),
                atom_decoder, bond_decoder, fp_radius, fp_bits)
            blk, sims = stats_block(gfp, smis, tfp)
            results[method].append({"target": tsmi, **blk})
            agg_sims[method].extend(sims.tolist())
            e.log(f"    {method:12s}: valid={blk['n_valid']} uniq={blk['n_unique']} "
                  f"<T>={blk['mean_tanimoto']}  maxT={blk['max_tanimoto']}")
            if method == e.GRID_METHOD and len(mols) > 0:
                order = np.argsort(-sims)[:e.GRID_N]
                gm = [tmol] + [mols[j] for j in order]
                leg = ["TARGET"] + [f"T={sims[j]:.2f}" for j in order]
                Draw.MolsToGridImage(gm, molsPerRow=5, subImgSize=(220, 220),
                                     legends=leg).save(os.path.join(e.path, f"grid_target{ti}.png"))

    # -- aggregate + artifacts ----------------------------------------------
    base_mean = float(np.mean(agg_sims["baseline"])) if agg_sims["baseline"] else float("nan")
    summary = {"methods": ["baseline"] + e.METHODS, "n_targets": e.N_TARGETS,
               "beta": e.BETA, "guide_w": e.GUIDE_W, "steps": e.STEPS,
               "per_target": results, "aggregate": {}}
    for m in ["baseline"] + e.METHODS:
        mean = float(np.mean(agg_sims[m])) if agg_sims[m] else float("nan")
        summary["aggregate"][m] = {"mean_tanimoto": mean, "lift_over_baseline": mean - base_mean}
    e.commit_json("fk_fingerprint_metrics.json", summary)

    # bar chart of mean Tanimoto per method
    fig, ax = plt.subplots(figsize=(8, 5))
    ms = ["baseline"] + e.METHODS
    means = [summary["aggregate"][m]["mean_tanimoto"] for m in ms]
    bars = ax.bar(ms, means, color=["0.6", "#4c72b0", "#dd8452", "#55a868"][:len(ms)])
    ax.axhline(base_mean, ls="--", color="0.5", lw=1)
    for b, mn in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, mn, f"{mn:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean Tanimoto to target")
    ax.set_title(f"Fingerprint steering: baseline vs guidance vs FK vs FK+guidance\n"
                 f"(beta={e.BETA}, guide_w={e.GUIDE_W}, {e.N_TARGETS} targets)")
    fig.tight_layout()
    e.commit_fig("method_comparison.png", fig)

    # distribution overlay
    fig2, ax2 = plt.subplots(figsize=(9, 5.2))
    bins = np.linspace(0, 1, 41)
    colors = {"baseline": "0.6", "guidance": "#4c72b0", "fk": "#dd8452", "fk+guidance": "#55a868"}
    for m in ms:
        if agg_sims[m]:
            ax2.hist(agg_sims[m], bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                     color=colors.get(m, None), label=f"{m} (<T>={np.mean(agg_sims[m]):.3f})")
    ax2.set_xlabel("Tanimoto to target"); ax2.set_ylabel("density")
    ax2.set_title("Tanimoto-to-target distributions by method")
    ax2.legend(fontsize=9); fig2.tight_layout()
    e.commit_fig("tanimoto_distributions.png", fig2)

    e.log("=" * 60)
    for m in ms:
        a = summary["aggregate"][m]
        e.log(f"{m:12s} <T>={a['mean_tanimoto']:.3f}  lift={a['lift_over_baseline']:+.3f}")
    e.log("Done.")


@experiment.testing
def testing(e: Experiment):
    e.N_TARGETS = 2
    e.N_PER_TARGET = 6
    e.N_BASELINE = 8
    e.EVAL_CHUNK = 6
    e.STEPS = 6
    e.WARMUP_FRAC = 0.4
    e.GRID_N = 4


experiment.run_if_main()
