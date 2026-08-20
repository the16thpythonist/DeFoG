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

from experiments.utils import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, AdaLNAdapter, AdapterComposition, ConditionBranch, AdaptedSampler,
    ConditionalSizeDistribution, FeynmanKacSampler, PropertyHead, LearnedPropertyEnergy,
)
from defog.domains import MoleculeDomain

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

# Single-property eval found w=1 (exact conditional) is the accurate sweet spot;
# w>1 overshoots. For a 2-branch joint, PRODUCT mode at w=1 each = the principled
# joint conditional p(x|logP,TPSA) (sum of log-ratios); mean-mode would under-steer.
COMPOSE_MODE: str = "product"
WEIGHT: float = 1.0             # per-branch guidance weight
N_PER_COMBO: int = 200
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
EVAL_CHUNK: int = 40
TARGET_PERCENTILES: list = [5, 95]
REF_SUBSAMPLE: int = 20000      # dataset molecules for the 2D density background
GRID_N: int = 12

# Background style for the dataset density. "contour" = smooth 2D gaussian KDE drawn
# as filled greys + thin iso-density lines; "hexbin" = the original log-binned hexagons.
# The hexagons only read as a density when the dataset fills every cell -- on smaller
# reference samples they look like scattered blobs, so contour is the default.
BACKGROUND: str = "contour"     # "contour" | "hexbin"
KDE_SUBSAMPLE: int = 8000       # molecules used to fit the background KDE (cost is O(n*grid^2))
KDE_GRID: int = 160             # background KDE evaluated on a KDE_GRID x KDE_GRID mesh
KDE_LEVELS: int = 12            # number of filled bands / iso-density lines

# Each conditioning also gets its own KDE drawn as same-colour iso-density lines
# beneath its scatter points -- the clusters overlap heavily in the middle band, and
# the rings show where each one actually concentrates.
CONDITION_KDE: bool = True
CONDITION_KDE_LEVELS: int = 4
CONDITION_KDE_ALPHA: float = 0.55
# Outermost ring as a fraction of that cluster's peak density. Low values (~0.15) drag
# a sprawling outer contour across the whole panel and the four clusters crisscross;
# 0.3 keeps each set of rings around its own core.
CONDITION_KDE_FLOOR: float = 0.30

# Graph size is drawn from a property-CONDITIONAL size distribution: logP and TPSA
# are strongly size-dependent, so a fixed global size prior caps the reachable
# range (esp. the high end). Keyed on normalized [logP, TPSA]; passed to
# AdaptedSampler.sample(condition=...) which drives sizing independently of the
# adapter modulation. Matches the direct-CFG conditional recipe.
USE_CONDITIONAL_SIZE: bool = True
SIZE_DIST_METHOD: str = "kernel"
SIZE_FIT_N: int = 20000         # molecules used to fit the conditional size distribution

# -- Feynman-Kac (SMC) refinement on top of the adapter composition ------------
# The adapters PROPOSE (product-of-experts conditioning) and FK reward-tilts the
# particle population + resamples it, scoring each particle's adapter-conditioned
# predicted-clean graph. Reward = the trained property HEADS (the same surrogate the
# head-RL adapters were tuned against), one per property.
USE_FK: bool = False
FK_PARTICLES: int = 16          # K: the resampling pool AND the GPU batch (VRAM-bound)
FK_BETA: float = 2.5            # tilt strength; FK potential = -beta * energy
FK_WARMUP_FRAC: float = 0.5     # fraction of steps before the FIRST resample
FK_RESAMPLE_INTERVAL: int = 0   # 0 -> the sampler's own default (sample_steps // 8)
FK_ESS_FRAC: float = 0.0        # 0 -> resample at every checkpoint (no adaptive gate)
LOGP_HEAD_CKPT: str = os.path.join(_PROJECT_DIR, "ckpts", "logp_head.ckpt")
TPSA_HEAD_CKPT: str = os.path.join(_PROJECT_DIR, "ckpts", "tpsa_head.ckpt")

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


class JointLearnedEnergy:
    """FK reward over SEVERAL properties, each scored by a trained PropertyHead.

    Neither stock energy covers this case: ``LearnedPropertyEnergy`` is single-property
    and ``MultiPropertyEnergy`` only takes RDKit callbacks, so a *learned* joint
    logP x TPSA reward has to be composed here.

    Each part's squared error is divided by that property's variance, which is what
    lets logP (O(1)) and TPSA (O(100)) contribute comparably -- the same normalization
    ``MultiPropertyEnergy`` applies. The caller must scale each part's
    ``invalid_energy`` by the same factor, otherwise an undecodable graph would be
    penalized ~500x more softly on TPSA than on logP and FK would happily keep junk
    that happens to score well on the wide-ranged property.
    """

    def __init__(self, parts, scales, weights=None):
        self.parts = list(parts)
        self.scales = [float(s) for s in scales]
        self.weights = list(weights) if weights is not None else [1.0] * len(self.parts)

    def __call__(self, X1, E1, node_mask):
        total = None
        for part, scale, w in zip(self.parts, self.scales, self.weights):
            term = (w / scale ** 2) * part(X1, E1, node_mask)
            total = term if total is None else total + term
        return total


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

    # dataset 2D density, high/low targets, and a property-CONDITIONAL size prior.
    # Build graphs + aligned RDKit logP/TPSA (matching how the adapters were trained).
    df = pd.read_csv(e.CSV_PATH)
    ref_smiles = df[e.SMILES_COLUMN].sample(min(e.REF_SUBSAMPLE, len(df)), random_state=e.SEED).tolist()
    fit_graphs, lp_list, tp_list = [], [], []
    for smi in ref_smiles[:e.SIZE_FIT_N]:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        g = smiles_to_pyg_data(smi, atom_encoder, bond_encoder)
        if g is None:
            continue
        try:
            lp, tp = float(Crippen.MolLogP(m)), float(Descriptors.TPSA(m))
        except Exception:
            continue
        fit_graphs.append(g); lp_list.append(lp); tp_list.append(tp)
    ds_lp, ds_tp = np.asarray(lp_list), np.asarray(tp_list)
    lp_mu, lp_sd = float(ds_lp.mean()), float(ds_lp.std() or 1.0)
    tp_mu, tp_sd = float(ds_tp.mean()), float(ds_tp.std() or 1.0)
    tgt = {
        "logp": dict(zip(["low", "high"], [float(x) for x in np.percentile(ds_lp, e.TARGET_PERCENTILES)])),
        "tpsa": dict(zip(["low", "high"], [float(x) for x in np.percentile(ds_tp, e.TARGET_PERCENTILES)])),
    }
    e["eval/targets"] = tgt
    e.log(f"targets: {tgt}  |  size-fit mols: {len(fit_graphs)}")

    # property-conditional size distribution keyed on normalized [logP, TPSA]
    size_dist = None
    if e.USE_CONDITIONAL_SIZE:
        from torch_geometric.loader import DataLoader
        for g, lp, tp in zip(fit_graphs, ds_lp, ds_tp):
            g.y = torch.tensor([[(lp - lp_mu) / lp_sd, (tp - tp_mu) / tp_sd]], dtype=torch.float)
        size_dist = ConditionalSizeDistribution.from_dataloader(
            DataLoader(fit_graphs, batch_size=256), method=e.SIZE_DIST_METHOD)
        e.log(f"conditional size distribution fitted (method={e.SIZE_DIST_METHOD})")

    def size_condition(lp_t, tp_t, n):
        v = torch.tensor([(lp_t - lp_mu) / lp_sd, (tp_t - tp_mu) / tp_sd], dtype=torch.float)
        return v.unsqueeze(0).expand(n, -1)

    # FK reward machinery: the property heads score each particle's predicted-clean
    # graph. Scales come from the heads' own prop_std (identical to the adapters'
    # cond_std), so reward normalization and adapter conditioning agree.
    fk_heads = fk_domain = None
    if e.USE_FK:
        fk_heads = {"logp": PropertyHead.load(e.LOGP_HEAD_CKPT, device=device),
                    "tpsa": PropertyHead.load(e.TPSA_HEAD_CKPT, device=device)}
        fk_domain = MoleculeDomain(atom_decoder, bond_decoder)
        fk_scales = {k: float(h.prop_std) for k, h in fk_heads.items()}
        e.log(f"FK: K={e.FK_PARTICLES} beta={e.FK_BETA} warmup_frac={e.FK_WARMUP_FRAC} "
              f"resample_interval={e.FK_RESAMPLE_INTERVAL or 'auto'} "
              f"ess_frac={e.FK_ESS_FRAC or 'off'} scales={fk_scales}")

    colors = {("low", "low"): "#2c7fb8", ("low", "high"): "#31a354",
              ("high", "low"): "#d95f0e", ("high", "high"): "#756bb1"}
    results, gen = {"mode": e.COMPOSE_MODE, "weight": e.WEIGHT, "targets": tgt, "combos": {}}, {}

    for lp_lvl, tp_lvl in itertools.product(["low", "high"], ["low", "high"]):
        lp_t, tp_t = tgt["logp"][lp_lvl], tgt["tpsa"][tp_lvl]
        comp = AdapterComposition([
            ConditionBranch(adapters["logp"], torch.tensor([lp_t]), e.WEIGHT),
            ConditionBranch(adapters["tpsa"], torch.tensor([tp_t]), e.WEIGHT),
        ], base=base, mode=e.COMPOSE_MODE)
        if e.USE_FK:
            # invalid_energy is pre-multiplied by scale^2 so that after JointLearnedEnergy
            # divides by it, every property penalizes an undecodable graph equally (1e3).
            energy = JointLearnedEnergy(
                parts=[LearnedPropertyEnergy(fk_heads["logp"], lp_t, fk_domain,
                                             atom_encoder, bond_encoder,
                                             invalid_energy=1e3 * fk_scales["logp"] ** 2),
                       LearnedPropertyEnergy(fk_heads["tpsa"], tp_t, fk_domain,
                                             atom_encoder, bond_encoder,
                                             invalid_energy=1e3 * fk_scales["tpsa"] ** 2)],
                scales=[fk_scales["logp"], fk_scales["tpsa"]])
            samp = FeynmanKacSampler(
                base, energy_fn=energy, composition=comp,
                beta=e.FK_BETA, warmup_frac=e.FK_WARMUP_FRAC,
                resample_interval=(e.FK_RESAMPLE_INTERVAL or None),
                ess_frac=(e.FK_ESS_FRAC or None),
                eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS,
                time_distortion=e.TIME_DISTORTION)
            # K IS the batch: the particles compete within one sample() call, so the
            # chunk cannot be raised past what fits in VRAM without shrinking the pool.
            chunk = e.FK_PARTICLES
        else:
            samp = AdaptedSampler(base, comp, eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS,
                                  time_distortion=e.TIME_DISTORTION)
            chunk = e.EVAL_CHUNK
        samples, rem = [], e.N_PER_COMBO
        while rem > 0:
            cur = min(chunk, rem)
            cond = size_condition(lp_t, tp_t, cur).to(device) if size_dist is not None else None
            samples += samp.sample(cur, size_dist=size_dist, condition=cond,
                                   device=device, show_progress=False)
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
    # raw per-molecule properties, so re-plotting (styling, extra contours) needs no
    # re-sampling -- the first contour version cost a full 23 min re-run for want of this.
    e.commit_json("compose_2d_points.json", {
        f"logp-{a}_tpsa-{b}": {"logp": _lp.tolist(), "tpsa": _tp.tolist()}
        for (a, b), (_lp, _tp) in gen.items()
    })

    # -- the 2D plot: dataset density + 4 generated clusters + target crosshairs
    fig, ax = plt.subplots(figsize=(9, 7.5))
    x_lo, x_hi = float(np.percentile(ds_lp, 0.5)), float(np.percentile(ds_lp, 99.5))
    y_lo, y_hi = max(0.0, float(np.percentile(ds_tp, 0.5))), float(np.percentile(ds_tp, 99.5))
    from scipy.stats import gaussian_kde
    # shared mesh: background density and the per-condition densities are evaluated on
    # the same grid, which is also exactly the visible axis extent (set below).
    XX, YY = np.meshgrid(np.linspace(x_lo, x_hi, e.KDE_GRID),
                         np.linspace(y_lo, y_hi, e.KDE_GRID))
    mesh = np.vstack([XX.ravel(), YY.ravel()])
    if e.BACKGROUND == "contour":
        rng = np.random.RandomState(e.SEED)
        idx = rng.choice(len(ds_lp), min(e.KDE_SUBSAMPLE, len(ds_lp)), replace=False)
        Z = gaussian_kde(np.vstack([ds_lp[idx], ds_tp[idx]]))(mesh).reshape(XX.shape)
        ax.contourf(XX, YY, Z, levels=e.KDE_LEVELS, cmap="Greys", alpha=0.85, zorder=1)
        ax.contour(XX, YY, Z, levels=e.KDE_LEVELS, colors="0.35", linewidths=0.6, alpha=0.5, zorder=1)
        e.log(f"background: KDE contour on {len(idx)} molecules, {e.KDE_GRID}^2 grid")
    else:
        ax.hexbin(ds_lp, ds_tp, gridsize=45, cmap="Greys", bins="log", mincnt=1, zorder=1, alpha=0.9)
    for (lp_lvl, tp_lvl), (glp, gtp) in gen.items():
        c = colors[(lp_lvl, tp_lvl)]
        lp_t, tp_t = tgt["logp"][lp_lvl], tgt["tpsa"][tp_lvl]
        # per-condition KDE: same-colour iso-density rings sitting UNDER the dots
        # (zorder 1.5), so each cluster's shape is readable where the dots overlap.
        # Levels are relative to each cluster's own peak -- an absolute scale would
        # make the tight clusters swamp the diffuse ones.
        if e.CONDITION_KDE and glp.size >= 5:
            try:
                Zc = gaussian_kde(np.vstack([glp, gtp]))(mesh).reshape(XX.shape)
                lv = np.linspace(Zc.max() * e.CONDITION_KDE_FLOOR, Zc.max() * 0.90,
                                 e.CONDITION_KDE_LEVELS)
                ax.contour(XX, YY, Zc, levels=lv, colors=c, linewidths=1.0,
                           alpha=e.CONDITION_KDE_ALPHA, zorder=1.5)
            except np.linalg.LinAlgError:
                e.log(f"[warn] singular KDE for logp-{lp_lvl}_tpsa-{tp_lvl}, skipping its contours")
        if glp.size:
            ax.scatter(glp, gtp, s=14, c=c, alpha=0.5, edgecolors="none", zorder=2,
                       label=f"logP-{lp_lvl}, TPSA-{tp_lvl}")
        ax.scatter([lp_t], [tp_t], marker="X", s=220, c=c, edgecolors="black", linewidths=1.5, zorder=4)
    ax.set_xlabel("logP (Crippen)"); ax.set_ylabel("TPSA")
    ax.set_title(f"Composed adapters (logP x TPSA) over ZINC density\n"
                 f"mode={e.COMPOSE_MODE}, w={e.WEIGHT}  (X = target, dots = generated)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
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
    e.SIZE_FIT_N = 120
    e.GRID_N = 4
    # LOGP_CKPT/TPSA_CKPT default "" -> fresh untrained adapters (mechanism smoke)


experiment.run_if_main()
