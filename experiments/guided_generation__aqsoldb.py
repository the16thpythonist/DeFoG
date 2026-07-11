"""
Amortized exact-guidance generation on AqSolDB (logP steering).

Trains a SINGLE target-conditioned guidance network (the target-amortized
extension of arXiv:2509.21912's exact posterior-based discrete guidance) on top of
a FROZEN unconditional AqSolDB DeFoG base, then steers generation toward the
low/medium/high logP percentiles of the dataset.

Unlike classifier-free guidance (``conditional_training__aqsoldb.py``), the base is
never retrained: guidance is a learned reweighting ``q ∝ h·p`` of the base's
predicted clean-graph marginals, and a single amortized ``h`` (conditioned on the
target logP, sampled during training) covers the whole target range.

For each target it produces:
  - a 5x5 grid of valid guided molecules (legended with their RDKit logP), and
  - it contributes one curve to a combined distribution figure: generated logP
    distributions over a grey dataset-logP backdrop, with target lines.

Usage:
    python experiments/guided_generation__aqsoldb.py
    python experiments/guided_generation__aqsoldb.py --__TESTING__ True
"""
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Draw
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, GuidedSampler, AmortizedPropertyGuidanceModule, ConditionalSizeDistribution,
)

RDLogger.DisableLog("rdApp.*")

# == configuration ==========================================================
# :param CKPT_PATH: frozen unconditional AqSolDB base checkpoint.
CKPT_PATH: str = os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt")
# :param CSV_PATH: dataset with `smiles` and precomputed `logp` columns.
CSV_PATH: str = "data/aqsoldb_conditional.csv"
# :param BOND_TYPES: bond vocabulary (order fixes edge classes; 0 = no bond).
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
# :param SEED: global seed for reproducible init / shuffling / sampling.
SEED: int = 0

# --- guidance training ---
# :param GAMMA: energy sharpness in r = exp(-GAMMA * (logP(x1) - target)^2).
GAMMA: float = 0.6
# :param H_LAYERS / H_HIDDEN: size of the guidance network h (can be < base).
H_LAYERS: int = 6
H_HIDDEN: int = 256
# :param EPOCHS / BATCH_SIZE / LR: Bregman training schedule for h.
EPOCHS: int = 60
BATCH_SIZE: int = 32
LR: float = 2e-4

# --- evaluation / sampling ---
# :param TARGET_PERCENTILES / LEVEL_NAMES: low/med/high logP targets.
TARGET_PERCENTILES: list = [10, 50, 90]
LEVEL_NAMES: list = ["low", "med", "high"]
# :param NUM_EVAL_SAMPLES: molecules generated per target.
NUM_EVAL_SAMPLES: int = 200
# :param CHUNK: sampling batch size (fit GPU memory).
CHUNK: int = 40
# :param GUIDANCE_WEIGHT: strength w of the reweight q ∝ h^w·p (1.0 = exact
#     Theorem-1 guidance; >1 sharpens the tilt toward the target).
GUIDANCE_WEIGHT: float = 2.0
# :param SAMPLE_STEPS / ETA / OMEGA / TIME_DISTORTION: CTMC sampling policy.
#     Lower ETA (less stochasticity) lets samples follow the guidance more directly.
SAMPLE_STEPS: int = 400
ETA: float = 2.0
OMEGA: float = 0.3
TIME_DISTORTION: str = "polydec"
# :param COND_SIZE / SIZE_METHOD / SIZE_BANDWIDTH: draw graph size from a
#     target-conditioned size distribution. logP is strongly size-dependent, so a
#     target-independent size caps the reachable logP (esp. the high end). The
#     kernel bandwidth (in logP units) controls how tightly size tracks the target;
#     a tight value (~0.5) gives the high target its true large scaffolds, whereas
#     the auto "median" bandwidth (~1.8) washes the separation out. Keep this on.
COND_SIZE: bool = True
SIZE_METHOD: str = "kernel"
SIZE_BANDWIDTH: float = 0.5
# :param GRID: molecule-grid shape per target.
GRID_ROWS: int = 5
GRID_COLS: int = 5

__DEBUG__: bool = True
__TESTING__: bool = False


# == helpers ================================================================
def derive_atom_types(smiles_list):
    counts = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] += 1
    return [sym for sym, _ in counts.most_common()]


def build_dataset(df, atom_encoder, bond_encoder):
    graphs = []
    skipped = 0
    for smi, lp in zip(df["smiles"], df["logp"]):
        try:
            data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder)
        except Exception:
            skipped += 1
            continue
        if data is None or data.edge_index.numel() == 0:
            skipped += 1
            continue
        data.prop_val = torch.tensor([float(lp)], dtype=torch.float)
        graphs.append(data)
    return graphs, skipped


@torch.no_grad()
def guided_sample(base, guidance, target, n, chunk, steps, eta, omega, distortion,
                  device, atom_decoder, bond_decoder, size_dist=None):
    guidance.set_target(target)
    sampler = GuidedSampler(base, guidance, eta=eta, omega=omega,
                            sample_steps=steps, time_distortion=distortion)
    mols, logps = [], []
    remaining = n
    while remaining > 0:
        cur = min(chunk, remaining)
        kw = {}
        if size_dist is not None:
            # target drives SIZE only (base is unconditional -> ignores it for the net)
            kw["size_dist"] = size_dist
            kw["condition"] = torch.full((cur, 1), float(target))
        for s in sampler.sample(cur, device=device, show_progress=False, **kw):
            mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
            if mol is not None and mol_to_smiles(mol) is not None:
                try:
                    logps.append(float(Crippen.MolLogP(mol)))
                    mols.append(mol)
                except Exception:
                    pass
        remaining -= cur
    return mols, np.array(logps)


def grid_figure(mols, logps, target, rows, cols):
    n = rows * cols
    img = Draw.MolsToGridImage(mols[:n], molsPerRow=cols, subImgSize=(220, 220),
                               legends=[f"logP={v:.2f}" for v in logps[:n]])
    fig, ax = plt.subplots(figsize=(cols * 2.2, rows * 2.3))
    ax.imshow(np.asarray(img)); ax.axis("off")
    ax.set_title(f"Guided molecules — target logP = {target:.2f}")
    fig.tight_layout()
    return fig


def distribution_figure(dataset_logp, per_target, targets, level_names):
    from scipy.stats import gaussian_kde
    lo, hi = -4.0, 9.0
    grid = np.linspace(lo, hi, 400)
    colors = {"low": "#2c7fb8", "med": "#31a354", "high": "#d95f0e"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    d = dataset_logp[(dataset_logp >= lo) & (dataset_logp <= hi)]
    ax.hist(d, bins=60, range=(lo, hi), density=True, color="0.8",
            label="AqSolDB dataset", zorder=1)
    for lvl in level_names:
        lp, c = per_target[lvl], colors.get(lvl, "purple")
        if len(lp) > 5 and np.std(lp) > 1e-3:
            ax.plot(grid, gaussian_kde(lp)(grid), color=c, lw=2.2,
                    label=f"generated ({lvl}, target={targets[lvl]:.2f})", zorder=3)
        ax.axvline(targets[lvl], color=c, ls="--", lw=1.6, zorder=2)
        if len(lp):
            ax.axvline(float(np.mean(lp)), color=c, ls="-", lw=1.0, alpha=0.5, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("logP (Crippen)"); ax.set_ylabel("density")
    ax.set_title("Amortized exact guidance on AqSolDB: logP steering")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# == experiment =============================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment) -> None:
    e.log("AqSolDB amortized exact-guidance generation (logP)")
    pl.seed_everything(e.SEED, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(e.CSV_PATH)
    atom_types = derive_atom_types(df["smiles"])
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, e.BOND_TYPES)
    e["config/atom_types"] = atom_types

    logp_all = df["logp"].values.astype(float)
    prop_mean, prop_std = float(logp_all.mean()), float(logp_all.std())
    lo_c, hi_c = np.percentile(logp_all, [1, 99])
    prop_values = logp_all[(logp_all >= lo_c) & (logp_all <= hi_c)]
    targets = dict(zip(e.LEVEL_NAMES, np.percentile(logp_all, e.TARGET_PERCENTILES)))
    e["eval/targets"] = {k: float(v) for k, v in targets.items()}
    e.log(f"targets: {e['eval/targets']}")

    graphs, skipped = build_dataset(df, atom_encoder, bond_encoder)
    e.log(f"built {len(graphs)} graphs ({skipped} skipped)")

    base = DeFoGModel.load(e.CKPT_PATH, device="cpu")
    assert len(atom_decoder) == base.num_node_classes, "atom vocab != base node classes"

    module = AmortizedPropertyGuidanceModule(
        base, prop_values=prop_values, prop_mean=prop_mean, prop_std=prop_std,
        gamma=e.GAMMA, prop_attr="prop_val", lr=e.LR,
        n_layers=e.H_LAYERS, hidden_dim=e.H_HIDDEN,
    )
    e["model/h_num_params"] = sum(p.numel() for p in module.h.parameters())
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(graphs, batch_size=e.BATCH_SIZE, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1, logger=False, enable_checkpointing=False,
        enable_model_summary=False, gradient_clip_val=1.0, log_every_n_steps=25,
    )
    e.log(f"training amortized guidance: epochs={e.EPOCHS} h={e.H_LAYERS}x{e.H_HIDDEN} gamma={e.GAMMA}")
    trainer.fit(module, train_loader)

    guidance = module.guidance().set_weight(e.GUIDANCE_WEIGHT)
    guidance.save(os.path.join(e.path, "guided_logp_amortized"))
    base.to(device).eval(); guidance.h.to(device).eval()

    # Target-conditioned graph-size distribution (logP -> n_nodes). Essential:
    # logP is size-driven, so a target-independent size caps the reachable logP.
    size_dist = None
    if e.COND_SIZE:
        conds = torch.tensor([[float(g.prop_val)] for g in graphs])
        sizes = torch.tensor([int(g.x.size(0)) for g in graphs])
        size_dist = ConditionalSizeDistribution(
            conds, sizes, method=e.SIZE_METHOD, bandwidth=e.SIZE_BANDWIDTH)
        e.log(f"conditional size distribution ({e.SIZE_METHOD}, bandwidth={e.SIZE_BANDWIDTH})")

    grid_metrics, per_target, all_generated = {}, {}, {}
    for lvl in e.LEVEL_NAMES:
        tgt = float(targets[lvl])
        mols, logps = guided_sample(
            base, guidance, tgt, e.NUM_EVAL_SAMPLES, e.CHUNK, e.SAMPLE_STEPS,
            e.ETA, e.OMEGA, e.TIME_DISTORTION, device, atom_decoder, bond_decoder,
            size_dist=size_dist,
        )
        per_target[lvl] = logps
        achieved = float(np.mean(logps)) if len(logps) else float("nan")
        mae = float(np.mean(np.abs(logps - tgt))) if len(logps) else float("nan")
        grid_metrics[lvl] = {
            "target": tgt, "n_valid": len(logps), "n_requested": e.NUM_EVAL_SAMPLES,
            "validity": len(logps) / e.NUM_EVAL_SAMPLES,
            "achieved_mean_logp": achieved, "logp_mae": mae,
        }
        all_generated[lvl] = [mol_to_smiles(m) for m in mols]
        e.log(f"  {lvl}: target={tgt:.2f} valid={len(logps)}/{e.NUM_EVAL_SAMPLES} "
              f"achieved_mean={achieved:.2f} MAE={mae:.2f}")
        if mols:
            e.commit_fig(f"guided_logp_grid_{lvl}.png",
                         grid_figure(mols, logps, tgt, e.GRID_ROWS, e.GRID_COLS))

    e.commit_fig("guided_logp_distributions.png",
                 distribution_figure(logp_all, per_target, targets, e.LEVEL_NAMES))
    e.commit_json("grid_metrics.json", grid_metrics)
    e.commit_json("generated_smiles.json", all_generated)
    e.log("done.")


@experiment.testing
def testing(e: Experiment) -> None:
    # Fast smoke: tiny h, few epochs, few samples, few steps, subsampled data.
    e.EPOCHS = 2
    e.H_LAYERS = 4
    e.H_HIDDEN = 128
    e.NUM_EVAL_SAMPLES = 10
    e.CHUNK = 10
    e.SAMPLE_STEPS = 50


experiment.run_if_main()
