"""
Conditional molecular generation on AqSolDB with logP + SAS conditioning.

Trains a DeFoG conditional model on AqSolDB molecules (preprocessed by
``scripts/prepare_aqsoldb.py`` into ``data/aqsoldb_conditional.csv`` with
recomputed ``logp`` and ``sas`` columns), conditioning jointly on both
properties via classifier-free guidance.

After training it evaluates the model on a 3x3 grid of joint (logP, SAS)
targets -- the low/medium/high percentiles of each property -- and, for each
target, produces a two-panel figure comparing:
  - gray  : the property distribution of the whole dataset,
  - red   : the property distribution of the generated molecules (RDKit-evaluated),
  - black : a vertical line at the requested target value.

Graph size is drawn from a property-conditional size distribution so the
sampled size stays consistent with the (size-correlated) target.

Usage:
    python scripts/prepare_aqsoldb.py            # once, to build the CSV
    python experiments/conditional_training__aqsoldb.py
    python experiments/conditional_training__aqsoldb.py --__TESTING__ True
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, RDConfig
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import (
    build_encoders,
    smiles_to_pyg_data,
    pyg_data_to_mol,
    mol_to_smiles,
)
from experiments.conditional_generation import (
    build_normalization_stats,
    build_condition_vector,
)
from defog.core import (
    DeFoGModel,
    TrainingMonitorCallback,
    SampleVisualizationCallback,
    ConditionalSizeDistribution,
)

# RDKit Contrib synthetic-accessibility scorer.
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = Path(__file__).parent.parent.resolve()

# ============================================================================
# Parameters
# ============================================================================

# :param CSV_PATH:
#     Preprocessed AqSolDB CSV (smiles, logp, sas). Build via prepare_aqsoldb.py.
CSV_PATH: str = str(_PROJECT_DIR / "data" / "aqsoldb_conditional.csv")

# :param SMILES_COLUMN:
SMILES_COLUMN: str = "smiles"

# :param BOND_TYPES:
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# :param PROPERTIES:
#     Joint conditioning on logP and SAS (both regression, z-score normalized).
#     Callbacks recompute the property from a generated molecule at eval time.
PROPERTIES: dict = {
    "logp": {
        "type": "regression",
        "callback": lambda mol: float(Crippen.MolLogP(mol)),
    },
    "sas": {
        "type": "regression",
        "callback": lambda mol: float(sascorer.calculateScore(mol)),
    },
}

# --- Model architecture ---
N_LAYERS: int = 6
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 10

# --- Conditioning ---
COND_DROP_PROB: float = 0.1
GUIDANCE_SCALE: float = 2.0

# --- Training ---
EPOCHS: int = 250
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-5
TRAIN_SPLIT: float = 0.9

# --- Sampling / evaluation ---
SAMPLE_STEPS: int = 100
ETA: float = 0.0
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "identity"

# :param SIZE_DIST_METHOD:
#     "kernel" / "regression" -> property-conditional size; "marginal" -> P(n).
SIZE_DIST_METHOD: str = "kernel"

# :param TARGET_PERCENTILES:
#     Percentiles used as low/medium/high target levels for each property.
TARGET_PERCENTILES: list = [10, 50, 90]

# :param LEVEL_NAMES:
LEVEL_NAMES: list = ["low", "med", "high"]

# :param NUM_EVAL_SAMPLES:
#     Molecules generated per (logP, SAS) target for the red histogram.
NUM_EVAL_SAMPLES: int = 500

# :param SAMPLE_VIS_EVERY_K:
SAMPLE_VIS_EVERY_K: int = 25

# --- Special ---
__DEBUG__: bool = True
__TESTING__: bool = False


# ============================================================================
# Helpers
# ============================================================================

def derive_atom_types(smiles_list) -> list:
    """Atom vocabulary observed in the (already filtered) dataset, most-common first."""
    counts = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    return [sym for sym, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def target_condition_vector(targets: dict, properties: dict, norm_stats: dict) -> torch.Tensor:
    """Build a normalized condition vector from raw target values (sorted prop order)."""
    parts = []
    for name, _cfg in sorted(properties.items()):
        mean = norm_stats[name]["mean"]
        std = norm_stats[name]["std"]
        z = (float(targets[name]) - mean) / std if std > 0 else 0.0
        parts.append(torch.tensor([z], dtype=torch.float))
    return torch.cat(parts)


def plot_target(dataset_df, generated, targets, property_names, title):
    """Two-panel figure: gray dataset hist, red generated hist, black target line."""
    fig, axes = plt.subplots(1, len(property_names), figsize=(6 * len(property_names), 4.5))
    if len(property_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, property_names):
        data_vals = dataset_df[name].values
        gen_vals = np.array(generated.get(name, []))
        lo = float(np.min(data_vals))
        hi = float(np.max(data_vals))
        bins = np.linspace(lo, hi, 40)
        ax.hist(data_vals, bins=bins, density=True, color="0.7",
                label="dataset", alpha=0.9)
        if gen_vals.size > 0:
            ax.hist(gen_vals, bins=bins, density=True, color="red",
                    label="generated", alpha=0.55)
        ax.axvline(targets[name], color="black", linestyle="--", linewidth=2,
                   label=f"target = {targets[name]:.2f}")
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ============================================================================
# Experiment
# ============================================================================

@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    e.log("AqSolDB conditional generation (logP + SAS)")
    e.log_parameters()

    # -- Data ---------------------------------------------------------------
    df = pd.read_csv(e.CSV_PATH)
    e.log(f"Loaded {len(df)} molecules from {e.CSV_PATH}")

    atom_types = derive_atom_types(df[e.SMILES_COLUMN])
    e.log(f"Atom vocabulary ({len(atom_types)}): {atom_types}")
    e["config/atom_types"] = atom_types
    e["config/bond_types"] = e.BOND_TYPES

    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(
        atom_types, e.BOND_TYPES
    )

    property_names = sorted(e.PROPERTIES.keys())
    norm_stats = build_normalization_stats(e.PROPERTIES, df)
    e["config/norm_stats"] = norm_stats
    e.log(f"Normalization: {norm_stats}")

    # -- Convert to graphs with condition vectors ---------------------------
    dataset = []
    skipped = 0
    for _, row in df.iterrows():
        data = smiles_to_pyg_data(row[e.SMILES_COLUMN], atom_encoder, bond_encoder)
        if data is None:
            skipped += 1
            continue
        data.y = build_condition_vector(row, e.PROPERTIES, norm_stats).unsqueeze(0)
        dataset.append(data)
    e.log(f"Converted {len(dataset)} graphs ({skipped} skipped)")

    from torch_geometric.loader import DataLoader
    n_train = int(len(dataset) * e.TRAIN_SPLIT)
    perm = torch.randperm(len(dataset)).tolist()
    train_set = [dataset[i] for i in perm[:n_train]]
    val_set = [dataset[i] for i in perm[n_train:]]
    train_loader = DataLoader(train_set, batch_size=e.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=e.BATCH_SIZE) if val_set else None
    e.log(f"Train: {len(train_set)}  Val: {len(val_set)}")

    # -- Model --------------------------------------------------------------
    cond_dim = len(property_names)  # both regression -> one dim each
    model = DeFoGModel.from_dataloader(
        train_loader,
        n_layers=e.N_LAYERS, hidden_dim=e.HIDDEN_DIM, hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS, dropout=e.DROPOUT, noise_type=e.NOISE_TYPE,
        extra_features_type=e.EXTRA_FEATURES_TYPE, rrwp_steps=e.RRWP_STEPS,
        lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY,
        sample_steps=e.SAMPLE_STEPS, eta=e.ETA, omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
        cond_dim=cond_dim, cond_drop_prob=e.COND_DROP_PROB,
        guidance_scale=e.GUIDANCE_SCALE,
    )
    e.log(f"Model: {model}")
    e["model/num_params"] = sum(p.numel() for p in model.parameters())

    # -- Train --------------------------------------------------------------
    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig)
    )
    sampler = SampleVisualizationCallback(
        num_samples=8, every_k_epochs=e.SAMPLE_VIS_EVERY_K, sample_steps=e.SAMPLE_STEPS,
        figure_callback=lambda fig: e.track("samples", fig),
    )
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, accelerator="auto", devices=1,
        enable_progress_bar=True, enable_checkpointing=False, logger=False,
        callbacks=[monitor, sampler],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model_path = model.save(os.path.join(e.path, "model"))
    e.log(f"Saved model to {model_path}")

    # -- Evaluation: 3x3 joint (logP, SAS) target grid ----------------------
    e.log("=" * 60)
    e.log("EVALUATION: joint target grid")
    model.eval()

    # Property-conditional size distribution built from the training pairs.
    size_dist = None
    if e.SIZE_DIST_METHOD in ("kernel", "regression"):
        size_dist = ConditionalSizeDistribution.from_dataloader(
            train_loader, method=e.SIZE_DIST_METHOD
        )
        e.log(f"Conditional size distribution (method={e.SIZE_DIST_METHOD})")

    # Low/med/high levels per property from dataset percentiles.
    levels = {
        name: dict(zip(e.LEVEL_NAMES,
                       np.percentile(df[name].values, e.TARGET_PERCENTILES)))
        for name in property_names
    }
    e["eval/levels"] = {n: {k: float(v) for k, v in lv.items()}
                        for n, lv in levels.items()}
    e.log(f"Target levels: {e['eval/levels']}")

    grid_metrics = []
    for lvl_a in e.LEVEL_NAMES:      # logp level
        for lvl_b in e.LEVEL_NAMES:  # sas level
            targets = {
                property_names[0]: float(levels[property_names[0]][lvl_a]),
                property_names[1]: float(levels[property_names[1]][lvl_b]),
            }
            cond_vec = target_condition_vector(targets, e.PROPERTIES, norm_stats)
            condition = cond_vec.unsqueeze(0).expand(e.NUM_EVAL_SAMPLES, -1)

            samples = model.sample(
                num_samples=e.NUM_EVAL_SAMPLES, condition=condition,
                guidance_scale=e.GUIDANCE_SCALE, sample_steps=e.SAMPLE_STEPS,
                eta=e.ETA, omega=e.OMEGA, time_distortion=e.SAMPLE_TIME_DISTORTION,
                size_dist=size_dist, show_progress=False,
            )

            generated = {name: [] for name in property_names}
            n_valid = 0
            for s in samples:
                mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
                smi = mol_to_smiles(mol) if mol is not None else None
                if smi is None:
                    continue
                emol = Chem.MolFromSmiles(smi)
                if emol is None:
                    continue
                n_valid += 1
                for name in property_names:
                    try:
                        generated[name].append(e.PROPERTIES[name]["callback"](emol))
                    except Exception:
                        pass

            tag = f"logp-{lvl_a}_sas-{lvl_b}"
            title = (f"logP {lvl_a}={targets['logp']:.2f}, "
                     f"SAS {lvl_b}={targets['sas']:.2f}  "
                     f"(valid {n_valid}/{e.NUM_EVAL_SAMPLES})")
            fig = plot_target(df, generated, targets, property_names, title)
            e.commit_fig(f"target_{tag}.png", fig)
            plt.close(fig)

            row_metrics = {"target_logp": targets["logp"], "target_sas": targets["sas"],
                           "level_logp": lvl_a, "level_sas": lvl_b, "n_valid": n_valid}
            for name in property_names:
                vals = generated[name]
                if vals:
                    row_metrics[f"{name}_mean"] = float(np.mean(vals))
                    row_metrics[f"{name}_mae"] = float(np.mean(np.abs(np.array(vals) - targets[name])))
            grid_metrics.append(row_metrics)
            e.log(f"  {tag}: valid={n_valid}/{e.NUM_EVAL_SAMPLES}  "
                  + "  ".join(f"{n}_mae={row_metrics.get(f'{n}_mae', float('nan')):.3f}"
                              for n in property_names))

    e.commit_json("grid_metrics.json", grid_metrics)
    e.log("Evaluation complete.")


@experiment.testing
def testing(e: Experiment):
    """Tiny end-to-end smoke run (all 9 target figures, few samples)."""
    e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.SAMPLE_STEPS = 5
    e.NUM_EVAL_SAMPLES = 20
    e.SAMPLE_VIS_EVERY_K = 1
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2
    # subset the dataset for speed
    df = pd.read_csv(e.CSV_PATH).head(150)
    smoke_path = os.path.join(folder_path(__file__), "_aqsoldb_smoke.csv")
    df.to_csv(smoke_path, index=False)
    e.CSV_PATH = smoke_path


experiment.run_if_main()
