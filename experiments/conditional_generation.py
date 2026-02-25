"""
Conditional molecular graph generation with DeFoG.

Trains a DeFoG model on molecules loaded from a CSV file (SMILES + property columns),
using classifier-free guidance for conditional generation. After training, generates
molecules at fixed target property values and evaluates how well they match.

Usage:
    python experiments/experiment_conditional_generation.py
    python experiments/experiment_conditional_generation.py --__TESTING__ True
    python experiments/experiment_conditional_generation.py --EPOCHS 100 --GUIDANCE_SCALE 3.0
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from rdkit import Chem
from rdkit.Chem import Crippen
from torch_geometric.loader import DataLoader
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import (
    build_encoders,
    smiles_to_pyg_data,
    pyg_data_to_mol,
    mol_to_smiles,
)
from defog.core import DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback

# Project root directory (repo root, one level up from experiments/)
_PROJECT_DIR = Path(__file__).parent.parent.resolve()

# ============================================================================
# Parameters
# ============================================================================

# --- Data ---

# :param CSV_PATH:
#     Path to the CSV file containing SMILES and property columns.
CSV_PATH: str = str(_PROJECT_DIR / "data" / "molecules.csv")

# :param SMILES_COLUMN:
#     Name of the column containing SMILES strings.
SMILES_COLUMN: str = "smiles"

# --- Atom/bond configuration ---

# :param ATOM_TYPES:
#     List of atom symbols the model can handle. Molecules with unknown atoms are skipped.
ATOM_TYPES: list = ["C", "N", "O", "F", "S", "Cl", "Br"]

# :param BOND_TYPES:
#     List of bond type names. Options: "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC".
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# --- Property conditioning ---

# :param PROPERTIES:
#     Configuration dict for conditioning properties. Each key is a CSV column name.
#     Each value is a dict with:
#       - "type": "regression" or "classification"
#       - "callback": callable(mol: Chem.Mol) -> float or str, used during evaluation
#       - "target": target value for evaluation sampling
#           regression: float (in original scale, will be z-scored internally)
#           classification: str (class label)
PROPERTIES: dict = {
    "logP": {
        "type": "regression",
        "callback": lambda mol: Crippen.MolLogP(mol),
        "target": 2.5,
    },
}

# --- Model architecture ---

# :param N_LAYERS:
#     Number of transformer layers.
N_LAYERS: int = 6

# :param HIDDEN_DIM:
#     Hidden dimension for the transformer.
HIDDEN_DIM: int = 256

# :param HIDDEN_MLP_DIM:
#     Hidden dimension for MLPs.
HIDDEN_MLP_DIM: int = 512

# :param N_HEADS:
#     Number of attention heads.
N_HEADS: int = 8

# :param DROPOUT:
#     Dropout probability.
DROPOUT: float = 0.1

# :param NOISE_TYPE:
#     Noise distribution type: "uniform", "marginal", or "absorbing".
NOISE_TYPE: str = "marginal"

# :param EXTRA_FEATURES_TYPE:
#     Extra features type: "none", "rrwp", or "cycles".
EXTRA_FEATURES_TYPE: str = "rrwp"

# :param RRWP_STEPS:
#     Number of random walk steps for RRWP features.
RRWP_STEPS: int = 10

# --- Conditioning ---

# :param COND_DROP_PROB:
#     Probability of dropping the condition during training for CFG.
COND_DROP_PROB: float = 0.1

# :param GUIDANCE_SCALE:
#     Guidance scale for CFG sampling. 1.0 = no guidance, >1.0 = amplified conditioning.
GUIDANCE_SCALE: float = 2.0

# --- Training ---

# :param EPOCHS:
#     Number of training epochs.
EPOCHS: int = 500

# :param BATCH_SIZE:
#     Training batch size.
BATCH_SIZE: int = 32

# :param LEARNING_RATE:
#     Learning rate for AdamW optimizer.
LEARNING_RATE: float = 1e-4

# :param WEIGHT_DECAY:
#     Weight decay for AdamW optimizer.
WEIGHT_DECAY: float = 1e-5

# :param TRAIN_SPLIT:
#     Fraction of data used for training (rest is validation).
TRAIN_SPLIT: float = 0.9

# --- Sampling ---

# :param SAMPLE_STEPS:
#     Number of denoising steps during sampling.
SAMPLE_STEPS: int = 100

# :param ETA:
#     Stochasticity parameter for sampling.
ETA: float = 0.0

# :param OMEGA:
#     Target guidance strength for sampling.
OMEGA: float = 0.0

# :param SAMPLE_TIME_DISTORTION:
#     Time distortion for sampling: "identity", "polydec", "cos".
SAMPLE_TIME_DISTORTION: str = "identity"

# :param NUM_EVAL_SAMPLES:
#     Number of molecules to generate during evaluation.
NUM_EVAL_SAMPLES: int = 100

# :param SAMPLE_VIS_EVERY_K:
#     Visualize sample graphs every k validation epochs during training.
SAMPLE_VIS_EVERY_K: int = 10

# --- Special ---
__DEBUG__: bool = True
__TESTING__: bool = False


# ============================================================================
# Helper functions
# ============================================================================

def compute_cond_dim(properties: dict, df: pd.DataFrame) -> int:
    """Compute total condition vector dimensionality."""
    dim = 0
    for name, cfg in sorted(properties.items()):
        if cfg["type"] == "regression":
            dim += 1
        elif cfg["type"] == "classification":
            dim += df[name].nunique()
    return dim


def build_normalization_stats(properties: dict, df: pd.DataFrame) -> dict:
    """
    Compute normalization stats and class mappings for all properties.

    Returns:
        Dict mapping property name to:
        - regression: {"mean": float, "std": float}
        - classification: {"classes": sorted list, "class_to_idx": dict}
    """
    stats = {}
    for name, cfg in sorted(properties.items()):
        if cfg["type"] == "regression":
            values = df[name].astype(float)
            stats[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        elif cfg["type"] == "classification":
            classes = sorted(df[name].unique().tolist(), key=str)
            stats[name] = {
                "classes": classes,
                "class_to_idx": {c: i for i, c in enumerate(classes)},
            }
    return stats


def build_condition_vector(
    row: pd.Series,
    properties: dict,
    norm_stats: dict,
) -> torch.Tensor:
    """
    Build a single condition vector from a DataFrame row.

    Returns:
        1D tensor of shape (cond_dim,)
    """
    parts = []
    for name, cfg in sorted(properties.items()):
        if cfg["type"] == "regression":
            val = float(row[name])
            mean = norm_stats[name]["mean"]
            std = norm_stats[name]["std"]
            z = (val - mean) / std if std > 0 else 0.0
            parts.append(torch.tensor([z], dtype=torch.float))
        elif cfg["type"] == "classification":
            class_to_idx = norm_stats[name]["class_to_idx"]
            num_classes = len(class_to_idx)
            idx = class_to_idx.get(row[name], 0)
            one_hot = torch.zeros(num_classes, dtype=torch.float)
            one_hot[idx] = 1.0
            parts.append(one_hot)
    return torch.cat(parts)


def build_target_condition(
    properties: dict,
    norm_stats: dict,
) -> torch.Tensor:
    """
    Build the target condition vector for evaluation sampling.

    Returns:
        1D tensor of shape (cond_dim,)
    """
    parts = []
    for name, cfg in sorted(properties.items()):
        target = cfg["target"]
        if cfg["type"] == "regression":
            mean = norm_stats[name]["mean"]
            std = norm_stats[name]["std"]
            z = (float(target) - mean) / std if std > 0 else 0.0
            parts.append(torch.tensor([z], dtype=torch.float))
        elif cfg["type"] == "classification":
            class_to_idx = norm_stats[name]["class_to_idx"]
            num_classes = len(class_to_idx)
            idx = class_to_idx.get(str(target), 0)
            one_hot = torch.zeros(num_classes, dtype=torch.float)
            one_hot[idx] = 1.0
            parts.append(one_hot)
    return torch.cat(parts)


# ============================================================================
# Experiment
# ============================================================================

@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    """Conditional molecular generation with DeFoG and classifier-free guidance."""

    e.log("Starting conditional generation experiment")
    e.log_parameters()

    # ------------------------------------------------------------------
    # Step 1: Data loading
    # ------------------------------------------------------------------
    e.log(f"Loading data from {e.CSV_PATH}")
    df = pd.read_csv(e.CSV_PATH)
    e.log(f"Loaded {len(df)} rows")

    # Validate columns
    assert e.SMILES_COLUMN in df.columns, f"Column '{e.SMILES_COLUMN}' not found in CSV"
    for name in e.PROPERTIES:
        assert name in df.columns, f"Property column '{name}' not found in CSV"

    # Build atom/bond encoders
    atom_encoder, _, bond_encoder, _ = build_encoders(
        e.ATOM_TYPES, e.BOND_TYPES,
    )
    e.log(f"Atom types ({len(atom_encoder)}): {e.ATOM_TYPES}")
    e.log(f"Bond types ({len(bond_encoder)}): {e.BOND_TYPES}")

    # ------------------------------------------------------------------
    # Step 2: Condition vector construction
    # ------------------------------------------------------------------
    cond_dim = compute_cond_dim(e.PROPERTIES, df)
    norm_stats = build_normalization_stats(e.PROPERTIES, df)
    e.log(f"Condition dimension: {cond_dim}")
    e.log(f"Normalization stats: {json.dumps(norm_stats, indent=2, default=str)}")

    # Store for use in analysis phase
    e["config/cond_dim"] = cond_dim
    e["config/norm_stats"] = norm_stats
    e["config/atom_types"] = e.ATOM_TYPES
    e["config/bond_types"] = e.BOND_TYPES

    # ------------------------------------------------------------------
    # Step 3: Convert SMILES to PyG Data objects
    # ------------------------------------------------------------------
    e.log("Converting SMILES to graphs...")
    dataset = []
    skipped = 0
    for _, row in df.iterrows():
        smiles = row[e.SMILES_COLUMN]
        data = smiles_to_pyg_data(smiles, atom_encoder, bond_encoder)
        if data is None:
            skipped += 1
            continue

        # Attach condition vector
        cond = build_condition_vector(row, e.PROPERTIES, norm_stats)
        data.y = cond.unsqueeze(0)  # (1, cond_dim)
        dataset.append(data)

    e.log(f"Converted {len(dataset)} molecules ({skipped} skipped)")
    e["data/num_molecules"] = len(dataset)
    e["data/num_skipped"] = skipped

    if len(dataset) == 0:
        e.log("ERROR: No valid molecules found. Check CSV_PATH and ATOM_TYPES.")
        return

    # ------------------------------------------------------------------
    # Step 4: Train/val split and dataloaders
    # ------------------------------------------------------------------
    n_train = int(len(dataset) * e.TRAIN_SPLIT)
    indices = torch.randperm(len(dataset)).tolist()
    train_set = [dataset[i] for i in indices[:n_train]]
    val_set = [dataset[i] for i in indices[n_train:]]

    train_loader = DataLoader(train_set, batch_size=e.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=e.BATCH_SIZE, shuffle=False) if val_set else None

    e.log(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # ------------------------------------------------------------------
    # Step 5: Model creation
    # ------------------------------------------------------------------
    model = DeFoGModel.from_dataloader(
        train_loader,
        n_layers=e.N_LAYERS,
        hidden_dim=e.HIDDEN_DIM,
        noise_type=e.NOISE_TYPE,
        hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS,
        dropout=e.DROPOUT,
        extra_features_type=e.EXTRA_FEATURES_TYPE,
        rrwp_steps=e.RRWP_STEPS,
        lr=e.LEARNING_RATE,
        weight_decay=e.WEIGHT_DECAY,
        sample_steps=e.SAMPLE_STEPS,
        eta=e.ETA,
        omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
        cond_dim=cond_dim,
        cond_drop_prob=e.COND_DROP_PROB,
        guidance_scale=e.GUIDANCE_SCALE,
    )

    num_params = sum(p.numel() for p in model.parameters())
    e.log(f"Model: {model}")
    e.log(f"Parameters: {num_params:,}")
    e["model/num_params"] = num_params

    # ------------------------------------------------------------------
    # Step 6: Training
    # ------------------------------------------------------------------
    e.log(f"Training for {e.EPOCHS} epochs...")

    monitor = TrainingMonitorCallback(
        smoothing_window=5,
        figure_callback=lambda fig: e.track("training_progress", fig),
    )

    sampler = SampleVisualizationCallback(
        num_samples=8,
        every_k_epochs=e.SAMPLE_VIS_EVERY_K,
        sample_steps=e.SAMPLE_STEPS,
        figure_callback=lambda fig: e.track("samples", fig),
    )

    trainer = pl.Trainer(
        max_epochs=e.EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
        callbacks=[monitor, sampler],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save model
    model_path = os.path.join(e.path, "model")
    saved_path = model.save(model_path)
    e.log(f"Model saved to {saved_path}")

    e.log("Training complete!")

    # ------------------------------------------------------------------
    # Step 7: Evaluation — generate conditional samples and evaluate
    # ------------------------------------------------------------------
    e.log("=" * 60)
    e.log("EVALUATION: Conditional sample evaluation")
    e.log("=" * 60)

    _, atom_decoder, _, bond_decoder = build_encoders(e.ATOM_TYPES, e.BOND_TYPES)

    # Build target condition vector
    target_cond = build_target_condition(e.PROPERTIES, norm_stats)
    condition = target_cond.unsqueeze(0).expand(e.NUM_EVAL_SAMPLES, -1)

    e.log(f"Target condition vector: {target_cond.tolist()}")
    for name, cfg in sorted(e.PROPERTIES.items()):
        e.log(f"  {name} ({cfg['type']}): target = {cfg['target']}")

    # Generate samples
    model.eval()
    e.log(f"Generating {e.NUM_EVAL_SAMPLES} samples...")
    samples = model.sample(
        num_samples=e.NUM_EVAL_SAMPLES,
        condition=condition,
        guidance_scale=e.GUIDANCE_SCALE,
        sample_steps=e.SAMPLE_STEPS,
        eta=e.ETA,
        omega=e.OMEGA,
        time_distortion=e.SAMPLE_TIME_DISTORTION,
        show_progress=True,
    )
    e.log(f"Generated {len(samples)} samples")

    # Reconstruct molecules and evaluate
    valid_smiles = []
    all_results = []

    for i, sample in enumerate(samples):
        mol = pyg_data_to_mol(sample, atom_decoder, bond_decoder)
        smiles = mol_to_smiles(mol) if mol is not None else None

        result = {"index": i, "valid": smiles is not None, "smiles": smiles}

        if smiles is not None:
            valid_smiles.append(smiles)

            eval_mol = Chem.MolFromSmiles(smiles)
            if eval_mol is not None:
                for name, cfg in sorted(e.PROPERTIES.items()):
                    try:
                        computed = cfg["callback"](eval_mol)
                        result[f"{name}_computed"] = computed
                        result[f"{name}_target"] = cfg["target"]
                    except Exception as ex:
                        result[f"{name}_error"] = str(ex)

        all_results.append(result)

    # Compute metrics
    num_valid = len(valid_smiles)
    validity_rate = num_valid / e.NUM_EVAL_SAMPLES if e.NUM_EVAL_SAMPLES > 0 else 0
    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / num_valid if num_valid > 0 else 0

    metrics = {
        "num_samples": e.NUM_EVAL_SAMPLES,
        "num_valid": num_valid,
        "validity_rate": validity_rate,
        "num_unique": len(unique_smiles),
        "uniqueness": uniqueness,
    }

    for name, cfg in sorted(e.PROPERTIES.items()):
        computed_key = f"{name}_computed"
        values = [r[computed_key] for r in all_results if computed_key in r]

        if cfg["type"] == "regression" and values:
            target = float(cfg["target"])
            errors = [abs(v - target) for v in values]
            metrics[f"{name}_mae"] = float(np.mean(errors))
            metrics[f"{name}_mean_computed"] = float(np.mean(values))
            metrics[f"{name}_std_computed"] = float(np.std(values))
            e.log(f"  {name}: MAE = {metrics[f'{name}_mae']:.4f} "
                  f"(mean={metrics[f'{name}_mean_computed']:.4f}, "
                  f"std={metrics[f'{name}_std_computed']:.4f}, "
                  f"target={target})")

        elif cfg["type"] == "classification" and values:
            target = str(cfg["target"])
            correct = sum(1 for v in values if str(v) == target)
            metrics[f"{name}_accuracy"] = correct / len(values) if values else 0
            e.log(f"  {name}: accuracy = {metrics[f'{name}_accuracy']:.4f} "
                  f"({correct}/{len(values)}, target={target})")

    e.log(f"Validity: {validity_rate:.2%} ({num_valid}/{e.NUM_EVAL_SAMPLES})")
    e.log(f"Uniqueness: {uniqueness:.2%} ({len(unique_smiles)}/{num_valid})")

    # Save artifacts
    e.commit_json("eval_metrics.json", metrics)
    e.commit_json("generated_smiles.json", list(unique_smiles))
    e.commit_json("eval_results.json", all_results)

    e["eval/validity_rate"] = validity_rate
    e["eval/uniqueness"] = uniqueness
    e["eval/metrics"] = metrics

    e.log("Evaluation complete!")


@experiment.testing
def testing(e: Experiment):
    """Reduce parameters for quick testing."""
    e.CSV_PATH = str(_PROJECT_DIR / "data" / "test_molecules.csv")
    e.EPOCHS = 2
    e.BATCH_SIZE = 4
    e.NUM_EVAL_SAMPLES = 5
    e.SAMPLE_STEPS = 3
    e.SAMPLE_VIS_EVERY_K = 1
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2


experiment.run_if_main()
