"""
Unconditional DeFoG training on AqSolDB (control for the conditional run).

Identical recipe to conditional_training__aqsoldb.py -- same data, 250 epochs,
6-layer/256-hidden model, marginal noise, RRWP -- but with NO conditioning
(cond_dim=0, no data.y). At the end it samples 1000 molecules unconditionally
(chunked on GPU) and reports validity + uniqueness, to test whether the
conditional training itself was responsible for the low validity.

Usage:
    python experiments/training__aqsoldb_uncond.py
    python experiments/training__aqsoldb_uncond.py --__TESTING__ True
"""
import os

import pandas as pd
import torch
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import (
    build_encoders,
    smiles_to_pyg_data,
    pyg_data_to_mol,
    mol_to_smiles,
)
from defog.core import DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters (identical to the conditional run, minus conditioning)
# ============================================================================

CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "aqsoldb_conditional.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# --- Model architecture (same as conditional) ---
N_LAYERS: int = 6
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 10

# --- Training (same as conditional) ---
EPOCHS: int = 250
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-5
TRAIN_SPLIT: float = 0.9

# --- Sampling / evaluation (same params as the conditional uncond eval) ---
SAMPLE_STEPS: int = 100
ETA: float = 0.0
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "identity"
NUM_EVAL_SAMPLES: int = 1000
EVAL_CHUNK: int = 32
SAMPLE_VIS_EVERY_K: int = 25

# --- Special ---
__DEBUG__: bool = True
__TESTING__: bool = False


def derive_atom_types(smiles_list) -> list:
    counts = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for a in mol.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    e.log("AqSolDB UNCONDITIONAL training (control run)")

    # -- Data ---------------------------------------------------------------
    df = pd.read_csv(e.CSV_PATH)
    e.log(f"Loaded {len(df)} molecules from {e.CSV_PATH}")

    atom_types = derive_atom_types(df[e.SMILES_COLUMN])
    e.log(f"Atom vocabulary ({len(atom_types)}): {atom_types}")
    e["config/atom_types"] = atom_types

    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(
        atom_types, e.BOND_TYPES
    )

    # No condition attached -- purely unconditional graphs.
    dataset = []
    skipped = 0
    for _, row in df.iterrows():
        data = smiles_to_pyg_data(row[e.SMILES_COLUMN], atom_encoder, bond_encoder)
        if data is None:
            skipped += 1
            continue
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

    # -- Model (cond_dim defaults to 0 -> unconditional) --------------------
    model = DeFoGModel.from_dataloader(
        train_loader,
        n_layers=e.N_LAYERS, hidden_dim=e.HIDDEN_DIM, hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS, dropout=e.DROPOUT, noise_type=e.NOISE_TYPE,
        extra_features_type=e.EXTRA_FEATURES_TYPE, rrwp_steps=e.RRWP_STEPS,
        lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY,
        sample_steps=e.SAMPLE_STEPS, eta=e.ETA, omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
    )
    e.log(f"Model: {model}")
    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"Params: {e['model/num_params']:,}  (cond_dim={model.cond_dim})")

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

    # -- Evaluation: 1000 unconditional samples (chunked on GPU) ------------
    e.log("=" * 60)
    e.log(f"EVALUATION: {e.NUM_EVAL_SAMPLES} unconditional samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    e.log(f"sampling on {next(model.parameters()).device}")

    samples = []
    remaining = e.NUM_EVAL_SAMPLES
    while remaining > 0:
        cur = min(e.EVAL_CHUNK, remaining)
        samples += model.sample(
            num_samples=cur, sample_steps=e.SAMPLE_STEPS, device=device,
            show_progress=False,
        )
        remaining -= cur

    valid_smiles = []
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi is not None and Chem.MolFromSmiles(smi) is not None:
            valid_smiles.append(smi)

    n = e.NUM_EVAL_SAMPLES
    n_valid = len(valid_smiles)
    n_unique = len(set(valid_smiles))
    validity = n_valid / n
    uniqueness = (n_unique / n_valid) if n_valid else 0.0
    e["eval/validity"] = validity
    e["eval/uniqueness"] = uniqueness
    e["eval/num_valid"] = n_valid
    e.commit_json("uncond_metrics.json",
                  {"num_samples": n, "num_valid": n_valid, "validity": validity,
                   "num_unique": n_unique, "uniqueness": uniqueness})
    e.log(f"validity:   {n_valid}/{n} = {validity:.1%}")
    e.log(f"uniqueness: {n_unique}/{n_valid} = {uniqueness:.1%} (of valid)")


@experiment.testing
def testing(e: Experiment):
    e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.SAMPLE_STEPS = 5
    e.NUM_EVAL_SAMPLES = 20
    e.EVAL_CHUNK = 8
    e.SAMPLE_VIS_EVERY_K = 1
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2
    df = pd.read_csv(e.CSV_PATH).head(150)
    smoke_path = os.path.join(folder_path(__file__), "_aqsoldb_uncond_smoke.csv")
    df.to_csv(smoke_path, index=False)
    e.CSV_PATH = smoke_path


experiment.run_if_main()
