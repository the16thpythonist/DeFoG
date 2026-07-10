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
    make_generation_metrics_fn,
)
from defog.core import (
    DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback, EMACallback
)
from defog.domains import MoleculeDomain

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters (identical to the conditional run, minus conditioning)
# ============================================================================

CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "aqsoldb_conditional.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# --- Model architecture (matched to authors' configs/experiment/aqsoldb.yaml) ---
N_LAYERS: int = 9
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 20

# --- Training (matched to authors' AqSolDB recipe, stabilized) ---
EPOCHS: int = 250
BATCH_SIZE: int = 16             # small batch -> many more grad steps/epoch (faster learning); EMA handles stability
LEARNING_RATE: float = 1.5e-4    # decreased from 2e-4 for stability
LR_SCHEDULER: str = "cosine"     # decay LR to settle late-training generation
LR_MIN: float = 1e-6
WEIGHT_DECAY: float = 1e-5
LAMBDA_EDGE: float = 5.0          # edge loss weighted 5x node loss (DeFoG default)
TRAIN_TIME_DISTORTION: str = "polydec"
EMA_DECAY: float = 0.9999        # EMA of weights (~18-epoch trailing avg here); val/eval/best-ckpt sample from EMA
TRAIN_SPLIT: float = 0.9

# :param MOLECULAR_FEATURES:
#     Add per-atom charge/valency + molecular-weight features (matches src).
MOLECULAR_FEATURES: bool = True

# Per-atom nominal valency and atomic weight (for molecular features), keyed by symbol.
ATOM_VALENCY: dict = {
    "C": 4, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1, "Br": 1, "P": 3,
    "I": 1, "Na": 1, "Si": 4, "B": 3,
}
ATOM_WEIGHT_TABLE: dict = {
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "S": 32.06, "Cl": 35.45,
    "Br": 79.904, "P": 30.974, "I": 126.904, "Na": 22.99, "Si": 28.085, "B": 10.81,
}
MAX_ATOM_WEIGHT: float = 350.0

# --- Sampling / evaluation ---
SAMPLE_STEPS: int = 100          # model default (general sampling)
EVAL_SAMPLE_STEPS: int = 1000    # definitive end-of-run eval
GEN_SAMPLE_STEPS: int = 500      # in-training probe steps
GEN_ETA: float = 5.0             # low eta for the in-training probe (unsaturated, trustworthy curve)
ETA: float = 100.0               # error-correction stochasticity (authors' AqSolDB value)
OMEGA: float = 0.3               # target guidance (authors' AqSolDB value)
SAMPLE_TIME_DISTORTION: str = "polydec"   # sampling schedule
NUM_EVAL_SAMPLES: int = 1000
EVAL_CHUNK: int = 32
SAMPLE_VIS_EVERY_K: int = 25

# --- Reproducibility ---
# :param SEED:
#     Global random seed (data split, weight init, DataLoader shuffling). Fixed so
#     runs are reproducible and an LR sweep isolates the LR effect from seed noise.
SEED: int = 42

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
    pl.seed_everything(e.SEED, workers=True)
    e.log(f"global seed set to {e.SEED} (reproducible init / split / shuffling)")

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
    dataset_smiles = []
    skipped = 0
    for _, row in df.iterrows():
        data = smiles_to_pyg_data(row[e.SMILES_COLUMN], atom_encoder, bond_encoder)
        if data is None:
            skipped += 1
            continue
        dataset.append(data)
        dataset_smiles.append(row[e.SMILES_COLUMN])
    e.log(f"Converted {len(dataset)} graphs ({skipped} skipped)")

    from torch_geometric.loader import DataLoader
    n_train = int(len(dataset) * e.TRAIN_SPLIT)
    perm = torch.randperm(len(dataset)).tolist()
    train_set = [dataset[i] for i in perm[:n_train]]
    val_set = [dataset[i] for i in perm[n_train:]]
    train_smiles = [dataset_smiles[i] for i in perm[:n_train]]
    train_loader = DataLoader(train_set, batch_size=e.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=e.BATCH_SIZE) if val_set else None
    e.log(f"Train: {len(train_set)}  Val: {len(val_set)}")

    # Molecular-feature tables aligned to the derived atom vocabulary.
    atom_valencies = [e.ATOM_VALENCY[a] for a in atom_types]
    atom_weights_list = [e.ATOM_WEIGHT_TABLE[a] for a in atom_types]

    # -- Model (cond_dim defaults to 0 -> unconditional) --------------------
    model = DeFoGModel.from_dataloader(
        train_loader,
        n_layers=e.N_LAYERS, hidden_dim=e.HIDDEN_DIM, hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS, dropout=e.DROPOUT, noise_type=e.NOISE_TYPE,
        extra_features_type=e.EXTRA_FEATURES_TYPE, rrwp_steps=e.RRWP_STEPS,
        molecular_features=e.MOLECULAR_FEATURES, atom_valencies=atom_valencies,
        atom_weights=atom_weights_list, max_atom_weight=e.MAX_ATOM_WEIGHT,
        lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY,
        lambda_edge=e.LAMBDA_EDGE, train_time_distortion=e.TRAIN_TIME_DISTORTION,
        lr_scheduler=e.LR_SCHEDULER, lr_min=e.LR_MIN,
        sample_steps=e.SAMPLE_STEPS, eta=e.ETA, omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
    )
    e.log(f"Model: {model}")
    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"Params: {e['model/num_params']:,}  (cond_dim={model.cond_dim})")

    # -- Train --------------------------------------------------------------
    gen_metrics_fn = make_generation_metrics_fn(atom_decoder, bond_decoder, train_smiles)
    # Single source of truth for in-training sampling: BOTH the validity/uniqueness
    # metric probe AND the molecule preview render use these settings, so the
    # rendered previews faithfully reflect the reported validity. (The model-default
    # eta=100 / 100-step sampling is far harsher and made previews look far worse
    # than the metric -- keep these two in lockstep going forward.)
    PROBE_SAMPLE_STEPS = e.GEN_SAMPLE_STEPS
    PROBE_ETA = e.GEN_ETA
    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig),
        generation_metrics_fn=gen_metrics_fn, gen_every_k=10, gen_num_samples=64,
        gen_sample_steps=PROBE_SAMPLE_STEPS, gen_eta=PROBE_ETA, checkpoint_dir=e.path,
    )
    # RDKit-backed molecule depictions for the in-training sample previews
    # (validity/SMILES captions). The metrics path stays independent (above).
    mol_domain = MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_smiles)
    sampler = SampleVisualizationCallback(
        num_samples=8, every_k_epochs=e.SAMPLE_VIS_EVERY_K,
        sample_steps=PROBE_SAMPLE_STEPS, eta=PROBE_ETA,
        domain=mol_domain,
        figure_callback=lambda fig: e.track("samples", fig),
    )
    # EMA first in the list so its weight-swap wraps validation sampling/metrics.
    callbacks = [monitor, sampler]
    if e.EMA_DECAY and e.EMA_DECAY > 0:
        callbacks = [EMACallback(decay=e.EMA_DECAY)] + callbacks
        e.log(f"EMA enabled (decay={e.EMA_DECAY}); validation/eval sample from EMA weights")
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, accelerator="auto", devices=1,
        enable_progress_bar=True, enable_checkpointing=False, logger=False,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model_path = model.save(os.path.join(e.path, "model"))
    e.log(f"Saved final model to {model_path}")

    # -- Evaluation: 1000 unconditional samples (chunked on GPU) ------------
    e.log("=" * 60)
    e.log(f"EVALUATION: {e.NUM_EVAL_SAMPLES} unconditional samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Evaluate the BEST-validity checkpoint if one was captured (late-training
    # generation quality can oscillate, so the final-epoch model may be worse).
    best_path = os.path.join(e.path, "best_model")
    if os.path.exists(best_path + ".ckpt"):
        e.log(f"Loading best-validity checkpoint (best in-train validity="
              f"{monitor.best_validity:.3f})")
        model = DeFoGModel.load(best_path)
    else:
        e.log("No best checkpoint captured; using final-epoch model.")
    model = model.to(device)
    model.eval()
    e.log(f"sampling on {next(model.parameters()).device}")

    samples = []
    remaining = e.NUM_EVAL_SAMPLES
    while remaining > 0:
        cur = min(e.EVAL_CHUNK, remaining)
        samples += model.sample(
            num_samples=cur, sample_steps=e.EVAL_SAMPLE_STEPS, device=device,
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
    e.EVAL_SAMPLE_STEPS = 5
    e.GEN_SAMPLE_STEPS = 5
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
