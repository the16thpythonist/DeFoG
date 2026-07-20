"""
Unconditional DeFoG foundation-model training on ChEMBL 37 (~2.44M molecules).

This is the first "release" base model: a larger-capacity, unconditional DeFoG
trained on the cleaned ChEMBL set produced by ``scripts/prepare_chembl.py``.
Downstream adapters / guidance / RL all bind to the frozen categorical schema
below, so the vocabulary and edge classes are hard-coded (NOT derived from data).

Key differences from the AqSolDB template:
  - FROZEN 12-element organic vocabulary (public contract), not data-derived.
  - Larger model: 12 layers / hidden 384 / 12 heads.
  - Lazy SMILES->graph dataset (2.44M graphs are never materialized at once).
  - Model built directly from the frozen marginals + size histogram in
    ``data/chembl/chembl_stats.json`` (no full stats pass over 2.44M graphs).
  - Extended evaluation: validity / uniqueness / novelty + connected /
    disconnected / sanity / ring diagnostics + logP/TPSA/QED KL divergence.

Usage:
    python experiments/training__chembl_uncond.py --__TESTING__ True   # smoke
    python experiments/training__chembl_uncond.py                       # full
"""
import json
import os

import numpy as np
import torch
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import (
    build_encoders,
    smiles_to_pyg_data,
    make_generation_metrics_fn,
    molecular_metrics,
    property_distributions,
    tag_generated_smiles,
)
from defog.core import (
    DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback, EMACallback
)
from defog.domains import MoleculeDomain

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHEMBL_DIR = os.path.join(_PROJECT_DIR, "data", "chembl")

# ============================================================================
# Parameters
# ============================================================================

# --- Data (produced by scripts/prepare_chembl.py) ---------------------------
TRAIN_SMILES_PATH: str = os.path.join(_CHEMBL_DIR, "chembl_train.smiles")
VAL_SMILES_PATH: str = os.path.join(_CHEMBL_DIR, "chembl_val.smiles")
STATS_PATH: str = os.path.join(_CHEMBL_DIR, "chembl_stats.json")
REF_DESC_PATH: str = os.path.join(_CHEMBL_DIR, "chembl_ref_descriptors.npz")

# FROZEN public-contract schema (must match scripts/prepare_chembl.py).
ATOM_DECODER: list = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# :param MAX_TRAIN / MAX_VAL:
#     Cap the number of SMILES read (None = all). Used to shrink smoke tests.
MAX_TRAIN: int = None
MAX_VAL: int = None

# --- Model architecture (moderate scale-up: ~12L / 384 / 12 heads) ----------
N_LAYERS: int = 12
HIDDEN_DIM: int = 384
HIDDEN_MLP_DIM: int = 768
N_HEADS: int = 12
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 20

# :param MOLECULAR_FEATURES:
#     Per-atom charge/valency + molecular-weight features (helps valence
#     correctness -> validity/sanity). Aligned to ATOM_DECODER order below.
MOLECULAR_FEATURES: bool = True
ATOM_VALENCY: dict = {
    "C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Br": 1, "Cl": 1, "I": 1,
    "P": 5, "S": 6, "Se": 2, "Si": 4,
}
ATOM_WEIGHT_TABLE: dict = {
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "B": 10.81, "Br": 79.904,
    "Cl": 35.45, "I": 126.904, "P": 30.974, "S": 32.06, "Se": 78.971, "Si": 28.085,
}
MAX_ATOM_WEIGHT: float = 700.0   # normalizer for total MW (<=48 heavy atoms)

# --- Training (placeholders; finalized in the HPC launch discussion) --------
EPOCHS: int = 100
# :param MAX_TIME_HOURS:
#     Wall-clock cap for trainer.fit so end-of-run eval still fits inside the
#     SLURM walltime (JUPITER hard limit = 12h). None = no cap (local runs).
MAX_TIME_HOURS: float = None
BATCH_SIZE: int = 256
NUM_WORKERS: int = 8
LEARNING_RATE: float = 2e-4
LR_SCHEDULER: str = "cosine"
LR_MIN: float = 1e-6
WEIGHT_DECAY: float = 1e-5
LAMBDA_EDGE: float = 5.0
TRAIN_TIME_DISTORTION: str = "polydec"
EMA_DECAY: float = 0.9999

# --- Sampling / evaluation --------------------------------------------------
SAMPLE_STEPS: int = 100
EVAL_SAMPLE_STEPS: int = 500      # end-of-run eval sampling steps
GEN_SAMPLE_STEPS: int = 250       # in-training probe steps
GEN_ETA: float = 5.0
ETA: float = 0.0
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "polydec"
NUM_EVAL_SAMPLES: int = 1000
EVAL_CHUNK: int = 64
KL_REF_SIZE: int = 25000          # reference molecules for the KL descriptors
SAMPLE_VIS_EVERY_K: int = 5
GEN_EVERY_K: int = 2
GEN_NUM_SAMPLES: int = 64

SEED: int = 42

__DEBUG__: bool = True
__TESTING__: bool = False


def read_smiles(path: str, limit=None) -> list:
    smis = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            s = line.strip()
            if s:
                smis.append(s)
    return smis


class SmilesGraphDataset(torch.utils.data.Dataset):
    """Lazy SMILES -> PyG Data conversion (keeps 2.44M graphs off the heap).

    All cleaned ChEMBL SMILES are guaranteed convertible (validated), so
    ``smiles_to_pyg_data`` should never return None here; the fallback loop is a
    defensive guard so a stray unconvertible entry can't crash a training run.
    """

    def __init__(self, smiles, atom_encoder, bond_encoder):
        self.smiles = smiles
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        n = len(self.smiles)
        for off in range(n):
            j = (idx + off) % n
            data = smiles_to_pyg_data(self.smiles[j], self.atom_encoder, self.bond_encoder)
            if data is not None:
                return data
        raise RuntimeError("no convertible SMILES in dataset")


def build_model_from_stats(e, stats, atom_valencies, atom_weights_list):
    """Construct DeFoGModel directly from frozen ChEMBL statistics."""
    node_marginals = torch.tensor(stats["node_marginals"], dtype=torch.float)
    edge_marginals = torch.tensor(stats["edge_marginals"], dtype=torch.float)
    max_nodes = int(stats["max_nodes"])
    node_counts = torch.zeros(max_nodes + 1)
    for k, v in stats["size_histogram"].items():
        node_counts[int(k)] = float(v)

    return DeFoGModel(
        num_node_classes=int(stats["num_node_classes"]),
        num_edge_classes=int(stats["num_edge_classes"]),
        n_layers=e.N_LAYERS, hidden_dim=e.HIDDEN_DIM, hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS, dropout=e.DROPOUT,
        noise_type=e.NOISE_TYPE, node_marginals=node_marginals,
        edge_marginals=edge_marginals, node_counts=node_counts, max_nodes=max_nodes,
        extra_features_type=e.EXTRA_FEATURES_TYPE, rrwp_steps=e.RRWP_STEPS,
        molecular_features=e.MOLECULAR_FEATURES, atom_valencies=atom_valencies,
        atom_weights=atom_weights_list, max_atom_weight=e.MAX_ATOM_WEIGHT,
        lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY, lambda_edge=e.LAMBDA_EDGE,
        train_time_distortion=e.TRAIN_TIME_DISTORTION,
        lr_scheduler=e.LR_SCHEDULER, lr_min=e.LR_MIN,
        sample_steps=e.SAMPLE_STEPS, eta=e.ETA, omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
    )


def load_reference_descriptors(e, train_smiles):
    """logP/TPSA/QED reference distributions (cached to disk across runs)."""
    if os.path.exists(e.REF_DESC_PATH):
        with np.load(e.REF_DESC_PATH) as z:
            return {k: z[k] for k in z.files}
    ref = property_distributions(train_smiles, max_n=e.KL_REF_SIZE, seed=e.SEED)
    try:
        np.savez(e.REF_DESC_PATH, **ref)
    except Exception:
        pass
    return ref


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment) -> None:
    e.log("ChEMBL UNCONDITIONAL foundation-model training")
    pl.seed_everything(e.SEED, workers=True)

    # -- Frozen schema ------------------------------------------------------
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(
        e.ATOM_DECODER, e.BOND_TYPES
    )
    e["config/atom_decoder"] = list(e.ATOM_DECODER)
    e.log(f"Frozen vocab ({len(e.ATOM_DECODER)}): {e.ATOM_DECODER}")

    # -- Data (lazy) --------------------------------------------------------
    train_smiles = read_smiles(e.TRAIN_SMILES_PATH, e.MAX_TRAIN)
    val_smiles = read_smiles(e.VAL_SMILES_PATH, e.MAX_VAL)
    e.log(f"Train SMILES: {len(train_smiles):,}  Val SMILES: {len(val_smiles):,}")

    from torch_geometric.loader import DataLoader
    train_set = SmilesGraphDataset(train_smiles, atom_encoder, bond_encoder)
    val_set = SmilesGraphDataset(val_smiles, atom_encoder, bond_encoder)
    train_loader = DataLoader(
        train_set, batch_size=e.BATCH_SIZE, shuffle=True,
        num_workers=e.NUM_WORKERS, persistent_workers=e.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=e.BATCH_SIZE, num_workers=e.NUM_WORKERS,
        persistent_workers=e.NUM_WORKERS > 0,
    ) if val_smiles else None

    # -- Model (built from frozen stats; no 2.44M stats pass) ---------------
    with open(e.STATS_PATH) as fh:
        stats = json.load(fh)
    atom_valencies = [e.ATOM_VALENCY[a] for a in e.ATOM_DECODER]
    atom_weights_list = [e.ATOM_WEIGHT_TABLE[a] for a in e.ATOM_DECODER]
    model = build_model_from_stats(e, stats, atom_valencies, atom_weights_list)
    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"Model params: {e['model/num_params']:,}  max_nodes={stats['max_nodes']}")

    # -- Callbacks ----------------------------------------------------------
    gen_metrics_fn = make_generation_metrics_fn(atom_decoder, bond_decoder, train_smiles)
    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig),
        generation_metrics_fn=gen_metrics_fn, gen_every_k=e.GEN_EVERY_K,
        gen_num_samples=e.GEN_NUM_SAMPLES, gen_sample_steps=e.GEN_SAMPLE_STEPS,
        gen_eta=e.GEN_ETA, checkpoint_dir=e.path,
    )
    mol_domain = MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_smiles)
    sampler = SampleVisualizationCallback(
        num_samples=8, every_k_epochs=e.SAMPLE_VIS_EVERY_K,
        sample_steps=e.GEN_SAMPLE_STEPS, eta=e.GEN_ETA, domain=mol_domain,
        figure_callback=lambda fig: e.track("samples", fig),
    )
    callbacks = [monitor, sampler]
    if e.EMA_DECAY and e.EMA_DECAY > 0:
        callbacks = [EMACallback(decay=e.EMA_DECAY)] + callbacks
        e.log(f"EMA enabled (decay={e.EMA_DECAY})")

    max_time = None
    if e.MAX_TIME_HOURS:
        hrs = int(e.MAX_TIME_HOURS)
        mins = int(round((e.MAX_TIME_HOURS - hrs) * 60))
        max_time = {"hours": hrs, "minutes": mins}
        e.log(f"Trainer max_time = {hrs}h{mins:02d}m (leaves room for eval before SLURM kill)")
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, max_time=max_time, accelerator="auto", devices=1,
        enable_progress_bar=True, enable_checkpointing=False, logger=False,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model_path = model.save(os.path.join(e.path, "model"))
    e.log(f"Saved final model to {model_path}")

    # -- Evaluation (extended metric suite) ---------------------------------
    e.log("=" * 60)
    e.log(f"EVALUATION: {e.NUM_EVAL_SAMPLES} unconditional samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_path = os.path.join(e.path, "best_model")
    if os.path.exists(best_path + ".ckpt"):
        e.log(f"Loading best-validity checkpoint (best={monitor.best_validity:.3f})")
        model = DeFoGModel.load(best_path)
    model = model.to(device).eval()

    samples = []
    remaining = e.NUM_EVAL_SAMPLES
    while remaining > 0:
        cur = min(e.EVAL_CHUNK, remaining)
        samples += model.sample(num_samples=cur, sample_steps=e.EVAL_SAMPLE_STEPS,
                                device=device, show_progress=False)
        remaining -= cur

    ref_desc = load_reference_descriptors(e, train_smiles)
    metrics = molecular_metrics(
        samples, atom_decoder, bond_decoder,
        reference_smiles=set(train_smiles), reference_descriptors=ref_desc,
        compute_kl=True,
    )
    for k, v in metrics.items():
        e[f"eval/{k}"] = v
    e.commit_json("chembl_uncond_metrics.json", metrics)
    e.log("-- extended metrics --")
    for k in ("validity", "uniqueness", "novelty", "connected", "disconnected",
              "sanity", "wonky_ring_frac", "kl_logp", "kl_tpsa", "kl_qed", "kl_score"):
        if k in metrics:
            e.log(f"  {k:16s} = {metrics[k]:.4f}")

    records = tag_generated_smiles(samples, atom_decoder, bond_decoder, train_smiles)
    e.commit_json("generated_smiles.json", records)
    e.log(f"saved {len(records)} generated molecules")


@experiment.testing
def testing(e: Experiment):
    e.MAX_TRAIN = 400
    e.MAX_VAL = 100
    e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.NUM_WORKERS = 0
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2
    e.RRWP_STEPS = 5
    e.SAMPLE_STEPS = 5
    e.EVAL_SAMPLE_STEPS = 5
    e.GEN_SAMPLE_STEPS = 5
    e.NUM_EVAL_SAMPLES = 20
    e.EVAL_CHUNK = 10
    e.KL_REF_SIZE = 200
    e.SAMPLE_VIS_EVERY_K = 1
    e.GEN_EVERY_K = 1
    # don't clobber the real cached reference descriptors during a smoke test
    e.REF_DESC_PATH = os.path.join(folder_path(__file__), "_chembl_ref_desc_smoke.npz")


experiment.run_if_main()
