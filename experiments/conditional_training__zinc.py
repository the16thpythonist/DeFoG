"""
Conditional molecular generation on ZINC 250k, jointly conditioning on THREE
properties (logP + TPSA + QED) via classifier-free guidance.

Same recipe as conditional_training__aqsoldb.py (9-layer/256, RRWP-20, marginal
noise, molecular features, EMA, seed 42, MoleculeDomain previews, SMILES-saving)
but on ZINC 250k, only 50 epochs (large dataset), and generalized to N properties.

After training it evaluates on ALL low/high combinations of the 3 properties
(2^3 = 8 scenarios; low/high = 5th/95th dataset percentile). For each scenario it
writes:
  - dist_<scenario>.png  : per-property histograms (gray dataset, red generated,
                           black target line),
  - grid_<scenario>.png  : a 5x5 grid of random valid molecules (RDKit MoleculeDomain),
  - and per-scenario tagged records in generated_smiles.json.

Graph size is drawn from a property-conditional size distribution.

Usage:
    python experiments/conditional_training__zinc.py
    python experiments/conditional_training__zinc.py --__TESTING__ True
"""
import os
import sys
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import (
    build_encoders,
    smiles_to_pyg_data,
    pyg_data_to_mol,
    mol_to_smiles,
    make_generation_metrics_fn,
    tag_generated_smiles,
)
from experiments.conditional_generation import (
    build_normalization_stats,
    build_condition_vector,
)
from defog.core import (
    DeFoGModel,
    TrainingMonitorCallback,
    SampleVisualizationCallback,
    EMACallback,
    ConditionalSizeDistribution,
)
from defog.domains import MoleculeDomain

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = Path(__file__).parent.parent.resolve()

# ============================================================================
# Parameters
# ============================================================================

CSV_PATH: str = str(_PROJECT_DIR / "data" / "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# :param PROPERTIES:
#     Joint conditioning on logP, TPSA and QED (all regression, z-score normalized).
PROPERTIES: dict = {
    "logp": {"type": "regression", "callback": lambda mol: float(Crippen.MolLogP(mol))},
    "tpsa": {"type": "regression", "callback": lambda mol: float(Descriptors.TPSA(mol))},
    "qed": {"type": "regression", "callback": lambda mol: float(QED.qed(mol))},
}

# --- Model architecture (matches the aqsoldb recipe) ---
N_LAYERS: int = 9
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 20

# Per-atom valency/weight for molecular features (covers ZINC atoms: C,N,O,F,S,Cl,Br,P,I).
MOLECULAR_FEATURES: bool = True
ATOM_VALENCY: dict = {
    "C": 4, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1, "Br": 1, "P": 3,
    "I": 1, "Na": 1, "Si": 4, "B": 3,
}
ATOM_WEIGHT_TABLE: dict = {
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "S": 32.06, "Cl": 35.45,
    "Br": 79.904, "P": 30.974, "I": 126.904, "Na": 22.99, "Si": 28.085, "B": 10.81,
}
MAX_ATOM_WEIGHT: float = 350.0

# --- Conditioning ---
COND_DROP_PROB: float = 0.1
GUIDANCE_SCALE: float = 2.0

# --- Training. batch 24 was the sweep-optimal batch size (best NUV). On 250k
#     mols that is ~10.4k grad steps/epoch, so 20 epochs => ~208k steps -- ~2x
#     the good aqsoldb budget (~91k), plenty while cutting wall-clock vs 50. ---
EPOCHS: int = 20
BATCH_SIZE: int = 24
LEARNING_RATE: float = 4e-4
LR_SCHEDULER: str = "cosine"
LR_MIN: float = 1e-6
WEIGHT_DECAY: float = 1e-5
LAMBDA_EDGE: float = 5.0
TRAIN_TIME_DISTORTION: str = "polydec"
EMA_DECAY: float = 0.9999
TRAIN_SPLIT: float = 0.9

# --- Sampling / evaluation ---
SAMPLE_STEPS: int = 100
EVAL_SAMPLE_STEPS: int = 1000
GEN_SAMPLE_STEPS: int = 500
GEN_ETA: float = 5.0
EVAL_CHUNK: int = 32
ETA: float = 100.0
OMEGA: float = 0.3
SAMPLE_TIME_DISTORTION: str = "polydec"

SIZE_DIST_METHOD: str = "kernel"      # property-conditional graph size

# :param TARGET_PERCENTILES / LEVEL_NAMES:
#     "low"/"high" target levels per property, from these dataset percentiles.
TARGET_PERCENTILES: list = [5, 95]
LEVEL_NAMES: list = ["low", "high"]

NUM_EVAL_SAMPLES: int = 500           # molecules generated per scenario
GRID_SIZE: int = 25                   # 5x5 example grid
SAMPLE_VIS_EVERY_K: int = 10

# --- Reproducibility / special ---
SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ============================================================================
# Helpers
# ============================================================================

def derive_atom_types(smiles_list) -> list:
    counts = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    return [sym for sym, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def target_condition_vector(targets, properties, norm_stats) -> torch.Tensor:
    parts = []
    for name, _cfg in sorted(properties.items()):
        mean = norm_stats[name]["mean"]
        std = norm_stats[name]["std"]
        z = (float(targets[name]) - mean) / std if std > 0 else 0.0
        parts.append(torch.tensor([z], dtype=torch.float))
    return torch.cat(parts)


def plot_dist(dataset_df, generated, targets, property_names, title):
    fig, axes = plt.subplots(1, len(property_names), figsize=(5.5 * len(property_names), 4.5))
    if len(property_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, property_names):
        data_vals = dataset_df[name].values
        gen_vals = np.array(generated.get(name, []))
        bins = np.linspace(float(np.min(data_vals)), float(np.max(data_vals)), 40)
        ax.hist(data_vals, bins=bins, density=True, color="0.7", label="dataset", alpha=0.9)
        if gen_vals.size > 0:
            ax.hist(gen_vals, bins=bins, density=True, color="red", label="generated", alpha=0.55)
        ax.axvline(targets[name], color="black", linestyle="--", linewidth=2,
                   label=f"target={targets[name]:.2f}")
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend(fontsize="small")
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
    e.log("ZINC 250k conditional generation (logP + TPSA + QED)")
    pl.seed_everything(e.SEED, workers=True)
    e.log(f"global seed set to {e.SEED}")

    # -- Data ---------------------------------------------------------------
    df = pd.read_csv(e.CSV_PATH)
    e.log(f"Loaded {len(df)} molecules from {e.CSV_PATH}")

    atom_types = derive_atom_types(df[e.SMILES_COLUMN])
    e.log(f"Atom vocabulary ({len(atom_types)}): {atom_types}")
    e["config/atom_types"] = atom_types
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, e.BOND_TYPES)

    property_names = sorted(e.PROPERTIES.keys())
    norm_stats = build_normalization_stats(e.PROPERTIES, df)
    e["config/norm_stats"] = norm_stats
    e.log(f"Normalization: {norm_stats}")

    dataset, dataset_smiles, skipped = [], [], 0
    for _, row in df.iterrows():
        data = smiles_to_pyg_data(row[e.SMILES_COLUMN], atom_encoder, bond_encoder)
        if data is None:
            skipped += 1
            continue
        data.y = build_condition_vector(row, e.PROPERTIES, norm_stats).unsqueeze(0)
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

    # -- Model --------------------------------------------------------------
    atom_valencies = [e.ATOM_VALENCY[a] for a in atom_types]
    atom_weights_list = [e.ATOM_WEIGHT_TABLE[a] for a in atom_types]
    cond_dim = len(property_names)
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
        cond_dim=cond_dim, cond_drop_prob=e.COND_DROP_PROB, guidance_scale=e.GUIDANCE_SCALE,
    )
    e.log(f"Model: {model}  (cond_dim={cond_dim})")
    e["model/num_params"] = sum(p.numel() for p in model.parameters())

    # -- Train --------------------------------------------------------------
    gen_metrics_fn = make_generation_metrics_fn(atom_decoder, bond_decoder, train_smiles)
    PROBE_SAMPLE_STEPS, PROBE_ETA = e.GEN_SAMPLE_STEPS, e.GEN_ETA
    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig),
        generation_metrics_fn=gen_metrics_fn, gen_every_k=10, gen_num_samples=64,
        gen_sample_steps=PROBE_SAMPLE_STEPS, gen_eta=PROBE_ETA, checkpoint_dir=e.path,
    )
    mol_domain = MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_smiles)
    sampler = SampleVisualizationCallback(
        num_samples=8, every_k_epochs=e.SAMPLE_VIS_EVERY_K,
        sample_steps=PROBE_SAMPLE_STEPS, eta=PROBE_ETA, domain=mol_domain,
        figure_callback=lambda fig: e.track("samples", fig),
    )
    callbacks = [monitor, sampler]
    if e.EMA_DECAY and e.EMA_DECAY > 0:
        callbacks = [EMACallback(decay=e.EMA_DECAY)] + callbacks
        e.log(f"EMA enabled (decay={e.EMA_DECAY})")
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, accelerator="auto", devices=1,
        enable_progress_bar=True, enable_checkpointing=False, logger=False, callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    e.log(f"Saved model to {model.save(os.path.join(e.path, 'model'))}")

    # -- Evaluation: all low/high combinations of the 3 properties ----------
    e.log("=" * 60)
    e.log(f"EVALUATION: {len(e.LEVEL_NAMES) ** len(property_names)} low/high scenarios")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_path = os.path.join(e.path, "best_model")
    if os.path.exists(best_path + ".ckpt"):
        e.log(f"Loading best-validity checkpoint (best={monitor.best_validity:.3f})")
        model = DeFoGModel.load(best_path)
    model = model.to(device)
    model.eval()

    size_dist = None
    if e.SIZE_DIST_METHOD in ("kernel", "regression"):
        size_dist = ConditionalSizeDistribution.from_dataloader(train_loader, method=e.SIZE_DIST_METHOD)
        e.log(f"Conditional size distribution (method={e.SIZE_DIST_METHOD})")

    levels = {name: dict(zip(e.LEVEL_NAMES, np.percentile(df[name].values, e.TARGET_PERCENTILES)))
              for name in property_names}
    e["eval/levels"] = {n: {k: float(v) for k, v in lv.items()} for n, lv in levels.items()}
    e.log(f"Target levels: {e['eval/levels']}")

    dom_eval = MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_smiles)
    grid_metrics, all_generated = [], []
    random.seed(e.SEED)

    for combo in itertools.product(e.LEVEL_NAMES, repeat=len(property_names)):
        combo_levels = dict(zip(property_names, combo))              # {logp:'low', tpsa:'high', ...}
        targets = {n: float(levels[n][lv]) for n, lv in combo_levels.items()}
        tag = "_".join(f"{n}-{lv}" for n, lv in combo_levels.items())
        cond_vec = target_condition_vector(targets, e.PROPERTIES, norm_stats)

        samples, remaining = [], e.NUM_EVAL_SAMPLES
        while remaining > 0:
            cur = min(e.EVAL_CHUNK, remaining)
            condition = cond_vec.unsqueeze(0).expand(cur, -1)
            samples += model.sample(
                num_samples=cur, condition=condition, guidance_scale=e.GUIDANCE_SCALE,
                sample_steps=e.EVAL_SAMPLE_STEPS, eta=e.ETA, omega=e.OMEGA,
                time_distortion=e.SAMPLE_TIME_DISTORTION, size_dist=size_dist,
                device=device, show_progress=False,
            )
            remaining -= cur

        # property distributions of the generated valid molecules
        generated = {n: [] for n in property_names}
        n_valid = 0
        for s in samples:
            mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
            smi = mol_to_smiles(mol) if mol is not None else None
            emol = Chem.MolFromSmiles(smi) if smi else None
            if emol is None:
                continue
            n_valid += 1
            for n in property_names:
                try:
                    generated[n].append(e.PROPERTIES[n]["callback"](emol))
                except Exception:
                    pass

        title = f"{tag}  (valid {n_valid}/{e.NUM_EVAL_SAMPLES})"
        e.commit_fig(f"dist_{tag}.png", plot_dist(df, generated, targets, property_names, title))

        # 5x5 grid of random valid molecules (RDKit MoleculeDomain)
        valid = [s for s in samples if dom_eval.is_valid(s)]
        random.shuffle(valid)
        grid = valid[:e.GRID_SIZE]
        gfig, axes = plt.subplots(5, 5, figsize=(15, 15))
        for i, ax in enumerate(axes.flatten()):
            if i < len(grid):
                dom_eval.render(ax, grid[i])
                cap = dom_eval.caption(grid[i])
                if cap:
                    ax.set_title(cap, fontsize=6)
            else:
                ax.axis("off")
        gfig.suptitle(f"ZINC conditional -- {tag} -- {len(grid)} random valid molecules", fontsize=12)
        gfig.tight_layout(rect=[0, 0, 1, 0.96])
        e.commit_fig(f"grid_{tag}.png", gfig)
        plt.close("all")

        recs = tag_generated_smiles(samples, atom_decoder, bond_decoder, train_smiles)
        for r in recs:
            r["scenario"] = tag
            for n in property_names:
                r[f"target_{n}"] = targets[n]
                r[f"level_{n}"] = combo_levels[n]
        all_generated.extend(recs)

        row = {"scenario": tag, "n_valid": n_valid}
        for n in property_names:
            row[f"target_{n}"] = targets[n]
            row[f"level_{n}"] = combo_levels[n]
            if generated[n]:
                row[f"{n}_mean"] = float(np.mean(generated[n]))
                row[f"{n}_mae"] = float(np.mean(np.abs(np.array(generated[n]) - targets[n])))
        grid_metrics.append(row)
        e.log(f"  {tag}: valid={n_valid}/{e.NUM_EVAL_SAMPLES}  "
              + "  ".join(f"{n}_mae={row.get(f'{n}_mae', float('nan')):.3f}" for n in property_names))

    e.commit_json("grid_metrics.json", grid_metrics)
    e.commit_json("generated_smiles.json", all_generated)
    e.log(f"saved {len(all_generated)} generated molecules -> generated_smiles.json")
    e.log("Evaluation complete.")


@experiment.testing
def testing(e: Experiment):
    """Tiny end-to-end smoke run (all 8 scenarios, few samples)."""
    e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.SAMPLE_STEPS = 5
    e.EVAL_SAMPLE_STEPS = 5
    e.GEN_SAMPLE_STEPS = 5
    e.EVAL_CHUNK = 8
    e.NUM_EVAL_SAMPLES = 16
    e.SAMPLE_VIS_EVERY_K = 1
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2
    df = pd.read_csv(e.CSV_PATH).head(200)
    smoke = os.path.join(folder_path(__file__), "_zinc_smoke.csv")
    df.to_csv(smoke, index=False)
    e.CSV_PATH = smoke


experiment.run_if_main()
