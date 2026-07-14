"""
Unconditional DeFoG training on the full GuacaMol dataset (ChEMBL-derived, 1.59M
molecules), for a single-node 4-GPU replica run on KCIST.

Best-known unconditional recipe (batch 24, LR 4e-4, 9L/256, RRWP-20, marginal
noise, molecular features, EMA) but on GuacaMol. Runs are wall-clock bounded via
Lightning `max_time` (~22h train) rather than a fixed epoch count -- EPOCHS is
just a high cap. Four arms differ only by seed (42/43/44/45).

GuacaMol specifics vs ZINC: `.smiles` file (no header, one SMILES/line); adds Se
to the atom tables; larger molecules (up to ~76 heavy atoms) -> higher
MAX_ATOM_WEIGHT, smaller EVAL_CHUNK; probes every 2 epochs so a best_model
checkpoint exists even though only ~8-12 epochs fit in 22h.

End-of-run eval: 2500 chunked samples -> NUV + KDE-KL (logP/QED/SAS/TPSA vs a
GuacaMol reference) + gray-dataset/foreground-generated plots.

Usage:
    python experiments/training__guacamol_uncond.py --SEED 43
    python experiments/training__guacamol_uncond.py --__TESTING__ True
"""
import os
import sys
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, RDConfig
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

from experiments.utils import (  # noqa: E402
    build_encoders,
    smiles_to_pyg_data,
    pyg_data_to_mol,
    mol_to_smiles,
    make_generation_metrics_fn,
    tag_generated_smiles,
)
from defog.core import (  # noqa: E402
    DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback, EMACallback
)
from defog.domains import MoleculeDomain  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================

CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "guacamol", "guacamol_all.smiles")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# --- Model architecture (best unconditional recipe) ---
N_LAYERS: int = 9
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 20

# --- Training: wall-clock bounded (~22h train) via Lightning max_time. EPOCHS is
#     just a cap that won't be reached (~8-12 epochs fit on 1.59M mols/22h). ---
EPOCHS: int = 100
MAX_TIME_HOURS: int = 22
BATCH_SIZE: int = 24
LEARNING_RATE: float = 4e-4
LR_SCHEDULER: str = "cosine"
LR_MIN: float = 1e-6
WEIGHT_DECAY: float = 1e-5
LAMBDA_EDGE: float = 5.0
TRAIN_TIME_DISTORTION: str = "polydec"
EMA_DECAY: float = 0.9999
TRAIN_SPLIT: float = 0.95            # 1.59M mols -> plenty; small val slice

MOLECULAR_FEATURES: bool = True
ATOM_VALENCY: dict = {
    "C": 4, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1, "Br": 1, "P": 3,
    "I": 1, "Si": 4, "B": 3, "Se": 2, "Na": 1,
}
ATOM_WEIGHT_TABLE: dict = {
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "S": 32.06, "Cl": 35.45,
    "Br": 79.904, "P": 30.974, "I": 126.904, "Si": 28.085, "B": 10.81,
    "Se": 78.971, "Na": 22.99,
}
MAX_ATOM_WEIGHT: float = 1000.0      # GuacaMol mols reach ~76 heavy atoms (~1000 Da)

# --- Sampling / evaluation ---
SAMPLE_STEPS: int = 100
EVAL_SAMPLE_STEPS: int = 500         # large mols -> 500 steps keeps eval ~affordable
GEN_SAMPLE_STEPS: int = 500
GEN_ETA: float = 5.0
ETA: float = 100.0
OMEGA: float = 0.3
SAMPLE_TIME_DISTORTION: str = "polydec"
GEN_EVERY_K: int = 2                 # probe often (few epochs) so best_model is captured
SAMPLE_VIS_EVERY_K: int = 2

NUM_EVAL_SAMPLES: int = 2500
EVAL_CHUNK: int = 25                 # smaller: GuacaMol mols up to 76 atoms
REF_SUBSAMPLE: int = 10000
KL_BINS: int = 60

SEED: int = 42                       # swept per-arm (42/43/44/45)
__DEBUG__: bool = False
__TESTING__: bool = False

PROPERTY_FNS: dict = {
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "qed": lambda m: float(QED.qed(m)),
    "sas": lambda m: float(sascorer.calculateScore(m)),
    "tpsa": lambda m: float(Descriptors.TPSA(m)),
}


def derive_atom_types(smiles_list) -> list:
    counts = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for a in mol.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def compute_props(smiles_iter, fns):
    out = {p: [] for p in fns}
    for smi in smiles_iter:
        if smi is None:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for p, fn in fns.items():
            try:
                out[p].append(fn(m))
            except Exception:
                pass
    return {p: np.asarray(v, dtype=float) for p, v in out.items()}


def property_kl_and_bins(ref, gen, n_bins_plot):
    """GuacaMol-style KL(P_dataset || Q_generated) via Gaussian KDE over the
    dataset 0.5-99.5 percentile range, plus histogram bins for plotting."""
    default_bins = np.linspace(0, 1, n_bins_plot + 1)
    if ref.size < 2 or gen.size < 10:
        return float("nan"), default_bins
    lo, hi = np.percentile(ref, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, n_bins_plot + 1)
    if np.std(gen) < 1e-9 or np.std(ref) < 1e-9:
        return float("nan"), bins
    grid = np.linspace(lo, hi, 1000)
    try:
        P = gaussian_kde(ref)(grid)
        Q = gaussian_kde(gen)(grid)
    except Exception:
        return float("nan"), bins
    P = P / P.sum(); Q = Q / Q.sum()
    eps = 1e-12
    P = P + eps; Q = Q + eps
    P = P / P.sum(); Q = Q / Q.sum()
    return float(np.sum(P * np.log(P / Q))), bins


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"GuacaMol UNCONDITIONAL training (seed={e.SEED}, max_time={e.MAX_TIME_HOURS}h)")
    pl.seed_everything(e.SEED, workers=True)

    # -- Data (GuacaMol .smiles: no header, one SMILES per line) ------------
    df = pd.read_csv(e.CSV_PATH, header=None, names=[e.SMILES_COLUMN])
    e.log(f"Loaded {len(df)} molecules from {e.CSV_PATH}")
    atom_types = derive_atom_types(df[e.SMILES_COLUMN].iloc[:50000])   # vocab from a big sample
    e.log(f"Atom vocabulary ({len(atom_types)}): {atom_types}")
    e["config/atom_types"] = atom_types
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, e.BOND_TYPES)

    dataset, dataset_smiles, skipped = [], [], 0
    for smi in df[e.SMILES_COLUMN]:
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder)
        if data is None:
            skipped += 1
            continue
        dataset.append(data)
        dataset_smiles.append(smi)
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

    # .get() defaults so an unexpected element can never KeyError a 24h run.
    atom_valencies = [e.ATOM_VALENCY.get(a, 4) for a in atom_types]
    atom_weights_list = [e.ATOM_WEIGHT_TABLE.get(a, 50.0) for a in atom_types]

    # -- Model (unconditional) ---------------------------------------------
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
    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"Model: {model}  Params: {e['model/num_params']:,}  (cond_dim={model.cond_dim})")

    # -- Train (wall-clock bounded) ----------------------------------------
    gen_metrics_fn = make_generation_metrics_fn(atom_decoder, bond_decoder, train_smiles)
    PROBE_SAMPLE_STEPS, PROBE_ETA = e.GEN_SAMPLE_STEPS, e.GEN_ETA
    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig),
        generation_metrics_fn=gen_metrics_fn, gen_every_k=e.GEN_EVERY_K, gen_num_samples=64,
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
        max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS},
        accelerator="auto", devices=1,
        enable_progress_bar=True, enable_checkpointing=False, logger=False, callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    e.log(f"Training stopped after {trainer.current_epoch} epochs")
    e.log(f"Saved final model to {model.save(os.path.join(e.path, 'model'))}")

    # -- Evaluation: 2500 unconditional samples (chunked) -------------------
    e.log("=" * 60)
    e.log(f"EVALUATION: {e.NUM_EVAL_SAMPLES} unconditional samples (chunk={e.EVAL_CHUNK})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_path = os.path.join(e.path, "best_model")
    if os.path.exists(best_path + ".ckpt"):
        e.log(f"Loading best-validity checkpoint (best={monitor.best_validity:.3f})")
        model = DeFoGModel.load(best_path)
    model = model.to(device)
    model.eval()

    samples, remaining = [], e.NUM_EVAL_SAMPLES
    while remaining > 0:
        cur = min(e.EVAL_CHUNK, remaining)
        samples += model.sample(num_samples=cur, sample_steps=e.EVAL_SAMPLE_STEPS,
                                device=device, show_progress=False)
        remaining -= cur
    e.log(f"generated {len(samples)} samples")

    records = tag_generated_smiles(samples, atom_decoder, bond_decoder, train_smiles)
    tot = len(records)
    n_valid = sum(r["valid"] for r in records)
    n_unique = sum(1 for r in records if r["valid"] and r["unique"])
    n_nuv = sum(1 for r in records if r["valid"] and r["unique"] and r["novel"])
    validity = n_valid / tot
    uniqueness = n_unique / n_valid if n_valid else 0.0
    novelty = n_nuv / n_unique if n_unique else 0.0
    nuv = n_nuv / tot
    e.commit_json("generated_smiles.json", records)
    e.log(f"validity={validity:.3f} uniqueness={uniqueness:.3f} novelty={novelty:.3f} NUV={nuv:.3f}")

    # -- KL divergence on cheap properties ----------------------------------
    ref_smiles = random.sample(dataset_smiles, min(e.REF_SUBSAMPLE, len(dataset_smiles)))
    ref_props = compute_props(ref_smiles, e.PROPERTY_FNS)
    gen_smiles = [r["smiles"] for r in records if r["valid"]]
    gen_props = compute_props(gen_smiles, e.PROPERTY_FNS)

    kls, norms = {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, p in zip(axes.flatten(), e.PROPERTY_FNS.keys()):
        ref, gen = ref_props[p], gen_props[p]
        kl, bins = property_kl_and_bins(ref, gen, e.KL_BINS)
        kls[p] = kl
        norms[p] = float(np.exp(-kl)) if kl == kl else float("nan")
        ax.hist(ref, bins=bins, density=True, color="0.6", label="dataset", zorder=1)
        ax.hist(gen, bins=bins, density=True, histtype="stepfilled", color="crimson",
                alpha=0.55, label="generated", zorder=2)
        ax.set_title(f"{p}   KL={kl:.3f}   exp(-KL)={norms[p]:.3f}")
        ax.set_xlabel(p); ax.set_ylabel("density"); ax.legend(fontsize="small")
        f1, a1 = plt.subplots(figsize=(6.5, 4.5))
        a1.hist(ref, bins=bins, density=True, color="0.6", label="dataset", zorder=1)
        a1.hist(gen, bins=bins, density=True, histtype="stepfilled", color="crimson",
                alpha=0.55, label="generated", zorder=2)
        a1.set_title(f"{p}   KL={kl:.3f}   exp(-KL)={norms[p]:.3f}")
        a1.set_xlabel(p); a1.set_ylabel("density"); a1.legend(fontsize="small")
        f1.tight_layout()
        e.commit_fig(f"dist_{p}.png", f1)

    avg_kl = float(np.mean([v for v in kls.values() if v == v]))
    norm_score = float(np.mean([v for v in norms.values() if v == v]))
    fig.suptitle(f"GuacaMol uncond seed={e.SEED}  |  NUV={nuv:.3f}  "
                 f"avg-KL={avg_kl:.3f}  normalized-KL={norm_score:.3f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    e.commit_fig("kl_distributions.png", fig)

    e.commit_json("uncond_metrics.json", {
        "num_samples": tot, "num_valid": n_valid,
        "validity": validity, "uniqueness": uniqueness, "novelty": novelty, "nuv": nuv,
        "kl_per_property": kls, "kl_normalized_per_property": norms,
        "avg_kl": avg_kl, "normalized_kl_score": norm_score,
        "seed": e.SEED, "epochs_trained": int(trainer.current_epoch),
    })
    e.log(f"avg-KL={avg_kl:.3f}  normalized-KL(GuacaMol)={norm_score:.3f}")
    e.log("Evaluation complete.")


@experiment.testing
def testing(e: Experiment):
    e.EPOCHS = 2
    e.MAX_TIME_HOURS = 1
    e.BATCH_SIZE = 16
    e.SAMPLE_STEPS = 5
    e.EVAL_SAMPLE_STEPS = 5
    e.GEN_SAMPLE_STEPS = 5
    e.GEN_EVERY_K = 1
    e.NUM_EVAL_SAMPLES = 40
    e.EVAL_CHUNK = 8
    e.REF_SUBSAMPLE = 150
    e.KL_BINS = 30
    e.SAMPLE_VIS_EVERY_K = 1
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2
    df = pd.read_csv(e.CSV_PATH, header=None, names=[e.SMILES_COLUMN]).head(200)
    smoke = os.path.join(folder_path(__file__), "_guacamol_smoke.smiles")
    df.to_csv(smoke, index=False, header=False)
    e.CSV_PATH = smoke


experiment.run_if_main()
