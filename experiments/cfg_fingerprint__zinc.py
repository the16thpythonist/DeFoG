"""
Pure classifier-free-guidance (CFG) model conditioned on a 128-bit Morgan
fingerprint, ZINC 250k. VERIFICATION run: does full CFG-conditional training
steer generation toward a target fingerprint (unlike the per-coordinate guidance
adapter and FK, which were ~null)?

This is NOT the frozen-base adapter (that's the next step). Here the WHOLE model
is trained conditioned on each molecule's own 128-bit fingerprint with
condition-dropout (cond_drop_prob), i.e. classic CFG. At sampling a target FP is
supplied and the rate matrices are blended with guidance_scale w (existing
denoise_step CFG path).

Eval (mirrors fingerprint_guidance / fk_fingerprint so results are comparable):
per held-out target molecule, condition on its FP, sample at w in {1,2,4}, and
measure Tanimoto(generated, target) vs an unconditional baseline. Success = a
clear lift, well beyond the ~+0.004 within-noise result of guidance/FK.
Generation is SIZE-INDEPENDENT (global size prior), matching those runs.

Usage:
    python experiments/cfg_fingerprint__zinc.py --__TESTING__ True
    python experiments/cfg_fingerprint__zinc.py --LEARNING_RATE 3e-4 --EPOCHS 20
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
import pytorch_lightning as pl
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Draw
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import (
    build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles,
    make_generation_metrics_fn,
)
from defog.core import (
    DeFoGModel, TrainingMonitorCallback, EMACallback,
)

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

FP_BITS: int = 128
FP_RADIUS: int = 2

# --- Model (settled ZINC recipe) ---
N_LAYERS: int = 9
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 20

MOLECULAR_FEATURES: bool = True
ATOM_VALENCY: dict = {"C": 4, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1, "Br": 1,
                      "P": 3, "I": 1, "Na": 1, "Si": 4, "B": 3}
ATOM_WEIGHT_TABLE: dict = {"C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
                           "S": 32.06, "Cl": 35.45, "Br": 79.904, "P": 30.974,
                           "I": 126.904, "Na": 22.99, "Si": 28.085, "B": 10.81}
MAX_ATOM_WEIGHT: float = 350.0

# --- Conditioning (CFG) ---
COND_DROP_PROB: float = 0.1
GUIDANCE_SCALE: float = 2.0          # model default; eval sweeps explicitly

# --- Training (from scratch, ~20 epochs) ---
EPOCHS: int = 20
BATCH_SIZE: int = 24
LEARNING_RATE: float = 2e-4          # swept per-arm: 1e-4/2e-4/3e-4/4e-4
LR_SCHEDULER: str = "cosine"
LR_MIN: float = 1e-6
WEIGHT_DECAY: float = 1e-5
LAMBDA_EDGE: float = 5.0
TRAIN_TIME_DISTORTION: str = "polydec"
EMA_DECAY: float = 0.9999
TRAIN_SPLIT: float = 0.9
MAX_TIME_HOURS: float = 5.0

# --- Sampling / evaluation ---
GEN_SAMPLE_STEPS: int = 500          # in-training validity probe
GEN_ETA: float = 5.0
EVAL_STEPS: int = 250
ETA: float = 5.0
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "polydec"

N_HOLDOUT: int = 2000
N_TARGETS: int = 6
N_PER_TARGET: int = 64
N_BASELINE: int = 256
EVAL_CHUNK: int = 32
GUIDANCE_SCALES: list = [1.0, 2.0, 4.0]
GRID_N: int = 24
GRID_SCALE: float = 2.0

# --- mid-training FP-steering probe ---
PROBE_EVERY_K: int = 5
PROBE_N_TARGETS: int = 2
PROBE_N: int = 24
PROBE_STEPS: int = 100
PROBE_SCALE: float = 2.0

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
        for a in mol.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def mol_morgan_bits(mol, radius, n_bits) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def tanimoto_to_target(fp_mat, target):
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


class CFGTanimotoProbe(pl.Callback):
    """Every K epochs: condition on a few held-out target FPs at a fixed scale,
    sample, and log the Tanimoto lift vs a cached unconditional baseline -- a live
    'is CFG steering yet' signal. Best-effort; never fatal. Also e.log's per-epoch
    loss (flushed / greppable)."""

    def __init__(self, e, atom_decoder, bond_decoder, radius, n_bits, cond_mean, cond_std,
                 targets_raw, targets_norm, baseline_tan, every_k, n, steps, scale, chunk):
        super().__init__()
        self.e = e
        self.ad, self.bd = atom_decoder, bond_decoder
        self.radius, self.n_bits = radius, n_bits
        self.cond_mean, self.cond_std = cond_mean, cond_std
        self.targets_raw, self.targets_norm = targets_raw, targets_norm
        self.baseline_tan = baseline_tan
        self.every_k, self.n, self.steps, self.scale, self.chunk = every_k, n, steps, scale, chunk

    def on_train_epoch_end(self, trainer, pl_module):
        ep = int(trainer.current_epoch)
        loss = trainer.callback_metrics.get("train_loss", trainer.callback_metrics.get("loss"))
        self.e.log(f"[epoch {ep}] train_loss={float(loss):.4f}" if loss is not None else f"[epoch {ep}] done")
        if not self.every_k or (ep + 1) % self.every_k != 0:
            return
        try:
            self._probe(pl_module, ep)
        except Exception as ex:
            self.e.log(f"[epoch {ep}] PROBE failed (non-fatal): {ex}")

    @torch.no_grad()
    def _probe(self, pl_module, ep):
        device = pl_module.device
        per = []
        for traw, tnorm in zip(self.targets_raw, self.targets_norm):
            samples, remaining = [], self.n
            while remaining > 0:
                cur = min(self.chunk, remaining)
                cond = torch.as_tensor(tnorm, dtype=torch.float32, device=device).unsqueeze(0).expand(cur, -1)
                samples += pl_module.sample(num_samples=cur, condition=cond, guidance_scale=self.scale,
                                            sample_steps=self.steps, device=device, show_progress=False)
                remaining -= cur
            _, _, gfp = decode_and_fp(samples, self.ad, self.bd, self.radius, self.n_bits)
            sims = tanimoto_to_target(gfp, traw)
            per.append(float(sims.mean()) if sims.size else float("nan"))
        guided = float(np.nanmean(per)) if per else float("nan")
        base = float(np.nanmean(self.baseline_tan)) if self.baseline_tan else float("nan")
        self.e.log(f"[epoch {ep}] PROBE(w={self.scale}) guided<T>={guided:.3f} baseline<T>={base:.3f} "
                   f"lift={guided - base:+.3f} per_target={[round(x, 3) for x in per]}")


# ============================================================================
# Experiment
# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log("ZINC 250k -- PURE CFG on 128-bit Morgan fingerprint (verification)")
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(e.CSV_PATH)
    e.log(f"Loaded {len(df)} molecules")
    atom_types = derive_atom_types(df[e.SMILES_COLUMN])
    e.log(f"Atom vocabulary ({len(atom_types)}): {atom_types}")
    e["config/atom_types"] = atom_types
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, e.BOND_TYPES)

    # graphs + per-molecule fingerprint (the CFG condition)
    graphs, smiles_kept = [], []
    for smi in df[e.SMILES_COLUMN]:
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder)
        if data is None:
            continue
        graphs.append(data)
        smiles_kept.append(smi)
    M = len(graphs)
    e.log(f"Converted {M} graphs; computing {e.FP_BITS}-bit Morgan FPs (r{e.FP_RADIUS}) ...")
    fp = np.stack([mol_morgan_bits(Chem.MolFromSmiles(s), e.FP_RADIUS, e.FP_BITS) for s in smiles_kept])
    cond_mean = fp.mean(0)
    cond_std = np.clip(fp.std(0), 1e-6, None)
    fp_norm = (fp - cond_mean) / cond_std
    for i, g in enumerate(graphs):
        g.y = torch.from_numpy(fp_norm[i]).float().unsqueeze(0)   # (1, 128) CFG condition

    # split: hold out targets for eval
    perm = torch.randperm(M).tolist()
    n_hold = min(e.N_HOLDOUT, M // 5)
    holdout_idx, train_idx = perm[:n_hold], perm[n_hold:]
    train_set = [graphs[i] for i in train_idx]
    train_smiles = [smiles_kept[i] for i in train_idx]
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_set, batch_size=e.BATCH_SIZE, shuffle=True)
    e.log(f"Train: {len(train_set)}  held-out target pool: {len(holdout_idx)}")

    atom_valencies = [e.ATOM_VALENCY[a] for a in atom_types]
    atom_weights_list = [e.ATOM_WEIGHT_TABLE[a] for a in atom_types]
    model = DeFoGModel.from_dataloader(
        train_loader,
        n_layers=e.N_LAYERS, hidden_dim=e.HIDDEN_DIM, hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS, dropout=e.DROPOUT, noise_type=e.NOISE_TYPE,
        extra_features_type=e.EXTRA_FEATURES_TYPE, rrwp_steps=e.RRWP_STEPS,
        molecular_features=e.MOLECULAR_FEATURES, atom_valencies=atom_valencies,
        atom_weights=atom_weights_list, max_atom_weight=e.MAX_ATOM_WEIGHT,
        lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY, lambda_edge=e.LAMBDA_EDGE,
        train_time_distortion=e.TRAIN_TIME_DISTORTION, lr_scheduler=e.LR_SCHEDULER, lr_min=e.LR_MIN,
        eta=e.ETA, omega=e.OMEGA, sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
        cond_dim=e.FP_BITS, cond_drop_prob=e.COND_DROP_PROB, guidance_scale=e.GUIDANCE_SCALE,
    )
    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"CFG model: {e['model/num_params']:,} params (cond_dim={model.cond_dim}, drop={e.COND_DROP_PROB})")

    # cached unconditional baseline for the in-training probe
    probe_idx = random.sample(holdout_idx, min(e.PROBE_N_TARGETS, len(holdout_idx)))
    probe_raw = [fp[i] for i in probe_idx]
    probe_norm = [fp_norm[i] for i in probe_idx]
    model = model.to(device)
    pb, remaining = [], max(32, e.PROBE_N)
    while remaining > 0:
        cur = min(e.EVAL_CHUNK, remaining)
        pb += model.sample(num_samples=cur, condition=None, sample_steps=e.PROBE_STEPS,
                           device=device, show_progress=False)
        remaining -= cur
    _, _, pb_fp = decode_and_fp(pb, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
    probe_baseline = [float(tanimoto_to_target(pb_fp, t).mean()) if pb_fp.shape[0] else float("nan")
                      for t in probe_raw]
    e.log(f"probe baseline <T>: {[round(x, 3) for x in probe_baseline]}")

    # -- Train --------------------------------------------------------------
    gen_metrics_fn = make_generation_metrics_fn(atom_decoder, bond_decoder, train_smiles)
    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig),
        generation_metrics_fn=gen_metrics_fn, gen_every_k=10, gen_num_samples=64,
        gen_sample_steps=e.GEN_SAMPLE_STEPS, gen_eta=e.GEN_ETA, checkpoint_dir=e.path,
    )
    probe = CFGTanimotoProbe(
        e, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS, cond_mean, cond_std,
        probe_raw, probe_norm, probe_baseline, every_k=e.PROBE_EVERY_K, n=e.PROBE_N,
        steps=e.PROBE_STEPS, scale=e.PROBE_SCALE, chunk=e.EVAL_CHUNK)
    callbacks = [monitor, probe]
    if e.EMA_DECAY and e.EMA_DECAY > 0:
        callbacks = [EMACallback(decay=e.EMA_DECAY)] + callbacks
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS}, accelerator="auto", devices=1,
        enable_progress_bar=True, enable_checkpointing=False, logger=False, callbacks=callbacks,
    )
    e.log(f"Training CFG model: epochs<={e.EPOCHS} max_time={e.MAX_TIME_HOURS}h batch={e.BATCH_SIZE} LR={e.LEARNING_RATE}")
    trainer.fit(model, train_dataloaders=train_loader)
    e.log(f"Saved model -> {model.save(os.path.join(e.path, 'model'))}")

    best_path = os.path.join(e.path, "best_model")
    if os.path.exists(best_path + ".ckpt"):
        e.log(f"Loading best-validity checkpoint (best={monitor.best_validity:.3f})")
        model = DeFoGModel.load(best_path)
    model = model.to(device).eval()

    # -- Evaluation: Tanimoto lift vs baseline, guidance-scale sweep --------
    e.log("=" * 60)
    e.log(f"EVALUATION: CFG fingerprint steering, w-sweep {e.GUIDANCE_SCALES}")
    tgt_idx = random.sample(holdout_idx, min(e.N_TARGETS, len(holdout_idx)))
    tgt_raw = [fp[i] for i in tgt_idx]
    tgt_norm = [fp_norm[i] for i in tgt_idx]
    tgt_mols = [Chem.MolFromSmiles(smiles_kept[i]) for i in tgt_idx]
    e.log(f"{len(tgt_idx)} target molecules: {[smiles_kept[i] for i in tgt_idx]}")

    # unconditional baseline pool (shared)
    base_samples, remaining = [], e.N_BASELINE
    while remaining > 0:
        cur = min(e.EVAL_CHUNK, remaining)
        base_samples += model.sample(num_samples=cur, condition=None, sample_steps=e.EVAL_STEPS,
                                     eta=e.ETA, omega=e.OMEGA, time_distortion=e.SAMPLE_TIME_DISTORTION,
                                     device=device, show_progress=False)
        remaining -= cur
    _, base_smis, base_fp = decode_and_fp(base_samples, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
    e.log(f"baseline valid: {base_fp.shape[0]}/{e.N_BASELINE}")

    methods = ["baseline"] + [f"w={w}" for w in e.GUIDANCE_SCALES]
    agg = {m: [] for m in methods}
    per_target = []
    for ti, (traw, tnorm, tmol) in enumerate(zip(tgt_raw, tgt_norm, tgt_mols)):
        base_sims = tanimoto_to_target(base_fp, traw)
        agg["baseline"].extend(base_sims.tolist())
        rec = {"index": int(tgt_idx[ti]), "smiles": smiles_kept[tgt_idx[ti]],
               "baseline_mean_tanimoto": float(base_sims.mean()) if base_sims.size else None, "per_w": {}}
        e.log(f"[target {ti+1}/{len(tgt_idx)}] baseline<T>={rec['baseline_mean_tanimoto']}")
        best_grid = None
        for w in e.GUIDANCE_SCALES:
            samples, remaining = [], e.N_PER_TARGET
            while remaining > 0:
                cur = min(e.EVAL_CHUNK, remaining)
                cond = torch.as_tensor(tnorm, dtype=torch.float32, device=device).unsqueeze(0).expand(cur, -1)
                samples += model.sample(num_samples=cur, condition=cond, guidance_scale=w,
                                        sample_steps=e.EVAL_STEPS, eta=e.ETA, omega=e.OMEGA,
                                        time_distortion=e.SAMPLE_TIME_DISTORTION, device=device, show_progress=False)
                remaining -= cur
            mols, smis, gfp = decode_and_fp(samples, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
            sims = tanimoto_to_target(gfp, traw)
            agg[f"w={w}"].extend(sims.tolist())
            rec["per_w"][str(w)] = {
                "n_valid": len(mols), "n_unique": len(set(smis)),
                "validity": len(mols) / len(samples) if samples else 0.0,
                "mean_tanimoto": float(sims.mean()) if sims.size else None,
                "median_tanimoto": float(np.median(sims)) if sims.size else None,
                "max_tanimoto": float(sims.max()) if sims.size else None,
            }
            e.log(f"    w={w}: valid={len(mols)}/{len(samples)} uniq={len(set(smis))} "
                  f"<T>={rec['per_w'][str(w)]['mean_tanimoto']} maxT={rec['per_w'][str(w)]['max_tanimoto']}")
            if abs(w - e.GRID_SCALE) < 1e-9 and mols:
                order = np.argsort(-sims)[:e.GRID_N]
                best_grid = ([tmol] + [mols[j] for j in order], ["TARGET"] + [f"T={sims[j]:.2f}" for j in order])
        if best_grid:
            Draw.MolsToGridImage(best_grid[0], molsPerRow=5, subImgSize=(220, 220),
                                 legends=best_grid[1]).save(os.path.join(e.path, f"grid_target{ti}.png"))
        per_target.append(rec)

    base_mean = float(np.mean(agg["baseline"])) if agg["baseline"] else float("nan")
    summary = {"methods": methods, "n_targets": len(tgt_idx), "learning_rate": e.LEARNING_RATE,
               "baseline_valid": int(base_fp.shape[0]), "per_target": per_target, "aggregate": {}}
    for m in methods:
        mean = float(np.mean(agg[m])) if agg[m] else float("nan")
        summary["aggregate"][m] = {"mean_tanimoto": mean, "lift_over_baseline": mean - base_mean}
    e.commit_json("cfg_fingerprint_metrics.json", summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [summary["aggregate"][m]["mean_tanimoto"] for m in methods]
    bars = ax.bar(methods, means, color=["0.6"] + ["#55a868"] * len(e.GUIDANCE_SCALES))
    ax.axhline(base_mean, ls="--", color="0.5", lw=1)
    for b, mn in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, mn, f"{mn:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean Tanimoto to target")
    ax.set_title(f"CFG fingerprint steering (LR={e.LEARNING_RATE}) | baseline<T>={base_mean:.3f}")
    fig.tight_layout()
    e.commit_fig("method_comparison.png", fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5.2))
    bins = np.linspace(0, 1, 41)
    for m in methods:
        if agg[m]:
            ax2.hist(agg[m], bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                     label=f"{m} (<T>={np.mean(agg[m]):.3f})")
    ax2.set_xlabel("Tanimoto to target"); ax2.set_ylabel("density")
    ax2.set_title("CFG fingerprint steering: Tanimoto-to-target by guidance scale")
    ax2.legend(fontsize=9); fig2.tight_layout()
    e.commit_fig("tanimoto_distributions.png", fig2)

    e.log("=" * 60)
    for m in methods:
        a = summary["aggregate"][m]
        e.log(f"{m:10s} <T>={a['mean_tanimoto']:.3f}  lift={a['lift_over_baseline']:+.3f}")
    e.log("Done.")


@experiment.testing
def testing(e: Experiment):
    e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.MAX_TIME_HOURS = 0.2
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2
    e.GEN_SAMPLE_STEPS = 5
    e.EVAL_STEPS = 5
    e.PROBE_STEPS = 5
    e.PROBE_EVERY_K = 1
    e.PROBE_N = 6
    e.PROBE_N_TARGETS = 2
    e.N_HOLDOUT = 40
    e.N_TARGETS = 2
    e.N_PER_TARGET = 8
    e.N_BASELINE = 16
    e.EVAL_CHUNK = 8
    e.GUIDANCE_SCALES = [1.0, 2.0]
    e.GRID_N = 4
    df = pd.read_csv(e.CSV_PATH).head(300)
    smoke = os.path.join(folder_path(__file__), "_cfg_fp_smoke.csv")
    df.to_csv(smoke, index=False)
    e.CSV_PATH = smoke


experiment.run_if_main()
