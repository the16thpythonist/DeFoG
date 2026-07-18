"""
Train ONE frozen-base AdaLN/FiLM CFG-adapter conditioned on a 128-bit Morgan
FINGERPRINT, ZINC 250k. The base (connectivity-improved unconditional ZINC model)
is FROZEN; only the adapter trains, with the base's own denoising CE loss (a
conditional denoiser p(x1|x_t, fp)).

This is the high-dimensional / holistic test of the frozen-base adapter: the
per-coordinate GUIDANCE adapter and FK were ~null on fingerprints, while the full
CFG-conditional model steered (+0.12 Tanimoto lift). This asks whether the FROZEN
base + adapter matches that (it injects the same conditional-denoiser signal via
FiLM, so it should).

Eval (mirrors cfg_fingerprint / fingerprint_guidance): per held-out target
molecule, condition on its FP, sample via AdaptedSampler at w in {1,2,4}, and
measure Tanimoto(generated, target) vs an unconditional baseline. Size-independent
generation (global size prior), matching those runs.

Usage:
    python experiments/adapter_fingerprint__zinc.py --__TESTING__ True
    python experiments/adapter_fingerprint__zinc.py --LEARNING_RATE 3e-4
"""
import os
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

from experiments.utils import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, AdaLNAdapter, AdapterModule, AdapterComposition, ConditionBranch,
    AdaptedSampler, Sampler,
)

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
ATOM_TYPES: list = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]   # match frozen base
BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")

FP_BITS: int = 512      # 512-bit Morgan/ECFP4: far fewer bit collisions than 128 -> more
FP_RADIUS: int = 2      # discriminative fingerprint + a cleaner Tanimoto signal

# --- Adapter architecture ---
H_HIDDEN: int = 256
TIME_CONDITIONED: bool = True
STREAMS: list = ["X", "E", "y"]

# --- Training (8h wall on JUPITER; base frozen, only the adapter trains) ---
EPOCHS: int = 50
BATCH_SIZE: int = 24
LEARNING_RATE: float = 2e-4      # swept per-arm (4 LRs)
COND_DROP_PROB: float = 0.0      # uncond branch IS the frozen base -> dropout not needed
MAX_TIME_HOURS: float = 8.0
N_HOLDOUT: int = 2000            # held out of training; eval targets drawn from here (unseen FPs)

# --- Sampling / evaluation ---
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
N_TARGETS: int = 6
N_PER_TARGET: int = 64
N_BASELINE: int = 256
EVAL_CHUNK: int = 32
GUIDANCE_WEIGHTS: list = [1.0, 2.0, 4.0]
GRID_N: int = 24
GRID_SCALE: float = 2.0

# --- mid-training probe ---
PROBE_EVERY_K: int = 5
PROBE_N_TARGETS: int = 2
PROBE_N: int = 24
PROBE_STEPS: int = 100
PROBE_WEIGHT: float = 2.0
PROBE_BASELINE_N: int = 48

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ============================================================================
# Helpers
# ============================================================================
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


def guided_sample(base, adapter, target_fp, weight, n, steps, eta, omega, td, chunk, device):
    comp = AdapterComposition([ConditionBranch(adapter, torch.as_tensor(target_fp, dtype=torch.float32), weight)],
                              base=base, mode="product")
    samp = AdaptedSampler(base, comp, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
    out, rem = [], n
    while rem > 0:
        cur = min(chunk, rem)
        out += samp.sample(cur, device=device, show_progress=False)
        rem -= cur
    return out


class FPAdapterProbe(pl.Callback):
    """Per-epoch loss log + every-K-epoch Tanimoto-steering probe vs a cached
    unconditional baseline, so training is visible."""

    def __init__(self, e, atom_decoder, bond_decoder, radius, n_bits, targets_raw,
                 baseline_tan, every_k, n, steps, weight, eta, omega, td, chunk):
        super().__init__()
        self.e = e
        self.ad, self.bd, self.radius, self.n_bits = atom_decoder, bond_decoder, radius, n_bits
        self.targets_raw, self.baseline_tan = targets_raw, baseline_tan
        self.every_k, self.n, self.steps, self.weight = every_k, n, steps, weight
        self.eta, self.omega, self.td, self.chunk = eta, omega, td, chunk

    def on_train_epoch_end(self, trainer, pl_module):
        ep = int(trainer.current_epoch)
        loss = trainer.callback_metrics.get("adapter/loss_epoch", trainer.callback_metrics.get("adapter/loss"))
        self.e.log(f"[epoch {ep}] adapter/loss={float(loss):.4f}" if loss is not None else f"[epoch {ep}] done")
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
        for traw in self.targets_raw:
            samples = guided_sample(pl_module.base, pl_module.adapter, traw, self.weight, self.n,
                                    self.steps, self.eta, self.omega, self.td, self.chunk, device)
            _, _, gfp = decode_and_fp(samples, self.ad, self.bd, self.radius, self.n_bits)
            sims = tanimoto_to_target(gfp, traw)
            per.append(float(sims.mean()) if sims.size else float("nan"))
        guided = float(np.nanmean(per)) if per else float("nan")
        base = float(np.nanmean(self.baseline_tan)) if self.baseline_tan else float("nan")
        self.e.log(f"[epoch {ep}] PROBE(w={self.weight}) guided<T>={guided:.3f} baseline<T>={base:.3f} "
                   f"lift={guided - base:+.3f} per_target={[round(x, 3) for x in per]}")


# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log("ZINC frozen-base AdaLN CFG-ADAPTER on 128-bit Morgan fingerprint")
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(e.CSV_PATH)
    atom_types = e.ATOM_TYPES
    e.log(f"Atom vocabulary ({len(atom_types)}): {atom_types}")
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, e.BOND_TYPES)

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
    fp_t = torch.from_numpy(fp)
    for i, g in enumerate(graphs):
        g.cond = fp_t[i].unsqueeze(0)   # (1, 128) RAW fingerprint (adapter normalizes internally)

    perm = torch.randperm(M).tolist()
    n_hold = min(e.N_HOLDOUT, M // 5)
    holdout_idx, train_idx = perm[:n_hold], perm[n_hold:]
    train_graphs = [graphs[i] for i in train_idx]
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=e.BATCH_SIZE, shuffle=True)
    e.log(f"Adapter train: {len(train_graphs)}   held-out target pool: {len(holdout_idx)}")

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0, f"expected unconditional base, cond_dim={base.cond_dim}"
    adapter = AdaLNAdapter.for_base(
        base, cond_dim=e.FP_BITS, hidden=e.H_HIDDEN, time_conditioned=e.TIME_CONDITIONED,
        streams=tuple(e.STREAMS), cond_mean=cond_mean, cond_std=cond_std,
        name="fp_adapter", cond_type="morgan128")
    e["adapter/num_params"] = sum(p.numel() for p in adapter.parameters())
    e.log(f"adapter: {e['adapter/num_params']:,} params (base {sum(p.numel() for p in base.parameters()):,} frozen)")
    module = AdapterModule(base, adapter, cond_attr="cond", cond_drop_prob=e.COND_DROP_PROB, lr=e.LEARNING_RATE)

    # probe targets (held out) + cached unconditional baseline
    probe_idx = random.sample(holdout_idx, min(e.PROBE_N_TARGETS, len(holdout_idx)))
    probe_raw = [fp[i] for i in probe_idx]
    pb = []
    pbs = Sampler(base, eta=e.ETA, omega=e.OMEGA, sample_steps=e.PROBE_STEPS, time_distortion=e.TIME_DISTORTION)
    rem = max(32, e.PROBE_N)
    while rem > 0:
        cur = min(e.EVAL_CHUNK, rem)
        pb += pbs.sample(cur, device=device, show_progress=False)
        rem -= cur
    _, _, pb_fp = decode_and_fp(pb, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
    probe_baseline = [float(tanimoto_to_target(pb_fp, t).mean()) if pb_fp.shape[0] else float("nan")
                      for t in probe_raw]
    e.log(f"probe baseline <T>: {[round(x, 3) for x in probe_baseline]}")
    probe = FPAdapterProbe(e, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS, probe_raw, probe_baseline,
                           e.PROBE_EVERY_K, e.PROBE_N, e.PROBE_STEPS, e.PROBE_WEIGHT,
                           e.ETA, e.OMEGA, e.TIME_DISTORTION, e.EVAL_CHUNK)

    trainer = pl.Trainer(max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS}, accelerator="auto",
                         devices=1, enable_progress_bar=True, enable_checkpointing=False, logger=False,
                         gradient_clip_val=1.0, callbacks=[probe])
    e.log(f"Training adapter: epochs<={e.EPOCHS} max_time={e.MAX_TIME_HOURS}h batch={e.BATCH_SIZE} LR={e.LEARNING_RATE}")
    trainer.fit(module, train_dataloaders=train_loader)

    ckpt = adapter.save(os.path.join(e.path, "fp_adapter"))
    with open(os.path.join(e.path, "fp_adapter_stats.json"), "w") as f:
        json.dump({"fp_bits": e.FP_BITS, "fp_radius": e.FP_RADIUS, "atom_types": atom_types,
                   "cond_mean": cond_mean.tolist(), "cond_std": cond_std.tolist(),
                   "learning_rate": e.LEARNING_RATE}, f)
    e.log(f"Saved adapter -> {ckpt}")

    # -- Evaluation: Tanimoto lift vs baseline, guidance-weight sweep ----------
    e.log("=" * 60)
    base = base.to(device).eval()
    adapter = adapter.to(device).eval()
    tgt_idx = random.sample(holdout_idx, min(e.N_TARGETS, len(holdout_idx)))
    tgt_raw = [fp[i] for i in tgt_idx]
    tgt_mols = [Chem.MolFromSmiles(smiles_kept[i]) for i in tgt_idx]
    e.log(f"{len(tgt_idx)} target molecules")

    base_sampler = Sampler(base, eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS, time_distortion=e.TIME_DISTORTION)
    bsamp, rem = [], e.N_BASELINE
    while rem > 0:
        cur = min(e.EVAL_CHUNK, rem)
        bsamp += base_sampler.sample(cur, device=device, show_progress=False)
        rem -= cur
    _, _, base_fp = decode_and_fp(bsamp, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
    e.log(f"baseline valid: {base_fp.shape[0]}/{e.N_BASELINE}")

    methods = ["baseline"] + [f"w={w}" for w in e.GUIDANCE_WEIGHTS]
    agg = {m: [] for m in methods}
    per_target = []
    for ti, (traw, tmol) in enumerate(zip(tgt_raw, tgt_mols)):
        base_sims = tanimoto_to_target(base_fp, traw)
        agg["baseline"].extend(base_sims.tolist())
        rec = {"index": int(tgt_idx[ti]), "smiles": smiles_kept[tgt_idx[ti]],
               "baseline_mean_tanimoto": float(base_sims.mean()) if base_sims.size else None, "per_w": {}}
        best_grid = None
        for w in e.GUIDANCE_WEIGHTS:
            samples = guided_sample(base, adapter, traw, w, e.N_PER_TARGET, e.EVAL_STEPS,
                                    e.ETA, e.OMEGA, e.TIME_DISTORTION, e.EVAL_CHUNK, device)
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
            e.log(f"  [t{ti}] w={w}: valid={len(mols)}/{len(samples)} uniq={len(set(smis))} "
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
               "eval_steps": e.EVAL_STEPS, "baseline_valid": int(base_fp.shape[0]),
               "per_target": per_target, "aggregate": {}}
    for m in methods:
        mean = float(np.mean(agg[m])) if agg[m] else float("nan")
        summary["aggregate"][m] = {"mean_tanimoto": mean, "lift_over_baseline": mean - base_mean}
    e.commit_json("adapter_fingerprint_metrics.json", summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [summary["aggregate"][m]["mean_tanimoto"] for m in methods]
    bars = ax.bar(methods, means, color=["0.6"] + ["#55a868"] * len(e.GUIDANCE_WEIGHTS))
    ax.axhline(base_mean, ls="--", color="0.5", lw=1)
    for b, mn in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, mn, f"{mn:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean Tanimoto to target")
    ax.set_title(f"FP adapter steering (LR={e.LEARNING_RATE}, {e.EVAL_STEPS} steps) | baseline<T>={base_mean:.3f}")
    fig.tight_layout()
    e.commit_fig("method_comparison.png", fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5.2))
    bins = np.linspace(0, 1, 41)
    for m in methods:
        if agg[m]:
            ax2.hist(agg[m], bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                     label=f"{m} (<T>={np.mean(agg[m]):.3f})")
    ax2.set_xlabel("Tanimoto to target"); ax2.set_ylabel("density")
    ax2.set_title("FP adapter steering: Tanimoto-to-target by guidance weight")
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
    e.H_HIDDEN = 32
    e.EVAL_STEPS = 5
    e.PROBE_STEPS = 5
    e.PROBE_EVERY_K = 1
    e.PROBE_N = 6
    e.PROBE_N_TARGETS = 2
    e.N_HOLDOUT = 40
    e.N_TARGETS = 2
    e.N_PER_TARGET = 8
    e.N_BASELINE = 8
    e.EVAL_CHUNK = 8
    e.GUIDANCE_WEIGHTS = [2.0]
    e.GRID_N = 4
    df = pd.read_csv(e.CSV_PATH).head(300)
    smoke = os.path.join(folder_path(__file__), "_adapter_fp_smoke.csv")
    df.to_csv(smoke, index=False)
    e.CSV_PATH = smoke
    e.BASE_CKPT = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")


experiment.run_if_main()
