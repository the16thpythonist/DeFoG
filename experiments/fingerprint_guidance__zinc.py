"""
Latent / high-dimensional conditioning for DeFoG via an amortized guidance
adapter -- prototype on ZINC 250k with 128-bit Morgan fingerprints as the
condition.

Idea
----
Generalize the amortized (scalar-property) guidance to an ARBITRARY d-dim
precomputed condition ``phi(x1)``. Here ``phi`` = a 128-bit Morgan/ECFP
fingerprint. A single guidance network ``h`` (cond_dim=128) is trained, against
the FROZEN improved ZINC base, so that supplying a target fingerprint ``c`` at
sample time steers generation toward molecules whose fingerprint is close to
``c`` (i.e. "give me molecules similar to this one").

The high-dimensional condition breaks the scalar recipe's independent target
sampling (a random (x1, c) pair almost never matches -> reward ~0 -> dead
gradients). So targets are drawn with POSITIVE-BIASED PAIRING: with prob
``POS_FRAC`` the target is the fingerprint of a precomputed high-Tanimoto
neighbor of ``x1`` (a genuine match), otherwise a random in-batch fingerprint
(a negative). See :class:`defog.core.LatentGuidanceModule`.

Pipeline
--------
1. Load frozen base (connectivity-improved ZINC uncond model).
2. Build graphs + a 128-bit Morgan FP per molecule; precompute top-K Tanimoto
   neighbors (GPU, cached).
3. Train the latent adapter (Tanimoto reward, positive-biased pairing).
4. Eval: for held-out target molecules, guided-sample (weight sweep) and measure
   Tanimoto(generated FP, target FP) vs an UNCONDITIONAL baseline. Artifacts:
   per-target grids (target + most-similar generations), Tanimoto distribution
   plots, and a metrics JSON.

Generation is SIZE-INDEPENDENT (graph size drawn from the base's global size
prior) -- a conservative test of pure fingerprint-type steering.

Usage:
    python experiments/fingerprint_guidance__zinc.py
    python experiments/fingerprint_guidance__zinc.py --__TESTING__ True
    python experiments/fingerprint_guidance__zinc.py --EPOCHS 20 --REWARD_SHARPEN 2.0
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

from experiments.utils import (  # noqa: E402
    build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles,
    make_generation_metrics_fn,
)
from defog.core import (  # noqa: E402
    DeFoGModel, LatentGuidanceModule, ExactGuidance, GuidedSampler, Sampler,
    tanimoto_similarity,
)

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# Frozen base = the connectivity-improved ZINC 250k unconditional model.
BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")

# --- Condition: Morgan fingerprint ---
# :param FP_BITS / FP_RADIUS: 128-bit ECFP4 (radius 2). 128 bits collide a lot
#     (folded), so baseline Tanimoto is elevated -- REWARD_SHARPEN compensates.
FP_BITS: int = 128
FP_RADIUS: int = 2

# --- Positive-biased pairing ---
# :param N_NEIGHBORS: top-K Tanimoto neighbors precomputed per molecule; a positive
#     target is a random one of these. :param POS_FRAC: fraction of each batch drawn
#     as positives (neighbor targets); the rest are in-batch negatives.
N_NEIGHBORS: int = 64
POS_FRAC: float = 0.5
# :param REWARD_SHARPEN: r <- Tanimoto**beta. >1 sharpens the (otherwise diffuse
#     for 128-bit FPs) reward so positives/negatives separate more.
REWARD_SHARPEN: float = 2.0

# --- Adapter (guidance net h) architecture: smaller than the 9L/256 base ---
H_LAYERS: int = 6
H_HIDDEN: int = 256
H_MLP_DIM: int = 512
H_HEADS: int = 8

# --- Adapter training ---
EPOCHS: int = 15
BATCH_SIZE: int = 24
LEARNING_RATE: float = 2e-4
LAMBDA_EDGE: float = 5.0
G_CLAMP: float = 20.0
MAX_TIME_HOURS: float = 9.0
# :param SUBSET: cap #molecules used to train the adapter (0/None = full 250k).
SUBSET: int = 0
# :param N_HOLDOUT: molecules held out of adapter training; eval targets are drawn
#     from here so we steer toward UNSEEN fingerprints (tests generalization).
N_HOLDOUT: int = 2000

# --- Guided-sampling evaluation ---
# :param GUIDED_WEIGHTS: guidance-strength sweep (w=1 exact; >1 sharpens steering).
GUIDED_WEIGHTS: list = [1.0, 3.0]
N_EVAL_TARGETS: int = 6
N_PER_TARGET: int = 64
BASELINE_POOL: int = 256
EVAL_CHUNK: int = 32
GUIDED_STEPS: int = 250
GUIDED_ETA: float = 5.0
GUIDED_OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "polydec"
GRID_N: int = 24   # most-similar generations shown per target grid

# --- Intermediate reporting (visible DURING training) ---
# :param PROBE_EVERY_K: every K epochs, guided-sample toward a few fixed held-out
#     targets and log the achieved Tanimoto vs a cached unconditional baseline, so
#     the steering curve is visible mid-run (not just at the final eval). Also
#     e.log's per-epoch loss/r_mean (flushed -> greppable) and saves adapter_latest.
PROBE_EVERY_K: int = 3
PROBE_N_TARGETS: int = 2
PROBE_N: int = 24
PROBE_STEPS: int = 100
PROBE_WEIGHT: float = 3.0
PROBE_BASELINE_N: int = 48

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ============================================================================
# Helpers
# ============================================================================
def derive_atom_types(smiles_list) -> list:
    """Frequency-descending atom vocabulary -- MUST match the base checkpoint's
    node-class order (the base was trained with this same function on this CSV)."""
    counts = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for a in mol.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def mol_morgan_bits(mol, radius, n_bits) -> np.ndarray:
    """(n_bits,) float32 {0,1} Morgan fingerprint, or zeros if mol is None."""
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def morgan_matrix(smiles_list, radius, n_bits) -> np.ndarray:
    """(M, n_bits) fingerprint matrix aligned to ``smiles_list``."""
    fp = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        fp[i] = mol_morgan_bits(Chem.MolFromSmiles(smi), radius, n_bits)
    return fp


def topk_tanimoto_neighbors(fp: np.ndarray, k: int, device, block: int = 1024) -> np.ndarray:
    """(M, k) int64 indices of each row's top-k Tanimoto neighbors (self excluded).

    Blocked GPU matmul: intersection = Q @ F^T, union = |Q| + |F| - inter. O(M^2 d)
    but bandwidth-bound and fully vectorized -- minutes for 250k x 128 on a 24GB GPU.
    """
    F = torch.as_tensor(fp, dtype=torch.float32, device=device)
    s = F.sum(1)                                  # popcount per row
    M = F.size(0)
    k = min(k, M - 1)
    out = np.empty((M, k), dtype=np.int64)
    for i in range(0, M, block):
        Q = F[i:i + block]                        # (b, d)
        inter = Q @ F.t()                         # (b, M)
        union = s[i:i + block, None] + s[None, :] - inter
        tan = inter / union.clamp_min(1e-8)
        rows = torch.arange(Q.size(0), device=device)
        tan[rows, torch.arange(i, i + Q.size(0), device=device)] = -1.0   # exclude self
        out[i:i + Q.size(0)] = tan.topk(k, dim=1).indices.cpu().numpy()
    return out


def tanimoto_to_target(fp_mat: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(G,) Tanimoto of each row of ``fp_mat`` (G, d) to a single ``target`` (d,)."""
    if fp_mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    inter = fp_mat @ target
    union = fp_mat.sum(1) + target.sum() - inter
    return inter / np.clip(union, 1e-8, None)


def decode_and_fp(samples, atom_decoder, bond_decoder, radius, n_bits):
    """Decode graph samples -> (list[mol], list[smiles], fp_matrix) for valid mols."""
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


class IntermediateReporter(pl.Callback):
    """Makes adapter training visible mid-run.

    On every train-epoch end it ``e.log``s the epoch loss / reward (flushed, so it
    shows up in the batch log unlike the suppressed progress bar) and tracks a
    training curve. Every ``every_k`` epochs it runs a small guided-sampling PROBE
    toward a few fixed held-out target fingerprints and logs the achieved Tanimoto
    against a cached unconditional baseline -- i.e. a live steering curve -- and
    overwrites ``adapter_latest.ckpt`` so a mid-run kill still leaves a usable
    adapter. Probes are best-effort: any failure is logged, never fatal.
    """

    def __init__(self, e, atom_decoder, bond_decoder, fp_radius, fp_bits,
                 probe_targets, baseline_tan, cond_mean, cond_std, every_k, probe_n,
                 probe_steps, probe_weight, eta, omega, time_distortion, chunk, ckpt_dir):
        super().__init__()
        self.e = e
        self.atom_decoder, self.bond_decoder = atom_decoder, bond_decoder
        self.fp_radius, self.fp_bits = fp_radius, fp_bits
        self.probe_targets = probe_targets          # list of (d,) np arrays
        self.baseline_tan = baseline_tan            # list of per-target baseline mean Tanimoto
        self.cond_mean, self.cond_std = cond_mean, cond_std
        self.every_k, self.probe_n, self.probe_steps = every_k, probe_n, probe_steps
        self.probe_weight, self.eta, self.omega = probe_weight, eta, omega
        self.time_distortion, self.chunk, self.ckpt_dir = time_distortion, chunk, ckpt_dir
        self.loss_hist, self.probe_hist = [], []

    def on_train_epoch_end(self, trainer, pl_module):
        ep = int(trainer.current_epoch)
        cm = trainer.callback_metrics
        loss = float(cm.get("guid/loss_epoch", cm.get("guid/loss", float("nan"))))
        rmean = float(cm.get("guid/r_mean_epoch", cm.get("guid/r_mean", float("nan"))))
        self.loss_hist.append((ep, loss, rmean))
        self.e.log(f"[epoch {ep}] guid/loss={loss:.4f}  r_mean={rmean:.4f}")
        self._track("training_curve", self.loss_hist, ("guid/loss", "r_mean"),
                    "Adapter training loss / reward")
        if self.every_k and (ep + 1) % self.every_k == 0:
            try:
                self._probe(pl_module, ep)
            except Exception as ex:  # never let a probe kill training
                self.e.log(f"[epoch {ep}] PROBE failed (non-fatal): {ex}")

    @torch.no_grad()
    def _probe(self, pl_module, ep):
        device = pl_module.device
        was_training = pl_module.h.training
        pl_module.h.eval()
        guidance = ExactGuidance(pl_module.h, prop_mean=self.cond_mean,
                                 prop_std=self.cond_std, weight=self.probe_weight)
        per_target = []
        for tvec in self.probe_targets:
            guidance.set_target(tvec)
            sampler = GuidedSampler(pl_module.base, guidance, eta=self.eta, omega=self.omega,
                                    sample_steps=self.probe_steps, time_distortion=self.time_distortion)
            samples, remaining = [], self.probe_n
            while remaining > 0:
                cur = min(self.chunk, remaining)
                samples += sampler.sample(cur, device=device, show_progress=False)
                remaining -= cur
            _, _, gfp = decode_and_fp(samples, self.atom_decoder, self.bond_decoder,
                                      self.fp_radius, self.fp_bits)
            sims = tanimoto_to_target(gfp, tvec)
            per_target.append(float(sims.mean()) if sims.size else float("nan"))
        if was_training:
            pl_module.h.train()
        guided = float(np.nanmean(per_target)) if per_target else float("nan")
        base_mean = float(np.nanmean(self.baseline_tan)) if self.baseline_tan else float("nan")
        self.probe_hist.append((ep, guided, base_mean))
        self.e.log(f"[epoch {ep}] PROBE guided<T>={guided:.3f}  baseline<T>={base_mean:.3f}  "
                   f"lift={guided - base_mean:+.3f}  per_target={[round(x, 3) for x in per_target]}")
        self._track("probe_tanimoto", self.probe_hist, ("guided <T>", "baseline <T>"),
                    f"Guided steering probe (w={self.probe_weight})", ylabel="Tanimoto to target")
        try:
            guidance.save(os.path.join(self.ckpt_dir, "adapter_latest"))
        except Exception:
            pass

    def _track(self, name, hist, labels, title, ylabel=None):
        try:
            eps = [h[0] for h in hist]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(eps, [h[1] for h in hist], "-o", color="steelblue", label=labels[0])
            if name == "probe_tanimoto":
                ax.plot(eps, [h[2] for h in hist], "--", color="0.5", label=labels[1])
                ax.set_ylabel(ylabel or labels[0])
            else:
                ax2 = ax.twinx()
                ax2.plot(eps, [h[2] for h in hist], "-s", color="crimson", label=labels[1])
                ax.set_ylabel(labels[0]); ax2.set_ylabel(labels[1])
            ax.set_xlabel("epoch"); ax.set_title(title); ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            self.e.track(name, fig)
            plt.close(fig)
        except Exception:
            pass


# ============================================================================
# Experiment
# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log("ZINC 250k -- LATENT (Morgan-fingerprint) guidance adapter")
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e.log(f"device={device}  base_ckpt={e.BASE_CKPT}")

    # -- Data ---------------------------------------------------------------
    df = pd.read_csv(e.CSV_PATH)
    if e.SUBSET and e.SUBSET > 0:
        df = df.head(e.SUBSET)
    e.log(f"Loaded {len(df)} molecules from {e.CSV_PATH}")
    atom_types = derive_atom_types(df[e.SMILES_COLUMN])
    e.log(f"Atom vocabulary ({len(atom_types)}): {atom_types}")
    e["config/atom_types"] = atom_types
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, e.BOND_TYPES)

    graphs, smiles_kept, skipped = [], [], 0
    for smi in df[e.SMILES_COLUMN]:
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder)
        if data is None:
            skipped += 1
            continue
        graphs.append(data)
        smiles_kept.append(smi)
    M = len(graphs)
    e.log(f"Converted {M} graphs ({skipped} skipped)")

    # -- Fingerprints + neighbors (cached for the full set) -----------------
    full = not (e.SUBSET and e.SUBSET > 0)
    cache_dir = os.path.join(_PROJECT_DIR, "data", "_fpcache")
    os.makedirs(cache_dir, exist_ok=True)
    fp_cache = os.path.join(cache_dir, f"zinc_fp{e.FP_BITS}_r{e.FP_RADIUS}.npy")
    nbr_cache = os.path.join(cache_dir, f"zinc_fp{e.FP_BITS}_r{e.FP_RADIUS}_nbr{e.N_NEIGHBORS}.npy")

    if full and os.path.exists(fp_cache) and os.path.exists(nbr_cache):
        fp = np.load(fp_cache)
        nbr = np.load(nbr_cache)
        e.log(f"Loaded cached FP {fp.shape} + neighbors {nbr.shape}")
        assert fp.shape[0] == M, f"cache/dataset mismatch ({fp.shape[0]} vs {M}) -- delete {fp_cache}"
    else:
        e.log(f"Computing {e.FP_BITS}-bit Morgan FPs (radius {e.FP_RADIUS}) for {M} molecules ...")
        fp = morgan_matrix(smiles_kept, e.FP_RADIUS, e.FP_BITS)
        e.log(f"Computing top-{e.N_NEIGHBORS} Tanimoto neighbors on {device} ...")
        nbr = topk_tanimoto_neighbors(fp, e.N_NEIGHBORS, device)
        if full:
            np.save(fp_cache, fp)
            np.save(nbr_cache, nbr)
            e.log(f"Cached FP + neighbors to {cache_dir}")
    e["config/fp_density"] = float(fp.mean())
    # sanity: mean top-1 neighbor Tanimoto (how "positive" positives are)
    top1 = float(np.mean([tanimoto_to_target(fp[nbr[i, :1]], fp[i])[0] for i in range(min(M, 500))]))
    e.log(f"FP density (frac bits on)={fp.mean():.3f}   mean top-1 neighbor Tanimoto~{top1:.3f}")

    # attach condition + neighbor indices to each graph
    fp_t = torch.from_numpy(fp)
    nbr_t = torch.from_numpy(nbr)
    for i, g in enumerate(graphs):
        g.cond = fp_t[i].unsqueeze(0)      # (1, d) -> collates to (bs, d)
        g.nbr = nbr_t[i].unsqueeze(0)      # (1, K)

    # -- Split: hold out targets for eval, train adapter on the rest --------
    perm = torch.randperm(M).tolist()
    n_hold = min(e.N_HOLDOUT, M // 5)
    holdout_idx = perm[:n_hold]
    train_idx = perm[n_hold:]
    train_graphs = [graphs[i] for i in train_idx]
    e.log(f"Adapter train: {len(train_graphs)}   held-out target pool: {len(holdout_idx)}")

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=e.BATCH_SIZE, shuffle=True)

    # -- Frozen base + latent adapter ---------------------------------------
    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device)
    base.eval()
    assert base.cond_dim == 0, f"expected an unconditional base, got cond_dim={base.cond_dim}"
    e.log(f"Loaded base ({sum(p.numel() for p in base.parameters()):,} params, cond_dim={base.cond_dim})")

    cond_mean = fp.mean(0)
    cond_std = fp.std(0)
    module = LatentGuidanceModule(
        base, cond_dim=e.FP_BITS, cond_bank=fp, cond_mean=cond_mean, cond_std=cond_std,
        reward_fn=tanimoto_similarity, reward_sharpen=e.REWARD_SHARPEN, pos_frac=e.POS_FRAC,
        lr=e.LEARNING_RATE, lambda_edge=e.LAMBDA_EDGE, g_clamp=e.G_CLAMP,
        n_layers=e.H_LAYERS, hidden_dim=e.H_HIDDEN, hidden_mlp_dim=e.H_MLP_DIM, n_heads=e.H_HEADS,
    )
    e["adapter/num_params"] = sum(p.numel() for p in module.h.parameters())
    e.log(f"Adapter h: {e['adapter/num_params']:,} params (cond_dim={e.FP_BITS})")

    # -- Intermediate-reporting probe: fixed held-out targets + cached baseline ---
    module = module.to(device)
    probe_idx = random.sample(holdout_idx, min(e.PROBE_N_TARGETS, len(holdout_idx)))
    probe_targets = [fp[i] for i in probe_idx]
    e.log(f"probe targets ({len(probe_idx)}): {[smiles_kept[i] for i in probe_idx]}")
    pb_sampler = Sampler(base, eta=e.GUIDED_ETA, omega=e.GUIDED_OMEGA,
                         sample_steps=e.PROBE_STEPS, time_distortion=e.SAMPLE_TIME_DISTORTION)
    pb_samples, rem = [], e.PROBE_BASELINE_N
    while rem > 0:
        cur = min(e.EVAL_CHUNK, rem)
        pb_samples += pb_sampler.sample(cur, device=device, show_progress=False)
        rem -= cur
    _, _, pb_fp = decode_and_fp(pb_samples, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
    probe_baseline = [float(tanimoto_to_target(pb_fp, t).mean()) if pb_fp.shape[0] else float("nan")
                      for t in probe_targets]
    e.log(f"probe baseline <T> per target: {[round(x, 3) for x in probe_baseline]}  "
          f"(from {pb_fp.shape[0]}/{len(pb_samples)} valid uncond samples)")
    reporter = IntermediateReporter(
        e, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS, probe_targets, probe_baseline,
        cond_mean, cond_std, every_k=e.PROBE_EVERY_K, probe_n=e.PROBE_N, probe_steps=e.PROBE_STEPS,
        probe_weight=e.PROBE_WEIGHT, eta=e.GUIDED_ETA, omega=e.GUIDED_OMEGA,
        time_distortion=e.SAMPLE_TIME_DISTORTION, chunk=e.EVAL_CHUNK, ckpt_dir=e.path,
    )

    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS},
        accelerator="auto", devices=1, enable_progress_bar=True,
        enable_checkpointing=False, logger=False, gradient_clip_val=1.0, callbacks=[reporter],
    )
    e.log(f"Training adapter: epochs<={e.EPOCHS}  max_time={e.MAX_TIME_HOURS}h  batch={e.BATCH_SIZE}  "
          f"(probe every {e.PROBE_EVERY_K} epochs)")
    trainer.fit(module, train_dataloaders=train_loader)

    guid_path = module.guidance().save(os.path.join(e.path, "fp_guidance"))
    with open(os.path.join(e.path, "fp_guidance_stats.json"), "w") as f:
        json.dump({"fp_bits": e.FP_BITS, "fp_radius": e.FP_RADIUS, "atom_types": atom_types,
                   "cond_mean": cond_mean.tolist(), "cond_std": cond_std.tolist(),
                   "reward_sharpen": e.REWARD_SHARPEN, "pos_frac": e.POS_FRAC}, f)
    e.log(f"Saved adapter -> {guid_path}")

    # -- Evaluation ---------------------------------------------------------
    e.log("=" * 60)
    e.log("EVALUATION: guided (fingerprint-conditioned) vs unconditional baseline")
    base = base.to(device).eval()
    guidance = ExactGuidance(module.h.to(device).eval(), prop_mean=cond_mean, prop_std=cond_std)

    # target molecules (from the held-out pool)
    tgt_idx = random.sample(holdout_idx, min(e.N_EVAL_TARGETS, len(holdout_idx)))
    e.log(f"{len(tgt_idx)} target molecules from the held-out pool")

    # unconditional baseline pool (shared across targets, same sampler settings)
    e.log(f"Sampling {e.BASELINE_POOL} unconditional baseline molecules ...")
    base_sampler = Sampler(base, eta=e.GUIDED_ETA, omega=e.GUIDED_OMEGA,
                           sample_steps=e.GUIDED_STEPS, time_distortion=e.SAMPLE_TIME_DISTORTION)
    base_samples, remaining = [], e.BASELINE_POOL
    while remaining > 0:
        cur = min(e.EVAL_CHUNK, remaining)
        base_samples += base_sampler.sample(cur, device=device, show_progress=False)
        remaining -= cur
    _, _, base_fp = decode_and_fp(base_samples, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
    e.log(f"baseline: {len(base_samples)} sampled, {base_fp.shape[0]} valid")

    def guided_sample(target_vec, weight, n):
        guidance.set_weight(weight).set_target(target_vec)
        sampler = GuidedSampler(base, guidance, eta=e.GUIDED_ETA, omega=e.GUIDED_OMEGA,
                                sample_steps=e.GUIDED_STEPS, time_distortion=e.SAMPLE_TIME_DISTORTION)
        out, remaining = [], n
        while remaining > 0:
            cur = min(e.EVAL_CHUNK, remaining)
            out += sampler.sample(cur, device=device, show_progress=False)
            remaining -= cur
        return out

    summary = {"targets": [], "weights": e.GUIDED_WEIGHTS,
               "baseline_valid": int(base_fp.shape[0])}
    # aggregate Tanimoto distributions for the overview plot
    agg_guided = {w: [] for w in e.GUIDED_WEIGHTS}
    agg_baseline = []

    for ti, idx in enumerate(tgt_idx):
        target_vec = fp[idx]
        target_mol = Chem.MolFromSmiles(smiles_kept[idx])
        base_sim = tanimoto_to_target(base_fp, target_vec)
        agg_baseline.extend(base_sim.tolist())
        rec = {"index": int(idx), "smiles": smiles_kept[idx],
               "baseline_mean_tanimoto": float(base_sim.mean()) if base_sim.size else None,
               "per_weight": {}}
        e.log(f"[target {ti+1}/{len(tgt_idx)}] {smiles_kept[idx]}  "
              f"baseline<T>={rec['baseline_mean_tanimoto']}")

        best_for_grid = None  # (sims, mols) at the largest weight, for the grid
        for w in e.GUIDED_WEIGHTS:
            samples = guided_sample(target_vec, w, e.N_PER_TARGET)
            mols, smis, gfp = decode_and_fp(samples, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
            sims = tanimoto_to_target(gfp, target_vec)
            agg_guided[w].extend(sims.tolist())
            n_valid = len(mols)
            n_unique = len(set(smis))
            rec["per_weight"][str(w)] = {
                "n_sampled": len(samples), "n_valid": n_valid, "n_unique": n_unique,
                "validity": n_valid / len(samples) if samples else 0.0,
                "uniqueness": n_unique / n_valid if n_valid else 0.0,
                "mean_tanimoto": float(sims.mean()) if sims.size else None,
                "median_tanimoto": float(np.median(sims)) if sims.size else None,
                "max_tanimoto": float(sims.max()) if sims.size else None,
            }
            e.log(f"    w={w}: valid={n_valid}/{len(samples)} uniq={n_unique}  "
                  f"<T>={rec['per_weight'][str(w)]['mean_tanimoto']}  "
                  f"maxT={rec['per_weight'][str(w)]['max_tanimoto']}")
            best_for_grid = (sims, mols)

        # grid: target + most-similar generations (at the largest weight)
        if best_for_grid is not None and len(best_for_grid[1]) > 0:
            sims, mols = best_for_grid
            order = np.argsort(-sims)[:e.GRID_N]
            grid_mols = [target_mol] + [mols[j] for j in order]
            legends = ["TARGET"] + [f"T={sims[j]:.2f}" for j in order]
            img = Draw.MolsToGridImage(grid_mols, molsPerRow=5, subImgSize=(220, 220), legends=legends)
            img.save(os.path.join(e.path, f"grid_target{ti}.png"))
        summary["targets"].append(rec)

    # -- Aggregate metrics + plots ------------------------------------------
    agg_baseline = np.array(agg_baseline)
    summary["aggregate"] = {
        "baseline_mean_tanimoto": float(agg_baseline.mean()) if agg_baseline.size else None,
        "guided_mean_tanimoto": {str(w): (float(np.mean(agg_guided[w])) if agg_guided[w] else None)
                                 for w in e.GUIDED_WEIGHTS},
        "lift_over_baseline": {str(w): (float(np.mean(agg_guided[w]) - agg_baseline.mean())
                                        if agg_guided[w] and agg_baseline.size else None)
                               for w in e.GUIDED_WEIGHTS},
    }
    e.commit_json("fingerprint_guidance_metrics.json", summary)

    # Tanimoto distribution: baseline vs guided (per weight), aggregated over targets
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bins = np.linspace(0, 1, 41)
    if agg_baseline.size:
        ax.hist(agg_baseline, bins=bins, density=True, color="0.6", label="unconditional baseline", zorder=1)
    cmap = plt.get_cmap("viridis")
    for wi, w in enumerate(e.GUIDED_WEIGHTS):
        if agg_guided[w]:
            ax.hist(agg_guided[w], bins=bins, density=True, histtype="stepfilled",
                    color=cmap(0.2 + 0.6 * wi / max(1, len(e.GUIDED_WEIGHTS) - 1)), alpha=0.5,
                    label=f"guided (w={w}, <T>={np.mean(agg_guided[w]):.3f})", zorder=2 + wi)
    b = summary["aggregate"]["baseline_mean_tanimoto"]
    ax.set_title(f"Fingerprint-guided steering on ZINC  |  baseline <T>={b:.3f}"
                 if b is not None else "Fingerprint-guided steering on ZINC")
    ax.set_xlabel("Tanimoto to target fingerprint")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    e.commit_fig("tanimoto_distributions.png", fig)

    e.log("=" * 60)
    e.log(f"baseline <T>={summary['aggregate']['baseline_mean_tanimoto']}")
    for w in e.GUIDED_WEIGHTS:
        e.log(f"guided w={w}: <T>={summary['aggregate']['guided_mean_tanimoto'][str(w)]}  "
              f"lift={summary['aggregate']['lift_over_baseline'][str(w)]}")
    e.log("Evaluation complete.")


@experiment.testing
def testing(e: Experiment):
    e.SUBSET = 300
    e.N_HOLDOUT = 40
    e.N_NEIGHBORS = 8
    e.EPOCHS = 1
    e.BATCH_SIZE = 16
    e.MAX_TIME_HOURS = 0.2
    e.H_LAYERS = 2
    e.H_HIDDEN = 32
    e.H_MLP_DIM = 64
    e.H_HEADS = 2
    e.GUIDED_WEIGHTS = [3.0]
    e.N_EVAL_TARGETS = 2
    e.N_PER_TARGET = 8
    e.BASELINE_POOL = 16
    e.EVAL_CHUNK = 8
    e.GUIDED_STEPS = 5
    e.GRID_N = 4
    e.PROBE_EVERY_K = 1
    e.PROBE_N_TARGETS = 2
    e.PROBE_N = 6
    e.PROBE_STEPS = 5
    e.PROBE_BASELINE_N = 8


experiment.run_if_main()
