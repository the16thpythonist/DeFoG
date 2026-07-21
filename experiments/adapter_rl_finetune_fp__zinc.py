"""RL-fine-tune (GDPO) a trained frozen-base FINGERPRINT adapter to tighten its
Tanimoto conditioning AND penalize disconnected/invalid molecules (connectivity
FIRST). Base frozen; only the adapter moves. Reward ordering (best->worst):
connected -> Tanimoto(FP(mol), target) in [0,1]; disconnected -> disconnect_reward;
invalid -> invalid_reward (< disconnect). GRPO advantage grouped by target FP.

Eval (pre & post): per held-out target FP, condition, sample (w-sweep, 500 steps),
report mean Tanimoto LIFT over the unconditional baseline + disconnection%.

Usage:
    python experiments/adapter_rl_finetune_fp__zinc.py --__TESTING__ True
    python experiments/adapter_rl_finetune_fp__zinc.py \
        --ADAPTER_CKPT "'.../fp_adapter.ckpt'" --KL_COEF 0.2
"""
import os
import json
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import build_encoders, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, AdaLNAdapter, AdapterComposition, ConditionBranch, AdaptedSampler,
    Sampler, AdapterGDPOTrainer,
)
from defog.core.data import dense_to_pyg

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
ATOM_TYPES: list = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")
ADAPTER_CKPT: str = ""            # trained fingerprint adapter to RL-finetune
FP_BITS: int = 512
FP_RADIUS: int = 2

# --- RL (GDPO) ---
MAX_TIME_HOURS: float = 4.0
MAX_ITERS: int = 100000
ROLLOUT_SIZE: int = 64
N_GROUPS: int = 8                 # distinct target FPs / iteration
ROLLOUT_STEPS: int = 250
ROLLOUT_ETA: float = 5.0
ROLLOUT_OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
SUBSAMPLE_STEPS: int = 16
MINIBATCH: int = 16
LR: float = 1e-5
KL_COEF: float = 0.2
EMA_DECAY: float = 0.999
GRAD_CLIP: float = 1.0
LAMBDA_EDGE: float = 1.0
# connectivity-FIRST: connected (Tanimoto in [0,1]) > disconnected > invalid
INVALID_REWARD: float = -1.0
DISCONNECT_REWARD: float = -0.5
N_HOLDOUT: int = 4000             # held-out molecules -> target FP pool (unseen)

# --- Evaluation (pre & post) ---
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
GUIDANCE_WEIGHTS: list = [1.0]  # eval the optimized policy: RL rolls out / scores at w=1 only
N_TARGETS: int = 6                # held-out eval target FPs
N_PER_TARGET: int = 64
N_BASELINE: int = 128
EVAL_CHUNK: int = 32
LOG_EVERY: int = 10
PROBE_EVERY: int = 0

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ============================================================================
def morgan_bits(mol, radius, n_bits):
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    a = np.zeros((n_bits,), dtype=np.float32); ConvertToNumpyArray(bv, a); return a


def tanimoto(fp_mat, tgt):
    inter = fp_mat @ tgt
    union = fp_mat.sum(1) + tgt.sum() - inter
    return np.where(union > 0, inter / union, 0.0)


def decode_fp_disc(samples, ad, bd, radius, n_bits):
    """(FP matrix of CONNECTED+valid mols, n_total, n_valid, n_disconnected)."""
    fps, n_valid, n_disc = [], 0, 0
    for s in samples:
        mol = pyg_data_to_mol(s, ad, bd)
        smi = mol_to_smiles(mol) if mol is not None else None
        m = Chem.MolFromSmiles(smi) if smi is not None else None
        if m is None:
            continue
        n_valid += 1
        if "." in smi:
            n_disc += 1
            continue
        try:
            fps.append(morgan_bits(m, radius, n_bits))
        except Exception:
            pass
    mat = np.stack(fps) if fps else np.zeros((0, n_bits), dtype=np.float32)
    return mat, len(samples), n_valid, n_disc


def chunked(sampler, n, chunk, device):
    out, rem = [], n
    while rem > 0:
        cur = min(chunk, rem)
        out += sampler.sample(cur, device=device, show_progress=False)
        rem -= cur
    return out


class FPMatchReward:
    """Connectivity-FIRST Tanimoto reward. connected -> Tanimoto(FP(mol), target) in
    [0,1]; disconnected -> disconnect_reward; invalid -> invalid_reward. Since Tanimoto
    is non-negative, ANY connected molecule outranks any disconnected/invalid one."""

    def __init__(self, atom_decoder, bond_decoder, radius, n_bits,
                 invalid_reward=-1.0, disconnect_reward=-0.5):
        self.ad, self.bd, self.radius, self.n_bits = atom_decoder, bond_decoder, radius, n_bits
        self.invalid, self.disconnect = float(invalid_reward), float(disconnect_reward)

    def __call__(self, X1, E1, node_mask, cond):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.invalid)
        tgt = cond.detach().cpu().float().numpy()          # (K, n_bits) RAW target FPs
        for i, d in enumerate(datas):
            mol = pyg_data_to_mol(d, self.ad, self.bd)
            smi = mol_to_smiles(mol) if mol is not None else None
            m = Chem.MolFromSmiles(smi) if smi is not None else None
            if m is None:
                continue                                   # invalid -> floor
            if "." in smi:
                out[i] = self.disconnect                   # disconnected -> penalty
                continue
            try:
                fp = morgan_bits(m, self.radius, self.n_bits)
                t = tgt[i]
                inter = float(fp @ t); union = float(fp.sum() + t.sum() - inter)
                out[i] = (inter / union) if union > 0 else 0.0
            except Exception:
                pass
        return out


def make_fp_condition_sampler(fp_pool_t, K, G, seed):
    gen = torch.Generator().manual_seed(seed)
    per = max(1, K // G)
    M = fp_pool_t.size(0)

    def sampler():
        idx = torch.randint(0, M, (G,), generator=gen)
        targs = fp_pool_t[idx]                              # (G, n_bits)
        cond = targs.repeat_interleave(per, dim=0)
        groups = torch.arange(G).repeat_interleave(per)
        if cond.size(0) < K:
            extra = K - cond.size(0)
            cond = torch.cat([cond, targs[:extra]], 0)
            groups = torch.cat([groups, torch.arange(extra)])
        return cond[:K].clone(), groups[:K].clone()
    return sampler


def fp_eval(base, adapter, ad, bd, radius, n_bits, target_fps, weights,
            steps, eta, omega, td, n_per, n_base, chunk, device):
    """Per w: mean Tanimoto LIFT over baseline + mean disconnection%, over the eval targets."""
    bs = Sampler(base, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
    bfp, bt, bvalid, bdisc = decode_fp_disc(chunked(bs, n_base, chunk, device), ad, bd, radius, n_bits)
    base_tan = [float(tanimoto(bfp, t).mean()) if bfp.shape[0] else np.nan for t in target_fps]
    out = {"baseline_tan": float(np.nanmean(base_tan)),
           "baseline_disc": (bdisc / bvalid) if bvalid else None, "per_w": {}}
    for w in weights:
        lifts, discs, gtans = [], [], []
        for k, tfp in enumerate(target_fps):
            comp = AdapterComposition([ConditionBranch(adapter, torch.as_tensor(tfp, dtype=torch.float32), w)],
                                      base=base, mode="product")
            samp = AdaptedSampler(base, comp, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
            gfp, gt, gvalid, gdisc = decode_fp_disc(chunked(samp, n_per, chunk, device), ad, bd, radius, n_bits)
            gtan = float(tanimoto(gfp, tfp).mean()) if gfp.shape[0] else np.nan
            gtans.append(gtan); lifts.append(gtan - base_tan[k])
            discs.append((gdisc / gvalid) if gvalid else np.nan)
        out["per_w"][str(w)] = {"tan": float(np.nanmean(gtans)), "lift": float(np.nanmean(lifts)),
                                "disc": float(np.nanmean(discs))}
    return out


# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"RL-finetune FINGERPRINT adapter (GDPO, connectivity-first) kl_coef={e.KL_COEF}")
    import pytorch_lightning as pl
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(e.ATOM_TYPES, e.BOND_TYPES)

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0
    if e.ADAPTER_CKPT:
        adapter = AdaLNAdapter.load(e.ADAPTER_CKPT, device=device)
        adapter.check_compatible(base)
        e.log(f"loaded FP adapter from {e.ADAPTER_CKPT} (cond_dim={adapter.cond_dim} "
              f"interior_ff={adapter.interior_ff} interior_attn={adapter.interior_attn})")
    else:
        cm = np.zeros(e.FP_BITS, dtype=np.float32); cs = np.ones(e.FP_BITS, dtype=np.float32)
        adapter = AdaLNAdapter.for_base(base, cond_dim=e.FP_BITS, hidden=32, cond_mean=cm, cond_std=cs,
                                        name="fp_adapter", cond_type=f"morgan{e.FP_BITS}").to(device)
        e.log("[fresh/untrained] adapter (smoke only)")

    # held-out target FP pool (unseen molecules)
    df = pd.read_csv(e.CSV_PATH)
    hold_smiles = df[e.SMILES_COLUMN].sample(min(e.N_HOLDOUT, len(df)), random_state=e.SEED + 1).tolist()
    fp_pool = []
    for smi in hold_smiles:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            fp_pool.append(morgan_bits(m, e.FP_RADIUS, e.FP_BITS))
    fp_pool = np.stack(fp_pool)
    fp_pool_t = torch.as_tensor(fp_pool, dtype=torch.float32)
    e.log(f"held-out FP pool: {fp_pool.shape[0]} ({e.FP_BITS}-bit r{e.FP_RADIUS})")

    eval_targets = fp_pool[:e.N_TARGETS]                    # fixed eval targets (first N)

    reward = FPMatchReward(atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS,
                           invalid_reward=e.INVALID_REWARD, disconnect_reward=e.DISCONNECT_REWARD)
    cond_sampler = make_fp_condition_sampler(fp_pool_t, e.ROLLOUT_SIZE, e.N_GROUPS, e.SEED)

    def _fmt(ev):
        parts = [f"baseline<T>={ev['baseline_tan']:.3f} disc={ev['baseline_disc']}"]
        for w in e.GUIDANCE_WEIGHTS:
            r = ev["per_w"][str(w)]
            parts.append(f"w{w}: <T>={r['tan']:.3f} lift={r['lift']:+.3f} disc={r['disc']*100:.0f}%")
        return " | ".join(parts)

    def eval_now(tag):
        ev = fp_eval(base, adapter, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS, eval_targets,
                     e.GUIDANCE_WEIGHTS, e.EVAL_STEPS, e.ETA, e.OMEGA, e.TIME_DISTORTION,
                     e.N_PER_TARGET, e.N_BASELINE, e.EVAL_CHUNK, device)
        e.log(f"[{tag}] {_fmt(ev)}")
        return ev

    e.log("=== PRE-RL eval ===")
    pre_ev = eval_now("pre-RL")

    trainer = AdapterGDPOTrainer(
        base, adapter, reward, kl_coef=e.KL_COEF, lr=e.LR, ema_decay=e.EMA_DECAY,
        rollout_size=e.ROLLOUT_SIZE, sample_steps=e.ROLLOUT_STEPS, eta=e.ROLLOUT_ETA,
        omega=e.ROLLOUT_OMEGA, time_distortion=e.TIME_DISTORTION, condition_sampler=cond_sampler,
        subsample_steps=e.SUBSAMPLE_STEPS, minibatch_size=e.MINIBATCH, lambda_edge=e.LAMBDA_EDGE,
        grad_clip=e.GRAD_CLIP, seed=e.SEED, device=device,
    )

    e.log(f"=== RL: max_time={e.MAX_TIME_HOURS}h rollout(K={e.ROLLOUT_SIZE},G={e.N_GROUPS},"
          f"{e.ROLLOUT_STEPS} steps) lr={e.LR} kl={e.KL_COEF} ===")
    t0 = time.time()
    deadline = t0 + e.MAX_TIME_HOURS * 3600
    history, it = [], 0
    while it < e.MAX_ITERS and time.time() < deadline:
        m = trainer.step()
        history.append(m)
        if it % e.LOG_EVERY == 0:
            e.log(f"[iter {it}] reward={m['reward_mean']:+.3f}(min {m['reward_min']:+.2f}) "
                  f"kl={m['kl']:.4f} adv_std={m['adv_std']:.2f} grad={m['grad_norm']:.2f}")
        it += 1
    e.log(f"RL done: {it} iterations in {(time.time()-t0)/60:.1f} min")
    e.commit_json("rl_history.json", history)

    ckpt = trainer.save(os.path.join(e.path, "fp_adapter_rl"))
    e.log(f"Saved RL'd adapter -> {ckpt}")
    if trainer.ema is not None:
        trainer.ema.copy_to(adapter)

    e.log("=== POST-RL eval ===")
    post_ev = eval_now("post-RL")

    summary = {"kl_coef": e.KL_COEF, "iterations": it, "pre": pre_ev, "post": post_ev}
    e.log("=" * 60)
    e.log("TANIMOTO LIFT (higher=tighter)  +  DISCONNECTION% (lower=better) — pre -> post")
    for w in e.GUIDANCE_WEIGHTS:
        pr, po = pre_ev["per_w"][str(w)], post_ev["per_w"][str(w)]
        e.log(f"w={w}: lift {pr['lift']:+.3f} -> {po['lift']:+.3f} | "
              f"disc {pr['disc']*100:.0f}% -> {po['disc']*100:.0f}% | <T> {pr['tan']:.3f} -> {po['tan']:.3f}")
    e.commit_json("rl_fp_metrics.json", summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ws = [str(w) for w in e.GUIDANCE_WEIGHTS]; x = np.arange(len(ws))
    ax1.bar(x - 0.2, [pre_ev["per_w"][w]["lift"] for w in ws], 0.4, label="pre-RL", color="0.6")
    ax1.bar(x + 0.2, [post_ev["per_w"][w]["lift"] for w in ws], 0.4, label="post-RL", color="#4c72b0")
    ax1.set_xticks(x); ax1.set_xticklabels([f"w{w}" for w in ws]); ax1.set_ylabel("Tanimoto lift")
    ax1.set_title("conditioning (higher=better)"); ax1.legend()
    ax2.bar(x - 0.2, [pre_ev["per_w"][w]["disc"] for w in ws], 0.4, label="pre-RL", color="0.6")
    ax2.bar(x + 0.2, [post_ev["per_w"][w]["disc"] for w in ws], 0.4, label="post-RL", color="#c44e52")
    ax2.set_xticks(x); ax2.set_xticklabels([f"w{w}" for w in ws]); ax2.set_ylabel("disconnection fraction")
    ax2.set_title("connectivity (lower=better)"); ax2.legend()
    fig.suptitle(f"FP adapter RL (connectivity-first, kl={e.KL_COEF})")
    fig.tight_layout()
    e.commit_fig("fp_rl_pre_post.png", fig)
    e.log("Done.")


@experiment.testing
def testing(e: Experiment):
    e.MAX_TIME_HOURS = 0.05
    e.MAX_ITERS = 2
    e.ROLLOUT_SIZE = 8
    e.N_GROUPS = 2
    e.ROLLOUT_STEPS = 5
    e.SUBSAMPLE_STEPS = 2
    e.MINIBATCH = 4
    e.EVAL_STEPS = 5
    e.N_TARGETS = 2
    e.N_PER_TARGET = 8
    e.N_BASELINE = 8
    e.EVAL_CHUNK = 8
    e.GUIDANCE_WEIGHTS = [1.0]
    e.N_HOLDOUT = 300
    e.LOG_EVERY = 1
    e.BASE_CKPT = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")


experiment.run_if_main()
