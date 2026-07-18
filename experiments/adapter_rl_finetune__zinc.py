"""
RL-fine-tune (GDPO) a trained frozen-base scalar-property ADAPTER to TIGHTEN its
conditioning. The base stays frozen; only the adapter's params move (composability
preserved). Reward = conditional property match ``-|prop(mol) - target|``, targets
sampled AMORTIZED over the full property range; GRPO advantage grouped by target;
KL guard to the PRE-RL adapter (reward-hacking guard). See defog.core.rl
.AdapterGDPOTrainer.

Evaluates the SAME steering protocol (achieved-property MAE, w-sweep, 500 steps) on
the low/high 5th/95th-percentile targets BEFORE and AFTER RL, to measure the
tightening.

Usage:
    python experiments/adapter_rl_finetune__zinc.py --__TESTING__ True
    python experiments/adapter_rl_finetune__zinc.py --PROPERTY "'logp'" \
        --ADAPTER_CKPT "'.../logp_adapter.ckpt'" --KL_COEF 0.1
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
from rdkit.Chem import Crippen, Descriptors
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
PROP_FNS = {"logp": lambda m: float(Crippen.MolLogP(m)),
            "tpsa": lambda m: float(Descriptors.TPSA(m))}

# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
ATOM_TYPES: list = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")
ADAPTER_CKPT: str = ""           # trained scalar-property adapter to RL-finetune
PROPERTY: str = "logp"           # logp | tpsa

# --- RL (GDPO) ---
MAX_TIME_HOURS: float = 4.0      # RL-loop wall budget (evals happen outside it)
MAX_ITERS: int = 100000          # unreached cap; time bounds it
ROLLOUT_SIZE: int = 64           # K trajectories / iteration
N_GROUPS: int = 8                # distinct targets / iteration (K/N_GROUPS rollouts each)
ROLLOUT_STEPS: int = 250
ROLLOUT_ETA: float = 5.0
ROLLOUT_OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
SUBSAMPLE_STEPS: int = 16
MINIBATCH: int = 16
LR: float = 1e-5
KL_COEF: float = 0.1             # swept per-arm; KL to the pre-RL adapter
EMA_DECAY: float = 0.999
GRAD_CLIP: float = 1.0
LAMBDA_EDGE: float = 1.0
INVALID_REWARD: float = -10.0
TARGET_RANGE_PCT: list = [1, 99]  # amortized RL targets sampled uniform over this range
LOG_EVERY: int = 10
PROBE_EVERY: int = 40

# --- Evaluation (pre & post RL) ---
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
GUIDANCE_WEIGHTS: list = [1.0, 2.0]
TARGET_PERCENTILES: list = [5, 95]
LEVEL_NAMES: list = ["low", "high"]
N_PER_TARGET: int = 128
N_BASELINE: int = 256
EVAL_CHUNK: int = 32
REF_SUBSAMPLE: int = 20000

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ============================================================================
# Helpers
# ============================================================================
def props_of(samples, atom_decoder, bond_decoder, prop_fn):
    vals = []
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi is not None and Chem.MolFromSmiles(smi) is not None:
            try:
                vals.append(prop_fn(mol))
            except Exception:
                pass
    return np.asarray(vals, dtype=float)


def chunked(sampler, n, chunk, device):
    out, rem = [], n
    while rem > 0:
        cur = min(chunk, rem)
        out += sampler.sample(cur, device=device, show_progress=False)
        rem -= cur
    return out


class PropertyMatchReward:
    """Conditional RL reward: ``-|property(mol) - target| / scale`` per row; invalid
    molecules get ``invalid_reward``. Higher = closer to the target."""

    def __init__(self, atom_decoder, bond_decoder, prop_fn, scale=1.0, invalid_reward=-10.0):
        self.ad, self.bd, self.prop_fn = atom_decoder, bond_decoder, prop_fn
        self.scale, self.invalid = float(scale), float(invalid_reward)

    def __call__(self, X1, E1, node_mask, cond):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.invalid)
        tgt = cond.reshape(-1).tolist()
        for i, d in enumerate(datas):
            mol = pyg_data_to_mol(d, self.ad, self.bd)
            if mol is None:
                continue
            try:
                out[i] = -abs(float(self.prop_fn(mol)) - tgt[i]) / self.scale
            except Exception:
                pass
        return out


def make_condition_sampler(p_lo, p_hi, K, G, seed):
    gen = torch.Generator().manual_seed(seed)
    per = max(1, K // G)

    def sampler():
        targets = torch.rand(G, generator=gen) * (p_hi - p_lo) + p_lo
        cond = targets.repeat_interleave(per).unsqueeze(-1)
        groups = torch.arange(G).repeat_interleave(per)
        if cond.size(0) < K:                       # pad to exactly K
            extra = K - cond.size(0)
            cond = torch.cat([cond, targets[:extra].unsqueeze(-1)])
            groups = torch.cat([groups, torch.arange(extra)])
        return cond[:K], groups[:K]
    return sampler


def steer_eval(base, adapter, atom_decoder, bond_decoder, prop_fn, targets, weights,
               steps, eta, omega, td, n_per, n_base, chunk, device):
    """Achieved property (mean, MAE) when steering to low/high targets + baseline."""
    bs = Sampler(base, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
    base_vals = props_of(chunked(bs, n_base, chunk, device), atom_decoder, bond_decoder, prop_fn)
    out = {"baseline_mean": float(base_vals.mean()) if base_vals.size else None, "levels": {}}
    for lvl, tgt in targets.items():
        out["levels"][lvl] = {"target": tgt, "per_w": {}}
        for w in weights:
            comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([tgt]), w)], base=base, mode="product")
            samp = AdaptedSampler(base, comp, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
            gv = props_of(chunked(samp, n_per, chunk, device), atom_decoder, bond_decoder, prop_fn)
            out["levels"][lvl]["per_w"][str(w)] = {
                "n": int(gv.size), "mean": float(gv.mean()) if gv.size else None,
                "mae": float(np.mean(np.abs(gv - tgt))) if gv.size else None}
    return out


def _fmt(ev, weights):
    parts = []
    for lvl, d in ev["levels"].items():
        for w in weights:
            r = d["per_w"][str(w)]
            parts.append(f"{lvl}->{d['target']:.1f}@w{w}: mean={r['mean']} mae={r['mae']}")
    return " | ".join(parts)


# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"RL-finetune adapter (GDPO) for property={e.PROPERTY}  kl_coef={e.KL_COEF}")
    import pytorch_lightning as pl
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prop_fn = PROP_FNS[e.PROPERTY]
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(e.ATOM_TYPES, e.BOND_TYPES)

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0
    if e.ADAPTER_CKPT:
        adapter = AdaLNAdapter.load(e.ADAPTER_CKPT, device=device)
        adapter.check_compatible(base)
        e.log(f"loaded {e.PROPERTY} adapter from {e.ADAPTER_CKPT} (cond_dim={adapter.cond_dim})")
    else:
        adapter = AdaLNAdapter.for_base(base, cond_dim=1, hidden=32, cond_mean=[0.0], cond_std=[1.0],
                                        name=f"{e.PROPERTY}_adapter", cond_type=e.PROPERTY).to(device)
        e.log("[fresh/untrained] adapter (smoke only)")

    # property distribution -> RL target range + eval targets
    df = pd.read_csv(e.CSV_PATH)
    ref_smiles = df[e.SMILES_COLUMN].sample(min(e.REF_SUBSAMPLE, len(df)), random_state=e.SEED).tolist()
    _mols = [Chem.MolFromSmiles(s) for s in ref_smiles]
    vals = np.asarray([prop_fn(m) for m in _mols if m is not None])
    p_lo, p_hi = [float(x) for x in np.percentile(vals, e.TARGET_RANGE_PCT)]
    prop_std = float(vals.std()) or 1.0
    targets = dict(zip(e.LEVEL_NAMES, [float(x) for x in np.percentile(vals, e.TARGET_PERCENTILES)]))
    e["eval/targets"] = targets
    e.log(f"RL target range [{p_lo:.1f}, {p_hi:.1f}]  eval targets {targets}  prop_std={prop_std:.2f}")

    reward = PropertyMatchReward(atom_decoder, bond_decoder, prop_fn, scale=prop_std, invalid_reward=e.INVALID_REWARD)
    cond_sampler = make_condition_sampler(p_lo, p_hi, e.ROLLOUT_SIZE, e.N_GROUPS, e.SEED)

    def eval_now(tag):
        ev = steer_eval(base, adapter, atom_decoder, bond_decoder, prop_fn, targets, e.GUIDANCE_WEIGHTS,
                        e.EVAL_STEPS, e.ETA, e.OMEGA, e.TIME_DISTORTION, e.N_PER_TARGET, e.N_BASELINE,
                        e.EVAL_CHUNK, device)
        e.log(f"[{tag}] baseline={ev['baseline_mean']}  {_fmt(ev, e.GUIDANCE_WEIGHTS)}")
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

    e.log(f"=== RL fine-tuning: max_time={e.MAX_TIME_HOURS}h rollout(K={e.ROLLOUT_SIZE},G={e.N_GROUPS},"
          f"{e.ROLLOUT_STEPS} steps) lr={e.LR} kl={e.KL_COEF} ===")
    t0 = time.time()
    deadline = t0 + e.MAX_TIME_HOURS * 3600
    history = []
    it = 0
    while it < e.MAX_ITERS and time.time() < deadline:
        m = trainer.step()
        history.append(m)
        if it % e.LOG_EVERY == 0:
            e.log(f"[iter {it}] reward={m['reward_mean']:+.3f}(min {m['reward_min']:+.2f}) "
                  f"kl={m['kl']:.4f} adv_std={m['adv_std']:.2f} grad={m['grad_norm']:.2f}")
        if e.PROBE_EVERY and it > 0 and it % e.PROBE_EVERY == 0:
            pv = eval_now(f"iter{it}")
            e.track("rl_progress_reward", None) if False else None
        it += 1
    e.log(f"RL done: {it} iterations in {(time.time()-t0)/60:.1f} min")
    e.commit_json("rl_history.json", history)

    ckpt = trainer.save(os.path.join(e.path, f"{e.PROPERTY}_adapter_rl"))
    e.log(f"Saved RL'd adapter -> {ckpt}")
    # load EMA weights into the live adapter for the post eval
    if trainer.ema is not None:
        trainer.ema.copy_to(adapter)

    e.log("=== POST-RL eval ===")
    post_ev = eval_now("post-RL")

    # -- compare + plot MAE pre vs post --------------------------------------
    summary = {"property": e.PROPERTY, "kl_coef": e.KL_COEF, "iterations": it,
               "targets": targets, "pre": pre_ev, "post": post_ev, "mae_delta": {}}
    e.log("=" * 60)
    for lvl in e.LEVEL_NAMES:
        for w in e.GUIDANCE_WEIGHTS:
            pre = pre_ev["levels"][lvl]["per_w"][str(w)]["mae"]
            post = post_ev["levels"][lvl]["per_w"][str(w)]["mae"]
            d = (post - pre) if (pre is not None and post is not None) else None
            summary["mae_delta"][f"{lvl}_w{w}"] = {"pre": pre, "post": post, "delta": d}
            e.log(f"{lvl} w={w}: MAE {pre} -> {post}  (delta {d})")
    e.commit_json("rl_finetune_metrics.json", summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{lvl}\nw{w}" for lvl in e.LEVEL_NAMES for w in e.GUIDANCE_WEIGHTS]
    pre_maes = [pre_ev["levels"][lvl]["per_w"][str(w)]["mae"] or np.nan for lvl in e.LEVEL_NAMES for w in e.GUIDANCE_WEIGHTS]
    post_maes = [post_ev["levels"][lvl]["per_w"][str(w)]["mae"] or np.nan for lvl in e.LEVEL_NAMES for w in e.GUIDANCE_WEIGHTS]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, pre_maes, 0.4, label="pre-RL", color="0.6")
    ax.bar(x + 0.2, post_maes, 0.4, label="post-RL", color="#55a868")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(f"{e.PROPERTY} MAE to target")
    ax.set_title(f"Adapter RL tightening: {e.PROPERTY} (kl={e.KL_COEF})  lower=better")
    ax.legend(); fig.tight_layout()
    e.commit_fig("mae_pre_post.png", fig)
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
    e.N_PER_TARGET = 8
    e.N_BASELINE = 8
    e.EVAL_CHUNK = 8
    e.GUIDANCE_WEIGHTS = [2.0]
    e.REF_SUBSAMPLE = 300
    e.PROBE_EVERY = 0
    e.LOG_EVERY = 1
    e.BASE_CKPT = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")


experiment.run_if_main()
