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
from rdkit.Chem import Crippen, Descriptors, QED
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import build_encoders, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, AdaLNAdapter, AdapterComposition, ConditionBranch, AdaptedSampler,
    Sampler, AdapterGDPOTrainer, PropertyHead,
)
from defog.core.data import dense_to_pyg, to_dense

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_prop_fns():
    """RDKit ground-truth property fns -- used for EVAL/validation even when the REWARD is a
    learned head. SAScore lives in RDKit's Contrib dir, so add it to sys.path."""
    fns = {"logp": lambda m: float(Crippen.MolLogP(m)),
           "tpsa": lambda m: float(Descriptors.TPSA(m)),
           "qed": lambda m: float(QED.qed(m))}
    try:
        import sys as _sys
        from rdkit.Chem import RDConfig
        _sa = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if _sa not in _sys.path:
            _sys.path.append(_sa)
        import sascorer
        fns["sascore"] = lambda m: float(sascorer.calculateScore(m))
    except Exception as _ex:
        print(f"[warn] SAScore (RDKit Contrib) unavailable: {_ex}")
    return fns


PROP_FNS = _make_prop_fns()

# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
ATOM_TYPES: list = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")
ADAPTER_CKPT: str = ""           # trained scalar-property adapter to RL-finetune
PROPERTY: str = "logp"           # logp | tpsa | qed | sascore
# --- reward source: RDKit ground-truth vs a learned property HEAD ---
# :param REWARD_SOURCE: "rdkit" (closed-form true fn) | "head" (learned PropertyHead -> tests
#     head-as-reward for properties with no true fn). With "head" the REWARD, the early-stop
#     PROBE and the best-SEED selection ALL use the head; EVAL additionally reports RDKit truth
#     to VALIDATE whether optimizing the head moved the real property.
REWARD_SOURCE: str = "rdkit"
# :param HEAD_CKPT: PropertyHead ckpt (required for REWARD_SOURCE="head").
HEAD_CKPT: str = ""

# --- RL (GDPO) ---
MAX_TIME_HOURS: float = 4.0      # RL-loop wall budget (evals happen outside it)
MAX_ITERS: int = 100000          # unreached cap; time bounds it
ROLLOUT_SIZE: int = 128          # K trajectories / iteration (= N_GROUPS * rollouts-per-group)
N_GROUPS: int = 8                # distinct targets / iteration -> 16 rollouts each
CRN: bool = True                 # common random numbers: share start noise+size within each target group
ROLLOUT_STEPS: int = 250
ROLLOUT_ETA: float = 1.0         # sweep winner (job 1006501): low eta best under CRN; sole within-group diversity source
ROLLOUT_OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
SUBSAMPLE_STEPS: int = 16
MINIBATCH: int = 16
LR: float = 1e-4                 # was 1e-5 (20x below the adapter's training LR) -> too weak to learn; raised
KL_COEF: float = 0.1             # swept per-arm; KL to the pre-RL adapter
EMA_DECAY: float = 0.9           # was 0.999 -> EMA lagged near-original on short runs; fast EMA tracks the live weights
GRAD_CLIP: float = 1.0
LAMBDA_EDGE: float = 1.0
# --- connectivity-FIRST reward: connected on-target > connected off-target >
#     disconnected > invalid. prop term clamped so ANY connected molecule outranks
#     ANY disconnected one, while property still gets a full gradient among connected. ---
INVALID_REWARD: float = -10.0     # unparseable molecule (worst)
DISCONNECT_REWARD: float = -4.0   # valid but fragmented ('.' in SMILES); < -PROP_CLAMP
PROP_CLAMP: float = 3.0           # connected reward in [-PROP_CLAMP, 0]
TARGET_RANGE_PCT: list = [1, 99]  # amortized RL targets sampled uniform over this range
LOG_EVERY: int = 10
PROBE_EVERY: int = 40
# --- early stopping: keep the best EMA snapshot by periodic probe MAE (validity-gated) ---
EARLY_STOP: bool = True
PROBE_STEPS: int = 100           # cheap denoising steps for the periodic probe
PROBE_N_PER: int = 48            # samples per eval target during a probe
VALIDITY_FLOOR_MARGIN: float = 0.05  # a snapshot must keep validity >= (pre-RL - this) to be eligible

# --- Evaluation (pre & post RL) ---
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
GUIDANCE_WEIGHTS: list = [1.0]  # eval the optimized policy: RL rolls out / scores at w=1 only
TARGET_PERCENTILES: list = [5, 95]
LEVEL_NAMES: list = ["low", "high"]
N_PER_TARGET: int = 128
N_BASELINE: int = 256
EVAL_CHUNK: int = 32
EVAL_SEED: int = 1234            # PAIRED eval: fixed rng so pre/post/probe sample the SAME draws -> the delta is real, not sampling noise
REF_SUBSAMPLE: int = 20000

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ============================================================================
# Helpers
# ============================================================================
def gen_eval(samples, atom_decoder, bond_decoder):
    """Return (connected+valid RDKit Mols, n_total, n_valid, n_disconnected). Mols are the
    SMILES-reparsed connected molecules; the caller computes RDKit and/or head properties on
    them so ONE generation can be scored under both."""
    mols, n_valid, n_disc = [], 0, 0
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        m = Chem.MolFromSmiles(smi) if smi is not None else None
        if m is None:
            continue
        n_valid += 1
        if "." in smi:
            n_disc += 1
            continue
        mols.append(m)
    return mols, len(samples), n_valid, n_disc


def _rdkit_stats(mols, tgt, prop_fn):
    """(MAE, mean, n) of prop_fn over mols vs tgt; skips prop errors."""
    vals = []
    for m in mols:
        try:
            v = float(prop_fn(m))
        except Exception:
            continue
        if v == v:
            vals.append(v)
    if not vals:
        return None, None, 0
    a = np.asarray(vals)
    return float(np.mean(np.abs(a - tgt))), float(a.mean()), len(a)


def chunked(sampler, n, chunk, device):
    out, rem = [], n
    while rem > 0:
        cur = min(chunk, rem)
        out += sampler.sample(cur, device=device, show_progress=False)
        rem -= cur
    return out


class PropertyMatchReward:
    """Connectivity-FIRST conditional reward. Ordering (best->worst): connected+valid
    on-target > connected+valid off-target > disconnected > invalid. The property term
    ``-min(|prop-target|/scale, clamp)`` lives in ``[-clamp, 0]``; ``disconnect_reward``
    is below ``-clamp`` so ANY connected molecule outranks ANY disconnected one, while
    property still gets a full gradient among connected molecules."""

    def __init__(self, atom_decoder, bond_decoder, prop_fn, scale=1.0,
                 invalid_reward=-10.0, disconnect_reward=-4.0, prop_clamp=3.0):
        self.ad, self.bd, self.prop_fn = atom_decoder, bond_decoder, prop_fn
        self.scale, self.invalid = float(scale), float(invalid_reward)
        self.disconnect, self.clamp = float(disconnect_reward), float(prop_clamp)

    def __call__(self, X1, E1, node_mask, cond):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.invalid)
        tgt = cond.reshape(-1).tolist()
        for i, d in enumerate(datas):
            mol = pyg_data_to_mol(d, self.ad, self.bd)
            smi = mol_to_smiles(mol) if mol is not None else None
            m = Chem.MolFromSmiles(smi) if smi is not None else None
            if m is None:
                continue                                   # invalid -> floor (-10)
            if "." in smi:
                out[i] = self.disconnect                   # disconnected -> flat penalty
                continue
            try:
                out[i] = -min(abs(float(self.prop_fn(m)) - tgt[i]) / self.scale, self.clamp)
            except Exception:
                pass                                       # prop error -> invalid floor
        return out


def head_predict_batch(mols, head, atom_encoder, bond_encoder, device):
    """The HEAD's predicted property for each RDKit Mol via its native re-encoding
    (SMILES -> smiles_to_pyg_data -> to_dense -> head.predict), exactly as LearnedPropertyEnergy
    does (the head is trained on that encoding; the raw graph mispredicts). Returns a list
    aligned to ``mols`` (None where re-encode fails). Batched over the molecules."""
    from torch_geometric.data import Batch
    from defog.domains.molecule import smiles_to_pyg_data
    reenc, idx = [], []
    for i, m in enumerate(mols):
        if m is None:
            continue
        try:
            rd = smiles_to_pyg_data(Chem.MolToSmiles(m), atom_encoder, bond_encoder)
        except Exception:
            rd = None
        if rd is not None and getattr(rd, "x", None) is not None:
            reenc.append(rd); idx.append(i)
    out = [None] * len(mols)
    if reenc:
        batch = Batch.from_data_list(reenc).to(device)
        dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        dense = dense.mask(mask)
        preds = head.predict(dense.X, dense.E, mask).reshape(-1).tolist()
        for j, i in enumerate(idx):
            out[i] = float(preds[j])
    return out


class HeadPropertyMatchReward:
    """Connectivity-FIRST conditional reward with a learned HEAD as the property term (instead
    of RDKit): connected+valid on-target > connected off-target > disconnected > invalid. The
    property term ``-min(|head(mol)-target|/scale, clamp)`` uses ``head.predict`` (batched over
    the connected mols). Same tiering/scale as :class:`PropertyMatchReward`."""

    def __init__(self, head, atom_encoder, bond_encoder, atom_decoder, bond_decoder, device,
                 scale=1.0, invalid_reward=-10.0, disconnect_reward=-4.0, prop_clamp=3.0):
        self.head = head
        self.ae, self.be = atom_encoder, bond_encoder
        self.ad, self.bd = atom_decoder, bond_decoder
        self.device = device
        self.scale, self.invalid = float(scale), float(invalid_reward)
        self.disconnect, self.clamp = float(disconnect_reward), float(prop_clamp)

    def __call__(self, X1, E1, node_mask, cond):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.invalid)
        tgt = cond.reshape(-1).tolist()
        conn_mols, conn_idx = [], []
        for i, d in enumerate(datas):
            mol = pyg_data_to_mol(d, self.ad, self.bd)
            smi = mol_to_smiles(mol) if mol is not None else None
            m = Chem.MolFromSmiles(smi) if smi is not None else None
            if m is None:
                continue                                   # invalid -> floor
            if "." in smi:
                out[i] = self.disconnect; continue         # disconnected -> flat penalty
            conn_mols.append(m); conn_idx.append(i)
        if conn_mols:
            preds = head_predict_batch(conn_mols, self.head, self.ae, self.be, self.device)
            for j, i in enumerate(conn_idx):
                p = preds[j]
                if p is None:
                    continue                               # re-encode failed -> invalid floor
                out[i] = -min(abs(p - tgt[i]) / self.scale, self.clamp)
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


def _disc(n_valid, n_disc):
    return (n_disc / n_valid) if n_valid else None


def steer_eval(base, adapter, atom_decoder, bond_decoder, prop_fn, targets, weights,
               steps, eta, omega, td, n_per, n_base, chunk, device, seed=None, head_fn=None):
    """Per (level,w): achieved-property MAE (RDKit truth) + disconnection% + validity%, and --
    when ``head_fn`` is given -- ALSO the HEAD's MAE (``head_mae``, the selection metric).
    ``seed`` fixes the rng (saved+restored) so pre/post evals sample the SAME draws (paired)."""
    _rng = torch.get_rng_state() if seed is not None else None
    if seed is not None:
        torch.manual_seed(seed)
    bs = Sampler(base, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
    bmols, _bt, bvalid, bdisc = gen_eval(chunked(bs, n_base, chunk, device), atom_decoder, bond_decoder)
    _, bmean, _ = _rdkit_stats(bmols, 0.0, prop_fn)
    out = {"baseline_mean": bmean, "baseline_disc": _disc(bvalid, bdisc), "levels": {}}
    for lvl, tgt in targets.items():
        out["levels"][lvl] = {"target": tgt, "per_w": {}}
        for w in weights:
            comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([tgt]), w)], base=base, mode="product")
            samp = AdaptedSampler(base, comp, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
            mols, tot, nvalid, ndisc = gen_eval(chunked(samp, n_per, chunk, device), atom_decoder, bond_decoder)
            mae, mean, ncon = _rdkit_stats(mols, tgt, prop_fn)
            rec = {"n": ncon, "valid": (nvalid / tot) if tot else None, "disc": _disc(nvalid, ndisc),
                   "mean": mean, "mae": mae}
            if head_fn is not None:
                hp = [v for v in head_fn(mols) if v is not None]
                if hp:
                    ha = np.asarray(hp, dtype=float)
                    rec["head_mae"], rec["head_mean"] = float(np.mean(np.abs(ha - tgt))), float(ha.mean())
                else:
                    rec["head_mae"] = rec["head_mean"] = None
            out["levels"][lvl]["per_w"][str(w)] = rec
    if _rng is not None:
        torch.set_rng_state(_rng)
    return out


def _fmt(ev, weights):
    parts = []
    for lvl, d in ev["levels"].items():
        for w in weights:
            r = d["per_w"][str(w)]
            disc = f"{r['disc']*100:.0f}%" if r["disc"] is not None else "?"
            val = f"{r['valid']*100:.0f}%" if r.get("valid") is not None else "?"
            hm = f" head_mae={r['head_mae']}" if "head_mae" in r else ""
            parts.append(f"{lvl}->{d['target']:.1f}@w{w}: mae={r['mae']}{hm} disc={disc} valid={val}")
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

    # Reward source: RDKit ground-truth OR the learned head. With the head, the reward + the
    # early-stop probe + best-seed selection all use it; EVAL still reports RDKit truth to validate.
    head_fn = None
    if e.REWARD_SOURCE == "head":
        assert e.HEAD_CKPT, "REWARD_SOURCE='head' needs --HEAD_CKPT"
        head = PropertyHead.load(e.HEAD_CKPT, device=device)
        head_fn = lambda mols: head_predict_batch(mols, head, atom_encoder, bond_encoder, device)
        reward = HeadPropertyMatchReward(head, atom_encoder, bond_encoder, atom_decoder, bond_decoder,
                                         device, scale=prop_std, invalid_reward=e.INVALID_REWARD,
                                         disconnect_reward=e.DISCONNECT_REWARD, prop_clamp=e.PROP_CLAMP)
        e.log(f"REWARD_SOURCE=head: PropertyHead from {e.HEAD_CKPT} "
              f"(prop_mean={float(head.prop_mean):.3f} prop_std={float(head.prop_std):.3f}); "
              f"probe/selection use HEAD, eval reports RDKit truth too")
    else:
        reward = PropertyMatchReward(atom_decoder, bond_decoder, prop_fn, scale=prop_std,
                                     invalid_reward=e.INVALID_REWARD, disconnect_reward=e.DISCONNECT_REWARD,
                                     prop_clamp=e.PROP_CLAMP)
    cond_sampler = make_condition_sampler(p_lo, p_hi, e.ROLLOUT_SIZE, e.N_GROUPS, e.SEED)

    def eval_now(tag):
        ev = steer_eval(base, adapter, atom_decoder, bond_decoder, prop_fn, targets, e.GUIDANCE_WEIGHTS,
                        e.EVAL_STEPS, e.ETA, e.OMEGA, e.TIME_DISTORTION, e.N_PER_TARGET, e.N_BASELINE,
                        e.EVAL_CHUNK, device, seed=e.EVAL_SEED, head_fn=head_fn)
        e.log(f"[{tag}] baseline={ev['baseline_mean']}  {_fmt(ev, e.GUIDANCE_WEIGHTS)}")
        return ev

    e.log("=== PRE-RL eval ===")
    pre_ev = eval_now("pre-RL")

    # Snapshot the INPUT adapter weights so we can prove the RL actually moved them
    # (guards against ever again shipping an unchanged adapter as a "result").
    input_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}

    trainer = AdapterGDPOTrainer(
        base, adapter, reward, kl_coef=e.KL_COEF, lr=e.LR, ema_decay=e.EMA_DECAY,
        rollout_size=e.ROLLOUT_SIZE, sample_steps=e.ROLLOUT_STEPS, eta=e.ROLLOUT_ETA,
        omega=e.ROLLOUT_OMEGA, time_distortion=e.TIME_DISTORTION, condition_sampler=cond_sampler,
        subsample_steps=e.SUBSAMPLE_STEPS, minibatch_size=e.MINIBATCH, lambda_edge=e.LAMBDA_EDGE,
        crn=e.CRN,
        grad_clip=e.GRAD_CLIP, seed=e.SEED, device=device,
    )

    e.log(f"=== RL fine-tuning: max_time={e.MAX_TIME_HOURS}h rollout(K={e.ROLLOUT_SIZE},G={e.N_GROUPS},"
          f"per={e.ROLLOUT_SIZE // e.N_GROUPS},crn={e.CRN},eta={e.ROLLOUT_ETA},{e.ROLLOUT_STEPS} steps) "
          f"lr={e.LR} kl={e.KL_COEF} early_stop={e.EARLY_STOP} ===")

    # early-stop bookkeeping: keep the best (validity-gated) EMA snapshot by probe MAE
    w1 = str(e.GUIDANCE_WEIGHTS[0])
    pre_valid_floor = 0.0
    if e.EARLY_STOP:
        vs = [pre_ev["levels"][lvl]["per_w"][w1]["valid"] or 0.0 for lvl in e.LEVEL_NAMES]
        pre_valid_floor = max(0.0, min(vs) - e.VALIDITY_FLOOR_MARGIN)
    best = {"mae": float("inf"), "state": None, "iter": -1}

    def probe_eval():
        """PAIRED conditional eval at the DEPLOYMENT policy (w=1, EVAL_STEPS, EMA weights):
        mean achieved-property MAE via the SELECTION property (the HEAD if REWARD_SOURCE='head',
        else RDKit) over connected mols + min validity across levels. Fixed rng (saved+restored)
        so the probe reflects the adapter, not sampling noise. Full EVAL_STEPS so early-stop
        selects a snapshot that is actually best at deployment."""
        _rng = torch.get_rng_state(); torch.manual_seed(e.EVAL_SEED)
        maes, valids = [], []
        for lvl, tgt in targets.items():
            comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([tgt]), 1.0)], base=base, mode="product")
            samp = AdaptedSampler(base, comp, eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS, time_distortion=e.TIME_DISTORTION)
            mols, tot, nvalid, _ = gen_eval(chunked(samp, e.PROBE_N_PER, e.EVAL_CHUNK, device), atom_decoder, bond_decoder)
            valids.append((nvalid / tot) if tot else 0.0)
            if mols:
                if head_fn is not None:
                    vals = [v for v in head_fn(mols) if v is not None]
                    if vals:
                        maes.append(float(np.mean(np.abs(np.asarray(vals, dtype=float) - tgt))))
                else:
                    mae_l, _, _ = _rdkit_stats(mols, tgt, prop_fn)
                    if mae_l is not None:
                        maes.append(mae_l)
        torch.set_rng_state(_rng)
        return (float(np.mean(maes)) if maes else float("inf")), (min(valids) if valids else 0.0)

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
        if e.EARLY_STOP and e.PROBE_EVERY and it > 0 and it % e.PROBE_EVERY == 0:
            backup = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
            if trainer.ema is not None:
                trainer.ema.copy_to(adapter)
            mae, valid = probe_eval()
            adapter.load_state_dict(backup)              # restore live training weights
            improved = (valid >= pre_valid_floor) and (mae < best["mae"])
            if improved:
                src = trainer.ema.state_dict() if trainer.ema is not None else adapter.state_dict()
                best = {"mae": mae, "iter": it, "state": {k: v.detach().clone() for k, v in src.items()}}
            e.track("probe_mae", float(mae)); e.track("probe_valid", float(valid))
            e.log(f"[probe iter{it}] mae={mae:.3f} valid={valid:.1%} floor={pre_valid_floor:.1%} "
                  f"best={best['mae']:.3f}@{best['iter']} {'*NEW*' if improved else ''}")
        it += 1
    e.log(f"RL done: {it} iterations in {(time.time()-t0)/60:.1f} min")
    e.commit_json("rl_history.json", history)

    # Deployment weights: the best early-stop snapshot (validity-gated probe minimum),
    # else the final EMA. Load into the live adapter, then save + eval THOSE weights.
    if e.EARLY_STOP and best["state"] is not None:
        adapter.load_state_dict(best["state"])
        e.log(f"early-stop: deploying best snapshot from iter {best['iter']} (probe mae={best['mae']:.3f})")
    elif trainer.ema is not None:
        trainer.ema.copy_to(adapter)
    # Sanity: how far did the deployed adapter actually move from the input? A tiny diff
    # means the RL never learned (LR too low / EMA lag) and any pre->post MAE change is
    # just eval noise -- surface it loudly instead of silently shipping a no-op.
    _dep = adapter.state_dict()
    wdiff = max(float((_dep[k] - input_state[k]).abs().max())
                for k in input_state if _dep[k].dtype.is_floating_point)
    e["results/deployed_weight_diff"] = wdiff
    e.log(f"deployed adapter max|Δweight| vs input = {wdiff:.3e}"
          + ("   ⚠️ NEAR-ZERO: RL did not learn (pre->post is noise)" if wdiff < 1e-4 else ""))
    ckpt = adapter.save(os.path.join(e.path, f"{e.PROPERTY}_adapter_rl"))
    e.log(f"Saved RL'd adapter -> {ckpt}")

    e.log("=== POST-RL eval ===")
    post_ev = eval_now("post-RL")

    # -- compare + plot MAE pre vs post --------------------------------------
    summary = {"property": e.PROPERTY, "kl_coef": e.KL_COEF, "iterations": it,
               "reward_source": e.REWARD_SOURCE, "crn": e.CRN, "rollout_eta": e.ROLLOUT_ETA,
               "rollouts_per_group": e.ROLLOUT_SIZE // e.N_GROUPS,
               "early_stop_best_iter": (best["iter"] if e.EARLY_STOP else None),
               "deployed_weight_diff": wdiff, "lr": e.LR, "ema_decay": e.EMA_DECAY,
               "targets": targets, "pre": pre_ev, "post": post_ev, "mae_delta": {}}
    summary["disc_delta"] = {}
    ds = lambda x: f"{x*100:.0f}%" if x is not None else "?"
    e.log("=" * 60)
    e.log(f"pre -> post   (RDKit-MAE = TRUTH; head-MAE = selection metric; reward={e.REWARD_SOURCE})")
    for lvl in e.LEVEL_NAMES:
        for w in e.GUIDANCE_WEIGHTS:
            pr, po = pre_ev["levels"][lvl]["per_w"][str(w)], post_ev["levels"][lvl]["per_w"][str(w)]
            pre, post = pr["mae"], po["mae"]
            d = (post - pre) if (pre is not None and post is not None) else None
            hpre, hpost = pr.get("head_mae"), po.get("head_mae")
            hd = (hpost - hpre) if (hpre is not None and hpost is not None) else None
            summary["mae_delta"][f"{lvl}_w{w}"] = {"pre": pre, "post": post, "delta": d,
                                                   "head_pre": hpre, "head_post": hpost, "head_delta": hd,
                                                   "valid_pre": pr.get("valid"), "valid_post": po.get("valid")}
            summary["disc_delta"][f"{lvl}_w{w}"] = {"pre": pr["disc"], "post": po["disc"]}
            hstr = f" | HEAD-MAE {hpre} -> {hpost} (Δ{hd})" if hpre is not None else ""
            e.log(f"{lvl} w={w}: RDKit-MAE {pre} -> {post} (Δ{d}){hstr} | "
                  f"disc {ds(pr['disc'])}->{ds(po['disc'])} | valid {ds(pr.get('valid'))}->{ds(po.get('valid'))}")
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
    e.GUIDANCE_WEIGHTS = [1.0]
    e.REF_SUBSAMPLE = 300
    e.PROBE_EVERY = 1
    e.PROBE_STEPS = 5
    e.PROBE_N_PER = 8
    e.EARLY_STOP = True
    e.CRN = True
    e.LOG_EVERY = 1
    e.BASE_CKPT = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")


experiment.run_if_main()
