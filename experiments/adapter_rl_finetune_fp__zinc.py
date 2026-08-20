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
# :param VOCABULARY: Which frozen base this adapter binds to. Bundles atom order,
#     bond set, kekulize flag and SMILES source -- the four must agree or the
#     conditions describe different molecules than the graphs do, silently.
#     ATOM_TYPES/BOND_TYPES are deliberately NOT parameters: a caller who could
#     set them independently could contradict the base with nothing to notice.
VOCABULARY: str = "e1_kekulized"
BASE_CKPT: str = os.path.join(_PROJECT_DIR, "ckpts", "zinc_rl2_seed42", "best_model")
ADAPTER_CKPT: str = ""            # trained fingerprint adapter to RL-finetune
FP_BITS: int = 1024
FP_RADIUS: int = 2
# :param FP_COUNTS/FP_FROM: MUST match how ADAPTER_CKPT was trained. The shipped
#     fingerprint@3.0.0 is counts/decoded at 1024 bits. Conditioning an adapter on
#     the encoding it was not trained with raises nothing -- it just steers badly,
#     and here it would also make the REWARD wrong, so RL would optimise noise.
FP_COUNTS: bool = True
FP_FROM: str = "decoded"

# --- RL (GDPO) ---
MAX_TIME_HOURS: float = 6.0
MAX_ITERS: int = 100000
ROLLOUT_SIZE: int = 64
N_GROUPS: int = 8                 # distinct target FPs / iteration
ROLLOUT_STEPS: int = 250
ROLLOUT_ETA: float = 5.0
ROLLOUT_OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
SUBSAMPLE_STEPS: int = 16
MINIBATCH: int = 16
# :param LR/EMA_DECAY: 1e-5 with ema 0.999 is what the earlier adapter-RL rounds
#     used, and those were VOID -- the adapter barely moved, so "RL did not help"
#     was indistinguishable from "RL did not run". 1e-4 with a faster EMA is what
#     made adapter-RL actually learn. WEIGHT_DIFF_EVERY below is the check that
#     keeps that failure detectable rather than silent.
LR: float = 1e-4
KL_COEF: float = 0.2
EMA_DECAY: float = 0.99
GRAD_CLIP: float = 1.0
#: Iterations between "has the adapter actually changed?" reports. 0 disables.
WEIGHT_DIFF_EVERY: int = 20
#: Iterations between checkpoints. A previous GDPO run peaked mid-training and the
#: peak was lost because only the final model was saved.
CKPT_EVERY: int = 50
LAMBDA_EDGE: float = 1.0
# connectivity-FIRST: connected (Tanimoto in [0,1]) > disconnected > invalid
INVALID_REWARD: float = -1.0
DISCONNECT_REWARD: float = -0.5
# :param DISCONNECT_DELTA: None keeps the flat DISCONNECT_REWARD above. A float
#     switches to largest-fragment partial credit minus this delta, which is what
#     lets similarity compete with connectivity at all -- see FPMatchReward.
#     Small delta -> similarity dominates; ~0.5 -> roughly round-1 behaviour.
DISCONNECT_DELTA: float = None
# :param DISC_FLOOR: Disconnection fraction this run must not regress past, as a
#     fraction (0.1049 = round 1's result). An arm that beats it on Tanimoto while
#     fragmenting more is REPORTED but flagged as failing the floor. Fixed before
#     the run on purpose: picking the acceptance rule after seeing the numbers is
#     how a marginal result gets argued into looking good. None disables.
DISC_FLOOR: float = None
N_HOLDOUT: int = 4000             # held-out molecules -> target FP pool (unseen)

# --- Evaluation (pre & post) ---
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
GUIDANCE_WEIGHTS: list = [1.0]  # eval the optimized policy: RL rolls out / scores at w=1 only
N_TARGETS: int = 12               # held-out eval target FPs. 6 was too few: the
                                  # target-set spread on this metric is ~0.008,
                                  # comparable to the effects being measured.
N_PER_TARGET: int = 64
N_BASELINE: int = 128
# :param SIZE_MATCHED: also evaluate with generation pinned to each target's
#     heavy-atom count. A PARALLEL STRAND, never the primary number -- supplying
#     the size hands the model information the fingerprint is meant to convey, so
#     it measures an easier task and is not comparable to free-size figures.
#     Worth its cost: measured +0.032 (RL adapter) to +0.048 (shipped) on the
#     comparison targets, which is larger than any architectural lever tried.
SIZE_MATCHED: bool = True
N_SIZE_BASELINE: int = 128        # unconditional samples drawn AT each target's size
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
    """(FP matrix of CONNECTED+valid mols, n_total, n_valid, n_disconnected, sizes).

    ``sizes`` is the heavy-atom count of each scored molecule. It exists for the
    size-inference diagnostic: bit count correlates with molecule size, so a target's
    size is information the fingerprint already carries, and whether the adapter
    USES it turned out to separate the shipped adapter (corr -0.13, ignores it) from
    the RL-tuned one (corr +0.55).
    """
    fps, n_valid, n_disc, sizes = [], 0, 0, []
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
            sizes.append(m.GetNumHeavyAtoms())
        except Exception:
            pass
    mat = np.stack(fps) if fps else np.zeros((0, n_bits), dtype=np.float32)
    return mat, len(samples), n_valid, n_disc, np.array(sizes, dtype=float)


def chunked(sampler, n, chunk, device, num_nodes=None):
    """``num_nodes`` pins every generated graph to that atom count (size-matched strand)."""
    out, rem = [], n
    while rem > 0:
        cur = min(chunk, rem)
        out += sampler.sample(cur, num_nodes=num_nodes, device=device, show_progress=False)
        rem -= cur
    return out


class FPMatchReward:
    """Connectivity-FIRST Tanimoto reward. connected -> Tanimoto(FP(mol), target) in
    [0,1]; disconnected -> disconnect_reward; invalid -> invalid_reward. Since Tanimoto
    is non-negative, ANY connected molecule outranks any disconnected/invalid one.

    The reward IS the evaluation metric -- binary Tanimoto against the target the
    rollout was conditioned on -- so there is no learned proxy here and nothing to
    game. That is the main reason this is a clean experiment: whatever RL finds, it
    found by moving the number we actually report.

    THE TARGET MUST BE BINARISED. ``cond`` is the RAW condition the adapter speaks,
    which for a counts-trained adapter is ``log1p(hashed counts)``. Feeding that
    straight into a Tanimoto would put log-counts in the numerator and denominator
    against a 0/1 generated vector -- a number that is not Tanimoto, is not the eval
    metric, and would have RL optimising something nobody measures. ``cond > 0``
    recovers exactly the set bits (log1p(x) > 0 iff x > 0), so binarising here makes
    the reward identical to the metric under BOTH encodings.
    """

    def __init__(self, atom_decoder, bond_decoder, radius, n_bits,
                 invalid_reward=-1.0, disconnect_reward=-0.5, disconnect_delta=None):
        self.ad, self.bd, self.radius, self.n_bits = atom_decoder, bond_decoder, radius, n_bits
        self.invalid, self.disconnect = float(invalid_reward), float(disconnect_reward)
        # None -> the flat penalty above (round-1 behaviour). A float switches to
        # LARGEST-FRAGMENT PARTIAL CREDIT: score the biggest piece, then subtract
        # delta for being in pieces.
        #
        # Round 1 spent its entire budget on connectivity and none on similarity,
        # and the flat penalty is why: a disconnected sample scored -0.5 against a
        # connected sample's ~0.30, so repairing one fragment paid ~0.80 while
        # improving an already-connected molecule's similarity paid ~0.05. Sixteen
        # times the return, for an easier edit. RL is not obliged to care which of
        # those we wanted.
        #
        # delta restores a usable ratio without ever inverting the ordering: the
        # same molecule always scores higher connected than in pieces, so nothing
        # here rewards fragmenting. A flat constant near the mean Tanimoto WOULD
        # invert -- it would make a 0.20-similarity connected molecule score below
        # a fragment -- which is why the credit is relative rather than absolute.
        self.delta = None if disconnect_delta is None else float(disconnect_delta)

    def __call__(self, X1, E1, node_mask, cond):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.invalid)
        # (K, n_bits) -> set bits. Correct for binary and counts conditioning alike.
        tgt = (cond.detach().cpu().float() > 0).numpy().astype(np.float32)
        for i, d in enumerate(datas):
            mol = pyg_data_to_mol(d, self.ad, self.bd)
            smi = mol_to_smiles(mol) if mol is not None else None
            m = Chem.MolFromSmiles(smi) if smi is not None else None
            if m is None:
                continue                                   # invalid -> floor
            scored, penalty = m, 0.0
            if "." in smi:
                if self.delta is None:
                    out[i] = self.disconnect               # flat penalty (round 1)
                    continue
                try:
                    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
                except Exception:                          # noqa: BLE001
                    continue                               # unreadable -> invalid floor
                if not frags:
                    continue
                scored = max(frags, key=lambda f: f.GetNumHeavyAtoms())
                penalty = self.delta
            try:
                fp = morgan_bits(scored, self.radius, self.n_bits)   # BINARY, = the metric
                t = tgt[i]
                inter = float(fp @ t); union = float(fp.sum() + t.sum() - inter)
                out[i] = ((inter / union) if union > 0 else 0.0) - penalty
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


def fp_eval(base, adapter, ad, bd, radius, n_bits, cond_fps, metric_fps, weights,
            steps, eta, omega, td, n_per, n_base, chunk, device, baseline_fp=None,
            target_sizes=None, size_matched=False, n_size_base=128):
    """Per w: mean Tanimoto LIFT over baseline + mean disconnection%, over the eval targets.

    ``cond_fps`` is what the adapter is CONDITIONED on (its trained encoding);
    ``metric_fps`` is what generations are SCORED against (always binary). Splitting
    the two is what keeps a counts-conditioned adapter's numbers comparable to a
    binary-conditioned one's.

    ``baseline_fp`` lets the caller reuse one unconditional sample across the pre- and
    post-RL evaluations. The baseline depends only on the FROZEN base, so re-drawing it
    would add sampling noise to the pre-vs-post difference -- the one quantity the whole
    run exists to measure -- for nothing.
    """
    if baseline_fp is None:
        bs = Sampler(base, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
        baseline_fp = decode_fp_disc(chunked(bs, n_base, chunk, device), ad, bd, radius, n_bits)
    bfp, bt, bvalid, bdisc, _bsizes = baseline_fp
    base_tan = [float(tanimoto(bfp, t).mean()) if bfp.shape[0] else np.nan for t in metric_fps]
    out = {"baseline_tan": float(np.nanmean(base_tan)),
           "baseline_disc": (bdisc / bvalid) if bvalid else None, "per_w": {}}
    for w in weights:
        lifts, discs, gtans, per_target = [], [], [], []
        sm_lifts, sm_tans = [], []
        for k, (cfp, mfp) in enumerate(zip(cond_fps, metric_fps)):
            comp = AdapterComposition([ConditionBranch(adapter, torch.as_tensor(cfp, dtype=torch.float32), w)],
                                      base=base, mode="product")
            samp = AdaptedSampler(base, comp, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
            gfp, gt, gvalid, gdisc, gsizes = decode_fp_disc(
                chunked(samp, n_per, chunk, device), ad, bd, radius, n_bits)
            gtan = float(tanimoto(gfp, mfp).mean()) if gfp.shape[0] else np.nan
            gtans.append(gtan); lifts.append(gtan - base_tan[k])
            discs.append((gdisc / gvalid) if gvalid else np.nan)
            rec = {"target": k, "tan": gtan, "baseline": base_tan[k],
                   "lift": gtan - base_tan[k],
                   "validity": gvalid / gt if gt else 0.0,
                   "gen_size": float(gsizes.mean()) if gsizes.size else float("nan"),
                   "target_size": (float(target_sizes[k]) if target_sizes is not None
                                   else float("nan"))}
            # --- SIZE-MATCHED STRAND -------------------------------------------
            # Reported alongside, never instead of, the free-size figure: pinning the
            # atom count supplies information the fingerprint is meant to carry, so it
            # measures an easier task. Its own baseline is re-drawn AT THAT SIZE --
            # a size-matched numerator over a free-size denominator would book the
            # size effect as steering, which is the confound this isolates.
            if size_matched and target_sizes is not None:
                n_at = int(round(float(target_sizes[k])))
                sfp, st_, svalid, sdisc, _ = decode_fp_disc(
                    chunked(samp, n_per, chunk, device, num_nodes=n_at), ad, bd, radius, n_bits)
                sbfp, _, _, _, _ = decode_fp_disc(
                    chunked(Sampler(base, eta=eta, omega=omega, sample_steps=steps,
                                    time_distortion=td), n_size_base, chunk, device,
                            num_nodes=n_at), ad, bd, radius, n_bits)
                s_t = float(tanimoto(sfp, mfp).mean()) if sfp.shape[0] else np.nan
                s_b = float(tanimoto(sbfp, mfp).mean()) if sbfp.shape[0] else np.nan
                rec.update({"sm_tan": s_t, "sm_baseline": s_b, "sm_lift": s_t - s_b,
                            "sm_validity": svalid / st_ if st_ else 0.0,
                            "sm_disc": (sdisc / svalid) if svalid else np.nan})
                sm_tans.append(s_t); sm_lifts.append(s_t - s_b)
            per_target.append(rec)
        entry = {"tan": float(np.nanmean(gtans)), "lift": float(np.nanmean(lifts)),
                 "disc": float(np.nanmean(discs)), "per_target": per_target}
        if sm_lifts:
            entry["sm_tan"] = float(np.nanmean(sm_tans))
            entry["sm_lift"] = float(np.nanmean(sm_lifts))
        # Whether the model INFERS size from the fingerprint. Mean-vs-mean would not
        # show this: a model that always emits ~23 atoms matches the average target
        # while missing every individual one. Measured -0.13 for the shipped adapter
        # (ignores size) against +0.55 for the RL-tuned one.
        if target_sizes is not None:
            ts = np.array([r["target_size"] for r in per_target], dtype=float)
            gs = np.array([r["gen_size"] for r in per_target], dtype=float)
            ok = np.isfinite(ts) & np.isfinite(gs)
            entry["corr_size"] = (float(np.corrcoef(ts[ok], gs[ok])[0, 1])
                                  if ok.sum() > 1 else float("nan"))
        out["per_w"][str(w)] = entry
    return out, baseline_fp


# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"RL-finetune FINGERPRINT adapter (GDPO, connectivity-first) kl_coef={e.KL_COEF}")
    import pytorch_lightning as pl
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Vocabulary + condition encoding come from the SAME definitions the adapter was
    # trained with, loaded from those modules rather than restated here. A second
    # copy of either would be a place for them to drift, and a drifted condition
    # encoding makes both the rollouts and the reward quietly wrong.
    import sys
    import importlib.util as _ilu

    def _load(mod_name, filename):
        spec = _ilu.spec_from_file_location(
            mod_name, os.path.join(_PROJECT_DIR, "experiments", filename))
        m = _ilu.module_from_spec(spec)
        sys.modules[mod_name] = m           # pycomex reads annotations off the module
        spec.loader.exec_module(m)
        return m

    _voc = _load("_advoc_rl", "adapter_training__zinc.py")
    _fpx = _load("_fpexp_rl", "adapter_fingerprint__zinc.py")
    atom_types, bond_types, kekulize, source = _voc._vocabulary(e.VOCABULARY)
    mol_morgan_bits = _fpx.mol_morgan_bits          # THE conditioning encoding
    e.log(f"vocabulary '{e.VOCABULARY}': {len(atom_types)} atoms {atom_types}")
    e.log(f"  bonds={bond_types} kekulize={kekulize} source={source}")
    e.log(f"  condition: {e.FP_BITS} bits r{e.FP_RADIUS} counts={e.FP_COUNTS} from={e.FP_FROM}")
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, bond_types)

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0

    # Decoding with the wrong vocabulary yields plausible molecules made of the wrong
    # elements, with nothing in the loop to object -- and every reward would be
    # computed on them.
    from defog.data import vocabulary as _vocab
    e.log(_vocab.check_model(base, atom_types, bond_types, what=f"base {e.BASE_CKPT}"))
    if e.ADAPTER_CKPT:
        adapter = AdaLNAdapter.load(e.ADAPTER_CKPT, device=device)
        adapter.check_compatible(base)
        # FP_BITS drives the condition pool, the reward and the metric. If it
        # disagrees with the adapter, the failure is a shape error thrown deep inside
        # the first sampling call, long after the dataset and base have been loaded --
        # and on a cluster that is an hour of a node to learn a one-line mistake.
        if int(adapter.cond_dim) != int(e.FP_BITS):
            raise ValueError(
                f"FP_BITS={e.FP_BITS} but {e.ADAPTER_CKPT} has cond_dim="
                f"{adapter.cond_dim}. The checkpoint is the authority: set "
                f"--FP_BITS {adapter.cond_dim} (and make sure FP_COUNTS matches how "
                f"it was trained -- that one cannot be checked automatically, and "
                f"getting it wrong makes both the rollouts and the reward wrong).")
        e.log(f"loaded FP adapter from {e.ADAPTER_CKPT} (cond_dim={adapter.cond_dim} "
              f"hidden={adapter.hidden} interior_ff={adapter.interior_ff} "
              f"interior_attn={adapter.interior_attn} "
              f"encoder={type(adapter.cond_encoder).__name__ if adapter.cond_encoder else 'none'})")
    else:
        cm = np.zeros(e.FP_BITS, dtype=np.float32); cs = np.ones(e.FP_BITS, dtype=np.float32)
        adapter = AdaLNAdapter.for_base(base, cond_dim=e.FP_BITS, hidden=32, cond_mean=cm, cond_std=cs,
                                        name="fp_adapter", cond_type=f"morgan{e.FP_BITS}").to(device)
        e.log("[fresh/untrained] adapter (smoke only)")

    # Held-out target pool. Conditions must describe what the model can actually
    # BUILD, so with FP_FROM="decoded" each molecule is round-tripped through the
    # graph encoding first -- the charges and any structure the vocabulary drops are
    # gone from the graph, and conditioning on the source molecule's fingerprint
    # would set a target the generator cannot reach even in principle.
    if source == "reference_split":
        from defog.data import zinc_reference as _zref
        all_smiles = list(_zref.load_reference_split().train_smiles)
    else:
        all_smiles = pd.read_csv(e.CSV_PATH)[e.SMILES_COLUMN].tolist()
    rng = np.random.default_rng(e.SEED + 1)
    pick = rng.permutation(len(all_smiles))[:min(e.N_HOLDOUT * 3, len(all_smiles))]

    from experiments.utils import smiles_to_pyg_data
    cond_pool, metric_pool, size_pool, n_dropped = [], [], [], 0
    for j in pick:
        smi = all_smiles[int(j)]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder, kekulize=kekulize)
        dec = pyg_data_to_mol(data, atom_decoder, bond_decoder) if data is not None else None
        back = mol_to_smiles(dec) if dec is not None else None
        dec = Chem.MolFromSmiles(back) if back else None
        if dec is None:
            n_dropped += 1
            continue                      # cannot describe the graph -> cannot label it
        src = dec if e.FP_FROM == "decoded" else m
        cond_pool.append(mol_morgan_bits(src, e.FP_RADIUS, e.FP_BITS, counts=e.FP_COUNTS))
        metric_pool.append(morgan_bits(dec, e.FP_RADIUS, e.FP_BITS))   # binary, = metric
        # size of the DECODED molecule -- what the model would actually have to build
        size_pool.append(dec.GetNumHeavyAtoms())
        if len(cond_pool) >= e.N_HOLDOUT:
            break
    fp_pool = np.stack(cond_pool)
    fp_pool_t = torch.as_tensor(fp_pool, dtype=torch.float32)
    e.log(f"target pool: {fp_pool.shape[0]} molecules ({e.FP_BITS}-bit r{e.FP_RADIUS} "
          f"counts={e.FP_COUNTS}); {n_dropped} dropped as undescribable")
    e.log(f"  condition nonzero/molecule ~{float((fp_pool > 0).sum(1).mean()):.1f}, "
          f"max value {float(fp_pool.max()):.3f}")

    # Eval conditions on the trained encoding but SCORES against the binary metric,
    # exactly as the training experiment and the shipped package do.
    eval_targets = fp_pool[:e.N_TARGETS]
    eval_metric_targets = np.stack(metric_pool[:e.N_TARGETS])
    eval_target_sizes = np.array(size_pool[:e.N_TARGETS], dtype=float)
    e.log(f"  eval target sizes: {eval_target_sizes.astype(int).tolist()}")

    reward = FPMatchReward(atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS,
                           invalid_reward=e.INVALID_REWARD, disconnect_reward=e.DISCONNECT_REWARD,
                           disconnect_delta=e.DISCONNECT_DELTA)
    e.log("reward: " + ("connected -> Tanimoto; disconnected -> "
                        f"Tanimoto(largest fragment) - {e.DISCONNECT_DELTA}; "
                        f"invalid -> {e.INVALID_REWARD}"
                        if e.DISCONNECT_DELTA is not None else
                        f"connected -> Tanimoto; disconnected -> {e.DISCONNECT_REWARD}; "
                        f"invalid -> {e.INVALID_REWARD}  [flat, connectivity-first]"))
    cond_sampler = make_fp_condition_sampler(fp_pool_t, e.ROLLOUT_SIZE, e.N_GROUPS, e.SEED)

    def _fmt(ev):
        parts = [f"baseline<T>={ev['baseline_tan']:.3f} disc={ev['baseline_disc']}"]
        for w in e.GUIDANCE_WEIGHTS:
            r = ev["per_w"][str(w)]
            seg = f"w{w}: <T>={r['tan']:.3f} lift={r['lift']:+.3f} disc={r['disc']*100:.0f}%"
            if "sm_lift" in r:
                seg += f" || size-matched lift={r['sm_lift']:+.3f}"
            if "corr_size" in r:
                seg += f" corr(size)={r['corr_size']:+.2f}"
            parts.append(seg)
        return " | ".join(parts)

    _baseline_cache = {"fp": None}

    def eval_now(tag):
        ev, bfp = fp_eval(base, adapter, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS,
                          eval_targets, eval_metric_targets,
                          e.GUIDANCE_WEIGHTS, e.EVAL_STEPS, e.ETA, e.OMEGA, e.TIME_DISTORTION,
                          e.N_PER_TARGET, e.N_BASELINE, e.EVAL_CHUNK, device,
                          baseline_fp=_baseline_cache["fp"],
                          target_sizes=eval_target_sizes,
                          size_matched=e.SIZE_MATCHED,
                          n_size_base=e.N_SIZE_BASELINE)
        _baseline_cache["fp"] = bfp     # frozen base -> same baseline pre and post
        e.log(f"[{tag}] {_fmt(ev)}")
        return ev

    e.log("=== PRE-RL eval ===")
    pre_ev = eval_now("pre-RL")

    # Snapshot for the weight-diff sanity check. The failure this catches is the one
    # that voided the earlier adapter-RL rounds: the loop runs, rewards get logged,
    # and the adapter never meaningfully moves -- so "RL did not help" and "RL did not
    # happen" produce identical output.
    _w0 = {k: v.detach().clone() for k, v in adapter.state_dict().items()
           if v.dtype.is_floating_point}

    def weight_drift():
        cur = adapter.state_dict()
        num = sum(float((cur[k] - v).pow(2).sum()) for k, v in _w0.items())
        den = sum(float(v.pow(2).sum()) for v in _w0.values())
        return (num / den) ** 0.5 if den > 0 else float("nan")

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
    best = {"reward": -float("inf"), "iter": -1}
    while it < e.MAX_ITERS and time.time() < deadline:
        m = trainer.step()
        history.append(m)
        if it % e.LOG_EVERY == 0:
            e.log(f"[iter {it}] reward={m['reward_mean']:+.3f}(min {m['reward_min']:+.2f}) "
                  f"kl={m['kl']:.4f} adv_std={m['adv_std']:.2f} grad={m['grad_norm']:.2f}")
        if e.WEIGHT_DIFF_EVERY and it > 0 and it % e.WEIGHT_DIFF_EVERY == 0:
            drift = weight_drift()
            m["weight_drift"] = drift
            e.log(f"[iter {it}] relative weight drift from init: {drift:.2e}"
                  + ("   <-- adapter is barely moving; treat any null result as VOID"
                     if drift < 1e-4 else ""))
        if e.CKPT_EVERY and it > 0 and it % e.CKPT_EVERY == 0:
            # Reward is the true metric here, so the running reward is a usable
            # selection signal -- keep the best-so-far rather than trusting the last.
            recent = float(np.mean([h["reward_mean"] for h in history[-e.CKPT_EVERY:]]))
            if recent > best["reward"]:
                best["reward"], best["iter"] = recent, it
                trainer.save(os.path.join(e.path, "fp_adapter_rl_best"))
                e.log(f"[iter {it}] new best (mean reward {recent:+.4f}) -> fp_adapter_rl_best")
        it += 1
    drift = weight_drift()
    e.log(f"RL done: {it} iterations in {(time.time()-t0)/60:.1f} min; "
          f"final relative weight drift {drift:.2e}"
          + (f"; best mean reward {best['reward']:+.4f} @ iter {best['iter']}"
             if best["iter"] >= 0 else ""))
    e["rl/weight_drift"] = drift
    e["rl/iterations"] = it
    if drift < 1e-4:
        e.log("WARNING: the adapter barely moved. A flat pre-vs-post result here says "
              "nothing about whether Tanimoto RL works -- it says this run did not "
              "train. Raise LR or lower KL_COEF and rerun before drawing conclusions.")
    e.commit_json("rl_history.json", history)

    ckpt = trainer.save(os.path.join(e.path, "fp_adapter_rl"))
    e.log(f"Saved RL'd adapter -> {ckpt}")
    if trainer.ema is not None:
        trainer.ema.copy_to(adapter)

    e.log("=== POST-RL eval ===")
    post_ev = eval_now("post-RL")

    # Every knob that defines this arm goes in the record. Downstream tooling
    # identifies arms from this file, and a field that is not written is an arm that
    # cannot be found later -- which is exactly what happened to round 2, whose
    # results carry no disconnect_delta and had to be identified from log filenames.
    summary = {"kl_coef": e.KL_COEF, "iterations": it, "weight_drift": drift,
               "vocabulary": e.VOCABULARY, "fp_bits": e.FP_BITS, "fp_counts": e.FP_COUNTS,
               "lr": e.LR, "disconnect_delta": e.DISCONNECT_DELTA,
               "disconnect_reward": e.DISCONNECT_REWARD, "disc_floor": e.DISC_FLOOR,
               "adapter_ckpt": e.ADAPTER_CKPT, "size_matched": e.SIZE_MATCHED,
               "best": best, "pre": pre_ev, "post": post_ev}
    e.log("=" * 60)
    e.log("TANIMOTO LIFT (higher=tighter)  +  DISCONNECTION% (lower=better) — pre -> post")
    for w in e.GUIDANCE_WEIGHTS:
        pr, po = pre_ev["per_w"][str(w)], post_ev["per_w"][str(w)]
        e.log(f"w={w}: lift {pr['lift']:+.3f} -> {po['lift']:+.3f} | "
              f"disc {pr['disc']*100:.0f}% -> {po['disc']*100:.0f}% | <T> {pr['tan']:.3f} -> {po['tan']:.3f}")
        # PAIRED read: same targets, same cached baseline, so the per-target delta is
        # the honest unit. How many targets improved matters more than the mean, which
        # one large mover can carry.
        deltas = [b["lift"] - a["lift"] for a, b in zip(pr["per_target"], po["per_target"])
                  if np.isfinite(a["lift"]) and np.isfinite(b["lift"])]
        if deltas:
            up = sum(d > 0 for d in deltas)
            e.log(f"       paired: {up}/{len(deltas)} targets improved, "
                  f"mean delta {np.mean(deltas):+.4f}, median {np.median(deltas):+.4f}")
            e.log(f"       NOTE target-set spread on this metric is ~0.008 and run-to-run "
                  f"noise ~0.012; a mean delta below that is not a result.")
        if "sm_lift" in pr and "sm_lift" in po:
            e.log(f"       SIZE-MATCHED strand: lift {pr['sm_lift']:+.4f} -> {po['sm_lift']:+.4f} "
                  f"(delta {po['sm_lift'] - pr['sm_lift']:+.4f})")
            e.log(f"       free-size vs size-matched gap: pre {pr['sm_lift'] - pr['lift']:+.4f}, "
                  f"post {po['sm_lift'] - po['lift']:+.4f}  <- what size mismatch is costing")
        if "corr_size" in pr and "corr_size" in po:
            e.log(f"       corr(target size, generated size): {pr['corr_size']:+.3f} -> "
                  f"{po['corr_size']:+.3f}  <- is the model inferring size from the fingerprint")
        if e.DISC_FLOOR is not None:
            ok = po["disc"] <= e.DISC_FLOOR + 1e-9
            summary.setdefault("floor", {})[str(w)] = {
                "disc_floor": e.DISC_FLOOR, "disc_post": po["disc"], "passes": bool(ok)}
            e.log(f"       connectivity floor {e.DISC_FLOOR:.4f}: post disc {po['disc']:.4f} "
                  + ("PASSES" if ok else
                     "FAILS -- this arm gave back connectivity; excluded from selection "
                     "regardless of its Tanimoto (rule fixed before the run)"))
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
    e.N_SIZE_BASELINE = 8
    e.EVAL_CHUNK = 8
    e.GUIDANCE_WEIGHTS = [1.0]
    e.N_HOLDOUT = 60
    e.LOG_EVERY = 1
    e.WEIGHT_DIFF_EVERY = 1
    e.CKPT_EVERY = 1
    if e.VOCABULARY == "legacy_aromatic" and not os.path.exists(e.BASE_CKPT + ".ckpt"):
        e.BASE_CKPT = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")

    # The reference-split path ignores CSV_PATH, so truncate it too or a smoke run
    # fingerprints the whole 224k training set before doing anything.
    from defog.data import zinc_reference as _zr
    _real = _zr.load_reference_split

    def _small(*a, **kw):
        s0 = _real(*a, **kw)
        return _zr.ZincReferenceSplit(
            train_smiles=s0.train_smiles[:300], val_smiles=s0.val_smiles[:50],
            test_smiles=s0.test_smiles[:50],
            provenance={**s0.provenance, "TRUNCATED_FOR_SMOKE_TEST": True})

    _zr.load_reference_split = _small


experiment.run_if_main()
