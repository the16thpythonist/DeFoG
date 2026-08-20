#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --job-name=xafo_train
#SBATCH --output=xafo_train_%j.out

# Stage 1: train a ZINC-kek clogP adapter with BOTH new mechanisms.
#
#   COND_FOURIER=3   Fourier bands on the target. Without them the trunk reads the
#                    property as ONE raw float while the flow-time gets a 64-dim
#                    sinusoidal embedding -- the spectral-bias setup (Tancik et al.,
#                    arXiv:2006.10739) that makes an MLP learn the smooth part of
#                    c -> modulation long before the part that separates 3.5 from 4.2.
#                    n=3 is a measured ceiling, not a guess: see the cosine table in
#                    AdaLNAdapter. Higher banks decorrelate neighbouring targets and
#                    destroy interpolation to targets never trained on.
#
#   XATTN_TOKENS=8   Node -> condition cross-attention. FiLM applies ONE diagonal
#                    affine map to every atom; this lets each atom's own representation
#                    select which condition tokens it reads. Nodes only -- edges and the
#                    global vector keep the FiLM path (RESEARCH.md §2.2 is about the
#                    per-atom action, and edge attention is n^2 x tokens at 500 steps).
#
# THIS IS ONE ARM WITH BOTH CHANGES, by choice. It answers "does the combination beat
# the shipped 0.52" and NOT "which of the two did it". If it wins, the attribution
# ablation (+FF alone, +XA alone) is the immediate follow-up; if it loses, note that one
# change could be helping while the other hurts and the arm cannot tell them apart.
#
# MATCHED TO THE SHIPPED RECIPE so the comparison against molsmith/clogp@1.2.0 means
# something: same base, property, vocabulary, label convention, width, LR, epochs.
#   * EPOCHS. 20 is the matched control; this job trains 40 and checkpoints every 10, so
#     ep20 is the like-for-like comparison and ep30/ep40 say whether it was still improving.
#   * BASE. ckpts/zinc_kek_base is what molsmith/zinc-kek serves -- token
#     -94.15126384728774, matching the shipped adapter. ckpts/zinc_e1_seed42_kek is a
#     DIFFERENT base (-94.09057337861941).
#   * LR=4e-4, NOT the module default 2e-4. docs/zinc_kek_shipping.md has TWO
#     "(shipped)" markers and they disagree: line 66 marks lr2h256 for the v1.0
#     aromatic-label grid, line 124 marks lr4h256 for the v1.1 DECODED-label grid, and
#     line 174 pins it at 4e-4. clogp@1.2.0 is the v1.1 lineage (same weights_hash as
#     1.1.0, plus a head), so 4e-4 is the matching arm. Taking the first marker is the
#     trap: a +2.05M-parameter adapter at HALF the shipped LR for the same number of
#     epochs is the least favourable setting in which to show a mechanism works.
#   * PROPERTY_FROM=decoded is NOT the module default ("source"). The shipped adapter
#     used decoded (run_zinc_clogp_adapter_v11_jupiter.sh:54); source gives cond_mean/std
#     2.458506/1.431754 instead of 2.825129/1.158113, so the adapter would read every
#     target as a different value and the eval script would refuse it.
#
# The added capacity is real (~2.0M params on top of 2.75M, mostly the 9 per-layer
# cross-attention blocks), so a null result at matched epochs is ambiguous between "the
# mechanism does not help" and "it did not finish converging". CKPT_EVERY_K=5 exists so
# that question can be answered by evaluating epoch 10/15/20 rather than argued about.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
BASE="ckpts/zinc_kek_base"
PROPERTY="clogp"
PROPERTY_FROM="decoded"
VOCAB="e1_kekulized"
# 20 epochs is the MATCHED control (the shipped recipe); 40 with a checkpoint every 10
# gives that comparison AND the convergence answer in one job. Measured on a near-identical
# arm on this cluster (adapter_improvements/capacity_results/train_A_base.log): ~10 min per
# epoch, so 40 epochs is ~7 h of a 16 h wall. Without this, a null result at +75%
# parameters has two explanations and no way to separate them.
EPOCHS=40
CKPT_EVERY_K=10
EVAL_ETA=25          # module default is 5.0; the shipped run used 25
HIDDEN=256
LR=4e-4
COND_FOURIER=3
XATTN_TOKENS=8
XATTN_DIM=128
XATTN_HEADS=8
WANT_TOKEN="-94.15126384728774"
# The shipped clogp@1.2.0 normalisation. The adapter must land on these or it is
# conditioned on a different scale than the thing it is being compared to.
WANT_MEAN="2.825129270553589"
WANT_STD="1.1581127643585205"

echo "xattn+fourier clogP adapter @ $(date) on $(hostname)"
echo "base=${BASE} property=${PROPERTY} from=${PROPERTY_FROM} vocab=${VOCAB}"
echo "fourier=${COND_FOURIER} xattn_tokens=${XATTN_TOKENS} dim=${XATTN_DIM} heads=${XATTN_HEADS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

# ---- preflight: right base, and the new code is actually present here --------
# "Verify remote, not local": this script may be running against a checkout that predates
# the feature. pycomex registers one argparse argument per module global and hard-exits
# with status 2 on an unknown --FLAG, so an old checkout CRASHES rather than quietly
# training a plain adapter -- the preflight is not what stands between us and a
# mislabelled run. It earns its place for the other checks: base identity, and that the
# two mechanisms are exact no-ops at init on THIS base.
$PY - <<PY || { echo "ERROR: preflight failed -- refusing to train"; exit 1; }
import inspect, sys
sys.path.insert(0, ".")
from defog.core import DeFoGModel, AdaLNAdapter
from defog.core.adapter import _base_token

sig = inspect.signature(AdaLNAdapter.__init__).parameters
missing = [k for k in ("cond_fourier", "xattn_tokens", "xattn_dim", "xattn_heads")
           if k not in sig]
if missing:
    print(f"FAIL: this checkout's AdaLNAdapter has no {missing} -- the flags would be "
          f"ignored and a plain adapter would be trained and mislabelled.")
    sys.exit(1)

import experiments.adapter_training__zinc as at
for k in ("COND_FOURIER", "XATTN_TOKENS", "XATTN_DIM", "XATTN_HEADS", "PROPERTY_FROM"):
    if not hasattr(at, k):
        print(f"FAIL: experiment module has no {k}"); sys.exit(1)

b = DeFoGModel.load("${BASE}", device="cpu")
tok = _base_token(b)
want = float("${WANT_TOKEN}")
print(f"base ${BASE}: token={tok!r} n_layers={len(b.model.tf_layers)} cond_dim={b.cond_dim}")
if b.cond_dim != 0:
    print("FAIL: base is not unconditional"); sys.exit(1)
if abs(tok - want) > 1e-6 * (1 + abs(want)):
    print(f"FAIL: base token {tok} != {want} (molsmith/zinc-kek). Wrong checkpoint.")
    sys.exit(1)

# and that the two mechanisms are exact no-ops at init on THIS base, which is the
# invariant the product-of-experts composition depends on
a = AdaLNAdapter.for_base(b, cond_dim=1, hidden=${HIDDEN}, cond_fourier=${COND_FOURIER},
                          xattn_tokens=${XATTN_TOKENS}, xattn_dim=${XATTN_DIM},
                          xattn_heads=${XATTN_HEADS})
gate_l1 = sum(float(p.detach().abs().sum()) for lay in a.gate for k in lay for p in lay[k].parameters())
# `a.xattn` exists only when xattn_tokens is truthy, so an ablation arm with
# XATTN_TOKENS=0 would die here instead of training.
out_l1 = (sum(float(m.out.weight.detach().abs().sum() + m.out.bias.detach().abs().sum())
              for m in a.xattn) if ${XATTN_TOKENS} else 0.0)
print(f"adapter: {sum(p.numel() for p in a.parameters()):,} params "
      f"(gate L1 {gate_l1:.1e}, xattn out L1 {out_l1:.1e} -- both must be 0 at init)")
if gate_l1 != 0.0 or out_l1 != 0.0:
    print("FAIL: not an exact no-op at init"); sys.exit(1)
print("preflight OK")
PY

mkdir -p experiments/results/adapter_training__zinc

$PY -u experiments/adapter_training__zinc.py \
    --VOCABULARY "'${VOCAB}'" \
    --PROPERTY "'${PROPERTY}'" \
    --PROPERTY_FROM "'${PROPERTY_FROM}'" \
    --BASE_CKPT "'${BASE}'" \
    --EPOCHS ${EPOCHS} \
    --H_HIDDEN ${HIDDEN} \
    --LEARNING_RATE ${LR} \
    --COND_FOURIER ${COND_FOURIER} \
    --XATTN_TOKENS ${XATTN_TOKENS} \
    --XATTN_DIM ${XATTN_DIM} \
    --XATTN_HEADS ${XATTN_HEADS} \
    --ETA ${EVAL_ETA} \
    --CKPT_EVERY_K ${CKPT_EVERY_K} \
    --MAX_TIME_HOURS 10.0 \
    --PROBE_EVERY_K 10 \
    --GUIDANCE_WEIGHTS "[2.0]" \
    --N_PER_TARGET 32 \
    --N_BASELINE 32 \
    --__DEBUG__ False

rc=$?
echo "training exited ${rc} at $(date)"
[ $rc -eq 0 ] || exit $rc

# ---- confirm what was written actually carries the new architecture ---------
RESULT_DIR=$(ls -td experiments/results/adapter_training__zinc/*/ 2>/dev/null | head -1)
echo "result dir: ${RESULT_DIR:-<none>}"

$PY - "$RESULT_DIR" "${WANT_MEAN}" "${WANT_STD}" "${COND_FOURIER}" "${XATTN_TOKENS}" <<'PY'
import glob, os, sys, torch
sys.path.insert(0, ".")
d, want_mean, want_std = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
# Read from the launcher instead of hardcoded, so the attribution ablation this
# experiment's own analysis prescribes (each mechanism alone) can be run with this script.
want_fourier, want_xattn = int(sys.argv[4]), int(sys.argv[5])
final = os.path.join(d, "clogp_adapter.ckpt")
if not os.path.exists(final):
    print(f"FAIL: {final} missing"); sys.exit(1)
bad = 0
for f in [final] + sorted(glob.glob(os.path.join(d, "clogp_adapter_ep*.ckpt"))):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    cfg, sd = ck["config"], ck["state_dict"]
    m, s = float(sd["cond_mean"][0]), float(sd["cond_std"][0])
    scale_ok = abs(m - want_mean) < 1e-4 and abs(s - want_std) < 1e-4
    arch_ok = (cfg.get("cond_fourier") == want_fourier
               and cfg.get("xattn_tokens") == want_xattn)
    # A trained adapter whose xattn output projections are still exactly zero learned
    # nothing through that path -- the mechanism would be inert and the run would be
    # reporting a FiLM adapter under a cross-attention label.
    xa = sum(float(v.abs().sum()) for k, v in sd.items()
             if k.startswith("xattn.") and ".out." in k)
    live = (xa > 0.0) if want_xattn else True
    bad += not (scale_ok and arch_ok and live)
    print(f"{os.path.basename(f):30s} fourier={cfg.get('cond_fourier')} "
          f"xattn={cfg.get('xattn_tokens')} cond=({m:.6f},{s:.6f}) xattn_out_L1={xa:.3e} "
          f"{'OK' if (scale_ok and arch_ok and live) else 'PROBLEM'}")
if bad:
    print(f"\nFAIL: {bad} checkpoint(s) are wrong (scale mismatch, missing architecture, "
          f"or an inert cross-attention path).")
    sys.exit(1)
print(f"\nOK. Evaluate with:")
print(f"  ADAPTER_CKPT={final} sbatch run_xattn_fourier_eval_kcist.sh")
PY
