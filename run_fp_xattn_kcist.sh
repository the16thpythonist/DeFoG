#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=fp_xattn
#SBATCH --output=fp_xattn_%j.out

# Morgan-fingerprint steering adapter on the NEW architecture.
#
# WHAT IS NEW: node -> condition cross-attention at the measured-best setting from the
# scalar-property sweep (64 tokens / dim 128 / 16 heads).
#
# WHAT IS DELIBERATELY ABSENT: the Fourier bands. AdaLNAdapter REFUSES cond_fourier
# together with a cond_encoder, and correctly -- Fourier features are a result about
# LOW-dimensional inputs, and expanding 1024 hashed substructure counts into frequency
# bands would be meaningless. So on this condition the new architecture reduces to
# cross-attention alone, which the attribution ablation showed is the half that carries
# the effect anyway (cross-attention alone -0.153 vs Fourier alone -0.024 on logP).
#
# THE COMPARISON IS NOT CLEAN, AND THAT IS A CHOICE. The shipped fp_adapter is 512-bit
# BINARY; this is 1024-bit COUNTS. Two axes move at once (encoding and architecture), so
# a difference cannot be attributed to cross-attention alone. The 512-binary arm is the
# control that would fix that, and is not run here.
#
# Related prior finding worth re-testing rather than assuming: 512 -> 1024 bits
# previously bought only +0.015 Tanimoto because the trunk's first layer was the
# bottleneck. Cross-attention moves where the narrow point is, so that conclusion may no
# longer hold.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
BASE="${BASE:-ckpts/zinc_kek_base}"
VOCAB="${VOCAB:-e1_kekulized}"
FP_BITS="${FP_BITS:-1024}"
FP_RADIUS="${FP_RADIUS:-2}"
FP_COUNTS="${FP_COUNTS:-True}"
XATTN_TOKENS="${XATTN_TOKENS:-64}"
XATTN_DIM="${XATTN_DIM:-128}"
XATTN_HEADS="${XATTN_HEADS:-16}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-4e-4}"
WANT_TOKEN="-94.15126384728774"

echo "fingerprint xattn adapter @ $(date) on $(hostname)"
echo "base=${BASE} vocab=${VOCAB} bits=${FP_BITS} radius=${FP_RADIUS} counts=${FP_COUNTS}"
echo "xattn=${XATTN_TOKENS}/${XATTN_DIM}/${XATTN_HEADS} epochs=${EPOCHS} lr=${LR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

# ---- preflight: right base, and the encoder+cross-attention combination builds --------
$PY - <<PY || { echo "ERROR: preflight failed -- refusing to train"; exit 1; }
import sys
sys.path.insert(0, ".")
from defog.core import DeFoGModel, AdaLNAdapter
from defog.core.adapter import _base_token
b = DeFoGModel.load("${BASE}", device="cpu")
tok = _base_token(b); want = float("${WANT_TOKEN}")
print(f"base token {tok!r}")
if b.cond_dim != 0:
    print("FAIL: base is not unconditional"); sys.exit(1)
if abs(tok - want) > 1e-6 * (1 + abs(want)):
    print(f"FAIL: base token {tok} != {want}"); sys.exit(1)
a = AdaLNAdapter.for_base(
    b, cond_dim=${FP_BITS}, hidden=256,
    cond_encoder={"kind": "mlp", "in_dim": ${FP_BITS}, "out_dim": 512,
                  "hidden": 1024, "n_blocks": 2},
    xattn_tokens=${XATTN_TOKENS}, xattn_dim=${XATTN_DIM}, xattn_heads=${XATTN_HEADS})
g = sum(float(p.detach().abs().sum()) for lay in a.gate for k in lay for p in lay[k].parameters())
o = sum(float(m.out.weight.detach().abs().sum() + m.out.bias.detach().abs().sum()) for m in a.xattn)
print(f"adapter: {sum(p.numel() for p in a.parameters()):,} params "
      f"(gate L1 {g:.1e}, xattn out L1 {o:.1e})")
if g != 0.0 or o != 0.0:
    print("FAIL: not an exact no-op at init"); sys.exit(1)
print("preflight OK")
PY

mkdir -p experiments/results/adapter_fingerprint__zinc
TRAIN_LOG="fp_xattn_trainlog_${SLURM_JOB_ID:-local}.txt"

$PY -u experiments/adapter_fingerprint__zinc.py \
    --VOCABULARY "'${VOCAB}'" \
    --BASE_CKPT "'${BASE}'" \
    --FP_BITS ${FP_BITS} \
    --FP_RADIUS ${FP_RADIUS} \
    --FP_COUNTS ${FP_COUNTS} \
    --COND_ENCODER "{'kind':'mlp','out_dim':512,'hidden':1024,'n_blocks':2}" \
    --XATTN_TOKENS ${XATTN_TOKENS} \
    --XATTN_DIM ${XATTN_DIM} \
    --XATTN_HEADS ${XATTN_HEADS} \
    --EPOCHS ${EPOCHS} \
    --LEARNING_RATE ${LR} \
    --MAX_TIME_HOURS 20.0 \
    --__DEBUG__ False 2>&1 | tee "$TRAIN_LOG"

rc=${PIPESTATUS[0]}
echo "training exited ${rc} at $(date)"
[ $rc -eq 0 ] || exit $rc

RESULT_DIR=$(grep -oE 'archive path:[[:space:]]+\S+' "$TRAIN_LOG" | head -1 | awk '{print $3}')
if [ -z "$RESULT_DIR" ] || [ ! -d "$RESULT_DIR" ]; then
    echo "ERROR: could not resolve this job's own result directory"; exit 1
fi
echo "result dir: $RESULT_DIR"
ls -la "$RESULT_DIR" | grep -i "\.ckpt" || echo "WARNING: no checkpoint found"

$PY - "$RESULT_DIR" "${XATTN_TOKENS}" "${FP_BITS}" <<'PY'
import glob, os, sys, torch
d, want_tok, want_bits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
found = sorted(glob.glob(os.path.join(d, "*.ckpt")))
if not found:
    print("FAIL: no checkpoint written"); sys.exit(1)
bad = 0
for f in found:
    cfg = torch.load(f, map_location="cpu", weights_only=False)["config"]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    enc = cfg.get("cond_encoder")
    xa = sum(float(v.abs().sum()) for k, v in sd.items()
             if k.startswith("xattn.") and ".out." in k)
    ok = (cfg.get("xattn_tokens") == want_tok and cfg.get("cond_dim") == want_bits
          and enc is not None and xa > 0.0)
    bad += not ok
    print(f"{os.path.basename(f):28s} cond_dim={cfg.get('cond_dim')} "
          f"xattn={cfg.get('xattn_tokens')}/{cfg.get('xattn_dim')}/{cfg.get('xattn_heads')} "
          f"encoder={'yes' if enc else 'NO'} xattn_out_L1={xa:.3e} {'OK' if ok else 'PROBLEM'}")
if bad:
    print(f"\nFAIL: {bad} checkpoint(s) wrong -- missing encoder, wrong width, or an "
          f"inert cross-attention path.")
    sys.exit(1)
print("\nOK")
PY
