#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=e2_fk
#SBATCH --output=e2_fk_%j.out

# E2 with Feynman-Kac steering on top of the cross-attention adapters, one arm per
# property, plus learned conditional size sampling.
#
# K=10 AND per-target=10 ARE THE SAME NUMBER, deliberately. The ten molecules returned
# for a target ARE one particle system of K=10, all kept. The tempting alternative --
# ten systems of K=8, keeping the best particle from each -- is best-of-8 selection and
# an 8x compute advantage over a baseline that simply draws ten times.
#
# THE COST OF THAT CHOICE IS COLLAPSE, WHICH IS WHY UNIQUENESS IS REPORTED. Resampling
# culls low-weight particles and duplicates high-weight ones, so a badly tuned run can
# return ten copies of one molecule and post an excellent MAE. Read uniqueness first; an
# MAE below the adapter's with uniqueness well under 1.0 is an artefact, not a result.
#
# EACH ARM RUNS AT ITS OWN BEST w, taken from the adapter-only sweep, so FK is the only
# thing that changed. Those were logP 1.5, QED 2.0, TPSA 1.5, SA 1.5.
#
# THE ENERGY IS A LEARNED HEAD, and its accuracy bounds what FK can do: resampling on a
# head noisier than the adapter's own error follows the head's noise. Measured against
# DECODED labels -- the convention the heads were trained under and the one the E2 metric
# uses -- all four are 4-18x better than the adapter they steer:
#   logp 0.0185 (18x)  qed 0.0212 (4x)  tpsa 0.6231 (8x)  sascore 0.0366 (12x)
#
# SIZE MODELS: fit per property, gains over the marginal of +0.194 (QED), +0.134 (TPSA),
# +0.093 (logP) and +0.024 nats (SA score). The SA-score model is essentially inert
# (shrink 0.987) -- it is used for consistency, but if SA improves here the size draw is
# almost certainly not the reason.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
NT=100
STEPS=500
ETA=25
R=experiments/results/adapter_training__zinc

#      0                     1                  2                   3
NAMES=( logp                 qed                tpsa                sascore )
CKPT=(  $R/28_08_2026__15_50__Rb1X/clogp_adapter.ckpt \
        $R/28_08_2026__15_50__p4LJ/qed_adapter.ckpt \
        $R/28_08_2026__15_50__oIoj/tpsa_adapter.ckpt \
        $R/28_08_2026__15_50__fJez/sascore_adapter.ckpt )
HEAD=(  ckpts/heads/logp_head.ckpt ckpts/heads/qed_head.ckpt \
        ckpts/heads/tpsa_head.ckpt ckpts/heads/sascore_head.ckpt )
SIZE=(  ckpts/heads/logp_head_size.ckpt ckpts/heads/qed_head_size.ckpt \
        ckpts/heads/tpsa_head_size.ckpt ckpts/heads/sascore_head_size.ckpt )
W=(     1.5                  2.0                1.5                 1.5 )
# adapter-only reference at that same w, for reading the FK delta off the table
REF=(   0.3250               0.0867             4.7045              0.4233 )

OUT="e2_fk_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
echo "E2 + Feynman-Kac @ $(date) on $(hostname)   K=10, targets=${NT}, steps=${STEPS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# ---- preflight: every file exists before any GPU time is spent ---------------
missing=0
for i in 0 1 2 3; do
  for f in "${CKPT[$i]}" "${HEAD[$i]}" "${SIZE[$i]}"; do
    [ -f "$f" ] || { echo "MISSING for ${NAMES[$i]}: $f"; missing=1; }
  done
done
[ $missing -eq 0 ] || { echo "ERROR: inputs missing -- refusing to start"; exit 1; }
echo "all adapters, heads and size models present"

declare -a PIDS
FAILED=0
for i in 0 1 2 3; do
    (
        CUDA_VISIBLE_DEVICES=$i $PY -u scripts/e2_targeting.py \
            --adapter-ckpt "${CKPT[$i]}" \
            --head-ckpt "${HEAD[$i]}" \
            --property "${NAMES[$i]}" --split validation \
            --method fk --n-targets ${NT} --per-target 10 \
            --weight ${W[$i]} --steps ${STEPS} --eta ${ETA} --blend-space prob \
            --fk-beta 2.5 --fk-warmup 0.6 --fk-ess 0.5 --fk-jump 10 \
            --size-mode learned --size-model "${SIZE[$i]}" \
            --seed 42 --out "${OUT}/${NAMES[$i]}.json"
    ) > "e2fk_${NAMES[$i]}_${SLURM_JOB_ID:-local}.out" 2>&1 &
    PIDS[$i]=$!
    echo "launched ${NAMES[$i]} (w=${W[$i]}, head=$(basename ${HEAD[$i]})) on GPU $i"
    sleep 3
done
for i in 0 1 2 3; do
    if wait "${PIDS[$i]}"; then echo "  ok   ${NAMES[$i]}"
    else echo "  FAIL ${NAMES[$i]} (exit $?) -- see e2fk_${NAMES[$i]}_${SLURM_JOB_ID:-local}.out"; FAILED=1; fi
done
echo "finished at $(date)"

echo
echo "=== E2 + FK (K=10, learned size) vs adapter-only, same w ==="
$PY - "$OUT" "${NAMES[*]}" "${REF[*]}" "${W[*]}" <<'PY'
import json, math, os, sys
OUT = sys.argv[1]
names, refs, ws = sys.argv[2].split(), [float(x) for x in sys.argv[3].split()], sys.argv[4].split()
print(f"{'property':9s}{'w':>5s}{'adapter':>10s}{'+FK':>10s}{'delta':>9s}"
      f"{'valid':>8s}{'uniq':>8s}{'dead':>6s}")
missing = []
for n, r, w in zip(names, refs, ws):
    f = os.path.join(OUT, n + ".json")
    if not os.path.exists(f):
        missing.append(n); continue
    d = json.load(open(f))
    dead = sum(1 for x in d["per_target"] if not math.isfinite(x["mae"]))
    m = d["mae_pooled"]
    print(f"{n:9s}{w:>5s}{r:>10.4f}{m:>10.4f}{m-r:>+9.4f}"
          f"{d['validity']:>8.3f}{d['uniqueness']:>8.3f}{dead:>6d}")
if missing:
    print(f"\n!! MISSING: {', '.join(missing)} -- a REFUSING: line in that arm's log means "
          f"it rejected its own configuration.")
print()
print("READ UNIQUENESS FIRST. FK resampling duplicates high-weight particles, so an MAE")
print("below the adapter's with uniqueness well under 1.0 is collapse, not steering.")
print("Note also that two things changed at once here -- FK AND learned size conditioning")
print("-- so a delta cannot be attributed to either alone. The size-model gains over the")
print("marginal were +0.194 (QED), +0.134 (TPSA), +0.093 (logP), +0.024 nats (SA score),")
print("so SA score is the arm where the size draw is least likely to be responsible.")
PY

exit $FAILED
