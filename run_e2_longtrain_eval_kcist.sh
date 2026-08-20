#!/bin/bash
#SBATCH --job-name=ltreval
#SBATCH --partition=small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG-probblend/ltreval_%j.out

# Evaluate every epoch checkpoint from run_e2_longtrain_kcist.sh, on BOTH pre-registered
# metrics. Submit with --dependency=afterok:<train job id>.
#
# WHY A SEPARATE JOB. The evaluation is the part most likely to need re-running (a
# different weight grid, more targets), and re-running it must never mean retraining. It
# also means a training overrun cannot eat the evaluation's wall time -- which is the
# same class of mistake that lost C_long.
#
# THE TWO METRICS, and why neither is dropped. Both come out of one invocation per
# checkpoint so they cannot drift apart in seed, sampler or weights:
#
#   SLOPE, p05..p95 grid at a FIXED w=2 -- the mechanism readout, comparable to the
#     capacity ladder's 0.369 (h256/20ep) and 0.323 (h1024/20ep), except those were taken
#     at w=1. Slope rather than MAE because a QED adapter emitting the dataset mean
#     already scores MAE ~0.15: MAE cannot distinguish "steers well" from "does nothing".
#   E2 MAE over w in {1,2,3} -- the shipping readout, FreeGress's protocol, and the
#     number the request was actually about. The grid is kept wide enough to detect the
#     optimum MOVING with training length, which would itself be the finding.
#
# COST. ~640 molecules for the slope grid plus 3x1000 for the E2 sweep, at 500 steps, per
# checkpoint. At the measured ~1047s per 1000-molecule E2 arm that is ~65 min per
# checkpoint, so ~5.5h per property; the two properties run concurrently on the two GPUs.
set -u
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
BASE=/home/tm4030/zinc_kek_base.ckpt
OUT=/home/tm4030/Programming/DeFoG-probblend/longtrain_results
mkdir -p "$OUT/eval"

EPOCHS_TO_EVAL="20 40 60 80 100"

eval_property () {   # gpu property
  local gpu=$1 prop=$2
  local arch
  arch=$(cat "$OUT/archive_$prop.txt" 2>/dev/null)
  if [ -z "$arch" ] || [ ! -d "$arch" ]; then
    echo "NO ARCHIVE for $prop (looked in $OUT/archive_$prop.txt) -- skipping"
    return 1
  fi
  for ep in $EPOCHS_TO_EVAL; do
    local ck="$arch/${prop}_adapter_ep${ep}.ckpt"
    if [ ! -f "$ck" ]; then
      # Absence is INFORMATION -- it means training stopped before this epoch. Say so
      # loudly rather than quietly producing a shorter trend than was asked for.
      echo "MISSING $ck -- training did not reach epoch $ep"
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/eval_adapter_ckpt.py \
        --base "$BASE" \
        --adapter-ckpt "$ck" \
        --property "$prop" \
        --vocabulary e1_kekulized \
        --epoch "$ep" \
        --split validation \
        --slope-weight 2.0 \
        --n-per-level 128 \
        --weights 1.0,2.0,3.0 \
        --n-targets 100 \
        --per-target 10 \
        --steps 500 --eta 5.0 --omega 0.0 \
        --blend-space prob \
        --seed 42 \
        --out "$OUT/eval/${prop}_ep${ep}.json" \
        > "$OUT/eval/${prop}_ep${ep}.log" 2>&1
    echo "  $prop ep$ep done -> $OUT/eval/${prop}_ep${ep}.json"
  done
}

eval_property 0 qed &
eval_property 1 logp &
wait
echo "ALL_EVAL_DONE"
ls -1 "$OUT"/eval/*.json 2>/dev/null

$PY -u adapter_improvements/analyze_longtrain.py --results "$OUT/eval" \
    --out "$OUT/longtrain_verdict.json" || echo "analyser failed (results are still on disk)"
