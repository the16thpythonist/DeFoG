#!/bin/bash
#SBATCH --job-name=longtrain
#SBATCH --partition=small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG-probblend/longtrain_%j.out

# IS THE ADAPTER UNDERTRAINED? The last open capacity hypothesis, on QED and logP.
#
# WHERE THIS COMES FROM. The capacity ladder (job 43124) ruled out width and depth on
# their own evidence -- 4.97M -> 20.6M params moved QED slope 0.369 -> 0.323, i.e. not at
# all -- but its C_long arm, the one testing training LENGTH, was killed by the 12h SLURM
# wall at epoch 51 with no final eval and no saved adapter. Its probe trend was
# +0.00425 slope/epoch, se 0.00242, p=0.118, R2=0.28, and it drops to +0.00164 (p=0.51)
# if the last point is removed. That is not evidence of flatness; it is too little power
# to tell. If the point estimate is real it is worth +0.19 slope over 45 epochs, which is
# large. So the hypothesis is open, and it is open specifically on the arm that died.
#
# WHAT IS DIFFERENT THIS TIME, and each of these is a fix for a specific past failure:
#
#   1. CKPT_EVERY_K=20 -- the adapter is written every 20 epochs instead of only after
#      trainer.fit returns. This is why C_long was lost ENTIRELY rather than truncated.
#   2. MAX_TIME_HOURS=36 against a 48h wall. C_long had 9.5h against 12h; Lightning's cap
#      did not fire in time. Margin now exceeds the whole expected runtime.
#   3. h256, NOT h1024. C_long was the WIDE adapter trained long, so it confounded the
#      two axes. Width is already ruled null, so the length question belongs on the
#      geometry that actually ships -- and a win here is directly shippable.
#   4. PROBE_WEIGHT=2.0, and the real evaluation happens at w=2 too. Every capacity-ladder
#      number was taken at w=1 because rate-space blending made anything higher unusable.
#      The blend now defaults to prob space, where w=2 is the optimum.
#
# TWO ARMS, TWO GPUS. One seed (42) per property, by choice: the budget goes into five
# real evaluation points per arm rather than into a second seed. That is the right trade
# HERE because the failure last time was trusting mid-training probes -- on the two arms
# that had both, probe-vs-final offsets went in OPPOSITE directions (+0.112, -0.131) on a
# quantity of ~0.35. Five checkpoints evaluated properly beat ten probes fitted through
# noise. It does mean there is no cross-seed replication, so the verdict rests on the
# within-arm trend across epochs; that is pre-registered in the analyser, not chosen after
# looking.
#
# BASE is the EXPORTED zinc-kek base, not ckpts/zinc_uncond_4e-4_connectivity.ckpt. Those
# two share geometry (9 layers, dx=256/de=64/dy=64, max_nodes=38) but differ in weights,
# so training against the wrong one converges fine and yields a useless adapter that
# passes every dimension check.
set -u
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
BASE=/home/tm4030/zinc_kek_base.ckpt
OUT=/home/tm4030/Programming/DeFoG-probblend/longtrain_results
mkdir -p "$OUT"

test -f "$BASE" || { echo "MISSING BASE $BASE"; exit 1; }
test -f "$PY"   || { echo "MISSING PY $PY"; exit 1; }

EPOCHS=100
CKPT_EVERY=20

run_arm () {   # gpu property
  CUDA_VISIBLE_DEVICES=$1 $PY -u experiments/adapter_training__zinc.py \
      --VOCABULARY '"e1_kekulized"' \
      --PROPERTY "\"$2\"" \
      --PROPERTY_FROM '"decoded"' \
      --BASE_CKPT "\"$BASE\"" \
      --H_HIDDEN 256 \
      --INTERIOR_FF True \
      --INTERIOR_ATTN False \
      --EPOCHS "$EPOCHS" \
      --CKPT_EVERY_K "$CKPT_EVERY" \
      --MAX_TIME_HOURS 36 \
      --LEARNING_RATE 2e-4 \
      --BATCH_SIZE 24 \
      --PROBE_EVERY_K 10 \
      --PROBE_WEIGHT 2.0 \
      --TARGET_PERCENTILES '[5,25,50,75,95]' \
      --LEVEL_NAMES '["p05","p25","p50","p75","p95"]' \
      --GUIDANCE_WEIGHTS '[2.0]' \
      --N_PER_TARGET 128 \
      > "$OUT/train_$2.log" 2>&1 &
  echo "launched $2 on GPU $1 (epochs=$EPOCHS ckpt_every=$CKPT_EVERY)"
  sleep 20   # stagger: concurrent pycomex starts have raced on the results mkdir before
}

run_arm 0 qed
run_arm 1 logp
wait
echo "ALL_TRAINING_DONE"

# Point the eval job at the archives without anyone having to read two logs for a path.
for p in qed logp; do
  ARCH=$(grep -oE '/home/tm4030/[^ ]*/results/adapter_training__zinc/[^ ]+' "$OUT/train_$p.log" | head -1)
  echo "$p archive: $ARCH"
  echo "$ARCH" > "$OUT/archive_$p.txt"
  ls -1 "$ARCH"/${p}_adapter_ep*.ckpt 2>/dev/null | sort -V || echo "  (no epoch checkpoints found for $p)"
done

for f in "$OUT"/train_*.log; do
  echo "== $(basename "$f")"
  grep -E "adapter:|Training adapter|checkpoint ->|PROBE|Finished|archive" "$f" | tail -8
done
