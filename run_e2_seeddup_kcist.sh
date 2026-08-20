#!/bin/bash
#SBATCH --job-name=e2seeddup
#SBATCH --partition=small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG-probblend/e2seeddup_%j.out

# Is qed@5.0.0's 0.0892 real, or a lucky sampling draw?
#
# It was the best of four RL seeds, and its lead over the rdkit-reward adapter qed@3.1.0
# (0.0920) is -0.0028 at p=0.44 -- inside the 0.008 noise floor. Round 2 is being trained
# from it, so whether that number is solid decides whether any round-2 gain is measured
# against a real baseline or an inflated one.
#
# The E2 sampling seed is the only thing changed: 1234 instead of 42. Both arms move
# together, so the PAIRED comparison at the new seed is the quantity of interest -- if
# s21's lead over 3.1.0 survives a different draw it is probably real, and if it inverts
# it was noise. Everything else is identical to job 43187.
set -u
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
OUT=/home/tm4030/Programming/DeFoG-probblend/e2seeddup_results
mkdir -p "$OUT"

run () {  # gpu name adapter
  CUDA_VISIBLE_DEVICES=$1 $PY -u scripts/e2_targeting.py \
    --base molsmith/zinc-kek --adapter "$3" --property qed \
    --split validation --method adapter --blend-space prob --weight 2.0 \
    --size-mode marginal --n-targets 100 --per-target 10 --steps 500 \
    --eta 25 --omega 0 --time-distortion polydec --seed 1234 \
    --out "$OUT/e2_$2.json" > "$OUT/arm_$2.log" 2>&1
  echo "  done $2 (rc=$?)"
}

run 0 headrl_s21_seed1234 molsmith/qed@5.0.0 &
run 1 rdkitrl_seed1234    molsmith/qed@3.1.0 &
wait
echo "ALL_ARMS_DONE"

$PY - <<'PY'
import json
b = "/home/tm4030/Programming/DeFoG-probblend/e2seeddup_results"
a = json.load(open(f"{b}/e2_headrl_s21_seed1234.json"))
c = json.load(open(f"{b}/e2_rdkitrl_seed1234.json"))
print(f"{'arm':28} {'MAE':>8} {'low':>8} {'mid':>8} {'high':>8} {'valid':>7}")
for lab, d in (("head-RL s21 @seed1234", a), ("rdkit-RL 3.1.0 @seed1234", c)):
    print(f"{lab:28} {d['mae_pooled']:8.4f} {d['mae_low_third']:8.4f} "
          f"{d['mae_mid_third']:8.4f} {d['mae_high_third']:8.4f} {d['validity']:7.4f}")
print(f"\nseed 42  (job 43187): s21 0.0892  3.1.0 0.0920  -> delta -0.0028")
print(f"seed 1234 (this job): s21 {a['mae_pooled']:.4f}  3.1.0 {c['mae_pooled']:.4f}  "
      f"-> delta {a['mae_pooled']-c['mae_pooled']:+.4f}")
PY
echo "SUMMARY_DONE"
