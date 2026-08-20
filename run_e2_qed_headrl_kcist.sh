#!/bin/bash
#SBATCH --job-name=e2headrl
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG-probblend/e2headrl_%j.out

# E2 on the two best head-reward RL adapters from job 43186, against the two references
# that make the number mean something.
#
# THE FOUR ARMS ARE ONE LADDER, all from the same starting adapter:
#   qed@2.0.0  pre-RL          -- what the adapter was before any RL
#   qed@3.1.0  1 round, rdkit  -- the like-for-like competitor (E2 MAE 0.0920 in job 43167)
#   qed@5.0.0  1 round, head, seed 21   } the two best of four seeds, chosen on TRUE
#   qed@5.1.0  1 round, head, seed 7    } RDKit MAE at the post-RL eval (0.1185, 0.1210)
#
# Comparing only the best two of four is a ship-the-best-checkpoint decision, not an
# unbiased estimate of what head-RL yields on average. The other two seeds (13, 42) also
# improved on every level, so the direction is not selection; the magnitude here is
# optimistic.
#
# qed@3.1.0 is a REPRODUCTION CHECK as well as a competitor: same seed, same protocol,
# same tree as job 43167, so it must return 0.0920. molsmith changed since (scale filling
# in load/sample), and --method adapter should be untouched by that. If 3.1.0 has moved,
# it did touch it and none of the other three arms can be read.
#
# Runs from DeFoG-probblend deliberately: that tree carries the prob-space blending, the
# fixed molsmith and the guarded e2_targeting. DeFoG/molsmith is still stale.
set -u
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
OUT=/home/tm4030/Programming/DeFoG-probblend/e2headrl_results
mkdir -p "$OUT"

run () {  # gpu name adapter
  CUDA_VISIBLE_DEVICES=$1 $PY -u scripts/e2_targeting.py \
    --base molsmith/zinc-kek --adapter "$3" --property qed \
    --split validation --method adapter --blend-space prob --weight 2.0 \
    --size-mode marginal --n-targets 100 --per-target 10 --steps 500 \
    --eta 25 --omega 0 --time-distortion polydec --seed 42 \
    --out "$OUT/e2_$2.json" > "$OUT/arm_$2.log" 2>&1
  echo "  done $2 (rc=$?)"
}

echo "=== ARMS ==="
run 0 headrl_s21 molsmith/qed@5.0.0 &
run 1 headrl_s7  molsmith/qed@5.1.0 &
run 2 rdkitrl    molsmith/qed@3.1.0 &
run 3 prerl      molsmith/qed@2.0.0 &
wait
echo "ALL_ARMS_DONE"

$PY - <<'PY'
import json, glob, os
base = "/home/tm4030/Programming/DeFoG-probblend/e2headrl_results"
order = [("prerl", "pre-RL (qed@2.0.0)"), ("rdkitrl", "rdkit-RL (qed@3.1.0)"),
         ("headrl_s21", "head-RL s21 (qed@5.0.0)"), ("headrl_s7", "head-RL s7 (qed@5.1.0)")]
rows = {}
for tag, label in order:
    p = f"{base}/e2_{tag}.json"
    if os.path.exists(p):
        rows[tag] = (label, json.load(open(p)))
print(f"{'arm':26} {'MAE':>8} {'low':>8} {'mid':>8} {'high':>8} {'valid':>7} {'uniq':>7}")
for tag, label in order:
    if tag not in rows:
        print(f"{label:26} (missing)"); continue
    _, d = rows[tag]
    print(f"{label:26} {d['mae_pooled']:8.4f} {d['mae_low_third']:8.4f} "
          f"{d['mae_mid_third']:8.4f} {d['mae_high_third']:8.4f} "
          f"{d['validity']:7.4f} {d['uniqueness']:7.4f}")
if "rdkitrl" in rows:
    got = rows["rdkitrl"][1]["mae_pooled"]
    ok = abs(got - 0.0920) < 1e-4
    print(f"\nreproduction check qed@3.1.0: {got:.4f} vs 0.0920 -> {'OK' if ok else 'DRIFTED'}")
PY
echo "SUMMARY_DONE"
