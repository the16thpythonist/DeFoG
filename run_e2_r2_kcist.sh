#!/bin/bash
#SBATCH --job-name=e2r2
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG-probblend/e2r2_%j.out

# Does head-reward RL round 2 improve on round 1, and should we stop selecting on the head?
#
# TWO QUESTIONS, ONE JOB:
#
#   1. ROUND 2 vs ROUND 1. qed@5.0.0 (round 1, seed 21) is the baseline, and it is a SOLID
#      one: re-measured at sampling seed 1234 it gave 0.0894 against 0.0892 at seed 42, a
#      swing of 0.0002. Round 2's own internal eval says -16.8% at low targets, -8.0% at
#      mid and a null at high. Whether that survives 100 targets is the question -- round
#      1's apparent lead over qed@3.1.0 did NOT survive a second sampling seed.
#
#   2. WHICH SELECTOR. After round 2 the head ranks the four checkpoints BACKWARDS against
#      RDKit truth: its best pick (s21, probe 0.096) is truth's worst (0.1106 mean), and
#      its worst pick (s42, probe 0.101) is truth's best (0.1012). Early-stop deploys on
#      the head probe, so every round-2 arm shipped a checkpoint chosen by that ranking.
#      Running both picks head-to-head says whether the head's selection is actively
#      costing us, which is directly actionable: PROBE would have to switch to RDKit for
#      any property that has a closed form.
#
#      In round 1 the two agreed. The disagreement appearing only after a second round of
#      optimising against the head is what a reward being ground down looks like when the
#      damage shows up in RANKING rather than in level -- both head-MAE and RDKit-MAE
#      still improved, which is the check I was watching and it did not catch this.
#
# qed@3.1.0 is the standing rdkit-reward competitor AND the reproduction check: same seed,
# protocol and tree as jobs 43187/43190, so it must land near 0.0920/0.0888.
set -u
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
OUT=/home/tm4030/Programming/DeFoG-probblend/e2r2_results
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
run 0 r2_s42_truthpick molsmith/qed@6.0.0 &
run 1 r2_s21_headpick  molsmith/qed@6.1.0 &
run 2 r1_s21_baseline  molsmith/qed@5.0.0 &
run 3 rdkitrl_ref      molsmith/qed@3.1.0 &
wait
echo "ALL_ARMS_DONE"

$PY - <<'PY'
import json, os
b="/home/tm4030/Programming/DeFoG-probblend/e2r2_results"
order=[("r1_s21_baseline","round 1 s21 (qed@5.0.0)"),
       ("r2_s42_truthpick","round 2 s42 TRUTH-pick (6.0.0)"),
       ("r2_s21_headpick","round 2 s21 HEAD-pick (6.1.0)"),
       ("rdkitrl_ref","rdkit-RL ref (qed@3.1.0)")]
R={}
print(f"{'arm':34} {'MAE':>8} {'low':>8} {'mid':>8} {'high':>8} {'valid':>7} {'uniq':>7}")
for t,l in order:
    p=f"{b}/e2_{t}.json"
    if not os.path.exists(p): print(f"{l:34} (missing)"); continue
    d=json.load(open(p)); R[t]=d
    print(f"{l:34} {d['mae_pooled']:8.4f} {d['mae_low_third']:8.4f} {d['mae_mid_third']:8.4f} "
          f"{d['mae_high_third']:8.4f} {d['validity']:7.4f} {d['uniqueness']:7.4f}")
if "r1_s21_baseline" in R:
    base=R["r1_s21_baseline"]["mae_pooled"]
    print(f"\nbaseline reproduction qed@5.0.0: {base:.4f} vs 0.0892 (seed 42) -> "
          f"{'OK' if abs(base-0.0892)<1e-4 else 'DRIFTED'}")
    for t,l in order[1:3]:
        if t in R: print(f"  {l:34} vs round 1: {R[t]['mae_pooled']-base:+.4f}")
if "r2_s42_truthpick" in R and "r2_s21_headpick" in R:
    d=R["r2_s42_truthpick"]["mae_pooled"]-R["r2_s21_headpick"]["mae_pooled"]
    print(f"\nselector test: truth-pick minus head-pick = {d:+.4f} "
          f"({'truth-pick better' if d<0 else 'head-pick better'})")
PY
echo "SUMMARY_DONE"
