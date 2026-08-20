#!/bin/bash
#SBATCH --job-name=fkknobs
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG-probblend/fkknobs_%j.out

# Two questions in one job, on top of the fixed FK (job 43167).
#
# Q1 -- CAN DIVERSITY BE BOUGHT BACK? Job 43167 left MAE still falling at the top of the
# ladder while uniqueness collapsed (logP 0.670, QED 0.551 at beta~=60). Every arm here sits
# at beta~=60 -- the worst-uniqueness / best-MAE point -- and turns ONE knob, so anything that
# moves is attributable. The question is not "does MAE improve" but whether any arm lands
# ABOVE the existing MAE-vs-uniqueness frontier rather than sliding along it.
#
#   rejuvenate  the purpose-built SMC remedy (resample-move); off in every run to date.
#               feynman_kac.py warns it wants a guided proposal_transform, which this path
#               does not pass -- it steers through the CFG-blended adapter -- so clones
#               diverge toward the adapter-conditioned distribution, not the raw base.
#               jump_length is the dial: more divergence, more drift back to adapter-only.
#   eta=50      orthogonal to FK entirely: DeFoG's own CTMC stochasticity. Makes clones
#               diverge between checkpoints without weakening selection at all.
#   warmup=0.3  resample earlier, so clones have 350 steps to separate instead of 200.
#               Also adds checkpoints, so the net is genuinely unknown.
#   ess=0.10    resample less often. A pure trade, included as the honest control: if the
#               others only match this, none of them is buying anything special.
#
# Q2 -- WHERE DOES THE LADDER TURN OVER? Both properties were still improving at beta~=60,
# the top rung. beta~=100 extends it. (I read QED as turning over at 20 on a partial ladder;
# beta~=60 beat it, and the 20-vs-33.5 gap was 0.0023 against a 0.008 noise floor.)
#
# The beta~=60 arms from job 43167 are the controls, same seed and settings -- not re-run.
set -u
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
OUT=/home/tm4030/Programming/DeFoG-probblend/fkknobs_results
mkdir -p "$OUT"

echo "=== GATE 1: both fixes still live ==="
CUDA_VISIBLE_DEVICES=0 $PY -u adapter_improvements/gate_fk.py 2>&1 | tee "$OUT/gate_fk.log"
grep -q "GATE PASSED" "$OUT/gate_fk.log" || { echo "ABORTING: fk gate failed"; exit 1; }

echo ""
echo "=== GATE 2: every knob in this sweep actually does something ==="
CUDA_VISIBLE_DEVICES=0 $PY -u adapter_improvements/gate_knobs.py 2>&1 | tee "$OUT/gate_knobs.log"
grep -q "GATE PASSED" "$OUT/gate_knobs.log" || { echo "ABORTING: a knob is inert"; exit 1; }

run () {  # gpu name property adapter beta eta warmup ess extra...
  local gpu=$1 name=$2 prop=$3 adp=$4 beta=$5 eta=$6 warm=$7 ess=$8; shift 8
  CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/e2_targeting.py \
    --base molsmith/zinc-kek --adapter "$adp" --property "$prop" \
    --split validation --method fk --blend-space prob --weight 2.0 \
    --size-mode marginal --n-targets 100 --per-target 10 --steps 500 \
    --eta "$eta" --omega 0 --time-distortion polydec --seed 42 \
    --fk-beta "$beta" --fk-warmup "$warm" --fk-ess "$ess" "$@" \
    --out "$OUT/e2_$name.json" > "$OUT/arm_$name.log" 2>&1
  echo "  done $name (rc=$?)"
}

LOGP=molsmith/clogp@1.2.0
QED=molsmith/qed@3.1.0

echo ""
echo "=== ARMS (3 per GPU) ==="
( run 0 logp_rejuv_j10 logp $LOGP 60  25 0.6  0.25 --fk-rejuvenate --fk-jump 10
  run 0 logp_rejuv_j25 logp $LOGP 60  25 0.6  0.25 --fk-rejuvenate --fk-jump 25
  run 0 logp_b100      logp $LOGP 100 25 0.6  0.25 ) &
( run 1 logp_eta50     logp $LOGP 60  50 0.6  0.25
  run 1 logp_warm03    logp $LOGP 60  25 0.3  0.25
  run 1 logp_ess010    logp $LOGP 60  25 0.6  0.10 ) &
( run 2 qed_rejuv_j10  qed  $QED  60  25 0.6  0.25 --fk-rejuvenate --fk-jump 10
  run 2 qed_rejuv_j25  qed  $QED  60  25 0.6  0.25 --fk-rejuvenate --fk-jump 25
  run 2 qed_b100       qed  $QED  100 25 0.6  0.25 ) &
( run 3 qed_eta50      qed  $QED  60  50 0.6  0.25
  run 3 qed_warm03     qed  $QED  60  25 0.3  0.25
  run 3 qed_ess010     qed  $QED  60  25 0.6  0.10 ) &
wait
echo "ALL_ARMS_DONE"

for f in "$OUT"/e2_*.json; do
  $PY -c "
import json
d=json.load(open('$f'))
print(f\"{'$(basename $f .json)'.replace('e2_',''):16s} MAE {d['mae_pooled']:.4f}  \"
      f\"valid {d['validity']:.4f}  uniq {d['uniqueness']:.4f}\")
"
done
echo "SUMMARY_DONE"
