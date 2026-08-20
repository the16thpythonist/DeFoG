#!/bin/bash
#SBATCH --job-name=fkfixed
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG-probblend/fkfixed_%j.out

# Feynman-Kac, re-run with two defects corrected. Everything else is bit-identical to the
# beta sweep (job 43161) so the two are directly comparable.
#
# FIX 1 -- kekulize. LearnedPropertyEnergy re-encoded each decoded molecule via
# Chem.MolToSmiles (aromatic) into a vocabulary with no AROMATIC class, so 94% of real
# molecules returned invalid_energy=1e3 and the head was never consulted. FK's dominant
# signal was "is this molecule non-aromatic", not the property.
#
# FIX 2 -- scale. The energy was raw squared error in the property's own units, so a shared
# beta meant ~76x less pressure on QED (std 0.13) than logP (std 1.16): the QED weights
# stayed within a few percent of uniform and the ESS gate never fired at any beta swept.
# The energy is now divided by the property's std^2, making beta DIMENSIONLESS.
#
# BETA IS THEREFORE IN NEW UNITS. The old known-good logP point (raw beta 25) is
# dimensionless 33.5. The ladder {5, 10, 20, 33.5, 60} is deliberately the SAME for both
# properties -- that comparability is the entire point of normalising, and the predicted
# resample rates now bracket sensibly on both (logP 1.6-53%, QED 8-72%).
#
# The two adapter-only arms are controls, not filler: neither fix touches the --method
# adapter path, so they MUST reproduce logP 0.5420 and QED 0.0920 exactly. If they do not,
# something else moved and the FK arms cannot be read.
set -u
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
OUT=/home/tm4030/Programming/DeFoG-probblend/fkfixed_results
mkdir -p "$OUT"

# ---------------------------------------------------------------- gate
# Refuse to spend four GPU-hours unless both fixes are live on the config that samples.
# Checking the source would pass even when the value never reaches the energy.
echo "=== GATE ==="
CUDA_VISIBLE_DEVICES=0 $PY -u adapter_improvements/gate_fk.py 2>&1 | tee "$OUT/gate.log"
if ! grep -q "GATE PASSED" "$OUT/gate.log"; then
  echo "ABORTING: gate did not pass. No arms launched."
  exit 1
fi

# ---------------------------------------------------------------- arms
run () {  # gpu name property adapter method beta
  local extra=""
  if [ "$5" = "fk" ]; then
    extra="--fk-beta $6 --fk-warmup 0.6 --fk-ess 0.25"
  fi
  CUDA_VISIBLE_DEVICES=$1 $PY -u scripts/e2_targeting.py \
    --base molsmith/zinc-kek --adapter "$4" --property "$3" \
    --split validation --method "$5" --blend-space prob --weight 2.0 \
    --size-mode marginal --n-targets 100 --per-target 10 --steps 500 \
    --eta 25 --omega 0 --time-distortion polydec --seed 42 $extra \
    --out "$OUT/e2_$2.json" > "$OUT/arm_$2.log" 2>&1
  echo "  done $2 (rc=$?)"
}

LOGP=molsmith/clogp@1.2.0
QED=molsmith/qed@3.1.0

echo "=== ARMS (3 per GPU, sequential within a GPU) ==="
( run 0 logp_adapter logp $LOGP adapter 0
  run 0 logp_b5      logp $LOGP fk      5
  run 0 logp_b10     logp $LOGP fk      10 ) &
( run 1 logp_b20     logp $LOGP fk      20
  run 1 logp_b33p5   logp $LOGP fk      33.5
  run 1 logp_b60     logp $LOGP fk      60 ) &
( run 2 qed_adapter  qed  $QED  adapter 0
  run 2 qed_b5       qed  $QED  fk      5
  run 2 qed_b10      qed  $QED  fk      10 ) &
( run 3 qed_b20      qed  $QED  fk      20
  run 3 qed_b33p5    qed  $QED  fk      33.5
  run 3 qed_b60      qed  $QED  fk      60 ) &
wait
echo "ALL_ARMS_DONE"

# ---------------------------------------------------------------- summary
for f in "$OUT"/e2_*.json; do
  $PY -c "
import json,sys
d=json.load(open('$f'))
fk=d.get('fk') or {}
print(f\"{'$(basename $f .json)'.replace('e2_',''):16s} beta={str(fk.get('beta','-')):>6s}  \"
      f\"MAE {d['mae_pooled']:.4f}  low {d['mae_low_third']:.4f} mid {d['mae_mid_third']:.4f} \"
      f\"high {d['mae_high_third']:.4f}  valid {d['validity']:.4f}  uniq {d['uniqueness']:.4f}\")
"
done
echo "SUMMARY_DONE"
