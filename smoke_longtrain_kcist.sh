#!/bin/bash
# Smoke test for the long-training pair: exercises the two things that are NEW and
# therefore the two things that can fail on a 15-hour job -- periodic adapter
# checkpointing, and the checkpoint evaluator that reads what it writes.
#
# Run under srun on one GPU. It is deliberately end-to-end rather than a unit test: the
# failure mode being guarded against is "the launcher's pycomex flags do not parse" or
# "the eval script cannot load what the trainer saved", neither of which a unit test sees.
set -eu
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
BASE=/home/tm4030/zinc_kek_base.ckpt
OUT=/home/tm4030/Programming/DeFoG-probblend/longtrain_smoke
rm -rf "$OUT"; mkdir -p "$OUT"

echo "### 1. defaults"
$PY -c "
from defog.core import AdapterComposition, ConditionBranch
import inspect
assert inspect.signature(AdapterComposition.__init__).parameters['blend_space'].default == 'prob'
assert ConditionBranch.__dataclass_fields__['weight'].default == 2.0
print('   blend_space=prob, weight=2.0 OK')
"

echo "### 2. training with periodic checkpointing (testing mode, 2 epochs, ckpt every 1)"
$PY -u experiments/adapter_training__zinc.py --__TESTING__ True \
    --VOCABULARY '"e1_kekulized"' \
    --PROPERTY '"qed"' \
    --PROPERTY_FROM '"decoded"' \
    --BASE_CKPT "\"$BASE\"" \
    --CKPT_EVERY_K 1 \
    > "$OUT/train.log" 2>&1 || { echo "TRAIN FAILED"; tail -30 "$OUT/train.log"; exit 1; }

ARCH=$(grep -oE '/home/tm4030/[^ ]*/results/adapter_training__zinc/[^ ]+' "$OUT/train.log" | head -1)
echo "   archive: $ARCH"
grep -E "checkpoint ->" "$OUT/train.log" || { echo "NO CHECKPOINTS WRITTEN"; exit 1; }
ls -1 "$ARCH"/qed_adapter_ep*.ckpt || { echo "NO ep CHECKPOINT FILES"; exit 1; }

CK=$(ls -1 "$ARCH"/qed_adapter_ep*.ckpt | sort -V | tail -1)
echo "### 3. evaluating $CK (tiny settings)"
$PY -u scripts/eval_adapter_ckpt.py \
    --base "$BASE" --adapter-ckpt "$CK" --property qed \
    --vocabulary e1_kekulized --epoch 2 \
    --percentiles 5,50,95 --slope-weight 2.0 --n-per-level 8 \
    --weights 1.0,2.0 --n-targets 4 --per-target 2 \
    --steps 20 --chunk 8 --seed 42 \
    --out "$OUT/smoke_eval.json" > "$OUT/eval.log" 2>&1 \
    || { echo "EVAL FAILED"; tail -30 "$OUT/eval.log"; exit 1; }
tail -8 "$OUT/eval.log"

echo "### 4. analyser on the single smoke result"
mkdir -p "$OUT/evaldir" && cp "$OUT/smoke_eval.json" "$OUT/evaldir/qed_ep2.json"
$PY -u adapter_improvements/analyze_longtrain.py --results "$OUT/evaldir" \
    --out "$OUT/verdict.json" 2>&1 | tail -12

echo "SMOKE_OK"
