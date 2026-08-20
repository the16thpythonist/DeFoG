#!/bin/bash
# Smoke test for the learned-size arm. The thing that must NOT be trusted untested is
# that --size-mode learned actually changes the node counts: a wrapper that quietly
# hands the model None still runs, still produces molecules, and still writes
# "size_mode": "learned" into the JSON.
set -eu
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
BASE=/home/tm4030/zinc_kek_base.ckpt
ARCH=$(cat longtrain_results/archive_qed.txt)
CK="$ARCH/qed_adapter_ep20.ckpt"
SM=/home/tm4030/Programming/DeFoG/ckpts/heads/qed_head_size.ckpt
OUT=/home/tm4030/Programming/DeFoG-probblend/size_smoke
rm -rf "$OUT"; mkdir -p "$OUT"

echo "### 1. property-mismatch guard must REFUSE (qed adapter + tpsa size model)"
if $PY -u scripts/eval_adapter_ckpt.py --base "$BASE" --adapter-ckpt "$CK" --property qed \
     --vocabulary e1_kekulized --size-mode learned \
     --size-model /home/tm4030/Programming/DeFoG/ckpts/heads/tpsa_head_size.ckpt \
     --percentiles 50 --n-per-level 4 --skip-e2 --steps 10 --chunk 4 \
     --out "$OUT/should_not_exist.json" > "$OUT/guard.log" 2>&1; then
  echo "   FAIL: mismatched size model was ACCEPTED"; exit 1
else
  grep -q "REFUSING: size model is for" "$OUT/guard.log" \
    && echo "   OK refused: $(grep -o 'REFUSING.*' "$OUT/guard.log" | head -1 | cut -c1-80)" \
    || { echo "   FAIL: refused for the wrong reason"; tail -5 "$OUT/guard.log"; exit 1; }
fi

echo "### 2. learned size must CHANGE the node counts vs marginal"
$PY - <<'PY'
import sys, torch
sys.path.insert(0, ".")
from defog.core import LearnedSizeDistribution, DeFoGModel
sys.path.insert(0, "scripts")
from eval_adapter_ckpt import _TargetedSize
m = LearnedSizeDistribution.load("/home/tm4030/Programming/DeFoG/ckpts/heads/qed_head_size.ckpt")
torch.manual_seed(0)
for tgt in (0.45, 0.90):
    n = _TargetedSize(m, tgt, 512).sample(512, device="cpu").float()
    print(f"   learned P(n|qed={tgt}): mean {n.mean():.2f}  sd {n.std():.2f}")
base = DeFoGModel.load("/home/tm4030/zinc_kek_base.ckpt")
torch.manual_seed(0)
nb = base._resolve_size_dist(None, None).sample(512, device="cpu").float()
print(f"   marginal P(n)          : mean {nb.mean():.2f}  sd {nb.std():.2f}")
PY

echo "### 3. tiny end-to-end with learned size"
$PY -u scripts/eval_adapter_ckpt.py --base "$BASE" --adapter-ckpt "$CK" --property qed \
    --vocabulary e1_kekulized --epoch 20 --size-mode learned --size-model "$SM" \
    --percentiles 5,95 --slope-weight 2.0 --n-per-level 8 \
    --weights 2.0 --n-targets 3 --per-target 2 --steps 20 --chunk 8 \
    --out "$OUT/size_eval.json" > "$OUT/eval.log" 2>&1 || { tail -20 "$OUT/eval.log"; exit 1; }
grep -E "size model:|SLOPE|E2 w=" "$OUT/eval.log"
$PY -c "import json;d=json.load(open('$OUT/size_eval.json'));print('   recorded:',d['sampling']['size_mode'],d['sampling']['size_model'])"
echo "SIZE_SMOKE_OK"
