#!/bin/bash
#SBATCH --job-name=wlaw
#SBATCH --partition=small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG-probblend/wlaw_%j.out

# DOES THE GUIDANCE-GAIN LAW HOLD ON AN ADAPTER IT WAS NOT DERIVED FROM?
#
# The law, from molsmith/qed@3.1.0: slope(w) ~= k*w with k = slope(w=1) = 0.477, and MAE
# minimised at w* = 1/k = 2.10. TPSA is the test case, and it is a fair one: it is a
# different property on the same base, shipped, and never used to fit the law.
#
# PRE-REGISTRATION IS STRUCTURAL, NOT A PROMISE. Stage 1 measures w=1 alone and writes
# prediction.json containing w*. Stage 2 then runs a grid CENTRED ON that prediction. The
# prediction file is therefore on disk, timestamped, before any confirming point exists --
# which is the only way this differs from drawing the target after seeing the arrows. The
# analyser judges claim B against prediction.json, not against a k refitted on everything.
#
# CONDITIONS MATCH closed_loop_qed.py, THE SOURCE OF THE LAW: eta=25 (not the eta=5 the
# long-training evaluator uses), omega=0, 500 steps, polydec, 128 per level, seed 42. Get
# this wrong and the comparison is against a different curve.
#
# A DIRECTIONAL PREDICTION WORTH RECORDING: TPSA is far more linearly readable from atom
# counts than QED (R^2 0.852 vs 0.201), so if adapters steer it more efficiently, k should
# come out LARGER than 0.477 and w* correspondingly SMALLER than 2.10. A 24-molecule
# smoke run at 100 steps put slope(w=1) at 0.98 (w* ~ 1.0), which is consistent -- and it
# also showed slope/w falling from 0.98 to 0.67 between w=1 and w=2, so claim A may well
# FAIL by saturation while claim B passes. Recorded here before the real run.
set -u
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD:/home/tm4030/Programming/DeFoG
export PYTHONUNBUFFERED=1
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
OUT=/home/tm4030/Programming/DeFoG-probblend/wlaw_results
ADAPTER=molsmith/tpsa@2.0.0
rm -rf "$OUT"; mkdir -p "$OUT"

COMMON="--adapter $ADAPTER --base molsmith/zinc-kek --property tpsa \
        --percentiles 5,25,50,75,95 --n-per-level 128 --steps 500 \
        --eta 25.0 --omega 0.0 --time-distortion polydec --blend-space prob --seed 42"

echo "### STAGE 1 -- probe at w=1, which fixes the prediction"
CUDA_VISIBLE_DEVICES=0 $PY -u adapter_improvements/verify_w_law.py $COMMON \
    --weights 1.0 --out "$OUT/w_1.0.json" 2>&1 | tail -8

$PY - "$OUT" <<'PY'
import json, sys
d = json.load(open(f"{sys.argv[1]}/w_1.0.json"))
k = d["per_w"]["1.0"]["slope"]
w = 1.0 / k if k else float("nan")
json.dump({"slope_at_w1": k, "w_star_predicted": w,
           "qed_reference": {"k": 0.477, "w_star": 2.10}},
          open(f"{sys.argv[1]}/prediction.json", "w"), indent=1)
print(f"PREDICTION: slope(w=1) = {k:.4f}  ->  w* = {w:.3f}   (QED had k=0.477, w*=2.10)")
PY

W_STAR=$($PY -c "import json;print(json.load(open('$OUT/prediction.json'))['w_star_predicted'])")
echo "### STAGE 2 -- grid centred on w* = $W_STAR"

# Grid: fractions of w*, so the density sits where claim B is decided, plus far points on
# both sides so claim A is tested over a real range. Clamped to [0.5, 4.0] -- past 4 the
# blend degrades for reasons that have nothing to do with the law being tested.
read -r GA GB <<< "$($PY - "$W_STAR" <<'PY'
import sys
w = float(sys.argv[1])
# Fractions of w* give DENSITY where claim B is decided; the fixed anchors give RANGE,
# without which claim A is untestable. The smoke run put TPSA's w* near 1.0, which would
# otherwise have confined the whole grid to 0.5-1.5 and made "is slope proportional to w"
# unanswerable over any interesting span.
fr = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
anchors = [2.0, 3.0]
g = sorted({round(min(max(f * w, 0.5), 4.0), 2) for f in fr} | set(anchors) - {1.0})
half = (len(g) + 1) // 2
print(",".join(map(str, g[:half])), ",".join(map(str, g[half:])))
PY
)"
echo "  GPU0 weights: $GA"
echo "  GPU1 weights: $GB"

CUDA_VISIBLE_DEVICES=0 $PY -u adapter_improvements/verify_w_law.py $COMMON \
    --weights "$GA" --out "$OUT/grid_a.json" > "$OUT/grid_a.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -u adapter_improvements/verify_w_law.py $COMMON \
    --weights "$GB" --out "$OUT/grid_b.json" > "$OUT/grid_b.log" 2>&1 &
wait
grep -hE "^w=" "$OUT"/grid_*.log || { echo "GRID FAILED"; tail -20 "$OUT"/grid_*.log; exit 1; }

echo "### VERDICT"
$PY -u adapter_improvements/analyze_w_law.py --results "$OUT" \
    --prediction "$OUT/prediction.json" --out "$OUT/verdict.json"
