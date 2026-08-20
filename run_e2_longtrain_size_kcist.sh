#!/bin/bash
#SBATCH --job-name=ltrsize
#SBATCH --partition=small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG-probblend/ltrsize_%j.out

# THE LEARNED-SIZE HALF OF THE PAIR. Submit with --dependency=afterany:<ltreval id>.
#
# WHY. run_e2_longtrain_eval_kcist.sh evaluates every checkpoint with the base's own
# UNCONDITIONAL P(n) -- the historical default, and the right constant to hold while
# asking "does training longer help?", since a fixed size policy cannot manufacture a
# trend in epochs. But it is NOT the configuration the shipped adapters run in, so those
# absolute MAEs are not comparable to the shipped logP 0.5420 / QED 0.0920. This job
# supplies the missing arm.
#
# It is deliberately a PAIR, not a replacement: every flag below is identical to the
# marginal run except --size-mode/--size-model, so the difference is attributable. That
# discipline exists because FreeGress Tab. 3 shows conditioned node inference alone moving
# MW MAE by -70% -- folded silently into one column it reads as "the adapter got better".
#
# ONLY ep20 AND ep100. The trend across five points is already bought by the marginal run;
# what is missing is the endpoints under the shipping configuration. Four evaluations
# rather than ten, for the same conclusion.
#
# EXPECT THE LOW END TO MOVE MOST. Wave 1 measured learned size as pooled-null but -8.3%
# on the low-logP third with validity RISING there (0.984 -> 0.991), and the low third is
# exactly where these arms are weakest. If the gain instead lands mid-range, that
# contradicts Wave 1 and is worth more than the MAE itself.
set -u
cd /home/tm4030/Programming/DeFoG-probblend
export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1
PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
BASE=/home/tm4030/zinc_kek_base.ckpt
OUT=/home/tm4030/Programming/DeFoG-probblend/longtrain_results
mkdir -p "$OUT/eval_size"

# property -> size model. Both are property_from="decoded", matching how these adapters
# were labelled; size_clogp_decoded reports property_name="logp" (clogp is an alias of
# logp, same Crippen estimate). eval_adapter_ckpt.py re-checks this and refuses on a
# mismatch rather than conditioning sizes on the wrong signal.
SIZE_logp=/home/tm4030/Programming/DeFoG-probblend/size_clogp_decoded.ckpt
SIZE_qed=/home/tm4030/Programming/DeFoG/ckpts/heads/qed_head_size.ckpt

eval_property () {   # gpu property size_model
  local gpu=$1 prop=$2 sm=$3
  local arch
  arch=$(cat "$OUT/archive_$prop.txt" 2>/dev/null)
  if [ -z "$arch" ] || [ ! -d "$arch" ]; then echo "NO ARCHIVE for $prop -- skipping"; return 1; fi
  if [ ! -f "$sm" ]; then echo "MISSING SIZE MODEL $sm for $prop -- skipping"; return 1; fi
  for ep in 20 100; do
    local ck="$arch/${prop}_adapter_ep${ep}.ckpt"
    if [ ! -f "$ck" ]; then echo "MISSING $ck"; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/eval_adapter_ckpt.py \
        --base "$BASE" --adapter-ckpt "$ck" --property "$prop" \
        --vocabulary e1_kekulized --epoch "$ep" --split validation \
        --size-mode learned --size-model "$sm" \
        --slope-weight 2.0 --n-per-level 128 \
        --weights 1.0,2.0,3.0 --n-targets 100 --per-target 10 \
        --steps 500 --eta 5.0 --omega 0.0 --blend-space prob --seed 42 \
        --out "$OUT/eval_size/${prop}_ep${ep}.json" \
        > "$OUT/eval_size/${prop}_ep${ep}.log" 2>&1
    echo "  $prop ep$ep (learned size) -> $OUT/eval_size/${prop}_ep${ep}.json"
  done
}

eval_property 0 qed  "$SIZE_qed"  &
eval_property 1 logp "$SIZE_logp" &
wait
echo "ALL_SIZE_EVAL_DONE"
ls -1 "$OUT"/eval_size/*.json 2>/dev/null

# The paired comparison, marginal vs learned, at each endpoint.
$PY -u adapter_improvements/compare_size_arms.py \
    --marginal "$OUT/eval" --learned "$OUT/eval_size" \
    --out "$OUT/size_pair_verdict.json" || echo "comparator failed (results are on disk)"
