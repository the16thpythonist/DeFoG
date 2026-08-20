#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --output=fp_bneck_compare_%j.out

# Matched-width re-scoring of the four bottleneck-ablation arms (job 1288793).
#
# WHY THIS IS NEEDED DESPITE THE IN-JOB CONTROL
# The ablation's control runs at 1024 bits and the other three at 2048, and the
# experiment scores each arm at its OWN FP_BITS. Tanimoto is not comparable
# across widths -- measured directly: the same frozen base scores 0.146 at 512
# bits and 0.128 at 1024. So arm0's numbers sit on a different scale from
# arm1-3's. An in-job control removes every cross-job difference EXCEPT the one
# the control is defined by, which here is the bit width itself.
#
# Empirically lifts (differences) transfer across widths far better than
# absolutes do -- v2 scored +0.1470 at 512 and +0.1467 at 1024 -- so the
# ablation's own table is probably close to right. "Probably close" is not the
# standard for a result that decides whether to keep spending nodes on this
# axis, and the effect sizes here (~0.015) are the same order as the residual.
#
# WHAT THIS DOES NOT FIX
# The arms trained for UNEQUAL numbers of epochs (ctrl 26, bits 21, width 23,
# enc 29) because all four hit the 7.5h wall-clock cap rather than a common
# epoch count, and no arm had plateaued -- every probe was still climbing at its
# last measurement. Re-scoring cannot undo that. Read any arm-to-arm difference
# with the epoch column in hand; a rerun with a fixed epoch budget is the only
# clean fix.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

R=experiments/results/adapter_fingerprint__zinc
CTRL=$R/09_08_2026__07_17__39cb/fp_adapter
BITS=$R/09_08_2026__07_17__PPFr/fp_adapter
WIDTH=$R/09_08_2026__07_17__MWDv/fp_adapter
ENC=$R/09_08_2026__07_17__7emY/fp_adapter

for p in "$CTRL" "$BITS" "$WIDTH" "$ENC"; do
    [ -f "${p}.ckpt" ] || { echo "ERROR: ${p}.ckpt missing"; exit 1; }
done

# Bit width is deliberately NOT passed: the script reads each checkpoint's own
# cond_dim, so a mistyped width cannot silently condition an adapter wrongly.
python -u scripts/compare_fp_adapters.py \
  --base ckpts/zinc_rl2_seed42/best_model \
  --adapter ctrl:"$CTRL" \
  --adapter bits:"$BITS" \
  --adapter width:"$WIDTH" \
  --adapter enc:"$ENC" \
  --n-per-target 64 --n-baseline 256 --steps 500 --eta 25 \
  --out fp_bneck_compare_${SLURM_JOB_ID}.json

echo
echo "EPOCHS ACTUALLY TRAINED (the confound this cannot fix):"
echo "  ctrl 26   bits 21   width 23   enc 29   -- none plateaued"
