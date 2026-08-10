#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=chembl_sweep_%j.out

# JUPITER: single-GPU eta/omega sampling sweep on a foundation checkpoint. Grids
# eta (error-correction stochasticity) x omega (target guidance) at fixed
# polydec time-distortion, scoring each config with the full metric suite, to pick
# the sampling config the released model ships with. Writes sweep_results.json
# next to the checkpoint.
#
#   $1 checkpoint   $2 representation   $3 data-dir   $4 prefix
#   $5 samples/config   $6 etas   $7 omegas
#
# $2/$3/$4 MATTER, for the same reason they do in run_chembl_eval_jupiter.sh: the
# representation decides how graphs decode (a kekulized ckpt is 12 atom / 4 edge
# against the released 12 / 5, and the wrong one silently yields molecules made of
# the wrong elements), while --data-dir/--prefix select the KL reference and the
# novelty denominator. Left at their defaults a union checkpoint gets scored
# against ChEMBL's distribution.
#
#   sbatch run_chembl_sweep_jupiter.sh <ckpt> kekulized_v2 data/zinc_chembl_union union
#
# ON SAMPLE COUNT: the default was 500, which cannot do the job it was asked to do.
# At validity ~0.99 the standard error at n=500 is +-0.0044, so differences of the
# size a sweep looks for (~0.005) are inside the noise band -- the original ChEMBL
# sweep's "all configs within sampling noise" conclusion was partly a property of
# the measurement, not of the configs. Resolving a 0.005 difference at 2 sigma
# needs ~3000 samples/config. Default raised to 1500 as a compromise for a coarse
# first pass; run the shortlist again at 5000 before choosing. Two cheap stages
# beat one underpowered grid.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

CKPT="${1:-ckpts/chembl_foundation_lr3e-4/best_model.ckpt}"
REP="${2:-}"
DATA_DIR="${3:-}"
PREFIX="${4:-}"
SAMPLES="${5:-1500}"
ETAS="${6:-0,5,25,50,100}"
OMEGAS="${7:-0,0.05,0.1}"

REP_ARG=""; [ -n "$REP" ] && REP_ARG="--representation $REP"
DATA_ARG=""; [ -n "$DATA_DIR" ] && DATA_ARG="--data-dir $DATA_DIR"
PREFIX_ARG=""; [ -n "$PREFIX" ] && PREFIX_ARG="--prefix $PREFIX"

echo "sweep on $CKPT ${REP:+(representation=$REP)} ${DATA_DIR:+(data=$DATA_DIR/$PREFIX)}"
echo "  etas=$ETAS omegas=$OMEGAS samples/config=$SAMPLES @ $(date)"
python -u scripts/train_chembl_ddp.py --sweep --eval-ckpt "$CKPT" \
    $REP_ARG $DATA_ARG $PREFIX_ARG \
    --sweep-etas "$ETAS" --sweep-omegas "$OMEGAS" \
    --sweep-samples "$SAMPLES" --eval-sample-steps 500
echo "sweep done @ $(date)"
