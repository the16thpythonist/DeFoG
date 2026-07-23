#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=chembl_sweep_%j.out

# JUPITER: single-GPU eta/omega sampling sweep on a ChEMBL foundation checkpoint.
# Grids eta (error-correction stochasticity) x omega (target guidance) at fixed
# polydec time-distortion, scoring each config with the full metric suite, to pick
# the sampling config the released model ships with. Writes sweep_results.json next
# to the checkpoint. Pass a checkpoint as $1 (default: best_model of lr=3e-4 run).

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

CKPT="${1:-ckpts/chembl_foundation_lr3e-4/best_model.ckpt}"
echo "sweep on $CKPT @ $(date)"
python -u scripts/train_chembl_ddp.py --sweep --eval-ckpt "$CKPT" \
    --sweep-etas 0,5,25,50,100 --sweep-omegas 0,0.05,0.1 \
    --sweep-samples 500 --eval-sample-steps 500
echo "sweep done @ $(date)"
