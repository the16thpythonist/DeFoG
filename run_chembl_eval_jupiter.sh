#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=chembl_eval_%j.out

# JUPITER: single-GPU extended eval of a ChEMBL foundation checkpoint -- 1000
# samples, 500 steps, eta=0, computing validity / uniqueness / novelty /
# connected / disconnected / sanity / wonky-ring / KL(logP,TPSA,QED). Writes
# eval_metrics.json next to the checkpoint. Pass a checkpoint as $1 (defaults to
# the best_model of the lr=3e-4 foundation run).

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

CKPT="${1:-ckpts/chembl_foundation_lr3e-4/best_model.ckpt}"
echo "eval $CKPT @ $(date)"
python -u scripts/train_chembl_ddp.py --eval-only --eval-ckpt "$CKPT"
echo "eval done @ $(date)"
