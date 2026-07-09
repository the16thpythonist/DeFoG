#!/bin/bash
# Unconditional AqSolDB training (control run) on kcist, plain SLURM.
#   sbatch experiments/run_aqsoldb_uncond_train_kcist.sh
#SBATCH --job-name=aqsoldb_uncond_train
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -e
cd "$HOME/Programming/DeFoG"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "host: $(hostname)  job: $SLURM_JOB_ID"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python experiments/training__aqsoldb_uncond.py "$@"
