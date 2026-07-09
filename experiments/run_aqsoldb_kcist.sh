#!/bin/bash
# Plain SLURM batch script for the AqSolDB conditional training experiment on kcist.
#
#   Full run:   sbatch experiments/run_aqsoldb_kcist.sh
#   Smoke run:  sbatch --time=00:20:00 experiments/run_aqsoldb_kcist.sh --__TESTING__ True
#
# Any extra CLI args are forwarded to the experiment (e.g. --EPOCHS, --__TESTING__).
#SBATCH --job-name=aqsoldb_cond
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -e
cd "$HOME/Programming/DeFoG"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "host: $(hostname)  job: $SLURM_JOB_ID  gpus: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python experiments/conditional_training__aqsoldb.py "$@"
