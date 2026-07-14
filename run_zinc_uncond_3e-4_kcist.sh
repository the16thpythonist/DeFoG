#!/bin/bash
#SBATCH --job-name=zinc_unc_3e4
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=zinc_uncond_lr_3e-4_rerun_%j.out

# Re-run of the 3e-4 arm that lost the pycomex prepare_path startup race in the
# 4-way sweep (26192). Solo -> no concurrency, no race.
cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u experiments/training__zinc_uncond.py --LEARNING_RATE 3e-4
echo "3e-4 rerun finished at $(date)"
