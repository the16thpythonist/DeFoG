#!/bin/bash
# Render 5x5 sample-molecule grids for the KCIST LR sweep (job 26144) on one GPU.
#   sbatch experiments/run_make_grids_kcist.sh
#SBATCH --job-name=aqsoldb_grids
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=aqsoldb_grids_%j.out
#SBATCH --error=aqsoldb_grids_%j.out

cd "$HOME/Programming/DeFoG" || exit 1
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
python -u experiments/make_sample_grids.py
