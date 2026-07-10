#!/bin/bash
# Backfill generated_smiles.json for the KCIST LR sweep (job 26144) on one GPU.
#   sbatch experiments/run_backfill_kcist.sh
#SBATCH --job-name=aqsoldb_backfill
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=aqsoldb_backfill_%j.out
#SBATCH --error=aqsoldb_backfill_%j.out

cd "$HOME/Programming/DeFoG" || exit 1
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
python -u experiments/backfill_smiles_kcist.py
