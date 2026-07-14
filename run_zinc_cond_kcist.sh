#!/bin/bash
#SBATCH --job-name=zinc_cond
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=zinc_cond_%j.out

# ZINC 250k conditional generation on logP + TPSA + QED (50 epochs, 1x RTX 4090).
# Evaluates all 8 low/high (5th/95th pct) scenarios -> per-scenario distribution
# plot + 5x5 grid + generated_smiles.json.

set -e
cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "starting zinc conditional run at $(date)"

python -u experiments/conditional_training__zinc.py

echo "finished at $(date)"
