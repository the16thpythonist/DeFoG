#!/bin/bash
# Unconditional generation (1000 molecules) from the trained AqSolDB checkpoint.
#   sbatch experiments/run_aqsoldb_uncond_kcist.sh
#SBATCH --job-name=aqsoldb_uncond
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -e
cd "$HOME/Programming/DeFoG"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python experiments/eval_uncond_aqsoldb.py \
  --ckpt experiments/results/conditional_training__aqsoldb/debug/model \
  --csv data/aqsoldb_conditional.csv \
  --num-samples 1000 --chunk 32 "$@"
