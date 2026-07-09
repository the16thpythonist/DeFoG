#!/bin/bash
# Standalone evaluation of the trained AqSolDB model on kcist (chunked GPU sampling).
#   sbatch experiments/run_aqsoldb_eval_kcist.sh
#SBATCH --job-name=aqsoldb_eval
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -e
cd "$HOME/Programming/DeFoG"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python experiments/eval_aqsoldb.py \
  --ckpt experiments/results/conditional_training__aqsoldb/debug/model \
  --csv data/aqsoldb_conditional.csv \
  --outdir experiments/results/aqsoldb_eval \
  --num-samples 500 --chunk 32 "$@"
