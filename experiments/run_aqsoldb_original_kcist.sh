#!/bin/bash
# Train the ORIGINAL DeFoG (src/) unconditionally on AqSolDB, for comparison
# against the reimplemented `defog` package. Uses the authors' own
# configs/experiment/aqsoldb.yaml (conditional defaults to False).
#
#   Full:   sbatch experiments/run_aqsoldb_original_kcist.sh
#   Smoke:  sbatch --time=00:30:00 experiments/run_aqsoldb_original_kcist.sh train.n_epochs=1 general.final_model_samples_to_generate=16
#
# Extra args are forwarded to Hydra (e.g. train.n_epochs=1).
#SBATCH --job-name=aqsoldb_orig
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -e
cd "$HOME/Programming/DeFoG/src"
source "$HOME/Programming/DeFoG/.venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "host: $(hostname)  job: $SLURM_JOB_ID"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python main.py +experiment=aqsoldb dataset=aqsoldb general.gpus=1 "$@"
