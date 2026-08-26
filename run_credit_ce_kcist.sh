#!/bin/bash
#SBATCH --job-name=credit_ce
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/credit_ce_%j.out
#
# Round 4: gauge-invariant loss. Reuses the round-3 conditional pools, so this is
# training only -- no resampling.
#
#   sbatch run_credit_ce_kcist.sh 42
set -euo pipefail
SEED="${1:-42}"
cd "$HOME/Programming/DeFoG-dam"
mkdir -p logs ckpts/credit
export PYTHONPATH="$PWD"
PY="$HOME/Programming/DeFoG/.venv/bin/python"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED}"
echo "commit=$(git log --oneline -1)"
srun "$PY" scripts/fit_credit_head_ce.py --seed "$SEED" \
  --pool "ckpts/credit/cpool_seed${SEED}.pt" \
  --lam 1.0 --iters 8000 --batch-train 16 --lr 1e-4 --readout scaled \
  --base ckpts/zinc_e1_seed42_kek.ckpt \
  --adapter ckpts/clogp_v11/clogp_adapter.ckpt \
  --out "ckpts/credit/credit_head_ce_seed${SEED}.ckpt"

# Gate 2 immediately: it is the measurement rounds 1-3 never moved, and the whole
# reason for this round.
srun "$PY" scripts/gate2_credit.py --head "ckpts/credit/credit_head_ce_seed${SEED}.ckpt" \
  --states 16 --k 256 --chunk 256 --steps 500 --t-int 375 --eta 30 --lam 1.0 \
  --out "ckpts/credit/gate2_ce_seed${SEED}.json"
