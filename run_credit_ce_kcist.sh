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
LAM="${2:-1.0}"
TAG="${3:-ce}"
cd "$HOME/Programming/DeFoG-dam"
mkdir -p logs ckpts/credit
export PYTHONPATH="$PWD"
PY="$HOME/Programming/DeFoG/.venv/bin/python"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED} lam=${LAM} tag=${TAG}"
echo "commit=$(git log --oneline -1)"
srun "$PY" scripts/fit_credit_head_ce.py --seed "$SEED" \
  --pool "ckpts/credit/cpool_seed${SEED}.pt" \
  --lam "$LAM" --iters 8000 --batch-train 16 --lr 1e-4 --readout scaled \
  --base ckpts/zinc_e1_seed42_kek.ckpt \
  --adapter ckpts/clogp_v11/clogp_adapter.ckpt \
  --out "ckpts/credit/credit_head_${TAG}_seed${SEED}.ckpt"

# Gate 2 immediately: it is the measurement rounds 1-3 never moved, and the whole
# reason for this round.
if [ "$LAM" = "0" ] || [ "$LAM" = "0.0" ]; then
  echo "lambda=0 control: no reward content, Gate 2 not meaningful"
  echo "GATE2-DONE"; exit 0
fi
srun "$PY" scripts/gate2_credit.py --head "ckpts/credit/credit_head_${TAG}_seed${SEED}.ckpt" \
  --states 16 --k 256 --chunk 256 --steps 500 --t-int 375 --eta 30 --lam 1.0 \
  --out "ckpts/credit/gate2_${TAG}_seed${SEED}.json"
