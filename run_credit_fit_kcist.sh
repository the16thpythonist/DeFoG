#!/bin/bash
#SBATCH --job-name=credit_fit
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/credit_fit_%j.out
#
# Gate 1 for the amortised credit head (docs/credit_head_design.md).
#
#   sbatch run_credit_fit_kcist.sh 42
#   sbatch run_credit_fit_kcist.sh 43
#
# Two INDEPENDENT seeds, each building its own endpoint pool, because the gate
# criterion is "beat the baselines by more than the seed-to-seed spread" -- sharing a
# pool would measure fit variance only and leave the data variance unmeasured.
#
# eta=30 / 500 steps: the realistic sampling regime (scripts/churn30.py measured 9.15
# type-changes per atom there against 1.08 at eta=1). The credit head is eta-invariant
# by construction -- it modifies the HEAD, and the sampler rebuilds the rate from it --
# but the endpoint POOL is not, so it is built where it will be deployed.
set -euo pipefail

SEED="${1:-42}"
cd "$HOME/Programming/DeFoG-dam"
mkdir -p logs ckpts/credit
export PYTHONPATH="$PWD"
PY="$HOME/Programming/DeFoG/.venv/bin/python"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED}"
echo "commit=$(git log --oneline -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$PY" scripts/fit_credit_head.py \
  --seed "$SEED" \
  --pool 8192 --batch 256 --steps 500 --eta 30 \
  --lam 1.0 --iters 8000 --batch-train 32 --lr 1e-4 \
  --base ckpts/zinc_e1_seed42_kek.ckpt \
  --adapter ckpts/clogp_v11/clogp_adapter.ckpt \
  --pool-cache "ckpts/credit/pool_seed${SEED}.pt" \
  --out "ckpts/credit/credit_head_seed${SEED}.ckpt"
