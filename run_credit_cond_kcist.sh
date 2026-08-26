#!/bin/bash
#SBATCH --job-name=credit_cond
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/credit_cond_%j.out
#
# Round 3: fit the credit head against the ACTUAL conditional expectation.
#
#   sbatch run_credit_cond_kcist.sh 42
#
# Rounds 1 and 2 re-noised ONE endpoint per state, so at the noise levels that matter
# x_t nearly determines its endpoint and the head could satisfy the loss by
# reconstructing it -- a per-state value function with no coordinate content. That fits
# every observation: Gate 1 passed, Gate 2 read as a shuffled null, Gate 3 was flat.
#
# Here each state carries K completions simulated from it, so the same x_t maps to K
# different rewards and the reconstruct-the-endpoint shortcut does not fit.
set -euo pipefail
SEED="${1:-42}"
cd "$HOME/Programming/DeFoG-dam"
mkdir -p logs ckpts/credit
export PYTHONPATH="$PWD"
PY="$HOME/Programming/DeFoG/.venv/bin/python"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED}"
echo "commit=$(git log --oneline -1)"
srun "$PY" scripts/fit_credit_head_cond.py --seed "$SEED" \
  --states 1024 --completions 8 --batch 32 --chunk 256 \
  --steps 500 --eta 30 --lam 1.0 --t-bias 1.6 \
  --iters 8000 --batch-train 32 --lr 1e-4 --readout scaled \
  --base ckpts/zinc_e1_seed42_kek.ckpt \
  --adapter ckpts/clogp_v11/clogp_adapter.ckpt \
  --pool-cache "ckpts/credit/cpool_seed${SEED}.pt" \
  --out "ckpts/credit/credit_head_cond_seed${SEED}.ckpt"
