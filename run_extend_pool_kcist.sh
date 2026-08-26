#!/bin/bash
#SBATCH --job-name=extend_pool
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/extend_pool_%j.out
#
#   sbatch run_extend_pool_kcist.sh 42 64
#
# Round 5 regressed a target whose split-half reliability is ~0.00 at K=6-8 -- two
# independent halves of the completions give UNCORRELATED tilt targets (logp +0.002,
# oxy +0.019). Every conclusion drawn from it was about noise, not amortisation.
#
# Extends the SAME states so K = 8, 16, 32, 64 can be compared on identical data. A
# dose-response in K is the convincing form: if Gate 1 and Gate 2 rise as the target's
# reliability rises, the earlier failures were sample size; if they stay flat while
# reliability climbs, amortisation is dead for a real reason.
set -euo pipefail
SEED="${1:-42}"; TK="${2:-64}"
cd "$HOME/Programming/DeFoG-dam"
mkdir -p logs ckpts/credit
export PYTHONPATH="$PWD"
PY="$HOME/Programming/DeFoG/.venv/bin/python"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED} target_k=${TK}"
echo "commit=$(git log --oneline -1)"
srun "$PY" scripts/extend_pool.py \
  --pool "ckpts/credit/cpool_seed${SEED}.pt" --target-k "$TK" \
  --batch 32 --chunk 256 --steps 500 --eta 30 \
  --base ckpts/zinc_e1_seed42_kek.ckpt \
  --adapter ckpts/clogp_v11/clogp_adapter.ckpt \
  --out "ckpts/credit/cpool_seed${SEED}_k${TK}.pt"
