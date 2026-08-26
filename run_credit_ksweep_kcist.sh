#!/bin/bash
#SBATCH --job-name=credit_ksweep
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/credit_ksweep_%j.out
#
#   sbatch run_credit_ksweep_kcist.sh 42
#
# The K dose-response, on IDENTICAL states (--use-k subsets the same pool), for both
# rewards. Registered prediction (docs/credit_head_design.md 6e):
#
#   oxygen  target reliability ~0.43 at K=64 -> Gate 1 and Gate 2 should move if
#           per-coordinate credit amortises at all
#   logP    no reliability trend up to K=24  -> expected to stay flat
#
# Gate 1 is cheap so every K gets it; Gate 2 costs ~25 min so only K=64 gets it, which
# is the only cell the prediction is about.
set -euo pipefail
SEED="${1:-42}"
cd "$HOME/Programming/DeFoG-dam"
mkdir -p logs ckpts/credit
export PYTHONPATH="$PWD"
PY="$HOME/Programming/DeFoG/.venv/bin/python"
POOL="ckpts/credit/cpool_seed${SEED}_k64.pt"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED}"
echo "commit=$(git log --oneline -1)"
test -f "$POOL" || { echo "MISSING POOL $POOL"; exit 1; }

for R in oxy logp; do
  for K in 8 16 32 64; do
    echo "##### reward=$R K=$K #####"
    H="ckpts/credit/ch_${R}_k${K}_seed${SEED}.ckpt"
    srun "$PY" scripts/fit_credit_head_ce.py --seed "$SEED" --pool "$POOL" \
      --use-k "$K" --reward "$R" --base-mode emp \
      --lam 1.0 --iters 8000 --batch-train 16 --lr 1e-4 --readout scaled \
      --base ckpts/zinc_e1_seed42_kek.ckpt \
      --adapter ckpts/clogp_v11/clogp_adapter.ckpt --out "$H"
  done
  echo "##### Gate 2, reward=$R, K=64 #####"
  srun "$PY" scripts/gate2_credit.py --head "ckpts/credit/ch_${R}_k64_seed${SEED}.ckpt" \
    --reward "$R" --states 16 --k 256 --chunk 256 --steps 500 --t-int 375 --eta 30 \
    --lam 1.0 --out "ckpts/credit/gate2_${R}_k64_seed${SEED}.json"
done
