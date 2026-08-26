#!/bin/bash
#SBATCH --job-name=credit_tilt
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/credit_tilt_%j.out
#
# Round 5: target the reward TILT only, and test a reward we proved carries signal.
#
#   sbatch run_credit_tilt_kcist.sh 42
#
# Two changes from round 4, both free (the round-3 pools are reused unchanged):
#
#  --base-mode emp  p* and p_emp come from the SAME completions, so the base model's
#                   miscalibration cancels exactly and only the reward tilt remains.
#                   Round 4 used the model's marginals, bundling the two, and the
#                   lambda=0 control showed the whole gain was calibration.
#
#  --reward oxy     the pool stores COMPLETIONS, so a different reward costs nothing.
#                   oxy-max was measured at 8.7x the directional signal of logp-match
#                   at matched tilt strength (coherence 1.000 vs 0.577). If Gate 2
#                   lifts for oxygen and not logP, the boundary is DEMONSTRATED rather
#                   than inferred; if neither lifts, amortisation fails even in the
#                   easy case, which is a stronger claim than four logP failures.
set -euo pipefail
SEED="${1:-42}"
cd "$HOME/Programming/DeFoG-dam"
mkdir -p logs ckpts/credit
export PYTHONPATH="$PWD"
PY="$HOME/Programming/DeFoG/.venv/bin/python"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED}"
echo "commit=$(git log --oneline -1)"

for R in logp oxy; do
  echo "##### reward=$R #####"
  H="ckpts/credit/credit_head_tilt_${R}_seed${SEED}.ckpt"
  srun "$PY" scripts/fit_credit_head_ce.py --seed "$SEED" \
    --pool "ckpts/credit/cpool_seed${SEED}.pt" \
    --reward "$R" --base-mode emp \
    --lam 1.0 --iters 8000 --batch-train 16 --lr 1e-4 --readout scaled \
    --base ckpts/zinc_e1_seed42_kek.ckpt \
    --adapter ckpts/clogp_v11/clogp_adapter.ckpt --out "$H"
  srun "$PY" scripts/gate2_credit.py --head "$H" --reward "$R" \
    --states 16 --k 256 --chunk 256 --steps 500 --t-int 375 --eta 30 --lam 1.0 \
    --out "ckpts/credit/gate2_tilt_${R}_seed${SEED}.json"
done
