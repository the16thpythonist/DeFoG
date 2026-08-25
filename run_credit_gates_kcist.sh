#!/bin/bash
#SBATCH --job-name=credit_gates
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/credit_gates_%j.out
#
# Gates 2 and 3 for a fitted credit head (docs/credit_head_design.md).
#
#   sbatch run_credit_gates_kcist.sh 42
#
# Gate 2 asks whether the head reproduces the per-coordinate instruction that was
# measured empirically; Gate 3 asks whether guided sampling makes better molecules.
# Gate 2 first and in the same job, because a head that fails it cannot pass Gate 3 for
# an honest reason, and Gate 3 is the expensive one.
set -euo pipefail

SEED="${1:-42}"
cd "$HOME/Programming/DeFoG-dam"
mkdir -p logs ckpts/credit
export PYTHONPATH="$PWD"
PY="$HOME/Programming/DeFoG/.venv/bin/python"
HEAD="ckpts/credit/credit_head_seed${SEED}.ckpt"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED} head=${HEAD}"
echo "commit=$(git log --oneline -1)"
test -f "$HEAD" || { echo "MISSING HEAD $HEAD -- fit job did not finish"; exit 1; }

srun "$PY" scripts/gate2_credit.py --head "$HEAD" \
  --states 16 --k 256 --chunk 256 --steps 500 --t-int 375 --eta 30 --lam 1.0 \
  --out "ckpts/credit/gate2_seed${SEED}.json"

# scale=0 is the control and is bit-identical to unguided sampling, so the comparison
# is paired within the same seeds rather than against a separately-run baseline.
srun "$PY" scripts/gate3_credit.py --head "$HEAD" \
  --n 256 --steps 500 --eta 30 --scales 0,0.5,1.0,2.0,4.0 --seeds 1,2,3 \
  --out "ckpts/credit/gate3_seed${SEED}.json"
