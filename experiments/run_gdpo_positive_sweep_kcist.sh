#!/bin/bash
#SBATCH --job-name=gdpo_pos_sweep
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_pos_sweep_%j.out

# GDPO connectivity -- COLLAPSE-PROOF recipe, KL_COEF sweep (4x RTX 4090, one per GPU).
# Best-guess stack to push past the ~6% floor without atom-soup collapse:
#   POSITIVE_ONLY=True  (RAFT: never push down bad endpoints -> no atom-soup)
#   ADVANTAGE_MODE=mean (gradient fades as reward saturates)
#   KL_ANCHOR=moving    (EMA-of-policy trust region -> drift past the pretrained floor)
# from the PRETRAINED ZINC model, 150 iters. We sweep the one value that sets how hard
# it can push -- the trust-region strength KL_COEF in {0.3, 0.1, 0.03, 0.0}. The 0.0
# arm is positive-only ALONE (no KL). Watch unique_frac: positive-only trades the
# atom-soup collapse for a possible DIVERSITY collapse, which uniqueness detects.
#
# PREREQUISITE: pretrained ZINC checkpoint on KCIST (already uploaded as
#   ~/zinc_uncond_pretrained.ckpt by the earlier from-pretrained sweep). Then:
#   git pull  &&  sbatch experiments/run_gdpo_positive_sweep_kcist.sh
# Override base ckpt with:  CKPT=/path/to/ckpt sbatch ...

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CKPT="${CKPT:-$HOME/zinc_uncond_pretrained.ckpt}"
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint '$CKPT' not found."; exit 1; fi

# pycomex prepare_path race guard: create the namespace dir ONCE before the parallel
# launch, so concurrent arms don't race on mkdir (FileExistsError).
mkdir -p experiments/results/gdpo_connectivity

KLS=(0.3 0.1 0.03 0.0)
echo "GDPO positive-only + moving-anchor KL_COEF sweep = ${KLS[*]} | 150 iters | base $CKPT | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  KL=${KLS[$i]}
  # NOTE: pycomex evals each --PARAM value as a Python expression, so STRING values
  # must be quoted Python literals ("'mean'"); numbers/bools pass through as-is.
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_connectivity.py \
      --__DEBUG__ False \
      --CKPT_PATH "'$CKPT'" \
      --KL_COEF "$KL" \
      --POSITIVE_ONLY True \
      --ADVANTAGE_MODE "'mean'" \
      --KL_ANCHOR "'moving'" --ANCHOR_DECAY 0.99 \
      --ROLLOUT_SIZE 128 --ITERATIONS 150 --MINIBATCH_SIZE 16 \
      --SAMPLE_STEPS 100 --SUBSAMPLE_STEPS 12 --ETA 0.0 --REDUCTION "'sum'" \
      --LR 2e-5 --EMA_DECAY 0.9 --EVAL_SAMPLES 2048 --EVAL_STEPS 100 \
      --CKPT_EVERY 25 --SEED 0 \
      > "gdpo_pos_KL${KL}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm KL_COEF=$KL on GPU $i (pid $!)"
  sleep 8   # stagger so the first arm creates the archive tree before the others start
done

wait
echo "all 4 arms finished at $(date)"
# Compare:  grep -H 'SUMMARY' gdpo_pos_KL*_${SLURM_JOB_ID}.out
# Each arm's log records its pycomex archive path under experiments/results/gdpo_connectivity/
