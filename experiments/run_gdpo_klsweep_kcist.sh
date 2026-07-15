#!/bin/bash
#SBATCH --job-name=gdpo_klsweep
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_klsweep_%j.out

# SWEEP A: kl_coef with the GOOD (fixed-anchor, GRPO) recipe + best-snapshot selection.
# We only ever ran kl_coef=0.3 with the fixed anchor (the sweep that collapsed used the
# MOVING anchor). With a fixed anchor -- which always pulls back to healthy pretrained --
# moderately lower kl_coef may be stable and sit at a lower floor. Lower KL collapses
# SOONER, so SELECT_BEST picks each arm's best PRE-COLLAPSE snapshot, making the arms
# comparable at their trough rather than their (possibly overshot) final weights.
#   arms:  kl_coef in {0.15, 0.2, 0.25, 0.3}   (one per GPU)
# From the PRETRAINED ZINC model. COMPARED eval (BEFORE/AFTER) at 500 steps -- the deploy
# point where the fine-tuned models produce the fewest disconnected fragments.
#
# PREREQUISITE: ~/zinc_uncond_pretrained.ckpt on KCIST (already there). Then:
#   git pull  &&  sbatch experiments/run_gdpo_klsweep_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CKPT="${CKPT:-$HOME/zinc_uncond_pretrained.ckpt}"
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint '$CKPT' not found."; exit 1; fi
mkdir -p experiments/results/gdpo_connectivity   # pycomex prepare_path race guard

KLS=(0.15 0.2 0.25 0.3)
echo "GDPO kl_coef sweep (fixed anchor + best-snapshot) = ${KLS[*]} | 100 iters | eval@500 | base $CKPT | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  KL=${KLS[$i]}
  # pycomex evals --PARAM values -> quote STRING literals; numbers/bools pass through.
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_connectivity.py \
      --__DEBUG__ False \
      --CKPT_PATH "'$CKPT'" \
      --KL_COEF "$KL" \
      --KL_ANCHOR "'fixed'" --ADVANTAGE_MODE "'grpo'" --REDUCTION "'sum'" \
      --ROUNDS 1 --ITERATIONS 100 --ROLLOUT_SIZE 128 --MINIBATCH_SIZE 16 \
      --SAMPLE_STEPS 100 --SUBSAMPLE_STEPS 12 --ETA 0.0 --LR 2e-5 --EMA_DECAY 0.9 \
      --CKPT_EVERY 20 --SELECT_BEST True --SELECT_EVAL_STEPS 100 --SELECT_EVAL_SAMPLES 1024 \
      --EVAL_SAMPLES 2048 --EVAL_STEPS 500 --SEED 0 \
      > "gdpo_klsweep_KL${KL}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm KL_COEF=$KL on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all 4 arms finished at $(date)"
# Compare:  grep -H 'SUMMARY' gdpo_klsweep_KL*_${SLURM_JOB_ID}.out
