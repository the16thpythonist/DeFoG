#!/bin/bash
#SBATCH --job-name=gdpo_reanchor
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_reanchor_%j.out

# GDPO connectivity -- ROUND-BASED RE-ANCHORING (4x RTX 4090, one config per GPU).
# The evidence-backed path to push below the ~6% floor: each round fine-tunes with a
# FIXED strong anchor (kl_coef=0.3, GRPO) to the round's starting weights (stable, no
# collapse), then the improved model becomes the next round's base AND anchor -- the
# floor ratchets down between rounds. (A continuous moving anchor collapsed; discrete
# re-anchoring is the controllable version, and round-1->round-2 already went 6.8->6.0.)
#
# Fixed TOTAL budget = ROUNDS x ITERATIONS = 300; sweep the RE-ANCHORING FREQUENCY:
#   ITERATIONS/ROUNDS = 30/10, 50/6, 75/4, 150/2   (150/2 ~= the manual round-1->round-2)
# Question: does re-anchoring more often ratchet lower / more stably at equal compute?
# From the PRETRAINED ZINC model.
#
# PREREQUISITE: ~/zinc_uncond_pretrained.ckpt on KCIST (already uploaded). Then:
#   git pull  &&  sbatch experiments/run_gdpo_reanchor_sweep_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CKPT="${CKPT:-$HOME/zinc_uncond_pretrained.ckpt}"
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint '$CKPT' not found."; exit 1; fi

# pycomex prepare_path race guard (create the namespace dir once before the parallel launch)
mkdir -p experiments/results/gdpo_connectivity

ITERS=(30 50 75 150)
RNDS=(10 6 4 2)
echo "GDPO re-anchoring sweep ITERATIONS/ROUNDS = 30/10,50/6,75/4,150/2 (300 total) | base $CKPT | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  IT=${ITERS[$i]}; RN=${RNDS[$i]}
  # pycomex evals --PARAM values as Python expressions -> quote STRING literals.
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_connectivity.py \
      --__DEBUG__ False \
      --CKPT_PATH "'$CKPT'" \
      --ROUNDS "$RN" --ITERATIONS "$IT" \
      --KL_COEF 0.3 --KL_ANCHOR "'fixed'" --ADVANTAGE_MODE "'grpo'" --REDUCTION "'sum'" \
      --ROLLOUT_SIZE 128 --MINIBATCH_SIZE 16 \
      --SAMPLE_STEPS 100 --SUBSAMPLE_STEPS 12 --ETA 0.0 \
      --LR 2e-5 --EMA_DECAY 0.9 \
      --EVAL_SAMPLES 2048 --ROUND_EVAL_SAMPLES 512 --EVAL_STEPS 100 \
      --CKPT_EVERY 25 --SEED 0 \
      > "gdpo_reanchor_R${RN}x${IT}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm ROUNDS=$RN x ITERATIONS=$IT on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all 4 arms finished at $(date)"
# Compare the ratchets:  grep -H 'SUMMARY' gdpo_reanchor_R*_${SLURM_JOB_ID}.out
