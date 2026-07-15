#!/bin/bash
#SBATCH --job-name=gdpo_ssteps
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_ssteps_%j.out

# SWEEP C: TRAIN-TIME rollout SAMPLE_STEPS. We found the model generates far cleaner
# molecules at more denoising steps; this trains the ROLLOUTS at more steps too, so the
# endpoints being reinforced are higher-quality/more-connected (a cleaner gradient) and
# match the higher-step deploy point. The COMPARED eval stays fixed at 500 steps, so
# this isolates the training-step effect from the eval-step effect.
#   arms:  SAMPLE_STEPS in {100, 200, 300, 400}   (100 = the tuned baseline)
# From PRETRAINED ZINC, kl_coef=0.2 (Sweep A winner), lambda_edge=1 (Sweep B: default
# is best), fixed anchor + GRPO + best-snapshot.
#
# PREREQUISITE: ~/zinc_uncond_pretrained.ckpt on KCIST. Then:
#   git pull  &&  sbatch experiments/run_gdpo_samplesteps_sweep_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CKPT="${CKPT:-$HOME/zinc_uncond_pretrained.ckpt}"
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint '$CKPT' not found."; exit 1; fi
mkdir -p experiments/results/gdpo_connectivity   # pycomex prepare_path race guard

SS=(100 200 300 400)
echo "GDPO train-time sample_steps sweep = ${SS[*]} | kl=0.2 lambda_edge=1 + best-snapshot | eval@500 | base $CKPT | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  S=${SS[$i]}
  # pycomex evals --PARAM values -> quote STRING literals; numbers/bools pass through.
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_connectivity.py \
      --__DEBUG__ False \
      --CKPT_PATH "'$CKPT'" \
      --SAMPLE_STEPS "$S" \
      --KL_COEF 0.2 --KL_ANCHOR "'fixed'" --ADVANTAGE_MODE "'grpo'" --REDUCTION "'sum'" \
      --ROUNDS 1 --ITERATIONS 100 --ROLLOUT_SIZE 128 --MINIBATCH_SIZE 16 \
      --SUBSAMPLE_STEPS 12 --ETA 0.0 --LR 2e-5 --EMA_DECAY 0.9 \
      --CKPT_EVERY 20 --SELECT_BEST True --SELECT_EVAL_STEPS 100 --SELECT_EVAL_SAMPLES 1024 \
      --EVAL_SAMPLES 2048 --EVAL_STEPS 500 --SEED 0 \
      > "gdpo_ssteps_S${S}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm SAMPLE_STEPS=$S on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all 4 arms finished at $(date)"
# Compare:  grep -H 'SUMMARY' gdpo_ssteps_S*_${SLURM_JOB_ID}.out
