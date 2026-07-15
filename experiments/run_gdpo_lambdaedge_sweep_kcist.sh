#!/bin/bash
#SBATCH --job-name=gdpo_ledge
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_ledge_%j.out

# SWEEP B: LAMBDA_EDGE -- weight of the bond term in the eager gradient. Connectivity
# is a pure bond property, so up-weighting edges aims the gradient exactly where the
# reward's information lives (more connectivity per unit of allowed drift), at the risk
# of valence errors -> watch validity. Orthogonal to kl_coef, so run at the swept-best
#   arms:  LAMBDA_EDGE in {1, 2, 3, 4}   (1 = baseline == the kl=0.2 arm from Sweep A)
# From the PRETRAINED ZINC model, kl_coef=0.2 (Sweep A winner), fixed anchor + GRPO +
# best-snapshot, compared at 500 steps.
#
# PREREQUISITE: ~/zinc_uncond_pretrained.ckpt on KCIST. Then:
#   git pull  &&  sbatch experiments/run_gdpo_lambdaedge_sweep_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CKPT="${CKPT:-$HOME/zinc_uncond_pretrained.ckpt}"
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint '$CKPT' not found."; exit 1; fi
mkdir -p experiments/results/gdpo_connectivity   # pycomex prepare_path race guard

LES=(1 2 3 4)
echo "GDPO lambda_edge sweep = ${LES[*]} | kl=0.2 fixed anchor + best-snapshot | eval@500 | base $CKPT | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  LE=${LES[$i]}
  # pycomex evals --PARAM values -> quote STRING literals; numbers/bools pass through.
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_connectivity.py \
      --__DEBUG__ False \
      --CKPT_PATH "'$CKPT'" \
      --LAMBDA_EDGE "$LE" \
      --KL_COEF 0.2 --KL_ANCHOR "'fixed'" --ADVANTAGE_MODE "'grpo'" --REDUCTION "'sum'" \
      --ROUNDS 1 --ITERATIONS 100 --ROLLOUT_SIZE 128 --MINIBATCH_SIZE 16 \
      --SAMPLE_STEPS 100 --SUBSAMPLE_STEPS 12 --ETA 0.0 --LR 2e-5 --EMA_DECAY 0.9 \
      --CKPT_EVERY 20 --SELECT_BEST True --SELECT_EVAL_STEPS 100 --SELECT_EVAL_SAMPLES 1024 \
      --EVAL_SAMPLES 2048 --EVAL_STEPS 500 --SEED 0 \
      > "gdpo_ledge_LE${LE}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm LAMBDA_EDGE=$LE on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all 4 arms finished at $(date)"
# Compare:  grep -H 'SUMMARY' gdpo_ledge_LE*_${SLURM_JOB_ID}.out
