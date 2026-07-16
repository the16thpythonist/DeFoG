#!/bin/bash
#SBATCH --job-name=gdpo_round2
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_round2_%j.out

# ROUND 2: continue fine-tuning FROM the tuned round-1 checkpoints (kl=0.2 result),
# one model per GPU, to squeeze a bit more. Each round-2 run fixed-anchors to its own
# round-1 weights (re-anchoring) at the tuned config. SELECT_INCLUDE_INPUT=True adds the
# round-1 checkpoint itself as a best-snapshot candidate, so round 2 CANNOT ship a model
# worse than round 1 -- worst case it keeps round 1. Diminishing returns expected.
#   config: kl=0.2, lambda_edge=1, SAMPLE_STEPS=100, GRPO, fixed anchor, 100 iters,
#           best-snapshot (input-inclusive), eval@500.
#
# PREREQUISITE (already copied on KCIST): ~/{aqsoldb,zinc,guacamol}_round1.ckpt.
#   git pull  &&  sbatch experiments/run_gdpo_round2_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

mkdir -p experiments/results/gdpo_connectivity   # pycomex prepare_path race guard

NAMES=(aqsoldb zinc guacamol)
CKPTS=("$HOME/aqsoldb_round1.ckpt" "$HOME/zinc_round1.ckpt" "$HOME/guacamol_round1.ckpt")
KS=(128 128 64)
MBS=(16 16 8)
echo "GDPO ROUND 2 (continue from round-1, kl=0.2, input-safe best-snapshot, eval@500) on ${NAMES[*]} | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2; do
  NM=${NAMES[$i]}; CK=${CKPTS[$i]}; K=${KS[$i]}; MB=${MBS[$i]}
  if [ ! -f "$CK" ]; then echo "SKIP $NM: $CK not found"; continue; fi
  # pycomex evals --PARAM values -> quote STRING literals; numbers/bools pass through.
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_connectivity.py \
      --__DEBUG__ False \
      --CKPT_PATH "'$CK'" \
      --KL_COEF 0.2 --KL_ANCHOR "'fixed'" --ADVANTAGE_MODE "'grpo'" --REDUCTION "'sum'" \
      --LAMBDA_EDGE 1 --ROUNDS 1 --ITERATIONS 100 --SAMPLE_STEPS 100 --SUBSAMPLE_STEPS 12 \
      --ROLLOUT_SIZE "$K" --MINIBATCH_SIZE "$MB" --ETA 0.0 --LR 2e-5 --EMA_DECAY 0.9 \
      --CKPT_EVERY 20 --SELECT_BEST True --SELECT_INCLUDE_INPUT True \
      --SELECT_EVAL_STEPS 100 --SELECT_EVAL_SAMPLES 1024 \
      --EVAL_SAMPLES 2048 --EVAL_STEPS 500 --SEED 1 \
      > "gdpo_round2_${NM}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched $NM (K=$K mb=$MB) on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all arms finished at $(date)"
# Results:  grep -H 'SUMMARY' gdpo_round2_*_${SLURM_JOB_ID}.out
