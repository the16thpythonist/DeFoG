#!/bin/bash
#SBATCH --job-name=gdpo_regen
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_regen_%j.out

# FINAL regeneration of the 3 unconditional models at the TUNED config, one per GPU.
# Settled by the sweeps:
#   kl_coef=0.2            (Sweep A winner: lowest disc at 500 with validity up)
#   lambda_edge=1          (Sweep B: null, default is best)
#   SAMPLE_STEPS=100       (Sweep C: higher train steps HURT)
#   best-snapshot + eval@500  (keep the trough; compare at the deploy point)
# plus the proven base: GRPO, fixed anchor, reduction=sum, 100 iters, lr=2e-5, ema=0.9, eta=0.
# GuacaMol (max_nodes=88) gets a trimmed K/minibatch.
#
# PREREQUISITE (already uploaded): ~/{aqsoldb,zinc_uncond,guacamol}_pretrained.ckpt.
#   git pull  &&  sbatch experiments/run_gdpo_final_regen_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

mkdir -p experiments/results/gdpo_connectivity   # pycomex prepare_path race guard

NAMES=(aqsoldb zinc guacamol)
CKPTS=("$HOME/aqsoldb_pretrained.ckpt" "$HOME/zinc_uncond_pretrained.ckpt" "$HOME/guacamol_pretrained.ckpt")
KS=(128 128 64)
MBS=(16 16 8)
echo "GDPO FINAL regen (kl=0.2, best-snapshot, eval@500) on ${NAMES[*]} | $(date)"
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
      --CKPT_EVERY 20 --SELECT_BEST True --SELECT_EVAL_STEPS 100 --SELECT_EVAL_SAMPLES 1024 \
      --EVAL_SAMPLES 2048 --EVAL_STEPS 500 --SEED 0 \
      > "gdpo_regen_${NM}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched $NM (K=$K mb=$MB) on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all arms finished at $(date)"
# Results:  grep -H 'SUMMARY' gdpo_regen_*_${SLURM_JOB_ID}.out
# Tuned checkpoints:  experiments/results/gdpo_connectivity/<archive>/gdpo_connected.ckpt
