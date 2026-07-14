#!/bin/bash
#SBATCH --job-name=zinc_lr_sweep
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=zinc_lr_sweep_%j.out

# 4-way ZINC 250k conditional LR sweep on one node (4x RTX 4090), one arm/GPU.
# Same seed + data + batch 24 / 20 epochs; only the learning rate differs.
# Each arm writes its own log (which records its pycomex archive path) so the
# LR -> archive mapping is recoverable.

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

LRS=(2e-4 3e-4 4e-4 5e-4)
echo "starting 4-way ZINC LR sweep (batch 24, 20 epochs) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  LR=${LRS[$i]}
  CUDA_VISIBLE_DEVICES=$i python -u experiments/conditional_training__zinc.py \
      --LEARNING_RATE "$LR" \
      > "zinc_lr_${LR}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm LR=$LR on GPU $i (pid $!)"
  sleep 5
done

wait
echo "all 4 arms finished at $(date)"
