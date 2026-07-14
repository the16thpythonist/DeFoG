#!/bin/bash
#SBATCH --job-name=gmol_seeds
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=26:00:00
#SBATCH --output=guacamol_uncond_%j.out

# 4 GuacaMol (1.59M) unconditional DeFoG replicas on one node (4x RTX 4090),
# one per GPU. Best recipe (batch 24, LR 4e-4); arms differ ONLY by seed.
# Each trains ~22h (Lightning max_time) then evaluates 2500 chunked samples
# (NUV + KDE-KL on logP/QED/SAS/TPSA + gray/foreground plots).

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12

# Pre-create the pycomex namespace dir: avoids the prepare_path TOCTOU race that
# killed an arm in the concurrent zinc uncond sweep (if-not-exists + os.mkdir).
mkdir -p experiments/results/training__guacamol_uncond

SEEDS=(42 43 44 45)
echo "starting 4-seed GuacaMol unconditional run (batch 24, LR 4e-4, max_time 22h) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  S=${SEEDS[$i]}
  CUDA_VISIBLE_DEVICES=$i python -u experiments/training__guacamol_uncond.py \
      --SEED "$S" \
      > "guacamol_seed${S}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm SEED=$S on GPU $i (pid $!)"
  sleep 10
done

wait
echo "all 4 arms finished at $(date)"
