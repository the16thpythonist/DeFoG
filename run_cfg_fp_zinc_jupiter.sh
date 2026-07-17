#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=07:00:00
#SBATCH --output=cfg_fp_zinc_%j.out

# JUPITER (GH200, ARM aarch64), whole node = 4 GPUs. Pure-CFG fingerprint model
# on ZINC 250k, one LR per GPU (1e-4/2e-4/3e-4/4e-4). From scratch, ~20 epochs /
# max_time 5h, then a held-out Tanimoto-lift eval with a guidance-scale sweep.

cd ~/Programming/DeFoG

# aarch64: use the JSC PyTorch module (do NOT build torch); run module load BARE.
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"     # APPEND (overwriting breaks torch/numpy from the module)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

# Pre-create the pycomex namespace dir: dodges the prepare_path TOCTOU race
# under 4 concurrent same-namespace arms.
mkdir -p experiments/results/cfg_fingerprint__zinc

LRS=(1e-4 2e-4 3e-4 4e-4)
echo "starting 4-arm CFG-fingerprint sweep at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  LR=${LRS[$i]}
  CUDA_VISIBLE_DEVICES=$i python -u experiments/cfg_fingerprint__zinc.py \
      --LEARNING_RATE "$LR" \
      > "cfg_fp_lr${LR}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm LR=$LR on GPU $i (pid $!)"
  sleep 10
done

wait
echo "all 4 CFG arms finished at $(date)"
