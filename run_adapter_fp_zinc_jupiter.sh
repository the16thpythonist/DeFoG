#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=10:00:00
#SBATCH --output=adapter_fp_%j.out

# JUPITER (GH200, aarch64), whole node = 4 GPUs. Train a frozen-base AdaLN CFG-
# adapter conditioned on a 512-bit Morgan fingerprint, on the connectivity ZINC
# base. 4 LR arms (1 per GPU): 1e-4/2e-4/3e-4/4e-4. ~8h train (max_time) then a
# 500-step Tanimoto-lift eval (w-sweep) + per-target grids.

cd "$SLURM_SUBMIT_DIR"    # repo under $PROJECT (submit from the repo dir)
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"     # APPEND (overwriting breaks module torch/numpy)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

mkdir -p experiments/results/adapter_fingerprint__zinc

LRS=(1e-4 2e-4 3e-4 4e-4)
echo "starting 4-arm fingerprint-adapter run at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  LR=${LRS[$i]}
  CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_fingerprint__zinc.py \
      --LEARNING_RATE "$LR" \
      --BASE_CKPT "'ckpts/zinc_uncond_4e-4_connectivity.ckpt'" \
      > "adapter_fp_lr${LR}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm LR=$LR on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all 4 fingerprint-adapter arms finished at $(date)"
