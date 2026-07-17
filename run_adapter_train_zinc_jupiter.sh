#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=07:00:00
#SBATCH --output=adapter_train_%j.out

# JUPITER (GH200, aarch64), whole node = 4 GPUs. Train frozen-base CFG-adapters
# for logP and TPSA on the connectivity ZINC base, 2 LR arms per property (2 GPUs
# each): logP@2e-4, logP@4e-4, TPSA@2e-4, TPSA@4e-4. From scratch adapter, ~20
# epochs / max_time 5h, then single-property steering eval.

cd ~/Programming/DeFoG
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"     # APPEND (overwriting breaks torch/numpy)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

mkdir -p experiments/results/adapter_training__zinc

# (property, LR) per GPU. pycomex eval()s CLI values -> string PROPERTY needs the
# extra single quotes; numeric LR does not.
ARMS=("logp 2e-4" "logp 4e-4" "tpsa 2e-4" "tpsa 4e-4")
echo "starting 4-arm adapter training at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  set -- ${ARMS[$i]}
  PROP=$1; LR=$2
  CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_training__zinc.py \
      --PROPERTY "'$PROP'" --LEARNING_RATE "$LR" \
      --BASE_CKPT "'ckpts/zinc_uncond_4e-4_connectivity.ckpt'" \
      > "adapter_${PROP}_lr${LR}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm PROPERTY=$PROP LR=$LR on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all 4 adapter arms finished at $(date)"
