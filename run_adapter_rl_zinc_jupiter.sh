#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=06:00:00
#SBATCH --output=adapter_rl_%j.out

# JUPITER: RL-finetune (GDPO) the trained logP + TPSA adapters to tighten their
# conditioning. 4 arms = 2 adapters x 2 KL-coef (the reward-vs-stability knob).
# Base frozen, only the adapter moves; amortized full-range targets; ~4h RL loop
# then pre/post 500-step steering eval.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

mkdir -p experiments/results/adapter_rl_finetune__zinc

# best-per-property adapters from job 966737
LOGP="experiments/results/adapter_training__zinc/17_07_2026__19_32__dBe2/logp_adapter.ckpt"
TPSA="experiments/results/adapter_training__zinc/17_07_2026__19_32__jFD3/tpsa_adapter.ckpt"
PROP=(logp logp tpsa tpsa)
KL=(0.05 0.2 0.05 0.2)
CKPT=("$LOGP" "$LOGP" "$TPSA" "$TPSA")

echo "starting 4-arm adapter RL fine-tune at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_rl_finetune__zinc.py \
      --PROPERTY "'${PROP[$i]}'" --KL_COEF ${KL[$i]} \
      --ADAPTER_CKPT "'${CKPT[$i]}'" \
      --BASE_CKPT "'ckpts/zinc_uncond_4e-4_connectivity.ckpt'" \
      > "adapter_rl_${PROP[$i]}_kl${KL[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm PROP=${PROP[$i]} KL=${KL[$i]} on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all 4 adapter-RL arms finished at $(date)"
