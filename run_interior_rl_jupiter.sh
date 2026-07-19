#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=08:00:00
#SBATCH --output=interior_rl_%j.out

# JUPITER: RL-finetune (GDPO, connectivity-FIRST reward) the NEW interior (L10+L4)
# adapters. Reward makes conditioning tighter AND penalizes disconnected/invalid.
#   GPU0 logP  (interior qY8v, lr2e-4)     GPU1 TPSA  (interior NXp2, lr4e-4)
#   GPU2 FP    (interior lsm8, lr4e-4)      GPU3 FP    (interior bljh, lr3e-4)
# kl_coef=0.2 (the better setting from the prior RL run), 4h RL + pre/post eval.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

BASE="ckpts/zinc_uncond_4e-4_connectivity.ckpt"
AT="experiments/results/adapter_training__zinc"
AF="experiments/results/adapter_fingerprint__zinc"
LOGP="$AT/18_07_2026__20_00__qY8v/logp_adapter.ckpt"
TPSA="$AT/18_07_2026__20_00__NXp2/tpsa_adapter.ckpt"
FP_LSM8="$AF/18_07_2026__20_00__lsm8/fp_adapter.ckpt"
FP_BLJH="$AF/18_07_2026__20_00__bljh/fp_adapter.ckpt"

echo "starting interior-adapter RL (connectivity-first) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

CUDA_VISIBLE_DEVICES=0 python -u experiments/adapter_rl_finetune__zinc.py \
    --PROPERTY "'logp'" --KL_COEF 0.2 --ADAPTER_CKPT "'$LOGP'" --BASE_CKPT "'$BASE'" \
    > "interior_rl_logp_${SLURM_JOB_ID}.out" 2>&1 &
echo "launched RL logP on GPU0 (pid $!)"; sleep 8

CUDA_VISIBLE_DEVICES=1 python -u experiments/adapter_rl_finetune__zinc.py \
    --PROPERTY "'tpsa'" --KL_COEF 0.2 --ADAPTER_CKPT "'$TPSA'" --BASE_CKPT "'$BASE'" \
    > "interior_rl_tpsa_${SLURM_JOB_ID}.out" 2>&1 &
echo "launched RL TPSA on GPU1 (pid $!)"; sleep 8

CUDA_VISIBLE_DEVICES=2 python -u experiments/adapter_rl_finetune_fp__zinc.py \
    --KL_COEF 0.2 --ADAPTER_CKPT "'$FP_LSM8'" --BASE_CKPT "'$BASE'" \
    > "interior_rl_fp_lsm8_${SLURM_JOB_ID}.out" 2>&1 &
echo "launched RL FP lsm8 on GPU2 (pid $!)"; sleep 8

CUDA_VISIBLE_DEVICES=3 python -u experiments/adapter_rl_finetune_fp__zinc.py \
    --KL_COEF 0.2 --ADAPTER_CKPT "'$FP_BLJH'" --BASE_CKPT "'$BASE'" \
    > "interior_rl_fp_bljh_${SLURM_JOB_ID}.out" 2>&1 &
echo "launched RL FP bljh on GPU3 (pid $!)"; sleep 8

wait
echo "all interior-adapter RL arms finished at $(date)"
