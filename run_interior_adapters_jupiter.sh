#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=06:00:00
#SBATCH --output=interior_adapters_%j.out

# JUPITER: train NEW adapters with interior conditioning (L4 pre-FFN + L10
# edge->attention-logit) both ON, on the frozen 9-layer connectivity ZINC base.
# Same recipe as the originals for a clean A/B vs the output-only champions:
#   GPU0 logP  lr2e-4  (vs dBe2)     GPU1 TPSA  lr4e-4  (vs jFD3)
#   GPU2 FP    lr3e-4  (vs 30AK)     GPU3 FP    lr4e-4  (2nd FP LR)

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

BASE="ckpts/zinc_uncond_4e-4_connectivity.ckpt"
echo "starting interior-adapter training at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

CUDA_VISIBLE_DEVICES=0 python -u experiments/adapter_training__zinc.py \
    --PROPERTY "'logp'" --LEARNING_RATE 2e-4 \
    --INTERIOR_FF True --INTERIOR_ATTN True --BASE_CKPT "'$BASE'" \
    > "interior_logp_2e-4_${SLURM_JOB_ID}.out" 2>&1 &
echo "launched logP lr2e-4 on GPU0 (pid $!)"; sleep 8

CUDA_VISIBLE_DEVICES=1 python -u experiments/adapter_training__zinc.py \
    --PROPERTY "'tpsa'" --LEARNING_RATE 4e-4 \
    --INTERIOR_FF True --INTERIOR_ATTN True --BASE_CKPT "'$BASE'" \
    > "interior_tpsa_4e-4_${SLURM_JOB_ID}.out" 2>&1 &
echo "launched TPSA lr4e-4 on GPU1 (pid $!)"; sleep 8

CUDA_VISIBLE_DEVICES=2 python -u experiments/adapter_fingerprint__zinc.py \
    --LEARNING_RATE 3e-4 \
    --INTERIOR_FF True --INTERIOR_ATTN True --BASE_CKPT "'$BASE'" \
    > "interior_fp_3e-4_${SLURM_JOB_ID}.out" 2>&1 &
echo "launched FP lr3e-4 on GPU2 (pid $!)"; sleep 8

CUDA_VISIBLE_DEVICES=3 python -u experiments/adapter_fingerprint__zinc.py \
    --LEARNING_RATE 4e-4 \
    --INTERIOR_FF True --INTERIOR_ATTN True --BASE_CKPT "'$BASE'" \
    > "interior_fp_4e-4_${SLURM_JOB_ID}.out" 2>&1 &
echo "launched FP lr4e-4 on GPU3 (pid $!)"; sleep 8

wait
echo "all interior-adapter arms finished at $(date)"
