#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --output=chembl_ddp_smoke_%j.out

# JUPITER GPU-DDP CONFIRMATION (short): confirm the 4-GPU srun+nccl launch works
# (SLURM rank/GPU binding, no nccl hang, real data sharding) on a tiny config
# before committing to the multi-day chain. Trains ~1 epoch on 5000 molecules.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "DDP smoke @ $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

srun python -u scripts/train_chembl_ddp.py \
    --devices 4 --num-nodes 1 --lr 3e-4 --epochs 1 \
    --max-train 5000 --max-val 500 --batch-size 128 --num-workers 8 \
    --gen-every-k 1 --gen-sample-steps 50 --sample-vis-every-k 1 \
    --ckpt-dir "ckpts/chembl_ddp_smoke"

echo "DDP smoke finished @ $(date)"
