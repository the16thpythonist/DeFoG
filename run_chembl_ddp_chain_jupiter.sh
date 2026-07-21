#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:30:00
#SBATCH --output=chembl_ddp_%j.out

# JUPITER: ONE LINK of the full ChEMBL foundation run -- 4-GPU DDP training of the
# 12L/384 DeFoG on ChEMBL (lr=3e-4, the ablation winner), with resumable
# checkpointing. Auto-resumes from $CKPT_DIR/last.ckpt (restoring optimizer + LR
# schedule + EMA shadow + epoch), trains until --max-time-hours, checkpoints every
# --ckpt-every-n-steps. Keep --epochs (the cosine horizon) FIXED across links so
# the single cosine spans the whole run; --max-time-hours cuts each 12h window.
#
# ---- Chain N links (each starts after the previous ENDS) ----------------------
#   PREV=""
#   for i in $(seq 1 5); do
#     if [ -z "$PREV" ]; then PREV=$(sbatch --parsable run_chembl_ddp_chain_jupiter.sh)
#     else PREV=$(sbatch --parsable --dependency=afterany:$PREV run_chembl_ddp_chain_jupiter.sh); fi
#     echo "link $i = $PREV"
#   done
#
# ---- After the chain: extended eval (single GPU) on the best checkpoint --------
#   python scripts/train_chembl_ddp.py --eval-only \
#       --eval-ckpt ckpts/chembl_foundation_lr3e-4/best_model.ckpt

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

LR=3e-4
EPOCHS=60                        # cosine horizon = total planned epochs (fixed across links)
CKPT_DIR="ckpts/chembl_foundation_lr${LR}"

if [ ! -f data/chembl/chembl_train.smiles ]; then
    echo "ERROR: data/chembl/chembl_train.smiles missing (stage prepared data first)"; exit 1
fi

echo "ChEMBL DDP foundation link @ $(date); CKPT_DIR=$CKPT_DIR"
[ -f "$CKPT_DIR/last.ckpt" ] && echo "  -> resuming from last.ckpt" || echo "  -> fresh start"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

srun python -u scripts/train_chembl_ddp.py \
    --devices 4 --num-nodes 1 --lr ${LR} --epochs ${EPOCHS} \
    --max-time-hours 9.5 --batch-size 256 --num-workers 8 \
    --ckpt-dir "${CKPT_DIR}"

echo "ChEMBL DDP foundation link finished @ $(date)"
