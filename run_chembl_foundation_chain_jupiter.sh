#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=11:30:00
#SBATCH --output=chembl_foundation_%j.out

# JUPITER: ONE LINK of the full ChEMBL foundation-model training chain.
#
# This resumes the *same* run across 12h windows: it auto-resumes from
# $CKPT_DIR/last.ckpt if present (restoring optimizer + LR schedule + EMA shadow +
# epoch), trains until MAX_TIME_HOURS, and writes last.ckpt every CKPT_EVERY_N_STEPS
# so a mid-epoch cut loses little. Run the SAME cosine horizon (EPOCHS) on every
# link so the single schedule spans the whole run; MAX_TIME_HOURS cuts each job.
#
# ---- Chain N links (each starts after the previous ENDS, success or not) -------
#   PREV=""
#   for i in $(seq 1 4); do
#     if [ -z "$PREV" ]; then
#       PREV=$(sbatch --parsable run_chembl_foundation_chain_jupiter.sh)
#     else
#       PREV=$(sbatch --parsable --dependency=afterany:$PREV run_chembl_foundation_chain_jupiter.sh)
#     fi
#     echo "submitted link $i as $PREV"
#   done
#
# ---- TWO THINGS TO FINALIZE BEFORE THE REAL RUN -------------------------------
#   1. LR: set LEARNING_RATE to the ablation winner (job 996855 eval) - TBD.
#   2. GPUs: this template pins ONE GPU (CUDA_VISIBLE_DEVICES=0) and wastes 3 on a
#      whole-node-exclusive booster node. The full run should use 4-GPU DDP
#      (devices=4, strategy=ddp) for ~4x throughput -- that needs a one-time DDP
#      validation pass (EMA/monitor/ModelCheckpoint under DDP) once the ablation
#      frees a node. Resumable checkpointing (this script's point) works either way.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

LEARNING_RATE=3e-4               # TODO: set to ablation winner
EPOCHS=60                        # cosine horizon = total planned epochs (fixed across links)
MAX_TIME_HOURS=9.5
BATCH_SIZE=256
NUM_WORKERS=8
CKPT_DIR="ckpts/chembl_foundation_lr${LEARNING_RATE}"   # stable across links

# sanity: prepared data must be present (staged via rsync earlier)
if [ ! -f data/chembl/chembl_train.smiles ]; then
    echo "ERROR: data/chembl/chembl_train.smiles missing (stage the prepared data first)"; exit 1
fi

echo "starting ChEMBL foundation link at $(date) (CKPT_DIR=$CKPT_DIR)"
[ -f "$CKPT_DIR/last.ckpt" ] && echo "  -> resuming from $CKPT_DIR/last.ckpt" || echo "  -> fresh start"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

CUDA_VISIBLE_DEVICES=0 python -u experiments/training__chembl_uncond.py \
    --LEARNING_RATE ${LEARNING_RATE} \
    --EPOCHS ${EPOCHS} \
    --MAX_TIME_HOURS ${MAX_TIME_HOURS} \
    --BATCH_SIZE ${BATCH_SIZE} \
    --NUM_WORKERS ${NUM_WORKERS} \
    --CKPT_DIR "'${CKPT_DIR}'" \
    --__DEBUG__ False

echo "ChEMBL foundation link finished at $(date)"
