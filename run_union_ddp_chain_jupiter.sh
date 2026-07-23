#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:30:00
#SBATCH --output=union_ddp_%j.out

# JUPITER: ONE LINK of the scaled ZINC∪ChEMBL foundation run -- 4-GPU DDP training
# of the SAME 25.9M model (12L/384) on the ~100M union set, to test whether scaling
# data broadens coverage (A/B vs the ChEMBL-only v1). lr=3e-4 (validated), per-rank
# batch 64 = eff 256. Auto-resumes from CKPT_DIR/last.ckpt; PerLinkTimer caps each
# 12h window. Keep EPOCHS (cosine horizon) fixed across links.
#
# ~100M @ ~30h/epoch -> EPOCHS=2 horizon ~= 6-7 links. Submit link-by-link and
# reassess coverage (like the ChEMBL run).
#
# Chain: PREV=""; for i in $(seq 1 N); do
#   if [ -z "$PREV" ]; then PREV=$(sbatch --parsable run_union_ddp_chain_jupiter.sh)
#   else PREV=$(sbatch --parsable --dependency=afterany:$PREV run_union_ddp_chain_jupiter.sh); fi
#   echo "link $i = $PREV"; done

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

DATA_DIR="data/zinc_chembl_union"
PREFIX="union"
LR=3e-4
EPOCHS=2                          # cosine horizon (total planned epochs, fixed across links)
CKPT_DIR="ckpts/foundation_union_lr${LR}"

if [ ! -f "$DATA_DIR/${PREFIX}_train.smiles" ]; then
    echo "ERROR: $DATA_DIR/${PREFIX}_train.smiles missing (stage the union data first)"; exit 1
fi

# KL reference descriptors (25k sample) for the extended eval -- generate once.
if [ ! -f "$DATA_DIR/${PREFIX}_ref_descriptors.npz" ]; then
    echo "precomputing ${PREFIX}_ref_descriptors.npz..."
    python -c "import numpy as np; from experiments.utils import property_distributions; \
s=[l.strip() for l in open('$DATA_DIR/${PREFIX}_train.smiles') if l.strip()]; \
np.savez('$DATA_DIR/${PREFIX}_ref_descriptors.npz', **property_distributions(s, 25000, 42))"
fi

echo "union DDP foundation link @ $(date); CKPT_DIR=$CKPT_DIR"
[ -f "$CKPT_DIR/last.ckpt" ] && echo "  -> resuming from last.ckpt" || echo "  -> fresh start"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

srun python -u scripts/train_chembl_ddp.py \
    --data-dir "$DATA_DIR" --prefix "$PREFIX" \
    --devices 4 --num-nodes 1 --lr ${LR} --epochs ${EPOCHS} \
    --max-time-hours 9.5 --batch-size 64 --num-workers 8 \
    --ckpt-dir "${CKPT_DIR}"

echo "union DDP foundation link finished @ $(date)"
