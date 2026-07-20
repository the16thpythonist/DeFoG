#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=11:30:00
#SBATCH --output=chembl_uncond_sweep_%j.out

# JUPITER: first ChEMBL foundation-model run -- an unconditional 12L/384/12-head
# DeFoG on the 2.44M cleaned ChEMBL set, as a 4-point LEARNING-RATE ablation
# (one LR per GPU) at fixed capacity. Selection metric = the extended eval suite
# (validity / uniqueness / novelty / connected / sanity / logP-TPSA-QED KL).
#   GPU0 lr=1e-4   GPU1 lr=2e-4   GPU2 lr=3e-4   GPU3 lr=4e-4
# Training is wall-clock-capped (MAX_TIME_HOURS) so end-of-run eval finishes
# inside the 12h SLURM limit.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

MAX_TIME_HOURS=9.5      # train cap; ~2h left for eval within the 11:30 walltime
BATCH_SIZE=256
NUM_WORKERS=8
LRS=(1e-4 2e-4 3e-4 4e-4)

# --- Data: prepare the cleaned ChEMBL splits on the compute node if missing ---
RAW="data/chembl/raw/chembl_37_chemreps.txt.gz"
if [ ! -f data/chembl/chembl_train.smiles ]; then
    if [ ! -f "$RAW" ]; then
        echo "ERROR: $RAW is missing. Download it on the login node first:"
        echo "  curl -sSL -o $RAW https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_chemreps.txt.gz"
        exit 1
    fi
    echo "preparing ChEMBL dataset (compute node)..."
    python scripts/prepare_chembl.py
fi

# --- Precompute the KL reference descriptors ONCE (avoid a 4-arm write race) ---
if [ ! -f data/chembl/chembl_ref_descriptors.npz ]; then
    echo "precomputing logP/TPSA/QED reference descriptors..."
    python -c "import numpy as np; from experiments.utils import property_distributions; \
s=[l.strip() for l in open('data/chembl/chembl_train.smiles') if l.strip()]; \
np.savez('data/chembl/chembl_ref_descriptors.npz', **property_distributions(s, 25000, 42))"
fi

echo "starting ChEMBL unconditional LR sweep at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u experiments/training__chembl_uncond.py \
        --LEARNING_RATE ${LRS[$i]} \
        --MAX_TIME_HOURS ${MAX_TIME_HOURS} \
        --BATCH_SIZE ${BATCH_SIZE} \
        --NUM_WORKERS ${NUM_WORKERS} \
        --__DEBUG__ False \
        > "chembl_uncond_lr${LRS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched lr=${LRS[$i]} on GPU $i (pid $!)"
    sleep 8
done

wait
echo "all ChEMBL unconditional arms finished at $(date)"
