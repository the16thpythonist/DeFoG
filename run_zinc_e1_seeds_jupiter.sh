#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:45:00
#SBATCH --output=zinc_e1_%j.out

# JUPITER: ONE LINK of the E1 ZINC250k base training -- 4 seeds of the SAME
# protocol recipe, one per GPU. See docs/unconditional-protocol.md.
#
# The four arms differ ONLY in the training seed. SPLIT_SEED is deliberately NOT
# varied: all four must see the identical train/val/test partition, or their
# validation losses are measured on different molecules and are not comparable
# to each other -- and the test split has to stay one fixed set of 24,887.
#
# Each arm trains to EPOCHS=300 (the cosine horizon, fixed across links) and is
# cut by --MAX_TIME_HOURS per link, resuming from its own
# ckpts/zinc_e1_seed<N>/last.ckpt. Training only: no sampling happens here,
# because eta/omega/steps have not been swept on validation yet.
#
# ---- Chain N links (each starts after the previous ENDS) --------------------
#   PREV=""
#   for i in $(seq 1 4); do
#     if [ -z "$PREV" ]; then PREV=$(sbatch --parsable run_zinc_e1_seeds_jupiter.sh)
#     else PREV=$(sbatch --parsable --dependency=afterany:$PREV run_zinc_e1_seeds_jupiter.sh); fi
#     echo "link $i = $PREV"
#   done
# A link that starts after all arms have reached 300 epochs costs a few minutes
# and exits, so over-provisioning the chain is safe.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SEEDS=(42 43 44 45)
MAX_TIME_HOURS=10.5      # of an 11:45 allocation: leaves room for startup + a clean final checkpoint
BATCH_SIZE=256
NUM_WORKERS=8

# The reference data is gitignored, so it is staged rather than cloned. Fetch it
# on the login node before submitting; a compute node may not have egress.
if [ ! -f data/zinc250k/zinc250k.csv ]; then
    echo "ERROR: data/zinc250k/zinc250k.csv missing."
    echo "  Run on the LOGIN node first:"
    echo "    python -c 'from defog.data import zinc_reference; zinc_reference.download_reference()'"
    exit 1
fi

# Bug A from the 978228 post-mortem: 4 concurrent pycomex runs sharing one
# namespace race on os.mkdir(namespace_dir) and the losers die with
# FileExistsError about a minute in. An 8 s stagger was NOT enough. Pre-creating
# the directory removes the race outright.
mkdir -p experiments/results/training__zinc_e1

echo "ZINC250k E1 seed run @ $(date)"
for s in "${SEEDS[@]}"; do
    if [ -f "ckpts/zinc_e1_seed${s}/last.ckpt" ]; then
        echo "  seed ${s}: resuming from ckpts/zinc_e1_seed${s}/last.ckpt"
    else
        echo "  seed ${s}: fresh start"
    fi
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u experiments/training__zinc_e1.py \
        --SEED ${s} \
        --CKPT_DIR "'ckpts/zinc_e1_seed${s}'" \
        --MAX_TIME_HOURS ${MAX_TIME_HOURS} \
        --BATCH_SIZE ${BATCH_SIZE} \
        --NUM_WORKERS ${NUM_WORKERS} \
        --SKIP_FINAL_EVAL True \
        --__DEBUG__ False \
        > "zinc_e1_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "all ZINC E1 arms finished at $(date)"
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "epochs completed|RESUMING|fresh start|new best val" \
        "zinc_e1_seed${s}_${SLURM_JOB_ID}.out" | tail -5
done
