#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --output=moses_probe_%j.out

# JUPITER: MOSES throughput/memory probe -- four batch sizes, one per GPU.
#
# The candidate batches are much larger than GuacaMol's because MOSES molecules
# are much smaller: 8-27 heavy atoms against GuacaMol's 72. Dense batches pad to
# the largest molecule and the edge tensor goes as n^2, so per-graph edge cost is
# roughly (27/72)^2 ~ 0.14x. Whether that translates into ~7x the batch is
# exactly what this measures rather than assumes -- the GuacaMol probe already
# showed the obvious extrapolation (batch 256, by analogy with ZINC) was wrong.
#
# An arm that OOMs records it and exits cleanly rather than taking the job down;
# finding the ceiling is the point.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCHES=(256 512 1024 2048)
SUBSET=50000
LINK_HOURS=10.0

if [ ! -f data/moses/train.csv ]; then
    echo "ERROR: MOSES split missing. On the LOGIN node run:"
    echo "  python -c 'from defog.data import moses_reference as m; m.download_reference()'"
    exit 1
fi

echo "MOSES throughput probe @ $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
    b=${BATCHES[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u scripts/probe_throughput.py \
        --dataset moses --batch-size ${b} --subset ${SUBSET} \
        --link-hours ${LINK_HOURS} \
        --out "moses_probe_bs${b}_${SLURM_JOB_ID}.json" \
        > "moses_probe_bs${b}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched batch_size=${b} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "probe finished at $(date)"
echo "================= RESULTS ================="
for b in "${BATCHES[@]}"; do
    f="moses_probe_bs${b}_${SLURM_JOB_ID}.json"
    if [ -f "$f" ]; then cat "$f"; else
        echo "batch ${b}: NO JSON -- tail of log:"
        tail -5 "moses_probe_bs${b}_${SLURM_JOB_ID}.out"
    fi
done
