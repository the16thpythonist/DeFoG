#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --output=guacamol_probe_%j.out

# JUPITER: GuacaMol throughput/memory probe -- four batch sizes, one per GPU.
#
# JUPITER is whole-node exclusive, so a serial sweep would waste three GPUs for
# the same allocation. Running the four candidate batch sizes concurrently costs
# the same node-minutes and returns the whole curve at once.
#
# An arm that OOMs reports it and exits cleanly rather than taking the job down,
# which is the point: the largest batch that FITS is exactly what we are looking
# for, so hitting the ceiling is a result, not a failure.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BATCHES=(64 128 256 512)
SUBSET=50000

if [ ! -f data/guacamol/guacamol_v1_train.smiles ]; then
    echo "ERROR: GuacaMol official split missing. On the LOGIN node run:"
    echo "  python -c 'from defog.data import guacamol_reference as g; g.download_reference()'"
    exit 1
fi

echo "GuacaMol throughput probe @ $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
    b=${BATCHES[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u scripts/probe_throughput.py \
        --dataset guacamol --batch-size ${b} --subset ${SUBSET} \
        --out "guacamol_probe_bs${b}_${SLURM_JOB_ID}.json" \
        > "guacamol_probe_bs${b}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched batch_size=${b} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "probe finished at $(date)"
echo "================= RESULTS ================="
for b in "${BATCHES[@]}"; do
    f="guacamol_probe_bs${b}_${SLURM_JOB_ID}.json"
    if [ -f "$f" ]; then cat "$f"; else
        echo "batch ${b}: NO JSON -- tail of log:"
        tail -5 "guacamol_probe_bs${b}_${SLURM_JOB_ID}.out"
    fi
done
