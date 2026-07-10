#!/bin/bash
# Unconditional AqSolDB training -- 4-way BATCH-SIZE sweep on ONE KCIST node.
# partition "small" = 4x RTX 4090 per node. Runs all 4 trainings in parallel, one
# per GPU, each a different --BATCH_SIZE, with LR fixed at 4e-4 (the winning value),
# seed 42, non-debug.
#
#   sbatch experiments/run_aqsoldb_bs_sweep_kcist.sh
#
#SBATCH --job-name=aqsoldb_bs_sweep
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=aqsoldb_bs_sweep_%j.out
#SBATCH --error=aqsoldb_bs_sweep_%j.out

cd "$HOME/Programming/DeFoG" || exit 1
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

BATCHES=(12 16 24 32)
echo "host $(hostname) | job $SLURM_JOB_ID | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n_gpus', torch.cuda.device_count())"

pids=()
for i in 0 1 2 3; do
    bs=${BATCHES[$i]}
    log="aqsoldb_bs${bs}_${SLURM_JOB_ID}.out"
    echo "launching BATCH_SIZE=$bs (LR 4e-4, seed 42) on local GPU $i -> $log"
    CUDA_VISIBLE_DEVICES=$i python -u experiments/training__aqsoldb_uncond.py \
        --__DEBUG__ False --LEARNING_RATE 4e-4 --BATCH_SIZE "$bs" > "$log" 2>&1 &
    pids+=($!)
done

status=0
for idx in 0 1 2 3; do
    if wait "${pids[$idx]}"; then
        echo "BATCH_SIZE=${BATCHES[$idx]} (GPU $idx) finished OK"
    else
        echo "BATCH_SIZE=${BATCHES[$idx]} (GPU $idx) FAILED (exit $?)"
        status=1
    fi
done
echo "sweep done (status=$status)"
exit $status
