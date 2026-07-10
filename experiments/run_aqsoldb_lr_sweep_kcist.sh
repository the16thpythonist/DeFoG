#!/bin/bash
# Unconditional AqSolDB training — 4-way learning-rate sweep on ONE KCIST node.
# partition "small" = 4x RTX 4090 per node; runs all 4 trainings in parallel,
# one per GPU, each a different --LEARNING_RATE, non-debug (permanent archive).
#
#   sbatch experiments/run_aqsoldb_lr_sweep_kcist.sh
#
#SBATCH --job-name=aqsoldb_lr_sweep
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=aqsoldb_lr_sweep_%j.out
#SBATCH --error=aqsoldb_lr_sweep_%j.out

cd "$HOME/Programming/DeFoG" || exit 1
source .venv/bin/activate || exit 1
# experiments/ is not an installed package; put the repo root on the path.
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

LRS=(5e-5 1e-4 2e-4 4e-4)

echo "host $(hostname) | job $SLURM_JOB_ID | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n_gpus', torch.cuda.device_count())"

pids=()
for i in 0 1 2 3; do
    lr=${LRS[$i]}
    log="aqsoldb_lr_${lr}_${SLURM_JOB_ID}.out"
    echo "launching LR=$lr on local GPU $i -> $log"
    CUDA_VISIBLE_DEVICES=$i python experiments/training__aqsoldb_uncond.py \
        --__DEBUG__ False --LEARNING_RATE "$lr" > "$log" 2>&1 &
    pids+=($!)
done

# Wait for all four; report each arm's exit status (don't let one failure abort the rest).
status=0
for idx in 0 1 2 3; do
    if wait "${pids[$idx]}"; then
        echo "LR=${LRS[$idx]} (GPU $idx) finished OK"
    else
        echo "LR=${LRS[$idx]} (GPU $idx) FAILED (exit $?)"
        status=1
    fi
done
echo "sweep done (status=$status)"
exit $status
