#!/bin/bash
# Unconditional AqSolDB training -- 4-way BATCH-SIZE sweep on ONE JUPITER node.
# JUPITER (JSC) "booster" partition = 4x GH200 (H100 96GB) per node, ARM aarch64,
# whole-node exclusive. Runs all 4 trainings in parallel, one per GPU, each a
# different --BATCH_SIZE, with LR fixed at 5e-5 and seed 42, non-debug.
#
#   sbatch experiments/run_aqsoldb_bs_sweep_jupiter.sh
#
#SBATCH --job-name=aqsoldb_bs_sweep
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=aqsoldb_bs_sweep_%j.out
#SBATCH --error=aqsoldb_bs_sweep_%j.out

cd "$HOME/Programming/DeFoG" || exit 1

# aarch64 environment: JSC module-provided PyTorch + a venv for the remaining deps.
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
# APPEND, never overwrite -- the PyTorch module puts torch/numpy on PYTHONPATH,
# and we also need the repo root so `import experiments` / `import defog` resolve.
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

BATCHES=(12 16 24 32)
echo "host $(hostname) | job $SLURM_JOB_ID | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n_gpus', torch.cuda.device_count())"

pids=()
for i in 0 1 2 3; do
    bs=${BATCHES[$i]}
    log="aqsoldb_bs${bs}_${SLURM_JOB_ID}.out"
    echo "launching BATCH_SIZE=$bs (LR 5e-5, seed 42) on local GPU $i -> $log"
    CUDA_VISIBLE_DEVICES=$i python -u experiments/training__aqsoldb_uncond.py \
        --__DEBUG__ False --LEARNING_RATE 5e-5 --BATCH_SIZE "$bs" > "$log" 2>&1 &
    pids+=($!)
done

# Wait for all four; report each arm's exit status without aborting the others.
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
