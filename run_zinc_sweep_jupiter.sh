#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=zinc_sweep_%j.out

# JUPITER: the ZINC sampling-parameter sweep (protocol section 5).
#
# The ZINC base finished its 300-epoch horizon (job 1125452, all four seeds).
# Before any number can go in a table, steps/eta/omega have to be chosen on the
# VALIDATION split and frozen -- the rate matrix is assembled at sampling time,
# so these are free parameters and tuning them on test would be exactly the
# benchmark overfitting the protocol forbids.
#
# 32 grid points (steps x eta x omega) split across the 4 GPUs, 8 each. Only
# ONE seed is swept: the sampling configuration is a property of the sampler,
# not of the seed, and sweeping all four would quadruple cost to answer the same
# question. Seed 42 is used, and the frozen config is then applied to all four
# seeds for the final test pass.
#
# This job only SAMPLES and writes SMILES. Scoring happens afterwards in the
# metrics env (FCD/NSPDK/scaffold are x86-only), which also makes the sweep
# re-scorable without re-sampling.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

CKPT="ckpts/zinc_e1_seed42/best_model"
OUT_DIR="sweep_zinc_seed42"
NUM_SAMPLES=1000        # enough to rank configs; FCD bias is constant across them

if [ ! -f "${CKPT}.ckpt" ]; then
    echo "ERROR: ${CKPT}.ckpt not found -- has the ZINC chain finished?"
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "ZINC sampling sweep @ $(date); ckpt=$CKPT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/sweep_sampling.py \
        --ckpt "$CKPT" --dataset zinc \
        --slice ${i}/4 --num-samples ${NUM_SAMPLES} \
        --out-dir "$OUT_DIR" \
        > "zinc_sweep_slice${i}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched slice ${i}/4 on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "sweep finished at $(date)"
echo "grid points completed: $(ls ${OUT_DIR}/*.json 2>/dev/null | wc -l) / 32"
