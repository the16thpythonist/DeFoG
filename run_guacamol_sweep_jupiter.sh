#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=guacamol_sweep_%j.out

# JUPITER: the GUACAMOL sampling-parameter sweep (protocol section 5).
# GuacaMol molecules reach 72 heavy atoms, so sampling is markedly slower
# than ZINC's (<=38) -- the edge tensor goes as n^2. Hence the longer walltime.
#
# NOTE the FCD direction: GuacaMol reports a NORMALISED exp(-0.2*FCD) where
# HIGHER is better, opposite to ZINC and MOSES. Scoring emits both forms with
# explicit direction fields; rank on the raw distance as usual (they are
# monotonically related, so the argmin is the same point).

#
# The guacamol base finished its cosine horizon, all four seeds.
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

CKPT="ckpts/guacamol_e1_seed42/best_model"
OUT_DIR="sweep_guacamol_seed42"
NUM_SAMPLES=1000        # enough to rank configs; FCD bias is constant across them

if [ ! -f "${CKPT}.ckpt" ]; then
    echo "ERROR: ${CKPT}.ckpt not found -- has the guacamol chain finished?"
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "GUACAMOL sampling sweep @ $(date); ckpt=$CKPT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# CUDA PREFLIGHT. Job 1137640 landed on jpbo-001-48, where nvidia-smi listed all
# four GH200s but torch._C._cuda_init() failed with "CUDA unknown error" on every
# slice -- a node-level fault, since the identical pattern was running fine on
# another node at the same moment. nvidia-smi succeeding is NOT evidence that
# torch can use the GPU, so check the thing we actually depend on and fail fast
# on a bad node instead of burning the allocation.
python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()   # forces a real context init
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/sweep_sampling.py \
        --ckpt "$CKPT" --dataset guacamol \
        --slice ${i}/4 --num-samples ${NUM_SAMPLES} \
        --out-dir "$OUT_DIR" \
        > "guacamol_sweep_slice${i}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched slice ${i}/4 on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "sweep finished at $(date)"
DONE=$(ls ${OUT_DIR}/*.json 2>/dev/null | wc -l)
echo "grid points completed: ${DONE} / 32"

# `wait` returns 0 even when every background arm died, so without this the job
# reports COMPLETED 0:0 having produced nothing -- which is exactly how job
# 1137640 looked. Make the exit code reflect reality.
if [ "$DONE" -eq 0 ]; then
    echo "ERROR: no grid points produced; arm tracebacks follow"
    grep -hA5 "Traceback" guacamol_sweep_slice*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
