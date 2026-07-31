#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:30:00
#SBATCH --output=zinc_final_%j.out

# JUPITER: the ONE evaluation pass on the ZINC test split (protocol section 5.3).
#
# Everything up to here ran on validation. This is the single pass over the
# sealed 24,887-molecule test set, one seed per GPU, at the configuration frozen
# from the sweep (job 1137654, 32 grid points scored against validation):
#
#     steps=500  eta=25  omega=0        FCD 2.691   validity 0.9930
#
# Chosen on FCD, which is what the ZINC table reports -- NOT on validity, which
# would have picked eta=50/omega=0.05 (validity 0.9950) at 4.5% worse FCD. omega
# worsened FCD in all 16 cells of the 500-step grid, exactly the target-guidance
# overfitting the protocol warns about.
#
# n=10,000 per seed: the ZINC convention, and what
# configs/experiment/zinc.yaml's final_model_samples_to_generate specifies.
#
# If this job has to be re-run for any reason other than an infrastructure
# failure, the protocol says that has to be disclosed.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SEEDS=(42 43 44 45)
STEPS=500
ETA=25
OMEGA=0
NUM_SAMPLES=10000
OUT_DIR="final_zinc_test"
SWEEP_DIR="sweep_zinc_seed42"

for s in "${SEEDS[@]}"; do
    if [ ! -f "ckpts/zinc_e1_seed${s}/best_model.ckpt" ]; then
        echo "ERROR: ckpts/zinc_e1_seed${s}/best_model.ckpt missing"; exit 1
    fi
done

mkdir -p "$OUT_DIR"
echo "ZINC FINAL TEST PASS @ $(date)"
echo "  frozen: steps=${STEPS} eta=${ETA} omega=${OMEGA} n=${NUM_SAMPLES}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# CUDA preflight: job 1137640 died on a node where nvidia-smi listed all four
# GPUs but torch could not init CUDA. Fail fast rather than burn the allocation.
python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

for i in 0 1 2 3; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u scripts/final_eval.py \
        --ckpt "ckpts/zinc_e1_seed${s}/best_model" --dataset zinc \
        --tag "seed${s}" \
        --sample-steps ${STEPS} --eta ${ETA} --omega ${OMEGA} \
        --num-samples ${NUM_SAMPLES} \
        --sweep-dir "${SWEEP_DIR}" \
        --out-dir "$OUT_DIR" \
        > "zinc_final_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "final pass finished at $(date)"

# `wait` returns 0 even when every arm died; make the exit code honest.
DONE=$(ls ${OUT_DIR}/seed*.json 2>/dev/null | wc -l)
echo "seeds completed: ${DONE} / 4"
if [ "$DONE" -lt 4 ]; then
    echo "ERROR: not all seeds produced results; tracebacks follow"
    grep -hA5 "Traceback" zinc_final_seed*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "validity|cumulative|per-valid" "zinc_final_seed${s}_${SLURM_JOB_ID}.out" | tail -6
done
