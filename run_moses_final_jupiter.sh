#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=moses_final_%j.out

# JUPITER: the ONE evaluation pass on the MOSES test split (protocol 5.3).
#
# Everything up to here ran on validation. Config frozen from the MOSES sweep
# (job 1206380, 32 grid points scored against validation):
#
#     steps=500  eta=25  omega=0        FCD 2.098   validity 0.8880
#
# Identical to ZINC's optimum, independently arrived at. Chosen on FCD, NOT on
# validity: validity would have picked eta=0/omega=0.25 (0.9100 vs 0.8880) at
# FCD 3.228 -- 54% worse. That divergence is an order of magnitude larger than
# ZINC's, which is the target-guidance overfitting warning showing up hardest on
# small drug-like molecules.
#
# n=25,000 per seed: the MOSES convention, matching
# configs/experiment/moses.yaml's final_model_samples_to_generate.
#
# final_eval.py also writes the test_scaffolds reference for this dataset. MOSES
# reports FCD against BOTH held-out sets, and DeFoG's published row uses the
# TestSF variants -- Scaf/Test would overstate by ~6x. Score with
# scripts/e1_metrics.py --dataset moses --reference-scaffolds <that file>.
#
# One-shot: a re-run for anything but infrastructure failure must be disclosed.

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
NUM_SAMPLES=25000
OUT_DIR="final_moses_test"
SWEEP_DIR="sweep_moses_seed42"

for s in "${SEEDS[@]}"; do
    if [ ! -f "ckpts/moses_e1_seed${s}/best_model.ckpt" ]; then
        echo "ERROR: ckpts/moses_e1_seed${s}/best_model.ckpt missing"; exit 1
    fi
done

mkdir -p "$OUT_DIR"
echo "MOSES FINAL TEST PASS @ $(date)"
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
        --ckpt "ckpts/moses_e1_seed${s}/best_model" --dataset moses \
        --tag "seed${s}" \
        --sample-steps ${STEPS} --eta ${ETA} --omega ${OMEGA} \
        --num-samples ${NUM_SAMPLES} \
        --sweep-dir "${SWEEP_DIR}" \
        --out-dir "$OUT_DIR" \
        > "moses_final_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
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
    grep -hA5 "Traceback" moses_final_seed*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "validity|cumulative|per-valid" "moses_final_seed${s}_${SLURM_JOB_ID}.out" | tail -6
done
