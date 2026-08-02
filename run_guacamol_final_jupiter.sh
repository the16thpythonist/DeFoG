#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=guacamol_final_%j.out

# JUPITER: the ONE evaluation pass on the GuacaMol test split (protocol 5.3).
#
# Config frozen from the validation sweep (job 1206379) PLUS a boundary check
# (job 1207239), and the boundary check changed the answer.
#
#     steps=500  eta=75  omega=0
#
# The main sweep's apparent winner was eta=50/omega=0.05 at FCD 2.6342 -- but
# eta=50 was the grid EDGE, so the extension re-measured all four eta=50 cells
# as a noise control. That cell came back at 2.7185, a swing of +0.084, and the
# four controls gave mean |diff| 0.042. So FCD gaps below ~0.09 at n=1000 are
# not meaningful and the original winner was a lucky draw, not an optimum.
#
# Averaged over the four omega values (which halves the noise), eta=75 scores
# 2.644 and eta=100 2.658 against eta=50's 2.721 -- so the optimum really was
# outside the original grid. eta=75 is preferred over eta=100: indistinguishable
# on the mean, but far better behaved per-cell (spread 0.09 against 0.31).
#
# omega=0 because omega shows no consistent benefit at any eta in either
# measurement. The single lowest cell anywhere is eta=100/omega=0.05 at 2.470;
# choosing it would be selecting the argmin of noisy measurements, which is the
# exact error the control just exposed.
#
# eta=75 is higher than ZINC's and MOSES's 25 -- GuacaMol molecules reach 72
# heavy atoms against 38 and 27, so there is more structure to get wrong and
# correspondingly more benefit from stochastic error correction.
#
# n=18,000: the GuacaMol convention from configs/experiment/guacamol.yaml.
#
# NOTE the FCD direction when reporting: GuacaMol uses the NORMALISED
# exp(-0.2*FCD) in [0,1] where HIGHER is better, opposite to ZINC and MOSES.
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
ETA=75
OMEGA=0
NUM_SAMPLES=18000
OUT_DIR="final_guacamol_test"
SWEEP_DIR="sweep_guacamol_seed42 + _ext (boundary check)"

for s in "${SEEDS[@]}"; do
    if [ ! -f "ckpts/guacamol_e1_seed${s}/best_model.ckpt" ]; then
        echo "ERROR: ckpts/guacamol_e1_seed${s}/best_model.ckpt missing"; exit 1
    fi
done

mkdir -p "$OUT_DIR"
echo "GUACAMOL FINAL TEST PASS @ $(date)"
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
        --ckpt "ckpts/guacamol_e1_seed${s}/best_model" --dataset guacamol \
        --tag "seed${s}" \
        --sample-steps ${STEPS} --eta ${ETA} --omega ${OMEGA} \
        --num-samples ${NUM_SAMPLES} \
        --sweep-dir "${SWEEP_DIR}" \
        --out-dir "$OUT_DIR" \
        > "guacamol_final_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
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
    grep -hA5 "Traceback" guacamol_final_seed*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "validity|cumulative|per-valid" "guacamol_final_seed${s}_${SLURM_JOB_ID}.out" | tail -6
done
