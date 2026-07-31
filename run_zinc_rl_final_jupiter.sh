#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:30:00
#SBATCH --output=zinc_rl_final_%j.out

# JUPITER: full n=10,000 evaluation of the RL-tuned ZINC models (job 1140572).
#
# Same frozen sampling configuration as the E1 test pass -- steps=500, eta=25,
# omega=0 -- and the same n, so each RL seed is directly comparable to its OWN
# E1 row rather than to a differently-sampled number. NO sweep is run: the
# configuration is inherited from the base model's validation sweep, not re-tuned
# for the RL model, which is what keeps the comparison paired.
#
# This is a SECOND read of the ZINC test split, for a DIFFERENT model. No tuning
# happened on test in between -- the sampling config came from the base's
# validation sweep and was not touched. That is legitimate, but it is a second
# read and has to be disclosed as such.
#
# The E1 checkpoints remain untouched; this reads ckpts/zinc_rl_seed<N>/best_model,
# the best-selected checkpoint (iterations 15/7/19/24 respectively).

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
OUT_DIR="final_zinc_rl_test"

for s in "${SEEDS[@]}"; do
    if [ ! -f "ckpts/zinc_rl_seed${s}/best_model.ckpt" ]; then
        echo "ERROR: ckpts/zinc_rl_seed${s}/best_model.ckpt missing"; exit 1
    fi
done

mkdir -p "$OUT_DIR"
echo "ZINC RL FINAL EVAL @ $(date)"
echo "  frozen: steps=${STEPS} eta=${ETA} omega=${OMEGA} n=${NUM_SAMPLES}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

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
        --ckpt "ckpts/zinc_rl_seed${s}/best_model" --dataset zinc \
        --tag "rl_seed${s}" \
        --sample-steps ${STEPS} --eta ${ETA} --omega ${OMEGA} \
        --num-samples ${NUM_SAMPLES} \
        --sweep-dir "sweep_zinc_seed42 (inherited from the E1 base; NOT re-swept)" \
        --out-dir "$OUT_DIR" \
        > "zinc_rl_final_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched RL seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"
DONE=$(ls ${OUT_DIR}/rl_seed*.json 2>/dev/null | wc -l)
echo "seeds completed: ${DONE} / 4"
if [ "$DONE" -lt 4 ]; then
    echo "ERROR: not all seeds produced results; tracebacks follow"
    grep -hA5 "Traceback" zinc_rl_final_seed*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
for s in "${SEEDS[@]}"; do
    echo "--- RL seed ${s} ---"
    grep -E "validity|cumulative|per-valid" "zinc_rl_final_seed${s}_${SLURM_JOB_ID}.out" | tail -6
done
