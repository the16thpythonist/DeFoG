#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:30:00
#SBATCH --output=zinc_rl_shiptest_%j.out

# JUPITER: THE test read for the shipped RL model.
#
# Model: ckpts/zinc_rl2_seed42/best_model -- GDPO sanity RL, two rounds from the
# ZINC E1 base. Selected ENTIRELY on validation: round-2 seeds were compared at
# n=10,000 against the 5,000-molecule validation reference, and seed42 was the
# only one of five improving on round 1 across both the targeted metrics AND
# the distribution metrics (FCD -0.097, scaffold +0.002, NSPDK -0.0004).
#
# Sampling config is the E1 frozen one (steps=500, eta=25, omega=0), inherited
# from the base model's validation sweep and never re-tuned for the RL model.
#
# TEST-READ LEDGER for ZINC, to be disclosed in the manuscript:
#   1. E1 base, 4 seeds        (job 1137783)
#   2. RL round 1, 4 seeds     (job 1141314)
#   3. this run, 1 model       (the shipped RL artifact)
# No model was ever SELECTED using a test number; all selection used validation.
#
# Four seeds are sampled from the SAME checkpoint with different sampling seeds,
# so the reported figure carries a sampling-noise band rather than being a single
# draw. That is a different quantity from the E1 row's spread, which is across
# four independently TRAINED models -- the two error bars are not comparable and
# must not be presented as if they were.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

CKPT="ckpts/zinc_rl2_seed42/best_model"
STEPS=500; ETA=25; OMEGA=0; N=10000
OUT_DIR="shiptest_zinc_rl"

[ -f "${CKPT}.ckpt" ] || { echo "ERROR: ${CKPT}.ckpt missing"; exit 1; }

mkdir -p "$OUT_DIR"
echo "ZINC RL SHIP TEST @ $(date); ckpt=$CKPT"
echo "  frozen: steps=${STEPS} eta=${ETA} omega=${OMEGA} n=${N} x 4 sampling seeds"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/final_eval.py \
        --ckpt "$CKPT" --dataset zinc --tag "ship_draw${i}" --split test \
        --sample-steps ${STEPS} --eta ${ETA} --omega ${OMEGA} --num-samples ${N} \
        --sweep-dir "sweep_zinc_seed42 (E1 base sweep; RL model never re-swept)" \
        --out-dir "$OUT_DIR" \
        > "zinc_rl_shiptest_draw${i}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched draw ${i} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"
DONE=$(ls ${OUT_DIR}/ship_draw*.json 2>/dev/null | wc -l)
echo "draws completed: ${DONE} / 4"
if [ "$DONE" -lt 4 ]; then
    echo "ERROR: not all draws produced results; tracebacks follow"
    grep -hA5 "Traceback" zinc_rl_shiptest_draw*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
for i in 0 1 2 3; do
    echo "--- draw ${i} ---"
    grep -E "validity|cumulative|per-valid" "zinc_rl_shiptest_draw${i}_${SLURM_JOB_ID}.out" | tail -6
done
