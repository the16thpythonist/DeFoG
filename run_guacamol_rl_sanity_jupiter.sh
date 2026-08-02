#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=guacamol_rl_%j.out

# JUPITER: GDPO sanity RL on the four ZINC E1 bases, one seed per GPU.
#
# Baseline from the GuacaMol E1 test pass (4 seeds, n=18,000):
#   validity 0.9799 | disconnected 0.0420 | wonky rings 0.0355 | sanity 0.9452
# A ZINC-like profile: all three failure modes present, ~5.5% headroom.
#
# EVAL_ETA=75 matches GuacaMol's OWN frozen deploy config, not ZINC's 25.
# Evaluating at the wrong eta would measure a policy nobody will run.
#
# Each arm starts from its OWN E1 checkpoint, giving a paired before/after per
# seed. Output goes to ckpts/DS_rl_seed<N>/; E1 checkpoints are inputs only, so
# the E1 table rows stay reproducible from untouched weights.
#
# Reward is graded (valid + connected + rings_ok, 0-3) rather than a single AND.
# kl_coef=0.3 guards against raising sanity by collapsing the distribution, and
# each arm dumps before/after SMILES so FCD can be checked externally.
#
# ITERATIONS=25 with CKPT_EVERY/SELECT_BEST: on ZINC the first attempt ran 60
# iterations, peaked near 20 and collapsed to sanity 0.14 by 59, and having only
# a final-model save made the good policy unrecoverable. Best-checkpoint
# selection is the real protection; the iteration count is secondary.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SEEDS=(42 43 44 45)

for s in "${SEEDS[@]}"; do
    if [ ! -f "ckpts/guacamol_e1_seed${s}/best_model.ckpt" ]; then
        echo "ERROR: ckpts/guacamol_e1_seed${s}/best_model.ckpt missing"; exit 1
    fi
done

mkdir -p experiments/results/gdpo_sanity
echo "GUACAMOL GDPO sanity RL @ $(date)"
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
    CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_sanity.py \
        --DATASET "'guacamol'" \
        --SEED ${s} \
        --EVAL_ETA 75.0 \
        --BASE_CKPT "'ckpts/guacamol_e1_seed${s}/best_model'" \
        --OUT_CKPT_DIR "'ckpts/guacamol_rl_seed${s}'" \
        --__DEBUG__ False \
        > "guacamol_rl_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "RL finished at $(date)"

# `wait` returns 0 even when every arm died; make the exit code honest.
OK=0
for s in "${SEEDS[@]}"; do
    grep -qE "saved final-iteration model|saved RL model" "guacamol_rl_seed${s}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that produced a model: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA5 "Traceback" guacamol_rl_seed*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "BEFORE|validity \(relaxed\)|disconnected|wonky rings|sanity \(all" \
        "guacamol_rl_seed${s}_${SLURM_JOB_ID}.out" | tail -6
done
