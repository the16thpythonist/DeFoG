#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=zinc_rl_%j.out

# JUPITER: GDPO sanity RL on the four ZINC E1 bases, one seed per GPU.
#
# Targets, from the E1 final test pass (n=10,000 x 4 seeds):
#   validity 0.9854 +- 0.0049 | disconnected 0.0615 +- 0.0219 | wonky rings 0.0356 +- 0.0129
#
# Each arm starts from its OWN E1 checkpoint, giving a paired before/after per
# seed and testing whether RL rescues the weak seeds (43-45, 6-9% disconnected)
# as well as the strong one (42, 2.7%).
#
#   GPU0  e1_seed42 -> rl_seed42     GPU2  e1_seed44 -> rl_seed44
#   GPU1  e1_seed43 -> rl_seed43     GPU3  e1_seed45 -> rl_seed45
#
# The E1 checkpoints are inputs only. Output goes to ckpts/zinc_rl_seed<N>/, so
# the E1 table row stays reproducible from untouched weights.
#
# Reward is graded (valid + connected + rings_ok, 0-3) rather than a single AND:
# 95% of samples already satisfy the AND, which would leave almost no
# group-relative advantage to learn from.
#
# kl_coef=0.3 pulls toward the frozen base. That is the primary guard against
# raising sanity by collapsing the distribution -- and each arm dumps before/after
# SMILES so FCD and NSPDK can be scored externally. A run whose sanity improves
# while FCD degrades is a FAILED run.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SEEDS=(42 43 44 45)

for s in "${SEEDS[@]}"; do
    if [ ! -f "ckpts/zinc_e1_seed${s}/best_model.ckpt" ]; then
        echo "ERROR: ckpts/zinc_e1_seed${s}/best_model.ckpt missing"; exit 1
    fi
done

mkdir -p experiments/results/gdpo_sanity__zinc
echo "ZINC GDPO sanity RL @ $(date)"
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
    CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_sanity__zinc.py \
        --SEED ${s} \
        --BASE_CKPT "'ckpts/zinc_e1_seed${s}/best_model'" \
        --OUT_CKPT_DIR "'ckpts/zinc_rl_seed${s}'" \
        --__DEBUG__ False \
        > "zinc_rl_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "RL finished at $(date)"

# `wait` returns 0 even when every arm died; make the exit code honest.
OK=0
for s in "${SEEDS[@]}"; do
    grep -q "saved RL model" "zinc_rl_seed${s}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that produced a model: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA5 "Traceback" zinc_rl_seed*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "BEFORE|validity \(relaxed\)|disconnected|wonky rings|sanity \(all" \
        "zinc_rl_seed${s}_${SLURM_JOB_ID}.out" | tail -6
done
