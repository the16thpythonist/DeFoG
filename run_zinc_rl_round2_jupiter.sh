#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=zinc_rl2_%j.out

# JUPITER: ROUND 2 of GDPO sanity RL, ratcheting from the best round-1 model.
#
# All four arms start from ckpts/zinc_rl_seed42/best_model -- the strongest
# model produced so far, on every metric, at n=10,000 on test:
#     validity 0.9954 | disconnected 0.0233 | wonky rings 0.0140 | FCD 1.4333
# They differ only in RL seed, so this measures round-2 variance from a single
# starting point rather than re-running four independent lineages.
#
# The KL reference is rebuilt from the round-1 model (GDPOTrainer freezes
# whatever policy it is handed), so the pull is toward round 1, not back toward
# the original E1 base. That is what makes this a ratchet rather than a reset.
#
# Round 1 moved validity +0.0045 and disconnected -0.0063 with FCD slightly
# improved, but scaffold similarity fell 0.018 on 3 of 4 seeds. That is the
# number to watch here: a small diversity cost per round could COMPOUND across
# rounds even while each round looks individually acceptable.
#
# E1 checkpoints are untouched, as always.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/zinc_rl_seed42/best_model"
SEEDS=(42 43 44 45)

if [ ! -f "${BASE}.ckpt" ]; then
    echo "ERROR: ${BASE}.ckpt missing -- did round 1 complete?"; exit 1
fi

mkdir -p experiments/results/gdpo_sanity
echo "ZINC GDPO sanity RL -- ROUND 2 from ${BASE} @ $(date)"
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
        --DATASET "'zinc'" \
        --SEED ${s} \
        --BASE_CKPT "'${BASE}'" \
        --OUT_CKPT_DIR "'ckpts/zinc_rl2_seed${s}'" \
        --__DEBUG__ False \
        > "zinc_rl2_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched RL-seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "round 2 finished at $(date)"
OK=0
for s in "${SEEDS[@]}"; do
    grep -qE "saved final-iteration model" "zinc_rl2_seed${s}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that produced a model: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA5 "Traceback" zinc_rl2_seed*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
for s in "${SEEDS[@]}"; do
    echo "--- RL-seed ${s} ---"
    grep -E "loading best checkpoint|^validity \(relaxed\)|^disconnected|^wonky rings|^sanity \(all" \
        "zinc_rl2_seed${s}_${SLURM_JOB_ID}.out" | tail -5
done
