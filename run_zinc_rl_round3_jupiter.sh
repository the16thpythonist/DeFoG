#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=zinc_rl3_%j.out

# JUPITER: ROUND 3 of GDPO sanity RL, ratcheting from the shipped round-2 model.
#
# All four arms start from ckpts/zinc_rl2_seed42/best_model -- the shipped
# round-2 model, measured on test at n=10,000:
#     validity 0.9959 | disconnected 0.0180 | wonky rings 0.0133
#     FCD 1.346 | scaffold 0.5464 | NSPDK 0.00190
# They differ only in RL seed.
#
# Scaffold trajectory on the seed-42 lineage, same test reference throughout:
#     E1 base 0.6035 -> round 1 0.5410 (-0.0625) -> round 2 0.5464 (+0.0054)
# The diversity cost was paid entirely in round 1; round 2 recovered slightly
# while also improving FCD below the base. So the cost looks front-loaded rather
# than per-round -- but that is two data points, and round 3 is the test of it.
# Watch scaffold: if it falls again here, the front-loaded reading was wrong.
#
# Expect small gains. The reward is near saturation (rollout sanity ~0.98) and
# the remaining failure rates are 1.8% disconnected / 1.3% wonky, so there is far
# less headroom than round 1 had. A further 0.002 on validity would be a good
# outcome, not a disappointment.
#
# E1 checkpoints are untouched, as always.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/zinc_rl2_seed42/best_model"
SEEDS=(42 43 44 45)

if [ ! -f "${BASE}.ckpt" ]; then
    echo "ERROR: ${BASE}.ckpt missing -- did round 1 complete?"; exit 1
fi

mkdir -p experiments/results/gdpo_sanity__zinc
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
    CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_sanity__zinc.py \
        --SEED ${s} \
        --BASE_CKPT "'${BASE}'" \
        --OUT_CKPT_DIR "'ckpts/zinc_rl3_seed${s}'" \
        --__DEBUG__ False \
        > "zinc_rl3_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched RL-seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "round 2 finished at $(date)"
OK=0
for s in "${SEEDS[@]}"; do
    grep -qE "saved final-iteration model" "zinc_rl3_seed${s}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that produced a model: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA5 "Traceback" zinc_rl3_seed*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
for s in "${SEEDS[@]}"; do
    echo "--- RL-seed ${s} ---"
    grep -E "loading best checkpoint|^validity \(relaxed\)|^disconnected|^wonky rings|^sanity \(all" \
        "zinc_rl3_seed${s}_${SLURM_JOB_ID}.out" | tail -5
done
