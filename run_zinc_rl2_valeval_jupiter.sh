#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=zinc_rl2_valeval_%j.out

# JUPITER: n=10,000 evaluation of RL round 2 -- on VALIDATION, not test.
#
# This run feeds a SELECTION decision (does round 2 beat round 1, and which
# seed), and selecting a model by its test numbers is tuning on test however
# defensible each individual pass looks. The ZINC test split has already been
# read twice (E1 base, RL round 1); it is reserved from here on for reporting a
# model that has already been chosen.
#
# The round-1 model is re-evaluated here TOO, on the same validation reference
# at the same n. Its existing numbers are on TEST against a 24,887-molecule
# reference, which is not comparable to validation's 5,000 -- FCD in particular
# is strongly reference-size dependent. Comparing round 2 to those would be
# comparing across reference sets, so round 1 is re-measured like-for-like.
#
# 5 models over 4 GPUs: GPU0 runs the round-1 baseline and then rl2_seed42.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

STEPS=500; ETA=25; OMEGA=0; N=10000
OUT_DIR="valeval_zinc_rl2"

for c in ckpts/zinc_rl_seed42/best_model ckpts/zinc_rl2_seed42/best_model \
         ckpts/zinc_rl2_seed43/best_model ckpts/zinc_rl2_seed44/best_model \
         ckpts/zinc_rl2_seed45/best_model; do
    [ -f "${c}.ckpt" ] || { echo "ERROR: ${c}.ckpt missing"; exit 1; }
done

mkdir -p "$OUT_DIR"
echo "ZINC RL round-2 VALIDATION eval @ $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

run_one () {  # $1=gpu  $2=ckpt  $3=tag
    CUDA_VISIBLE_DEVICES=$1 python -u scripts/final_eval.py \
        --ckpt "$2" --dataset zinc --tag "$3" --split validation \
        --sample-steps ${STEPS} --eta ${ETA} --omega ${OMEGA} \
        --num-samples ${N} \
        --sweep-dir "sweep_zinc_seed42 (inherited; NOT re-swept)" \
        --out-dir "$OUT_DIR" > "zinc_rl2_valeval_$3_${SLURM_JOB_ID}.out" 2>&1
}

# GPU0 takes two models sequentially; the others take one each.
( run_one 0 ckpts/zinc_rl_seed42/best_model  round1_seed42
  run_one 0 ckpts/zinc_rl2_seed42/best_model round2_seed42 ) &
run_one 1 ckpts/zinc_rl2_seed43/best_model round2_seed43 &
run_one 2 ckpts/zinc_rl2_seed44/best_model round2_seed44 &
run_one 3 ckpts/zinc_rl2_seed45/best_model round2_seed45 &
echo "launched 5 evaluations across 4 GPUs"

wait
echo "finished at $(date)"
DONE=$(ls ${OUT_DIR}/round*.json 2>/dev/null | wc -l)
echo "models evaluated: ${DONE} / 5"
if [ "$DONE" -lt 5 ]; then
    echo "ERROR: not all models evaluated; tracebacks follow"
    grep -hA5 "Traceback" zinc_rl2_valeval_*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
