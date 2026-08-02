#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=moses_rl_valeval_%j.out

# JUPITER: n=25,000 evaluation of the MOSES RL models -- on VALIDATION.
#
# This feeds a SELECTION decision (does RL beat the base, and which seed), and
# selecting by test numbers is tuning on test. MOSES's test split has been read
# once (the E1 row) and is reserved for reporting an already-chosen model.
#
# The four E1 bases are re-evaluated here too, on the same validation reference
# at the same n. Their existing numbers are on TEST against a 176,074-molecule
# reference; validation has 5,000, and FCD is strongly reference-size dependent
# (~1/n). Comparing RL-on-validation against base-on-test would attribute a
# reference-set artefact to the model.
#
# RL at n=2,048 showed validity 0.8856 -> 0.9381, a +5.3 point gain on all four
# seeds -- by far the largest RL result in this project. This run checks the
# magnitude at scale AND checks FCD: a jump that size is exactly when
# distribution narrowing is most likely, and the MOSES FCD/TestSF baseline is
# 1.265 +- 0.037. Improved validity with degraded FCD is a FAILED result.
#
# 8 models over 4 GPUs: each GPU takes one base then its RL counterpart.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

STEPS=500; ETA=25; OMEGA=0; N=25000
OUT_DIR="valeval_moses_rl"

for i in 0 1 2 3; do
    s=$((42+i))
    for c in "ckpts/moses_e1_seed${s}/best_model" "ckpts/moses_rl_seed${s}/best_model"; do
        [ -f "${c}.ckpt" ] || { echo "ERROR: ${c}.ckpt missing"; exit 1; }
    done
done

mkdir -p "$OUT_DIR"
echo "MOSES RL VALIDATION eval @ $(date)"
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
        --ckpt "$2" --dataset moses --tag "$3" --split validation \
        --sample-steps ${STEPS} --eta ${ETA} --omega ${OMEGA} \
        --num-samples ${N} \
        --sweep-dir "sweep_moses_seed42 (inherited from the E1 base; NOT re-swept)" \
        --out-dir "$OUT_DIR" > "moses_rl_valeval_$3_${SLURM_JOB_ID}.out" 2>&1
}

# Each GPU runs one E1 base then its RL counterpart, so every pair is measured
# on the same reference under identical conditions.
for i in 0 1 2 3; do
  s=$((42+i))
  ( run_one $i "ckpts/moses_e1_seed${s}/best_model" "base_seed${s}"
    run_one $i "ckpts/moses_rl_seed${s}/best_model" "rl_seed${s}" ) &
  echo "launched base+rl seed ${s} on GPU ${i}"
done
echo "launched 8 evaluations across 4 GPUs"

wait
echo "finished at $(date)"
DONE=$(ls ${OUT_DIR}/*_seed*.json 2>/dev/null | wc -l)
echo "models evaluated: ${DONE} / 8"
if [ "$DONE" -lt 8 ]; then
    echo "ERROR: not all models evaluated; tracebacks follow"
    grep -hA5 "Traceback" moses_rl_valeval_*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi
