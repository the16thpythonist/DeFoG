#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=zinc_rl_sweep_%j.out

# JUPITER: sampling sweep for ckpts/zinc_rl2_seed42 -- the RL round-2 model
# being packaged for the molsmith store.
#
# WHY THIS MODEL NEEDS ITS OWN SWEEP
# zinc_rl2_seed42 has only ever been evaluated at the E1 BASE's frozen config
# (steps=500, eta=25, omega=0). That was the right call for the RL comparison --
# holding the sampler fixed isolates the policy change -- but it is the wrong
# thing to ship. RL fine-tuning moved the policy, and eta is error-correction
# stochasticity whose optimum depends on where the policy's errors are. The
# base's optimum is a starting guess, not a shipped default.
#
# GRID: 4 x 4 x 3 = 48 points, wider on steps than the E1 sweep's 2.
#   steps  50, 100, 250, 500
#   eta    0, 5, 25, 50
#   omega  0, 0.05, 0.1
#
# The extra step ladder is specifically for shipping. The E1 sweep only needed
# to know which config produced the best numbers for a table, where 500 steps
# costs nothing but wall clock in a batch job. A web UI generates interactively,
# so if 100 steps is within noise of 500 that is a 5x latency win for every user
# and the right default. That trade-off cannot be read off a 2-point ladder.
#
# One seed only: the sampling configuration is a property of the sampler, not
# the seed.
#
# This job only SAMPLES and writes SMILES. FCD/NSPDK/scaffold scoring happens
# afterwards in the x86 metrics env, which also makes the sweep re-scorable
# without re-sampling.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

CKPT="ckpts/zinc_rl2_seed42/best_model"
OUT_DIR="sweep_zinc_rl2_seed42"
NUM_SAMPLES=1000
STEPS="50,100,250,500"
ETAS="0,5,25,50"
OMEGAS="0,0.05,0.1"

if [ ! -f "${CKPT}.ckpt" ]; then
    echo "ERROR: ${CKPT}.ckpt not found"; exit 1
fi

mkdir -p "$OUT_DIR"
echo "ZINC RL sampling sweep @ $(date); ckpt=$CKPT"
echo "grid: steps=[$STEPS] eta=[$ETAS] omega=[$OMEGAS]"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/sweep_sampling.py \
        --ckpt "$CKPT" --dataset zinc \
        --steps "$STEPS" --eta "$ETAS" --omega "$OMEGAS" \
        --slice ${i}/4 --num-samples ${NUM_SAMPLES} \
        --out-dir "$OUT_DIR" \
        > "zinc_rl_sweep_slice${i}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched slice ${i}/4 on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "sweep finished at $(date)"
DONE=$(ls ${OUT_DIR}/*.json 2>/dev/null | wc -l)
echo "grid points completed: ${DONE} / 48"

# `wait` returns 0 even if every arm died, so make the exit code reflect reality.
if [ "$DONE" -eq 0 ]; then
    echo "ERROR: no grid points produced; arm tracebacks follow"
    grep -hA5 "Traceback" zinc_rl_sweep_slice*_${SLURM_JOB_ID}.out 2>/dev/null | head -20
    exit 1
fi

echo
echo "top points by validity (validity alone does NOT choose the config --"
echo "FCD is scored afterwards in the metrics env and is what decides):"
python - "$OUT_DIR" <<'PY'
import json, glob, sys, os
rows = []
for p in glob.glob(os.path.join(sys.argv[1], "*.json")):
    try:
        d = json.load(open(p))
        rows.append((d.get("validity_relaxed_largest_frag", 0), d.get("uniqueness", 0),
                     os.path.basename(p)[:-5]))
    except Exception:
        pass
for v, u, t in sorted(rows, reverse=True)[:10]:
    print(f"  {t:34s} validity {v:.4f}  uniq {u:.4f}")
PY

echo
echo "NEXT: score ${OUT_DIR}/*.smi against the validation reference with"
echo "scripts/e1_metrics.py, then pick the config for the molsmith package's"
echo "default_sampling. Prefer the cheapest step count within noise of the best,"
echo "since this default runs on every interactive generation in the web UI."
