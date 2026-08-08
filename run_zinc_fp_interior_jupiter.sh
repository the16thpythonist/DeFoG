#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=09:00:00
#SBATCH --output=zinc_fp_interior_%j.out

# JUPITER: does DEEPER conditioning injection raise the fingerprint steering
# ceiling? This is the experiment the v3 result actually pointed at.
#
# WHY, AND WHY NOT MORE FINGERPRINT WORK
# Going binary/512 -> counts/1024 bought +0.014 lift at matched metric width
# (0.1428 -> 0.1566), and did NOT reduce the size dependence it was meant to fix
# (corr(size,lift) -0.673 -> -0.761, if anything worse). So the conditioning
# INFORMATION was not the binding constraint.
#
# The guidance-weight curve says where the constraint is. _blend_rates computes
#     log R_blend = (1-w) log R_uncond + w log R_cond
# so w=1.0 is not a tunable ceiling -- it IS the adapter's own conditional
# denoiser, undiluted, and w>1 extrapolates until the >1e5 rate clamp kills it.
# Measured:
#     w=0    0.128     w=0.5   0.176
#     w=0.25 0.147     w=1.0   0.322
# The 0.5 -> 1.0 step is 5x the 0.25 -> 0.5 step, because the blend is geometric
# in log space and only w=1 delivers the conditional cleanly. There is no
# headroom left in w.
#
# So ~0.32 is simply how sharp this conditional denoiser is. The adapter is
# handed the target's EXACT fingerprint -- perfect information -- and converts it
# into FiLM modulation of a frozen base. That conversion is the bottleneck, and
# pouring more information into a saturated channel is what +0.014 looks like.
#
# WHAT THIS RUN CHANGES
# Two deeper injection points exist in AdaLNAdapter and are both OFF by default:
#     INTERIOR_FF    L4  pre-FFN adaLN-Zero FiLM on X and E
#     INTERIOR_ATTN  L10 conditions e_mul, i.e. edge features -> attention logits
# Output-side FiLM can only rescale what the base already decided. L10 lets the
# fingerprint bias WHICH ATOMS ATTEND TO WHICH, which is structural control
# rather than post-hoc rescaling. L10_LR_SCALE=0.3 already exists in the code as
# a validity guard for exactly these heads, so the risk is anticipated.
#
#     arm0  ff=F attn=F   CONTROL -- reproduces the v3 1e-4 arm
#     arm1  ff=T attn=F
#     arm2  ff=F attn=T
#     arm3  ff=T attn=T
#
# THE CONTROL IS IN-JOB ON PURPOSE. The v3-vs-v2 comparison was invalid because
# the metric width moved with the adapter; a separate job would risk the same
# class of mistake with a different variable. Here every arm shares one job, one
# target draw, one metric width, one base. Only the injection points differ.
#
# Adapter size grows with the injection points (3.0M / 5.2M / 4.8M / 7.0M
# params at cond_dim=1024), so a win could be capacity rather than placement.
# arm1 vs arm2 separates those: similar sizes, different injection points.
#
# WHAT WOULD COUNT
# Lift meaningfully above the control at matched validity. If NOTHING moves,
# that is the informative answer: the frozen base itself is the limit, and
# fingerprint steering is near its ceiling without unfreezing it.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/zinc_rl2_seed42/best_model"
VOCAB="e1_kekulized"
FP_FROM="decoded"
FP_BITS=1024
LR=1e-4                 # best of the v3 sweep (+0.194 decoded lift)
EVAL_ETA=25
MAX_HOURS=7.0

#       arm0    arm1    arm2    arm3
FF=(    False   True    False   True  )
ATTN=(  False   False   True    True  )
TAGS=(  ctrl    ff      attn    both  )

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

mkdir -p experiments/results/adapter_fingerprint__zinc
echo "ZINC fingerprint adapter -- interior conditioning @ $(date)"
echo "base=${BASE} bits=${FP_BITS} counts=True lr=${LR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

python - "$BASE" "$VOCAB" <<'PY'
import sys, importlib.util
try:
    from defog.core import DeFoGModel
    from defog.data import vocabulary
    spec = importlib.util.spec_from_file_location("at", "experiments/adapter_training__zinc.py")
    m = importlib.util.module_from_spec(spec); sys.modules["at"] = m
    spec.loader.exec_module(m)
    atoms, bonds, kek, src = m._vocabulary(sys.argv[2])
    base = DeFoGModel.load(sys.argv[1], device="cpu")
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})"); raise SystemExit(2)
try:
    print(vocabulary.check_model(base, atoms, bonds, what=sys.argv[1]))
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}"); raise SystemExit(1)
print("atoms:", atoms, "| bonds:", bonds)
PY
rc=$?
if [ $rc -eq 1 ]; then echo "ERROR: base and vocabulary disagree -- refusing"; exit 1; fi
if [ $rc -ne 0 ]; then echo "ERROR: preflight could not run (exit $rc)"; exit 1; fi

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_fingerprint__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --FP_FROM "'${FP_FROM}'" \
        --FP_COUNTS True \
        --FP_BITS ${FP_BITS} \
        --INTERIOR_FF ${FF[$i]} \
        --INTERIOR_ATTN ${ATTN[$i]} \
        --BASE_CKPT "'${BASE}'" \
        --LEARNING_RATE ${LR} \
        --ETA ${EVAL_ETA} \
        --MAX_TIME_HOURS ${MAX_HOURS} \
        --__DEBUG__ False \
        > "zinc_fp_interior_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (interior_ff=${FF[$i]} interior_attn=${ATTN[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "Saved adapter" "zinc_fp_interior_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that saved an adapter: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_fp_interior_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== interior conditioning: all arms share targets, metric and base ==="
echo "    read arm0 (ctrl) as the in-job baseline, NOT the v3 numbers"
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]} (ff=${FF[$i]} attn=${ATTN[$i]}) ---"
    grep -E "adapter: [0-9,]+ params|^decoded +(baseline|w=1.0)" \
        "zinc_fp_interior_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -4
done

echo
echo "NEXT: compare arms on decoded lift with validity as a guard. arm1 vs arm2"
echo "separates capacity from placement (similar parameter counts, different"
echo "injection points). If nothing beats arm0, the frozen base is the limit and"
echo "fingerprint steering is near its ceiling without unfreezing it -- which is"
echo "a real answer and means stop spending nodes here."
