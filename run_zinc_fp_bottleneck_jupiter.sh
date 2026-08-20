#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --output=zinc_fp_bottleneck_%j.out

# JUPITER: is the 256-d TRUNK the thing limiting fingerprint steering?
#
# THE HOLE THIS FILLS
# Three levers have each bought ~+0.015 Tanimoto lift: counts, width (512->1024),
# and injection depth (interior_ff). I read that as evidence the frozen base is
# the ceiling. That reading has a gap, and the gap is architectural.
#
# The conditioning path is:
#     cond -> normalize -> [encoder: NONE] -> concat time -> trunk -> FiLM heads
# and the trunk's first layer is Linear(FP_BITS + 64, H_HIDDEN) with H_HIDDEN
# pinned at 256 for every run so far. So 512 bits and 1024 bits were BOTH
# compressed to 256 dimensions before anything downstream saw them. We never
# measured "more fingerprint information helps"; we measured "more fingerprint
# information pushed through an unchanged 256-d bottleneck helps". Those are
# different claims and only the second was tested. At 2048 bits the squeeze
# becomes 8:1 in a single linear layer.
#
# WHAT EACH ARM ISOLATES
#     arm0  1024  h256  enc=None   CONTROL -- reproduces the shipped ff adapter
#     arm1  2048  h256  enc=None   bits alone, bottleneck unchanged
#     arm2  2048  h512  enc=None   bits + a wider bottleneck
#     arm3  2048  h512  enc=MLP    + a residual encoder before the trunk
#
# arm1 vs arm0 asks whether more bits do anything on their own (prediction: no,
# same as 512->1024). arm2 vs arm1 asks whether the trunk width was the binding
# constraint. arm3 vs arm2 asks whether DEPTH in the conditioning path adds
# anything beyond width. If arm1 ~ arm0 but arm2 >> arm1, the bottleneck was
# real and every earlier "more bits didn't help" conclusion was confounded.
#
# THE CONTROL IS IN-JOB. Same base, same targets, same metric width, same eval.
# Only the conditioning path differs. Chaining comparisons across jobs is what
# invalidated the first v2-vs-v3 result; not repeating it.
#
# THE REGIME DIAGNOSTIC (the arm that can end this line of work)
# Every arm now also scores targets it TRAINED on. If lift on training targets
# is no better than on held-out ones, the adapter is not failing to generalize,
# it is failing to fit -- and no amount of conditioning capacity (bits, width,
# depth) can be the answer. That result would settle the ceiling question
# directly rather than by accumulating null effects, and it costs one extra
# target group per arm.
#
# EVAL TARGETS: the original 6 held-out (drawn by the identical call, so the
# numbers stay comparable with every previous run) + 10 fresh held-out (the
# original 6 have now driven selection four times and are no longer a clean
# estimate) + 6 training. GUIDANCE_WEIGHTS is cut to [1.0] because the w-curve
# is already established and w=1.0 is the conditional denoiser undiluted --
# spending eval on w=0.25/0.5 would re-measure a known monotone curve.
#
# VALIDITY BUDGET: down to ~0.96 is acceptable (the shipped v2 sits at 0.982,
# the ff adapter at 0.971). An arm that wins on lift while falling below that
# is a diagnostic, not a shippable artifact -- read the validity column.
#
# MEMORY: the condition array is ~1.8 GB per arm at 2048 bits, ~3.6 GB at the
# np.stack peak, x3 arms at 2048 = ~11 GB transient. The node absorbs this.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/zinc_rl2_seed42/best_model"
VOCAB="e1_kekulized"
FP_FROM="decoded"
LR=1e-4                 # best of the v3 sweep, and what the ff arm used
EVAL_ETA=25
MAX_HOURS=7.5

#       arm0    arm1    arm2    arm3
BITS=(  1024    2048    2048    2048  )
HID=(   256     256     512     512   )
ENC=(   "None"  "None"  "None"  "{'kind':'mlp','out_dim':512,'hidden':1024,'n_blocks':2}" )
TAGS=(  ctrl    bits    width   enc   )

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

mkdir -p experiments/results/adapter_fingerprint__zinc
echo "ZINC fingerprint adapter -- conditioning BOTTLENECK ablation @ $(date)"
echo "base=${BASE} counts=True interior_ff=True lr=${LR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary guard + a direct check that the new encoder actually builds. A
# typo in the spec would otherwise surface only after the dataset and base were
# loaded, minutes into a 7-hour arm.
python - "$BASE" "$VOCAB" <<'PY'
import sys, importlib.util
try:
    from defog.core import DeFoGModel, AdaLNAdapter
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
try:
    a = AdaLNAdapter.for_base(
        base, cond_dim=2048, hidden=512, interior_ff=True,
        cond_encoder={"kind": "mlp", "in_dim": 2048, "out_dim": 512,
                      "hidden": 1024, "n_blocks": 2})
    import torch
    out = a(torch.zeros(2, 2048), t=torch.rand(2, 1))
    print(f"encoder preflight OK: {sum(p.numel() for p in a.parameters()):,} params, "
          f"{len(out.layers)} modulated layers")
except Exception as exc:
    print(f"ENCODER PREFLIGHT FAILED ({type(exc).__name__}: {exc})"); raise SystemExit(3)
print("atoms:", atoms, "| bonds:", bonds)
PY
rc=$?
if [ $rc -eq 1 ]; then echo "ERROR: base and vocabulary disagree -- refusing"; exit 1; fi
if [ $rc -eq 3 ]; then echo "ERROR: cond_encoder does not build -- refusing"; exit 1; fi
if [ $rc -ne 0 ]; then echo "ERROR: preflight could not run (exit $rc)"; exit 1; fi

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_fingerprint__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --FP_FROM "'${FP_FROM}'" \
        --FP_COUNTS True \
        --FP_BITS ${BITS[$i]} \
        --H_HIDDEN ${HID[$i]} \
        --COND_ENCODER "${ENC[$i]}" \
        --INTERIOR_FF True \
        --INTERIOR_ATTN False \
        --BASE_CKPT "'${BASE}'" \
        --LEARNING_RATE ${LR} \
        --ETA ${EVAL_ETA} \
        --GUIDANCE_WEIGHTS "[1.0]" \
        --MAX_TIME_HOURS ${MAX_HOURS} \
        --__DEBUG__ False \
        > "zinc_fp_bneck_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (bits=${BITS[$i]} hidden=${HID[$i]} enc=${ENC[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "Saved adapter" "zinc_fp_bneck_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that saved an adapter: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_fp_bneck_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== bottleneck ablation: all arms share base, targets, metric and eval ==="
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]}  bits=${BITS[$i]} hidden=${HID[$i]} enc=$([ "${ENC[$i]}" = "None" ] && echo no || echo yes) ---"
    grep -E "conditioning path:|adapter: [0-9,]+ params" \
        "zinc_fp_bneck_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -2
    grep -E "^(holdout_orig|holdout_fresh|train) +[0-9]+ +w=" \
        "zinc_fp_bneck_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null
    grep -E "train-minus-heldout" "zinc_fp_bneck_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null
done

echo
echo "HOW TO READ THIS"
echo "  arm1 ~ arm0            -> more bits alone still do nothing (expected)"
echo "  arm2 >> arm1           -> the 256-d trunk WAS the bottleneck; every earlier"
echo "                            'more bits did not help' result was confounded"
echo "  arm3 >> arm2           -> depth in the conditioning path buys more than width"
echo "  all four within ~0.015 -> the conditioning path is not the constraint at all"
echo
echo "CAPACITY CONFOUND, stated up front: arm0/arm1 are near capacity-matched"
echo "  (5.23M vs 5.49M params) so arm1-vs-arm0 is a clean test of bits alone."
echo "  arm2 (11.1M) and arm3 (15.0M) are NOT matched to them -- widening the trunk"
echo "  necessarily adds parameters, so a win there is 'wider path OR more capacity'."
echo "  The interior-injection job is the partial answer: its largest arm (7.0M,"
echo "  ff+attn) was its WORST, so capacity alone did not buy lift there. That makes"
echo "  a capacity-only explanation less likely here, but it does not exclude it."
echo "  train ~ holdout_fresh  -> not generalization-limited; capacity cannot be the"
echo "                            answer and this axis is genuinely finished"
echo "Compare on the holdout_fresh row (honest estimate); holdout_orig exists only"
echo "to stay comparable with the four runs that already used those six targets."
