#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=zinc_clogp_adapter_%j.out

# JUPITER: clogP CFG-adapter for the NEW kekulized ZINC base
# (ckpts/zinc_rl2_seed42/best_model), the model being packaged for molsmith.
#
# WHY A NEW ADAPTER AT ALL
# The eight adapters currently in the store (logp, tpsa, qed, sascore,
# fingerprint, logd x3) are all bound to molsmith/zinc-base -- the OLD aromatic
# checkpoint. That base and this one differ in both the bond set (5 edge classes
# against 4) and the atom ORDER:
#
#     old   C N O S F Cl Br I P     bonds: none single double triple aromatic
#     new   C N O F P S Cl Br I     bonds: none single double triple
#
# so their schema hashes differ and no existing adapter transfers. The old base
# stays installed, so those eight keep working; this is the first adapter for
# the new one.
#
# VOCABULARY=e1_kekulized selects the atom order, bond set, kekulize flag and
# SMILES source together. They are not separately overridable on purpose: an
# adapter trained against a vocabulary the base does not use converges normally
# and is meaningless, and nothing in the training loop would say so. The
# experiment also asserts the base's channel counts against the vocabulary
# before training starts.
#
# TARGET RANGE, measured on the hash-pinned ZINC train split:
#     clogP  mean 2.997  std 1.140   5th pct 0.92   95th pct 4.70
# TARGET_PERCENTILES=[5,95] steers to those two ends.
#
# FOUR ARMS: two learning rates x two adapter widths. The ZINC logP/TPSA
# adapters were validated with a 2-LR sweep, and LR was the parameter that
# mattered; width is included here because this base is a different (RL
# fine-tuned) model and its representation may want more or less capacity.
#
# A note on the property label: it is computed from the SOURCE molecule, which
# carries formal charges the graph does not represent. That is what the
# validated legacy adapters did, and changing it alongside the vocabulary would
# confound the result. Documented in the experiment.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/zinc_rl2_seed42/best_model"
PROPERTY="clogp"
VOCAB="e1_kekulized"

#        arm0     arm1     arm2     arm3
LRS=(    2e-4     4e-4     2e-4     4e-4 )
HIDDEN=( 256      256      512      512  )
TAGS=(   lr2h256  lr4h256  lr2h512  lr4h512 )

# Sampling config for the in-training probe and the end-of-run steering eval.
# ETA is the E1 base's frozen value; the RL model's own sweep (job 1237968) is
# running in parallel and its winner should replace this once known. Steering
# quality is measured as achieved-vs-target, which is far less eta-sensitive
# than FCD, so this is an acceptable placeholder rather than a silent guess.
EVAL_ETA=25
EVAL_STEPS=500

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

mkdir -p experiments/results/adapter_training__zinc
echo "ZINC clogP adapter @ $(date)"
echo "base=${BASE} vocabulary=${VOCAB} property=${PROPERTY}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Fail before the fit loop if the base and the vocabulary disagree, rather than
# after four arms have each burned an hour.
#
# The module MUST be registered in sys.modules before exec_module: pycomex's
# @Experiment decorator runs at import and calls inspect.get_annotations on the
# module it looks up by name, which returns None for an unregistered module and
# raises a TypeError that has nothing to do with vocabularies. Job 1238075 died
# exactly this way and reported it as a mismatch.
python - "$BASE" "$VOCAB" <<'PY'
import sys, importlib.util
try:
    from defog.core import DeFoGModel
    from defog.data import vocabulary
    spec = importlib.util.spec_from_file_location("at", "experiments/adapter_training__zinc.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["at"] = m              # <- pycomex needs this
    spec.loader.exec_module(m)
    atoms, bonds, kek, src = m._vocabulary(sys.argv[2])
    base = DeFoGModel.load(sys.argv[1], device="cpu")
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc}) -- this is a bug in the "
          f"check itself, not evidence about the base.")
    raise SystemExit(2)
try:
    print(vocabulary.check_model(base, atoms, bonds, what=sys.argv[1]))
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}")
    raise SystemExit(1)
print("atoms:", atoms)
print("bonds:", bonds, "kekulize=", kek, "source=", src)
PY
rc=$?
if [ $rc -eq 1 ]; then echo "ERROR: base and vocabulary disagree -- refusing to train"; exit 1; fi
if [ $rc -ne 0 ]; then echo "ERROR: preflight could not run (exit $rc)"; exit 1; fi

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_training__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --PROPERTY "'${PROPERTY}'" \
        --BASE_CKPT "'${BASE}'" \
        --LEARNING_RATE ${LRS[$i]} \
        --H_HIDDEN ${HIDDEN[$i]} \
        --ETA ${EVAL_ETA} \
        --EVAL_STEPS ${EVAL_STEPS} \
        --__DEBUG__ False \
        > "zinc_clogp_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (lr=${LRS[$i]} hidden=${HIDDEN[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "adapter training finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "Saved adapter" "zinc_clogp_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that saved an adapter: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_clogp_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== steering quality per arm (achieved vs target) ==="
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]} (lr=${LRS[$i]} hidden=${HIDDEN[$i]}) ---"
    grep -E "vocabulary |targets |MAE|achieved|baseline" \
        "zinc_clogp_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -8
done

echo
echo "NEXT: pick the arm with the best achieved-vs-target separation between the"
echo "5th and 95th percentile targets, then package it with 'molsmith adapter"
echo "migrate' against the new base. An adapter that moves clogP but wrecks"
echo "validity is not a win -- check the validity line in the eval block too."
