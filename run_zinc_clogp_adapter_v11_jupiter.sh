#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=zinc_clogp_v11_%j.out

# JUPITER: clogP adapter v1.1 for molsmith/zinc-kek -- relabelled.
#
# WHAT CHANGED, AND ONLY THIS
# PROPERTY_FROM: source -> decoded. Everything else is identical to the v1.0
# run (job 1238148): same base, same vocabulary, same LR x width grid, same
# epochs. One variable.
#
# WHY
# A DeFoG graph stores atoms and bonds but NOT formal charges. 33% of ZINC
# carries one, and protonated amines / carboxylates are exactly what make a
# molecule low-logP. So the v1.0 labels, taken from the source SMILES,
# described molecules the graphs were not -- and only at the low end:
#
#     5th pct:   source -0.90   graph +0.73   error +1.64   92.5% charged
#     95th pct:  source +4.90   graph +4.90   error -0.00    7.0% charged
#
# That is visible in v1.0's steering: high-end MAE 0.65-0.76 across all four
# arms, low-end MAE 1.51-1.73. The adapter was not failing, it was faithfully
# reproducing what it had been taught -- graphs labelled -0.1 really are ~+0.7.
#
# WHAT THIS DOES NOT DO
# It does not give the model new reach. clogP -0.1 needs charges the
# representation cannot express, and no relabelling changes that. It makes the
# target scale honest: the declared range becomes roughly 0.8 to 4.9 instead of
# -0.1 to 4.5, so a requested target means what it says.
#
# SUCCESS CRITERION, fixed in advance
# Low-end MAE should fall well below v1.0's 1.51-1.73. High-end MAE should be
# ~unchanged (0.65-0.76), since the high-end labels were already correct --
# if the high end MOVES much, something other than the relabelling changed and
# the run is suspect rather than successful.
#
# The v1.0 package stays installed. This ships as clogp@1.1.0 only if it wins.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/zinc_rl2_seed42/best_model"
PROPERTY="clogp"
VOCAB="e1_kekulized"
PROPERTY_FROM="decoded"

#        arm0     arm1     arm2     arm3
LRS=(    2e-4     4e-4     2e-4     4e-4 )
HIDDEN=( 256      256      512      512  )
TAGS=(   lr2h256  lr4h256  lr2h512  lr4h512 )

EVAL_ETA=25
EVAL_STEPS=500

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

mkdir -p experiments/results/adapter_training__zinc
echo "ZINC clogP adapter v1.1 (relabelled) @ $(date)"
echo "base=${BASE} vocabulary=${VOCAB} property_from=${PROPERTY_FROM}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary preflight. sys.modules registration is required before
# exec_module or pycomex's decorator raises a TypeError that looks like a
# mismatch but is not -- job 1238075 died exactly that way.
python - "$BASE" "$VOCAB" <<'PY'
import sys, importlib.util
try:
    from defog.core import DeFoGModel
    from defog.data import vocabulary
    spec = importlib.util.spec_from_file_location("at", "experiments/adapter_training__zinc.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["at"] = m
    spec.loader.exec_module(m)
    atoms, bonds, kek, src = m._vocabulary(sys.argv[2])
    base = DeFoGModel.load(sys.argv[1], device="cpu")
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})")
    raise SystemExit(2)
try:
    print(vocabulary.check_model(base, atoms, bonds, what=sys.argv[1]))
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}")
    raise SystemExit(1)
print("atoms:", atoms, "| bonds:", bonds, "| kekulize:", kek)
PY
rc=$?
if [ $rc -eq 1 ]; then echo "ERROR: base and vocabulary disagree -- refusing"; exit 1; fi
if [ $rc -ne 0 ]; then echo "ERROR: preflight could not run (exit $rc)"; exit 1; fi

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_training__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --PROPERTY "'${PROPERTY}'" \
        --PROPERTY_FROM "'${PROPERTY_FROM}'" \
        --BASE_CKPT "'${BASE}'" \
        --LEARNING_RATE ${LRS[$i]} \
        --H_HIDDEN ${HIDDEN[$i]} \
        --ETA ${EVAL_ETA} \
        --EVAL_STEPS ${EVAL_STEPS} \
        --__DEBUG__ False \
        > "zinc_clogp_v11_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (lr=${LRS[$i]} hidden=${HIDDEN[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "Saved adapter" "zinc_clogp_v11_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that saved an adapter: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_clogp_v11_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== steering per arm, compare against v1.0 (job 1238148) ==="
echo "    v1.0 at w=1.0:  low MAE 1.51-1.73   high MAE 0.65-0.76"
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]} (lr=${LRS[$i]} hidden=${HIDDEN[$i]}) ---"
    grep -E "property_from|targets \(clogp\)|baseline clogp|target=.*w=1.0" \
        "zinc_clogp_v11_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -6
done

echo
echo "NEXT: if low-end MAE dropped and the high end held, package the best arm"
echo "as molsmith/clogp@1.1.0 against molsmith/zinc-kek. Targets are now in"
echo "DECODED space, so 1.1.0's declared range will differ from 1.0.0's -- that"
echo "is the fix, not a regression."
