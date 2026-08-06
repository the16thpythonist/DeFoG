#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=09:00:00
#SBATCH --output=zinc_fp_kek_%j.out

# JUPITER: Morgan-fingerprint steering adapter for molsmith/zinc-kek.
#
# WHY: zinc-kek has no fingerprint adapter at all. The existing
# molsmith/fingerprint@1.0.0 is bound to the OLD aromatic base, whose schema
# differs in both bond set and atom ORDER, so it cannot transfer. This ships
# regardless of the measured lift, because "no reference-molecule steering" is
# worse than "some" -- but the lift is reported honestly either way.
#
# FP_FROM=decoded. Morgan bits encode formal charges; a DeFoG graph does not
# store them, and 32% of ZINC molecules carry one. Measured over ZINC train,
# 512-bit r=2, Tanimoto( FP(source), FP(decoded) ):
#
#     neutral molecules (68%)   1.0000   <- identical, nothing lost
#     charged molecules (32%)   0.6813
#     overall                   0.8990
#
# Unlike clogP, stereochemistry is irrelevant here (Morgan bits ignore it), so
# the damage is confined entirely to charged molecules -- and for those it is
# severe. Labelling from the decoded molecule costs the 68% nothing and fixes
# the 32%. Same fix that took clogp low-end MAE from 1.51-1.73 to 0.63-0.71.
#
# THE CEILING, AND WHY BOTH CONVENTIONS ARE REPORTED
# 0.899 is not just a label defect, it is a hard limit on the reported metric: a
# model that reproduced a target GRAPH perfectly still scores only 0.899 against
# a source-derived target fingerprint, because it cannot express the charges.
# The evaluation samples ONCE per (target, weight) and scores those same
# molecules against both conventions:
#
#     decoded targets  ceiling 1.000  measures steering ability cleanly
#     source targets   ceiling 0.899  measures what a user pasting a real
#                                     molecule actually experiences
#
# The gap between the two IS the charge limitation, shown rather than absorbed
# into a single number that would look like underperformance.
#
# ETA=25 for evaluation, not the original experiment's 5. That is zinc-kek's
# swept deployment default (job 1237968), so the measured steering reflects how
# the adapter will actually be run.
#
# Reference point: the old aromatic-base fingerprint adapter achieved a +0.12
# Tanimoto lift over unconditional. Not directly comparable -- different base,
# different vocabulary, different eta, different label convention -- but it is
# the only prior number and worth beating.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/zinc_rl2_seed42/best_model"
VOCAB="e1_kekulized"
FP_FROM="decoded"
EVAL_ETA=25
MAX_HOURS=7.0

#        arm0   arm1   arm2   arm3
LRS=(    1e-4   2e-4   3e-4   4e-4 )
TAGS=(   lr1    lr2    lr3    lr4  )

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

mkdir -p experiments/results/adapter_fingerprint__zinc
echo "ZINC fingerprint adapter for zinc-kek @ $(date)"
echo "base=${BASE} vocabulary=${VOCAB} fp_from=${FP_FROM} eval_eta=${EVAL_ETA}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary preflight. sys.modules registration before exec_module is required
# or pycomex's decorator raises a TypeError that looks like a mismatch but is
# not -- job 1238075 died exactly that way.
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
    CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_fingerprint__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --FP_FROM "'${FP_FROM}'" \
        --BASE_CKPT "'${BASE}'" \
        --LEARNING_RATE ${LRS[$i]} \
        --ETA ${EVAL_ETA} \
        --MAX_TIME_HOURS ${MAX_HOURS} \
        --__DEBUG__ False \
        > "zinc_fp_kek_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (lr=${LRS[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "Saved adapter" "zinc_fp_kek_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that saved an adapter: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_fp_kek_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== steering per arm: <T> and lift, BOTH target conventions ==="
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]} (lr=${LRS[$i]}) ---"
    grep -E "ceiling when scoring|^(decoded|source) +(baseline|w=)" \
        "zinc_fp_kek_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -10
done

echo
echo "NEXT: pick on LIFT over the unconditional baseline (decoded convention),"
echo "with validity as a guard -- an adapter that raises Tanimoto by breaking"
echo "molecules is not steering. Then package as molsmith/fingerprint bound to"
echo "molsmith/zinc-kek. Ship regardless of magnitude: zinc-kek currently has no"
echo "fingerprint adapter at all. Report the lift honestly either way."
