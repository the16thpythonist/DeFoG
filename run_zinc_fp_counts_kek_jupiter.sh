#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=09:00:00
#SBATCH --output=zinc_fp_counts_%j.out

# JUPITER: fingerprint adapter v3 for zinc-kek -- COUNT fingerprints, 1024 bits.
#
# WHAT CHANGED FROM v2.0.0 (job 1263970), and only this:
#     FP_COUNTS  False -> True     (hashed counts, log1p)
#     FP_BITS    512   -> 1024
# Same base, vocabulary, FP_FROM, epochs, batch, eval protocol. Four LRs at
# H_HIDDEN=256, exactly mirroring v2's grid, so any difference is attributable
# to the fingerprint encoding rather than to a co-varying architecture change.
#
# WHY
# A binary Morgan vector records which substructures are present, not how many.
# It calls hexane and eicosane 0.875 similar. v2.0.0 inherited that, and the
# consequence is measurable: steering quality falls off sharply with target size
# (corr(heavy atoms, lift) = -0.92 across six held-out targets, monotone), and
# v2's own analogues include a molecule carrying the reference motif TWICE
# scoring 0.705 -- which binary barely penalises.
#
# Counts give the model the missing signal. On this data the encoding reaches
# log1p(14) for one molecule, i.e. an environment occurring fourteen times, and
# 1024 bits resolves ~44 distinct environments per molecule against 512's ~35.
#
# log1p, NOT raw counts. Counts are small integers with a heavy tail and the
# adapter normalises per-bit by mean/std, so a raw 3 on a rare bit becomes
# roughly a ten-sigma input -- which is where FiLM conditioning destabilises.
# The transform is part of the package declaration, not an implementation
# detail, because molsmith has to apply the identical one at serving time.
#
# THE METRIC DOES NOT CHANGE. Tanimoto is still computed on BINARY fingerprints,
# for both the target and the generated molecules. That is deliberate: if the
# metric moved with the conditioning, "did counts help" would be unanswerable.
# v2.0.0's numbers are therefore directly comparable:
#
#     v2.0.0 (binary, 512):  baseline <T> 0.150  ->  w=1.0  0.323   lift +0.173
#                            validity at w=1.0  0.9826
#
# SHIPPABILITY IS ALREADY HANDLED. molsmith could previously only serve binary;
# FingerprintSpec.counts and a count path in morgan_bits were added first
# (defog-web 335e087, 437 tests passing, existing packages verified unchanged).
# The serving encoding is read from the package, so training and serving cannot
# drift apart once this ships.

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
EVAL_ETA=25
MAX_HOURS=7.0

LRS=(  1e-4  2e-4  3e-4  4e-4 )
TAGS=( lr1   lr2   lr3   lr4  )

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

mkdir -p experiments/results/adapter_fingerprint__zinc
echo "ZINC fingerprint adapter v3 (counts, ${FP_BITS} bits) @ $(date)"
echo "base=${BASE} vocab=${VOCAB} fp_from=${FP_FROM} counts=True bits=${FP_BITS}"
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
    m = importlib.util.module_from_spec(spec)
    sys.modules["at"] = m
    spec.loader.exec_module(m)
    atoms, bonds, kek, src = m._vocabulary(sys.argv[2])
    base = DeFoGModel.load(sys.argv[1], device="cpu")
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})"); raise SystemExit(2)
try:
    print(vocabulary.check_model(base, atoms, bonds, what=sys.argv[1]))
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}"); raise SystemExit(1)
print("atoms:", atoms, "| bonds:", bonds, "| kekulize:", kek)
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
        --BASE_CKPT "'${BASE}'" \
        --LEARNING_RATE ${LRS[$i]} \
        --ETA ${EVAL_ETA} \
        --MAX_TIME_HOURS ${MAX_HOURS} \
        --__DEBUG__ False \
        > "zinc_fp_counts_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (lr=${LRS[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "Saved adapter" "zinc_fp_counts_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that saved an adapter: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_fp_counts_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== v3 (counts, 1024) vs v2.0.0 (binary, 512): binary Tanimoto, same metric ==="
echo "    v2.0.0 reference:  baseline 0.150  w=1.0 0.323  lift +0.173  validity 0.9826"
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]} (lr=${LRS[$i]}) ---"
    grep -E "condition:|^(decoded|source) +(baseline|w=)" \
        "zinc_fp_counts_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -10
done

echo
echo "NEXT: pick on decoded-convention lift with validity as a guard, then check"
echo "the SIZE effect specifically -- corr(heavy atoms, lift) was -0.92 for v2,"
echo "and weakening that is the whole point of counts. Per-target numbers are in"
echo "each arm's adapter_fingerprint_metrics.json."
echo "Ship as molsmith/fingerprint@3.0.0 with --fp-counts (the flag exists now)."
