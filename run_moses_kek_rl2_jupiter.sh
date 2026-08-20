#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=07:00:00
#SBATCH --output=moses_kek_rl2_%j.out

# JUPITER: ROUND 2 of sanity RL on kekulized MOSES, from the round-1 winner.
#
# BASE: ckpts/moses_kekrl_b0s43/best_model -- now the working MOSES checkpoint.
# Confirmed at n=10,000 with replicates (job 1248827):
#     FCD          0.5928  against its base's 0.6313   (-6.1%, 4.6x the noise floor)
#     valid+conn   0.9942  against 0.9931              (inside noise, floor 0.0020)
# So round 1 bought distribution match, NOT validity. That is the inverse of the
# aromatic-base hack, which bought validity by wrecking FCD.
#
# TWO DESIGN CHANGES FROM ROUND 1, both forced by what round 1 taught us.
#
# 1. ALL FOUR ARMS AT beta=0, four different RL seeds.
#    Round 1 ran 2 arms at beta=0 and 2 at beta=14. The penalty arms gained
#    nothing extra (same sanity, FCD equally flat) which is what you expect when
#    there is no exploit left to defend against -- the kekulized base has no
#    AROMATIC class to drop. Spending arms on a control that controls for
#    nothing is waste. Four beta=0 seeds instead answer the question round 1
#    left open: b0_s43 was the best of four arms, so its gain could have been
#    one lucky arm. Four replicates show whether beta=0 RL improves FCD
#    RELIABLY or only sometimes.
#
#    The safety argument for keeping a penalty arm is weak here: FCD is measured
#    on every arm afterwards, and that measurement -- not the penalty -- is what
#    actually catches a hack. The penalty only helps if you intend to keep
#    training through one.
#
# 2. EVAL_SAMPLES 2048 -> 8192.
#    Round 1's per-arm FCD numbers were unusable. The four `before` runs sampled
#    one checkpoint four times and their FCD spread was 0.046 -- larger than the
#    ~0.04 effect we were trying to see. That is why b0_s43's apparent -0.076
#    needed a separate 10,000-sample job to confirm. At n=8192 the floor should
#    land near 0.012 (the n=10,000 replicates measured 0.0084), which is small
#    enough to read the in-run numbers directly instead of re-running.
#
# WHAT TO EXPECT: little. ZINC's round 2 gave consistent but small further
# gains and round 3 was entirely inside the noise floor. This base has
# valid+connected 0.9942 and wonky rings near zero, so the sanity reward has
# almost nothing left to buy. The honest hypothesis is that round 2 either
# repeats round 1's FCD improvement at a smaller magnitude, or does nothing.
# "Does nothing on 4/4 seeds" is a real answer and means stop.
#
# E1 and round-1 checkpoints are untouched; this writes to ckpts/moses_kekrl2_*.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/moses_kekrl_b0s43/best_model"
REPRESENTATION="kekulized_v2"
ITERATIONS=50
EVAL_SAMPLES=8192
EVAL_ETA=25            # inherited; this lineage still has no sweep of its own
SEEDS=(101 202 303 404)

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

mkdir -p experiments/results/gdpo_sanity
echo "MOSES kekulized sanity RL -- ROUND 2 @ $(date)"
echo "base=${BASE} representation=${REPRESENTATION} iterations=${ITERATIONS} eval_n=${EVAL_SAMPLES}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

python - "$BASE" "$REPRESENTATION" <<'PY'
import sys
try:
    from defog.core import DeFoGModel
    from defog.data import moses_reference as mref, vocabulary
    base = DeFoGModel.load(sys.argv[1], device="cpu")
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})"); raise SystemExit(2)
try:
    a, b, _ad, _bd, rep, msg = vocabulary.resolve_and_check(
        mref, base, sys.argv[2], what=sys.argv[1])
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}"); raise SystemExit(1)
print(msg); print("representation:", rep.name, a, b)
PY
rc=$?
if [ $rc -eq 1 ]; then echo "ERROR: base and representation disagree -- refusing"; exit 1; fi
if [ $rc -ne 0 ]; then echo "ERROR: preflight could not run (exit $rc)"; exit 1; fi

for i in 0 1 2 3; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_sanity.py \
        --DATASET "'moses'" \
        --REPRESENTATION "'${REPRESENTATION}'" \
        --SEED ${s} \
        --BETA_MMD "0.0" \
        --ALPHA_FRAG "0.0" \
        --ITERATIONS ${ITERATIONS} \
        --EVAL_SAMPLES ${EVAL_SAMPLES} \
        --EVAL_ETA ${EVAL_ETA} \
        --BASE_CKPT "'${BASE}'" \
        --OUT_CKPT_DIR "'ckpts/moses_kekrl2_s${s}'" \
        --__DEBUG__ False \
        > "moses_kek_rl2_s${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched rl-seed=${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for s in "${SEEDS[@]}"; do
    grep -q "saved final-iteration model" "moses_kek_rl2_s${s}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that produced a model: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" moses_kek_rl2_s*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus|_setup_socket" | head -25
fi

echo
echo "=== per-arm summary, eval n=${EVAL_SAMPLES} ==="
for s in "${SEEDS[@]}"; do
    echo "--- rl-seed ${s} ---"
    grep -E "loading best checkpoint|^validity \(relaxed\)|^disconnected|^wonky rings|^sanity \(all" \
        "moses_kek_rl2_s${s}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -5
done

echo
echo "NEXT: score every arm's before.smi and after.smi for FCD."
echo "The four before.smi files sample ONE checkpoint four times -- their spread"
echo "IS the noise floor at this n, so compute it before believing any delta."
echo "Round 1 reference, n=10,000: this base scored FCD 0.5928 against its own"
echo "base's 0.6313. Those are NOT comparable to n=8192 numbers; FCD is n-biased."
