#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=moses_kek_rl_%j.out

# JUPITER: sanity RL on the KEKULIZED MOSES base -- and a test of whether the
# distribution penalty is still needed.
#
# THE QUESTION THIS RUN ANSWERS
# On the AROMATIC base, plain sanity RL hacked: validity +5.4 points while FCD
# went 1.465 -> 2.307 on 4/4 seeds. The diagnosis, established later, is that
# 118 of 120 hard failures on that base were kekulization errors, so the
# cheapest route to reward was "emit fewer aromatic rings" -- and the policy
# took it (aromatic rings 1.88 -> 1.63).
#
# The kekulized base has NO aromatic bond class. That exploit is impossible by
# construction. If the diagnosis is right, beta=0 RL should now improve sanity
# WITHOUT degrading FCD -- the way ZINC's RL always did, ZINC being kekulized.
#
#   2 arms at beta=0   test the hypothesis
#   2 arms at beta=14  control, the penalty that worked on the aromatic base
#
# Two RL seeds per condition, so a difference has to survive within-condition
# replication before it means anything.
#
# HEADROOM IS THIN -- read the result with this in mind.
# Kekulized base, 4 seeds, n=2048:  sanity 0.9877 +- 0.0039
#   validity 0.9969   disconnected 0.0062   wonky rings 0.0033
# So at most 1.23 points are available and the seed spread is 0.39. A gain
# under ~0.008 is NOT separable from noise at this n. For scale, the shipped
# ZINC RL model reached 0.9877 after two rounds; this base starts there.
#
# The aromatic RL's +6.05 points came almost entirely from the exploit. Nobody
# should expect a repeat, and a small gain here is the honest expectation.
#
# WHAT WOULD MAKE THIS RUN INFORMATIVE, in order:
#   1. beta=0 improves sanity and FCD holds  -> confirms the representation was
#      the root cause of the hack. This is the valuable outcome even if the
#      sanity gain is tiny.
#   2. beta=0 improves sanity and FCD degrades -> a NEW exploit exists that
#      kekulization did not remove. Worth knowing, and the beta=14 arms show
#      whether the penalty still catches it.
#   3. Nothing moves beyond noise -> the base is saturated. Also a result: it
#      says stop spending RL on MOSES.
#
# BASE: seed 44, the best of the four (validity 0.9941, FCD 1.084, disconnected
# 0.0044). Note this breaks direct comparability with the aromatic RL runs,
# which all started from seed 42 -- deliberate, since this base is a candidate
# artifact rather than a comparison point.
#
# REPRESENTATION=kekulized_v2 is REQUIRED. Without it gdpo_sanity resolves the
# dataset default (8 atoms / 5 edge classes) and the channel-count guard refuses
# to run -- which is the point: decoding a 7/4 model with an 8/5 vocabulary
# produces plausible molecules made of the wrong elements, and the reward would
# happily optimise them.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/moses_kek_seed44/best_model"
REPRESENTATION="kekulized_v2"
ITERATIONS=50          # round 2 on the aromatic base showed 50 > 25 at moderate
                       # beta; CKPT_EVERY + SELECT_BEST cover the cliff risk.
EVAL_ETA=25            # the MOSES frozen value, inherited. This model has NOT
                       # had its own sweep, so it is a comparability choice, not
                       # a tuned one -- same caveat as the final evaluation.

#        arm0    arm1    arm2    arm3
BETAS=(  0       0       14      14   )
SEEDS=(  42      43      44      45   )
TAGS=(   b0s42   b0s43   b14s44  b14s45 )

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

mkdir -p experiments/results/gdpo_sanity
echo "MOSES kekulized sanity RL @ $(date)"
echo "base=${BASE} representation=${REPRESENTATION} iterations=${ITERATIONS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary preflight. Fails in seconds rather than after four arms have each
# spent an hour training against mis-decoded molecules. Distinguishes a real
# mismatch from a crash in the check itself -- job 1238075 conflated the two.
python - "$BASE" "$REPRESENTATION" <<'PY'
import sys
try:
    from defog.core import DeFoGModel
    from defog.data import moses_reference as mref, vocabulary
    base = DeFoGModel.load(sys.argv[1], device="cpu")
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})")
    raise SystemExit(2)
try:
    a, b, _ad, _bd, rep, msg = vocabulary.resolve_and_check(
        mref, base, sys.argv[2], what=sys.argv[1])
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}")
    raise SystemExit(1)
print(msg)
print("representation:", rep.name, a, b)
PY
rc=$?
if [ $rc -eq 1 ]; then echo "ERROR: base and representation disagree -- refusing"; exit 1; fi
if [ $rc -ne 0 ]; then echo "ERROR: preflight could not run (exit $rc)"; exit 1; fi

for i in 0 1 2 3; do
    b=${BETAS[$i]}; s=${SEEDS[$i]}; tag=${TAGS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_sanity.py \
        --DATASET "'moses'" \
        --REPRESENTATION "'${REPRESENTATION}'" \
        --SEED ${s} \
        --BETA_MMD "${b}" \
        --ALPHA_FRAG "0.0" \
        --MMD_KERNEL "'descriptor'" \
        --ITERATIONS ${ITERATIONS} \
        --EVAL_ETA ${EVAL_ETA} \
        --BASE_CKPT "'${BASE}'" \
        --OUT_CKPT_DIR "'ckpts/moses_kekrl_${tag}'" \
        --__DEBUG__ False \
        > "moses_kek_rl_${tag}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched beta=${b} rl-seed=${s} (${tag}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "saved final-iteration model" "moses_kek_rl_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that produced a model: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" moses_kek_rl_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus|_setup_socket" | head -25
fi

echo
echo "=== per-arm summary (raw rates, comparable across arms) ==="
for i in 0 1 2 3; do
    echo "--- beta=${BETAS[$i]} rl-seed=${SEEDS[$i]} (${TAGS[$i]}) ---"
    grep -E "representation=|loading best checkpoint|^validity \(relaxed\)|^disconnected|^wonky rings|^sanity \(all" \
        "moses_kek_rl_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -6
done

echo
echo "NEXT: score every arm's before.smi/after.smi against"
echo "validation_reference.smi with scripts/e1_metrics.py."
echo "The beta=0 arms are the test: sanity up AND FCD held means the"
echo "kekulized representation removed the need for the penalty."
echo "Base reference, same sampler and n=2048: sanity 0.9877 +- 0.0039,"
echo "seed 44 specifically FCD 1.084."
