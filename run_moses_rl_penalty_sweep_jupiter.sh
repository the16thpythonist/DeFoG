#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=moses_rl_pensweep_%j.out

# JUPITER: MOSES sanity-RL with a distribution-fidelity penalty, weight sweep.
#
# WHY THIS RUN EXISTS
# The unpenalised MOSES RL run raised validity 0.885 -> 0.939 on 4/4 seeds while
# FCD against validation went 0.863 -> 1.706 on 4/4 seeds. Every quantity the
# reward could see improved; the one it could not see collapsed. Of the three
# datasets that run was the largest apparent gain and the only fraudulent one.
#
# WHAT THE PENALTY IS
# A per-sample decomposition of MMD^2 under an RBF kernel on standardised
# physicochemical descriptors, referenced to the TRAIN split:
#
#     r = sanity(0..3) - BETA_MMD * (sim_sibling - 2 * sim_reference)
#
# The sibling term is what makes it anti-collapse: it rises when a sample
# resembles the rest of its own rollout batch, so a narrowing policy is
# penalised even when each molecule it emits is individually ordinary.
#
# The fragment-typicality penalty is deliberately OFF (ALPHA_FRAG=0). The
# offline gate scored it at run-level AUC 0.048 against the known hacked
# policy -- below 0.5, meaning it PREFERS the hacked samples. See
# docs/penalty_gate_moses.json.
#
# WHERE THE WEIGHTS COME FROM
# Not round numbers. On the four hacked runs the hack bought +0.157 of mean
# sanity reward at a cost of only +0.011 of MMD penalty, so the weight at which
# the trade becomes exactly reward-neutral is beta* = 0.157/0.011 ~= 14
# (12.5-20.2 across seeds). The ladder brackets it geometrically:
#
#     arm 0  beta = 0     control -- byte-identical to the original code path,
#                         no penalty object is even constructed
#     arm 1  beta = 3.5   beta*/4  -- sanity still dominates the gradient
#     arm 2  beta = 7     beta*/2
#     arm 3  beta = 14    beta*    -- the hack is exactly reward-neutral
#
# All four share SEED=42 so the contrast is the weight, not seed noise. Seeds
# come later, on whichever weight wins.
#
# WHAT WOULD COUNT AS SUCCESS
# NOT "FCD is preserved". Holding FCD by giving back the whole +5.4 validity
# gain is as much a failure as the hack was -- it just reproduces the base
# model. Success is an arm that keeps a real part of the validity gain while
# holding FCD near the base 0.863. If no arm manages that, the honest reading
# is that MOSES has no sanity headroom that is not paid for in distribution
# fidelity, which is a genuine result and the reason the control arm is here.
#
# The E1 checkpoints are untouched, as always.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/moses_e1_seed42/best_model"
BETAS=(0 3.5 7 14)
TAGS=(b0 b3p5 b7 b14)
SEED=42

if [ ! -f "${BASE}.ckpt" ]; then
    echo "ERROR: ${BASE}.ckpt missing"; exit 1
fi

mkdir -p experiments/results/gdpo_sanity
echo "MOSES RL penalty sweep @ $(date)"
echo "base=${BASE} seed=${SEED} betas=${BETAS[*]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Build the BRICS vocabulary once, before the arms start. Four processes racing
# on the same cache path would each pay the ~7 min build; the save is atomic so
# it would be correct, just wasteful. Only needed if any arm sets ALPHA_FRAG,
# which none currently do -- kept so turning it on does not silently cost 28
# CPU-minutes.
if [ "${BUILD_FRAG_VOCAB:-0}" = "1" ]; then
    echo "pre-building the fragment vocabulary..."
    python - <<'PY'
from defog.data import moses_reference as mref
from defog.core.distribution_penalty import FragmentVocabulary
split = mref.load_reference_split(download=False)
FragmentVocabulary.build_or_load("moses", split.train_smiles, max_molecules=250_000, seed=0)
PY
fi

for i in 0 1 2 3; do
    b=${BETAS[$i]}
    tag=${TAGS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_sanity.py \
        --DATASET "'moses'" \
        --SEED ${SEED} \
        --BETA_MMD "${b}" \
        --ALPHA_FRAG "0.0" \
        --MMD_KERNEL "'descriptor'" \
        --BASE_CKPT "'${BASE}'" \
        --OUT_CKPT_DIR "'ckpts/moses_rlpen_${tag}'" \
        --__DEBUG__ False \
        > "moses_rlpen_${tag}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched beta=${b} (${tag}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "sweep finished at $(date)"

OK=0
for tag in "${TAGS[@]}"; do
    grep -qE "saved final-iteration model" "moses_rlpen_${tag}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that produced a model: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA5 "Traceback" moses_rlpen_*_${SLURM_JOB_ID}.out 2>/dev/null | head -30
    exit 1
fi

echo
echo "=== per-arm summary (validity/disconnected/wonky are RAW rates, comparable across arms) ==="
for i in 0 1 2 3; do
    echo "--- beta=${BETAS[$i]} (${TAGS[$i]}) ---"
    grep -E "loading best checkpoint|^validity \(relaxed\)|^disconnected|^wonky rings|^sanity \(all|min_valid" \
        "moses_rlpen_${TAGS[$i]}_${SLURM_JOB_ID}.out" | tail -6
done

echo
echo "NEXT: FCD is NOT computed here. Score each arm's before.smi/after.smi against"
echo "validation_reference.smi with scripts/e1_metrics.py in .venv_metrics."
echo "An arm whose sanity rose while FCD degraded is a FAILED arm, whatever beta was."
