#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --job-name=qed_rl2
#SBATCH --output=qed_rl2_%j.out

# ROUND 2 of the targeting-weighted RL ratchet on the QED adapter.
#
# WHAT MAKES THIS A RATCHET RATHER THAN LONGER TRAINING: the KL guard anchors to
# whatever adapter is LOADED, so starting from a round-1 checkpoint re-anchors the
# guard to it. Round 1's gains are therefore kept as the new reference point instead
# of being pulled back toward the original adapter. It also means this job's own
# pre->post eval IS the round-1 -> round-2 delta, with no extra bookkeeping.
#
# ROUND 2 CAN OVERSHOOT, and the precedent is mixed: logP/TPSA/QED all improved
# further in round 2 historically, but SAScore REGRESSED on every seed. The saving
# grace was that the selection metric reported the regression correctly, so a
# "keep the better round" rule kept round 1. That rule applies here: if every arm
# comes back worse than its round-1 starting point, round 1 ships and this job was
# the measurement that established it.
#
# WHAT IS SWEPT, AND WHY IT IS NOT KL. Round 1 swept KL in {0.05, 0.10, 0.20} and all
# three landed inside the seed-duplicate noise floor, so another KL sweep would buy
# nothing. The open question is which round-1 checkpoint to build on: measured under
# the deployment stack the two are tied --
#     k010  E2 MAE 0.0766   low 0.1005  mid 0.0569  high 0.0715
#     k005  E2 MAE 0.0774   low 0.0992  mid 0.0665  high 0.0656
# -- a 0.0008 difference, well inside noise, but with different SHAPES: k010 trades
# the high third for low/mid, k005 is even across all three. Two seeds from each
# settles it by measurement rather than by argument, and gives a noise floor per
# starting point rather than one pooled guess.
#
#     arm0  from k010  seed 42        arm2  from k005  seed 42
#     arm1  from k010  seed 7         arm3  from k005  seed 7
#
# EVERYTHING ELSE IS ROUND 1's CONFIGURATION, unchanged so the comparison is clean:
# REWARD_SHAPE=weighted at 3:1, REWARD_SOURCE=rdkit, rollouts under P(n|QED),
# KL_COEF 0.10, LR 1e-4, EMA 0.9, CRN, K=128, eta=1, 4h budget.
#
# NOTE ON SEED: SEED changes both the RL stream AND the reference subsample used to
# set targets, so the two seeds per starting point differ slightly in their eval
# targets too. That makes the pair a measure of run-to-run variability rather than of
# seed noise alone -- which is the honest quantity to compare arms against anyway.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=.venv/bin/python
BASE="ckpts/zinc_rl2_seed42/best_model"
SIZE_CKPT="ckpts/heads/qed_head_size.ckpt"
R1="experiments/results/adapter_rl_finetune__zinc/13_08_2026__21_45__"
A_K010="${R1}6cUJ/qed_adapter_rl.ckpt"
A_K005="${R1}dpSM/qed_adapter_rl.ckpt"
VOCAB="e1_kekulized"

echo "QED adapter RL ROUND 2 (ratchet) @ $(date) on $(hostname)"
echo "  round-1 k010: ${A_K010}"
echo "  round-1 k005: ${A_K005}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for f in "${BASE}.ckpt" "$A_K010" "$A_K005" "$SIZE_CKPT"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# The two starting checkpoints must actually differ, and both must differ from the
# ORIGINAL pre-RL adapter -- otherwise this "round 2" would silently redo round 1.
$PY - "$A_K010" "$A_K005" <<'PY'
import sys, hashlib
from pathlib import Path
orig = ("experiments/results/adapter_training__zinc/"
        "12_08_2026__13_35__3Oiw/qed_adapter.ckpt")
paths = {"k010": sys.argv[1], "k005": sys.argv[2], "pre-RL": orig}
h = {}
for k, p in paths.items():
    d = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    h[k] = d
    print(f"  {k:7s} {d[:24]}  {p}")
if len(set(h.values())) != 3:
    print("REFUSING: two of these checkpoints are byte-identical -- this would not "
          "be a second round"); sys.exit(1)
print("  three distinct checkpoints: round 2 starts from genuinely trained weights")
PY
[ $? -eq 0 ] || { echo "ERROR: checkpoint preflight failed"; exit 1; }

#        arm0      arm1      arm2      arm3
ADAPT=(  "$A_K010" "$A_K010" "$A_K005" "$A_K005" )
SEEDS=(  42        7         42        7 )
TAGS=(   k010s42   k010s7    k005s42   k005s7 )

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PY -u experiments/adapter_rl_finetune__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --PROPERTY "'qed'" \
        --PROPERTY_FROM "'decoded'" \
        --BASE_CKPT "'${BASE}'" \
        --ADAPTER_CKPT "'${ADAPT[$i]}'" \
        --SIZE_MODEL_CKPT "'${SIZE_CKPT}'" \
        --REWARD_SOURCE "'rdkit'" \
        --REWARD_SHAPE "'weighted'" \
        --W_PROP 3.0 --W_SANITY 1.0 --PROP_SPAN 3.0 \
        --KL_COEF 0.10 \
        --SEED ${SEEDS[$i]} \
        --MAX_TIME_HOURS 4.0 \
        --TARGET_PERCENTILES "[5,50,95]" \
        --LEVEL_NAMES "['low','mid','high']" \
        --__DEBUG__ False \
        > "qed_rl2_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (from $(basename $(dirname ${ADAPT[$i]})), seed ${SEEDS[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "pre -> post" "qed_rl2_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that reached the post-RL eval: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms finished; tracebacks follow"
    grep -hA8 "Traceback" qed_rl2_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -30
fi

echo
echo "=== ROUND 2: round-1 -> round-2 (RDKit truth, w=1, 500 steps) ==="
echo "    'pre' here IS the round-1 adapter, so these deltas are round-over-round."
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]} ---"
    grep -E "early-stop: deploying|pre -> post|^(low|mid|high) w=" \
        "qed_rl2_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -5
done

echo
echo "THE KEEP-BETTER-ROUND RULE"
echo "  Round 1 reached, on ITS OWN protocol: k010 mean 0.1082, k005 mean 0.1144."
echo "  An arm here whose post-RL mean is ABOVE its starting point OVERSHOT, and"
echo "  round 1 ships for that lineage. That is a real outcome, not a failed job --"
echo "  SAScore regressed in round 2 historically and the guard is what caught it."
echo "  Compare the two seeds within each lineage FIRST: round 1 showed -26% vs"
echo "  -9.8% for identical settings, so a single arm proves nothing about magnitude."
echo
echo "NEXT: whichever lineage wins goes through the E2 harness at FK beta=1000 +"
echo "learned size, against round 1's 0.0766 (k010) / 0.0774 (k005) and the pre-RL"
echo "0.0865. E2 at 1000 molecules (SE ~0.004) is the only instrument here fine"
echo "enough to separate these; the internal probe (144 mols, SE ~0.01) is not."
