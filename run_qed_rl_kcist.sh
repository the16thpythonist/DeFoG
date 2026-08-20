#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --job-name=qed_rl
#SBATCH --output=qed_rl_%j.out

# GDPO RL fine-tune of the QED adapter: better TARGETING first, sanity second.
#
# WHAT IS NEW HERE, AND WHY
# The historical reward (PropertyMatchReward) is connectivity-FIRST: the disconnect
# penalty (-4) sits BELOW the property clamp (-3), so ANY connected molecule outranks
# ANY on-target disconnected one. Sanity vetoes targeting outright. That is the
# opposite of what this run is for, so REWARD_SHAPE=weighted selects
# WeightedSanityPropertyReward:
#
#     r = 3 * targeting  +  1 * sanity          (both normalised to [0,1], so r in [0,4])
#     targeting = 1 - min(|QED - target| / (3 * std), 1)
#     sanity    = 1.00 sane (valid + connected + every ring size in [3,8])
#                 0.67 valid + connected but a ring outside [3,8]
#                 0.33 disconnected
#                 0.00 invalid
#
# At 3:1 a well-targeted molecule with one oversized ring CAN outrank a sane but
# off-target one -- verified on real molecules before submitting: at target 0.50 a
# 11-membered-ring molecule at QED 0.488 scores 3.571 against a sane molecule at
# QED 0.790 scoring 1.601; at target 0.90 the ordering correctly flips (3.093 vs
# 0.670). Sanity is the project's own definition (ring_sizes_ok), not a proxy, so the
# thing optimised is the thing the metric reports.
#
# ROLLOUTS USE THE CONDITIONAL SIZE DRAW. P(n | QED) rather than the dataset marginal,
# because that is how the adapter will be deployed -- measured yesterday at 10-15% MAE
# on its own (job 43071), so training under the marginal would optimise against a
# distribution we no longer use. The trainer forwards RAW per-row targets as the
# condition, which is exactly what LearnedSizeDistribution normalises internally.
#
# REWARD_SOURCE=rdkit, deliberately. QED is closed-form, so there is no reason to
# optimise a learned proxy: the head exists for FK at sampling time, where the true
# function is not available mid-trajectory. Using RDKit here removes head-gaming as a
# failure mode entirely.
#
# WHAT IS SWEPT: KL_COEF only. Everything else is fixed by the user's decisions or by
# earlier sweeps (LR 1e-4, fast EMA 0.9, CRN on, K=128, eta=1 rollouts). Arm 3 is a
# SEED DUPLICATE of arm 1, not a fourth KL value -- without it a 0.005 MAE difference
# between KL values cannot be told from run-to-run noise, and this project has been
# burned by exactly that before.
#
#     arm0  KL 0.05  seed 42        arm2  KL 0.20  seed 42
#     arm1  KL 0.10  seed 42        arm3  KL 0.10  seed 7   <- noise floor
#
# ROLLOUT ETA=1 vs DEPLOYMENT ETA=25 is a known and accepted mismatch. eta=1 was the
# winner of an earlier sweep because under CRN it is the sole within-group diversity
# source, and job 43036 showed conditioning skill is FLAT across eta in {1,5,25} for
# both properties -- so the policy learned at eta=1 transfers to eta=25.
#
# ONE ROUND. The 2-round ratchet is the historically useful depth, but round 2
# regressed SAScore, so the decision is to see round 1 first.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=.venv/bin/python
BASE="ckpts/zinc_rl2_seed42/best_model"
ADAPTER="experiments/results/adapter_training__zinc/12_08_2026__13_35__3Oiw/qed_adapter.ckpt"
SIZE_CKPT="ckpts/heads/qed_head_size.ckpt"
VOCAB="e1_kekulized"

echo "QED adapter RL (targeting-weighted) @ $(date) on $(hostname)"
echo "  base=${BASE}"
echo "  adapter=${ADAPTER}   (this is molsmith/qed@2.0.0's source ckpt)"
echo "  size model=${SIZE_CKPT}   rollouts use P(n|QED)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for f in "${BASE}.ckpt" "$ADAPTER" "$SIZE_CKPT"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Preflight the pieces this run adds, on the CLUSTER's copies -- the reward shape, the
# vocabulary switch and the size-model handoff are all new code paths here.
$PY - "$ADAPTER" "$SIZE_CKPT" "$BASE" <<'PY'
import sys, importlib.util, torch
sys.path.insert(0, ".")
adapter_ckpt, size_ckpt, base_ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("rl", "experiments/adapter_rl_finetune__zinc.py")
rl = importlib.util.module_from_spec(spec); sys.modules["rl"] = rl; spec.loader.exec_module(rl)

if not hasattr(rl, "WeightedSanityPropertyReward"):
    print("REFUSING: WeightedSanityPropertyReward missing -- stale experiment file"); sys.exit(1)
atoms, bonds, kek, src = rl._vocabulary("e1_kekulized")
print(f"vocabulary OK: {atoms} / {bonds} kekulize={kek} source={src}")

from defog.core import LearnedSizeDistribution, AdaLNAdapter, DeFoGModel
sm = LearnedSizeDistribution.load(size_ckpt)
print(f"size model OK: grid {sm.min_size}..{sm.max_size} "
      f"property={sm.property_name!r} from={sm.property_from!r}")
if sm.property_from != "decoded":
    print(f"REFUSING: size model property_from={sm.property_from!r}, expected 'decoded'"); sys.exit(1)

base = DeFoGModel.load(base_ckpt, device="cpu")
ad = AdaLNAdapter.load(adapter_ckpt, device="cpu")
ad.check_compatible(base)
print(f"adapter OK: cond_dim={ad.cond_dim}, {sum(p.numel() for p in ad.parameters()):,} params")

# The size model must actually move the node draw, or the whole point of this run is
# lost silently -- the same failure that made job 43067's 2x2 meaningless.
lo = sm.sample(512, condition=torch.full((512, 1), 0.48)).float().mean()
hi = sm.sample(512, condition=torch.full((512, 1), 0.91)).float().mean()
print(f"size draw: E[n|QED=0.48]={lo:.2f}  E[n|QED=0.91]={hi:.2f}  (table: 26.6 vs 21.8)")
if abs(lo - hi) < 1.0:
    print("REFUSING: the size model barely responds to the condition"); sys.exit(1)
print("PREFLIGHT OK")
PY
[ $? -eq 0 ] || { echo "ERROR: preflight failed -- refusing"; exit 1; }

#        arm0     arm1     arm2     arm3
KLS=(    0.05     0.10     0.20     0.10 )
SEEDS=(  42       42       42       7    )
TAGS=(   kl005    kl010    kl020    kl010s7 )

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PY -u experiments/adapter_rl_finetune__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --PROPERTY "'qed'" \
        --PROPERTY_FROM "'decoded'" \
        --BASE_CKPT "'${BASE}'" \
        --ADAPTER_CKPT "'${ADAPTER}'" \
        --SIZE_MODEL_CKPT "'${SIZE_CKPT}'" \
        --REWARD_SOURCE "'rdkit'" \
        --REWARD_SHAPE "'weighted'" \
        --W_PROP 3.0 --W_SANITY 1.0 --PROP_SPAN 3.0 \
        --KL_COEF ${KLS[$i]} \
        --SEED ${SEEDS[$i]} \
        --MAX_TIME_HOURS 4.0 \
        --TARGET_PERCENTILES "[5,50,95]" \
        --LEVEL_NAMES "['low','mid','high']" \
        --__DEBUG__ False \
        > "qed_rl_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (KL=${KLS[$i]} seed=${SEEDS[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "POST-RL eval\|post-RL" "qed_rl_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that reached the post-RL eval: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms finished; tracebacks follow"
    grep -hA8 "Traceback" qed_rl_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -30
fi

echo
echo "=== pre -> post RL, per arm (RDKit truth, w=1, 500 steps) ==="
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]}  KL=${KLS[$i]} seed=${SEEDS[$i]} ---"
    grep -E "weight drift|pre -> post|^(low|mid|high) w=|reward mean|adapter MOVED|early-stop|best snapshot" \
        "qed_rl_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -12
done

echo
echo "HOW TO PICK THE WINNER"
echo "  Mean RDKit MAE over low/mid/high AFTER RL, lower is better."
echo "  Compare arm1 (KL 0.10 seed 42) against arm3 (KL 0.10 seed 7) FIRST: that gap"
echo "  is the run-to-run noise floor, and no KL difference smaller than it is real."
echo "  Reject any arm whose validity fell more than 5 points below pre-RL -- the"
echo "  3:1 weighting deliberately lets targeting outrank sanity, so a validity"
echo "  collapse is the specific risk this weighting takes on."
echo "  Also read the SANITY numbers, not just validity: the reward scores rings, so"
echo "  wonky_ring_frac is the term that should improve if the sanity half is working."
echo
echo "NEXT: the winning adapter goes through the E2 harness with the frozen best"
echo "config (FK beta=1000 + learned size draw, 100 validation targets) against the"
echo "pre-RL numbers 0.1212 / 0.1033 / 0.0865."
