#!/bin/bash
#SBATCH --job-name=qedrlhd2
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG/qedrlhd2_%j.out

# QED adapter RL, ROUND 2, learned head as the reward, starting from round 1's best seed.
#
# ROUND 1 (job 43186) took the pre-RL adapter to E2 MAE 0.0892 with seed 21, significantly
# better than its own starting point (0.1045, p<1e-4) and statistically tied with the
# rdkit-reward adapter qed@3.1.0 (0.0920, p=0.44). True RDKit QED improved alongside the
# head metric in 12 of 12 seed x level cells, so the head is a usable reward and is not
# being gamed.
#
# WHY A SECOND ROUND AND NOT JUST A LONGER FIRST. Seeds 21 and 7 both deployed their LAST
# probe (iter 200) -- neither plateaued, they ran out of the 4h budget. Reloading the round-1
# adapter makes it both the initialisation and the KL reference, so the 0.05 leash resets and
# the policy may drift as far again from s21 as round 1 drifted from pre-RL. Continuing under
# the original anchor would instead keep total drift bounded by one leash. The re-anchored
# form is the 2-round ratchet that has been the useful depth on this project before.
#
# Head reward again, deliberately: the point is a pure 2-round head-reward lineage, which is
# the thing with no closed-form analogue. Switching to rdkit here would likely produce a
# better QED adapter and a worse experiment.
#
# 4 seeds at KL=0.05, identical to round 1, so round 2 is comparable to round 1 arm for arm.
#
# EXPECT A SMALLER GAIN THAN ROUND 1. Round 1 moved 0.0154. A second round plausibly gives
# under the 0.008 pooled noise floor. The paired E2 resolved -0.0125 at p=1e-4 and failed on
# -0.0028, so the practical detection threshold is ~0.005-0.008; a null here is a real answer,
# not a failed run.
#
# The kekulize preflight below is retained and still matters: it is what stands between this
# and job 43175, where 94% of good molecules scored as unparseable.
set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
BASE="ckpts/zinc_rl2_seed42/best_model"
ADAPTER="experiments/results/adapter_rl_finetune__zinc/18_08_2026__16_11__JRAw/qed_adapter_rl.ckpt"   # molsmith/qed@5.0.0 = head-RL round 1, seed 21
HEAD="ckpts/heads/qed_head.ckpt"                  # na=9 nb=4, holdout MAE 0.0219, r=0.968
SIZE_CKPT="ckpts/heads/qed_head_size.ckpt"
VOCAB="e1_kekulized"

echo "QED head-reward RL ROUND 2 @ $(date) on $(hostname)"
echo "  base=${BASE}  adapter=${ADAPTER}  head=${HEAD}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for f in "${BASE}.ckpt" "$ADAPTER" "$HEAD" "$SIZE_CKPT"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

$PY - "$ADAPTER" "$HEAD" "$SIZE_CKPT" "$BASE" <<'PY'
import sys, importlib.util, torch
sys.path.insert(0, ".")
adapter_ckpt, head_ckpt, size_ckpt, base_ckpt = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("rl", "experiments/adapter_rl_finetune__zinc.py")
rl = importlib.util.module_from_spec(spec); sys.modules["rl"] = rl; spec.loader.exec_module(rl)

atoms, bonds, kek, src = rl._vocabulary("e1_kekulized")
print(f"vocabulary OK: {atoms} / {bonds} kekulize={kek} source={src}")

from defog.core import (AdaLNAdapter, DeFoGModel, LearnedSizeDistribution,
                        PropertyHead)
from defog.core import head_predict_batch as _lib_hpb
from defog.domains.molecule import build_encoders, needs_kekulize

# Test the symbol THE EXPERIMENT WILL USE, not the library's. The first version of this
# gate imported head_predict_batch from defog.core, reported 0/96 rejected, and passed the
# job through -- while the experiment ran a byte-drifted local copy that rejected 94%. A
# gate that exercises a different implementation than the run manufactures confidence.
head_predict_batch = rl.head_predict_batch
if head_predict_batch is not _lib_hpb:
    print("REFUSING: the experiment does not use defog.core.head_predict_batch. It had a "
          "local duplicate once; that duplicate silently missed the kekulize fix and cost "
          "job 43175. Import it, do not re-declare it."); sys.exit(1)

ae, adec, be, bdec = build_encoders(atoms, bonds)

# ---- THE GATE THIS RUN EXISTS BEHIND -------------------------------------------------
# head_predict_batch must return a real number for ordinary aromatic molecules on this
# kekulized vocabulary. Before the fix it returned None for ~94% of them and every one of
# those scored -10, the invalid floor, which is a stronger training signal than anything
# the property term can produce.
if not needs_kekulize(be):
    print("REFUSING: this vocabulary has an AROMATIC class, so this is not the kekulized "
          "base the run is supposed to use."); sys.exit(1)

head = PropertyHead.load(head_ckpt, device="cpu")
print(f"head OK: na/nb from ckpt, prop_mean={float(head.prop_mean):.4f} "
      f"prop_std={float(head.prop_std):.4f}")

from rdkit import Chem, RDLogger
from rdkit.Chem import QED
RDLogger.DisableLog("rdApp.*")
from defog.data import zinc_reference as zref
smis = zref.load_reference_split().val_smiles[:96]
mols = [Chem.MolFromSmiles(s) for s in smis]
mols = [m for m in mols if m is not None]
n_arom = sum(any(b.GetIsAromatic() for b in m.GetBonds()) for m in mols)
preds = head_predict_batch(mols, head, ae, be, "cpu")
n_none = sum(p is None for p in preds)
print(f"head_predict_batch on {len(mols)} real molecules ({n_arom} aromatic): "
      f"{n_none} returned None")
if n_none > 0:
    print(f"REFUSING: {n_none}/{len(mols)} real molecules score as un-encodable, so they "
          f"would take invalid_reward=-10 and the RL would train the adapter away from "
          f"aromatic rings. The kekulize fix is NOT live in this tree."); sys.exit(1)

true = [QED.qed(m) for m in mols]
got = [p for p in preds]
mae = sum(abs(a - b) for a, b in zip(true, got)) / len(true)
import statistics
print(f"head vs RDKit QED on those molecules: MAE {mae:.4f} "
      f"(property std {statistics.pstdev(true):.4f})")
if mae > 0.06:
    print("REFUSING: the head disagrees with RDKit far more than its holdout MAE of "
          "0.022 -- wrong head, wrong vocabulary, or wrong encoding."); sys.exit(1)

sm = LearnedSizeDistribution.load(size_ckpt)
lo = sm.sample(512, condition=torch.full((512, 1), 0.48)).float().mean()
hi = sm.sample(512, condition=torch.full((512, 1), 0.91)).float().mean()
print(f"size draw: E[n|QED=0.48]={lo:.2f}  E[n|QED=0.91]={hi:.2f}")
if abs(lo - hi) < 1.0:
    print("REFUSING: the size model barely responds to the condition"); sys.exit(1)

base = DeFoGModel.load(base_ckpt, device="cpu")
ad = AdaLNAdapter.load(adapter_ckpt, device="cpu")
ad.check_compatible(base)
print(f"adapter OK: cond_dim={ad.cond_dim}, {sum(p.numel() for p in ad.parameters()):,} params")
print("PREFLIGHT OK")
PY
[ $? -eq 0 ] || { echo "ERROR: preflight failed -- refusing"; exit 1; }

SEEDS=( 42 7 13 21 )
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PY -u experiments/adapter_rl_finetune__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --PROPERTY "'qed'" \
        --PROPERTY_FROM "'decoded'" \
        --BASE_CKPT "'${BASE}'" \
        --ADAPTER_CKPT "'${ADAPTER}'" \
        --HEAD_CKPT "'${HEAD}'" \
        --SIZE_MODEL_CKPT "'${SIZE_CKPT}'" \
        --REWARD_SOURCE "'head'" \
        --KL_COEF 0.05 \
        --SEED ${SEEDS[$i]} \
        --MAX_TIME_HOURS 4.0 \
        --TARGET_PERCENTILES "[5,50,95]" \
        --LEVEL_NAMES "['low','mid','high']" \
        --__DEBUG__ False \
        > "qed_rlhd2_s${SEEDS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed=${SEEDS[$i]} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "ALL_ARMS_DONE at $(date)"

OK=0
for s in "${SEEDS[@]}"; do
    grep -q "POST-RL eval\|pre -> post" "qed_rlhd2_s${s}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that reached the post-RL eval: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms finished; tracebacks follow"
    grep -hA8 "Traceback" qed_rlhd2_s*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -30
fi

echo
echo "=== pre -> post, per seed. RDKit-MAE is TRUTH; head-MAE is the objective. ==="
echo "=== head improving while RDKit does not = the head is being gamed.        ==="
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "weight drift|pre -> post|^(low|mid|high) w=|reward mean|adapter MOVED|early-stop|best snapshot|RDKit-MAE|head-MAE" \
        "qed_rlhd2_s${s}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -14
done
echo "SUMMARY_DONE"
