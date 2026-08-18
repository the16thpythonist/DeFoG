#!/bin/bash
#SBATCH --job-name=qedrlhead
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/tm4030/Programming/DeFoG/qedrlhead_%j.out

# QED adapter RL with the LEARNED HEAD as the reward, on the kekulized base.
#
# WHY THIS RUN EXISTS. This is the combination that has never been run: every
# head-reward run to date used the AROMATIC base (zinc-base, 5 edge classes), and every
# kekulized-base RL run used REWARD_SOURCE=rdkit, chosen deliberately because QED has a
# closed form. So the head-as-reward mechanism -- the whole point of a head, since it is
# the only option for a property with no closed form -- has never been exercised where
# it matters most.
#
# AND IT WOULD HAVE BEEN DESTROYED UNTIL YESTERDAY. head_predict_batch re-encoded each
# decoded molecule from an aromatic SMILES into a vocabulary with no aromatic bond class
# and got None back for ~94% of real molecules. HeadPropertyMatchReward leaves those at
# invalid_reward = -10.0, BELOW disconnect_reward = -4.0 and below the worst on-target
# score of -PROP_CLAMP = -3.0. With grouped advantages, ~94% pinned at the floor and ~6%
# scoring normally, the gradient would have trained the adapter to stop making aromatic
# rings -- i.e. away from drug-likeness. The preflight below refuses to start unless the
# fix is live, because that failure produces a plausible-looking run, not a crash.
#
# DESIGN. Head reward, 4 seeds, one variable. Start point is the PRE-RL adapter
# (molsmith/qed@2.0.0, exported to ckpts/qed_adapter_pre_rl.ckpt) so the head reward's
# effect is not confounded by two prior rounds of rdkit-reward RL. KL fixed at 0.05, the
# value behind the shipped qed@3.1.0. Rollouts draw n from P(n|QED), matching the rdkit
# runs so the reward is the difference that remains.
#
# HOW TO READ IT. The head defines the objective AND picks the checkpoint; RDKit QED is
# logged alongside as ground truth. Those two moving together means the head is a usable
# reward. Head-MAE improving while RDKit-MAE does not is the head being gamed, and is the
# result this run is actually designed to detect. Reference: the rdkit-reward round-1 runs
# from the same start point, and qed@3.1.0's adapter-only E2 MAE of 0.0920.
#
# NOTE ON REWARD_SHAPE. REWARD_SOURCE=head takes the TIERED reward and ignores
# REWARD_SHAPE, so this is not shape-matched to the rdkit runs, which used weighted 3:1.
# That is a real difference; do not read a head-vs-rdkit gap as purely the reward source.
set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=/home/tm4030/Programming/DeFoG/.venv/bin/python
BASE="ckpts/zinc_rl2_seed42/best_model"
ADAPTER="ckpts/qed_adapter_pre_rl.ckpt"           # molsmith/qed@2.0.0, pre-RL
HEAD="ckpts/heads/qed_head.ckpt"                  # na=9 nb=4, holdout MAE 0.0219, r=0.968
SIZE_CKPT="ckpts/heads/qed_head_size.ckpt"
VOCAB="e1_kekulized"

echo "QED head-reward RL @ $(date) on $(hostname)"
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
        > "qed_rlhead_s${SEEDS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed=${SEEDS[$i]} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "ALL_ARMS_DONE at $(date)"

OK=0
for s in "${SEEDS[@]}"; do
    grep -q "POST-RL eval\|pre -> post" "qed_rlhead_s${s}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that reached the post-RL eval: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms finished; tracebacks follow"
    grep -hA8 "Traceback" qed_rlhead_s*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -30
fi

echo
echo "=== pre -> post, per seed. RDKit-MAE is TRUTH; head-MAE is the objective. ==="
echo "=== head improving while RDKit does not = the head is being gamed.        ==="
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "weight drift|pre -> post|^(low|mid|high) w=|reward mean|adapter MOVED|early-stop|best snapshot|RDKit-MAE|head-MAE" \
        "qed_rlhead_s${s}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -14
done
echo "SUMMARY_DONE"
