#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=zinc_heads
#SBATCH --output=zinc_heads_%j.out

# Train PropertyHeads for logP and QED on the zinc-kek base.
#
# WHY: E2 uses Feynman-Kac steering, and FK needs a learned energy --
# LearnedPropertyEnergy scores each predicted-clean particle by the squared error
# of a PropertyHead's prediction to the target. NO zinc-kek adapter ships a head:
# clogp@1.0.0, clogp@1.1.0 and all three fingerprint adapters have
# head.present = false. The heads that exist (logp, qed, tpsa, sascore, logd) are
# all on the OLD aromatic zinc-base and cannot be served here -- the atom ORDER
# differs, not just the bond set.
#
# A head does NOT need its adapter. fit_property_head grounds a graph -> scalar
# regressor on the same (graph, label) pairs, with no adapter in the loop, so
# these attach to clogp@1.1.0 (already trained) and to the QED adapter currently
# training in job 43027.
#
# LABELS MUST MATCH THE ADAPTER'S. property_from=decoded, same as clogp@1.1.0 and
# the running QED arms. A head trained on source-SMILES labels paired with an
# adapter trained on decoded-graph labels would disagree about what a target means
# exactly where the charge loss bites (the low-logP end), and the FK energy would
# then pull against the adapter instead of with it.
#
# ONE SEED PER PROPERTY, by decision. A second head at a different seed would be
# an independent ruler for checking the first; not done here. Note the consequence:
# nothing cross-checks these heads, so the held-out MAE printed below is the only
# evidence they carry signal. The script warns if MAE is no better than predicting
# the dataset mean, which would make the FK energy a silent no-op.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"

# Overridable. ckpts/zinc_rl2_seed42/best_model and ckpts/zinc_kek_base are
# weight-identical (verified: 524 tensors, 0 differing), so either names the same model,
# but the adapters under test were trained against zinc_kek_base and the head should say
# so. PROPS is settable so a single missing head can be trained without retraining the
# ones that already exist.
BASE="${BASE:-ckpts/zinc_kek_base}"
VOCAB="${VOCAB:-e1_kekulized}"
PY=.venv/bin/python
mkdir -p ckpts/heads

[ -f "${BASE}.ckpt" ] || { echo "ERROR: ${BASE}.ckpt missing"; exit 1; }

echo "ZINC property heads (zinc-kek) @ $(date) on $(hostname)"
echo "  base=${BASE}  vocab=${VOCAB}  property_from=decoded  seed=0"
echo "  md5(base)=$(md5sum ${BASE}.ckpt | cut -d' ' -f1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False on a GPU node"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

IFS=" " read -r -a PROPS <<< "${PROPS:-logp qed}"
for i in "${!PROPS[@]}"; do
    p=${PROPS[$i]}
    CUDA_VISIBLE_DEVICES=$(( i % 2 )) $PY -u scripts/train_property_head.py \
        --base "$BASE" \
        --vocabulary "$VOCAB" \
        --property "$p" \
        --property-from decoded \
        --hidden 128 --layers 3 \
        --epochs 60 --lr 1e-3 --batch-size 32 \
        --seed 0 --holdout 5000 \
        --out "ckpts/heads/${p}_head" \
        > "zinc_head_${p}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${p} head on GPU $(( i % 2 )) (pid $!)"
    sleep 3
done

wait
echo "finished at $(date)"

OK=0
for p in "${PROPS[@]}"; do
    [ -f "ckpts/heads/${p}_head.ckpt" ] && OK=$((OK+1))
done
echo "heads written: ${OK} / 2"
if [ "$OK" -lt 2 ]; then
    echo "ERROR: not all heads trained; tracebacks follow"
    grep -hA6 "Traceback" zinc_head_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -20
fi

echo
echo "=== held-out quality (the only evidence these heads carry signal) ==="
for p in "${PROPS[@]}"; do
    echo "--- ${p} ---"
    grep -E "graphs \(skipped|held-out MAE|WARNING|head: [0-9,]+ params" \
        "zinc_head_${p}_${SLURM_JOB_ID}.out" 2>/dev/null
done

echo
echo "HOW TO READ THIS"
echo "  MAE/std is the number that matters: a head at MAE >= std is no better than"
echo "  predicting the dataset mean and would make LearnedPropertyEnergy a no-op,"
echo "  so FK steering would appear to run and change nothing."
echo "  QED spans only ~0.45 between its 5th and 95th percentile, so its ABSOLUTE"
echo "  MAE will look small next to logP's regardless of quality -- compare each"
echo "  against its own std, not against the other property."
echo
echo "NEXT: repackage with molsmith adapter migrate --head, giving clogp a new"
echo "version and bundling the QED head with the winning arm of job 43027."
