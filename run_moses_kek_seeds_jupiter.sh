#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:45:00
#SBATCH --output=moses_kek_%j.out

# JUPITER: ONE LINK of MOSES trained under the KEKULIZED representation.
# 4 seeds, one per GPU. Identical to run_moses_e1_seeds_jupiter.sh in every
# respect except REPRESENTATION -- same epochs, batch, LR, architecture, seeds,
# split and cosine horizon -- so the comparison isolates the representation.
#
# WHY
# scripts/diagnose_validity.py on the aromatic base, n=1024 at the deploy
# config, classified every hard failure:
#
#     ok              896   0.8750
#     kekulize        118   0.1152     <- 98% of hard failures
#     disconnected      8   0.0078
#     other_sanitize    1   0.0010
#     valence           1   0.0010     <- exactly one
#
# The model is not producing impossible valences. It is producing aromatic ring
# systems RDKit cannot kekulize. An AROMATIC bond class is a promise about a
# whole ring system, checked by kekulization; the model asserts it per-edge and
# cannot keep it. Removing the class makes that failure impossible by
# construction. ZINC trains kekulized and reaches ~0.99 validity against
# MOSES's ~0.90.
#
# It also explains the RL reward hack: the cheapest route to validity was
# "emit fewer aromatic rings" (1.88 -> 1.63), because that is where essentially
# all the invalidity lived.
#
# THE H CLASS IS ALSO DROPPED (8 -> 7 atom types). 'H' never appears as an atom
# in MOSES: verified across a random 200,000 train molecules AND all 220 whose
# SMILES literally contain "[H]" -- RDKit folds those into implicit hydrogen
# counts. Those 220 are almost all imino tautomers (214 exocyclic N-H
# double-bonded to an aromatic ring); the amino/imino distinction lives in bond
# ORDER, which the graph carries, so nothing is lost.
#
# ENCODING IS LOSSLESS: 50,000 random train molecules and all 220 "[H]" cases
# round-trip to identical canonical SMILES, zero encode failures.
#
# WHAT WOULD FALSIFY THE HYPOTHESIS
# If validity does NOT rise well above 0.90, the kekulization failures were a
# symptom rather than the cause, and the deficit lives in capacity or training
# instead. That is a real possible outcome: GuacaMol is ALSO aromatic and
# reaches ~0.98, so aromaticity alone does not explain why MOSES specifically
# sits at 0.90. A second factor exists and has not been identified.
#
# THE AROMATIC CHECKPOINTS ARE UNTOUCHED. This writes to ckpts/moses_kek_seed*.
# A kekulized model has 7 atom / 4 edge classes against the aromatic model's
# 8 / 5, so the two are NOT interchangeable -- decoding either with the other's
# vocabulary mis-decodes silently. The experiment asserts model dims against
# the declared representation at startup, and records it in provenance.
#
# ---- Chain 3 links -----------------------------------------------------------
#   PREV=""
#   for i in $(seq 1 3); do
#     if [ -z "$PREV" ]; then PREV=$(sbatch --parsable run_moses_kek_seeds_jupiter.sh)
#     else PREV=$(sbatch --parsable --dependency=afterany:$PREV run_moses_kek_seeds_jupiter.sh); fi
#     echo "link $i = $PREV"
#   done

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SEEDS=(42 43 44 45)
REPRESENTATION="kekulized_v2"
EPOCHS=100              # cosine horizon; MUST stay fixed across links
BATCH_SIZE=256          # unchanged from the aromatic run, deliberately
MAX_TIME_HOURS=10.0
NUM_WORKERS=8

if [ ! -f data/moses/train.csv ]; then
    echo "ERROR: MOSES split missing. On the LOGIN node run:"
    echo "  python -c 'from defog.data import moses_reference as m; m.download_reference()'"
    exit 1
fi

mkdir -p experiments/results/training__moses_e1

# Build the kekulized graph cache once, before the arms start. Four arms
# encoding 1.58M molecules simultaneously would each pay the cost and race on
# the same file. This is a DIFFERENT cache from the aromatic one -- the key
# includes the vocabulary and the kekulize flag -- so it does not collide with
# or invalidate the existing aromatic cache.
echo "warming the kekulized graph cache (encoding 1.58M molecules)..."
REPRESENTATION="$REPRESENTATION" python - <<'PY'
import os, torch
from defog.data import moses_reference as mr
import experiments.training__moses_e1 as exp

rep = mr.get_representation(os.environ["REPRESENTATION"])
split = mr.load_reference_split()
os.makedirs(exp.GRAPH_CACHE_DIR, exist_ok=True)
path = exp._cache_path(exp.GRAPH_CACHE_DIR, split.provenance, rep)
print("representation:", rep.name, rep.atom_types, rep.bond_types,
      "kekulize=%s" % rep.kekulize, flush=True)
if os.path.exists(path):
    print("cache already present:", os.path.basename(path), flush=True)
else:
    tr_g, tr_s, n_skip = mr.build_graphs(split.train_smiles, representation=rep)
    va_g, _, _ = mr.build_graphs(split.val_smiles, representation=rep)
    # A non-trivial skip count means kekulization failed on real data, which
    # would invalidate the premise of this whole run. Fail loudly here rather
    # than train on a silently truncated dataset.
    if n_skip > 0.001 * len(split.train_smiles):
        raise SystemExit("ERROR: %d/%d train molecules failed to encode under %s"
                         % (n_skip, len(split.train_smiles), rep.name))
    tmp = path + ".partial"
    torch.save({"train_graphs": tr_g, "train_smiles": tr_s,
                "val_graphs": va_g, "n_skipped": n_skip}, tmp)
    os.replace(tmp, path)
    print("cache built: %d train (%d skipped), %d val -> %s"
          % (len(tr_g), n_skip, len(va_g), os.path.basename(path)), flush=True)
PY
if [ $? -ne 0 ]; then echo "ERROR: cache warm-up failed"; exit 1; fi

echo "MOSES KEKULIZED seed run @ $(date)"
for s in "${SEEDS[@]}"; do
    if [ -f "ckpts/moses_kek_seed${s}/last.ckpt" ]; then
        echo "  seed ${s}: resuming from ckpts/moses_kek_seed${s}/last.ckpt"
    else
        echo "  seed ${s}: fresh start"
    fi
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

for i in 0 1 2 3; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u experiments/training__moses_e1.py \
        --SEED ${s} \
        --REPRESENTATION "'${REPRESENTATION}'" \
        --EPOCHS ${EPOCHS} \
        --BATCH_SIZE ${BATCH_SIZE} \
        --CKPT_DIR "'ckpts/moses_kek_seed${s}'" \
        --MAX_TIME_HOURS ${MAX_TIME_HOURS} \
        --NUM_WORKERS ${NUM_WORKERS} \
        --SKIP_FINAL_EVAL True \
        --__DEBUG__ False \
        > "moses_kek_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "all MOSES kekulized arms finished at $(date)"
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "representation '|representation check|epochs completed|new best|val_loss" \
        "moses_kek_seed${s}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -4
done

echo
echo "NEXT once the chain reaches 100 epochs:"
echo "  python scripts/diagnose_validity.py --ckpt ckpts/moses_kek_seed42/best_model \\"
echo "      --dataset moses --representation kekulized_v2 --n 1024"
echo "  If the hypothesis holds, the 'kekulize' category should be ~0 (it is"
echo "  unreachable without an AROMATIC class) and validity well above the"
echo "  aromatic base's 0.90. The --representation flag is REQUIRED: without it"
echo "  the script decodes against the 8-type aromatic vocabulary and the"
echo "  channel-count guard will refuse to run."
echo
echo "  Then the validation sweep and one frozen test pass, as for any E1 row."
echo "  scripts/sweep_sampling.py and final_eval.py still assume the default"
echo "  vocabulary and will need the same flag before they can score this model."
