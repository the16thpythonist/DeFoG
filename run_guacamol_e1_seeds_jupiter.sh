#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:45:00
#SBATCH --output=guacamol_e1_%j.out

# JUPITER: ONE LINK of the E1 GuacaMol base training -- 4 seeds of the same
# protocol recipe, one per GPU. See docs/unconditional-protocol.md.
#
# Settings come from the throughput probe (job 1116289), not from extrapolating
# ZINC:
#   batch  it/s  peak GPU   epochs/10h
#     64   7.58    16.1 GB     15.6
#    128   4.42    38.6 GB     18.2   <- chosen
#    256   2.19    91.4 GB     18.0
#    512    OOM
# Throughput saturates at 128: batch 256 is no faster and sits at 96% of the
# 95 GB card, which is an OOM waiting for an unlucky batch composition.
#
# EPOCHS=75 is the COSINE HORIZON and must stay fixed across all links, or the
# LR schedule restarts on every resume. It is also the actual target: ~19 epochs
# per link x 4 links. 75 epochs x 8,730 steps = ~655k updates, 2.5x the total of
# the ZINC 300-epoch run, because GuacaMol is a much larger dataset.
#
# ---- Chain 4 links -----------------------------------------------------------
#   PREV=""
#   for i in $(seq 1 4); do
#     if [ -z "$PREV" ]; then PREV=$(sbatch --parsable run_guacamol_e1_seeds_jupiter.sh)
#     else PREV=$(sbatch --parsable --dependency=afterany:$PREV run_guacamol_e1_seeds_jupiter.sh); fi
#     echo "link $i = $PREV"
#   done

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SEEDS=(42 43 44 45)
EPOCHS=75
BATCH_SIZE=128
MAX_TIME_HOURS=10.0     # of 11:45, leaving room for the cache build + a clean final checkpoint
NUM_WORKERS=8

if [ ! -f data/guacamol/guacamol_v1_train.smiles ]; then
    echo "ERROR: GuacaMol official split missing. On the LOGIN node run:"
    echo "  python -c 'from defog.data import guacamol_reference as g; g.download_reference()'"
    exit 1
fi

# Bug A from the 978228 post-mortem: concurrent pycomex runs race on
# os.mkdir(namespace_dir) and the losers die with FileExistsError.
mkdir -p experiments/results/training__guacamol_e1

# Same class of race, second instance: all four arms would otherwise encode
# 1.27M molecules simultaneously and each write the ~6 GB cache. Build it once
# here, so the arms only load. Uses the experiment's own _cache_path and
# constants so the key cannot drift from what the arms will look for.
echo "warming the graph cache (encode + round-trip filter over 1.27M molecules)..."
python - <<'PY'
import os, torch
from defog.data import guacamol_reference as gm
import experiments.training__guacamol_e1 as exp

split = gm.load_reference_split()
os.makedirs(exp.GRAPH_CACHE_DIR, exist_ok=True)
path = exp._cache_path(exp.GRAPH_CACHE_DIR, split.provenance,
                       exp.ATOM_TYPES, exp.BOND_TYPES, exp.FILTER_ROUNDTRIP)
if os.path.exists(path):
    print("cache already present:", os.path.basename(path), flush=True)
else:
    tr_g, tr_s, tr_d, st = gm.build_graphs(
        split.train_smiles, atom_types=exp.ATOM_TYPES, bond_types=exp.BOND_TYPES,
        filter_roundtrip=exp.FILTER_ROUNDTRIP, progress=False)
    va_g, va_s, _, _ = gm.build_graphs(
        split.val_smiles, atom_types=exp.ATOM_TYPES, bond_types=exp.BOND_TYPES,
        filter_roundtrip=exp.FILTER_ROUNDTRIP)
    tmp = path + ".partial"
    torch.save({"train_graphs": tr_g, "train_src": tr_s, "train_dec": tr_d,
                "train_stats": st, "val_graphs": va_g, "val_src": va_s}, tmp)
    os.replace(tmp, path)   # rename only when complete: a killed job cannot
                            # leave a half-written cache that loads as valid
    print("cache built: %d train (kept %.4f), %d val -> %s"
          % (len(tr_g), st["kept_fraction"], len(va_g), os.path.basename(path)), flush=True)
PY
if [ $? -ne 0 ]; then echo "ERROR: cache warm-up failed"; exit 1; fi

echo "GuacaMol E1 seed run @ $(date)"
for s in "${SEEDS[@]}"; do
    if [ -f "ckpts/guacamol_e1_seed${s}/last.ckpt" ]; then
        echo "  seed ${s}: resuming from ckpts/guacamol_e1_seed${s}/last.ckpt"
    else
        echo "  seed ${s}: fresh start"
    fi
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u experiments/training__guacamol_e1.py \
        --SEED ${s} \
        --EPOCHS ${EPOCHS} \
        --BATCH_SIZE ${BATCH_SIZE} \
        --CKPT_DIR "'ckpts/guacamol_e1_seed${s}'" \
        --MAX_TIME_HOURS ${MAX_TIME_HOURS} \
        --NUM_WORKERS ${NUM_WORKERS} \
        --SKIP_FINAL_EVAL True \
        --__DEBUG__ False \
        > "guacamol_e1_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "all GuacaMol E1 arms finished at $(date)"
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "epochs completed|RESUMING|fresh start|new best val|train graphs" \
        "guacamol_e1_seed${s}_${SLURM_JOB_ID}.out" | tail -5
done
