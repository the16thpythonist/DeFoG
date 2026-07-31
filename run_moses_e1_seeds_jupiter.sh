#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:45:00
#SBATCH --output=moses_e1_%j.out

# JUPITER: ONE LINK of the E1 MOSES base training -- 4 seeds, one per GPU.
#
# Settings come from the MOSES probe (job 1137600):
#
#   batch  it/s  reserved GB  steps/epoch  epochs/10h  updates/10h
#    256   7.81     13.0          6170        45.6        281k   <- chosen
#    512   4.48     25.8          3085        52.3        161k
#   1024   2.37     51.3          1542        55.4         85k
#   2048    OOM
#
# Batch 256 despite NOT being the fastest in epochs/hour. Larger batches buy
# more epochs but fewer optimiser steps -- step time grows sub-linearly with
# batch, so 1024 gives +21% epochs while cutting updates by 70%, at a learning
# rate tuned for 256. Maximising epochs/hour here would quietly undertrain.
# 256 is also exactly the published recipe (configs/experiment/moses.yaml:
# batch_size 256, lr 2e-4), so no LR rescaling argument is needed, and it leaves
# the widest memory headroom of the three E1 runs (13 of 95.6 GB).
#
# EPOCHS=100 is the COSINE HORIZON and must stay fixed across links, or the LR
# schedule restarts on every resume. At ~45.6 epochs/link on paper, ~41 in
# practice (validation, checkpointing and generation probes are not in the
# probe's measured loop -- GuacaMol's 18.2 became 16-17), 3 links should reach
# it with margin.
#
# No PYTORCH_CUDA_ALLOC_CONF here, unlike the GuacaMol launcher. That flag
# exists to fight fragmentation from wildly varying molecule sizes; MOSES
# molecules are uniformly small (max_nodes_in_a_batch was 26 at EVERY batch
# size), and the probe measured allocated 12.69 GB against reserved 13.00 --
# essentially no fragmentation to fight.
#
# ---- Chain 3 links -----------------------------------------------------------
#   PREV=""
#   for i in $(seq 1 3); do
#     if [ -z "$PREV" ]; then PREV=$(sbatch --parsable run_moses_e1_seeds_jupiter.sh)
#     else PREV=$(sbatch --parsable --dependency=afterany:$PREV run_moses_e1_seeds_jupiter.sh); fi
#     echo "link $i = $PREV"
#   done

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SEEDS=(42 43 44 45)
EPOCHS=100
BATCH_SIZE=256
MAX_TIME_HOURS=10.0     # of 11:45, leaving room for the cache build on link 1
NUM_WORKERS=8

if [ ! -f data/moses/train.csv ]; then
    echo "ERROR: MOSES split missing. On the LOGIN node run:"
    echo "  python -c 'from defog.data import moses_reference as m; m.download_reference()'"
    exit 1
fi

# Bug A from the 978228 post-mortem: concurrent pycomex runs race on
# os.mkdir(namespace_dir) and the losers die with FileExistsError.
mkdir -p experiments/results/training__moses_e1

# Same class of race: all four arms would otherwise encode 1.58M molecules at
# once and each write the same cache. Build it once here so the arms only load.
# Uses the experiment's own _cache_path and constants so the key cannot drift.
echo "warming the graph cache (encoding 1.58M molecules)..."
python - <<'PY'
import os, torch
from defog.data import moses_reference as mr
import experiments.training__moses_e1 as exp

split = mr.load_reference_split()
os.makedirs(exp.GRAPH_CACHE_DIR, exist_ok=True)
path = exp._cache_path(exp.GRAPH_CACHE_DIR, split.provenance,
                       exp.ATOM_TYPES, exp.BOND_TYPES)
if os.path.exists(path):
    print("cache already present:", os.path.basename(path), flush=True)
else:
    tr_g, tr_s, n_skip = mr.build_graphs(
        split.train_smiles, atom_types=exp.ATOM_TYPES, bond_types=exp.BOND_TYPES)
    va_g, _, _ = mr.build_graphs(
        split.val_smiles, atom_types=exp.ATOM_TYPES, bond_types=exp.BOND_TYPES)
    tmp = path + ".partial"
    torch.save({"train_graphs": tr_g, "train_smiles": tr_s,
                "val_graphs": va_g, "n_skipped": n_skip}, tmp)
    os.replace(tmp, path)   # rename only when complete, so a killed job cannot
                            # leave a half-written cache that loads as valid
    print("cache built: %d train (%d skipped), %d val -> %s"
          % (len(tr_g), n_skip, len(va_g), os.path.basename(path)), flush=True)
PY
if [ $? -ne 0 ]; then echo "ERROR: cache warm-up failed"; exit 1; fi

echo "MOSES E1 seed run @ $(date)"
for s in "${SEEDS[@]}"; do
    if [ -f "ckpts/moses_e1_seed${s}/last.ckpt" ]; then
        echo "  seed ${s}: resuming from ckpts/moses_e1_seed${s}/last.ckpt"
    else
        echo "  seed ${s}: fresh start"
    fi
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u experiments/training__moses_e1.py \
        --SEED ${s} \
        --EPOCHS ${EPOCHS} \
        --BATCH_SIZE ${BATCH_SIZE} \
        --CKPT_DIR "'ckpts/moses_e1_seed${s}'" \
        --MAX_TIME_HOURS ${MAX_TIME_HOURS} \
        --NUM_WORKERS ${NUM_WORKERS} \
        --SKIP_FINAL_EVAL True \
        --__DEBUG__ False \
        > "moses_e1_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed ${s} on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "all MOSES E1 arms finished at $(date)"
for s in "${SEEDS[@]}"; do
    echo "--- seed ${s} ---"
    grep -E "epochs completed|RESUMING|fresh start|new best val|train graphs" \
        "moses_e1_seed${s}_${SLURM_JOB_ID}.out" | tail -5
done
