#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:30:00
#SBATCH --output=union_ddp_%j.out

# JUPITER: ONE LINK of the scaled ZINC∪ChEMBL foundation run -- 4-GPU DDP training
# of the SAME 25.9M model (12L/384) on the ~100M union set, KEKULIZED.
#
# ---- Why kekulized (measured, not assumed) -----------------------------------
# The aromatic vocabulary cannot round-trip its own training data: 14.07% of real
# ChEMBL molecules and 8.07% of cleaned ZINC fail to survive encode->decode, almost
# entirely KekulizeException. Kekulized loses 0.02% and 1-in-299,794 respectively.
# The controlled A/B (job 1282763) held every knob fixed and moved validity
# 0.838 -> 0.984 and sanity 0.772 -> 0.940 in ONE 9.5h link -- beating aromatic v1
# (39 epochs, 3 links) and v2 (v1 + a GDPO RL round) while holding kl_score.
# Weighted for this union (~100M ZINC + 2.4M ChEMBL), aromatic would be lossy on
# ~8.2% of the training set.
#
# ---- Scale ------------------------------------------------------------------
# ~100M @ ~30h/epoch -> EPOCHS=2 horizon ~= 6-7 links. Note that ONE epoch here is
# ~390k optimizer steps at effective batch 256, which is MORE total optimization
# than the entire released ChEMBL model received (~371k steps over 39 epochs). So
# even a partial first epoch is a complete training run, not a warm-up. Submit
# link-by-link and reassess, as the ChEMBL run did.
#
# ---- Two things that will silently produce garbage if skipped ----------------
# 1. The stats file is the NOISE PRIOR under noise_type="marginal", and it is
#    per-vocabulary: dropping AROMATIC moves ~3.5% of all atom pairs into
#    single/double. prepare_smiles_union.py writes an AROMATIC union_stats.json,
#    which is NOT usable here -- generate the kekulized one with
#    compute_graph_stats.py (below). train_chembl_ddp.py refuses to start on a
#    mismatch rather than training from the wrong prior.
# 2. These checkpoints are 12 atom / 4 edge. The released v1/v2 are 12 / 5.
#    Decoding either with the other's vocabulary yields plausible molecules made
#    of the wrong elements rather than an error, so pass --representation
#    kekulized_v2 to every eval.
#
# Chain: PREV=""; for i in $(seq 1 N); do
#   if [ -z "$PREV" ]; then PREV=$(sbatch --parsable run_union_ddp_chain_jupiter.sh)
#   else PREV=$(sbatch --parsable --dependency=afterany:$PREV run_union_ddp_chain_jupiter.sh); fi
#   echo "link $i = $PREV"; done

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

DATA_DIR="data/zinc_chembl_union"
PREFIX="union"
REPRESENTATION=kekulized_v2
LR=3e-4
EPOCHS=2                          # cosine horizon (total planned epochs, fixed across links)
CKPT_DIR="ckpts/foundation_union_kek_lr${LR}"

if [ ! -f "$DATA_DIR/${PREFIX}_train.smiles" ]; then
    echo "ERROR: $DATA_DIR/${PREFIX}_train.smiles missing (stage the union data first)"; exit 1
fi
if [ ! -f "$DATA_DIR/${PREFIX}_kek_stats.json" ]; then
    echo "ERROR: $DATA_DIR/${PREFIX}_kek_stats.json missing. The aromatic"
    echo "union_stats.json written by prepare_smiles_union.py is the WRONG prior."
    echo "Generate with:"
    echo "  python scripts/compute_graph_stats.py --dataset chembl \\"
    echo "      --representation kekulized_v2 --smiles $DATA_DIR/${PREFIX}_train.smiles \\"
    echo "      --out $DATA_DIR/${PREFIX}_kek_stats.json"
    exit 1
fi

# KL reference descriptors (25k sample) for the extended eval -- generate once.
if [ ! -f "$DATA_DIR/${PREFIX}_ref_descriptors.npz" ]; then
    echo "precomputing ${PREFIX}_ref_descriptors.npz..."
    python -c "import numpy as np; from experiments.utils import property_distributions; \
s=[l.strip() for l in open('$DATA_DIR/${PREFIX}_train.smiles') if l.strip()]; \
np.savez('$DATA_DIR/${PREFIX}_ref_descriptors.npz', **property_distributions(s, 25000, 42))"
fi

echo "union KEKULIZED DDP foundation link @ $(date); CKPT_DIR=$CKPT_DIR"
[ -f "$CKPT_DIR/last.ckpt" ] && echo "  -> resuming from last.ckpt" || echo "  -> fresh start"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# --gen-num-samples 256 --gen-eta 0: the historical 64-sample/eta=5 probe does NOT
# track the metric decisions are made on. On the ChEMBL A/B it selected a
# best_model that the eta=0 eval rated WORSE than the end-of-link model on every
# metric. Prefer the end-of-link EMA model (foundation_model.ckpt) over
# best_model.ckpt unless the probe is this large.
srun python -u scripts/train_chembl_ddp.py \
    --data-dir "$DATA_DIR" --prefix "$PREFIX" \
    --representation ${REPRESENTATION} \
    --devices 4 --num-nodes 1 --lr ${LR} --epochs ${EPOCHS} \
    --max-time-hours 9.5 --batch-size 64 --num-workers 8 \
    --gen-num-samples 256 --gen-eta 0.0 \
    --ckpt-dir "${CKPT_DIR}"

echo "union KEKULIZED DDP foundation link finished @ $(date)"

# ---- After a link: extended eval, single GPU ---------------------------------
#   sbatch run_chembl_eval_jupiter.sh \
#       ckpts/foundation_union_kek_lr3e-4/foundation_model.ckpt kekulized_v2
