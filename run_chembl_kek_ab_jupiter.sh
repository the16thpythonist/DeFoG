#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:30:00
#SBATCH --output=chembl_kek_ab_%j.out

# JUPITER: ONE LINK of a kekulized ChEMBL run, as a controlled A/B against the
# released aromatic v1.
#
# ---- What this tests ---------------------------------------------------------
# The shipped aromatic vocabulary cannot round-trip its own training data:
# encoding and decoding real ChEMBL molecules through pyg_data_to_mol loses
# 14.07% of them (n=50,000), essentially all KekulizeException. Under the
# kekulized vocabulary the same measurement loses 0.02%, and the residue is 10
# AtomValenceExceptions that are IDENTICAL under both -- i.e. representation-
# independent. On the model side, diagnose_validity.py on the v2 checkpoint
# (n=2048) finds 120 of 129 hard failures are kekulization: 93%, matching the
# 98% measured on MOSES, where removing the AROMATIC class moved validity
# 0.884 -> 0.991.
#
# ---- Falsification condition -------------------------------------------------
# EVERY knob below is identical to run_chembl_ddp_chain_jupiter.sh -- same lr,
# cosine horizon, per-rank batch, wall-clock cap, architecture and data -- so
# the representation is the only variable. Compare at MATCHED step count against
# the aromatic link-1 numbers:
#
#     aromatic link1 (~ep13):  validity 0.838   sanity 0.772   connected 0.938
#
# If validity does not rise well clear of 0.84 toward the ~0.99 that kekulized
# MOSES and ZINC reach, then kekulization is not what limits ChEMBL and the
# union run should NOT be launched on this premise.
#
# NOTE the noise floor: at n=2048 differences below ~0.01 are not real. The
# expected effect here is ~0.10, well clear of it.
#
# ---- Not interchangeable with the aromatic lineage ---------------------------
# This produces 12 atom / 4 edge checkpoints. The released v1/v2 are 12 / 5.
# Decoding either with the other's vocabulary yields plausible molecules made of
# the wrong elements rather than an error, so pass --representation kekulized_v2
# to every eval touching these checkpoints. train_chembl_ddp.py refuses on a
# channel-count mismatch, and refuses to start if the stats file (the noise
# prior) was built for a different bond vocabulary.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

LR=3e-4
EPOCHS=60                        # cosine horizon -- MUST match the aromatic run
REPRESENTATION=kekulized_v2
CKPT_DIR="ckpts/chembl_kek_ab_lr${LR}"

if [ ! -f data/chembl/chembl_train.smiles ]; then
    echo "ERROR: data/chembl/chembl_train.smiles missing (stage prepared data first)"; exit 1
fi
# The marginals ARE the noise prior and differ per bond vocabulary: dropping the
# AROMATIC class moves 3.55% of all atom pairs into single/double. Training from
# the aromatic prior would confound the very thing this run isolates.
if [ ! -f data/chembl/chembl_kek_stats.json ]; then
    echo "ERROR: data/chembl/chembl_kek_stats.json missing. Generate with:"
    echo "  python scripts/compute_graph_stats.py --dataset chembl \\"
    echo "      --representation kekulized_v2 --smiles data/chembl/chembl_train.smiles \\"
    echo "      --out data/chembl/chembl_kek_stats.json"
    exit 1
fi

echo "ChEMBL KEKULIZED A/B link @ $(date); CKPT_DIR=$CKPT_DIR"
[ -f "$CKPT_DIR/last.ckpt" ] && echo "  -> resuming from last.ckpt" || echo "  -> fresh start"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

srun python -u scripts/train_chembl_ddp.py \
    --representation ${REPRESENTATION} \
    --devices 4 --num-nodes 1 --lr ${LR} --epochs ${EPOCHS} \
    --max-time-hours 9.5 --batch-size 64 --num-workers 8 \
    --ckpt-dir "${CKPT_DIR}"

echo "ChEMBL KEKULIZED A/B link finished @ $(date)"

# ---- After this link: extended eval, single GPU -------------------------------
#   python scripts/train_chembl_ddp.py --eval-only --representation kekulized_v2 \
#       --eval-ckpt ckpts/chembl_kek_ab_lr3e-4/best_model.ckpt
