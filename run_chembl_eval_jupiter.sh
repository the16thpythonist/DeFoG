#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=chembl_eval_%j.out

# JUPITER: single-GPU extended eval of a ChEMBL foundation checkpoint -- 1000
# samples, 500 steps, eta=0, computing validity / uniqueness / novelty /
# connected / disconnected / sanity / wonky-ring / KL(logP,TPSA,QED). Writes
# eval_metrics.json next to the checkpoint. Pass a checkpoint as $1 (defaults to
# the best_model of the lr=3e-4 foundation run), and optionally a representation
# as $2.
#
# $2 MATTERS: a kekulized checkpoint is 12 atom / 4 edge and the released v1/v2
# are 12 / 5. Decoding one with the other's vocabulary yields plausible molecules
# made of the wrong elements rather than an error, so every metric below would be
# a number describing nothing. Defaulting to empty keeps the historical aromatic
# behaviour for existing checkpoints; train_chembl_ddp.py refuses on a
# channel-count mismatch either way.
#
#   sbatch run_chembl_eval_jupiter.sh ckpts/chembl_kek_ab_lr3e-4/foundation_model.ckpt kekulized_v2

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

CKPT="${1:-ckpts/chembl_foundation_lr3e-4/best_model.ckpt}"
REP="${2:-}"
REP_ARG=""
[ -n "$REP" ] && REP_ARG="--representation $REP"
echo "eval $CKPT ${REP:+(representation=$REP)} @ $(date)"
python -u scripts/train_chembl_ddp.py --eval-only --eval-ckpt "$CKPT" $REP_ARG
echo "eval done @ $(date)"
