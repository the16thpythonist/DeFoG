#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
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
DATA_DIR="${3:-}"
PREFIX="${4:-}"

# $3/$4 MATTER for anything that is not the ChEMBL set. --data-dir/--prefix select
# BOTH the KL reference descriptors AND the training SMILES used as the novelty
# denominator. Left at their defaults, a union checkpoint gets scored against
# ChEMBL's property distribution -- which for a ~97% ZINC-trained model measures
# fit to the wrong target -- and its novelty is computed against a 2.4M subset of
# its own 99.8M training set, which inflates it to ~1.0. Both happened on the
# first union link 1 eval (job 1287871): kl_score read 0.607 and novelty 1.000,
# neither of which described the model. Structural metrics (validity, sanity,
# connected, wonky_ring, uniqueness) use no reference and were unaffected.
#
#   sbatch run_chembl_eval_jupiter.sh <ckpt> kekulized_v2 data/zinc_chembl_union union

# $5 = sample count (default 1000). Raise it when comparing two checkpoints that
# are close: at validity ~0.99 the n=1000 standard error is +-0.0026, so link-to-
# link differences of ~0.003 are unresolvable. n=5000 halves that to +-0.0011.
# Cheap next to the alternative -- one more training link costs ~38 GPU-hours, a
# higher-precision eval costs ~1.3.
N="${5:-}"
# $6/$7 = eta/omega. Until 2026-08-11 evaluate() ignored these entirely and always
# sampled at eta=0/omega=0, so an eval labelled with a sweep-winning config
# silently reported baseline numbers.
ETA="${6:-}"
OMEGA="${7:-}"

REP_ARG=""; [ -n "$REP" ] && REP_ARG="--representation $REP"
DATA_ARG=""; [ -n "$DATA_DIR" ] && DATA_ARG="--data-dir $DATA_DIR"
PREFIX_ARG=""; [ -n "$PREFIX" ] && PREFIX_ARG="--prefix $PREFIX"
N_ARG=""; [ -n "$N" ] && N_ARG="--num-eval-samples $N"
ETA_ARG=""; [ -n "$ETA" ] && ETA_ARG="--eval-eta $ETA"
OM_ARG=""; [ -n "$OMEGA" ] && OM_ARG="--eval-omega $OMEGA"

echo "eval $CKPT ${REP:+(representation=$REP)} ${DATA_DIR:+(data=$DATA_DIR/$PREFIX)} ${N:+(n=$N)} (eta=${ETA:-0} omega=${OMEGA:-0}) @ $(date)"
python -u scripts/train_chembl_ddp.py --eval-only --eval-ckpt "$CKPT" $REP_ARG $DATA_ARG $PREFIX_ARG $N_ARG $ETA_ARG $OM_ARG
echo "eval done @ $(date)"
