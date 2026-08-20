#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=fp_size_matched_%j.out

# SIZE-MATCHED companion metric for the current best adapter.
#
# THE QUESTION
# When we condition on a target's fingerprint and score the similarity of what
# comes out, generation currently draws its atom count from the base's
# unconditional size prior. A generated molecule of the wrong size carries a
# residual dissimilarity by construction, so the principled companion measurement
# is: pin the atom count to the target's and score again.
#
# WHAT IS ALREADY KNOWN, so this run is read correctly
# Size mismatch is NOT what caps achieved similarity:
#   - a 4-atom gap caps Tanimoto at 0.88, a 12-atom gap at 0.71; we sit at ~0.30
#   - a target's TRUE nearest neighbours in ZINC average a 3.94-atom gap, barely
#     better than the 4.70 of random pairs -- high similarity does not require
#     matching size
#   - the 16-atom target's 50 best matches average 23.5 atoms (a 7.6-atom gap)
#     and score the HIGHEST similarity of any target, 0.419
#   - forcing exact size-matching in a random ZINC pool moves mean Tanimoto by
#     +0.0012
# So the expected delta is small and strongly size-dependent (+0.016 for the
# 28-atom target, -0.020 for the 16-atom one): a redistribution across targets
# rather than a gain. This run measures it directly instead of arguing about it.
#
# WHY IT IS A COMPANION AND NEVER THE PRIMARY NUMBER
# Morgan bit count correlates with molecule size, so "how big should this be" is
# part of what the fingerprint conveys. Supplying the count externally hands the
# model an answer it is supposed to infer, making this a strictly easier task
# whose numbers are not comparable to any free-size figure we have recorded.
#
# THE BASELINE IS ALSO RE-DRAWN AT EACH TARGET'S SIZE. A size-matched numerator
# over a free-size denominator would book the size effect itself as steering,
# which is precisely the confound this metric exists to isolate.
#
# THE MORE INTERESTING QUESTION, which nothing has ever measured: is the model
# inferring the right size on its own? Reported as corr(target size, generated
# size) ACROSS targets -- not mean-vs-mean, which a model that always emits
# ~22-atom molecules would pass while missing every individual target.
#
# ADAPTERS: the current best (round-1 kl=0.5, which beat the shipped 3.0.0 on
# lift, validity and fragments on this same target set) plus 3.0.0 for reference,
# so the companion metric has something to be read against.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

R=experiments/results
SHIPPED=$R/adapter_fingerprint__zinc/08_08_2026__13_03__pkM7/fp_adapter          # fingerprint@3.0.0
BEST=$R/adapter_rl_finetune_fp__zinc/09_08_2026__20_13__5SfL/fp_adapter_rl       # round-1 kl=0.5, current best

for p in "$BEST" "$SHIPPED"; do
    [ -f "${p}.ckpt" ] || { echo "ERROR: ${p}.ckpt missing"; exit 1; }
done

python -u scripts/compare_fp_adapters.py \
  --base ckpts/zinc_rl2_seed42/best_model \
  --adapter best:"$BEST" \
  --adapter v3:"$SHIPPED" \
  --size-matched \
  --n-per-target 64 --n-baseline 256 --n-size-baseline 128 \
  --steps 500 --eta 25 \
  --out fp_size_matched_${SLURM_JOB_ID}.json

echo
echo "HOW TO READ THIS"
echo "  'sm lift' is the size-matched figure; 'delta' is how much similarity the"
echo "  size mismatch was costing. Prior evidence says delta should be SMALL"
echo "  (~+0.001 on average) and concentrated on the large targets."
echo "  A large delta would overturn that analysis and would be worth chasing."
echo "  corr(target size, generated size) answers the separate question of whether"
echo "  the model infers size from the fingerprint at all. Near 0 would mean the"
echo "  size information in the fingerprint is simply not being used -- a concrete,"
echo "  fixable gap, and a better lead than anything left on the similarity axis."
