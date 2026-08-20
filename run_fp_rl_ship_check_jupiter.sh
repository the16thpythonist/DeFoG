#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=fp_rl_ship_check_%j.out

# Shipping check: does round 1's kl=0.5 adapter beat the shipped fingerprint@3.0.0
# on an INDEPENDENT target set, and does the fragment reduction hold there?
#
# WHY THIS IS NOT REDUNDANT WITH THE RL JOB'S OWN PRE/POST
# That comparison was paired and used a cached baseline, so its delta is sound --
# but it ran on the 12 targets the RL experiment draws from its own pool, one
# seed. This scores both adapters on the six long-standing comparison targets
# instead, at three metric widths, with the disconnection reporting that was
# added after the fragment problem surfaced. Two independent target sets agreeing
# is worth an hour of a node before changing a user-facing package; the whole
# reason the fragment defect went unnoticed for four experiments is that one
# measurement, repeated, kept telling the same incomplete story.
#
# WHAT WOULD BLOCK SHIPPING
#   - lift materially below 3.0.0's (beyond the ~0.012 noise), or
#   - disconnection not actually lower here
# A small lift loss WITH a real fragment reduction is the expected and acceptable
# outcome: 3.0.0 currently returns roughly one fragment in seven, and validity as
# reported never showed it.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

R=experiments/results
SHIPPED=$R/adapter_fingerprint__zinc/08_08_2026__13_03__pkM7/fp_adapter      # = fingerprint@3.0.0
RL1=$R/adapter_rl_finetune_fp__zinc/09_08_2026__20_13__5SfL/fp_adapter_rl    # round 1, kl=0.5, final EMA

for p in "$SHIPPED" "$RL1"; do
    [ -f "${p}.ckpt" ] || { echo "ERROR: ${p}.ckpt missing"; exit 1; }
done

# Widths come from each checkpoint's own cond_dim (both 1024 here), so they
# cannot be mistyped into conditioning an adapter wrongly.
python -u scripts/compare_fp_adapters.py \
  --base ckpts/zinc_rl2_seed42/best_model \
  --adapter v3:"$SHIPPED" \
  --adapter rl1:"$RL1" \
  --n-per-target 64 --n-baseline 256 --steps 500 --eta 25 \
  --out fp_rl_ship_check_${SLURM_JOB_ID}.json

echo
echo "For reference, the RL job's own paired measurement on ITS 12 targets:"
echo "  fingerprint@3.0.0 (pre)  lift +0.1631   disc 14.65%"
echo "  round-1 kl=0.5   (post)  lift +0.1588   disc 10.49%"
echo "If the six targets here disagree in DIRECTION on either quantity, do not"
echo "ship -- that would mean the effect is target-set-specific, and target-set"
echo "spread on this metric is already known to be ~0.008."
