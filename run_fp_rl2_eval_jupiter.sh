#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=fp_rl2_eval_%j.out

# Cross-adapter evaluation of RL round 2, with the SIZE-MATCHED strand alongside
# the primary free-size metric.
#
# Round 2 (job 1299180) was launched before the size-matched strand existed, so
# its own pre/post evaluation reports free-size numbers only. This scores every
# round-2 arm against the current best and the shipped 3.0.0 on ONE shared target
# set, at matched metric widths, on both strands.
#
# WHY THE SIZE-MATCHED STRAND IS HERE AT ALL
# Pinning generation to the target's heavy-atom count was worth +0.032 (RL
# adapter) to +0.048 (shipped 3.0.0) -- larger than every architectural lever
# tried on this axis put together (counts +0.014, interior injection +0.024,
# wider trunk and pre-encoder nothing). It is reported as a PARALLEL STRAND and
# never as the headline: supplying the atom count hands the model information the
# fingerprint is supposed to convey, so it measures an easier task and its
# numbers are not comparable to any free-size figure on record.
#
# THE DIAGNOSTIC THAT CAME WITH IT
# corr(target size, generated size) across targets -- whether the adapter infers
# size from the fingerprint at all. Measured -0.132 for the shipped adapter
# (ignores target size entirely, emits ~23.5 atoms regardless) against +0.552 for
# the round-1 RL adapter. Mean-vs-mean would have missed this completely: a model
# that always emits ~23 atoms matches the AVERAGE target size while missing every
# individual target. Watch whether round 2 pushes it further.
#
# Arms are DISCOVERED rather than hard-coded, because round 2's result
# directories do not exist until it finishes.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

R=experiments/results
SHIPPED=$R/adapter_fingerprint__zinc/08_08_2026__13_03__pkM7/fp_adapter
BEST=$R/adapter_rl_finetune_fp__zinc/09_08_2026__20_13__5SfL/fp_adapter_rl

ARGS=( --adapter v3:"$SHIPPED" --adapter rl1:"$BEST" )

# Round-2 arms are identified from the LAUNCH LOGS, not from the metrics JSON.
# Round 2 was submitted before disconnect_delta was recorded in the summary, so
# there is no field in its results that names which arm is which. The log
# filename carries the delta and the log body carries the checkpoint path, and
# that mapping is written by the run itself -- so it cannot be mixed up the way a
# hand-maintained path list can.
RL2_JOB="${RL2_JOB:-1299180}"
n_found=0
for tag in d005 d015 d030 d050; do
    f="zinc_fp_rl2_${tag}_${RL2_JOB}.out"
    [ -f "$f" ] || { echo "  (no log for ${tag})"; continue; }
    # The log line is "Saved RL'd adapter -> <path>" -- match on the ARROW and the
    # filename rather than the prose, so a reworded log message cannot silently
    # reduce this to "no arms found".
    ck=$(grep -o -- "-> /.*fp_adapter_rl\.ckpt" "$f" | tail -1 | sed 's/^-> //')
    if [ -z "$ck" ] || [ ! -f "$ck" ]; then
        echo "  (${tag}: no saved adapter yet)"; continue
    fi
    ARGS+=( --adapter "${tag}:${ck%.ckpt}" )
    n_found=$((n_found+1))
    echo "  ${tag} -> ${ck}"
done

echo "round-2 arms discovered: ${n_found}"
if [ "$n_found" -eq 0 ]; then
    echo "ERROR: no round-2 arms found -- has job 1299180 finished and written"
    echo "rl_fp_metrics.json with disconnect_delta? Refusing to report a comparison"
    echo "that silently contains only the two reference adapters."
    exit 1
fi

python -u scripts/compare_fp_adapters.py \
  --base ckpts/zinc_rl2_seed42/best_model \
  "${ARGS[@]}" \
  --size-matched \
  --n-per-target 64 --n-baseline 256 --n-size-baseline 128 \
  --steps 500 --eta 25 \
  --out fp_rl2_eval_${SLURM_JOB_ID}.json

echo
echo "HOW TO READ THIS"
echo "  PRIMARY is the free-size lift, at any single metric width (they agree on"
echo "  ordering; absolutes are not comparable across widths)."
echo "  CONNECTIVITY FLOOR: round 1 reached disc 0.105. An arm above that gave back"
echo "  connectivity and is excluded from selection -- rule fixed before round 2 ran."
echo "  SIZE-MATCHED is the parallel strand; read its 'delta' as headroom that size"
echo "  mismatch is still costing, not as a score."
echo "  corr(target size, generated size): shipped -0.13, round-1 RL +0.55. If an"
echo "  arm pushes this higher its free-size lift should rise too -- that would tie"
echo "  the two findings together and make size inference an explicit training goal."
echo
echo "  Noise floors, so nothing marginal gets over-read: run-to-run ~0.012 lift,"
echo "  target-set spread ~0.008."
