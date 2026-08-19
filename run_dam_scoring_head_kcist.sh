#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=dam_head
#SBATCH --output=dam_head_%j.out

# Fit the scoring head DAM's adjoint needs (docs/dam_design.md, step 8.5 follow-up).
#
# WHY. DAM compares candidate clean graphs drawn from the base's own head. Measured
# on zinc-kek, 84-98% of those do not decode, and PropertyMatchReward floors every
# failure at the same value -- so g(Z) == g(X1_k) for nearly every sample, the
# adjoint collapses to 1, and the update has nothing to learn. The in-situ residual
# said exactly that: median 1.21-1.28, i.e. worse than doing nothing, and it stayed
# there after the estimator's variance was fixed (m=1 -> m=8 narrowed the range from
# [0.87, 7.03] to [1.01, 2.42] without moving the median).
#
# PropertyHead.forward already takes dense tensors and needs no RDKit. What it lacks
# is training on THESE graphs. That is what this fixes.
#
# TWO ARMS, because the reach measurement says the base matters. Node projection gap
# is 0.09-0.58 on the pre-RL E1 base against 0.65-0.87 on the shipped (2x sanity-RL)
# one, same vocabulary -- the RL rounds made the head markedly harder to steer
# through the x1-parameterisation. Fitting a head for each settles which base the
# DAM arm should run against, at no extra wall clock since they share the node.
#
# SCOPE. These heads are for g(Z) and g(X1_k) INSIDE the adjoint only. The RL reward
# on rollout endpoints stays RDKit ground truth, so the GDPO and RAM arms are
# untouched and remain comparable to the historical runs.
#
# Submit from the worktree:  cd ~/Programming/DeFoG-dam && sbatch run_dam_scoring_head_kcist.sh

set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/Programming/DeFoG-dam}"

PY=$HOME/Programming/DeFoG/.venv/bin/python     # worktree shares the main repo's venv
export PYTHONPATH="$PWD"                        # ...but must import defog from HERE
ADAPTER="ckpts/clogp_v11/clogp_adapter.ckpt"
mkdir -p ckpts/heads

for f in "$ADAPTER" ckpts/zinc_kek_shipped.ckpt ckpts/zinc_e1_seed42_kek.ckpt; do
  [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done

echo "DAM scoring heads @ $(date) on $(hostname)"
echo "  defog resolves to: $($PY -c 'import defog;print(defog.__file__)')"
echo "  md5(shipped)=$(md5sum ckpts/zinc_kek_shipped.ckpt | cut -d' ' -f1)"
echo "  md5(pre-RL) =$(md5sum ckpts/zinc_e1_seed42_kek.ckpt | cut -d' ' -f1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False on a GPU node"); sys.exit(1)
print(f"CUDA ok: {torch.cuda.device_count()} device(s)")
PY

run_arm () {           # $1 gpu  $2 tag  $3 base-ckpt
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/fit_dam_scoring_head.py \
      --base "$3" --adapter "$ADAPTER" --property logp \
      --out "ckpts/heads/dam_scoring_head_$2.ckpt" \
      --endpoints 2048 --batch 128 --rollout-steps 250 \
      --t-per-endpoint 4 --draws-per-state 4 --epochs 60 --seed 42 \
      > "dam_head_$2_${SLURM_JOB_ID:-local}.out" 2>&1 &
}

run_arm 0 shipped ckpts/zinc_kek_shipped.ckpt
sleep 15
run_arm 1 prerl   ckpts/zinc_e1_seed42_kek.ckpt
wait

echo "=== done $(date) ==="
for tag in shipped prerl; do
  echo "--- $tag ---"
  tail -6 "dam_head_${tag}_${SLURM_JOB_ID:-local}.out" 2>/dev/null || echo "(no log)"
  [ -f "ckpts/heads/dam_scoring_head_${tag}.ckpt" ] \
    && echo "OK  ckpts/heads/dam_scoring_head_${tag}.ckpt" \
    || echo "FAILED: no checkpoint written for $tag"
done
