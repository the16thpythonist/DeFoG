#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=moses_kekrl_b0s43_eval_%j.out

# JUPITER: settle whether the b0_s43 arm's apparent FCD improvement is real.
#
# THE CLAIM UNDER TEST
# In job 1246227, arm b0_s43 (beta=0, RL seed 43) improved sanity +0.0024 AND
# FCD 1.087 -> 1.011, a drop of 0.076. That is the only arm that moved both the
# right way.
#
# WHY IT IS NOT YET BELIEVABLE
# The four arms' `before` measurements sample the SAME base checkpoint four
# times, so their spread IS the measurement noise: FCD std 0.046 at n=2048. A
# before/after difference therefore carries noise 0.046*sqrt(2) = 0.064, making
# -0.076 about 1.2 sigma. And b0_s43 was picked as the best of four arms, which
# is a selection effect: with four draws at that noise, the best-looking one
# shows about 1 sigma of apparent improvement by chance alone. Reporting it as
# measured would be the winner's curse.
#
# THE DESIGN THAT CAN ANSWER IT
# n=10,000 instead of 2,048, and -- more importantly -- REPLICATES. Two
# independent sampling seeds for each model:
#
#   GPU 0   b0_s43 RL model   seed 1000
#   GPU 1   b0_s43 RL model   seed 2000     <- replicate
#   GPU 2   moses_kek_seed44  seed 1000     <- the base it started from
#   GPU 3   moses_kek_seed44  seed 2000     <- replicate
#
# The two base runs give the noise floor AT THIS n directly, rather than
# inferring it from a different n. The two RL runs show whether the RL model's
# own value is stable. Only if the RL-vs-base gap is large relative to the
# within-model spread does the improvement mean anything.
#
# FCD IS NOT COMPARABLE ACROSS n -- it is strongly n-biased (measured on this
# project: 5.18 at n=500 against 0.218 at n=12443 for the same distribution).
# So the n=2048 numbers from job 1246227 cannot be compared to these. Everything
# here is measured at one n, in one job, against one reference.
#
# Validation only. This is a selection question, so test stays sealed. If the
# improvement survives, THEN a single frozen test pass is the reportable number.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

RL_CKPT="ckpts/moses_kekrl_b0s43/best_model"
BASE_CKPT="ckpts/moses_kek_seed44/best_model"
N=10000
STEPS=500
ETA=25

#          GPU0        GPU1        GPU2         GPU3
CKPTS=(    "$RL_CKPT"  "$RL_CKPT"  "$BASE_CKPT" "$BASE_CKPT" )
SEEDS=(    1000        2000        1000         2000 )
TAGS=(     rl_s1000    rl_s2000    base_s1000   base_s2000 )

for c in "$RL_CKPT" "$BASE_CKPT"; do
    if [ ! -f "${c}.ckpt" ]; then echo "ERROR: ${c}.ckpt missing"; exit 1; fi
done

echo "b0_s43 confirmation eval @ $(date)"
echo "RL=${RL_CKPT}  BASE=${BASE_CKPT}  n=${N} steps=${STEPS} eta=${ETA}"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK")
PY

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/diagnose_validity.py \
        --ckpt "${CKPTS[$i]}" --dataset moses --representation kekulized_v2 \
        --n ${N} --steps ${STEPS} --eta ${ETA} --seed ${SEEDS[$i]} \
        --dump-smiles "kekrl_eval_${TAGS[$i]}_${SLURM_JOB_ID}.smi" \
        --out "kekrl_eval_${TAGS[$i]}_${SLURM_JOB_ID}.json" \
        > "kekrl_eval_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (${CKPTS[$i]}, seed ${SEEDS[$i]}) on GPU ${i}"
    sleep 3
done
wait

echo
echo "=== sanity-side results, n=${N} ==="
python - "$SLURM_JOB_ID" <<'PY'
import json, sys
jid = sys.argv[1]
print(f"{'run':12s}{'valid+conn':>12s}{'kekulize':>10s}{'valence':>9s}{'disconn':>9s}{'uniq':>8s}")
vals = {}
for tag in ("rl_s1000", "rl_s2000", "base_s1000", "base_s2000"):
    try:
        d = json.load(open(f"kekrl_eval_{tag}_{jid}.json"))
    except Exception as exc:
        print(f"{tag:12s} unreadable ({exc})"); continue
    n = max(1, d["n"]); c = d["counts"]
    ok = c.get("ok", 0) / n
    vals[tag] = ok
    print(f"{tag:12s}{ok:>12.4f}{c.get('kekulize',0)/n:>10.4f}"
          f"{c.get('valence',0)/n:>9.4f}{c.get('disconnected',0)/n:>9.4f}"
          f"{'-':>8s}")
if len(vals) == 4:
    rl = (vals['rl_s1000'] + vals['rl_s2000']) / 2
    ba = (vals['base_s1000'] + vals['base_s2000']) / 2
    spread = max(abs(vals['rl_s1000'] - vals['rl_s2000']),
                 abs(vals['base_s1000'] - vals['base_s2000']))
    print(f"\nvalid+connected: RL {rl:.4f}  base {ba:.4f}  diff {rl-ba:+.4f}")
    print(f"within-model replicate spread: {spread:.4f}  <- the floor")
    if abs(rl - ba) <= spread:
        print("  -> the difference is NOT larger than the noise. Not a real gain.")
PY

echo
echo "NEXT: score the four .smi files for FCD against the validation reference."
echo "Read them as: |FCD(rl) - FCD(base)| must exceed the within-model replicate"
echo "spread before the improvement is real. Two base seeds give that floor"
echo "directly, at this n, in this job."
