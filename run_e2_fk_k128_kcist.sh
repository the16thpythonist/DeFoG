#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=e2_k128
#SBATCH --output=e2_fk_k128_%j.out

# Does Feynman-Kac steering improve with a larger particle population?
#
# SMC theory says it should: more particles approximate the target distribution
# better and are less prone to impoverishment. This measures it rather than
# assuming it.
#
# CONFIGURATION IS FROZEN FROM THE VALIDATION SWEEPS (jobs 43030, 43031):
#   beta=2.5, ess_frac=0.25, warmup_frac=0.6, rejuvenate OFF, w=1.0,
#   500 steps, eta=25, omega=0, polydec
# Only K varies. Note rejuvenate=off is a MEASURED choice, not a default: the
# controlled pair fk_e25 vs fk_e25_rj showed switching it on DROPPED uniqueness
# 0.902 -> 0.796 and raised MAE 0.558 -> 0.601. Resampling less often was what
# worked, not regenerating particles after the fact.
#
# ALL K PARTICLES ARE SCORED. At K>10 this is deliberately NOT the FreeGress
# protocol -- MAE is over 32/64/128 molecules per target rather than ten -- so
# these numbers are a scaling curve for the component-ablation section, NOT a
# Table 2 row. Table 2 stays at the budget the baselines actually spend: adapter
# (no particles) and FK at K=10.
#
# Keeping ten of a larger K would be the alternative, and both ways of doing it
# are worse: best-of-K is selection the baselines do not get, and random-ten
# spends up to 12.8x the compute for the same ten reported molecules. Scoring all
# K at least measures exactly what it claims to measure.
#
# K=10 IS INCLUDED at 100 targets even though the sweeps already covered it --
# they ran at 40 targets, and a curve whose anchor point came from a different
# sample size is not a curve.
#
# K=128 WAS DROPPED ONCE ON A BAD COST ESTIMATE, then reinstated. The decomposition
# (base 258 ms/step, +adapter 573, +adapter+FK 646 at K=64/20 steps) was measured on
# a local RTX 2060 and then wrongly applied to KCIST, which is ~5-6x faster. Measured
# here: K=10 15.8 s/target, K=32 27, K=64 56 -- so K=128 is ~112 s/target, about 3 h
# for 100 targets rather than the 18 h originally quoted.
#
# The real lesson stands though: past K=32 the GPU is saturated (~79 ms per graph per
# step, flat), so cost is LINEAR in K from there, and the adapter -- not FK -- is what
# doubles it, because classifier-free guidance runs the network twice per step.
#
# WRITES INTO job 43032's OUTPUT DIRECTORY on purpose, so all four K values form one
# curve. Same frozen configuration, same seed, same 100 validation targets; only K
# differs. A fourth point produced under any other conditions would not belong on the
# same axis.
#
# WATCH UNIQUENESS AS WELL AS MAE. A larger population could improve MAE simply
# by having more chances to land near the target while still collapsing onto a
# few distinct molecules. Uniqueness is reported per arm for exactly that reason.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=.venv/bin/python
ADAPTER="molsmith/clogp@1.2.0"
NT=100
OUT="e2_kscale_43032"
mkdir -p "$OUT"

echo "E2 FK particle-count scaling @ $(date) on $(hostname)"
echo "  frozen: beta=2.5 ess=0.25 warm=0.6 rejuvenate=off w=1.0 steps=500 eta=25"
echo "  varying: K = 128, ${NT} validation targets, ALL K scored"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

$PY -c "
import sys; sys.path.insert(0,'.')
from molsmith import store
m = store.resolve_package('$ADAPTER').metadata
assert m.head.present, 'adapter bundles no property head -- FK would be meaningless'
print('head present on $ADAPTER: FK energy available')
" || exit 1

KS=( 128 )
for i in 0; do
    K=${KS[$i]}
    CUDA_VISIBLE_DEVICES=$i $PY -u scripts/e2_targeting.py \
        --adapter "$ADAPTER" --property logp --split validation \
        --method fk --n-targets ${NT} --per-target ${K} \
        --weight 1.0 --steps 500 --eta 25 \
        --fk-beta 2.5 --fk-warmup 0.6 --fk-ess 0.25 \
        --seed 42 --out "${OUT}/K${K}.json" \
        > "e2ks_K${K}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched K=${K} on GPU ${i} (pid $!)"
    sleep 3
done

wait
echo "finished at $(date)"

OK=0
for K in "${KS[@]}"; do [ -f "${OUT}/K${K}.json" ] && OK=$((OK+1)); done
echo "arms complete: ${OK} / 1"
if [ "$OK" -lt 1 ]; then
    echo "ERROR: incomplete; tracebacks follow"
    grep -hA6 "Traceback" e2ks_K*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -20
fi

echo
echo "=== FK scaling with particle count K (validation, ${NT} targets) ==="
$PY - "$OUT" <<'PY'
import json, sys, glob, os, re
rows = []
for f in glob.glob(os.path.join(sys.argv[1], "K*.json")):
    d = json.load(open(f))
    rows.append((int(re.search(r"K(\d+)", os.path.basename(f)).group(1)), d))
rows.sort()
print(f"{'K':>5s}{'molecules':>11s}{'MAE':>9s}{'low':>8s}{'mid':>8s}{'high':>8s}"
      f"{'valid':>8s}{'uniq':>8s}{'sec/target':>12s}")
for K, d in rows:
    n = K * d["n_targets"]
    print(f"{K:>5d}{n:>11d}{d['mae_pooled']:>9.4f}{d['mae_low_third']:>8.4f}"
          f"{d['mae_mid_third']:>8.4f}{d['mae_high_third']:>8.4f}"
          f"{d['validity']:>8.3f}{d['uniqueness']:>8.3f}{'':>12s}")
if len(rows) > 1:
    base = rows[0][1]["mae_pooled"]
    print()
    print("relative to K=10:")
    for K, d in rows[1:]:
        rel = (d["mae_pooled"] - base) / base * 100
        print(f"  K={K:<4d} MAE {d['mae_pooled']:.4f}  {rel:+.1f}%  "
              f"uniqueness {d['uniqueness']:.3f}  ({K/rows[0][0]:.0f}x the compute)")
PY

echo
echo "HOW TO READ THIS"
echo "  A falling MAE with rising K supports the hypothesis that FK is"
echo "  population-limited rather than mechanism-limited."
echo "  But check uniqueness in the same row: a larger population can lower MAE"
echo "  simply by having more chances near the target while still collapsing onto"
echo "  few distinct molecules, and that is not the same claim."
echo "  These are NOT Table 2 numbers -- MAE at K>10 is over more molecules than"
echo "  the protocol's ten, and the compute per target scales with K."
