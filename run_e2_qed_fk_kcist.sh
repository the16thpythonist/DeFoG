#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --job-name=e2_qed_fk
#SBATCH --output=e2_qed_fk_%j.out

# How much does Feynman-Kac steering buy on QED, in the direct E2 comparison?
#
# This is the QED analogue of the two logP rows: same adapter, same protocol, the
# only difference being whether FK resampling sits on top. The adapter is
# molsmith/qed@2.0.0 -- the FF-site adapter from job 43027, which is the one we
# have measured end to end.
#
# BETA MUST BE RESCALED FOR QED, AND THIS IS THE WHOLE REASON THIS IS A SWEEP
# RATHER THAN ONE FK ARM. The FK potential is phi = -beta * energy with
# energy = (head.predict(mol) - target)^2, i.e. SQUARED ERROR IN PROPERTY UNITS.
# The logP configuration used beta=2.5, where typical errors are ~0.5-1.0, so
# beta*err^2 lands at 0.6-2.5 and particles are meaningfully discriminated.
# QED's whole distribution has sd 0.121, so typical errors are ~0.13 and
# beta*err^2 = 2.5 * 0.017 = 0.04 -- every particle gets essentially the same
# weight and FK degenerates into plain adapter sampling while appearing to run.
# Matching the logP tilt requires beta scaled by the variance ratio,
# (1.18/0.121)^2 ~= 95, i.e. beta ~= 240. The sweep brackets that on a log scale.
#
# Reusing beta=2.5 here would have produced "FK does nothing for QED", which is a
# statement about unit scaling and not about FK.
#
# ARMS (4 GPUs)
#   adapter   the control -- no FK, the number FK has to beat
#   fk_b50    below the scale estimate
#   fk_b250   at the scale estimate
#   fk_b1000  above it, to find where over-tilting starts costing diversity
#
# EVERYTHING ELSE IS THE FROZEN logP CONFIGURATION: w=1.0, 500 steps, eta=25,
# omega=0, polydec, seed 42, warmup_frac=0.6, ess_frac=0.25, rejuvenate OFF,
# K=10 particles with all ten kept. eta=25 is retained deliberately: the eta sweep
# (job 43036) showed skill flat across eta in {1,5,25} for both properties, so
# changing it here would only break comparability with the logP rows.
#
# READ SKILL, NOT RAW MAE. On 100 real-molecule targets about a third sit near the
# unconditional mean where a constant predictor is unbeatable. The do-nothing
# baseline for QED at eta=25 is MAE 0.1083 against the model's unconditional mean
# of 0.743 (measured, job 43036). The adapter alone scored -12% skill there; the
# question is whether FK turns that positive.
#
# WATCH UNIQUENESS. Resampling clones high-weight particles, so a large beta can
# post a fine MAE with ten near-copies. At high beta this is the expected failure
# mode and it is why 1000 is in the sweep -- to locate it, not to ship it.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=.venv/bin/python
ADAPTER="molsmith/qed@2.0.0"
NT=100
OUT="e2_qedfk_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "E2 QED: adapter vs Feynman-Kac @ $(date) on $(hostname)"
echo "  adapter=${ADAPTER}  ${NT} validation targets x 10"
echo "  frozen: w=1.0 steps=500 eta=25 omega=0 polydec seed=42 warmup=0.6 ess=0.25"
echo "  sweeping FK beta in {50, 250, 1000}; logP used 2.5 on a ~95x larger variance"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Without a head the FK arms silently reduce to the control and we would be
# comparing a configuration against itself three times.
$PY -c "
import sys; sys.path.insert(0,'.')
from molsmith import store
m = store.resolve_package('$ADAPTER').metadata
assert m.head.present, '$ADAPTER bundles no property head -- FK arms would be meaningless'
print('head present on $ADAPTER: FK energy available')
print('base:', m.base.id)
" || exit 1

NAMES=( adapter  fk_b50  fk_b250  fk_b1000 )
METH=(  adapter  fk      fk       fk )
BETA=(  0        50      250      1000 )

for i in 0 1 2 3; do
    if [ "${METH[$i]}" = "fk" ]; then
        CUDA_VISIBLE_DEVICES=$i $PY -u scripts/e2_targeting.py \
            --adapter "$ADAPTER" --property qed --split validation \
            --method fk --n-targets ${NT} --per-target 10 \
            --weight 1.0 --steps 500 --eta 25 \
            --fk-beta ${BETA[$i]} --fk-warmup 0.6 --fk-ess 0.25 \
            --seed 42 --out "${OUT}/${NAMES[$i]}.json" \
            > "e2qfk_${NAMES[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$i $PY -u scripts/e2_targeting.py \
            --adapter "$ADAPTER" --property qed --split validation \
            --method adapter --n-targets ${NT} --per-target 10 \
            --weight 1.0 --steps 500 --eta 25 \
            --seed 42 --out "${OUT}/${NAMES[$i]}.json" \
            > "e2qfk_${NAMES[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    fi
    echo "launched ${NAMES[$i]} (${METH[$i]} beta=${BETA[$i]}) on GPU ${i} (pid $!)"
    sleep 3
done

wait
echo "finished at $(date)"

OK=0
for n in "${NAMES[@]}"; do [ -f "${OUT}/${n}.json" ] && OK=$((OK+1)); done
echo "arms complete: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: incomplete; tracebacks follow"
    grep -hA6 "Traceback" e2qfk_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== QED: adapter vs FK (validation, ${NT} targets, w=1.0, eta=25) ==="
$PY - "$OUT" <<'PY'
import json, os, sys
import numpy as np

d = sys.argv[1]
UNC = 0.743          # model's unconditional QED mean at eta=25 (job 43036, measured)
order = ["adapter", "fk_b50", "fk_b250", "fk_b1000"]
print(f"{'arm':10s}{'beta':>7s}{'MAE':>9s}{'skill':>8s}"
      f"{'low':>8s}{'mid':>8s}{'high':>8s}{'valid':>8s}{'uniq':>8s}{'distinct':>10s}")
base_mae = None
for n in order:
    f = os.path.join(d, f"{n}.json")
    if not os.path.exists(f):
        print(f"{n:10s}   MISSING"); continue
    r = json.load(open(f))
    tg = np.array([row["target"] for row in r["per_target"]])
    dn = float(np.abs(tg - UNC).mean())
    sk = (1 - r["mae_pooled"] / dn) * 100
    fk = r.get("fk") or {}
    # distinct molecules per target, not the ratio -- the ratio falls simply
    # because 10 distinct molecules at one narrow target is harder than 1
    uniq = r["uniqueness"]
    dist = uniq * 10
    print(f"{n:10s}{fk.get('beta',0):>7.0f}{r['mae_pooled']:>9.4f}{sk:>7.0f}%"
          f"{r['mae_low_third']:>8.4f}{r['mae_mid_third']:>8.4f}{r['mae_high_third']:>8.4f}"
          f"{r['validity']:>8.3f}{uniq:>8.3f}{dist:>10.1f}")
    if n == "adapter":
        base_mae = r["mae_pooled"]

if base_mae:
    print()
    print("FK gain over the adapter alone:")
    for n in order[1:]:
        f = os.path.join(d, f"{n}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        rel = (r["mae_pooled"] - base_mae) / base_mae * 100
        print(f"  {n:10s} MAE {r['mae_pooled']:.4f}  {rel:+.1f}%  "
              f"uniqueness {r['uniqueness']:.3f}")

print()
print("reference points")
print("  do-nothing (always emit 0.743)      MAE 0.1083")
print("  adapter alone at eta=25 (job 43036) MAE 0.1212  = -12% skill")
print("  FreeGress unconditional 0.15, FreeGress best 0.04, DiGress 0.14-0.15")
print("  logP for scale: FK at beta=2.5 took K=10 MAE 0.607 -> 0.508 at K=128;")
print("  at K=10 the adapter and FK rows were much closer.")
print()
print("HOW TO READ IT")
print("  A beta that leaves MAE unchanged from the adapter row means the tilt is")
print("  too weak for QED's units -- that is the failure this sweep exists to")
print("  avoid, not evidence about FK.")
print("  A beta that improves MAE while uniqueness falls toward 0.1 is buying the")
print("  number with duplicated particles; check the distinct column before")
print("  believing it.")
PY
