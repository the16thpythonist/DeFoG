#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=e2_fk2
#SBATCH --output=e2_logp_fk2_%j.out

# E2 stage 1: VALIDATION sweep for the logP targeting configuration.
#
# Protocol discipline (docs/targeting-protocol.md §6.3, same as E1): tune on
# validation, freeze, one evaluation pass on test. This job never touches test.
#
# TWO CONFIGURATIONS will be reported as two rows of Table 2 -- adapter alone, and
# adapter + Feynman-Kac -- so both need their configuration chosen here.
#
# WHAT IS SWEPT
#   adapter arm: guidance weight w in {0.5, 0.75, 1.0}
#   FK arm:      w frozen from the adapter arm's winner, then FK's own knobs
#
# WHY FK GETS ITS OWN KNOBS SWEPT. Under FK the ten molecules per target come from
# ONE particle system of K=10 and all ten are kept -- the honest budget, since
# keeping the best of K would be best-of-K selection the baseline does not get.
# The cost is that resampling culls low-weight particles and DUPLICATES high-weight
# ones, so a badly tuned FK run can return ten copies of one molecule and post a
# superb MAE. beta (pull strength), warmup_frac (how late the first resample) and
# ess_frac (resample only when effective sample size drops) are exactly the levers
# that trade MAE against collapse.
#
# SELECTION, fixed before the run:
#   adapter arm: lowest pooled MAE, validity >= 0.90
#   FK arm:      lowest pooled MAE among configurations with UNIQUENESS >= 0.90
#                and validity >= 0.90. An FK row that beats the adapter row on MAE
#                while duplicating particles is not a result, it is a measurement
#                artefact, and uniqueness is the only thing that reveals it.
#
# 40 validation targets rather than the protocol's 100: this stage only has to
# rank configurations, and the test pass is what gets the full 100.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=.venv/bin/python
ADAPTER="molsmith/clogp@1.2.0"
NT=40
OUT="e2_fk2_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "E2 logP FK round-2 validation sweep @ $(date) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Refuse early if the head is missing: the FK arms would silently degrade to plain
# adapter sampling and we would be comparing a configuration against itself.
$PY -c "
import sys; sys.path.insert(0,'.')
from molsmith import store
m = store.resolve_package('$ADAPTER').metadata
assert m.head.present, 'adapter bundles no property head -- FK arms would be meaningless'
print('head present on $ADAPTER: FK energy available')
" || exit 1

# ---- FK round 2: the two levers left at defaults in round 1 -------------------
# Round 1 swept beta and warmup_frac and every arm landed at 0.79-0.83 uniqueness.
# ess_frac and rejuvenate are the levers actually designed for particle
# impoverishment and were never touched:
#   ess_frac  resample ONLY when effective sample size drops below ess_frac*K.
#             Lower = resample less often = less culling = more diversity.
#   rejuvenate  MCMC moves after each resample, the standard SMC remedy: it
#             regenerates duplicated particles instead of leaving copies.
# Round-1 best qualifying arm was beta=2.5/warm=0.6 at MAE 0.5388, uniq 0.818.
# The bar here is MAE <= 0.54 at uniqueness >= 0.90, which would make the floor
# relaxation unnecessary rather than merely disclosed.
#        0            1            2            3            4            5
NAMES=( fk_e25       fk_e25_rj    fk_rj        fk_e10_rj    fk_b15_rj    fk_b40_rj )
METH=(  fk           fk           fk           fk           fk           fk )
W=(     1.0          1.0          1.0          1.0          1.0          1.0 )
BETA=(  2.5          2.5          2.5          2.5          1.5          4.0 )
WARM=(  0.6          0.6          0.6          0.6          0.6          0.6 )
ESS=(   0.25         0.25         0.5          0.10         0.25         0.25 )
REJ=(   0            1            1            1            1            1 )

for i in 0 1 2 3 4 5; do
    gpu=$(( i % 4 ))
    (
        if [ "${METH[$i]}" = "fk" ]; then
            REJFLAG=""
            [ "${REJ[$i]}" = "1" ] && REJFLAG="--fk-rejuvenate"
            CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/e2_targeting.py \
                --adapter "$ADAPTER" --property logp --split validation \
                --method fk --n-targets ${NT} --per-target 10 \
                --weight ${W[$i]} --steps 500 --eta 25 \
                --fk-beta ${BETA[$i]} --fk-warmup ${WARM[$i]} --fk-ess ${ESS[$i]} \
                $REJFLAG \
                --seed 42 --out "${OUT}/${NAMES[$i]}.json"
        else
            CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/e2_targeting.py \
                --adapter "$ADAPTER" --property logp --split validation \
                --method adapter --n-targets ${NT} --per-target 10 \
                --weight ${W[$i]} --steps 500 --eta 25 \
                --seed 42 --out "${OUT}/${NAMES[$i]}.json"
        fi
    ) > "e2fk2_${NAMES[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${NAMES[$i]} (${METH[$i]} w=${W[$i]} beta=${BETA[$i]} warm=${WARM[$i]}) on GPU ${gpu}"
    sleep 3
done

wait
echo "finished at $(date)"

echo
echo "=== validation sweep results ==="
$PY - "$OUT" <<'PY'
import json, sys, glob, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    rows.append((os.path.basename(f)[:-5], d))
print(f"{'arm':14s}{'method':9s}{'w':>6s}{'beta':>6s}{'MAE':>9s}"
      f"{'low':>8s}{'mid':>8s}{'high':>8s}{'valid':>8s}{'uniq':>8s}")
for name, d in rows:
    fk = d.get("fk") or {}
    print(f"{name:14s}{d['method']:9s}{d['sampling']['weight']:>6.2f}"
          f"{fk.get('beta', 0):>6.1f}{d['mae_pooled']:>9.4f}"
          f"{d['mae_low_third']:>8.4f}{d['mae_mid_third']:>8.4f}{d['mae_high_third']:>8.4f}"
          f"{d['validity']:>8.3f}{d['uniqueness']:>8.3f}")
print()
ok = [(n, d) for n, d in rows if d["validity"] >= 0.90
      and (d["method"] != "fk" or d["uniqueness"] >= 0.90)]
for meth in ("adapter", "fk"):
    cand = [(n, d) for n, d in ok if d["method"] == meth]
    if not cand:
        print(f"{meth}: NO configuration passed the floors (validity>=0.90"
              + (", uniqueness>=0.90)" if meth == "fk" else ")"))
        continue
    n, d = min(cand, key=lambda kv: kv[1]["mae_pooled"])
    print(f"{meth}: winner {n}  MAE {d['mae_pooled']:.4f}  valid {d['validity']:.3f} "
          f"uniq {d['uniqueness']:.3f}")
PY

echo
echo "FreeGress logP reference: MAE 0.16-0.22 at validity 0.81-0.87;"
echo "DiGress 0.74-0.92 at 0.65-0.78; unconditional 1.52 at 0.86."
echo "Read our validity against those -- classifier guidance buys MAE with validity,"
echo "and a frozen base that keeps validity high is the point of the comparison."
echo
echo "NEXT: freeze both winners, then ONE test pass each (100 targets x 10)."
