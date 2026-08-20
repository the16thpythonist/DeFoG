#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=e2_qed_size
#SBATCH --output=e2_qed_size_%j.out

# QED targeting with a CONDITIONAL size draw, on top of Feynman-Kac.
#
# THE IDEA. Every E2 number so far drew the atom count from the dataset's marginal
# P(n), which ignores the target completely: a request for QED 0.48 and a request
# for QED 0.91 got the same size prior, and the denoiser then spent the whole
# trajectory fighting a size draw that was wrong for the target. This job draws
# from P(n | QED = target) instead.
#
# THE HEADROOM IS MEASURED, NOT ASSUMED (ZINC reference train, 20k molecules):
#     E[n | QED decile]  26.6 atoms at the bottom -> 21.8 at the top
#     marginal           23.2 regardless
#     swing 4.9 atoms = 1.07 sigma of the marginal
# So the low-QED end wants ~26.6 atoms and the marginal hands it 23.2 -- a 3.4-atom
# bias pointing the wrong way, and the low third is exactly where the adapter is
# weakest (E2 MAE 0.1603 against 0.1001 at the high end).
# QED also NARROWS with the target (sd 5.81 at the bottom decile, 2.46 at the top),
# so at high QED this is a variance reduction as well as a bias correction. logP,
# for comparison, swings further (1.50 sigma) but keeps sd ~4.0 throughout.
#
# TWO STAGES, because the size model does not exist yet.
#   1. fit P(n | QED) with train_property_head.py --size-only. This writes
#      ckpts/heads/qed_head_size.ckpt and DOES NOT TOUCH qed_head.ckpt (the script
#      returns before the head branch) -- which matters, because the FK arms below
#      need that head intact.
#   2. the four E2 arms.
# Stage 1 is dominated by dataset preparation (~2 h: encode -> decode -> QED over
# 219,568 molecules, no cache), not by fitting the MLP.
#
# --property-from decoded MATCHES THE ADAPTER AND THE HEAD. A size model fit on
# source labels paired with an adapter trained on decoded labels would disagree
# about what "QED = 0.48" means exactly where the charge loss bites.
#
# THE DESIGN IS A 2x2, so the size effect can be attributed rather than folded into
# a single improved row:
#                        marginal P(n)      learned P(n|QED)
#     adapter alone      0.1212 (job 43064)  arm: ad_size
#     FK beta=1000       0.0965 (job 43064)  arm: fk1000_size
# The fourth arm re-runs adapter+marginal here as a REPRODUCTION CONTROL. Job 43064
# used the same seed and the same 100 targets, so it must come back at 0.1212; if it
# does not, something in the environment moved and the cross-job cells above are not
# comparable.
#
# FK beta=250 also gets a size arm: it is the configuration inside the 80% uniqueness
# floor (0.840 against beta=1000's 0.762), so it is the one most likely to ship.
#
# WATCH gain_nats IN STAGE 1. The fitter warns below 0.02 nats, meaning the property
# carries no size information and the learned draw is a moving part that buys
# nothing. Given the 1.07 sigma swing above that should not trigger -- if it does,
# the measurement and the fit disagree and stage 2 is not worth reading.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=.venv/bin/python
BASE="ckpts/zinc_rl2_seed42/best_model"
ADAPTER="molsmith/qed@2.0.0"
SIZE_CKPT="ckpts/heads/qed_head_size.ckpt"
NT=100
OUT="e2_qedsize_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "E2 QED + conditional size @ $(date) on $(hostname)"
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
assert m.head.present, '$ADAPTER bundles no property head -- the FK arms need it'
print('head present on $ADAPTER; base', m.base.id)
" || exit 1

HEAD_MD5_BEFORE=$(md5sum ckpts/heads/qed_head.ckpt | cut -d' ' -f1)
echo "qed_head.ckpt md5 before stage 1: ${HEAD_MD5_BEFORE}"

# ---- stage 1: fit P(n | QED) -------------------------------------------------
if [ -f "$SIZE_CKPT" ]; then
    echo "stage 1 SKIPPED: ${SIZE_CKPT} already exists"
else
    echo "=== stage 1: fitting P(n | QED) @ $(date) ==="
    $PY -u scripts/train_property_head.py \
        --base "$BASE" --vocabulary e1_kekulized \
        --property qed --property-from decoded \
        --size-only --size-hidden 512 --size-layers 2 --size-epochs 200 \
        --seed 0 --out ckpts/heads/qed_head \
        2>&1 | tee "qed_size_fit_${SLURM_JOB_ID}.out"
    echo "stage 1 finished at $(date)"
fi

[ -f "$SIZE_CKPT" ] || { echo "ERROR: ${SIZE_CKPT} was not written"; exit 1; }

# The head must be byte-identical: the FK arms load it, and a --size-only run that
# silently rewrote it would change what FK is steering toward mid-experiment.
HEAD_MD5_AFTER=$(md5sum ckpts/heads/qed_head.ckpt | cut -d' ' -f1)
if [ "$HEAD_MD5_BEFORE" != "$HEAD_MD5_AFTER" ]; then
    echo "ERROR: qed_head.ckpt CHANGED during the size fit (${HEAD_MD5_BEFORE} -> ${HEAD_MD5_AFTER})"
    exit 1
fi
echo "qed_head.ckpt unchanged: ${HEAD_MD5_AFTER}"

echo
echo "=== size model summary ==="
grep -E "size model:|held-out NLL|WARNING|dropping" "qed_size_fit_${SLURM_JOB_ID}.out" 2>/dev/null || true

# ---- stage 1.5: prove the size draw actually CHANGES something ----------------
# THE FIRST VERSION OF THIS GATE WAS WORTHLESS AND PASSED A DEAD EXPERIMENT.
# It ran the learned-size path for two targets and checked only that a JSON came
# out. It did -- and every number in job 43067 was byte-identical to the marginal
# run (100/100 per-target rows equal to 1e-12), because the cluster's molsmith
# predated SamplingConfig.size_dist and silently ignored the field. Setting an
# undeclared attribute on a dataclass is legal Python, so nothing raised.
#
# A gate that asserts "did not crash" cannot distinguish a working feature from a
# no-op. This one asserts the EFFECT: draw molecules at a LOW QED target with and
# without the size model and require the mean heavy-atom count to move. The table
# says low QED wants ~26.6 atoms against the marginal's 23.2, so a working size
# draw must shift by several atoms; anything under 0.5 means it is not plumbed.
echo
echo "=== stage 1.5: EFFECT test of the learned-size path @ $(date) ==="
$PY - "$SIZE_CKPT" "$ADAPTER" <<'PY'
import sys, importlib.util
sys.path.insert(0, ".")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from molsmith import sample as ms
from defog.core import LearnedSizeDistribution

size_ckpt, adapter = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("e2", "scripts/e2_targeting.py")
e2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e2)
sm = LearnedSizeDistribution.load(size_ckpt)

TARGET = 0.48                      # low QED: wants ~26.6 atoms, marginal gives ~23.2
def run(use_size):
    c = ms.SamplingConfig(
        base="molsmith/zinc-kek", n=24, seed=7, steps=60,
        eta=25, omega=0, time_distortion="polydec",
        adapters=[ms.AdapterTarget(package=adapter, target=TARGET, weight=1.0)],
        method="none")
    if use_size:
        c.size_dist = e2._TargetedSize(sm, TARGET, 24)
    r = ms.sample(c, ms.load(c))
    n = [Chem.MolFromSmiles(s).GetNumHeavyAtoms()
         for s in r.smiles if s and Chem.MolFromSmiles(s)]
    return sum(n) / len(n)

a, b = run(False), run(True)
print(f"  marginal size draw: {a:.2f} heavy atoms")
print(f"  learned  size draw: {b:.2f} heavy atoms   (target QED {TARGET}, table says ~26.6)")
if abs(a - b) < 0.5:
    print("*** NO EFFECT: the size model is not reaching the sampler. REFUSING. ***")
    sys.exit(1)
print("EFFECT CONFIRMED")
PY
[ $? -eq 0 ] || { echo "ERROR: size draw has no effect; not spending four GPUs on a no-op"; exit 1; }

# ---- stage 2: the four E2 arms ----------------------------------------------
echo
echo "=== stage 2: E2 arms @ $(date) ==="
NAMES=( ad_size   fk250_size  fk1000_size  ad_marginal )
METH=(  adapter   fk          fk           adapter )
BETA=(  0         250         1000         0 )
SMODE=( learned   learned     learned      marginal )

for i in 0 1 2 3; do
    EXTRA=""
    [ "${SMODE[$i]}" = "learned" ] && EXTRA="--size-mode learned --size-model ${SIZE_CKPT}"
    if [ "${METH[$i]}" = "fk" ]; then
        CUDA_VISIBLE_DEVICES=$i $PY -u scripts/e2_targeting.py \
            --adapter "$ADAPTER" --property qed --split validation \
            --method fk --n-targets ${NT} --per-target 10 \
            --weight 1.0 --steps 500 --eta 25 \
            --fk-beta ${BETA[$i]} --fk-warmup 0.6 --fk-ess 0.25 \
            ${EXTRA} --seed 42 --out "${OUT}/${NAMES[$i]}.json" \
            > "e2qs_${NAMES[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$i $PY -u scripts/e2_targeting.py \
            --adapter "$ADAPTER" --property qed --split validation \
            --method adapter --n-targets ${NT} --per-target 10 \
            --weight 1.0 --steps 500 --eta 25 \
            ${EXTRA} --seed 42 --out "${OUT}/${NAMES[$i]}.json" \
            > "e2qs_${NAMES[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    fi
    echo "launched ${NAMES[$i]} (${METH[$i]} beta=${BETA[$i]} size=${SMODE[$i]}) on GPU ${i} (pid $!)"
    sleep 3
done

wait
echo "finished at $(date)"

OK=0
for n in "${NAMES[@]}"; do [ -f "${OUT}/${n}.json" ] && OK=$((OK+1)); done
echo "arms complete: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: incomplete; tracebacks follow"
    grep -hA6 "Traceback" e2qs_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== QED: conditional size draw (validation, ${NT} targets, w=1.0, eta=25) ==="
$PY - "$OUT" <<'PY'
import json, os, sys
import numpy as np

d = sys.argv[1]
UNC = 0.743          # unconditional QED mean at eta=25 (job 43036, measured)
PRIOR = {"ad_marginal": 0.1212, "fk1000_marginal": 0.0965, "fk250_marginal": 0.1054}
order = ["ad_marginal", "ad_size", "fk250_size", "fk1000_size"]
print("%-13s%8s%8s%9s%8s%8s%8s%8s%8s%8s" %
      ("arm","beta","size","MAE","skill","low","mid","high","valid","uniq"))
got = {}
for n in order:
    f = os.path.join(d, f"{n}.json")
    if not os.path.exists(f):
        print("%-13s  MISSING" % n); continue
    r = json.load(open(f)); got[n] = r
    tg = np.array([row["target"] for row in r["per_target"]])
    dn = float(np.abs(tg - UNC).mean())
    sk = (1 - r["mae_pooled"] / dn) * 100
    fk = r.get("fk") or {}
    print("%-13s%8.0f%8s%9.4f%7.0f%%%8.4f%8.4f%8.4f%8.3f%8.3f" %
          (n, fk.get("beta", 0), "learned" if "size" in n else "marginal",
           r["mae_pooled"], sk, r["mae_low_third"], r["mae_mid_third"],
           r["mae_high_third"], r["validity"], r["uniqueness"]))

print()
print("REPRODUCTION CONTROL")
if "ad_marginal" in got:
    v = got["ad_marginal"]["mae_pooled"]
    delta = abs(v - PRIOR["ad_marginal"])
    verdict = "OK" if delta < 0.002 else "*** DRIFT -- cross-job cells are NOT comparable ***"
    print(f"  adapter+marginal here {v:.4f} vs job 43064 {PRIOR['ad_marginal']:.4f}"
          f"  |diff| {delta:.4f}  {verdict}")

print()
print("THE 2x2 (what the conditional size draw is worth)")
rows = [("adapter", "ad_marginal", "ad_size", PRIOR["ad_marginal"]),
        ("FK b=250", "fk250_marginal", "fk250_size", PRIOR["fk250_marginal"]),
        ("FK b=1000", "fk1000_marginal", "fk1000_size", PRIOR["fk1000_marginal"])]
for label, mkey, skey, mval in rows:
    s = got.get(skey)
    if s is None:
        continue
    sv = s["mae_pooled"]
    print(f"  {label:10s} marginal {mval:.4f} -> learned {sv:.4f}   "
          f"{(sv-mval)/mval*100:+.1f}%")
print("  (marginal cells for the FK rows come from job 43064, same seed and targets;")
print("   the control above is what licenses reading across jobs.)")

print()
print("reference points")
print("  do-nothing (always emit 0.743)  MAE 0.1083  <- the bar for POSITIVE skill")
print("  FreeGress unconditional 0.15, FreeGress best 0.04, DiGress 0.14-0.15")
print("  measured headroom: E[n|QED] 26.6 -> 21.8 atoms, marginal 23.2 (1.07 sigma)")
print()
print("HOW TO READ IT")
print("  The size draw is a BIAS CORRECTION on the node count, so expect the gain")
print("  in the LOW third first -- that is where the marginal is most wrong (wants")
print("  26.6, gets 23.2). A gain that shows up only in the mid third is more likely")
print("  noise than mechanism.")
print("  Uniqueness should NOT fall relative to the matching marginal arm. If it")
print("  does, the size draw is narrowing the output distribution rather than")
print("  aiming it, and that is a different and less welcome result.")
PY
