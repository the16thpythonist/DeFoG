#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=ag_pilot
#SBATCH --output=ag_pilot_%j.out

# Stage 2: AUTOGUIDANCE PILOT -- does a degraded conditional negative branch buy
# validity headroom at higher guidance weight?
#
# THE HYPOTHESIS, WRITTEN DOWN BEFORE THE RUN
# Standard CFG pushes away from the UNCONDITIONAL base, so raising w drags samples
# off the data manifold and validity collapses: 0.982 / 0.898 / 0.466 at w = 2/3/4
# (RESEARCH.md, prob-space blend). Autoguidance pushes away from a WEAK CONDITIONAL
# instead. Both models share the base's flaws, so the flaws cancel in the difference
# and what is amplified is closer to pure quality. MolGuidance found exactly this
# asymmetry: CFG wins property alignment, autoguidance is the balanced arm and
# IMPROVES structural validity.
#
# PREDICTION (pre-registered):
#   * at w=2 the two arms are close on MAE -- autoguidance is not expected to win here
#   * at w=3 and w=4 autoguidance holds validity materially above CFG
#   * if autoguidance's MAE at w=3-4 beats CFG's best-validity-constrained MAE, the
#     lever is real and earns a full 100-target run
# A uniform improvement at every w, including w=2, is NOT the predicted signature and
# should be read as suspicion of a bug (most likely the guide not actually being wired
# into group 0) rather than as a stronger result. `test_guide_at_init_is_plain_cfg`
# pins the wiring down, but a passing unit test is not a passing experiment.
#
# WHY 30 TARGETS. This stage only has to tell us whether to spend a 100-target run.
# The harness's measured seed spread is 0.0082 MAE, so 30 targets resolves anything
# above ~0.02. The two SEED-DUPLICATE arms at w=3 exist to measure that spread HERE,
# on validity as well as MAE, rather than importing a number measured elsewhere --
# without them a validity difference has no scale to be read against.
#
# NOT A TEST-SPLIT RUN. Validation only. This informs a choice, so it may not touch
# test (docs/targeting-protocol.md).

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
ADAPTER="molsmith/clogp@1.2.0"
NT=30
STEPS=500
ETA=25

# ---------------------------------------------------------------------------
# SET THIS from run_autoguidance_guide_kcist.sh's output. ep2 of 3 is the default
# pick: ep1 risks being so weak it is effectively the unconditional base (which
# would silently reduce autoguidance to plain CFG), ep3 risks being good enough
# that the difference vanishes.
# ---------------------------------------------------------------------------
GUIDE="${GUIDE:-}"
if [ -z "$GUIDE" ]; then
    GUIDE=$(ls -t experiments/results/adapter_training__zinc/*/clogp_adapter_ep2.ckpt 2>/dev/null | head -1)
fi
if [ -z "$GUIDE" ] || [ ! -f "$GUIDE" ]; then
    echo "ERROR: no guide checkpoint. Run run_autoguidance_guide_kcist.sh first, or"
    echo "       set GUIDE=/path/to/clogp_adapter_epN.ckpt"
    exit 1
fi

OUT="ag_pilot_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

echo "autoguidance pilot @ $(date) on $(hostname)"
echo "adapter=${ADAPTER}  guide=${GUIDE}  targets=${NT}  steps=${STEPS}  eta=${ETA}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# ---- preflight: the guide is wired, compatible, and actually degraded --------
# e2_targeting.py refuses on a bad guide, but it would do so eight times in parallel
# after loading a base each time. Fail once, here.
$PY - "$GUIDE" "$ADAPTER" <<'PY' || { echo "ERROR: guide preflight failed"; exit 1; }
import sys
sys.path.insert(0, "."); sys.path.insert(0, "/media/ssd2/Programming/defog-web")
import torch
from molsmith import sample as ms, store
from molsmith.weights import load as wl
from defog.core import AdaLNAdapter
from defog.core.adapter import _base_token

guide_path, adapter_ref = sys.argv[1], sys.argv[2]

if "guide" not in ms.SamplingConfig.__dataclass_fields__:
    print("FAIL: molsmith has no SamplingConfig.guide -- the run would be plain CFG "
          "labelled as autoguidance. Update molsmith.")
    sys.exit(1)

base_pkg = store.resolve_package("molsmith/zinc-kek")
base = wl.load_base(base_pkg, device="cpu")
main = wl.load_adapter(store.resolve_package(adapter_ref), base_pkg, base, device="cpu")
guide = AdaLNAdapter.load(guide_path, device="cpu")

tok = _base_token(base)
if guide.base_token is None or abs(tok - guide.base_token) > 1e-3 * (1 + abs(guide.base_token)):
    print(f"FAIL: guide base token {guide.base_token} != {tok}"); sys.exit(1)
for f in ("cond_mean", "cond_std"):
    gv, mv = getattr(guide, f), getattr(main, f)
    # shape first: torch.allclose RAISES on a shape mismatch rather than returning False,
    # which would surface as a traceback instead of the intended FAIL line.
    if gv.shape != mv.shape or not torch.allclose(gv, mv, atol=1e-4):
        print(f"FAIL: guide {f} {getattr(guide,f).tolist()} != adapter "
              f"{getattr(main,f).tolist()}"); sys.exit(1)

# The guide must be DEGRADED, not dead. A guide whose gates are ~0 IS the frozen base,
# which makes autoguidance silently identical to plain CFG -- the run would complete,
# the numbers would look fine, and the comparison would be against itself.
gn = sum(float(p.detach().abs().sum()) for lay in guide.gate for k in lay for p in lay[k].parameters())
mn = sum(float(p.detach().abs().sum()) for lay in main.gate  for k in lay for p in lay[k].parameters())
print(f"guide gate L1 = {gn:.4f}   adapter gate L1 = {mn:.4f}   ratio = {gn/mn if mn else float('nan'):.3f}")
if gn < 1e-6:
    print("FAIL: guide gates are all zero -- it IS the base, autoguidance == CFG"); sys.exit(1)
if gn > mn:
    print("WARNING: guide gates are LARGER than the shipped adapter's. It may not be "
          "the weaker model; check you picked an early epoch.")
print("guide preflight OK")
PY

# ---- arms: {CFG, autoguidance} x w in {2,3,4}, + a seed duplicate at w=3 ------
#        0        1        2        3        4        5        6           7
NAMES=( cfg_w2   cfg_w3   cfg_w4   ag_w2    ag_w3    ag_w4    cfg_w3_s43  ag_w3_s43 )
ARM=(   cfg      cfg      cfg      ag       ag       ag       cfg         ag )
W=(     2.0      3.0      4.0      2.0      3.0      4.0      3.0         3.0 )
SEED=(  42       42       42       42       42       42       43          43 )

declare -a PIDS
FAILED=0

reap () {   # reap <first> <last> -- wait on those arms, record any failure
    local j
    for j in $(seq "$1" "$2"); do
        if wait "${PIDS[$j]}"; then
            echo "  ok   ${NAMES[$j]}"
        else
            echo "  FAIL ${NAMES[$j]} (exit $?) -- see ag_${NAMES[$j]}_${SLURM_JOB_ID:-local}.out"
            FAILED=1
        fi
    done
}

for i in 0 1 2 3 4 5 6 7; do
    gpu=$(( i % 4 ))
    # Two full invocations rather than an optional-argument array: "${ARR[@]}" on an
    # EMPTY array is an unbound-variable error under `set -u` on bash < 4.4, and the
    # CFG arms are exactly the empty case. Matches the fk/adapter split in
    # run_e2_logp_sweep_kcist.sh.
    (
        if [ "${ARM[$i]}" = "ag" ]; then
            CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/e2_targeting.py \
                --adapter "$ADAPTER" --property logp --split validation \
                --method adapter --n-targets ${NT} --per-target 10 \
                --weight ${W[$i]} --steps ${STEPS} --eta ${ETA} \
                --blend-space prob --guide "$GUIDE" \
                --seed ${SEED[$i]} --out "${OUT}/${NAMES[$i]}.json"
        else
            CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/e2_targeting.py \
                --adapter "$ADAPTER" --property logp --split validation \
                --method adapter --n-targets ${NT} --per-target 10 \
                --weight ${W[$i]} --steps ${STEPS} --eta ${ETA} \
                --blend-space prob \
                --seed ${SEED[$i]} --out "${OUT}/${NAMES[$i]}.json"
        fi
    ) > "ag_${NAMES[$i]}_${SLURM_JOB_ID:-local}.out" 2>&1 &
    PIDS[$i]=$!
    echo "launched ${NAMES[$i]} (${ARM[$i]} w=${W[$i]} seed=${SEED[$i]}) on GPU ${gpu} (pid ${PIDS[$i]})"
    sleep 3
    # 8 arms over 4 GPUs: let the first wave finish before starting the second.
    # Arms exit via sys.exit("REFUSING: ...") on a bad guide, leaving NO json and no
    # other trace -- so exit codes are collected rather than a bare `wait`.
    if [ $i -eq 3 ]; then echo "reaping wave 1:"; reap 0 3; echo "wave 1 done at $(date)"; fi
done

echo "reaping wave 2:"
reap 4 7
echo "finished at $(date)"
[ $FAILED -eq 0 ] || echo "WARNING: at least one arm failed; the table below is INCOMPLETE"

echo
echo "=== autoguidance pilot: validity headroom as w rises ==="
$PY - "$OUT" "${NAMES[@]}" <<'PY'
import json, os, sys
import numpy as np

# EXPECT comes from the NAMES array itself rather than a second hardcoded list: an arm
# added to NAMES but forgotten here would be silently omitted from the table, which is
# the exact failure the MISSING ARMS banner exists to prevent.
OUT, EXPECT = sys.argv[1], sys.argv[2:]
rows, missing = {}, []
for n in EXPECT:
    f = os.path.join(OUT, n + ".json")
    if os.path.exists(f):
        rows[n] = json.load(open(f))
    else:
        missing.append(n)

# A silently short table reads exactly like a complete one. It must not.
if missing:
    print(f"!! MISSING ARMS ({len(missing)}/{len(EXPECT)}): {', '.join(missing)}")
    print("!! Their logs are ag_<arm>_<jobid>.out. A REFUSING: line there means the arm")
    print("!! rejected its own configuration and wrote nothing.\n")
if not rows:
    print("no result files at all"); sys.exit(1)

print(f"{'arm':13s}{'guide':>6s}{'w':>5s}{'seed':>6s}{'MAE':>9s}{'low':>8s}"
      f"{'mid':>8s}{'high':>8s}{'valid':>8s}{'uniq':>8s}")
for n in EXPECT:
    if n not in rows:
        continue
    d = rows[n]
    print(f"{n:13s}{('yes' if d.get('guide') else 'no'):>6s}"
          f"{d['sampling']['weight']:>5.1f}{d['seed']:>6d}{d['mae_pooled']:>9.4f}"
          f"{d['mae_low_third']:>8.4f}{d['mae_mid_third']:>8.4f}{d['mae_high_third']:>8.4f}"
          f"{d['validity']:>8.3f}{d['uniqueness']:>8.3f}")

# --- the PAIRED comparison, which is what the design actually supports -------
# --seed feeds draw_targets (e2_targeting.py), so cfg_w3 and cfg_w3_s43 run on
# DIFFERENT target sets: their spread is unpaired run-to-run variation dominated by
# the target draw. The cfg-vs-ag contrast at a FIXED seed is a different and much
# tighter quantity -- same targets, same node counts, same noise draws (adding a guide
# does not change RNG consumption: both arms run rep=2 groups). So pair per target.
def paired(a, b):
    ra, rb = a["per_target"], b["per_target"]
    if len(ra) != len(rb):
        return None
    if any(abs(x["target"] - y["target"]) > 1e-9 for x, y in zip(ra, rb)):
        return None                      # not the same targets: pairing is invalid
    dm = np.array([y["mae"] - x["mae"] for x, y in zip(ra, rb)], dtype=float)
    dv = np.array([y["validity"] - x["validity"] for x, y in zip(ra, rb)], dtype=float)
    # DO NOT mask dv by dm's finiteness. e2_targeting.py writes mae=NaN for a target
    # where an arm produced ZERO valid molecules -- and those are exactly the targets
    # this experiment is about. Measured on the real w-sweep: 0 such targets at w=1/2/3,
    # 7 of 100 at w=4, 78 of 100 at w=6. If autoguidance rescues a target where CFG
    # collapsed to 0/10, that target carries the LARGEST positive dv in the sample; the
    # old `dv[np.isfinite(dm)]` deleted precisely those, turning a clean win into a
    # clean null with a zero standard error. Validity is always finite -- 0/10 is 0.0,
    # not NaN -- so dv needs no mask at all.
    ok = np.isfinite(dm)
    return dm[ok], dv, int(ok.sum())

def _se(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")

print()
print("PAIRED per-target difference (autoguidance - CFG), same targets:")
print(f"{'w':>5s}{'dMAE':>10s}{'se':>8s}{'ag<cfg':>11s}{'n_mae':>7s}"
      f"{'dvalid':>10s}{'se':>8s}{'n_val':>7s}{'dead':>6s}")
for w in ("2", "3", "4"):
    c, a = rows.get(f"cfg_w{w}"), rows.get(f"ag_w{w}")
    if not (c and a):
        continue
    pr = paired(c, a)
    if pr is None:
        print(f"{w:>5s}  targets differ between arms -- cannot pair (check seed/--n-targets)")
        continue
    dm, dv, n_mae = pr
    dead = len(dv) - n_mae            # targets where SOME arm produced nothing at all
    print(f"{w:>5s}{dm.mean():>+10.4f}{_se(dm):>8.4f}"
          f"{str(int((dm < 0).sum())) + '/' + str(len(dm)):>11s}{n_mae:>7d}"
          f"{dv.mean():>+10.4f}{_se(dv):>8.4f}{len(dv):>7d}{dead:>6d}")
    if dead:
        print(f"       ^ {dead} target(s) had an arm produce 0 valid molecules. dMAE is "
              f"computed on the OTHER {n_mae} -- a strictly easier subset -- while "
              f"dvalidity covers all {len(dv)}. The two columns describe different "
              f"targets at this w; read dvalidity as the honest one.")

# Kept, but labelled as what it is: an UNPAIRED spread over a different target draw.
print()
for arm in ("cfg", "ag"):
    a, b = rows.get(f"{arm}_w3"), rows.get(f"{arm}_w3_s43")
    if a and b:
        print(f"{arm} w=3, seed 42 vs 43 (DIFFERENT targets -> unpaired, an upper bound "
              f"on noise): dMAE {abs(a['mae_pooled']-b['mae_pooled']):.4f}  "
              f"dvalidity {abs(a['validity']-b['validity']):.4f}")

print()
print("NOTE: the table's MAE is molecule-weighted (pooled over all generated molecules)")
print("while the paired dMAE is target-weighted. Where per-target valid counts vary a")
print("lot -- which is the whole point at high w -- the two can disagree in sign.")
print()
print("READING THIS. Judge the cfg-vs-ag contrast against the PAIRED standard errors")
print("above, not against the seed-42-vs-43 spread -- that spread is inflated by the")
print("target draw and is not the quantity being compared. The pre-registered signature")
print("is autoguidance holding validity at w=3/4 where CFG collapses, with the arms")
print("close at w=2. A uniform gain at every w INCLUDING w=2 is the bug signature, not")
print("a better result: check the 'guide wiring check' line in each ag arm's log.")
PY

exit $FAILED
