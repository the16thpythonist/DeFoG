#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=xafo_eval
#SBATCH --output=xafo_eval_%j.out

# Stage 2: the full E2 protocol on the xattn+fourier adapter.
#
# PROTOCOL (docs/targeting-protocol.md, matching FreeGress Tab. 2): 100 targets drawn
# from the split, 10 molecules each, MAE over all 1000, validity reported beside it.
#
# VALIDATION SPLIT ONLY. This is exploratory architecture work, so it informs a choice
# and may not touch test. The one-shot test pass stays intact for whatever ends up being
# the frozen configuration.
#
# WHY A w SWEEP AND NOT JUST w=2. w=2.0 is the measured optimum for the SHIPPED FiLM
# adapter in probability space (0.6410 -> 0.5420 -> 0.5818 -> 0.5943 at w=1/2/2.5/3).
# There is no reason a different conditioning architecture has the same optimum, and
# reporting a new architecture at someone else's operating point is how a real gain gets
# missed. Validation is exactly where choosing w is allowed.
#
# THE SEED DUPLICATE IS NOT OPTIONAL. Every number here gets read against the shipped
# 0.5420, and the harness's run-to-run spread is ~0.008 MAE. The seed-43 arm is what
# says whether a difference is a result. Note it draws a DIFFERENT 100 targets, so it
# measures unpaired spread -- the right yardstick for comparing against a number
# measured on another run, which is what the 0.5420 comparison is.
#
# BLEND SPACE IS PINNED. --blend-space prob explicitly rather than by default: in rate
# space w>1 collapses (MAE 5.59, validity 0.526 at w=2), which would make the sweep
# measure the clamp rather than the architecture.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
NT=100
STEPS=500
ETA=25
BASELINE_MAE="0.5420"          # shipped clogp@1.2.0, prob blend, w=2, 100 val targets
BASELINE_VAL="0.982"

# Set ADAPTER_CKPT, or the newest training result is used.
ADAPTER_CKPT="${ADAPTER_CKPT:-}"
if [ -z "$ADAPTER_CKPT" ]; then
    ADAPTER_CKPT=$(ls -t experiments/results/adapter_training__zinc/*/clogp_adapter.ckpt 2>/dev/null | head -1)
fi
if [ -z "$ADAPTER_CKPT" ] || [ ! -f "$ADAPTER_CKPT" ]; then
    echo "ERROR: no adapter checkpoint. Run run_xattn_fourier_train_kcist.sh first, or"
    echo "       set ADAPTER_CKPT=/path/to/clogp_adapter.ckpt"
    exit 1
fi

OUT="xafo_eval_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

echo "xattn+fourier E2 eval @ $(date) on $(hostname)"
echo "adapter=${ADAPTER_CKPT}  targets=${NT}  steps=${STEPS}  eta=${ETA}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# ---- preflight: the checkpoint really is the architecture we are reporting ----
$PY - "$ADAPTER_CKPT" <<'PY' || { echo "ERROR: adapter preflight failed"; exit 1; }
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "/media/ssd2/Programming/defog-web")
from defog.core import AdaLNAdapter
a = AdaLNAdapter.load(sys.argv[1], device="cpu")
cfg = a._config()
print(f"adapter: {sum(p.numel() for p in a.parameters()):,} params  "
      f"fourier={cfg.get('cond_fourier')} xattn_tokens={cfg.get('xattn_tokens')} "
      f"hidden={cfg['hidden']} n_layers={cfg['n_layers']}")
print(f"  cond_mean={a.cond_mean.tolist()} cond_std={a.cond_std.tolist()}")
if not cfg.get("cond_fourier") or not cfg.get("xattn_tokens"):
    print("FAIL: this checkpoint does not carry both mechanisms -- it is not the arm "
          "this job claims to evaluate."); sys.exit(1)
# An untrained cross-attention path is zero-init by construction, so a zero here means
# the mechanism never learned anything and the run would report a FiLM adapter under a
# cross-attention label.
xa = sum(float(m.out.weight.abs().sum() + m.out.bias.abs().sum()) for m in a.xattn)
print(f"  xattn output-projection L1 = {xa:.4e}")
if xa == 0.0:
    print("FAIL: cross-attention output projections are still exactly zero -- the "
          "mechanism is inert."); sys.exit(1)
print("adapter preflight OK")
PY

# ---- arms: w sweep at the protocol's 100 targets, plus a seed duplicate -------
#        0        1        2        3
NAMES=( w1       w2       w3       w2_s43 )
W=(     1.0      2.0      3.0      2.0 )
SEED=(  42       42       42       43 )

declare -a PIDS
FAILED=0

for i in 0 1 2 3; do
    (
        CUDA_VISIBLE_DEVICES=$i $PY -u scripts/e2_targeting.py \
            --adapter-ckpt "$ADAPTER_CKPT" --property logp --split validation \
            --method adapter --n-targets ${NT} --per-target 10 \
            --weight ${W[$i]} --steps ${STEPS} --eta ${ETA} \
            --blend-space prob \
            --seed ${SEED[$i]} --out "${OUT}/${NAMES[$i]}.json"
    ) > "xafo_${NAMES[$i]}_${SLURM_JOB_ID:-local}.out" 2>&1 &
    PIDS[$i]=$!
    echo "launched ${NAMES[$i]} (w=${W[$i]} seed=${SEED[$i]}) on GPU ${i} (pid ${PIDS[$i]})"
    sleep 3
done

for i in 0 1 2 3; do
    if wait "${PIDS[$i]}"; then
        echo "  ok   ${NAMES[$i]}"
    else
        echo "  FAIL ${NAMES[$i]} (exit $?) -- see xafo_${NAMES[$i]}_${SLURM_JOB_ID:-local}.out"
        FAILED=1
    fi
done
echo "finished at $(date)"
[ $FAILED -eq 0 ] || echo "WARNING: at least one arm failed; the table below is INCOMPLETE"

echo
echo "=== E2 logP, validation, xattn+fourier adapter ==="
$PY - "$OUT" "$BASELINE_MAE" "$BASELINE_VAL" "$ADAPTER_CKPT" <<'PY'
import json, os, sys
OUT, base_mae, base_val, ckpt = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
EXPECT = ["w1", "w2", "w3", "w2_s43"]
rows, missing = {}, []
for n in EXPECT:
    f = os.path.join(OUT, n + ".json")
    (rows.__setitem__(n, json.load(open(f))) if os.path.exists(f) else missing.append(n))
if missing:
    print(f"!! MISSING ARMS ({len(missing)}/{len(EXPECT)}): {', '.join(missing)}")
    print(f"!! A REFUSING: line in xafo_<arm>_*.out means the arm rejected its config.\n")
if not rows:
    print("no result files at all"); sys.exit(1)

print(f"{'arm':9s}{'w':>5s}{'seed':>6s}{'MAE':>9s}{'low':>8s}{'mid':>8s}{'high':>8s}"
      f"{'valid':>8s}{'uniq':>8s}")
for n in EXPECT:
    if n not in rows: continue
    d = rows[n]
    print(f"{n:9s}{d['sampling']['weight']:>5.1f}{d['seed']:>6d}{d['mae_pooled']:>9.4f}"
          f"{d['mae_low_third']:>8.4f}{d['mae_mid_third']:>8.4f}{d['mae_high_third']:>8.4f}"
          f"{d['validity']:>8.3f}{d['uniqueness']:>8.3f}")

a, b = rows.get("w2"), rows.get("w2_s43")
spread = abs(a["mae_pooled"] - b["mae_pooled"]) if (a and b) else None
if spread is not None:
    print(f"\nseed spread at w=2 (different 100 targets): dMAE {spread:.4f}  "
          f"dvalidity {abs(a['validity']-b['validity']):.4f}")

print(f"\nSHIPPED FiLM BASELINE (clogp@1.2.0, prob blend, w=2, 100 val targets): "
      f"MAE {base_mae:.4f}, validity {base_val}")
best = min((d for d in rows.values() if d["validity"] >= 0.90),
           key=lambda d: d["mae_pooled"], default=None)
if best is None:
    print("No arm cleared validity >= 0.90; MAE is not comparable below that.")
else:
    delta = best["mae_pooled"] - base_mae
    print(f"best arm clearing validity>=0.90: w={best['sampling']['weight']} "
          f"MAE {best['mae_pooled']:.4f} ({delta:+.4f} vs shipped) "
          f"validity {best['validity']:.3f}")
    if spread is not None:
        n_sig = abs(delta) / max(spread, 1e-9)
        print(f"that difference is {n_sig:.1f}x the measured seed spread "
              f"({'READ IT' if n_sig >= 2 else 'DO NOT over-read it'})")

print(f"\nWHAT THIS CAN AND CANNOT SAY. One arm carries BOTH mechanisms, so a win does")
print(f"not attribute to Fourier bands or to cross-attention, and a null does not rule")
print(f"out one helping while the other hurts. The attribution ablation is the")
print(f"follow-up, and it is cheap: the same job with COND_FOURIER/XATTN_TOKENS set")
print(f"one at a time.")
print(f"\nadapter: {ckpt}")
PY
