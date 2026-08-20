#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
# The shipped adapter's OWN per-target output, not just its headline number: same seed,
# split, target count, steps, eta and blend space, so the comparison is target-paired.
BASELINE_JSON="adapter_improvements/blend_results/e2_prob_w2.0.json"

# Set ADAPTER_CKPT, or the newest training result is used.
# The preflight below refuses a checkpoint missing either mechanism, so a plain FiLM
# adapter cannot slip through -- but a DIFFERENT xattn+fourier run (an ablation arm, or a
# re-run) would be accepted silently, and "newest" is not the same as "the one I meant".
# So the fallback refuses when it is ambiguous instead of guessing. Stage 1 prints the
# exact ADAPTER_CKPT= line to use.
ADAPTER_CKPT="${ADAPTER_CKPT:-}"
if [ -z "$ADAPTER_CKPT" ]; then
    mapfile -t _CANDS < <(ls -t experiments/results/adapter_training__zinc/*/clogp_adapter.ckpt 2>/dev/null)
    if [ "${#_CANDS[@]}" -eq 0 ]; then
        echo "ERROR: no adapter checkpoint found. Run run_xattn_fourier_train_kcist.sh"
        echo "       first, or set ADAPTER_CKPT=/path/to/clogp_adapter.ckpt"
        exit 1
    elif [ "${#_CANDS[@]}" -gt 1 ]; then
        echo "ERROR: ${#_CANDS[@]} candidate checkpoints; refusing to guess which one this"
        echo "       job is meant to evaluate. Set ADAPTER_CKPT= explicitly:"
        printf '         %s\n' "${_CANDS[@]}"
        exit 1
    fi
    ADAPTER_CKPT="${_CANDS[0]}"
    echo "note: ADAPTER_CKPT not set; using the only candidate found"
fi
if [ ! -f "$ADAPTER_CKPT" ]; then
    echo "ERROR: ADAPTER_CKPT=$ADAPTER_CKPT does not exist"
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
sys.path.insert(0, ".")   # molsmith is found the same way e2_targeting.py finds it
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

# A NON-ZERO OUTPUT PROJECTION IS NOT EVIDENCE OF ROUTING. At init the keys are
# unnormalised and the token producer's output is small, so the attention is almost
# uniform and every atom reads the same token average -- cross-attention starts life as
# a per-graph broadcast and only becomes content-addressed if training moves q/k. A
# checkpoint whose attention is still uniform IS a FiLM adapter with extra parameters,
# and the L1 check above would call it healthy. Entropy against ln(m) is what tells them
# apart. Reported, not enforced: a partially-sharpened router is still a real result, and
# the number belongs beside the MAE when it is interpreted.
import math, torch
m_tok = a.xattn_tokens
c = torch.tensor([[a.cond_mean.reshape(-1)[0].item()]])
cn = a.normalize(c)
parts = [cn] + ([a._fourier(cn)] if a.cond_fourier else [])
if a.time_conditioned:
    from defog.core.layers import timestep_embedding
    parts.append(timestep_embedding(torch.full((1, 1), 0.5), a.time_emb_dim))
h = a.trunk(torch.cat(parts, dim=-1))
tokens = a.tok(h).view(1, m_tok, a.xattn_dim)
torch.manual_seed(0)
X = torch.randn(1, 24, a.dims["dx"])
ents = []
for mod in a.xattn:
    nh, dh = mod.n_heads, mod.dx // mod.n_heads
    q = mod.q(mod.norm(X)).view(1, -1, nh, dh).transpose(1, 2)
    k = mod.k(tokens).view(1, -1, nh, dh).transpose(1, 2)
    att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
    ents.append(float(-(att * att.clamp_min(1e-12).log()).sum(-1).mean()))
uni = math.log(m_tok)
mean_ent, min_ent = sum(ents) / len(ents), min(ents)
print(f"  attention entropy: mean {mean_ent:.4f}, min {min_ent:.4f} "
      f"(uniform max {uni:.4f} = a pure broadcast)")
# Reported as a ratio, NOT as a pass/fail verdict: the queries here are Gaussian rather
# than the frozen base's real activations, and no trained adapter has yet been measured
# to calibrate a threshold against. 1.0000 means a pure per-graph broadcast (every atom
# reads the same token mixture, i.e. FiLM with extra parameters); lower means the router
# discriminates. Quote it beside the MAE.
print(f"  entropy ratio {mean_ent/uni:.4f} of uniform "
      f"({(1 - mean_ent/uni)*100:.2f}% below; 1.0000 would be a pure broadcast)")
if mean_ent > 0.999 * uni:
    print(f"  NOTE: essentially indistinguishable from uniform -- treat any result below "
          f"as 'FiLM with more parameters' until this number is understood.")
print("adapter preflight OK")
PY

# ---- arms: w sweep at the protocol's 100 targets, plus a seed duplicate -------
# The shipped 0.5420 is the MINIMUM over a 6-point grid (1, 2, 2.5, 3, 4, 6) on these same
# targets. Giving the new architecture 3 points and the baseline 6 would hand it a
# handicap that has nothing to do with the mechanism -- and the shipped adapter's own w=3
# sits at validity 0.898, below the floor used below, so a sparse grid can leave only two
# usable points. Eight arms is two clean waves of four; each arm is ~1000 s.
#        0        1        2        3        4        5        6           7
NAMES=( w1       w1.5     w2       w2.5     w3       w4       w2_s43      w3_s43 )
W=(     1.0      1.5      2.0      2.5      3.0      4.0      2.0         3.0 )
SEED=(  42       42       42       42       42       42       43          43 )

declare -a PIDS
FAILED=0

for i in 0 1 2 3 4 5 6 7; do
    gpu=$(( i % 4 ))
    (
        CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/e2_targeting.py \
            --adapter-ckpt "$ADAPTER_CKPT" --property logp --split validation \
            --method adapter --n-targets ${NT} --per-target 10 \
            --weight ${W[$i]} --steps ${STEPS} --eta ${ETA} \
            --blend-space prob \
            --seed ${SEED[$i]} --out "${OUT}/${NAMES[$i]}.json"
    ) > "xafo_${NAMES[$i]}_${SLURM_JOB_ID:-local}.out" 2>&1 &
    PIDS[$i]=$!
    echo "launched ${NAMES[$i]} (w=${W[$i]} seed=${SEED[$i]}) on GPU ${gpu} (pid ${PIDS[$i]})"
    sleep 3
    if [ $i -eq 3 ]; then
        echo "reaping wave 1:"
        for j in 0 1 2 3; do
            if wait "${PIDS[$j]}"; then echo "  ok   ${NAMES[$j]}"
            else echo "  FAIL ${NAMES[$j]} (exit $?)"; FAILED=1; fi
        done
        echo "wave 1 done at $(date)"
    fi
done

echo "reaping wave 2:"
for j in 4 5 6 7; do
    if wait "${PIDS[$j]}"; then echo "  ok   ${NAMES[$j]}"
    else echo "  FAIL ${NAMES[$j]} (exit $?) -- see xafo_${NAMES[$j]}_${SLURM_JOB_ID:-local}.out"; FAILED=1; fi
done
echo "finished at $(date)"
[ $FAILED -eq 0 ] || echo "WARNING: at least one arm failed; the table below is INCOMPLETE"

echo
echo "=== E2 logP, validation, xattn+fourier adapter ==="
$PY - "$OUT" "$BASELINE_JSON" "$ADAPTER_CKPT" "${NAMES[@]}" <<'PY'
import json, math, os, sys

OUT, BASELINE, CKPT = sys.argv[1], sys.argv[2], sys.argv[3]
EXPECT = sys.argv[4:]                 # from NAMES, not a second hardcoded copy

rows, missing = {}, []
for n in EXPECT:
    f = os.path.join(OUT, n + ".json")
    (rows.__setitem__(n, json.load(open(f))) if os.path.exists(f) else missing.append(n))
if missing:
    print(f"!! MISSING ARMS ({len(missing)}/{len(EXPECT)}): {', '.join(missing)}")
    print("!! A REFUSING: line in xafo_<arm>_*.out means the arm rejected its config.\n")
if not rows:
    print("no result files at all"); sys.exit(1)

def dead_of(d):
    # mae is NaN exactly when an arm produced zero valid molecules for that target, so
    # those targets vanish from every MAE number while still counting in validity.
    return sum(1 for r in d["per_target"] if not math.isfinite(r["mae"]))

print(f"{'arm':9s}{'w':>5s}{'seed':>6s}{'MAE':>9s}{'low':>8s}{'mid':>8s}{'high':>8s}"
      f"{'valid':>8s}{'uniq':>8s}{'dead':>6s}")
for n in EXPECT:
    if n not in rows: continue
    d = rows[n]
    print(f"{n:9s}{d['sampling']['weight']:>5.1f}{d['seed']:>6d}{d['mae_pooled']:>9.4f}"
          f"{d['mae_low_third']:>8.4f}{d['mae_mid_third']:>8.4f}{d['mae_high_third']:>8.4f}"
          f"{d['validity']:>8.3f}{d['uniqueness']:>8.3f}{dead_of(d):>6d}")
    if dead_of(d):
        print(f"{'':9s}^ {dead_of(d)}/{len(d['per_target'])} targets produced 0 valid "
              f"molecules; this arm's MAE describes only the other "
              f"{len(d['per_target'])-dead_of(d)}.")

# ---- PAIRED against the shipped adapter, which is the whole point ------------
# The sampling pipeline is DETERMINISTIC (PLAN.md Wave 1: an identical-config rerun
# reproduced to 1e-9), and the shipped 0.5420 was measured at seed 42, validation,
# 100x10, 500 steps, eta 25, prob blend, marginal size -- the same configuration as the
# seed-42 arms here. So those arms draw the SAME 100 targets as the baseline and the
# comparison is target-paired: the right statistic is a paired difference, not a
# difference of two pooled means judged against a seed spread.
#
# The seed-43 arms are NOT the yardstick for that comparison. Their spread is
# target-draw variation, which CANCELS in a paired test; using it as the denominator
# inflates it with variance that is not there and can stamp "do not over-read" on a real
# effect. They are kept for what they do measure: how much a 100-target estimate wobbles
# when the targets change.
base = None
if os.path.exists(BASELINE):
    base = json.load(open(BASELINE))
    print(f"\nshipped baseline: {base.get('adapter')} seed {base['seed']} "
          f"{base['split']} n={len(base['per_target'])} w={base['sampling']['weight']} "
          f"MAE {base['mae_pooled']:.4f} validity {base['validity']:.3f} "
          f"dead {dead_of(base)}")
else:
    print(f"\n(no shipped baseline JSON at {BASELINE}; paired comparison skipped)")

def paired_vs_base(d):
    ra, rb = base["per_target"], d["per_target"]
    if len(ra) != len(rb) or any(abs(x["target"] - y["target"]) > 1e-9
                                 for x, y in zip(ra, rb)):
        return None
    dm = [y["mae"] - x["mae"] for x, y in zip(ra, rb)
          if math.isfinite(x["mae"]) and math.isfinite(y["mae"])]
    dv = [y["validity"] - x["validity"] for x, y in zip(ra, rb)]   # never NaN
    return dm, dv

def mean_se(v):
    if len(v) < 2: return (float("nan"), float("nan"))
    m = sum(v) / len(v)
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var / len(v))

if base is not None:
    print("\nPAIRED vs the shipped adapter (same 100 targets; seed-42 arms only):")
    print(f"{'arm':9s}{'dMAE':>10s}{'se':>8s}{'better':>10s}{'n_mae':>7s}"
          f"{'dvalid':>10s}{'se':>8s}")
    for n in EXPECT:
        d = rows.get(n)
        if d is None or d["seed"] != base["seed"]:
            continue
        pr = paired_vs_base(d)
        if pr is None:
            print(f"{n:9s}  targets differ from the baseline -- cannot pair")
            continue
        dm, dv = pr
        mm, sm = mean_se(dm)
        mv, sv = mean_se(dv)
        print(f"{n:9s}{mm:>+10.4f}{sm:>8.4f}"
              f"{str(sum(1 for x in dm if x < 0)) + '/' + str(len(dm)):>10s}{len(dm):>7d}"
              f"{mv:>+10.4f}{sv:>8.4f}")
    print("negative dMAE = the new adapter is closer to target than the shipped one.")

# ---- best arm, restricted to the paired (seed-42) arms ----------------------
print()
for floor in (0.90, 0.97):
    cand = [(n, d) for n, d in rows.items()
            if d["seed"] == 42 and d["validity"] >= floor]
    if not cand:
        print(f"validity >= {floor:.2f}: no seed-42 arm clears it")
        continue
    n, d = min(cand, key=lambda kv: kv[1]["mae_pooled"])
    print(f"validity >= {floor:.2f}: best is {n} (w={d['sampling']['weight']}) "
          f"MAE {d['mae_pooled']:.4f}, validity {d['validity']:.3f}")
# The two floors are printed because 0.90 admits an arm at 0.91 against a baseline at
# 0.982 -- MAE bought with 7 points of validity is not a win on FreeGress's terms, and
# 0.97 is the floor at which the comparison is like-for-like.

a43 = [(n, rows[n]) for n in EXPECT if n in rows and rows[n]["seed"] == 43]
for n, d in a43:
    twin = next((rows[m] for m in EXPECT if m in rows and rows[m]["seed"] == 42
                 and rows[m]["sampling"]["weight"] == d["sampling"]["weight"]), None)
    if twin:
        print(f"\ntarget-draw wobble at w={d['sampling']['weight']}: "
              f"|dMAE| {abs(twin['mae_pooled']-d['mae_pooled']):.4f}, "
              f"|dvalidity| {abs(twin['validity']-d['validity']):.4f} "
              f"(a DIFFERENT 100 targets -- how much the estimate itself moves, not the "
              f"yardstick for the paired rows above)")

print("\nWHAT THIS CAN AND CANNOT SAY. One arm carries BOTH mechanisms, so a win does")
print("not attribute to Fourier bands or to cross-attention, and a null does not rule out")
print("one helping while the other hurts. The attribution ablation is the follow-up and")
print("this script supports it: rerun stage 1 with COND_FOURIER/XATTN_TOKENS set one at a")
print("time. Also read the entropy ratio from the preflight above -- if the router stayed")
print("near-uniform, this arm is closer to 'FiLM with more parameters' than to")
print("node-resolved conditioning, whatever the MAE says.")
print(f"\nadapter: {CKPT}")
PY

exit $FAILED
