#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=e2_qed_r2
#SBATCH --output=e2_qed_r2_%j.out

# Does the RL gain survive into the DEPLOYMENT configuration?
#
# The RL job (43073) measured itself on 3 percentile targets at eta=5 with a bare
# adapter, and reported mean MAE 0.1466 -> 0.1082 (-26%). That is its own internal
# protocol, not the one the paper reports. This job asks the question that matters:
# under the FreeGress protocol (100 real-molecule targets x 10) with the frozen best
# deployment stack -- FK beta=1000 plus the conditional size draw -- how much of that
# -26% is left?
#
# THE GAIN MIGHT NOT SURVIVE, AND THAT IS THE POINT OF ASKING. FK and the size draw
# already fix some of what RL fixes: all three attack the same tails. RL improved the
# LOW and HIGH levels most (-0.049 / -0.042); the size draw's gain was also in both
# tails (-16.6% / -23.2%); FK's was in the low third (-23%). Three corrections aimed
# at the same place do not add up, so the honest expectation is that the stacked gain
# is smaller than the sum, and it could be near zero.
#
# ARMS (4 GPUs)
#   rl_fk        RL-kl010 + size + FK1000   <- the headline: full stack, best adapter
#   rl_nofk      RL-kl010 + size, no FK     <- isolates RL+size from FK
#   rl005_fk     RL-kl005 + size + FK1000   <- second RL arm; the seed duplicate showed
#                                              the RL magnitude is seed-sensitive, so a
#                                              single RL adapter is not evidence
#   pre_fk       pre-RL     + size + FK1000 <- REPRODUCTION CONTROL, must return 0.0865
#
# The control is not optional. The molsmith on this cluster was patched by hand
# yesterday to honour SamplingConfig.size_dist, and two adapters were installed into
# the store since job 43071 ran. If pre_fk does not come back at 0.0865 then something
# in the environment moved and none of the deltas below can be read.
#
# EVERYTHING ELSE IS FROZEN AT THE MEASURED BEST: w=1.0, 500 steps, eta=25, omega=0,
# polydec, seed 42, FK beta=1000 / warmup 0.6 / ess 0.25 / rejuvenate off, K=10 with
# all ten kept, learned P(n|QED) from ckpts/heads/qed_head_size.ckpt.
#
# VALIDATION SPLIT. The test pass is still unspent and the configuration is still
# moving, so nothing here touches test.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=.venv/bin/python
SIZE_CKPT="ckpts/heads/qed_head_size.ckpt"
NT=100
OUT="e2_qedr2_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "E2 QED: does the RL gain survive the deployment stack? @ $(date) on $(hostname)"
echo "  pre-RL  molsmith/qed@2.0.0   RL-kl010 molsmith/qed@3.0.0   RL-kl005 molsmith/qed@3.1.0"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# All three adapters must bind zinc-kek and carry the head, and they must be DISTINCT
# -- packaging three versions from the same file is a silent way to compare an adapter
# against itself.
$PY -c "
import sys, hashlib; sys.path.insert(0,'.')
from pathlib import Path
from molsmith import store
seen = {}
for ref in ('molsmith/qed@2.0.0', 'molsmith/qed@3.0.0', 'molsmith/qed@4.0.0', 'molsmith/qed@4.1.0'):
    p = store.resolve_package(ref); m = p.metadata
    assert m.head.present, ref + ' bundles no head -- FK needs it'
    assert m.base.id == 'molsmith/zinc-kek', ref + ' does not bind zinc-kek'
    # Hash the WEIGHTS on disk, not any store bookkeeping: three versions migrated
    # from the same ckpt would carry three distinct store entries and identical
    # tensors, which is exactly the mistake this is meant to catch.
    w = Path(p.primary_weights())
    h = hashlib.sha256(w.read_bytes()).hexdigest()
    print(f'  {ref:22s} {w.name} sha256 {h[:24]}')
    seen.setdefault(h, []).append(ref)
dupes = {h: r for h, r in seen.items() if len(r) > 1}
assert not dupes, f'IDENTICAL weights behind different versions: {dupes}'
print('  four distinct adapters, all on zinc-kek with heads')
" || exit 1

[ -f "$SIZE_CKPT" ] || { echo "ERROR: ${SIZE_CKPT} missing"; exit 1; }

NAMES=( r2_best            r2_pair            r1_k010            pre_fk )
ADAPT=( molsmith/qed@4.0.0 molsmith/qed@4.1.0 molsmith/qed@3.0.0 molsmith/qed@2.0.0 )
METH=(  fk                 fk                 fk                 fk )

for i in 0 1 2 3; do
    if [ "${METH[$i]}" = "fk" ]; then
        CUDA_VISIBLE_DEVICES=$i $PY -u scripts/e2_targeting.py \
            --adapter "${ADAPT[$i]}" --property qed --split validation \
            --method fk --n-targets ${NT} --per-target 10 \
            --weight 1.0 --steps 500 --eta 25 \
            --fk-beta 1000 --fk-warmup 0.6 --fk-ess 0.25 \
            --size-mode learned --size-model "$SIZE_CKPT" \
            --seed 42 --out "${OUT}/${NAMES[$i]}.json" \
            > "e2qr2_${NAMES[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$i $PY -u scripts/e2_targeting.py \
            --adapter "${ADAPT[$i]}" --property qed --split validation \
            --method adapter --n-targets ${NT} --per-target 10 \
            --weight 1.0 --steps 500 --eta 25 \
            --size-mode learned --size-model "$SIZE_CKPT" \
            --seed 42 --out "${OUT}/${NAMES[$i]}.json" \
            > "e2qr2_${NAMES[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    fi
    echo "launched ${NAMES[$i]} (${ADAPT[$i]}, ${METH[$i]}) on GPU ${i} (pid $!)"
    sleep 3
done

wait
echo "finished at $(date)"

OK=0
for n in "${NAMES[@]}"; do [ -f "${OUT}/${n}.json" ] && OK=$((OK+1)); done
echo "arms complete: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: incomplete; tracebacks follow"
    grep -hA6 "Traceback" e2qr2_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== QED: RL vs pre-RL under the deployment stack (validation, ${NT} targets) ==="
$PY - "$OUT" <<'PY'
import json, os, sys
import numpy as np

d = sys.argv[1]
UNC = 0.743                 # unconditional QED mean at eta=25 (measured, job 43036)
PRIOR = {"pre_fk": 0.0865, "r1_k010": 0.0766}  # jobs 43071 / 43082, identical config
rows = [("pre_fk",  "pre-RL      + size + FK1000"),
        ("r1_k010", "RL rnd1     + size + FK1000"),
        ("r2_pair", "RL rnd2 s42 + size + FK1000"),
        ("r2_best", "RL rnd2 s7  + size + FK1000")]
print("%-26s%9s%8s%9s%9s%9s%8s%8s" %
      ("arm", "MAE", "skill", "low", "mid", "high", "valid", "uniq"))
got = {}
for key, label in rows:
    f = os.path.join(d, f"{key}.json")
    if not os.path.exists(f):
        print("%-26s  MISSING" % label); continue
    r = json.load(open(f)); got[key] = r
    tg = np.array([x["target"] for x in r["per_target"]])
    sk = (1 - r["mae_pooled"] / float(np.abs(tg - UNC).mean())) * 100
    print("%-26s%9.4f%7.0f%%%9.4f%9.4f%9.4f%8.3f%8.3f" %
          (label, r["mae_pooled"], sk, r["mae_low_third"], r["mae_mid_third"],
           r["mae_high_third"], r["validity"], r["uniqueness"]))

print()
print("REPRODUCTION CONTROLS (two, because two earlier jobs are being read across)")
for key, job, label in (("pre_fk", "43071", "pre-RL  + size + FK1000"),
                        ("r1_k010", "43082", "RL rnd1 + size + FK1000")):
    if key not in got:
        continue
    v = got[key]["mae_pooled"]; delta = abs(v - PRIOR[key])
    print(f"  {label} here {v:.4f} vs job {job} {PRIOR[key]:.4f}  |diff| {delta:.4f}  "
          + ("OK" if delta < 0.003 else
             "*** DRIFT -- the deltas below are NOT readable ***"))

print()
print("WHAT THE RL IS WORTH ON TOP OF THE STACK")
if "pre_fk" in got:
    base = got["pre_fk"]["mae_pooled"]
    for key, label in rows:
        if key == "pre_fk" or key not in got:
            continue
        v = got[key]["mae_pooled"]
        print(f"  {label} {v:.4f}   {(v-base)/base*100:+.1f}% vs pre-RL")

print()
print("reference points")
print("  RL's OWN protocol (3 percentile targets, eta=5, bare adapter): -26%")
print("  do-nothing 0.1083 | adapter alone 0.1212 | +size 0.1033 | +size+FK 0.0865")
print("  FreeGress unconditional 0.15, best 0.04; DiGress 0.14-0.15")
print()
print("HOW TO READ IT")
print("  Expect LESS than -26%. RL, FK and the size draw all correct the same tails,")
print("  so they cannot each be worth their solo number when stacked. A small gain")
print("  here is the honest result, not a disappointment -- and rl_nofk says how much")
print("  of it FK was already providing.")
print("  Check uniqueness against pre_fk (0.828). RL tightening the conditional could")
print("  narrow the output distribution, and MAE bought that way is not a real gain.")
PY
