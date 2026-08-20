#!/bin/bash
# `small`, not `batch`. The default `batch` partition spans both node classes and
# the scheduler will happily place a 4-GPU job on a 128-CPU/980GB/8-GPU large node,
# which is what it did on the first submission of this job. A small node is
# 64 CPU / 489 GB / 4 GPU -- this job's whole request fits one exactly.
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --job-name=e2_eta
#SBATCH --output=e2_eta_sweep_%j.out

# Is eta=25 the wrong operating point for CONDITIONAL sampling?
#
# WHY THIS EXISTS. Every adapter number this project has ever validated was
# measured at eta=5 -- adapter_training__zinc.py, adapter_rl_finetune__zinc.py and
# adapter_selfconsistency__zinc.py all set ETA: float = 5.0, and the four shipped
# ZINC bundles were scored there. The E2 harness inherited eta=25 from the
# UNCONDITIONAL E1 sweep, where it was tuned for base validity and sanity with no
# adapter in the loop. Nobody chose 25 for conditional sampling; it was carried
# across from a different question.
#
# THE MECHANISM THAT MAKES THIS PLAUSIBLE. eta scales R^DB, the detailed-balance
# term. R^DB is constructed to leave the marginal invariant -- it is error
# correction toward the UNCONDITIONAL distribution. That is exactly antagonistic
# to an adapter whose whole job is to hold the chain away from the marginal. It
# should also hurt narrow-band properties much more than wide ones: QED's entire
# usable spread is std 0.139, logP's is 1.46, so the same diffusive churn eats a
# tenfold larger fraction of the QED signal.
#
# WHAT THIS IS NOT. It does NOT explain the QED training number (MAE 0.1400) --
# that eval already ran at eta=5. Checked: run_zinc_qed_adapter_kcist.sh passes no
# ETA override, so the experiment default applied. This sweep is about the E2
# rows, which are the numbers that go in the paper, and where QED has never been
# run through the harness at all.
#
# READ SKILL, NOT MAE. On 100 targets drawn from real molecules, roughly a third
# land near the unconditional mean, where a constant predictor is unbeatable and
# any generative sampler loses on its own spread alone. Raw MAE therefore mixes
# "how well does it steer" with "how many targets were worth steering to". The
# summary below reports skill = 1 - MAE/MAE_donothing per third, against the
# unconditional mean MEASURED AT THE SAME ETA -- which is why the baseline arms
# exist rather than a hard-coded constant.
#
# THE BASELINE ARMS DO DOUBLE DUTY. They also answer whether the sanity-RL base
# narrowed the reachable QED range: the RL cut disconnected fragments -33% and
# wonky rings -23%, which are precisely what QED's alert and structural terms
# punish. If the unconditional QED std comes back well under the dataset's 0.133,
# the low targets are unreachable no matter how good the adapter is, and that is a
# base problem the adapter cannot be blamed for.
#
# WATCH VALIDITY. eta is error correction; removing it is not free. If validity
# collapses at eta=1 then the MAE gain is bought with molecules we cannot use, and
# the row is not reportable. Validity is printed beside every arm.
#
# VALIDATION SPLIT THROUGHOUT. Nothing here touches test.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY=.venv/bin/python
NT=100
OUT="e2_eta_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "E2 eta sweep @ $(date) on $(hostname)"
echo "  adapters: molsmith/qed@2.0.0 (NEW) and molsmith/clogp@1.2.0"
echo "  varying eta in {1, 5, 25}; everything else frozen:"
echo "    w=1.0, 500 steps, omega=0, polydec, seed 42, ${NT} validation targets, 10/target"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Both adapters must bind the SAME base, or the eta comparison silently spans two
# models. The schema_hash gate would catch a cross-serve at load time, but only
# after an hour of sampling.
$PY -c "
import sys; sys.path.insert(0,'.')
from molsmith import store
for ref in ('molsmith/qed@2.0.0', 'molsmith/clogp@1.2.0'):
    m = store.resolve_package(ref).metadata
    print(f'  {ref:24s} base={m.base.id} head={m.head.present}')
    assert m.base.id == 'molsmith/zinc-kek', f'{ref} does not bind zinc-kek'
" || exit 1

# ---- build the arm list ------------------------------------------------------
# 6 targeting arms (2 properties x 3 etas) + 3 unconditional baselines (1 per eta).
CMDS=(); NAMES=()
for ETA in 1 5 25; do
    CMDS+=("$PY -u scripts/e2_targeting.py --adapter molsmith/qed@2.0.0 --property qed \
--split validation --method adapter --n-targets ${NT} --per-target 10 \
--weight 1.0 --steps 500 --eta ${ETA} --seed 42 --out ${OUT}/qed_eta${ETA}.json")
    NAMES+=("qed_eta${ETA}")
    CMDS+=("$PY -u scripts/e2_targeting.py --adapter molsmith/clogp@1.2.0 --property logp \
--split validation --method adapter --n-targets ${NT} --per-target 10 \
--weight 1.0 --steps 500 --eta ${ETA} --seed 42 --out ${OUT}/logp_eta${ETA}.json")
    NAMES+=("logp_eta${ETA}")
    CMDS+=("$PY -u scripts/e2_uncond_baseline.py --base molsmith/zinc-kek --n 250 \
--steps 500 --eta ${ETA} --seed 42 --out ${OUT}/base_eta${ETA}.json")
    NAMES+=("base_eta${ETA}")
done

# ---- run at most 4 at a time (one per GPU) -----------------------------------
MAXPAR=4
declare -A GPU_OF
running=0; idx=0
for i in "${!CMDS[@]}"; do
    while [ "$running" -ge "$MAXPAR" ]; do wait -n; running=$((running-1)); done
    gpu=$(( idx % MAXPAR )); idx=$((idx+1))
    CUDA_VISIBLE_DEVICES=$gpu bash -c "${CMDS[$i]}" > "e2eta_${NAMES[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${NAMES[$i]} on GPU ${gpu} (pid $!)"
    running=$((running+1))
    sleep 3
done
wait
echo "finished at $(date)"

OK=0
for n in "${NAMES[@]}"; do [ -f "${OUT}/${n}.json" ] && OK=$((OK+1)); done
echo "arms complete: ${OK} / ${#NAMES[@]}"
if [ "$OK" -lt "${#NAMES[@]}" ]; then
    echo "ERROR: incomplete; tracebacks follow"
    grep -hA6 "Traceback" e2eta_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -30
fi

echo
echo "=== eta sweep (validation, ${NT} targets, adapter-only, w=1.0) ==="
$PY - "$OUT" <<'PY'
import json, os, sys
import numpy as np

d = sys.argv[1]
def load(p):
    f = os.path.join(d, p)
    return json.load(open(f)) if os.path.exists(f) else None

for prop, key in (("qed", "qed"), ("logp", "logp")):
    print(f"\n--- {prop.upper()} ---")
    print(f"{'eta':>5s}{'uncond mean':>13s}{'uncond sd':>11s}{'MAE':>9s}"
          f"{'do-nothing':>12s}{'skill':>8s}{'  low/mid/high skill':>22s}{'valid':>8s}")
    for eta in (1, 5, 25):
        r = load(f"{prop}_eta{eta}.json"); b = load(f"base_eta{eta}.json")
        if r is None or b is None:
            print(f"{eta:>5d}   MISSING"); continue
        mu, sd = b[key]["mean"], b[key]["sd"]
        # the do-nothing predictor: emit the unconditional mean at THIS eta
        tg = np.array([row["target"] for row in r["per_target"]])
        order = np.argsort(tg); thirds = np.array_split(order, 3)
        dn_all = float(np.abs(tg - mu).mean())
        skills = []
        for part, got in zip(thirds, (r["mae_low_third"], r["mae_mid_third"],
                                      r["mae_high_third"])):
            dn = float(np.abs(tg[part] - mu).mean())
            skills.append((1 - got / dn) * 100 if dn > 0 else float("nan"))
        sk = (1 - r["mae_pooled"] / dn_all) * 100
        print(f"{eta:>5d}{mu:>13.3f}{sd:>11.3f}{r['mae_pooled']:>9.4f}"
              f"{dn_all:>12.4f}{sk:>7.0f}%"
              f"{skills[0]:>8.0f}%{skills[1]:>7.0f}%{skills[2]:>7.0f}%"
              f"{r['validity']:>8.3f}")

print("\nreference points")
print("  QED  dataset std 0.139 (source) / 0.133 (decoded).  If uncond sd is far")
print("       below that, the sanity-RL base cannot reach the low targets and the")
print("       adapter is being blamed for a base limitation.")
print("  logP dataset std 1.46 (source) / 1.18 (decoded).")
print("  skill = 1 - MAE/MAE_donothing. NEGATIVE means worse than ignoring the")
print("  condition. The MID third is expected to be negative for any generative")
print("  sampler -- do-nothing is a point mass sitting on the target -- so judge")
print("  the LOW and HIGH thirds, and read pooled skill only against itself.")
print("\n  prior art at eta=5, 5/50/95-percentile targets, same QED adapter:")
print("    low +26%, mid -174%, high +25%   (mean MAE 0.1403)")
print("  old aromatic QED adapter, 5/95 targets: +23% pre-RL, +39% after 2 RL rounds")
PY

echo
echo "NEXT"
echo "  If a lower eta lifts skill on the LOW/HIGH thirds without dropping validity"
echo "  below ~0.90, the E2 rows should be re-frozen at that eta -- which affects"
echo "  the logP row too, not just QED."
echo "  If skill is flat in eta, the operating point is exonerated and the next"
echo "  lever is capacity (INTERIOR_ATTN) or an RL round."
