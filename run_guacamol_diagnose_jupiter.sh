#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
#SBATCH --output=guacamol_diagnose_%j.out

# JUPITER: is GuacaMol leaving validity on the table the way MOSES was?
#
# THE QUESTION
# MOSES trained on an AROMATIC bond class and lost 11.5 validity points to
# kekulization failures -- 118 of 120 hard failures, against exactly one valence
# error. Switching to a kekulized representation took it 0.8843 -> 0.9907 with
# FCD improving 1.478 -> 1.126.
#
# GuacaMol is ALSO aromatic (12 atom types, AROMATIC in the bond set) and has
# never been checked. It sits at ~0.98 validity, so the ceiling is far lower
# than MOSES's was -- but the diagnostic costs minutes and the alternative is
# guessing.
#
# This is also the loose end in the MOSES story. Aromatic encoding alone does
# NOT explain why MOSES was bad: GuacaMol uses the same encoding and reaches
# 0.98. Something else made MOSES unusually prone to unkekulizable output, and
# it was never identified. GuacaMol's failure breakdown is direct evidence
# either way:
#
#   mostly kekulize failures  -> the encoding is the shared cause and GuacaMol
#                                has the same fix available, just less of it
#   mostly something else     -> MOSES had a second, distinct problem, and
#                                whatever it was is still unexplained
#
# Four seeds, one per GPU, so the answer carries a spread rather than resting on
# one checkpoint. eta=75 is GuacaMol's frozen deploy value (from its own sweep),
# not an inherited guess.
#
# No representation flag: GuacaMol defines no named representations, so the
# dataset default is the only vocabulary and the channel-count guard checks it.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

N=2048
STEPS=500
ETA=75
SEEDS=(42 43 44 45)

for s in "${SEEDS[@]}"; do
    if [ ! -f "ckpts/guacamol_e1_seed${s}/best_model.ckpt" ]; then
        echo "ERROR: ckpts/guacamol_e1_seed${s}/best_model.ckpt missing"; exit 1
    fi
done

echo "GuacaMol failure-mode diagnostic @ $(date)   n=${N} steps=${STEPS} eta=${ETA}"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK")
PY

for i in 0 1 2 3; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u scripts/diagnose_validity.py \
        --ckpt "ckpts/guacamol_e1_seed${s}/best_model" --dataset guacamol \
        --n ${N} --steps ${STEPS} --eta ${ETA} --seed $((1000 + s)) \
        --out "guacamol_diag_seed${s}_${SLURM_JOB_ID}.json" \
        > "guacamol_diag_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed ${s} on GPU ${i}"
    sleep 3
done
wait

echo
echo "=== GuacaMol failure breakdown, 4 seeds, n=${N} ==="
python - "$SLURM_JOB_ID" <<'PY'
import json, sys, statistics as st
jid = sys.argv[1]
cats = ["ok", "kekulize", "valence", "disconnected", "other_sanitize", "decode_failed"]
rows = {}
for s in (42, 43, 44, 45):
    try:
        d = json.load(open(f"guacamol_diag_seed{s}_{jid}.json"))
    except Exception as exc:
        print(f"  seed {s}: unreadable ({exc})"); continue
    n = max(1, d["n"])
    rows[s] = {c: d["counts"].get(c, 0) / n for c in cats}
if not rows:
    raise SystemExit("no results")
print(f"{'seed':>5s}" + "".join(f"{c:>16s}" for c in cats))
for s, r in rows.items():
    print(f"{s:>5d}" + "".join(f"{r[c]:>16.4f}" for c in cats))
print(f"{'mean':>5s}" + "".join(
    f"{st.mean([r[c] for r in rows.values()]):>16.4f}" for c in cats))

hard = {c: st.mean([r[c] for r in rows.values()])
        for c in ("kekulize", "valence", "other_sanitize", "decode_failed")}
tot = sum(hard.values())
print()
if tot <= 0:
    print("no hard failures at all -- nothing to fix")
else:
    share = {c: v / tot for c, v in hard.items()}
    top = max(share, key=share.get)
    print(f"hard-failure rate {tot:.4f}; dominant mode '{top}' at {share[top]:.1%} of them")
    print(f"  MOSES for comparison: kekulize was 98.3% of hard failures (118 of 120)")
    if top == "kekulize" and share[top] > 0.5:
        print("  -> SAME cause as MOSES. A kekulized GuacaMol should recover most")
        print(f"     of {tot:.4f}, i.e. up to {100*tot:.1f} validity points.")
    else:
        print("  -> DIFFERENT from MOSES. Kekulizing GuacaMol would not buy much,")
        print("     and MOSES's second, unidentified problem stays unexplained.")
PY

echo
echo "NOTE: this only says whether the failures are kekulization. It does not"
echo "prove a kekulized GuacaMol would train as well -- MOSES needed a full"
echo "retrain to establish that. It says whether that retrain is worth costing."
