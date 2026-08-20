#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=moses_sweep2_%j.out

# E1 stage 2: separate the stage-1 leaders at n=10,000, still VALIDATION only.
#
# WHY A SECOND STAGE
# Stage 1 (job 1313054) settled the coarse structure -- 500 steps beats 50 by a
# mile, eta does error correction, omega=0.25 is bad everywhere -- and could NOT
# settle the winner. Its top five span 0.080 FCD against a +/-0.046 noise floor
# at n=2000. Crowning the argmin of 32 draws at that resolution is a winner's
# curse, so the leaders are re-run at n=10,000 where the floor is ~5x tighter.
#
# stage-1 leaders (validation FCD at n=2000):
#     500/25/0.05   1.0083
#     500/50/0.05   1.0194
#     500/50/0.00   1.0269
#     500/50/0.10   1.0795
#     500/25/0.10   1.0884
#     500/25/0.00   1.1434   <- the INHERITED config, rank 8, carried as the
#                               reference point every existing MOSES number was
#                               produced at. Without it in this run there is no
#                               way to say whether the sweep bought anything.
#
# THE SEED-DUPLICATE ARM IS THE POINT OF THIS JOB AS MUCH AS THE RANKING.
# The +/-0.0084 FCD floor on record was measured against a much LARGER reference
# than MOSES validation, which is only 5,000 molecules. A floor measured under
# other conditions cannot license a claim here. So 500/25/0.05 is run TWICE at
# different seeds: two independent generations from one checkpoint at one
# config, whose spread IS the floor against this exact reference. Any gap
# between configs smaller than that spread is not a result.
#
# SELECTION CRITERION -- unchanged from stage 1, fixed before either ran:
#     primary     validation FCD, lower better
#     constraint  validity >= 0.985  (all 32 stage-1 points cleared this)
#     tie-break   within the floor prefer FEWER STEPS, then LOWER eta
# The tie-break is why the inherited config matters: if it lands inside the
# floor of the leader, the honest answer is that the sweep found no improvement
# and eta=25/omega=0 stays -- which is a perfectly good outcome and much easier
# to defend than a 0.05 FCD "win" pulled from noise.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

CKPT="ckpts/moses_kekrl_b0s43/best_model"
REP="kekulized_v2"
N=10000
OUT="sweep2_moses_${SLURM_JOB_ID}"

[ -f "${CKPT}.ckpt" ] || { echo "ERROR: ${CKPT}.ckpt missing"; exit 1; }
mkdir -p "$OUT"

#            0        1        2        3        4        5        6
ETAS=(      25       50       50       50       25       25       25 )
OMEGAS=(  0.05     0.05     0.00     0.10     0.10     0.00     0.05 )
SEEDS=(     42       42       42       42       42       42      777 )   # arm6 = seed duplicate of arm0
TAGS=( a_e25w005 b_e50w005 c_e50w000 d_e50w010 e_e25w010 f_e25w000_INHERITED g_e25w005_SEED777 )

echo "MOSES kekulized E1 stage-2 VALIDATION refinement @ $(date)"
echo "ckpt=${CKPT} representation=${REP} n=${N} steps=500 (stage 1 settled steps)"
echo "7 arms on 4 GPUs; arm g is a SEED DUPLICATE of arm a -> their gap IS the noise floor"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

python - "$CKPT" "$REP" <<'PY'
import sys
try:
    from defog.core import DeFoGModel
    from defog.data import moses_reference as mref, vocabulary
    model = DeFoGModel.load(sys.argv[1], device="cpu")
    atoms, bonds, adec, bdec, rep, msg = vocabulary.resolve_and_check(mref, model, sys.argv[2])
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}"); raise SystemExit(1)
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})"); raise SystemExit(2)
print(msg)
split = mref.load_reference_split()
print(f"representation={rep.name}  val={len(split.val_smiles)}  test={len(split.test_smiles)}"
      f"   <- VALIDATION only; test stays untouched until the frozen pass")
PY
rc=$?
[ $rc -eq 1 ] && { echo "ERROR: checkpoint and representation disagree -- refusing"; exit 1; }
[ $rc -ne 0 ] && { echo "ERROR: preflight could not run (exit $rc)"; exit 1; }

# 7 arms over 4 GPUs: GPU i takes arms i, i+4.
for i in 0 1 2 3 4 5 6; do
    gpu=$(( i % 4 ))
    (
        CUDA_VISIBLE_DEVICES=$gpu python -u scripts/sweep_sampling.py \
            --ckpt "$CKPT" \
            --dataset moses \
            --representation "$REP" \
            --steps 500 \
            --eta "${ETAS[$i]}" \
            --omega "${OMEGAS[$i]}" \
            --num-samples ${N} \
            --chunk 500 \
            --seed "${SEEDS[$i]}" \
            --out-dir "${OUT}/${TAGS[$i]}" \
            > "moses_sweep2_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1
    ) &
    echo "launched ${TAGS[$i]} (eta=${ETAS[$i]} omega=${OMEGAS[$i]} seed=${SEEDS[$i]}) on GPU ${gpu}"
    sleep 3
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    [ -n "$(ls ${OUT}/${t}/*.smi 2>/dev/null)" ] && OK=$((OK+1))
done
echo "arms that wrote SMILES: ${OK} / 7"
if [ "$OK" -lt 7 ]; then
    echo "ERROR: incomplete; tracebacks follow"
    grep -hA6 "Traceback" moses_sweep2_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "NEXT: pull ${OUT}/ locally, score every arm against the SAME validation"
echo "reference in .venv_metrics, and read arm a vs arm g FIRST -- that gap is the"
echo "noise floor, and no ranking below it means anything. If the inherited"
echo "config (arm f) sits inside that floor of the leader, the sweep found"
echo "nothing and eta=25/omega=0 is kept."
