#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=moses_sweep_%j.out

# E1 stage 1: VALIDATION sampling sweep for the kekulized MOSES lineage.
#
# WHY THIS EXISTS
# Every MOSES number on record -- validity 0.9907, FCD 0.5928, the whole
# kekulized-vs-aromatic comparison -- was produced at eta=25, a value inherited
# from the AROMATIC model. This lineage has never had its own sampling sweep, so
# those figures are lower bounds, not E1 rows. DeFoG assembles the rate matrix at
# sampling time, so steps/eta/omega are free parameters that were simply never
# set for this model.
#
# PROTOCOL (docs/unconditional-protocol.md section 5)
#   sweep on validation -> freeze -> ONE evaluation pass on test
# This script touches VALIDATION ONLY. It writes SMILES; scoring happens
# afterwards in the x86 metrics env, which cannot install on JUPITER's aarch64.
#
# SELECTION CRITERION, FIXED BEFORE ANY RESULT EXISTS
#   primary     FCD on validation, lower is better
#   constraint  validity >= 0.985 (model is 0.9907 +/- 0.0035; this disqualifies
#               a config that buys distribution match with validity)
#   tie-break   within the FCD noise floor prefer FEWER STEPS, then LOWER eta
# The tie-break matters: argmin over 32 noisy points is a winner's curse, and
# without a stated rule the temptation is to crown whichever exotic corner of the
# grid happened to draw well.
#
# WHY n=2000 AND A SECOND STAGE
# Measured FCD noise is +/-0.046 at n=2048 and +/-0.0084 at n=10,000. Stage 1
# resolves coarse structure only -- 50 vs 500 steps, whether high eta hurts --
# which is all a 32-point grid can honestly support. The top few configs then get
# re-run at n=10,000 (stage 2) to be separated above the floor. Picking the
# winner straight off stage 1 would be selecting noise.
#
# REPRESENTATION IS NOT OPTIONAL. This lineage is kekulized_v2 (7 atom / 3 bond).
# Decoding it with the default aromatic_v1 (8/5) does not raise -- it yields
# plausible molecules made of the wrong elements. The guard in
# defog/data/vocabulary.py refuses on a dimension mismatch; passing the flag is
# what makes it check the right thing.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

CKPT="ckpts/moses_kekrl_b0s43/best_model"
REP="kekulized_v2"
N=2000
OUT="sweep_moses_${SLURM_JOB_ID}"

[ -f "${CKPT}.ckpt" ] || { echo "ERROR: ${CKPT}.ckpt missing"; exit 1; }
mkdir -p "$OUT"

echo "MOSES kekulized E1 stage-1 VALIDATION sweep @ $(date)"
echo "ckpt=${CKPT} representation=${REP} n=${N} grid=2x4x4=32 points"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# The vocabulary guard, run once before four GPUs spend hours decoding with the
# wrong atom table.
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
print(f"representation={rep.name}  atoms={atoms}  bonds={bonds}")
split = mref.load_reference_split()
print(f"split: train={len(split.train_smiles)}  val={len(split.val_smiles)}  "
      f"test={len(split.test_smiles)}   <- the sweep scores against VAL only")
PY
rc=$?
[ $rc -eq 1 ] && { echo "ERROR: checkpoint and representation disagree -- refusing"; exit 1; }
[ $rc -ne 0 ] && { echo "ERROR: preflight could not run (exit $rc)"; exit 1; }

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/sweep_sampling.py \
        --ckpt "$CKPT" \
        --dataset moses \
        --representation "$REP" \
        --slice ${i}/4 \
        --num-samples ${N} \
        --chunk 500 \
        --out-dir "$OUT" \
        > "moses_sweep_s${i}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched slice ${i}/4 on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

n_pts=$(ls "$OUT"/*.smi 2>/dev/null | wc -l)
echo "grid points written: ${n_pts} / 32"
if [ "$n_pts" -lt 32 ]; then
    echo "ERROR: incomplete sweep; tracebacks follow"
    grep -hA6 "Traceback" moses_sweep_s*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "per-point validity (sampling-side; FCD needs the x86 metrics env):"
grep -hE "steps=|validity" moses_sweep_s*_${SLURM_JOB_ID}.out 2>/dev/null | tail -40

echo
echo "NEXT: pull ${OUT}/ locally and score against the MOSES VALIDATION reference"
echo "  in .venv_metrics via scripts/e1_metrics.py. Do NOT score against test --"
echo "  the whole point of this stage is that test stays untouched until one"
echo "  frozen pass at the end."
