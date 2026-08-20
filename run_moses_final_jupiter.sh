#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=moses_final_%j.out

# E1 stage 3: THE ONE EVALUATION PASS ON TEST for the kekulized MOSES lineage.
#
# Protocol section 5 step 3. Everything up to here happened on validation. This
# is the only place the MOSES test split is read, and it is meant to run once.
# If it runs again, the paper has to say so.
#
# THE FROZEN CONFIGURATION AND WHERE IT CAME FROM
#   steps 500, eta 25, omega 0, polydec, n=30,000, seed 42
#
# Chosen on validation across two stages:
#   stage 1 (job 1313054, n=2000, 32 points) settled the coarse structure --
#     500 steps beats 50 decisively, omega=0.25 is bad everywhere -- and could
#     not separate the leaders.
#   stage 2 (job 1317727, n=10,000, 7 arms) re-ran the leaders plus the
#     inherited config plus a SEED DUPLICATE. The duplicate is what decided it:
#     one configuration run at seeds 42 and 777 scored 0.5611 and 0.5919, a gap
#     of 0.0308, and the two draws ranked 3rd and LAST. The entire spread among
#     the top five was 0.0268 -- smaller than that. So no configuration is
#     separably better than any other, and the pre-registered tie-break (within
#     the floor: fewer steps, then lower eta, then lower omega) lands on
#     eta=25/omega=0.
#
# That happens to be the configuration every previous MOSES number was produced
# at, inherited from the AROMATIC model. The sweep's result is therefore a null
# -- and a useful one: those numbers stop being "settings copied from a
# different model, never checked" and become "settings confirmed within noise of
# the best of 32 alternatives tuned on validation".
#
# n=30,000 is the MOSES convention; the test reference is the full 176,074-
# molecule split, and MOSES additionally scores FCD against test_scaffolds,
# which final_eval.py writes out alongside.
#
# ONE PROCESS ON ONE GPU, THREE IDLE, ON PURPOSE. Splitting the draw across
# devices would give the one artifact that must be reproducible several RNG
# streams and no single seed that regenerates it. Ninety minutes of idle GPU is
# a fair price for a number that can be reproduced exactly.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

CKPT="ckpts/moses_kekrl_b0s43/best_model"
REP="kekulized_v2"
OUT="final_moses_${SLURM_JOB_ID}"

[ -f "${CKPT}.ckpt" ] || { echo "ERROR: ${CKPT}.ckpt missing"; exit 1; }

echo "MOSES kekulized E1 -- FROZEN TEST PASS @ $(date)"
echo "  ckpt=${CKPT}  representation=${REP}"
echo "  frozen: steps=500 eta=25 omega=0 polydec n=30000 seed=42"
echo "  chosen on validation, jobs 1313054 (stage 1) + 1317727 (stage 2)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# The vocabulary guard matters most here: a mis-decoded one-shot test pass would
# produce a plausible table row made of the wrong elements.
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
print(f"representation={rep.name}  atoms={atoms}")
print(f"test reference: {len(split.test_smiles)} molecules"
      + (f"  + test_scaffolds: {len(split.test_scaffolds_smiles)}"
         if hasattr(split, "test_scaffolds_smiles") else ""))
PY
rc=$?
[ $rc -eq 1 ] && { echo "ERROR: checkpoint and representation disagree -- refusing"; exit 1; }
[ $rc -ne 0 ] && { echo "ERROR: preflight could not run (exit $rc)"; exit 1; }

CUDA_VISIBLE_DEVICES=0 python -u scripts/final_eval.py \
    --ckpt "$CKPT" \
    --dataset moses \
    --representation "$REP" \
    --tag seed42 \
    --sample-steps 500 \
    --eta 25 \
    --omega 0 \
    --time-distortion polydec \
    --num-samples 30000 \
    --chunk 500 \
    --split test \
    --seed 42 \
    --sweep-dir "sweep2_moses_1317727" \
    --out-dir "$OUT"

rc=$?
echo "finished at $(date) (exit ${rc})"
if [ $rc -ne 0 ]; then echo "ERROR: final eval failed"; exit $rc; fi

echo
ls -la "$OUT"/
echo
echo "NEXT: pull ${OUT}/ locally and score in .venv_metrics:"
echo "  e1_metrics.py --generated ${OUT}/seed42.smi \\"
echo "    --reference ${OUT}/_test_reference.smi --dataset moses"
echo "This yields the official MOSES suite (Filters/SNN/Frag/Scaf/FCD) via"
echo "molsets 0.3.1 against the full test split -- the numbers that go in E1."
echo
echo "Report in the table caption, per protocol section 5 step 4:"
echo "  steps=500 eta=25 omega=0 polydec n=30000 seed=42"
echo "  representation=kekulized_v2 (7 atom / 3 bond, remove_h=True, aromatic=False)"
echo "  validity convention: state which of relaxed_largest_frag /"
echo "  strict_largest_frag / whole_molecule is quoted -- they differ, and"
echo "  quoting a corrected number against someone else's uncorrected one"
echo "  inflates the result."
