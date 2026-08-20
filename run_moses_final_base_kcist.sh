#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=moses_base
#SBATCH --output=moses_final_base_%j.out

# E1: frozen test pass for the PRE-RL kekulized MOSES base (moses_kek_seed44),
# the parent of moses_kekrl_b0s43. Reported as a second table row so the E1 row
# shows the base and the RL'd model side by side rather than only the tuned one.
#
# RUNNING HERE BECAUSE JUPITER IS DOWN: all 1971 booster nodes drained for a
# system deployment, zero jobs running cluster-wide. The equivalent JUPITER job
# (1324954) is queued behind that and will never start before this does.
#
# NOT a second pass on the same model. One pass per model, no re-tuning after
# seeing test numbers. The sampling configuration is the one frozen from the RL
# model's validation sweep, applied UNCHANGED here -- re-sweeping per model would
# be more tuning, not less, and the sweep separated nothing beyond its noise
# floor anyway. The transfer is disclosed in the manuscript caption.
#
# VERSION SKEW IS THE REAL RISK HERE, not the compute. This repo checkout predates
# defog/data entirely; the module was uploaded on top of an older defog/core. The
# import chain was verified before submitting, and SMOKE_N below runs a tiny
# sample first: if the two halves disagree the run dies in seconds rather than
# producing a plausible wrong number for a paper table 40 minutes later.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"

CKPT="ckpts/moses_kek_seed44/best_model"
REP="kekulized_v2"
OUT="final_mosesbase_${SLURM_JOB_ID}"
PY=.venv/bin/python

[ -f "${CKPT}.ckpt" ] || { echo "ERROR: ${CKPT}.ckpt missing"; exit 1; }

echo "MOSES kekulized E1 -- FROZEN TEST PASS (PRE-RL BASE) @ $(date) on $(hostname)"
echo "  ckpt=${CKPT}  representation=${REP}"
echo "  frozen: steps=500 eta=25 omega=0 polydec n=30000 seed=42"
echo "  md5(ckpt)=$(md5sum ${CKPT}.ckpt | cut -d' ' -f1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False on a GPU node"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary guard + a real 32-sample generate/decode, so version skew between the
# uploaded defog/data and this checkout's defog/core surfaces now.
$PY - "$CKPT" "$REP" <<'PY'
import sys
try:
    import torch
    from defog.core import DeFoGModel
    from defog.data import moses_reference as mref, vocabulary
    from defog.domains.molecule import validity_report
    model = DeFoGModel.load(sys.argv[1]).to("cuda").eval()
    atoms, bonds, adec, bdec, rep, msg = vocabulary.resolve_and_check(mref, model, sys.argv[2])
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}"); raise SystemExit(1)
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})"); raise SystemExit(2)
print(msg)
s = mref.load_reference_split()
print(f"split: train={len(s.train_smiles)} test={len(s.test_smiles)} "
      f"test_scaffolds={len(s.test_scaffolds_smiles)}")
try:
    smp = model.sample(num_samples=32, sample_steps=50, eta=25.0, omega=0.0,
                       time_distortion="polydec", device="cuda", show_progress=False)
    r = validity_report(smp, adec, bdec)
    v = r["validity_relaxed_largest_frag"]
except Exception as exc:
    print(f"SMOKE SAMPLE FAILED ({type(exc).__name__}: {exc})"); raise SystemExit(3)
print(f"smoke: 32 samples at 50 steps -> validity {v:.3f}")
# 50 steps is a weak setting; anything above 0.5 means the stack is coherent.
# Near-zero would mean the old core and new data module disagree about the graph.
if v < 0.5:
    print(f"SMOKE VALIDITY {v:.3f} TOO LOW -- suspect version skew, refusing"); raise SystemExit(4)
PY
rc=$?
[ $rc -eq 1 ] && { echo "ERROR: checkpoint/representation disagree -- refusing"; exit 1; }
[ $rc -eq 3 ] && { echo "ERROR: sampling broken on this checkout -- refusing"; exit 1; }
[ $rc -eq 4 ] && { echo "ERROR: smoke validity too low -- suspect version skew"; exit 1; }
[ $rc -ne 0 ] && { echo "ERROR: preflight could not run (exit $rc)"; exit 1; }

$PY scripts/final_eval.py \
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
    --sweep-dir "sweep2_moses_1317727 (on the RL model; config applied unchanged)" \
    --out-dir "$OUT"
rc=$?
echo "finished at $(date) (exit ${rc})"
[ $rc -ne 0 ] && exit $rc

ls -la "$OUT"/
echo
echo "NEXT: score in .venv_metrics on this same cluster:"
echo "  .venv_metrics/bin/python scripts/e1_metrics.py \\"
echo "    --generated ${OUT}/seed42.smi \\"
echo "    --reference ${OUT}/_test_reference.smi \\"
echo "    --reference-scaffolds ${OUT}/_test_scaffolds_reference.smi \\"
echo "    --dataset moses --out ${OUT}/moses_test_metrics.json"
echo "Quote the TestSF columns; take validity from seed42.json, NOT from"
echo "moses_validity (which is 1.0 by construction on a pre-filtered file)."
