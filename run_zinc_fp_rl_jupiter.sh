#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=09:00:00
#SBATCH --output=zinc_fp_rl_%j.out

# JUPITER: RL-finetune the shipped fingerprint adapter with TANIMOTO as the reward.
#
# WHY THIS AND NOT MORE ARCHITECTURE
# Four explanations for the ~0.33 Tanimoto plateau have now been eliminated by
# measurement, not argument:
#   generalization   train-vs-heldout gap <= 0 in all four ablation arms
#   capacity         2048 bits, 512-wide trunk, MLP pre-encoder: all null or worse
#   guidance weight  w=1.0 IS the conditional denoiser undiluted; no headroom
#   size mismatch    perfect size-matching is worth +0.0012
#
# What is left is the objective. The adapter is trained by denoising cross-entropy
# -- reconstruct the exact molecule from its own fingerprint -- and at high noise
# most of that loss is coarse structure: size, atom counts, gross connectivity. The
# fine substructure agreement Tanimoto measures is a small share of what is being
# minimised. On that reading the adapter is fitting its objective correctly and the
# objective is not the metric, which also explains why capacity did nothing: you
# cannot fix an objective mismatch with parameters.
#
# NOTHING IN THIS LINE HAS EVER OPTIMISED TANIMOTO. This does.
#
# WHY THE REWARD IS TRUSTWORTHY
# reward = binary Tanimoto to the target the rollout was conditioned on = exactly
# the eval metric. No learned head, so no proxy to game -- unlike the property-head
# RL, where the reward was a model. Connectivity-first ordering keeps it from
# collapsing to fragments: invalid < disconnected < any connected molecule.
#
# THIS IS ALSO THE DIAGNOSTIC FOR THE FROZEN BASE
# If direct reward optimisation cannot move Tanimoto either, the objective was not
# the constraint and the frozen base is -- which is the evidence that would justify
# unfreezing it (LoRA), a step that costs the swappable-adapter property molsmith's
# whole design rests on. Worth paying for that evidence before paying for that.
#
# THE FAILURE MODE THIS RUN IS BUILT TO DETECT
# Earlier adapter-RL rounds were VOID: lr=1e-5 with ema=0.999 left the adapter
# barely moving, so "RL did not help" was indistinguishable from "RL did not run".
# Here lr=1e-4, ema=0.99, and relative weight drift is reported every 20 iterations
# and again at the end, with an explicit VOID warning below 1e-4. A flat result with
# healthy drift is a real answer; a flat result with dead drift is a broken run.
#
# ARMS: the KL coefficient, which sets how far RL may drag the adapter from the
# trained reference. We have no prior for the right value under THIS reward.
#     arm0  kl=0.0    unconstrained -- max movement, max risk of wrecking validity
#     arm1  kl=0.05
#     arm2  kl=0.2    what the property-adapter RL used
#     arm3  kl=0.5    conservative
# Read validity and disconnection alongside lift: an arm that wins on Tanimoto by
# emitting fragments has not won anything.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/zinc_rl2_seed42/best_model"
ADAPTER="experiments/results/adapter_fingerprint__zinc/08_08_2026__13_03__pkM7/fp_adapter.ckpt"
VOCAB="e1_kekulized"
FP_BITS=1024            # must equal the adapter's cond_dim; the run refuses otherwise
LR=1e-4
MAX_HOURS=6.0

KLS=(  0.0   0.05  0.2   0.5  )
TAGS=( kl000 kl005 kl020 kl050 )

[ -f "${BASE}.ckpt" ] || { echo "ERROR: ${BASE}.ckpt missing"; exit 1; }
[ -f "$ADAPTER" ]     || { echo "ERROR: $ADAPTER missing"; exit 1; }

echo "ZINC fingerprint adapter -- TANIMOTO-REWARD RL @ $(date)"
echo "base=${BASE}"
echo "adapter=${ADAPTER}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary guard + the FP_BITS/adapter agreement that would otherwise surface as a
# shape error deep inside the first sampling call, an hour into the allocation.
python - "$BASE" "$VOCAB" "$ADAPTER" "$FP_BITS" <<'PY'
import sys, importlib.util
try:
    from defog.core import DeFoGModel, AdaLNAdapter
    from defog.data import vocabulary
    spec = importlib.util.spec_from_file_location("at", "experiments/adapter_training__zinc.py")
    m = importlib.util.module_from_spec(spec); sys.modules["at"] = m
    spec.loader.exec_module(m)
    atoms, bonds, kek, src = m._vocabulary(sys.argv[2])
    base = DeFoGModel.load(sys.argv[1], device="cpu")
    ad = AdaLNAdapter.load(sys.argv[3], device="cpu")
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})"); raise SystemExit(2)
try:
    print(vocabulary.check_model(base, atoms, bonds, what=sys.argv[1]))
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}"); raise SystemExit(1)
if int(ad.cond_dim) != int(sys.argv[4]):
    print(f"BITS MISMATCH: adapter cond_dim={ad.cond_dim} but FP_BITS={sys.argv[4]}")
    raise SystemExit(3)
ad.check_compatible(base)
print(f"adapter OK: cond_dim={ad.cond_dim} hidden={ad.hidden} "
      f"interior_ff={ad.interior_ff} params={sum(p.numel() for p in ad.parameters()):,}")
print("atoms:", atoms, "| bonds:", bonds)
PY
rc=$?
[ $rc -eq 1 ] && { echo "ERROR: base and vocabulary disagree -- refusing"; exit 1; }
[ $rc -eq 3 ] && { echo "ERROR: FP_BITS disagrees with the adapter -- refusing"; exit 1; }
[ $rc -ne 0 ] && { echo "ERROR: preflight could not run (exit $rc)"; exit 1; }

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_rl_finetune_fp__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --BASE_CKPT "'${BASE}'" \
        --ADAPTER_CKPT "'${ADAPTER}'" \
        --FP_BITS ${FP_BITS} \
        --FP_COUNTS True \
        --FP_FROM "'decoded'" \
        --KL_COEF ${KLS[$i]} \
        --LR ${LR} \
        --ROLLOUT_SIZE 64 \
        --N_GROUPS 4 \
        --ETA 25.0 \
        --MAX_TIME_HOURS ${MAX_HOURS} \
        --__DEBUG__ False \
        > "zinc_fp_rl_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (kl=${KLS[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "POST-RL eval" "zinc_fp_rl_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that reached post-RL eval: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_fp_rl_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== Tanimoto-reward RL: pre -> post, per KL coefficient ==="
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]}  kl=${KLS[$i]} ---"
    grep -E "relative weight drift|w=1.0: lift|paired:|barely moved" \
        "zinc_fp_rl_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -5
done

echo
echo "HOW TO READ THIS"
echo "  FIRST check weight drift. Below 1e-4 the arm did not train and its result is"
echo "  VOID regardless of what the lift says -- that is what invalidated the earlier"
echo "  adapter-RL rounds and it must be ruled out before anything else is read."
echo "  Then: paired improvement count matters more than the mean, which one large"
echo "  mover can carry. Target-set spread here is ~0.008 and run-to-run noise ~0.012,"
echo "  so a mean delta below ~0.012 is not a result."
echo "  Watch disconnection%: Tanimoto gained by emitting fragments is not a gain."
echo
echo "IF EVERY ARM IS FLAT WITH HEALTHY DRIFT, that is the informative outcome: the"
echo "objective was not the constraint either, and the frozen base is the remaining"
echo "explanation. That is what would justify the LoRA/unfreeze experiment."
