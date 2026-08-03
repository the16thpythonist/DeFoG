#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=moses_rl_pensweep2_%j.out

# JUPITER: MOSES penalty sweep, round 2. Two levers, both measured -- plus a
# base-model diagnostic that may matter more than either.
#
# WHAT ROUND 1 (job 1219218) ESTABLISHED
#   beta      0     3.5      7      14
#   d valid  +.048  +.037  +.041  +.029
#   d FCD    +.842  +.753  +.616  +.334      <- monotone in beta
#   ratio     .057   .049   .067   .088      <- the trade itself improves
# So the term works. No arm gave validity for free.
#
# LEVER 1 -- MORE ITERATIONS. beta=14 selected iteration 24 of 25, i.e. the
# last one, and was still climbing (sanity .859/.875/.891 at it 13-15 ->
# .930/.938/.938 at it 22-24). The control plateaued by it 17 and selected 21.
# That is direct evidence beta=14 had not converged: blocking the cheap route
# (drop aromatic rings) makes the policy take longer to find gains that do not
# cost distribution. Arms C and D run 50 iterations.
#
# LEVER 2 -- MORE BETA. FCD damage fell monotonically across the whole ladder
# with no sign of saturation, so the ladder is extended to 28 and 56.
#
# WHITENING WAS TRIED AND REJECTED -- do not re-propose it without re-reading
# this. The idea was that per-axis standardisation under-weights the correlated
# polarity cluster (TPSA/HBA/nN/HBD) that survives at beta=14. Measuring the
# hack's direction in the reference covariance eigenbasis killed it: the shift
# loads almost entirely on the TOP eigenvalues (3.26, 2.54, 1.80, 1.50, 1.28),
# and whitening divides by sqrt(eigenvalue), so
#     ||shift||^2 standardised = 0.349  ->  whitened = 0.173   (0.50x)
# Whitening HALVES the signal for this hack. It rides the dominant axis of
# drug-like space, so it is genuinely less surprising in Mahalanobis terms even
# though FCD still penalises it. The kernel is kept in the codebase for the
# opposite case (a hack hiding in a rigid low-variance direction).
#
# ADDING DESCRIPTORS WAS ALSO REJECTED. Every residual axis at beta=14 is
# already in the set, and the kernel's own MMD^2 still reads 3.7x base -- it
# can see the gap and is not weighting it. Breadth is not the missing piece.
#
# WATCH FOR DISPLACEMENT. At beta=14 the penalty restored aromatic rings
# (-0.47 -> -0.17 sigma) but TPSA got WORSE than the unpenalised arm
# (-0.253 -> -0.318). Constraining one axis pushed the policy to buy validity
# on another. If beta=28/56 shows the same pattern with no net MMD^2 gain, the
# honest conclusion is that MOSES sanity headroom is simply not free, and the
# remaining lever is the base model, not the reward.
#
# The E1 checkpoints are untouched, as always.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/moses_e1_seed42/best_model"
SEED=42
#        A      B      C      D
BETAS=(  28     56     14     28   )
ITERS=(  25     25     50     50   )
TAGS=(   b28    b56    b14i50 b28i50 )

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

mkdir -p experiments/results/gdpo_sanity
echo "MOSES RL penalty sweep 2 @ $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# ---------------------------------------------------------------------------
# Base-model validity diagnostic. Runs FIRST, on one GPU, ~3 min. The other
# three sit idle for that time, which is the price of getting this answer in
# the same allocation rather than queueing a second job for it.
#
# This is the highest-leverage open question: MOSES base validity ~0.90 against
# GuacaMol ~0.98, on the easier dataset. If most failures turn out to be
# kekulization, the fix is the representation (ZINC trains kekulized and hits
# 0.99) and no amount of reward shaping competes with it.
# ---------------------------------------------------------------------------
echo
echo "=== base-model failure-mode diagnostic ==="
CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_validity.py \
    --ckpt "${BASE}" --dataset moses --n 1024 --steps 500 --eta 25 \
    --out "moses_validity_diagnosis_${SLURM_JOB_ID}.json" \
    2>&1 | tee "moses_diag_${SLURM_JOB_ID}.out"
echo "=== diagnostic done ==="
echo

for i in 0 1 2 3; do
    b=${BETAS[$i]}; it=${ITERS[$i]}; tag=${TAGS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_sanity.py \
        --DATASET "'moses'" \
        --SEED ${SEED} \
        --BETA_MMD "${b}" \
        --ALPHA_FRAG "0.0" \
        --MMD_KERNEL "'descriptor'" \
        --ITERATIONS "${it}" \
        --BASE_CKPT "'${BASE}'" \
        --OUT_CKPT_DIR "'ckpts/moses_rlpen2_${tag}'" \
        --__DEBUG__ False \
        > "moses_rlpen2_${tag}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched beta=${b} iters=${it} (${tag}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "sweep finished at $(date)"

OK=0
for tag in "${TAGS[@]}"; do
    grep -qE "saved final-iteration model" "moses_rlpen2_${tag}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that produced a model: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA5 "Traceback" moses_rlpen2_*_${SLURM_JOB_ID}.out 2>/dev/null | head -30
    exit 1
fi

echo
echo "=== per-arm summary ==="
for i in 0 1 2 3; do
    echo "--- beta=${BETAS[$i]} iters=${ITERS[$i]} (${TAGS[$i]}) ---"
    grep -E "loading best checkpoint|^validity \(relaxed\)|^disconnected|^wonky rings|^sanity \(all" \
        "moses_rlpen2_${TAGS[$i]}_${SLURM_JOB_ID}.out" | tail -5
done

echo
echo "NEXT: score each arm's after.smi against validation_reference.smi with"
echo "scripts/e1_metrics.py. Round-1 reference points, same reference and n:"
echo "  base FCD 1.465 | beta=0 2.307 | beta=7 2.081 | beta=14 1.800"
