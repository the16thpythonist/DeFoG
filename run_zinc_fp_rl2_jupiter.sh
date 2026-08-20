#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=09:00:00
#SBATCH --output=zinc_fp_rl2_%j.out

# ROUND 2: ratchet on the round-1 kl=0.5 adapter, with a reward that lets
# similarity compete.
#
# WHAT ROUND 1 DID, AND WHY IT DID IT
#   lift 0.1631 -> 0.1588 (-0.004, inside noise)
#   disconnected 14.65% -> 10.49% (-28% relative)
# RL spent its whole budget on connectivity and none on similarity. That was the
# reward's doing, not a property of the problem: a disconnected sample scored a
# flat -0.5 against a connected sample's ~0.30, so repairing one fragment paid
# ~0.80 while improving an already-connected molecule 0.30 -> 0.35 paid ~0.05.
# Sixteen times the return, for an easier edit.
#
# THE FIX: LARGEST-FRAGMENT PARTIAL CREDIT
#   connected     -> Tanimoto
#   disconnected  -> Tanimoto(largest fragment) - delta
#   invalid       -> -1.0
# delta sets the repair-vs-similarity ratio directly, and unlike a flat constant
# it can never invert the ordering -- the same molecule always scores exactly
# delta higher intact than in pieces, so nothing rewards fragmenting. (A constant
# near the mean Tanimoto WOULD invert: it would put a 0.20-similarity connected
# molecule below a fragment.) Verified offline before this run.
#
#     arm0  delta=0.05   ratio  1:1   similarity competes equally
#     arm1  delta=0.15   ratio  3:1
#     arm2  delta=0.30   ratio  6:1
#     arm3  delta=0.50   ratio 10:1   (round 1's flat penalty was ~16:1)
#
# KL is held at 0.5 -- the efficient point in round 1 (5 points of fragments for
# 0.0008 lift per point, 3-5x better than any other arm) -- so only the reward
# balance moves. The KL reference is the ROUND-1 ADAPTER, not the original: this
# is a ratchet, and round 2 is anchored to what round 1 achieved.
#
# STARTING POINT is round 1's FINAL EMA adapter, which is the checkpoint that was
# actually measured (+0.1588 lift, 10.49% disc). fp_adapter_rl_best.ckpt exists
# too but was selected on ROLLOUT-config reward and never evaluated at deploy
# config, so starting there would make round 2's result unreadable.
#
# CONNECTIVITY FLOOR 0.1049, FIXED BEFORE THE RUN. An arm that improves Tanimoto
# while fragmenting more than round 1 is reported but fails the floor and is
# excluded from selection. The connectivity gain is banked, not currency.
# Choosing that rule after seeing the numbers is how a marginal result gets
# argued into looking good.
#
# WHAT WOULD COUNT
# An arm that holds disc <= 10.49% AND raises lift above 0.1588 by more than the
# ~0.012 run-to-run noise. If every arm is flat on lift again even at delta=0.05,
# where similarity and repair pay equally, then direct optimisation genuinely
# cannot move this metric and the frozen base is the remaining explanation.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BASE="ckpts/zinc_rl2_seed42/best_model"
# Round 1, kl=0.5, FINAL EMA -- the measured artifact.
ADAPTER="experiments/results/adapter_rl_finetune_fp__zinc/09_08_2026__20_13__5SfL/fp_adapter_rl.ckpt"
VOCAB="e1_kekulized"
FP_BITS=1024
LR=1e-4
KL=0.5
DISC_FLOOR=0.1049
MAX_HOURS=6.0

DELTAS=( 0.05  0.15  0.30  0.50 )
TAGS=(   d005  d015  d030  d050 )

[ -f "${BASE}.ckpt" ] || { echo "ERROR: ${BASE}.ckpt missing"; exit 1; }
[ -f "$ADAPTER" ]     || { echo "ERROR: $ADAPTER missing"; exit 1; }

echo "ZINC fingerprint adapter -- RL ROUND 2 (delta sweep) @ $(date)"
echo "starting adapter: ${ADAPTER}"
echo "round 1 result being built on: lift +0.1588, disc 10.49%"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python - <<'PY' || { echo "ERROR: CUDA preflight failed on $(hostname) -- bad node, resubmit"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary + adapter agreement, and a direct check that the delta reward orders
# correctly -- a reward that scored fragments ABOVE intact molecules would train
# happily and produce exactly the artifact this run exists to avoid.
python - "$BASE" "$VOCAB" "$ADAPTER" "$FP_BITS" <<'PY'
import sys, importlib.util
import numpy as np
try:
    from defog.core import DeFoGModel, AdaLNAdapter
    from defog.data import vocabulary
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    spec = importlib.util.spec_from_file_location("at", "experiments/adapter_training__zinc.py")
    m = importlib.util.module_from_spec(spec); sys.modules["at"] = m
    spec.loader.exec_module(m)
    atoms, bonds, kek, src = m._vocabulary(sys.argv[2])
    base = DeFoGModel.load(sys.argv[1], device="cpu")
    ad = AdaLNAdapter.load(sys.argv[3], device="cpu")
    rl = importlib.util.spec_from_file_location(
        "rlfp", "experiments/adapter_rl_finetune_fp__zinc.py")
    rlm = importlib.util.module_from_spec(rl); sys.modules["rlfp"] = rlm
    rl.loader.exec_module(rlm)
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})"); raise SystemExit(2)
try:
    print(vocabulary.check_model(base, atoms, bonds, what=sys.argv[1]))
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}"); raise SystemExit(1)
if int(ad.cond_dim) != int(sys.argv[4]):
    print(f"BITS MISMATCH: adapter cond_dim={ad.cond_dim} vs FP_BITS={sys.argv[4]}")
    raise SystemExit(3)
ad.check_compatible(base)

# reward ordering: intact must beat the SAME molecule fragmented, at every delta
T = "COCCNC(=O)c1ccc2c(c1)OCO2"
tgt = rlm.morgan_bits(Chem.MolFromSmiles(T), 2, 1024)
def sc(smi, delta):
    r = rlm.FPMatchReward(None, None, 2, 1024, disconnect_delta=delta)
    mol = Chem.MolFromSmiles(smi); pen = 0.0; scored = mol
    if "." in smi:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        scored, pen = max(frags, key=lambda f: f.GetNumHeavyAtoms()), r.delta
    fp = rlm.morgan_bits(scored, 2, 1024)
    i = float(fp @ tgt); u = float(fp.sum() + tgt.sum() - i)
    return (i / u if u > 0 else 0.0) - pen
for d in (0.05, 0.15, 0.30, 0.50):
    a, b = sc(T, d), sc(T + ".CC", d)
    if not b < a:
        print(f"REWARD ORDERING BROKEN at delta={d}: fragmented {b} >= intact {a}")
        raise SystemExit(4)
print("reward ordering OK at every delta (intact > same molecule fragmented)")
print(f"adapter OK: cond_dim={ad.cond_dim} params={sum(p.numel() for p in ad.parameters()):,}")
PY
rc=$?
[ $rc -eq 1 ] && { echo "ERROR: base and vocabulary disagree -- refusing"; exit 1; }
[ $rc -eq 3 ] && { echo "ERROR: FP_BITS disagrees with the adapter -- refusing"; exit 1; }
[ $rc -eq 4 ] && { echo "ERROR: delta reward orders fragments above intact -- refusing"; exit 1; }
[ $rc -ne 0 ] && { echo "ERROR: preflight could not run (exit $rc)"; exit 1; }

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_rl_finetune_fp__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --BASE_CKPT "'${BASE}'" \
        --ADAPTER_CKPT "'${ADAPTER}'" \
        --FP_BITS ${FP_BITS} \
        --FP_COUNTS True \
        --FP_FROM "'decoded'" \
        --KL_COEF ${KL} \
        --DISCONNECT_DELTA ${DELTAS[$i]} \
        --DISC_FLOOR ${DISC_FLOOR} \
        --LR ${LR} \
        --ROLLOUT_SIZE 64 \
        --N_GROUPS 4 \
        --ETA 25.0 \
        --MAX_TIME_HOURS ${MAX_HOURS} \
        --__DEBUG__ False \
        > "zinc_fp_rl2_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (delta=${DELTAS[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "POST-RL eval" "zinc_fp_rl2_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that reached post-RL eval: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_fp_rl2_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== ROUND 2: similarity-vs-connectivity trade, from round 1's kl=0.5 adapter ==="
echo "    round 1 endpoint (= every arm's starting point): lift +0.1588  disc 10.49%"
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]}  delta=${DELTAS[$i]} ---"
    grep -E "relative weight drift from init: .*$|w=1.0: lift|paired:|connectivity floor|barely moved" \
        "zinc_fp_rl2_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -5
done

echo
echo "HOW TO READ THIS"
echo "  1. Weight drift first. Below 1e-4 the arm did not train and is VOID."
echo "  2. Connectivity floor: an arm above 10.49% disc is excluded from selection"
echo "     whatever its Tanimoto. That rule was fixed before the run."
echo "  3. Among arms that pass the floor, lift must beat +0.1588 by more than the"
echo "     ~0.012 run-to-run noise to count. Paired improvement count over the 12"
echo "     targets matters more than the mean, which one large mover can carry."
echo
echo "IF EVERY ARM IS FLAT ON LIFT -- including delta=0.05, where repairing a"
echo "fragment and improving similarity pay EQUALLY -- then direct optimisation"
echo "cannot move this metric, and the frozen base is what is left. That would be"
echo "the evidence justifying an unfreeze/LoRA experiment, which no result so far"
echo "has earned."
