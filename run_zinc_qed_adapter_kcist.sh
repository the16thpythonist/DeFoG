#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --job-name=qed_adapter
#SBATCH --output=zinc_qed_adapter_%j.out

# Train a QED adapter on the zinc-kek base -- the missing column of E2.
#
# WHY THIS EXISTS
# E2 (property targeting, FreeGress protocol) needs logP and QED singly plus
# logP+QED jointly. On molsmith/zinc-kek we have clogp@1.1.0 (property `logp`,
# Crippen) and three fingerprint adapters. QED exists ONLY on the old aromatic
# zinc-base, and the two are not interchangeable: different atom ORDER, not just
# a bond-set change, and molsmith's schema_hash gate refuses the cross-serve.
# So the joint column -- the composition claim, the paper's most distinctive
# result -- is blocked on this one adapter.
#
# RUNNING HERE BECAUSE JUPITER IS IN MAINTENANCE (4914 nodes). This checkout had
# no defog/core/adapter.py at all; the whole defog/ tree was synced from the
# working copy so core and data are internally consistent. Verified before
# submitting: base loads, `forward` accepts cond_modulation, adapter builds with
# interior_ff at 4,968,256 params, atom order matches the kekulized reference.
#
# THE FAILURE THIS GATES AGAINST. If the synced core did not thread
# cond_modulation through, the adapter would receive no gradient and remain at
# its zero-init -- which is an EXACT no-op by construction. Training would run,
# loss would look plausible (the frozen base still predicts), and the result
# would be an adapter that steers nothing while looking trained. The weight-drift
# check below is the only thing that separates those two outcomes.
#
# ARMS: 2 learning rates x 2 trunk widths, the grid that produced the shipped
# clogp@1.1.0 on this same base. interior_ff is ON IN ALL FOUR rather than swept
# -- it bought +0.024 Tanimoto on the fingerprint adapter and is cheap, and
# spending arms on it here would cost the capacity comparison that clogp's grid
# provides a direct precedent for.
#
#     arm0  lr 2e-4  hidden 256      arm2  lr 2e-4  hidden 512
#     arm1  lr 4e-4  hidden 256      arm3  lr 4e-4  hidden 512
#
# SELECTION: mean MAE over LOW / MID / HIGH targets, arm rejected if validity
# < 0.95. QED is bounded in [0,1] with its mass mid-range, so an arm can win on
# average by regressing to the middle -- which is exactly where conditioning does
# not need to work. Hence the 5/50/95 percentile targets rather than 5/95.
#
# SCALE, so a good-looking number is not over-read: FreeGress reports an
# UNCONDITIONAL MAE of 0.15 on QED and its own best at 0.04. DiGress barely moves
# it (0.14-0.15). The useful band is narrow and a strong QED result is worth much
# less than a strong logP one.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"

BASE="ckpts/zinc_rl2_seed42/best_model"
VOCAB="e1_kekulized"
PROPERTY="qed"
PROPERTY_FROM="decoded"      # label the molecule the GRAPH is, not the source SMILES
PY=.venv/bin/python
MAX_HOURS=8.0

#        arm0    arm1    arm2    arm3
LRS=(    2e-4    4e-4    2e-4    4e-4 )
HIDDEN=( 256     256     512     512  )
TAGS=(   lr2h256 lr4h256 lr2h512 lr4h512 )

[ -f "${BASE}.ckpt" ] || { echo "ERROR: ${BASE}.ckpt missing"; exit 1; }

echo "ZINC QED adapter (zinc-kek) @ $(date) on $(hostname)"
echo "  base=${BASE}  vocab=${VOCAB}  property=${PROPERTY} from=${PROPERTY_FROM}"
echo "  interior_ff=True in all arms; grid = 2 LR x 2 hidden"
echo "  md5(base)=$(md5sum ${BASE}.ckpt | cut -d' ' -f1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False on a GPU node"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary guard + the gradient check. A zero-init adapter is an exact no-op, so
# "trained but never moved" and "trained and useless" are indistinguishable from
# the loss curve alone. This takes a few optimiser steps and asserts the weights
# actually changed, before four GPUs spend eight hours each.
$PY - "$BASE" "$VOCAB" "$PROPERTY" <<'PY'
import sys
try:
    import torch, importlib.util
    from defog.core import DeFoGModel, AdaLNAdapter, AdapterModule
    from defog.data import zinc_reference as zref, vocabulary
    spec = importlib.util.spec_from_file_location("at", "experiments/adapter_training__zinc.py")
    at = importlib.util.module_from_spec(spec); sys.modules["at"] = at
    spec.loader.exec_module(at)
    atoms, bonds, kek, src = at._vocabulary(sys.argv[2])
    if sys.argv[3] not in at.PROP_FNS:
        print(f"PROPERTY {sys.argv[3]!r} not in PROP_FNS {sorted(at.PROP_FNS)}"); raise SystemExit(5)
    base = DeFoGModel.load(sys.argv[1], device="cpu")
except Exception as exc:
    print(f"PREFLIGHT CRASHED ({type(exc).__name__}: {exc})"); raise SystemExit(2)
try:
    print(vocabulary.check_model(base, atoms, bonds, what=sys.argv[1]))
except vocabulary.VocabularyMismatch as exc:
    print(f"REAL MISMATCH: {exc}"); raise SystemExit(1)

# does the property function work on a real molecule from this vocabulary
from rdkit import Chem
q = at.PROP_FNS[sys.argv[3]](Chem.MolFromSmiles("CCOc1ccc(CN2CCN(C)CC2)cc1"))
print(f"property {sys.argv[3]} on a probe molecule: {q:.4f}")

# THE GRADIENT GATE
try:
    base = base.to("cuda").eval()
    ad = AdaLNAdapter.for_base(base, cond_dim=1, hidden=256, interior_ff=True).to("cuda")
    before = {k: v.detach().clone() for k, v in ad.state_dict().items() if v.dtype.is_floating_point}
    opt = torch.optim.AdamW(ad.parameters(), lr=1e-3)
    mod = ad(torch.randn(4, 1, device="cuda"), t=torch.rand(4, 1, device="cuda"))
    loss = sum(v.square().mean() for d in mod.layers for v in d.values())
    loss.backward()
    gnorm = sum(float(p.grad.norm()) for p in ad.parameters() if p.grad is not None)
    opt.step()
    after = ad.state_dict()
    drift = sum(float((after[k] - v).abs().sum()) for k, v in before.items())
except Exception as exc:
    print(f"GRADIENT CHECK CRASHED ({type(exc).__name__}: {exc})"); raise SystemExit(3)
print(f"gradient check: grad_norm={gnorm:.4g}  weight_drift={drift:.4g}")
if not (gnorm > 0 and drift > 1e-6):
    print("ADAPTER RECEIVES NO GRADIENT -- the synced core does not thread the")
    print("modulation through. Training would produce a zero-init no-op that looks")
    print("trained. Refusing.")
    raise SystemExit(4)
print("atoms:", atoms, "| bonds:", bonds)
PY
rc=$?
[ $rc -eq 1 ] && { echo "ERROR: base and vocabulary disagree -- refusing"; exit 1; }
[ $rc -eq 4 ] && { echo "ERROR: adapter gets no gradient -- refusing"; exit 1; }
[ $rc -eq 5 ] && { echo "ERROR: property not registered in PROP_FNS -- refusing"; exit 1; }
[ $rc -ne 0 ] && { echo "ERROR: preflight could not run (exit $rc)"; exit 1; }

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $PY -u experiments/adapter_training__zinc.py \
        --VOCABULARY "'${VOCAB}'" \
        --PROPERTY "'${PROPERTY}'" \
        --PROPERTY_FROM "'${PROPERTY_FROM}'" \
        --BASE_CKPT "'${BASE}'" \
        --LEARNING_RATE ${LRS[$i]} \
        --H_HIDDEN ${HIDDEN[$i]} \
        --INTERIOR_FF True \
        --INTERIOR_ATTN False \
        --TARGET_PERCENTILES "[5,50,95]" \
        --LEVEL_NAMES "['low','mid','high']" \
        --MAX_TIME_HOURS ${MAX_HOURS} \
        --__DEBUG__ False \
        > "zinc_qed_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (lr=${LRS[$i]} hidden=${HIDDEN[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "Saved adapter" "zinc_qed_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that saved an adapter: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_qed_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== QED adapter arms: MAE by target level (lower better) ==="
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]}  lr=${LRS[$i]} hidden=${HIDDEN[$i]} ---"
    grep -E "adapter: [0-9,]+ params|^(low|mid|high) |MAE|validity" \
        "zinc_qed_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -12
done

echo
echo "HOW TO PICK"
echo "  Mean MAE over low/mid/high, and reject any arm with validity < 0.95."
echo "  QED is bounded [0,1] with its mass mid-range, so check the LOW and HIGH"
echo "  rows specifically -- an arm that regresses to the middle scores well on"
echo "  average while doing nothing that conditioning is for."
echo "  Scale: FreeGress unconditional 0.15, FreeGress best 0.04, DiGress 0.14."
echo "  A mid-range-only win is not worth shipping."
echo
echo "NEXT: the winner plus clogp@1.1.0 both need the FreeGress evaluation mode"
echo "(100 real-molecule targets from the TEST split, 10 samples each, MAE +"
echo "validity) before either can enter Table 2. That harness does not exist yet."
