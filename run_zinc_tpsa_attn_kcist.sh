#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=tpsa_attn
#SBATCH --output=zinc_tpsa_attn_%j.out

# TPSA adapter for the zinc-kek base, conditioning at the ATTENTION site.
#
# WHY IT EXISTS: molsmith/tpsa@1.0.0 binds the OLD aromatic zinc-base and cannot be
# served on zinc-kek -- the atom ORDER differs (C N O S F Cl Br I P against
# C N O F P S Cl Br I), not just the bond set, so molsmith's schema_hash gate refuses
# the cross-serve and would be right to. This restores TPSA to the shipped set on the
# current base.
#
# ARCHITECTURE: INTERIOR_ATTN=True, INTERIOR_FF=False (~4.52M params), by decision.
# The QED A/B put ATTN ahead of FF on the 4-arm mean (0.1396 vs 0.1500) though only
# -2.4% best-arm, so this is the better bet rather than a settled result. Grid is the
# full 2 LR x 2 hidden, which ATTN-only affords because the site is fixed.
#
#     arm0  lr 2e-4  hidden 256      arm2  lr 2e-4  hidden 512
#     arm1  lr 4e-4  hidden 256      arm3  lr 4e-4  hidden 512
#
# WHAT TPSA MEASURES DIFFERENTLY FROM logP AND QED (measured on 8000 train molecules):
#   * DECODE GAP IS NEGLIGIBLE: signed bias -0.85 on a std of 23.3, i.e. 0.036 of a
#     std, against logP's 0.263 and QED's 0.107. The charge-stripping that wrecks
#     logP's low end barely touches TPSA, so property_from matters little here --
#     `decoded` is still used, to match the head and size model.
#   * SIZE HEADROOM IS THE LARGEST OF THE THREE: E[n | TPSA decile] runs 18.8 -> 26.5
#     heavy atoms, a 1.70 sigma swing against logP's 1.50 and QED's 1.07.
#
# THAT SECOND POINT IS A WARNING, NOT JUST AN OPPORTUNITY. TPSA is a sum over polar
# atoms, so it is close to mechanically tied to molecule size. An adapter can post a
# good MAE by getting the ATOM COUNT right while conditioning nothing about
# structure -- and the conditional size draw, which is a separate mechanism, would
# then be doing the work. Read the eval with that in mind, and note that the honest
# ablation (adapter with marginal sizes vs adapter with P(n|TPSA)) is a later E2 run,
# not something this training job can answer.
#
# SCALE FOR READING THE MAE. Decoded TPSA: mean 63.6, std 23.3, p5 26.3, p50 63.1,
# p95 103.9. A do-nothing predictor that always emits the mean scores about 26.0 mean
# MAE over the 5/50/95 targets (37.3 / 0.5 / 40.3). So an adapter at ~8 is roughly
# +69% skill; judge against 26.0, never against zero.
# The OLD aromatic TPSA adapter reached MAE 6.75 (low) / 10.91 (high) on its own base
# and protocol -- indicative only, since base, vocabulary and targets all differ.
#
# The head and the conditional size model are trained CONCURRENTLY by
# run_zinc_tpsa_head_kcist.sh on a separate node; neither is needed here.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Thread contention: each concurrent arm otherwise spawns ~47 threads (torch intra-op
# defaults), so 4 arms oversubscribe the 16 allocated CPUs ~2.6x -- measured on job
# 43126 as load 41 against an allocation of 16. The dataset pass is RDKit plus tiny
# one-hot ops, where intra-op threading buys nothing and the contention is pure loss:
# the single-process head job encoded the SAME 219,568 molecules in 407s against
# ~2h50m here. Give each arm its own quarter of the allocation.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

BASE="ckpts/zinc_rl2_seed42/best_model"
VOCAB="e1_kekulized"
PROPERTY="tpsa"
PROPERTY_FROM="decoded"
PY=.venv/bin/python
MAX_HOURS=8.0

#        arm0    arm1    arm2    arm3
LRS=(    2e-4    4e-4    2e-4    4e-4 )
HIDDEN=( 256     256     512     512  )
TAGS=(   lr2h256 lr4h256 lr2h512 lr4h512 )

[ -f "${BASE}.ckpt" ] || { echo "ERROR: ${BASE}.ckpt missing"; exit 1; }

echo "ZINC TPSA adapter, ATTENTION site (zinc-kek) @ $(date) on $(hostname)"
echo "  base=${BASE}  vocab=${VOCAB}  property=${PROPERTY} from=${PROPERTY_FROM}"
echo "  INTERIOR_FF=False  INTERIOR_ATTN=True   grid = 2 LR x 2 hidden"
echo "  md5(base)=$(md5sum ${BASE}.ckpt | cut -d' ' -f1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False on a GPU node"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary guard + a gradient gate built on the L10/attention path THIS job trains.
# L4 and L10 are threaded separately, so a gate built on the wrong flag would pass an
# adapter that never receives gradient -- and a zero-init adapter is an exact no-op,
# indistinguishable from a trained-but-useless one in the loss curve.
$PY - "$BASE" "$VOCAB" "$PROPERTY" <<'PY'
import sys
try:
    import torch, importlib.util
    from defog.core import DeFoGModel, AdaLNAdapter
    from defog.data import vocabulary
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

from rdkit import Chem
q = at.PROP_FNS[sys.argv[3]](Chem.MolFromSmiles("CCOc1ccc(CN2CCN(C)CC2)cc1"))
print(f"property {sys.argv[3]} on a probe molecule: {q:.4f}")

try:
    base = base.to("cuda").eval()
    ad = AdaLNAdapter.for_base(base, cond_dim=1, hidden=256,
                               interior_ff=False, interior_attn=True).to("cuda")
    n_par = sum(p.numel() for p in ad.parameters())
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
print(f"attn-only adapter: {n_par:,} params")
print(f"gradient check: grad_norm={gnorm:.4g}  weight_drift={drift:.4g}")
if not (gnorm > 0 and drift > 1e-6):
    print("ADAPTER RECEIVES NO GRADIENT on the L10/attention path. Refusing.")
    raise SystemExit(4)
l10 = {k: v for k, v in before.items() if "attn" in k.lower() or "e_mul" in k.lower()}
if l10:
    d10 = sum(float((after[k] - v).abs().sum()) for k, v in l10.items())
    print(f"L10/attention heads drift: {d10:.4g} over {len(l10)} tensors")
    if d10 <= 1e-9:
        print("L10 HEADS DID NOT MOVE -- interior_attn is not wired. Refusing.")
        raise SystemExit(4)
print("atoms:", atoms, "| bonds:", bonds)
PY
rc=$?
[ $rc -eq 1 ] && { echo "ERROR: base and vocabulary disagree -- refusing"; exit 1; }
[ $rc -eq 4 ] && { echo "ERROR: adapter gets no gradient on the attention path -- refusing"; exit 1; }
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
        --INTERIOR_FF False \
        --INTERIOR_ATTN True \
        --TARGET_PERCENTILES "[5,50,95]" \
        --LEVEL_NAMES "['low','mid','high']" \
        --MAX_TIME_HOURS ${MAX_HOURS} \
        --__DEBUG__ False \
        > "zinc_tpsa_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (lr=${LRS[$i]} hidden=${HIDDEN[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "Saved adapter" "zinc_tpsa_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that saved an adapter: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_tpsa_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== TPSA adapter arms: MAE by target level (lower better) ==="
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]}  lr=${LRS[$i]} hidden=${HIDDEN[$i]} ---"
    grep -E "adapter: [0-9,]+ params|^(low|mid|high) |MAE|validity|baseline" \
        "zinc_tpsa_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -12
done

echo
echo "HOW TO PICK"
echo "  Mean MAE over low/mid/high; reject any arm with validity < 0.95."
echo "  JUDGE AGAINST 26.0, the do-nothing score at these targets (37.3/0.5/40.3)."
echo "  As with QED, the MID level is unwinnable -- the target sits on the"
echo "  unconditional mean, where a constant predictor beats any sampler on spread"
echo "  alone. Read LOW and HIGH; the mean is dragged by a level nobody can win."
echo "  Watch the ACHIEVED means, not only MAE: TPSA is a sum over polar atoms and"
echo "  is 1.70 sigma tied to graph size, so an arm can score well by getting the"
echo "  atom count right. The size ablation that separates those is a later E2 run."
echo
echo "NEXT: package the winner as molsmith/tpsa@2.0.0 on zinc-kek WITH the head from"
echo "run_zinc_tpsa_head_kcist.sh (check head.present, not the metadata block), then"
echo "the E2 arms -- and note FK beta for TPSA is ~0.0064, four orders of magnitude"
echo "below QED's ~1000, because beta multiplies a SQUARED error in property units."
