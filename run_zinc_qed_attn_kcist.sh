#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --job-name=qed_attn
#SBATCH --output=zinc_qed_attn_%j.out

# QED adapter, conditioning at the ATTENTION site instead of the FFN site.
#
# THE A/B. This is job 43027's grid with exactly one flag flipped:
#     43027 (shipped):  INTERIOR_FF=True   INTERIOR_ATTN=False
#     this job:         INTERIOR_FF=False  INTERIOR_ATTN=True
# Same base, same vocabulary, same labels, same 2 LR x 2 hidden grid, same
# 5/50/95 targets, same eta=5 eval. Winner-to-winner is therefore a clean read on
# the SITE, which is the only thing that differs.
#
# THIS IS NOT A CAPACITY INCREASE -- it is a site swap at slightly LOWER capacity.
# On the real 9-layer base: output-only 2.75M, +L4 (pre-FFN FiLM) 2.22M, +L10
# (edge->attention-logit FiLM) 1.78M. So FF-only is 4.97M and ATTN-only is 4.53M.
# If ATTN-only wins it wins on where the conditioning is injected, not on size.
#
# WHY THE ATTENTION SITE IS THE INTERESTING ONE FOR QED. The conditioning audit
# ranked L10 as the only lever that actually re-routes attention -- e_mul is
# per-pair and rides Y_ij, so it survives softmax-over-keys (a generic logit bias
# cancels and is a mathematical no-op). QED is a composite of structural terms
# (rings, alerts, rotatable bonds, aromaticity) far more than logP is, and logP is
# nearly an additive atom-contribution sum. If any property should care about
# re-routing attention rather than rescaling an FFN, it is this one.
#
# WHAT THIS RUN IS NOT EXPECTED TO FIX, stated up front so the result is read
# honestly. The measured gap between this adapter family and the QED adapter that
# is remembered as better is NOT architecture:
#     old aromatic QED, 5/95 targets:  +23% skill pre-RL,  +39% after 2 RL rounds
#     new zinc-kek QED, same measure:  +26% skill
# The new adapter already matches the old one pre-RL. The 0.130 number came after
# two rounds of head-reward RL. So the most likely outcome here is a small change
# either way, and the RL round is the lever with a measured -21% behind it.
#
# TWO HYPOTHESES ALREADY ELIMINATED, so nobody re-runs them:
#   * eta is NOT the problem (job 43036). Skill is flat across eta in {1,5,25}:
#     QED pooled -15/-13/-12%, logP pooled 50/50/50%. The E2 harness inheriting
#     eta=25 from the unconditional sweep cost nothing.
#   * The sanity-RL base did NOT compress the QED range. Unconditional QED sd
#     0.132/0.121/0.139 at eta 1/5/25 against a dataset 0.133 decoded, mean 0.745
#     against 0.748, p5 0.496 against 0.483. The base reproduces the dataset
#     distribution; the low targets are reachable and the adapter is not being
#     blamed for a base limitation.
#
# L10_LR_SCALE stays at its 0.3 default: the attention heads train at a smaller LR
# because re-routing attention is the change most likely to cost validity, and
# validity is the reject gate below.

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
PROPERTY="qed"
PROPERTY_FROM="decoded"      # label the molecule the GRAPH is, not the source SMILES
PY=.venv/bin/python
MAX_HOURS=8.0

#        arm0    arm1    arm2    arm3      -- identical to job 43027
LRS=(    2e-4    4e-4    2e-4    4e-4 )
HIDDEN=( 256     256     512     512  )
TAGS=(   lr2h256 lr4h256 lr2h512 lr4h512 )

[ -f "${BASE}.ckpt" ] || { echo "ERROR: ${BASE}.ckpt missing"; exit 1; }

echo "ZINC QED adapter, ATTENTION site (zinc-kek) @ $(date) on $(hostname)"
echo "  base=${BASE}  vocab=${VOCAB}  property=${PROPERTY} from=${PROPERTY_FROM}"
echo "  INTERIOR_FF=False  INTERIOR_ATTN=True  (43027 was the reverse)"
echo "  md5(base)=$(md5sum ${BASE}.ckpt | cut -d' ' -f1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False on a GPU node"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# Vocabulary guard + the gradient check, with the gate built on the SITE THIS JOB
# TRAINS. A zero-init adapter is an exact no-op, so "trained but never moved" and
# "trained and useless" are indistinguishable from the loss curve alone -- and the
# L10 path is threaded through NodeEdgeBlock.forward separately from L4, so a core
# that threads one and not the other would pass a gate built on the wrong flag.
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

# The L10 heads specifically must move -- the whole point of the job. If only the
# output-FiLM heads drift, this arm is just an output-only adapter wearing a flag.
l10 = {k: v for k, v in before.items() if "attn" in k.lower() or "e_mul" in k.lower()}
if l10:
    d10 = sum(float((after[k] - v).abs().sum()) for k, v in l10.items())
    print(f"L10/attention heads drift: {d10:.4g} over {len(l10)} tensors")
    if d10 <= 1e-9:
        print("L10 HEADS DID NOT MOVE -- interior_attn is not wired. Refusing.")
        raise SystemExit(4)
else:
    print("WARNING: no tensor name matched the attention heads; drift check is")
    print("         only the global one above. Verify naming before trusting a null result.")
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
        > "zinc_qedattn_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched ${TAGS[$i]} (lr=${LRS[$i]} hidden=${HIDDEN[$i]}) on GPU ${i} (pid $!)"
    sleep 5
done

wait
echo "finished at $(date)"

OK=0
for t in "${TAGS[@]}"; do
    grep -q "Saved adapter" "zinc_qedattn_${t}_${SLURM_JOB_ID}.out" 2>/dev/null && OK=$((OK+1))
done
echo "arms that saved an adapter: ${OK} / 4"
if [ "$OK" -lt 4 ]; then
    echo "ERROR: not all arms completed; tracebacks follow"
    grep -hA6 "Traceback" zinc_qedattn_*_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -vE "dbus|desktop_notifier|message_bus" | head -25
fi

echo
echo "=== QED attention-site arms: MAE by target level (lower better) ==="
for i in 0 1 2 3; do
    echo "--- ${TAGS[$i]}  lr=${LRS[$i]} hidden=${HIDDEN[$i]} ---"
    grep -E "adapter: [0-9,]+ params|^(low|mid|high) |MAE|validity" \
        "zinc_qedattn_${TAGS[$i]}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -12
done

echo
echo "=== THE COMPARISON THIS JOB EXISTS FOR (job 43027, FF site, same grid) ==="
echo "  winner lr4h256 @ w=1.0:  low 0.1908   mid 0.1037   high 0.1255   mean 0.1403"
echo "  achieved means:          low 0.628    mid 0.739    high 0.788"
echo "  targets:                 low 0.4786   mid 0.7782   high 0.9109"
echo "  unconditional mean 0.7418, so do-nothing scores low 0.259 / mid 0.038 /"
echo "  high 0.169 -- i.e. FF-site skill was +26% / -174% / +25%."
echo
echo "HOW TO READ IT"
echo "  Compare the LOW and HIGH rows, not the mean. The mid target sits on the"
echo "  unconditional mean, where a constant predictor is unbeatable and every"
echo "  generative sampler loses on its own spread alone; averaging it in is what"
echo "  made the FF-site adapter look broken when it was not."
echo "  Reject any arm with validity < 0.95. The attention site is the one most"
echo "  likely to buy steering with validity, and a tighter MAE at 0.90 validity"
echo "  is not a better adapter."
echo "  A null result here is INFORMATIVE: it says the site is not the constraint,"
echo "  and points at the RL round (-21% measured on the old adapter) instead."
