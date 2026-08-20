#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=tpsa_head
#SBATCH --output=zinc_tpsa_head_%j.out

# TPSA property head AND conditional size model, in ONE dataset pass.
#
# Runs CONCURRENTLY with the adapter grid (run_zinc_tpsa_attn_kcist.sh) on a separate
# node. Neither needs the other: a head is fit on (graph, label) pairs with no adapter
# in the loop, which is why it can attach to an adapter trained later.
#
# --with-size-model is the reason this is one job rather than two. Both the head and
# the LearnedSizeDistribution need the same encode -> decode -> measure pass over
# 219,568 molecules, which dominates the runtime; fitting them separately pays it
# twice. It writes ckpts/heads/tpsa_head.ckpt and ckpts/heads/tpsa_head_size.ckpt.
#
# WHY BOTH ARE WANTED HERE
#   head        Feynman-Kac needs a learned energy: LearnedPropertyEnergy scores each
#               predicted-clean particle by squared error to the target. Without it
#               --method fk refuses (correctly -- it would silently degrade to plain
#               adapter sampling).
#   size model  P(n | TPSA) instead of the dataset marginal. TPSA has the LARGEST size
#               headroom of the three properties measured: E[n | decile] runs
#               18.8 -> 26.5 heavy atoms, a 1.70 sigma swing (logP 1.50, QED 1.07).
#               On QED the conditional draw was worth 10-15% MAE on its own.
#
# --property-from decoded MATCHES THE ADAPTER. A head or size model fit on source
# labels, paired with an adapter trained on decoded ones, disagrees about what a
# target means. For TPSA the gap is small (bias -0.85 on a std of 23.3, i.e. 0.036 of
# a std, against logP's 0.263) so this matters less here than elsewhere -- but
# consistency costs nothing and the mismatch is invisible once it is baked in.
#
# TWO NUMBERS TO CHECK IN THE OUTPUT
#   head MAE/std   the head is the FK energy; at MAE >= std it is no better than
#                  predicting the mean and FK becomes a silent no-op. Compare against
#                  TPSA's OWN std (23.3), not against another property's -- TPSA's
#                  absolute MAE will look huge next to QED's purely from units.
#   gain_nats      the size model must beat the marginal by >0.02 nats or the learned
#                  draw is a moving part that buys nothing. QED reached +0.1943; given
#                  TPSA's larger headroom this should comfortably clear it, and if it
#                  does NOT then the decile measurement and the fit disagree and
#                  something is wrong.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BASE="ckpts/zinc_rl2_seed42/best_model"
VOCAB="e1_kekulized"
PY=.venv/bin/python
mkdir -p ckpts/heads

[ -f "${BASE}.ckpt" ] || { echo "ERROR: ${BASE}.ckpt missing"; exit 1; }

echo "TPSA head + size model (zinc-kek) @ $(date) on $(hostname)"
echo "  base=${BASE}  vocab=${VOCAB}  property_from=decoded  seed=0"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False on a GPU node"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK:", torch.cuda.device_count(), "device(s)")
PY

# The flags this job depends on are recent; verify the CLUSTER's copy has them rather
# than trusting a local file (a stale trainer here cost 2h39m of dead job time once).
$PY scripts/train_property_head.py --help 2>&1 | grep -q -- "--with-size-model" \
    || { echo "ERROR: cluster's train_property_head.py has no --with-size-model"; exit 1; }
echo "trainer supports --with-size-model"

$PY -u scripts/train_property_head.py \
    --base "$BASE" \
    --vocabulary "$VOCAB" \
    --property tpsa \
    --property-from decoded \
    --hidden 128 --layers 3 \
    --epochs 60 --lr 1e-3 --batch-size 32 \
    --with-size-model --size-hidden 512 --size-layers 2 --size-epochs 200 \
    --seed 0 --holdout 5000 \
    --out ckpts/heads/tpsa_head

echo "finished at $(date)"

OK=0
for f in ckpts/heads/tpsa_head.ckpt ckpts/heads/tpsa_head_size.ckpt; do
    [ -f "$f" ] && OK=$((OK+1)) || echo "MISSING: $f"
done
echo "artefacts written: ${OK} / 2"
[ "$OK" -lt 2 ] && exit 1

echo
echo "=== quality gates ==="
$PY - <<'PY'
import torch
from defog.core import LearnedSizeDistribution
sm = LearnedSizeDistribution.load("ckpts/heads/tpsa_head_size.ckpt")
print(f"size model: grid {sm.min_size}..{sm.max_size}  property={sm.property_name!r} "
      f"from={sm.property_from!r}")
# The measured decile table says low TPSA wants ~18.8 atoms and high ~26.5. A size
# model that does not reproduce that is not usable, and the only way to find out is
# to ask it -- job 43067 shipped a 2x2 built on a size draw that never fired.
lo = sm.sample(512, condition=torch.full((512, 1), 26.3)).float().mean()
hi = sm.sample(512, condition=torch.full((512, 1), 103.9)).float().mean()
print(f"E[n | TPSA=26.3] = {lo:.2f}   E[n | TPSA=103.9] = {hi:.2f}   "
      f"(measured deciles: 18.8 -> 26.5)")
if hi - lo < 1.0:
    print("*** the size model barely responds to its condition -- do NOT use it ***")
else:
    print(f"size model responds: {hi - lo:.2f} atoms across the range")
PY

echo
echo "HOW TO READ THIS"
echo "  MAE/std is the number that matters for the head, not absolute MAE: TPSA's std"
echo "  is 23.3, so an MAE of 2 is excellent here and would be catastrophic for QED."
echo "  gain_nats < 0.02 means the size model buys nothing; QED reached +0.1943 and"
echo "  TPSA has MORE headroom (1.70 sigma vs 1.07), so expect at least that."
echo
echo "NEXT: package the winning adapter arm as molsmith/tpsa@2.0.0 with"
echo "--head ckpts/heads/tpsa_head.ckpt, then E2 arms with --size-mode learned"
echo "--size-model ckpts/heads/tpsa_head_size.ckpt and FK beta ~0.0064."
