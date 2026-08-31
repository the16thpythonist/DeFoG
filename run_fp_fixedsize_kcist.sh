#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=fp_fixsz
#SBATCH --output=fp_fixsz_%j.out

# Fingerprint steering with the generated graphs PINNED to the target's exact node count.
#
# NOT ORACLE LEAKAGE. In the intended use -- find analogs of a query molecule -- the query
# is in hand (it is where the fingerprint comes from), so its heavy-atom count is free
# information. This is a deployable setting, not an upper bound.
#
# BUT IT IS STILL EXTRA INFORMATION the free-size arm did not have, and Tanimoto rewards
# size agreement on its own: two molecules of the same size share more bits than two of
# different sizes even with no steering at all. So a lift here is "fingerprint steering
# PLUS size agreement" and CANNOT be attributed to the adapter. The control that would
# separate them is an unsteered arm at the same pinned size (w=0, adapter off); it was
# considered and deliberately not run. Do not report the delta as an adapter improvement.
#
# THE FREE-SIZE COMPARISON COMES FROM A DIFFERENT RUN (job 43583), on the same 22 targets
# under the same seed. Same targets means the comparison is paired per target, but not
# same-run, so it carries that run's sampling noise -- measured at ~0.013 mean Tanimoto
# on the six original targets when the identical w=1.0 configuration was re-evaluated.
# Treat differences below ~0.015 as unresolved.
#
# THREE WEIGHTS, not just the free-size optimum of 1.5: pinning the size removes work the
# adapter was previously spending guidance on, so the optimum plausibly moves. At ~0.9 min
# per target-weight the extra two points cost under an hour.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
CKPT="${CKPT:-experiments/results/adapter_fingerprint__zinc/29_08_2026__08_07__1y5w/fp_adapter.ckpt}"
WEIGHTS="${WEIGHTS:-[1.0,1.5,2.0]}"
[ -f "$CKPT" ] || { echo "ERROR: $CKPT missing"; exit 1; }
echo "fingerprint FIXED-SIZE run @ $(date) on $(hostname)"
echo "adapter: $CKPT   weights: $WEIGHTS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ---- preflight: the size pin actually pins, on THIS checkout's sampler ---------------
# Cheap (untrained no-op adapter, 20 steps, CPU-or-GPU) and it fails before any real GPU
# time if num_nodes stops being honoured -- a silent regression here would produce a full
# set of plausible "size-matched" numbers that were never size-matched.
$PY - <<'PY' || { echo "ERROR: size-pin preflight failed -- refusing to run"; exit 1; }
import re, sys, numpy as np, torch
sys.path.insert(0, ".")
from defog.core import DeFoGModel, AdaLNAdapter
base = DeFoGModel.load("ckpts/zinc_kek_base", device="cpu").eval()
a = AdaLNAdapter.for_base(base, cond_dim=1024, hidden=256,
        cond_encoder={"kind": "mlp", "in_dim": 1024, "out_dim": 512,
                      "hidden": 1024, "n_blocks": 2},
        xattn_tokens=64, xattn_dim=128, xattn_heads=16).eval()
src = open("experiments/adapter_fingerprint__zinc.py").read()
ns = {}
exec("import torch, numpy as np\nfrom defog.core import AdapterComposition, ConditionBranch\n"
     "from defog.core.sampler import AdaptedSampler\n", ns)
m = re.search(r"^def guided_sample.*?\n    return out\n", src, re.S | re.M)
if m is None:
    print("FAIL: could not find guided_sample in the experiment module"); sys.exit(1)
exec(m.group(0), ns)
torch.manual_seed(0)
fp = np.random.RandomState(0).poisson(0.2, 1024).astype("float32")
bad = 0
for k in (11, 23, int(base.max_nodes)):
    out = ns["guided_sample"](base, a, fp, 1.5, 4, 20, 5.0, 0.0, "polydec", 4,
                              torch.device("cpu"), num_nodes=k)
    sizes = {int(g.x.shape[0]) for g in out}
    ok = sizes == {k}
    bad += not ok
    print(f"pin={k}: node counts {sorted(sizes)} {'OK' if ok else 'MISMATCH'}")
if bad:
    print("FAIL: num_nodes is not being honoured"); sys.exit(1)
print(f"size-pin preflight OK (base max_nodes={base.max_nodes})")
PY

$PY -u experiments/adapter_fingerprint__zinc.py \
    --VOCABULARY "'e1_kekulized'" \
    --BASE_CKPT "'ckpts/zinc_kek_base'" \
    --FP_BITS 1024 --FP_RADIUS 2 --FP_COUNTS True \
    --COND_ENCODER "{'kind':'mlp','out_dim':512,'hidden':1024,'n_blocks':2}" \
    --XATTN_TOKENS 64 --XATTN_DIM 128 --XATTN_HEADS 16 \
    --LOAD_ADAPTER "'$CKPT'" \
    --FIXED_SIZE True \
    --GUIDANCE_WEIGHTS "$WEIGHTS" \
    --SEED 42 \
    --__DEBUG__ False
rc=$?
echo "exited $rc at $(date)"
[ $rc -eq 0 ] || exit $rc

echo
echo "REMINDER: size_hit should be 1.000 on every line. Anything else means the pin did"
echo "not take for that target and its Tanimoto is not a size-matched number."
exit 0
