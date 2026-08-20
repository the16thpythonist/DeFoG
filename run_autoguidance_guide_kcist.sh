#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=ag_guide
#SBATCH --output=ag_guide_%j.out

# Stage 1 of the autoguidance test: TRAIN THE DEGRADED GUIDE.
#
# Autoguidance (Karras et al.) replaces CFG's unconditional negative branch with a
# deliberately WORSE version of the conditional model, so the flaws the two share
# cancel and only the quality difference is amplified. MolGuidance (arXiv 2512.12198)
# reports it improving structural validity where CFG costs it -- which is the reason
# to try it here: our CFG validity falls 0.982 / 0.898 / 0.466 as w goes 2 / 3 / 4,
# and the question is whether autoguidance buys headroom to push w further.
#
# WHY A TRAINING RUN AT ALL. The plan said "the checkpoints already exist". They do
# not: CKPT_EVERY_K defaults to 0 in adapter_training__zinc.py, so no intermediate
# adapter was ever written, here or on any cluster. This job creates them.
#
# WHAT MAKES THIS A VALID GUIDE
#   * SAME BASE. ckpts/zinc_kek_base is the exact model molsmith/zinc-kek serves --
#     verified by base token -94.15126384728774, which also matches the shipped
#     molsmith/clogp@1.2.0 adapter. Note ckpts/zinc_e1_seed42_kek.ckpt is a DIFFERENT
#     base (-94.09057337861941); grabbing it by mistake would produce a guide that
#     shares none of this base's flaws and a result that means nothing.
#   * SAME RECIPE, STOPPED EARLY. Same property, vocabulary, width, LR *and label
#     convention* as the shipped adapter -- undertrained, not different. A guide that
#     differs in architecture or data is degraded along the wrong axis.
#
#     PROPERTY_FROM IS PART OF THAT RECIPE AND IS NOT THE MODULE DEFAULT. The default
#     is "source" (adapter_training__zinc.py:113); clogp@1.2.0 was trained with
#     "decoded" (run_zinc_clogp_adapter_v11_jupiter.sh:54). The two label conventions
#     differ by up to 1.6 log units at the low end and give cond_mean/cond_std of
#     2.458506 / 1.431754 versus 2.825129 / 1.158113 -- about 3700x the 1e-4 tolerance
#     the checks below use. Omitting this flag trains a guide that reads every target
#     as a different value, and the mismatch is only caught after the training spend.
#   * SAME CONDITIONING SCALE. The adapter normalises the target with its own
#     cond_mean/cond_std buffers; the shipped clogp@1.2.0 has 2.825129 / 1.158113.
#     If this run produces different ones, the guide reads the same target as a
#     different value and autoguidance is meaningless. e2_targeting.py REFUSES on
#     that mismatch, and the check below catches it here instead of at eval time.
#
# THE STEERING EVAL IS CUT TO A STUB. `MAX_TIME_HOURS` bounds `trainer.fit` and
# nothing else; the experiment then runs an UNCAPPED end-of-run steering eval --
# N_BASELINE + 2 levels x |GUIDANCE_WEIGHTS| x N_PER_TARGET generations at
# EVAL_STEPS=500, i.e. 256 + 2*5*128 = 1536 x 500 steps on the defaults. On a 3 h wall
# with fit allowed 2.5 h that overruns, and the overrun kills the VERIFICATION BLOCK
# below -- the one thing that would catch a bad guide. It is also pure waste: this
# adapter is deliberately bad, so measuring how well it steers answers nothing. Cut to
# a stub, with the wall raised to 4 h for margin.
#
# THREE GUIDES FOR THE PRICE OF ONE RUN. CKPT_EVERY_K=1 with EPOCHS=3 writes
# ep1/ep2/ep3, so how-degraded becomes a knob at evaluation time rather than a
# training decision made blind.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
BASE="ckpts/zinc_kek_base"
PROPERTY="clogp"
VOCAB="e1_kekulized"
PROPERTY_FROM="decoded"      # NOT the module default; see above
EPOCHS=3
HIDDEN=256
LR=2e-4
# Base token of molsmith/zinc-kek, hard-coded so a swapped checkpoint is caught here
# rather than three hours later.
WANT_TOKEN="-94.15126384728774"

echo "autoguidance guide training @ $(date) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

if [ ! -f "${BASE}.ckpt" ]; then echo "ERROR: ${BASE}.ckpt missing"; exit 1; fi

# ---- preflight: right base, and the vocabulary agrees with it -----------------
$PY - <<PY || { echo "ERROR: base preflight failed -- refusing to train"; exit 1; }
import sys
sys.path.insert(0, ".")
from defog.core import DeFoGModel
from defog.core.adapter import _base_token
b = DeFoGModel.load("${BASE}", device="cpu")
tok = _base_token(b)
want = float("${WANT_TOKEN}")
print(f"base ${BASE}: token={tok!r} n_layers={len(b.model.tf_layers)} cond_dim={b.cond_dim}")
if b.cond_dim != 0:
    print("FAIL: base is not unconditional"); sys.exit(1)
if abs(tok - want) > 1e-6 * (1 + abs(want)):
    print(f"FAIL: base token {tok} != {want} (molsmith/zinc-kek). Wrong checkpoint.")
    sys.exit(1)
print("base preflight OK")
PY

mkdir -p experiments/results/adapter_training__zinc

$PY -u experiments/adapter_training__zinc.py \
    --VOCABULARY "'${VOCAB}'" \
    --PROPERTY "'${PROPERTY}'" \
    --BASE_CKPT "'${BASE}'" \
    --PROPERTY_FROM "'${PROPERTY_FROM}'" \
    --EPOCHS ${EPOCHS} \
    --CKPT_EVERY_K 1 \
    --H_HIDDEN ${HIDDEN} \
    --LEARNING_RATE ${LR} \
    --MAX_TIME_HOURS 2.0 \
    --GUIDANCE_WEIGHTS "[2.0]" \
    --N_PER_TARGET 8 \
    --N_BASELINE 8 \
    --PROBE_EVERY_K 0 \
    --__DEBUG__ False

rc=$?
echo "training exited ${rc} at $(date)"
[ $rc -eq 0 ] || exit $rc

# ---- report what was written, and check the conditioning scale ---------------
RESULT_DIR=$(ls -td experiments/results/adapter_training__zinc/*/ 2>/dev/null | head -1)
echo "result dir: ${RESULT_DIR:-<none>}"
ls -la "${RESULT_DIR}" 2>/dev/null | grep -i "_ep[0-9]*\.ckpt" || echo "WARNING: no ep checkpoints found"

$PY - "$RESULT_DIR" <<'PY'
import glob, os, sys, torch
sys.path.insert(0, ".")
d = sys.argv[1]
# The shipped adapter's normalisation. A guide that disagrees is conditioned on a
# different value than the adapter it is supposed to negate, and every log line would
# still print the same target.
WANT = (2.825129270553589, 1.1581127643585205)
found = sorted(glob.glob(os.path.join(d, "*_adapter_ep*.ckpt")))
if not found:
    print("FAIL: no epoch checkpoints written -- CKPT_EVERY_K did not take effect")
    sys.exit(1)
bad = 0
for f in found:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    sd, cfg = ck["state_dict"], ck["config"]
    m, s = float(sd["cond_mean"][0]), float(sd["cond_std"][0])
    ok = abs(m - WANT[0]) < 1e-4 and abs(s - WANT[1]) < 1e-4
    bad += not ok
    print(f"{os.path.basename(f):32s} hidden={cfg['hidden']} n_layers={cfg['n_layers']} "
          f"cond_mean={m:.6f} cond_std={s:.6f} token={cfg.get('base_token')} "
          f"{'OK' if ok else 'MISMATCH vs clogp@1.2.0 ' + str(WANT)}")
if bad:
    print(f"\nFAIL: {bad} checkpoint(s) disagree with the shipped adapter's conditioning "
          f"scale. e2_targeting.py will refuse these. Check PROPERTY/VOCABULARY.")
    sys.exit(1)
print(f"\nOK: {len(found)} guide checkpoints, conditioning scale matches clogp@1.2.0")
print("Next: set GUIDE= one of these in run_autoguidance_pilot_kcist.sh")
PY
