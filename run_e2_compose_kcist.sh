#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=e2_comp
#SBATCH --output=e2_comp_%j.out

# E2 JOINT targeting: two properties at once, product-of-experts composition, with FK
# and composed conditional size sampling on top.
#
# COMPARABLE TO FreeGress's multi-property row. Their appendix reports per-property MAEs
# and a "Total MAE" which is the unweighted MEAN of the two -- verified against their
# DiGress rows, where (0.83+0.14)/2 = 0.49 and (0.55+0.14)/2 = 0.35 reproduce the printed
# Totals exactly. Their best JOINT row is logP ~0.18 / QED ~0.06 -> Total 0.12 at 80.7%
# validity, and note both components are WORSE than their single-property bests of
# 0.16 / 0.04: joint targeting costs them accuracy on each. Compare against the joint row.
#
# THREE THINGS DIFFER FROM THE SINGLE-PROPERTY RUNS AT ONCE -- composition, FK, and a
# composed size draw. That is deliberate (the question is what the stack achieves
# jointly), but it means a joint number cannot be attributed to any one of them. The
# single-property FK runs are the reference point for how much FK alone was worth.
#
# WEIGHTS are each property's own single-property optimum, in product mode. The
# composition docstring warns that product mode lets the effective unconditional
# coefficient grow with the number of branches, so this may over-steer; "mean" mode
# exists for that reason and is the obvious follow-up if validity drops.
#
# logP+TPSA is the second pair on purpose: logP+QED pairs a well-steered property with a
# poorly-steered one, while logP and TPSA are both steered well AND are chemically
# correlated -- so the two pairs probe different failure modes of composition.

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
NT=100
R=experiments/results/adapter_training__zinc
A_LOGP=$R/28_08_2026__15_50__Rb1X/clogp_adapter.ckpt
A_QED=$R/28_08_2026__15_50__p4LJ/qed_adapter.ckpt
A_TPSA=$R/28_08_2026__15_50__oIoj/tpsa_adapter.ckpt
H=ckpts/heads

OUT="e2_comp_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
echo "E2 joint composition @ $(date) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for f in "$A_LOGP" "$A_QED" "$A_TPSA" \
         $H/logp_head.ckpt $H/qed_head.ckpt $H/tpsa_head.ckpt \
         $H/logp_head_size.ckpt $H/qed_head_size.ckpt $H/tpsa_head_size.ckpt; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
echo "all adapters, heads and size models present"

declare -a PIDS
FAILED=0

# pair 0: logP + QED   (the FreeGress row)
( CUDA_VISIBLE_DEVICES=0 $PY -u scripts/e2_compose.py \
    --properties logp,qed \
    --adapter-ckpts "$A_LOGP,$A_QED" \
    --head-ckpts "$H/logp_head.ckpt,$H/qed_head.ckpt" \
    --size-models "$H/logp_head_size.ckpt,$H/qed_head_size.ckpt" \
    --weights 1.5,2.0 --composite-mode product \
    --split validation --method fk --n-targets $NT --per-target 10 \
    --steps 500 --eta 25 --blend-space prob \
    --fk-beta 2.5 --fk-warmup 0.6 --fk-ess 0.5 \
    --seed 42 --out "$OUT/logp_qed.json"
) > "e2comp_logp_qed_${SLURM_JOB_ID:-local}.out" 2>&1 &
PIDS[0]=$!; echo "launched logp+qed on GPU 0"

# pair 1: logP + TPSA  (both well-steered, and correlated)
( CUDA_VISIBLE_DEVICES=1 $PY -u scripts/e2_compose.py \
    --properties logp,tpsa \
    --adapter-ckpts "$A_LOGP,$A_TPSA" \
    --head-ckpts "$H/logp_head.ckpt,$H/tpsa_head.ckpt" \
    --size-models "$H/logp_head_size.ckpt,$H/tpsa_head_size.ckpt" \
    --weights 1.5,1.5 --composite-mode product \
    --split validation --method fk --n-targets $NT --per-target 10 \
    --steps 500 --eta 25 --blend-space prob \
    --fk-beta 2.5 --fk-warmup 0.6 --fk-ess 0.5 \
    --seed 42 --out "$OUT/logp_tpsa.json"
) > "e2comp_logp_tpsa_${SLURM_JOB_ID:-local}.out" 2>&1 &
PIDS[1]=$!; echo "launched logp+tpsa on GPU 1"

for i in 0 1; do
    if wait "${PIDS[$i]}"; then echo "  ok   pair $i"
    else echo "  FAIL pair $i (exit $?)"; FAILED=1; fi
done
echo "finished at $(date)"

echo
echo "=== E2 joint targeting ==="
$PY - "$OUT" <<'PY'
import json, os, sys
OUT = sys.argv[1]
# single-property references at the same w, WITH FK, from the per-property runs
SOLO = {"logp": 0.2988, "qed": 0.0666, "tpsa": 4.5598}
print(f"{'pair':12s}{'property':9s}{'joint':>9s}{'solo+FK':>10s}{'cost':>9s}"
      f"{'/std':>8s}   Total   valid   uniq")
for name in ("logp_qed", "logp_tpsa"):
    f = os.path.join(OUT, name + ".json")
    if not os.path.exists(f):
        print(f"{name:12s} MISSING -- see e2comp_{name}_*.out"); continue
    d = json.load(open(f))
    for p in d["properties"]:
        m = d["mae_per_property"][p]; s = d["target_std"][p]
        print(f"{name:12s}{p:9s}{m:>9.4f}{SOLO[p]:>10.4f}{m-SOLO[p]:>+9.4f}"
              f"{m/s:>8.3f}", end="")
        print(f"   {d['mae_total']:.4f}   {d['validity']:.3f}   {d['uniqueness']:.3f}"
              if p == d["properties"][-1] else "")
print()
print("FreeGress joint reference: logP 0.18 / QED 0.06 -> Total 0.12 at 80.7% validity.")
print("'cost' is what each property gives up relative to steering it ALONE with FK --")
print("the quantity FreeGress also pays (0.16 -> 0.18 logP, 0.04 -> 0.06 QED).")
print("Read uniqueness: FK resampling under two competing energies can collapse harder")
print("than under one, and a joint MAE bought with duplicates is not joint steering.")
PY

exit $FAILED
