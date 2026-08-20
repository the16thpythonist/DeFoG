#!/bin/bash
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=moses_score
#SBATCH --output=moses_score_%j.out

# Score the FROZEN MOSES test pass with the official molsets suite.
#
# WHY HERE AND NOT WHERE IT WAS GENERATED
# Sampling ran on JUPITER (aarch64, GPU). The metrics stack is x86-only and does
# not install there. It was then run on the local workstation, where it held
# ~22 GB of RAM for an hour alongside the user's other work -- too much. KCIST is
# x86 with a scheduler, so the memory belongs to a job allocation instead of
# competing with an interactive desktop.
#
# The samples were transferred, not regenerated: md5 12808c49b9d195cb00fc1c084dffd1cd
# on both sides. Scoring reads SMILES, so it is fully re-runnable without
# touching the 43 minutes of GPU time that produced them.
#
# WHAT IS BEING SCORED
#   generated  final_moses_1318337/seed42.smi        29,954 valid of 30,000
#   test       _test_reference.smi                  176,074
#   testSF     _test_scaffolds_reference.smi        176,225
#
# BOTH references are passed, and that is not optional. MOSES reports every
# metric against test AND test_scaffolds, and they are far apart -- calibrated
# against DeFoG's own published samples, Scaf/Test reads 0.868 where the paper
# reports 0.144, while Scaf/TestSF reads 0.107. An E1 row must quote the TestSF
# columns; quoting Test would look like a 6x improvement that does not exist.
#
# Reference size is load-bearing too: MOSES metrics are strongly reference-size
# dependent and published numbers use the full split, so no --limit here.
#
# MEMORY: the local run peaked ~22 GB (SNN is 30,000 x 176,074 nearest-neighbour
# similarities, done twice). 64 GB requested to leave headroom rather than
# discover the ceiling in a six-hour job.

set -u
cd "${SLURM_SUBMIT_DIR:-/home/tm4030/Programming/DeFoG}"

D=final_moses_1318337
PY=.venv_metrics/bin/python

for f in "$D/seed42.smi" "$D/_test_reference.smi" "$D/_test_scaffolds_reference.smi"; do
    [ -f "$f" ] || { echo "ERROR missing: $f"; exit 1; }
done
[ -x "$PY" ] || { echo "ERROR: $PY not executable"; exit 1; }

echo "MOSES frozen-test scoring @ $(date) on $(hostname)"
echo "  generated  $(wc -l < "$D/seed42.smi")"
echo "  test       $(wc -l < "$D/_test_reference.smi")"
echo "  testSF     $(wc -l < "$D/_test_scaffolds_reference.smi")"
echo "  md5(gen)   $(md5sum "$D/seed42.smi" | cut -d' ' -f1)"

$PY scripts/e1_metrics.py \
    --generated "$D/seed42.smi" \
    --reference "$D/_test_reference.smi" \
    --reference-scaffolds "$D/_test_scaffolds_reference.smi" \
    --dataset moses \
    --out "$D/moses_test_metrics.json"
rc=$?

echo "exit=${rc} at $(date)"
if [ $rc -ne 0 ]; then
    echo "ERROR: scoring failed"
    exit $rc
fi

echo
echo "=== E1 MOSES row -- quote the TestSF columns ==="
$PY - "$D" <<'PY'
import json, sys
d = json.load(open(f"{sys.argv[1]}/moses_test_metrics.json"))
order = ["moses_validity", "moses_unique", "moses_filters",
         "moses_snn_test_scaffolds", "moses_frag_test_scaffolds",
         "moses_scaf_test_scaffolds", "moses_fcd_test_scaffolds",
         "moses_snn_test", "moses_frag_test", "moses_scaf_test", "moses_fcd_test"]
for k in order:
    if k in d:
        tag = "  <- TestSF (report this)" if "test_scaffolds" in k else ""
        print(f"  {k:32s} {d[k]:.4f}{tag}")
PY
