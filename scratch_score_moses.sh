#!/bin/bash
# Score the frozen MOSES test pass -- the full official suite against BOTH
# held-out sets. --reference-scaffolds is not optional: DeFoG reports the TestSF
# columns, and Scaf/Test vs Scaf/TestSF differ by ~6x, so quoting the wrong one
# is not a rounding error.
B=/media/ssd2/Programming/DeFoG
D="$B/final_moses_1318337"
for f in "$D/seed42.smi" "$D/_test_reference.smi" "$D/_test_scaffolds_reference.smi"; do
    [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
echo "generated  $(wc -l < "$D/seed42.smi")"
echo "test ref   $(wc -l < "$D/_test_reference.smi")"
echo "testSF ref $(wc -l < "$D/_test_scaffolds_reference.smi")"
"$B/.venv_metrics/bin/python" "$B/scripts/e1_metrics.py" \
    --generated "$D/seed42.smi" \
    --reference "$D/_test_reference.smi" \
    --reference-scaffolds "$D/_test_scaffolds_reference.smi" \
    --dataset moses \
    --out "$D/moses_test_metrics.json"
rc=$?
[ $rc -ne 0 ] && { echo "SCORING FAILED (exit $rc)"; exit $rc; }
echo "OK -> $D/moses_test_metrics.json"
