#!/usr/bin/env python
"""Head-to-head for the blend-space sweep.

All four arms use the same 100 targets, drawn with the same seed, so the comparison
should be PAIRED: for each target, rate-space MAE vs prob-space MAE on that same target.
A paired test removes target difficulty, which is the dominant source of variance here --
comparing pooled means throws that away and needs a much bigger effect to see anything.

Reports, per weight:
  mean paired difference, a Wilcoxon signed-rank test, and a plain sign count.
And across weights: whether prob-space changes the w=1 vs w=2 ordering, which is the
question behind the whole exercise (our sweeps always pick w=1; FreeGress gains 27% from
s>1, and a miscalibrated guidance direction is the suspected reason we cannot).
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "blend_results")


def load(space, w):
    p = RESULTS / f"e2_{space}_w{w}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def paired(da, db):
    """Align two arms row by row, and VERIFY the alignment rather than trust it.

    Every arm calls draw_targets with the same split and seed, so row i is the same
    target in both. Keying on target_smiles instead looks safer but is not: duplicate
    SMILES silently collapse in a dict and drop targets from the comparison, which is how
    a 100-target pairing quietly becomes a 9-target one. Index-pairing plus an assertion
    on the target values catches misalignment loudly instead.
    """
    ra, rb = da["per_target"], db["per_target"]
    if len(ra) != len(rb):
        raise SystemExit(f"arm length mismatch: {len(ra)} vs {len(rb)} -- not comparable")
    ta = np.array([r["target"] for r in ra])
    tb = np.array([r["target"] for r in rb])
    if not np.allclose(ta, tb):
        raise SystemExit("arms were run on DIFFERENT targets; the pairing would be bogus")
    return (ta,
            np.array([r["mae"] for r in ra], dtype=float),
            np.array([r["mae"] for r in rb], dtype=float))


def discover():
    """Every arm on disk, keyed (space, weight-string). Globbed rather than hardcoded so
    the weight curve can grow without touching this script."""
    arms = {}
    for p in sorted(RESULTS.glob("e2_*_w*.json")):
        stem = p.stem[len("e2_"):]
        space, _, w = stem.rpartition("_w")
        if space in ("rate", "prob"):
            arms[(space, w)] = json.loads(p.read_text())
    return arms


def weight_curve(arms, space="prob"):
    """MAE against guidance weight, with the validity it costs.

    Reported together because they trade off: the useful operating point is where MAE
    bottoms out BEFORE validity starts falling away, and reading either column alone
    picks the wrong w."""
    pts = sorted(((float(w), d) for (s, w), d in arms.items() if s == space),
                 key=lambda kv: kv[0])
    if len(pts) < 3:
        return
    print()
    print(f"GUIDANCE-WEIGHT CURVE ({space} space)")
    print("-" * 74)
    print(f"{'w':>6} {'MAE':>8} {'low':>8} {'mid':>8} {'high':>8} {'validity':>9} {'uniq':>8}")
    for w, d in pts:
        print(f"{w:>6.1f} {d['mae_pooled']:>8.4f} {d['mae_low_third']:>8.4f} "
              f"{d['mae_mid_third']:>8.4f} {d['mae_high_third']:>8.4f} "
              f"{d['validity']:>9.4f} {d['uniqueness']:>8.4f}")
    best_w, best = min(pts, key=lambda kv: kv[1]["mae_pooled"])
    print(f"\n  MAE optimum at w={best_w:g}: {best['mae_pooled']:.4f} "
          f"(validity {best['validity']:.4f})")
    base = dict(pts)[pts[0][0]]
    print(f"  vs w={pts[0][0]:g}: {100*(best['mae_pooled']-base['mae_pooled'])/base['mae_pooled']:+.1f}% MAE, "
          f"{100*(best['validity']-base['validity']):+.2f} pts validity")
    # where validity starts to go: first w losing >1 point against the lowest w
    knee = next((w for w, d in pts if d["validity"] < base["validity"] - 0.01), None)
    print(f"  validity knee (first w losing >1 pt): "
          f"{('w=%g' % knee) if knee is not None else 'none in range'}")
    if best_w == pts[-1][0]:
        print("  NOTE: the optimum is at the edge of the sweep -- the curve has not turned "
              "over, so the best w may lie beyond it.")


def main():
    arms = discover()
    if not arms:
        sys.exit(f"no results found in {RESULTS}")

    print("=" * 74)
    print("E2 logP targeting -- validation split, 100 targets x 10, adapter-only")
    print("zinc-kek + clogp@1.2.0, seed 42, 500 steps, eta=25, omega=0, size=marginal")
    print("=" * 74)
    print(f"{'arm':<14} {'MAE':>8} {'low':>8} {'mid':>8} {'high':>8} {'valid':>8} {'uniq':>8}")
    print("-" * 74)
    for (s, w), d in sorted(arms.items()):
        print(f"{s+' w='+w:<14} {d['mae_pooled']:>8.4f} {d['mae_low_third']:>8.4f} "
              f"{d['mae_mid_third']:>8.4f} {d['mae_high_third']:>8.4f} "
              f"{d['validity']:>8.4f} {d['uniqueness']:>8.4f}")

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    print()
    print("PAIRED rate -> prob, per target (negative = prob-space is better)")
    print("-" * 74)
    for w in ("1.0", "2.0"):
        if ("rate", w) not in arms or ("prob", w) not in arms:
            continue
        tv, da, db = paired(arms[("rate", w)], arms[("prob", w)])
        ok = np.isfinite(da) & np.isfinite(db)
        tv, da, db = tv[ok], da[ok], db[ok]
        diff = db - da
        better = int((diff < 0).sum())
        line = (f"  w={w}: n={len(diff)}  mean {diff.mean():+.4f}  median {np.median(diff):+.4f}"
                f"  prob better on {better}/{len(diff)}")
        if wilcoxon is not None and len(diff) > 10 and np.any(diff != 0):
            try:
                line += f"  wilcoxon p={wilcoxon(da, db).pvalue:.4g}"
            except Exception:                                    # noqa: BLE001
                pass
        print(line)
        # the tails are where the open-loop diagnosis predicts movement
        order = np.argsort(tv)
        for name, part in zip(("low", "mid", "high"), np.array_split(order, 3)):
            print(f"      {name:>4} third: {diff[part].mean():+.4f}")

    weight_curve(arms, "prob")

    print()
    print("DOES PROB-SPACE UNLOCK w>1?")
    print("-" * 74)
    for s in ("rate", "prob"):
        if (s, "1.0") in arms and (s, "2.0") in arms:
            m1, m2 = arms[(s, "1.0")]["mae_pooled"], arms[(s, "2.0")]["mae_pooled"]
            winner = "w=2.0" if m2 < m1 else "w=1.0"
            print(f"  {s:<5}: w=1.0 {m1:.4f} vs w=2.0 {m2:.4f}  ->  {winner} wins "
                  f"({100*(m2-m1)/m1:+.1f}% going to w=2)")
    print()
    print("Reference: FreeGress logP MAE 0.22 at s=1, 0.16 at best s (ZINC-250k, Table 2),")
    print("at 81-87% validity against our ~99%; their ZINC is preprocessed differently.")


if __name__ == "__main__":
    main()
