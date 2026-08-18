#!/usr/bin/env python
"""Read job 43167: Feynman-Kac re-run with the kekulize and scale fixes.

Three things this reports that a bare MAE table would not:

CONTROLS FIRST. Neither fix touches the ``--method adapter`` path, so the two adapter-only
arms must reproduce logP 0.5420 and QED 0.0920. If they do not, something else moved between
jobs and none of the FK arms can be attributed to the fixes. That check gates the rest.

PAIRED SIGNIFICANCE, not thresholds. Every arm sees the same 100 targets under the same seed,
so per-target MAEs pair and a paired test is the right one. Two verdicts today were wrong
because they compared a point estimate against a hand-picked cutoff with no error bar; the
seed spread on pooled MAE is ~0.008, which is larger than several deltas that looked real.

UNIQUENESS BESIDE MAE. FK resampling clones high-weight particles, so ten copies of one
molecule posts an excellent MAE. A gain with uniqueness far below 1.0 is bought partly with
diversity and has to be reported that way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).parent / "fkfixed_results"
NOISE = 0.008                       # seed spread on pooled MAE, measured earlier
CONTROLS = {"logp": 0.5420, "qed": 0.0920}
LADDER = ["b5", "b10", "b20", "b33p5", "b60"]
BETA = {"b5": 5.0, "b10": 10.0, "b20": 20.0, "b33p5": 33.5, "b60": 60.0}
SIGMA = {"logp": 1.1581127643585205, "qed": 0.1327676922082901}


def load(name):
    p = RESULTS / f"e2_{name}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def per_target_mae(d):
    return np.array([r["mae"] for r in d["per_target"] if r["mae"] is not None])


def main() -> int:
    missing = [n for n in
               [f"{p}_{s}" for p in ("logp", "qed") for s in ["adapter"] + LADDER]
               if load(n) is None]
    if missing:
        print(f"MISSING ARMS: {', '.join(missing)}")
        if len(missing) > 6:
            return 1

    print("=" * 78)
    print("CONTROL CHECK -- the adapter-only arms must reproduce the pre-fix numbers")
    print("=" * 78)
    ok = True
    for prop, expect in CONTROLS.items():
        d = load(f"{prop}_adapter")
        if d is None:
            print(f"  {prop:5s} MISSING")
            ok = False
            continue
        got = d["mae_pooled"]
        good = abs(got - expect) < 1e-4
        ok &= good
        print(f"  {prop:5s} expected {expect:.4f}  got {got:.4f}  "
              f"{'OK' if good else 'DRIFTED -- FK arms are not attributable'}")
    if not ok:
        print("\nControls failed. Everything below is descriptive only.\n")

    for prop in ("logp", "qed"):
        base = load(f"{prop}_adapter")
        if base is None:
            continue
        b_mae, b_pt = base["mae_pooled"], per_target_mae(base)
        print("\n" + "=" * 78)
        print(f"{prop.upper()}   adapter-only: MAE {b_mae:.4f}  "
              f"validity {base['validity']:.4f}  uniq {base['uniqueness']:.4f}")
        print(f"  (dimensionless beta B == raw-units beta B/{SIGMA[prop]**2:.4f})")
        print("=" * 78)
        print(f"  {'beta':>6} {'raw-eq':>8} {'MAE':>8} {'low':>8} {'mid':>8} {'high':>8} "
              f"{'valid':>7} {'uniq':>7} {'vs adpt':>9} {'p':>8}  verdict")
        for tag in LADDER:
            d = load(f"{prop}_{tag}")
            if d is None:
                print(f"  {BETA[tag]:>6.1f}  (missing)")
                continue
            pt = per_target_mae(d)
            n = min(len(pt), len(b_pt))
            p = stats.wilcoxon(pt[:n], b_pt[:n]).pvalue if n > 10 else float("nan")
            delta = d["mae_pooled"] - b_mae
            if p < 0.05 and abs(delta) > NOISE:
                verdict = "BETTER" if delta < 0 else "WORSE"
            elif abs(delta) < NOISE:
                verdict = "within noise"
            else:
                verdict = "not significant"
            print(f"  {BETA[tag]:>6.1f} {BETA[tag]/SIGMA[prop]**2:>8.1f} "
                  f"{d['mae_pooled']:>8.4f} {d['mae_low_third']:>8.4f} "
                  f"{d['mae_mid_third']:>8.4f} {d['mae_high_third']:>8.4f} "
                  f"{d['validity']:>7.4f} {d['uniqueness']:>7.4f} "
                  f"{delta:>+9.4f} {p:>8.4f}  {verdict}")

        arms = [(BETA[t], load(f"{prop}_{t}")) for t in LADDER if load(f"{prop}_{t}")]
        if len(arms) >= 3:
            bs = np.array([a[0] for a in arms])
            ms = np.array([a[1]["mae_pooled"] for a in arms])
            us = np.array([a[1]["uniqueness"] for a in arms])
            best = int(np.argmin(ms))
            print(f"\n  best: beta={bs[best]:.1f}  MAE {ms[best]:.4f} "
                  f"({ms[best]-b_mae:+.4f} vs adapter)  uniq {us[best]:.4f}")
            print(f"  monotone in beta: {bool(np.all(np.diff(ms) < 0))}  "
                  f"| turned over: {best not in (0, len(bs)-1)}  "
                  f"| at ladder edge: {best in (0, len(bs)-1)}")
            r = stats.spearmanr(bs, us)
            print(f"  uniqueness vs beta: rho {r.statistic:+.3f} (p {r.pvalue:.3f}) "
                  f"-- falling uniqueness is diversity paying for MAE")

    print(f"\nnoise floor on pooled MAE: {NOISE:.3f} (seed spread)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
