#!/usr/bin/env python
"""Read job 43169: diversity knobs at beta~=60, plus the beta~=100 ladder extension.

The question is NOT "does this knob improve MAE". Every knob that weakens selection improves
uniqueness and costs MAE, and every knob that strengthens it does the reverse -- so a lone
MAE or uniqueness number says nothing. The question is whether a knob lands ABOVE the
MAE-vs-uniqueness frontier that the beta ladder already traces, i.e. whether it beats simply
turning beta down to buy the same diversity.

So each arm is scored against the ladder INTERPOLATED AT ITS OWN UNIQUENESS. Negative is
better (lower MAE at equal diversity). The interpolation is linear between five rungs and is
only as good as they are -- flagged per-arm when the bracketing rungs are close together in
uniqueness, where the slope is ill-conditioned and the comparison should not be trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LADDER_DIR = HERE / "fkfixed_results"
KNOB_DIR = HERE / "fkknobs_results"
NOISE = 0.008
LADDER = [(5, "b5"), (10, "b10"), (20, "b20"), (33.5, "b33p5"), (60, "b60")]
KNOBS = [("eta50", "eta=50"), ("warm03", "warmup=0.3"), ("ess010", "ess=0.10"),
         ("rejuv_j10", "rejuvenate j=10"), ("rejuv_j25", "rejuvenate j=25")]


def load(d, name):
    p = d / f"e2_{name}.json"
    return json.load(open(p)) if p.exists() else None


def frontier_at(pts, u):
    """Ladder MAE interpolated at uniqueness u. Returns (mae, ill_conditioned)."""
    pts = sorted(pts)                                   # by uniqueness
    us = np.array([p[0] for p in pts])
    ms = np.array([p[1] for p in pts])
    if u <= us[0]:
        return float(ms[0]), True                       # extrapolating past the end
    if u >= us[-1]:
        return float(ms[-1]), True
    i = int(np.searchsorted(us, u))
    du = us[i] - us[i - 1]
    frac = (u - us[i - 1]) / du
    return float(ms[i - 1] + frac * (ms[i] - ms[i - 1])), bool(du < 0.02)


def main() -> int:
    for prop in ("logp", "qed"):
        adapter = load(LADDER_DIR, f"{prop}_adapter")
        pts, rows = [], []
        for beta, tag in LADDER:
            d = load(LADDER_DIR, f"{prop}_{tag}")
            if d:
                pts.append((d["uniqueness"], d["mae_pooled"], beta))
                rows.append((beta, d))
        b100 = load(KNOB_DIR, f"{prop}_b100")
        base = load(LADDER_DIR, f"{prop}_b60")

        print("=" * 88)
        print(f"{prop.upper()}   adapter-only MAE {adapter['mae_pooled']:.4f} (uniq 1.000)")
        print("=" * 88)
        print("  LADDER (beta~ dimensionless)")
        print(f"    {'beta':>6} {'MAE':>8} {'uniq':>7} {'valid':>7}")
        for beta, d in rows:
            print(f"    {beta:>6.1f} {d['mae_pooled']:>8.4f} {d['uniqueness']:>7.4f} "
                  f"{d['validity']:>7.4f}")
        if b100:
            prev = rows[-1][1]["mae_pooled"]
            turn = b100["mae_pooled"] > prev + NOISE
            print(f"    {100.0:>6.1f} {b100['mae_pooled']:>8.4f} {b100['uniqueness']:>7.4f} "
                  f"{b100['validity']:>7.4f}   <- extension: "
                  f"{'TURNED OVER' if turn else 'still improving or flat'} "
                  f"({b100['mae_pooled'] - prev:+.4f} vs beta=60)")

        print(f"\n  KNOBS at beta~=60 (baseline MAE {base['mae_pooled']:.4f}, "
              f"uniq {base['uniqueness']:.4f})")
        print(f"    {'knob':>18} {'MAE':>8} {'uniq':>7} {'valid':>7} {'dMAE':>8} "
              f"{'duniq':>8} {'vs frontier':>12}  verdict")
        for tag, label in KNOBS:
            d = load(KNOB_DIR, f"{prop}_{tag}")
            if not d:
                print(f"    {label:>18}  (missing)")
                continue
            u, m = d["uniqueness"], d["mae_pooled"]
            lm, ill = frontier_at(pts, u)
            gap = m - lm
            dominated = (m > base["mae_pooled"] + NOISE) and (u < base["uniqueness"])
            if dominated:
                verdict = "DOMINATED (worse on both)"
            elif ill:
                verdict = "frontier ill-conditioned here"
            elif gap < -NOISE:
                verdict = "ABOVE frontier"
            elif gap > NOISE:
                verdict = "below frontier"
            else:
                verdict = "on frontier (within noise)"
            print(f"    {label:>18} {m:>8.4f} {u:>7.4f} {d['validity']:>7.4f} "
                  f"{m - base['mae_pooled']:>+8.4f} {u - base['uniqueness']:>+8.4f} "
                  f"{gap:>+12.4f}  {verdict}")
        print()
    print(f"noise floor on pooled MAE: {NOISE:.3f}. Uniqueness has NO measured error bar -- "
          f"no seed duplicate has been run for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
