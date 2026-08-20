#!/usr/bin/env python
"""PRE-REGISTERED analysis for the long-training arms. Written before the data existed.

The point of writing this first is that the capacity ladder's verdict already flipped
once -- from FLAT to RISING on a single extra observation -- because the analysis judged
a trend against a hard threshold (`m > 0.002`) with no error bar and no stated decision
rule. Everything below is fixed in advance so the result cannot be re-read after seeing
it.

THE HYPOTHESES

  H1 (mechanism)  slope at a fixed w=2 RISES with training epochs.
                  Predicted sign: positive. From C_long's +0.00425/epoch point estimate.
  H2 (shipping)   E2 MAE at each checkpoint's own best w FALLS with training epochs.
                  Predicted sign: negative. This is the one the request was about.

THE DECISION RULE

  "Training longer helps" iff H2's trend is negative with p < 0.05 across the five
  checkpoints. H1 is reported alongside as the mechanism, and the two are allowed to
  disagree -- slope can rise while MAE worsens if guidance overshoots, which is not a
  hypothetical: sliding-mode control at w0=3 reached slope 1.278 with WORSE MAE and
  validity than w=2.

  A single seed means there is no cross-seed error bar, so the trend across five points
  IS the estimator and its residual se is the only uncertainty available. That is stated
  here rather than discovered later.

ATTRIBUTION, also pre-registered. The open-loop diagnosis in RESEARCH.md predicts that
any real steering gain shows up at the TAILS (low and high thirds) and not in the middle,
because an open-loop controller holds the centre and fails the ends. A UNIFORM
improvement across all three thirds is evidence for "more capacity", not "better
steering". Agreeing this in advance matters because a uniform gain is easy to over-read
as confirmation of whichever story one prefers.

CONFOUND WATCH. If best_w moves upward with epochs, part of any MAE gain is "tolerates
stronger guidance" rather than "steers better at a fixed strength". Reported explicitly.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

#: The measured run-to-run spread for logP E2 MAE, from the two-seed Wave 1 pair
#: (0.5420 vs 0.5338 at identical config). A 20->100 delta smaller than this is inside
#: noise. No equivalent has ever been measured for QED, so none is asserted for it --
#: inventing a threshold is exactly the kind of after-the-fact freedom this file exists
#: to remove.
SEED_SPREAD = {"logp": 0.0082, "qed": None}


def trend(x, y):
    """OLS slope with se, t, two-sided p and R2. NaN-safe."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return {"n": int(n), "slope": float("nan"), "se": float("nan"),
                "t": float("nan"), "p": float("nan"), "r2": float("nan")}
    res = stats.linregress(x, y)
    return {"n": int(n), "slope": float(res.slope), "se": float(res.stderr),
            "t": float(res.slope / res.stderr) if res.stderr else float("nan"),
            "p": float(res.pvalue), "r2": float(res.rvalue ** 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir of {prop}_ep{N}.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = {}
    for path in sorted(glob.glob(os.path.join(args.results, "*_ep*.json"))):
        with open(path) as f:
            d = json.load(f)
        runs.setdefault(d["property"], []).append(d)

    if not runs:
        print(f"NO RESULTS in {args.results}")
        return 1

    verdict = {}
    for prop, ds in runs.items():
        ds = sorted(ds, key=lambda d: d["epoch"])
        eps = [d["epoch"] for d in ds]
        slopes = [d["slope_grid"]["slope"] for d in ds]
        maes = [d["e2"]["best_mae"] if d.get("e2") else float("nan") for d in ds]
        best_ws = [d["e2"]["best_w"] if d.get("e2") else float("nan") for d in ds]
        # MAE at a FIXED w=2 as well: "best w" can improve merely by the optimum moving,
        # and the fixed-w series is what isolates steering quality from that.
        mae_w2 = [d["e2"]["per_w"].get("2.0", {}).get("mae", float("nan"))
                  if d.get("e2") else float("nan") for d in ds]
        thirds = {k: [d["e2"]["per_w"].get("2.0", {}).get("mae_by_third", {}).get(k, float("nan"))
                      if d.get("e2") else float("nan") for d in ds]
                  for k in ("low", "mid", "high")}
        validity = [d["e2"]["per_w"].get("2.0", {}).get("validity", float("nan"))
                    if d.get("e2") else float("nan") for d in ds]

        h1, h2 = trend(eps, slopes), trend(eps, maes)
        h2_fixed = trend(eps, mae_w2)
        spread = SEED_SPREAD.get(prop)
        delta = (maes[-1] - maes[0]) if len(maes) > 1 else float("nan")

        # THE decision, by the rule stated at the top of this file.
        #
        # Three outcomes, not two. "Insufficient data" is a distinct verdict from "does
        # not help", and conflating them is the precise error that produced the last
        # capacity result: C_long was killed early, and its truncated evidence was first
        # written up as a null ("undertrained: no -- flat") when it was actually a
        # question left open. Fewer than three checkpoints cannot support a trend, so it
        # says so instead of returning False.
        if len(eps) < 3 or not np.isfinite(h2["p"]):
            decision = "INSUFFICIENT DATA"
            helps = None
        elif h2["p"] < 0.05 and h2["slope"] < 0:
            decision = "HELPS"
            helps = True
        else:
            decision = "does NOT help"
            helps = False
        if spread is None:
            noise_note = (f"no measured seed spread for {prop}; the {delta:+.4f} delta "
                          f"cannot be compared against run-to-run noise")
        else:
            noise_note = (f"delta {delta:+.4f} vs measured seed spread {spread:.4f}: "
                          f"{'EXCEEDS' if abs(delta) > spread else 'INSIDE'} noise")

        third_deltas = {k: (v[-1] - v[0]) if len(v) > 1 else float("nan")
                        for k, v in thirds.items()}
        finite_td = [v for v in third_deltas.values() if np.isfinite(v)]
        if len(finite_td) == 3:
            tails = (third_deltas["low"] + third_deltas["high"]) / 2
            attribution = ("tails (consistent with the open-loop diagnosis)"
                           if tails < third_deltas["mid"] - 1e-9 else
                           "uniform across thirds -> reads as capacity, NOT steering")
        else:
            attribution = "incomplete"

        verdict[prop] = {
            "epochs": eps, "slope_at_w2": slopes, "best_mae": maes,
            "best_w": best_ws, "mae_at_w2": mae_w2, "validity_at_w2": validity,
            "mae_by_third_at_w2": thirds,
            "H1_slope_trend": h1, "H2_mae_trend": h2, "H2_mae_trend_fixed_w2": h2_fixed,
            "delta_20_to_last": delta, "noise_note": noise_note,
            "third_deltas": third_deltas, "attribution": attribution,
            "best_w_moved": bool(np.isfinite(best_ws[0]) and np.isfinite(best_ws[-1])
                                 and best_ws[-1] != best_ws[0]),
            "n_checkpoints": len(eps),
            "VERDICT": decision,
            "VERDICT_training_longer_helps": helps,   # None == could not be determined
        }

        print(f"\n=== {prop}  ({len(eps)} checkpoints: {eps})")
        print(f"  slope@w2   {[f'{s:.3f}' for s in slopes]}")
        print(f"  best MAE   {[f'{m:.4f}' for m in maes]}   (best w {best_ws})")
        print(f"  MAE@w2     {[f'{m:.4f}' for m in mae_w2]}")
        print(f"  H1 slope trend : {h1['slope']:+.5f}/epoch  se {h1['se']:.5f}  "
              f"p={h1['p']:.3f}  R2={h1['r2']:.2f}")
        print(f"  H2 MAE trend   : {h2['slope']:+.5f}/epoch  se {h2['se']:.5f}  "
              f"p={h2['p']:.3f}  R2={h2['r2']:.2f}")
        print(f"  {noise_note}")
        print(f"  by-third delta : {', '.join(f'{k} {v:+.4f}' for k, v in third_deltas.items())}")
        print(f"  attribution    : {attribution}")
        print(f"  VERDICT: training longer -> {decision} "
              f"(pre-registered rule: H2 trend negative at p<0.05; "
              f"{len(eps)} checkpoint(s), 3 needed for a trend)")

    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
