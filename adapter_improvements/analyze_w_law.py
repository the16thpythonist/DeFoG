#!/usr/bin/env python
"""Verdict on the guidance-gain law. Rules fixed before the confirming data exists.

CLAIM A -- LINEARITY:  slope(w) = k*w.
  Test: regress slope on w through the origin, report k and the spread of slope/w.
  PASSES if max|slope/w - k| / k <= 0.15, i.e. the per-point gain is constant to within
  15%. On QED the observed spread was 0.457..0.527 around k=0.477, i.e. 10.5% -- so 15%
  is a real bar rather than one drawn around the answer.

CLAIM B -- OPTIMALITY: MAE is minimised at w* = 1/k(w=1).
  Test: the empirically MAE-minimising weight on the grid vs the PREDICTED w*.
  PASSES if the predicted w* is within one grid step of the observed minimum. Judged
  against the prediction made from the w=1 probe ALONE (recorded in prediction.json
  before the rest of the grid ran), not against a k refitted on all the data.

The two are reported separately and can disagree: A is about the mechanism, B adds the
assumption that minimising bias minimises MAE, which only holds while the conditional
spread is roughly flat in w.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir of verify_w_law JSONs")
    ap.add_argument("--prediction", default=None, help="prediction.json from the w=1 probe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_w, meta = {}, None
    for f in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        if os.path.basename(f) in ("prediction.json", "verdict.json"):
            continue
        d = json.load(open(f))
        meta = meta or {k: d[k] for k in ("adapter", "base", "property", "sampling")}
        per_w.update(d["per_w"])
    if not per_w:
        print(f"NO RESULTS in {args.results}")
        return 1

    ws = np.array(sorted(float(w) for w in per_w))
    slope = np.array([per_w[str(w)]["slope"] for w in ws])
    mae = np.array([per_w[str(w)]["mean_mae"] for w in ws])
    sd = np.array([per_w[str(w)]["mean_sd"] for w in ws])
    bias = np.array([per_w[str(w)]["mean_abs_bias"] for w in ws])
    val = np.array([per_w[str(w)]["mean_validity"] for w in ws])

    print(f"\n=== {meta['adapter']} on {meta['base']}  ({meta['property']}, "
          f"eta={meta['sampling']['eta']})")
    print(f"{'w':>6}{'slope':>8}{'slope/w':>9}{'MAE':>9}{'|bias|':>9}{'sd':>8}{'val':>8}")
    for i, w in enumerate(ws):
        print(f"{w:>6.2f}{slope[i]:>8.3f}{slope[i]/w:>9.3f}{mae[i]:>9.4f}"
              f"{bias[i]:>9.4f}{sd[i]:>8.4f}{val[i]:>8.3f}")

    # --- claim A -------------------------------------------------------------
    k_fit = float((ws * slope).sum() / (ws * ws).sum())
    ratios = slope / ws
    spread = float(np.max(np.abs(ratios - k_fit)) / k_fit) if k_fit else float("nan")
    A_pass = bool(np.isfinite(spread) and spread <= 0.15)

    # --- claim B -------------------------------------------------------------
    pred = None
    if args.prediction and os.path.exists(args.prediction):
        pred = json.load(open(args.prediction))
    k_probe = pred["slope_at_w1"] if pred else (
        float(per_w["1.0"]["slope"]) if "1.0" in per_w else float("nan"))
    w_star_pred = (1.0 / k_probe) if k_probe else float("nan")
    w_best = float(ws[int(np.nanargmin(mae))])
    steps = np.diff(ws)
    step = float(np.median(steps)) if len(steps) else float("nan")
    B_pass = bool(np.isfinite(w_star_pred) and abs(w_best - w_star_pred) <= step + 1e-9)

    # Where slope actually crosses 1, for contrast with where MAE actually bottoms out.
    w_slope1 = float(1.0 / k_fit) if k_fit else float("nan")

    print(f"\nCLAIM A (linearity): k = {k_fit:.3f}; slope/w spread {spread*100:.1f}% "
          f"(bar: 15%)  -> {'PASS' if A_pass else 'FAIL'}")
    print(f"CLAIM B (optimality): predicted w* = 1/{k_probe:.3f} = {w_star_pred:.2f}; "
          f"observed MAE minimum at w = {w_best:.2f}; grid step {step:.2f}"
          f"  -> {'PASS' if B_pass else 'FAIL'}")
    print(f"  (slope crosses 1 at w = {w_slope1:.2f}; QED reference: k=0.477, w*=2.10)")
    if np.isfinite(sd).all() and sd.max() > 0:
        print(f"  conditional sd moves {(sd.max()-sd.min())/sd.max()*100:.1f}% across the "
              f"grid -- claim B relies on this being small")

    verdict = {
        "meta": meta, "w": ws.tolist(), "slope": slope.tolist(), "mae": mae.tolist(),
        "sd": sd.tolist(), "abs_bias": bias.tolist(), "validity": val.tolist(),
        "k_fit": k_fit, "slope_over_w_spread": spread, "claim_A_linearity": A_pass,
        "k_probe_w1": k_probe, "w_star_predicted": w_star_pred,
        "w_best_observed": w_best, "grid_step": step, "claim_B_optimality": B_pass,
        "w_where_slope_1": w_slope1,
        "qed_reference": {"k": 0.477, "w_star": 2.10},
    }
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
