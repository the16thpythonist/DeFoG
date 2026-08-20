#!/usr/bin/env python
"""Marginal P(n) vs learned P(n|target), paired at each checkpoint.

The two arms differ in exactly one flag, so a difference here is attributable to the size
policy. Reported per target-third as well as pooled, because Wave 1's result was NOT a
pooled gain -- it was pooled-null (+0.0002, 100/200, p~0.80) with a -8.3% improvement
confined to the low-logP third. A pooled-only comparison would have called that nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load(d):
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "*_ep*.json"))):
        with open(path) as f:
            r = json.load(f)
        out[(r["property"], r["epoch"])] = r
    return out


def at_best_w(r):
    """(w, mae, thirds, validity) at the arm's own best weight."""
    if not r.get("e2"):
        return None
    w = str(r["e2"]["best_w"])
    row = r["e2"]["per_w"][w]
    return w, row["mae"], row.get("mae_by_third", {}), row.get("validity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marginal", required=True)
    ap.add_argument("--learned", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m, l = load(args.marginal), load(args.learned)
    keys = sorted(set(m) & set(l))
    if not keys:
        print(f"NO OVERLAPPING CHECKPOINTS between {args.marginal} and {args.learned}")
        return 1

    res = {}
    for prop, ep in keys:
        am, al = at_best_w(m[(prop, ep)]), at_best_w(l[(prop, ep)])
        if am is None or al is None:
            continue
        wm, mm, tm, vm = am
        wl, ml, tl, vl = al
        thirds = {k: (tl.get(k, float("nan")) - tm.get(k, float("nan")))
                  for k in ("low", "mid", "high")}
        sm_, sl_ = m[(prop, ep)]["slope_grid"], l[(prop, ep)]["slope_grid"]
        res[f"{prop}_ep{ep}"] = {
            "marginal": {"best_w": wm, "mae": mm, "by_third": tm, "validity": vm,
                         "slope": sm_["slope"]},
            "learned": {"best_w": wl, "mae": ml, "by_third": tl, "validity": vl,
                        "slope": sl_["slope"]},
            "delta_mae": ml - mm,
            "delta_pct": (ml - mm) / mm * 100 if mm else float("nan"),
            "delta_by_third": thirds,
            "delta_slope": sl_["slope"] - sm_["slope"],
            "delta_validity": (vl - vm) if (vl is not None and vm is not None) else None,
        }
        print(f"\n=== {prop} ep{ep}")
        print(f"  marginal  best_w={wm}  MAE {mm:.4f}  slope {sm_['slope']:.3f}  val {vm:.4f}")
        print(f"  learned   best_w={wl}  MAE {ml:.4f}  slope {sl_['slope']:.3f}  val {vl:.4f}")
        print(f"  delta MAE {ml-mm:+.4f} ({(ml-mm)/mm*100:+.1f}%)   "
              f"by third: " + ", ".join(f"{k} {v:+.4f}" for k, v in thirds.items()))
        # Wave 1's prediction, stated so it can fail: the gain should sit at the LOW end.
        lo, mid = thirds.get("low", float("nan")), thirds.get("mid", float("nan"))
        if lo == lo and mid == mid:
            print(f"  Wave-1 prediction (low end gains most): "
                  f"{'HELD' if lo < mid else 'NOT held'}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
