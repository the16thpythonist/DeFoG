#!/usr/bin/env python
"""Did the 4-quadrant stacking claim survive the move to prob-space blending?

The claim: two adapters stacked over one frozen base steer INDEPENDENTLY, so asking for
(low logP, high QED) lands in that corner rather than blurring toward the base mean.
It was established in rate space, which the joint sweep showed is materially worse with
two branches even at w=1.

Headline metric is quadrant accuracy -- correct side of the midpoint on BOTH axes at once,
chance 0.25. Reported per axis too, because a pair of adapters can look good jointly while
one of them is doing all the work.
"""
import json
import sys
from pathlib import Path

import numpy as np

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "quadrant_results")
QUADS = ["logp-lo_qed-lo", "logp-lo_qed-hi", "logp-hi_qed-lo", "logp-hi_qed-hi"]


def load(space, seed):
    p = RES / f"quad_{space}_seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    arms = {(s, seed): load(s, seed) for s in ("rate", "prob") for seed in (42, 43)}
    have = {k: v for k, v in arms.items() if v is not None}
    if not have:
        sys.exit(f"no results in {RES}")
    any_arm = next(iter(have.values()))

    print("=" * 84)
    print("4-QUADRANT STACKING -- clogp@1.2.0 + qed@3.1.0 over zinc-kek, w=1.0, 250/quadrant")
    print(f"targets: logP {tuple(round(v,2) for v in any_arm['targets']['logp'])}  "
          f"QED {tuple(round(v,3) for v in any_arm['targets']['qed'])}  "
          f"(20th/80th pct, not extremes)")
    print(f"midpoints: logP {any_arm['midpoints']['logp']:.2f}  "
          f"QED {any_arm['midpoints']['qed']:.3f}")
    print("=" * 84)

    print(f"{'arm':<16} {'quad acc':>9} {'acc logP':>9} {'acc QED':>9} {'validity':>9}")
    print("-" * 84)
    summary = {}
    for (s, seed), d in sorted(have.items()):
        qs = d["quadrants"]
        accs = np.mean([qs[q]["acc_both"] for q in QUADS if q in qs])
        al = np.mean([qs[q]["acc_logp"] for q in QUADS if q in qs])
        aq = np.mean([qs[q]["acc_qed"] for q in QUADS if q in qs])
        val = np.mean([qs[q]["validity"] for q in QUADS if q in qs])
        summary[(s, seed)] = (accs, al, aq, val)
        print(f"{s + ' seed' + str(seed):<16} {accs:>9.4f} {al:>9.4f} {aq:>9.4f} {val:>9.4f}")

    print()
    print("PER-QUADRANT achieved means (target in brackets)")
    print("-" * 84)
    for (s, seed), d in sorted(have.items()):
        if seed != 42:
            continue
        print(f"  {s}:")
        for q in QUADS:
            if q not in d["quadrants"]:
                continue
            r = d["quadrants"][q]
            print(f"    {q:<16} logP {r['achieved_logp_mean']:>6.2f}+-{r['achieved_logp_sd']:.2f} "
                  f"[{r['target_logp']:.2f}]   QED {r['achieved_qed_mean']:.3f}"
                  f"+-{r['achieved_qed_sd']:.3f} [{r['target_qed']:.3f}]   "
                  f"acc {r['acc_both']:.3f}")

    print()
    print("VERDICT")
    print("-" * 84)
    r = [summary[("rate", s)][0] for s in (42, 43) if ("rate", s) in summary]
    p = [summary[("prob", s)][0] for s in (42, 43) if ("prob", s) in summary]
    if r and p:
        rm, pm = float(np.mean(r)), float(np.mean(p))
        spread = max(abs(r[0] - r[-1]), abs(p[0] - p[-1]))
        print(f"  rate quadrant accuracy {rm:.4f}   prob {pm:.4f}   "
              f"delta {pm - rm:+.4f}  (seed spread {spread:.4f})")
        if min(rm, pm) < 0.25 + 0.05:
            print("  !! at or near chance -- the stacking claim does NOT hold here")
        elif abs(pm - rm) <= spread:
            print("  Both placements separate the quadrants and the difference is within")
            print("  seed noise: the shipped stacking claim STANDS as reported.")
        elif pm > rm:
            print("  Prob-space separates better by more than seed noise. The claim holds,")
            print("  but the published figure understates it -- worth regenerating.")
        else:
            print("  Rate-space separates BETTER, which contradicts the joint-MAE result and")
            print("  needs explaining before either number is used.")


if __name__ == "__main__":
    main()
