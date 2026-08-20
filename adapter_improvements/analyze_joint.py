#!/usr/bin/env python
"""Joint logP+QED composition: does the blend space matter more with two branches?

Two questions, and the second is the uncomfortable one.

1. Does prob-space help the joint column, and does w>1 work here as it does for logP alone?
2. IS THE rate/w=1.0 ARM SOUND? That is the configuration every previously reported
   composition result was produced under -- the 4-quadrant separation, the product-vs-mean
   PoE finding. If prob-space beats it materially at w=1, those need revisiting. If the two
   agree at w=1 and diverge only above it, the earlier work stands.

Paired per target, since all arms share the 100 targets and the seed.
"""
import json
import sys
from pathlib import Path

import numpy as np

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "joint_results")


def load(space, w):
    p = RES / f"joint_{space}_w{w}.json"
    return json.loads(p.read_text()) if p.exists() else None


def col(d, key):
    return np.array([r[key] for r in d["per_target"]], dtype=float)


def targets(d, prop):
    return np.array([r["target"][prop] for r in d["per_target"]], dtype=float)


def main():
    arms = {(s, w): load(s, w) for s in ("rate", "prob") for w in ("1.0", "2.0")}
    have = {k: v for k, v in arms.items() if v is not None}
    if not have:
        sys.exit(f"no results in {RES}")

    print("=" * 82)
    print("E2 JOINT logP+QED -- validation, 100 targets x 10, clogp@1.2.0 + qed@3.1.0")
    print("composed product-of-experts over zinc-kek; seed 42, 500 steps, eta=25")
    print("=" * 82)
    print(f"{'arm':<14} {'logP MAE':>9} {'QED MAE':>9} {'joint':>9} {'valid':>8} {'uniq':>8}")
    print("-" * 82)
    for (s, w), d in sorted(have.items()):
        print(f"{s + ' w=' + w:<14} {d['mae_logp']:>9.4f} {d['mae_qed']:>9.4f} "
              f"{d['mae_mean']:>9.4f} {d['validity']:>8.4f} {d['uniqueness']:>8.4f}")

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    print()
    print("PAIRED rate -> prob (negative = prob-space better)")
    print("-" * 82)
    for w in ("1.0", "2.0"):
        a, b = have.get(("rate", w)), have.get(("prob", w))
        if a is None or b is None:
            continue
        if not np.allclose(targets(a, "logp"), targets(b, "logp")):
            sys.exit(f"w={w}: arms ran on different targets; pairing would be bogus")
        for prop in ("logp", "qed"):
            da, db = col(a, f"mae_{prop}"), col(b, f"mae_{prop}")
            ok = np.isfinite(da) & np.isfinite(db)
            diff = db[ok] - da[ok]
            line = (f"  w={w} {prop:>4}: mean {diff.mean():+.4f}  "
                    f"prob better on {int((diff < 0).sum())}/{len(diff)}")
            if wilcoxon is not None and np.any(diff != 0):
                line += f"  p={wilcoxon(da[ok], db[ok]).pvalue:.4g}"
            print(line)

    # -- the check on previously reported composition work ----------------------
    print()
    print("IS THE HISTORICAL COMPOSITION CONFIG (rate, w=1.0) SOUND?")
    print("-" * 82)
    a, b = have.get(("rate", "1.0")), have.get(("prob", "1.0"))
    if a and b:
        d = b["mae_mean"] - a["mae_mean"]
        rel = 100 * d / a["mae_mean"]
        if abs(rel) < 5:
            print(f"  joint MAE {a['mae_mean']:.4f} -> {b['mae_mean']:.4f} ({rel:+.1f}%): "
                  f"the two placements agree at w=1, as they do for a single adapter.")
            print("  => earlier composition results stand; prob-space is upside, not a fix.")
        else:
            print(f"  joint MAE {a['mae_mean']:.4f} -> {b['mae_mean']:.4f} ({rel:+.1f}%): "
                  f"MATERIAL disagreement at w=1 with two branches.")
            print("  Why the single-adapter case was a no-op here and this is not:")
            print("    N=1, w=1:  lu + (lc1-lu) = lc1          -> the uncond term CANCELS")
            print("    N=2, w=1:  lu + (lc1-lu) + (lc2-lu)     -> R1*R2/R0, it does NOT")
            print("  So multi-branch rate-space blending is a genuine 3-way product of rate")
            print("  matrices built from three independent discrete draws of three different")
            print("  distributions -- incoherent in a way the single-branch case never was.")
            print("  => AFFECTED: any multi-adapter result computed through denoise_step in")
            print("     rate space, e.g. the 4-quadrant logP x QED stacking. Re-run those.")
            print("  => NOT affected: the size-distribution product-vs-mean finding, which")
            print("     composes over the node-count grid, not over rate matrices.")

    print()
    print("BEST ARM vs REFERENCES")
    print("-" * 82)
    (bs, bw), best = min(have.items(), key=lambda kv: kv[1]["mae_mean"])
    print(f"  best: {bs} w={bw}  joint {best['mae_mean']:.4f}  "
          f"(logP {best['mae_logp']:.4f}, QED {best['mae_qed']:.4f})  "
          f"validity {best['validity']:.4f}")
    print("  FreeGress joint: 0.12 best, 0.15-0.16 typical, at validity 0.73-0.83")
    print("  FreeGress unconditional joint: 0.83 at validity 0.861")
    print("  NB their ZINC is preprocessed differently (228k, no stereo, N+/O- only) and")
    print("  their whole regime sits far below our validity, so this is not like-for-like.")


if __name__ == "__main__":
    main()
