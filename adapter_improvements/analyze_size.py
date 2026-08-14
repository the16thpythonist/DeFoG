#!/usr/bin/env python
"""Size ablation on the frozen blend config (prob, w=2.0): marginal vs learned P(n|y).

Three things at once, because the design gives them for free:

1. THE ABLATION -- paired marginal vs learned, per seed. Paired because both arms use the
   same 100 targets, so target difficulty cancels.
2. THE SEED DUPLICATE -- the whole weight curve behind the w=2.0 optimum was a single seed.
   marginal/seed43 is the first independent draw of that configuration, so the spread
   between the two marginal arms IS the error bar on the -16% headline.
3. AN EXACT REPRODUCTION CHECK -- marginal/seed42 is bit-for-bit the same configuration as
   the earlier e2_prob_w2.0.json arm. It should reproduce EXACTLY. If it does not, the
   pipeline is not deterministic and every paired comparison here is on sand.
"""
import json
import sys
from pathlib import Path

import numpy as np

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "size_results")
PRIOR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("blend_results/e2_prob_w2.0.json")


def load(mode, seed):
    p = RES / f"e2_size-{mode}_seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def mae_of(d):
    return np.array([r["mae"] for r in d["per_target"]], dtype=float)


def targets_of(d):
    return np.array([r["target"] for r in d["per_target"]], dtype=float)


def thirds(x, order):
    return [float(np.nanmean(x[p])) for p in np.array_split(order, 3)]


def main():
    arms = {(m, s): load(m, s) for m in ("marginal", "learned") for s in (42, 43)}
    have = {k: v for k, v in arms.items() if v is not None}
    if not have:
        sys.exit(f"no results in {RES}")

    print("=" * 78)
    print("SIZE ABLATION -- E2 logP, validation, prob-space blend, w=2.0, 100 targets x 10")
    print("=" * 78)
    print(f"{'arm':<20} {'MAE':>8} {'low':>8} {'mid':>8} {'high':>8} {'valid':>8} {'uniq':>8}")
    print("-" * 78)
    for (m, s), d in sorted(have.items()):
        print(f"{'size=' + m + ' seed' + str(s):<20} {d['mae_pooled']:>8.4f} "
              f"{d['mae_low_third']:>8.4f} {d['mae_mid_third']:>8.4f} "
              f"{d['mae_high_third']:>8.4f} {d['validity']:>8.4f} {d['uniqueness']:>8.4f}")

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    # -- 1. the ablation, paired ------------------------------------------------
    print()
    print("PAIRED marginal -> learned (negative = learned size helps)")
    print("-" * 78)
    pooled = []
    for s in (42, 43):
        a, b = have.get(("marginal", s)), have.get(("learned", s))
        if a is None or b is None:
            continue
        ta, tb = targets_of(a), targets_of(b)
        if not np.allclose(ta, tb):
            sys.exit(f"seed {s}: arms ran on different targets; pairing would be bogus")
        da, db = mae_of(a), mae_of(b)
        ok = np.isfinite(da) & np.isfinite(db)
        diff = db[ok] - da[ok]
        pooled.append(diff)
        line = (f"  seed {s}: n={len(diff)}  mean {diff.mean():+.4f}  "
                f"median {np.median(diff):+.4f}  learned better on "
                f"{int((diff < 0).sum())}/{len(diff)}")
        if wilcoxon is not None and np.any(diff != 0):
            line += f"  p={wilcoxon(da[ok], db[ok]).pvalue:.4g}"
        print(line)
        order = np.argsort(ta[ok])
        for name, val in zip(("low", "mid", "high"), thirds(diff, order)):
            print(f"      {name:>4} third: {val:+.4f}")
    if len(pooled) == 2:
        alld = np.concatenate(pooled)
        print(f"  BOTH SEEDS: n={len(alld)}  mean {alld.mean():+.4f}  "
              f"learned better on {int((alld < 0).sum())}/{len(alld)}")

    # -- 2. how big is run-to-run noise? ---------------------------------------
    print()
    print("SEED SPREAD (the error bar the single-seed weight curve never had)")
    print("-" * 78)
    for m in ("marginal", "learned"):
        a, b = have.get((m, 42)), have.get((m, 43))
        if a is None or b is None:
            continue
        print(f"  size={m:<9} seed42 {a['mae_pooled']:.4f}  seed43 {b['mae_pooled']:.4f}  "
              f"|diff| {abs(a['mae_pooled']-b['mae_pooled']):.4f}")

    # -- 3. does the pipeline reproduce exactly? --------------------------------
    print()
    print("REPRODUCTION CHECK vs the earlier run of the identical configuration")
    print("-" * 78)
    a = have.get(("marginal", 42))
    if a is not None and PRIOR.exists():
        prior = json.loads(PRIOR.read_text())
        d1, d2 = mae_of(prior), mae_of(a)
        if len(d1) == len(d2):
            md = np.nanmax(np.abs(d1 - d2))
            verdict = ("EXACT -- pipeline is deterministic" if md < 1e-9 else
                       f"DIFFERS by up to {md:.4f} -- pipeline is NOT deterministic, so "
                       f"treat every paired comparison above as noisier than it looks")
            print(f"  {PRIOR.name} vs size=marginal seed42: {verdict}")
            print(f"    pooled MAE {prior['mae_pooled']:.4f} vs {a['mae_pooled']:.4f}")
        else:
            print("  length mismatch; skipped")
    else:
        print(f"  {PRIOR} not found; skipped")


if __name__ == "__main__":
    main()
