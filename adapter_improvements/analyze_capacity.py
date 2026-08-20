#!/usr/bin/env python
"""Capacity ladder: does a bigger / longer-trained adapter recover more of QED's range?

Reads BOTH sources, because the two long arms were killed before their final eval:

* final eval  (A_base, B_wide) -- 128 molecules, 500 steps, the trustworthy number
* PROBE series (all arms)      -- 31 molecules, 100 steps, every 5 epochs

The probe is noisy, so it is only usable if calibrated. A_base and B_wide have both,
which lets us measure the probe-vs-final offset on arms where we know the answer and
carry it to C_long / D_attn where we only have probes. If that offset is large or
inconsistent between A and B, the probe cannot substitute and the script says so
rather than quietly reporting a number it cannot support.

Metric is SLOPE of achieved-vs-requested QED (1.0 = perfect tracking, 0 = ignores the
condition), not MAE: a QED adapter emitting the dataset mean already scores MAE ~0.15.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "capacity_results")
ARMS = ["A_base", "B_wide", "C_long", "D_attn"]
CONFIG = {"A_base": "h256  20ep       (4.97M)",
          "B_wide": "h1024 20ep       (20.6M)",
          "C_long": "h1024 60ep*      (20.6M)",
          "D_attn": "h1024 60ep* +attn(27.7M)"}

PROBE_RE = re.compile(r"\[epoch (\d+)\] PROBE\(w=[\d.]+\)\s*(.*)")
PAIR_RE = re.compile(r"(p\d+)->([\d.]+): achieved=([\d.]+)")


def probes(arm):
    """[(epoch, targets, achieved)] from the training log."""
    out = []
    p = RES / f"train_{arm}.log"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = PROBE_RE.search(line)
        if not m:
            continue
        pairs = PAIR_RE.findall(m.group(2))
        if len(pairs) < 3:
            continue
        t = np.array([float(x[1]) for x in pairs])
        a = np.array([float(x[2]) for x in pairs])
        out.append((int(m.group(1)), t, a))
    return out


def final(arm):
    p = RES / f"{arm}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    lv = list(d["targets"])
    t = np.array([d["targets"][k] for k in lv])
    a = np.array([d["per_level"][k]["per_w"]["1.0"]["mean"] for k in lv])
    mae = float(np.mean([d["per_level"][k]["per_w"]["1.0"]["mae"] for k in lv]))
    return t, a, mae, float(d["baseline_mean"])


def slope(t, a):
    return float(np.polyfit(t, a, 1)[0])


def main():
    print("=" * 78)
    print("CAPACITY LADDER -- QED range recovery. Slope 1.0 = perfect, 0 = ignores target.")
    print("* C_long/D_attn were killed by the SLURM wall mid-training; no final eval.")
    print("=" * 78)

    print("\nFINAL EVAL (128 mol, 500 steps) -- the trustworthy numbers")
    print("-" * 78)
    print(f"{'arm':<26} {'slope':>7} {'span':>7} {'mean MAE':>9} {'uncond':>8}")
    fin = {}
    for a in ARMS:
        f = final(a)
        if f is None:
            print(f"{CONFIG[a]:<26} {'--':>7} {'--':>7} {'--':>9} {'--':>8}   (killed before eval)")
            continue
        t, ac, mae, base = f
        fin[a] = slope(t, ac)
        print(f"{CONFIG[a]:<26} {fin[a]:>7.3f} {ac.max()-ac.min():>7.3f} {mae:>9.4f} {base:>8.3f}")

    print("\nPROBE SERIES (31 mol, 100 steps) -- slope by epoch")
    print("-" * 78)
    series = {a: probes(a) for a in ARMS}
    eps = sorted({e for a in ARMS for e, _, _ in series[a]})
    print("epoch:  " + "".join(f"{e:>6}" for e in eps))
    for a in ARMS:
        by = {e: slope(t, ac) for e, t, ac in series[a]}
        print(f"{a:<8}" + "".join(f"{by[e]:>6.2f}" if e in by else f"{'':>6}" for e in eps))

    # -- is the probe a usable stand-in? -----------------------------------
    print("\nCALIBRATION: probe vs final, on the arms that have both")
    print("-" * 78)
    offs = []
    for a in ("A_base", "B_wide"):
        if a in fin and series[a]:
            last_ep, t, ac = series[a][-1]
            ps = slope(t, ac)
            offs.append(fin[a] - ps)
            print(f"  {a}: last probe (ep {last_ep}) slope {ps:.3f}  vs final {fin[a]:.3f}"
                  f"   offset {fin[a]-ps:+.3f}")
    if len(offs) == 2 and abs(offs[0] - offs[1]) < 0.10:
        print(f"  offsets agree within {abs(offs[0]-offs[1]):.3f} -> probe is usable as a "
              f"stand-in, corrected by {np.mean(offs):+.3f}")
        for a in ("C_long", "D_attn"):
            if series[a]:
                last_ep, t, ac = series[a][-1]
                print(f"  {a}: probe(ep {last_ep}) {slope(t,ac):.3f} -> "
                      f"estimated final {slope(t,ac)+np.mean(offs):.3f}")
    else:
        print("  offsets DISAGREE between A and B -- the probe cannot stand in for the "
              "final eval. Report C_long/D_attn as trend-only, not as point estimates.")

    print("\nTREND TEST: does more training raise the slope?")
    print("-" * 78)
    print("  Judged by SIGNIFICANCE and robustness, not by a threshold on the point")
    print("  estimate. An earlier version of this script called +0.002/epoch 'RISING',")
    print("  which flipped C_long's verdict when a single tenth probe arrived — a")
    print("  magic number with no error bar is not a test.")
    try:
        from scipy import stats
    except ImportError:
        print("  scipy unavailable; cannot test significance")
        return
    for a in ("C_long", "D_attn"):
        s = series[a]
        if len(s) < 4:
            continue
        e = np.array([x[0] for x in s], float)
        v = np.array([slope(x[1], x[2]) for x in s])
        r = stats.linregress(e, v)
        drop = stats.linregress(e[:-1], v[:-1]).slope
        print(f"\n  {a}: {len(s)} probes, epochs {int(e.min())}-{int(e.max())}, "
              f"slope range {v.min():.2f}-{v.max():.2f}")
        print(f"      trend {r.slope:+.5f}/epoch  se {r.stderr:.5f}  p={r.pvalue:.3f}  "
              f"R2={r.rvalue**2:.2f}")
        print(f"      without the last probe: {drop:+.5f}/epoch")
        if r.pvalue < 0.05:
            print("      => SIGNIFICANT rise; more epochs help")
        elif abs(r.slope) > 2 * r.stderr:
            print("      => suggestive but not significant")
        else:
            print("      => NOT RESOLVED. Consistent with flat, but the data cannot")
            print("         exclude a real effect either — underpowered, not negative.")


if __name__ == "__main__":
    main()
