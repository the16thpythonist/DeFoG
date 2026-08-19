#!/usr/bin/env python
"""
The Run A readout (docs/dam_design.md section 9).

Run A compares three estimators. The in-experiment probe reads out at TWO targets
(TARGET_PERCENTILES = [5, 95]) with N_PER_TARGET = 128, which reconstructs to about
0.03 SE per band before RL-seed variance -- so the two-arm minimum detectable effect
is roughly 0.06, which is the ENTIRE effect the plan cites as its prior. That probe
stays where it is, for early stopping; it is not the readout.

The readout is the E2 protocol this repo already has: 100 targets drawn from the
validation split at 10 samples each, produced by scripts/eval_adapter_ckpt.py.
Measured on adapter_improvements/*/e2_*.json, the across-target SD is 0.18, i.e.
SE 0.018 -- and because draw_targets is seeded, every arm sees the IDENTICAL target
list, so the comparison can be paired per target rather than pooled.

This script does that comparison. It does not sample anything; it reads the JSONs.

    # per arm, per seed:
    python scripts/eval_adapter_ckpt.py --base <base.ckpt> \\
        --adapter-ckpt <run>/logp_adapter_rl.ckpt --property logp \\
        --n-targets 100 --per-target 10 --seed 42 --out e2_dam_s42.json

    python scripts/dam_paired_analysis.py --arm gdpo e2_gdpo_s*.json \\
                                          --arm ram  e2_ram_s*.json \\
                                          --arm dam  e2_dam_s*.json
"""

import argparse
import json
import math
from collections import defaultdict


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    rows = d["per_target"]
    keys = [r.get("target_smiles", r.get("target")) for r in rows]
    mae = {k: r["mae"] for k, r in zip(keys, rows) if r.get("mae") is not None}
    return d, mae


def paired(a, b):
    """Per-target differences b - a over the targets both arms scored."""
    common = [k for k in a if k in b]
    d = [b[k] - a[k] for k in common]
    n = len(d)
    if n < 2:
        return None
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "t": (mean / se if se > 0 else float("nan"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", action="append", required=True,
                    metavar=("NAME", "JSON"),
                    help="--arm NAME file1.json [file2.json ...], repeatable")
    ap.add_argument("--baseline", default="gdpo", help="arm every other arm is compared to")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, files = spec[0], spec[1:]
        if not files:
            raise SystemExit(f"arm {name!r} has no json files")
        arms[name] = [load(f) for f in files]

    print(f"{'arm':>8} {'seeds':>6} {'targets':>8} {'pooled MAE':>11} {'per-target':>11} "
          f"{'SE':>7} {'valid':>7}")
    for name, runs in arms.items():
        maes = [sum(m.values()) / len(m) for _, m in runs]
        n_t = min(len(m) for _, m in runs)
        allm = [v for _, m in runs for v in m.values()]
        sd = math.sqrt(sum((x - sum(allm) / len(allm)) ** 2 for x in allm) / (len(allm) - 1))
        val = [d.get("validity") for d, _ in runs if d.get("validity") is not None]
        print(f"{name:>8} {len(runs):6d} {n_t:8d} "
              f"{sum(d.get('mae_pooled', float('nan')) for d, _ in runs) / len(runs):11.4f} "
              f"{sum(maes) / len(maes):11.4f} {sd / math.sqrt(len(allm)):7.4f} "
              f"{(sum(val) / len(val) if val else float('nan')):7.3f}")

    if args.baseline not in arms:
        raise SystemExit(f"baseline arm {args.baseline!r} not given")

    print(f"\nPAIRED per-target difference against {args.baseline!r} "
          f"(negative = better targeting)")
    print(f"{'arm':>8} {'seed':>6} {'n':>5} {'mean d':>9} {'SE':>8} {'t':>7}")
    base_runs = arms[args.baseline]
    for name, runs in arms.items():
        if name == args.baseline:
            continue
        for i, (_, m) in enumerate(runs):
            ref = base_runs[min(i, len(base_runs) - 1)][1]
            st = paired(ref, m)
            if st is None:
                print(f"{name:>8} {i:6d}     -  (no overlapping targets)")
                continue
            print(f"{name:>8} {i:6d} {st['n']:5d} {st['mean']:+9.4f} {st['se']:8.4f} "
                  f"{st['t']:+7.2f}")

    print("\nPre-registered in docs/dam_design.md section 9: the HIGH band carries the")
    print("confirmatory claim, paired two-sided t on the per-target differences,")
    print("alpha = 0.05. Seeds are blocks; do not pool them before pairing.")
    print("A run whose iteration count differs from the others is not comparable --")
    print("check 'iterations' and 'rl_iters_pinned' in each run's summary JSON.")


if __name__ == "__main__":
    main()
