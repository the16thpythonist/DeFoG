#!/usr/bin/env python
"""In-job gate: prove every knob this sweep varies actually changes the output.

`warmup_frac`, `ess_frac`, `rejuvenate` and `jump_length` have been passed at their defaults
in every run on this project, so none of them has ever been observed to do anything. A knob
that is silently ignored produces arms that are bit-identical to the baseline and an analysis
that reads "this lever does not help" when the lever was never connected -- which is exactly
how the size ablation was voided earlier, and how the whole FK beta sweep came to be run on
an energy that was never consulted.

`rejuvenate` deserves the suspicion most: feynman_kac.py warns that it needs a guided
proposal_transform, which this path does not pass (it steers through the CFG-blended adapter
instead). The warning does not stop the run, so "on" could mean "on and inert".

Cheap on purpose: 3 targets x 10 molecules x 100 steps per variant. The fingerprint is the
vector of per-target MAEs, not the pooled scalar -- two configs can collide on one number.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
BASE_ARGS = [
    "--base", "molsmith/zinc-kek", "--adapter", "molsmith/clogp@1.2.0",
    "--property", "logp", "--split", "validation", "--method", "fk",
    "--blend-space", "prob", "--weight", "2.0", "--size-mode", "marginal",
    "--n-targets", "3", "--per-target", "10", "--steps", "100",
    "--eta", "25", "--omega", "0", "--time-distortion", "polydec", "--seed", "42",
    "--fk-beta", "60", "--fk-warmup", "0.6", "--fk-ess", "0.25",
]

VARIANTS = {
    "baseline":   [],
    "rejuv_j10":  ["--fk-rejuvenate", "--fk-jump", "10"],
    "rejuv_j25":  ["--fk-rejuvenate", "--fk-jump", "25"],
    "eta50":      ["--eta", "50"],
    "warm03":     ["--fk-warmup", "0.3"],
    "ess010":     ["--fk-ess", "0.10"],
}


def run(name, extra, outdir):
    out = outdir / f"{name}.json"
    args = list(BASE_ARGS)
    # A later --eta / --fk-warmup / --fk-ess overrides the earlier one in argparse.
    args += extra + ["--out", str(out)]
    r = subprocess.run([PY, "-u", str(REPO / "scripts" / "e2_targeting.py")] + args,
                       capture_output=True, text=True, cwd=str(REPO))
    if r.returncode != 0:
        print(f"  {name}: FAILED rc={r.returncode}")
        print("  " + "\n  ".join(r.stderr.strip().splitlines()[-8:]))
        return None
    d = json.load(open(out))
    return np.array([t["mae"] for t in d["per_target"]], dtype=float), d


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)
        results = {}
        for name, extra in VARIANTS.items():
            print(f"running {name} ...", flush=True)
            got = run(name, extra, outdir)
            if got is None:
                print(f"GATE FAILED: variant {name} did not run")
                return 1
            results[name] = got

        base_vec, base_d = results["baseline"]
        print(f"\nbaseline  MAE {base_d['mae_pooled']:.4f}  uniq {base_d['uniqueness']:.4f}")
        print(f"{'variant':>12} {'MAE':>9} {'uniq':>7}  {'max|d per-target|':>18}  verdict")
        dead = []
        for name, (vec, d) in results.items():
            if name == "baseline":
                continue
            n = min(len(vec), len(base_vec))
            delta = float(np.abs(vec[:n] - base_vec[:n]).max())
            live = delta > 1e-9
            if not live:
                dead.append(name)
            print(f"{name:>12} {d['mae_pooled']:>9.4f} {d['uniqueness']:>7.4f} "
                  f"{delta:>18.6f}  {'live' if live else 'INERT -- flag ignored'}")

        # jump_length must matter on its own, not just the rejuvenate switch.
        j10, j25 = results["rejuv_j10"][0], results["rejuv_j25"][0]
        n = min(len(j10), len(j25))
        if float(np.abs(j10[:n] - j25[:n]).max()) <= 1e-9:
            dead.append("jump_length (j10 == j25)")
            print("\njump_length is INERT: j10 and j25 give identical results")

        if dead:
            print(f"\nGATE FAILED: inert knobs -> {', '.join(dead)}. "
                  f"Arms varying them would be indistinguishable from the baseline and the "
                  f"analysis would read as 'this lever does not help'.")
            return 1
        print("\nGATE PASSED -- every knob in this sweep changes the output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
