#!/usr/bin/env python
"""What does the base produce when nobody steers it?

Two uses, both needed to read an E2 targeting number honestly.

1. THE DO-NOTHING BASELINE. "MAE 0.14" means nothing on its own. The comparison
   that gives it meaning is a predictor that ignores the condition entirely and
   always emits the model's unconditional mean. Under the FreeGress protocol the
   targets are real molecules' own property values, so roughly a third of them sit
   near that mean and the constant predictor is unbeatable there. Skill against
   this baseline separates "the adapter steers" from "the targets were easy".

   The baseline has to be measured at the SAME eta as the arm it judges. eta
   scales the detailed-balance term, which is constructed to preserve the
   marginal -- so it moves the unconditional distribution, and a constant lifted
   from a different eta would silently compare two different models.

2. THE REACHABILITY CHECK. If the base's unconditional spread is already much
   narrower than the dataset's, the extreme targets are unreachable and no adapter
   can be blamed for missing them. This matters for zinc-kek specifically: it is
   sanity-RL'd (disconnected -33%, wonky rings -23%), and those are exactly the
   defects QED's alert and structural terms punish, so the RL is a plausible
   suspect for having compressed the low-QED tail out of the model.

Usage:
    python scripts/e2_uncond_baseline.py --base molsmith/zinc-kek --n 250 \\
        --steps 500 --eta 5 --seed 42 --out base_eta5.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/ssd2/Programming/defog-web")

import numpy as np  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, QED  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROP_FNS = {
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "qed": lambda m: float(QED.qed(m)),
    "tpsa": lambda m: float(Descriptors.TPSA(m)),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from molsmith import sample as ms

    cfg = ms.SamplingConfig(
        base=args.base, n=args.n, seed=args.seed, steps=args.steps,
        eta=args.eta, omega=args.omega, time_distortion=args.time_distortion,
        adapters=[], method="none")

    t0 = time.time()
    res = ms.sample(cfg, ms.load(cfg))
    smis = [s for s in res.smiles if s]

    vals = {k: [] for k in PROP_FNS}
    n_parsed = 0
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n_parsed += 1
        for k, fn in PROP_FNS.items():
            try:
                vals[k].append(float(fn(m)))
            except Exception:                                   # noqa: BLE001
                pass

    summary = {
        "base": args.base, "n_requested": args.n, "n_valid": n_parsed,
        # validity over what was ASKED for, matching e2_targeting's convention
        "validity": n_parsed / args.n,
        "uniqueness": (len(set(smis)) / len(smis)) if smis else float("nan"),
        "sampling": {"steps": args.steps, "eta": args.eta, "omega": args.omega,
                     "time_distortion": args.time_distortion, "seed": args.seed},
        "seconds": time.time() - t0,
    }
    for k, v in vals.items():
        a = np.array(v)
        summary[k] = {
            "n": int(a.size),
            "mean": float(a.mean()) if a.size else float("nan"),
            "sd": float(a.std()) if a.size else float("nan"),
            "p5": float(np.percentile(a, 5)) if a.size else float("nan"),
            "p50": float(np.percentile(a, 50)) if a.size else float("nan"),
            "p95": float(np.percentile(a, 95)) if a.size else float("nan"),
        }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print(f"=== unconditional baseline, eta={args.eta} ===")
    print(f"  validity {summary['validity']:.4f} over {args.n} requested "
          f"({n_parsed} parsed), uniqueness {summary['uniqueness']:.4f}")
    for k in ("logp", "qed"):
        s = summary[k]
        print(f"  {k:5s} mean {s['mean']:.4f}  sd {s['sd']:.4f}   "
              f"p5 {s['p5']:.4f}  p50 {s['p50']:.4f}  p95 {s['p95']:.4f}")
    print(f"  {summary['seconds']:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
