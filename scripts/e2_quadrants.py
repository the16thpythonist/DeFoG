#!/usr/bin/env python
"""The 4-quadrant stacking check, re-run in both blend spaces.

The shipped composition claim is that two adapters stacked over one frozen base steer
INDEPENDENTLY -- ask for (low logP, high QED) and you get that corner, not a blur around
the base's mean. That claim was established in rate space, and the joint sweep showed
multi-branch rate-space blending is materially worse than prob-space even at w=1 (joint
MAE 0.4335 -> 0.3647). So the claim needs re-checking on the placement we now believe.

Targets are the 20th/80th percentiles of each property over the split -- deliberately not
the extremes, which would make separation look easy. Quadrant accuracy is the headline:
the fraction of generated molecules landing on the correct side of the MIDPOINT for BOTH
properties at once. Chance is 25%.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                        # noqa: E402
from rdkit import Chem, RDLogger                          # noqa: E402
from rdkit.Chem import Crippen, QED                       # noqa: E402

RDLogger.DisableLog("rdApp.*")
PROPS = {"logp": lambda m: float(Crippen.MolLogP(m)),
         "qed": lambda m: float(QED.qed(m))}


def reference_quantiles(split, lo=20, hi=80, limit=5000):
    from defog.data import zinc_reference as zref
    s = zref.load_reference_split()
    pool = {"validation": s.val_smiles, "test": s.test_smiles}[split][:limit]
    vals = {k: [] for k in PROPS}
    for smi in pool:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            for k, f in PROPS.items():
                vals[k].append(f(m))
        except Exception:                                  # noqa: BLE001
            continue
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi)))
            for k, v in vals.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--logp-adapter", default="molsmith/clogp@1.2.0")
    ap.add_argument("--qed-adapter", default="molsmith/qed@3.1.0")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--blend-space", default="rate", choices=("rate", "prob"))
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--n-per-quadrant", type=int, default=250)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from molsmith import sample as ms
    if "blend_space" not in ms.SamplingConfig.__dataclass_fields__:
        sys.exit("REFUSING: molsmith has no blend_space; the arms would be mislabelled.")

    q = reference_quantiles(args.split)
    mid = {k: float(np.mean(v)) for k, v in q.items()}
    print(f"targets  logp {q['logp']}  qed {q['qed']}", flush=True)
    print(f"midpoints {mid}   blend={args.blend_space} w={args.weight}", flush=True)

    quadrants = [(lp, qd) for lp in ("lo", "hi") for qd in ("lo", "hi")]

    def cfg_for(lp, qd, seed):
        return ms.SamplingConfig(
            base=args.base, n=args.n_per_quadrant, seed=seed, steps=args.steps,
            eta=args.eta, omega=args.omega, time_distortion=args.time_distortion,
            adapters=[
                ms.AdapterTarget(package=args.logp_adapter,
                                 target=q["logp"][0 if lp == "lo" else 1],
                                 weight=args.weight, property="logp"),
                ms.AdapterTarget(package=args.qed_adapter,
                                 target=q["qed"][0 if qd == "lo" else 1],
                                 weight=args.weight, property="qed"),
            ],
            composite_mode="product", blend_space=args.blend_space, method="none")

    loaded = ms.load(cfg_for("lo", "lo", args.seed))
    out = {"targets": q, "midpoints": mid, "blend_space": args.blend_space,
           "weight": args.weight, "seed": args.seed,
           "n_per_quadrant": args.n_per_quadrant, "quadrants": {}}
    t0 = time.time()
    correct_all, n_all = 0, 0
    for k, (lp, qd) in enumerate(quadrants):
        res = ms.sample(cfg_for(lp, qd, args.seed + k), loaded)
        got = {"logp": [], "qed": []}
        n_valid = 0
        for s in [x for x in res.smiles if x]:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            try:
                vals = {p: f(m) for p, f in PROPS.items()}
            except Exception:                              # noqa: BLE001
                continue
            n_valid += 1
            for p, v in vals.items():
                got[p].append(v)
        a_lp, a_qd = np.array(got["logp"]), np.array(got["qed"])
        want_lp_hi, want_qd_hi = lp == "hi", qd == "hi"
        ok_lp = (a_lp > mid["logp"]) == want_lp_hi
        ok_qd = (a_qd > mid["qed"]) == want_qd_hi
        both = ok_lp & ok_qd
        correct_all += int(both.sum())
        n_all += len(both)
        name = f"logp-{lp}_qed-{qd}"
        out["quadrants"][name] = {
            "target_logp": q["logp"][0 if lp == "lo" else 1],
            "target_qed": q["qed"][0 if qd == "lo" else 1],
            "n_valid": n_valid, "validity": n_valid / args.n_per_quadrant,
            "achieved_logp_mean": float(a_lp.mean()) if a_lp.size else float("nan"),
            "achieved_logp_sd": float(a_lp.std()) if a_lp.size else float("nan"),
            "achieved_qed_mean": float(a_qd.mean()) if a_qd.size else float("nan"),
            "achieved_qed_sd": float(a_qd.std()) if a_qd.size else float("nan"),
            "acc_logp": float(ok_lp.mean()) if ok_lp.size else float("nan"),
            "acc_qed": float(ok_qd.mean()) if ok_qd.size else float("nan"),
            "acc_both": float(both.mean()) if both.size else float("nan"),
            "logp": got["logp"], "qed": got["qed"],
        }
        print(f"  {name}: n={n_valid} logp {a_lp.mean():.2f}+-{a_lp.std():.2f} "
              f"qed {a_qd.mean():.3f}+-{a_qd.std():.3f}  acc_both {both.mean():.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    out["quadrant_accuracy"] = correct_all / max(1, n_all)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print()
    print(f"=== 4-QUADRANT / blend={args.blend_space} / w={args.weight} ===")
    print(f"  quadrant accuracy {out['quadrant_accuracy']:.4f}  (chance 0.25)")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
