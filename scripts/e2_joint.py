#!/usr/bin/env python
"""E2 joint targeting: logP AND QED at once, the FreeGress Table 2 third column.

`e2_targeting.py` steers one property. This steers two, through two adapters composed
product-of-experts over the same frozen base -- the configuration the composition work
exists for, and the one no published baseline covers except FreeGress's joint column.

Protocol, as in the single-property case: 100 molecules from the split, each one's OWN
measured (logP, QED) becomes a joint target, 10 generations per target. FreeGress reports
the joint column as the MEAN of the two per-property MAEs (their rho=0 row is logP 0.23,
QED 0.07, joint 0.15), so that is what `mae_mean` here is.

WHY THIS RUN EXISTS. Rate-space blending averages one rate matrix per branch, and each is
built from an independent DISCRETE sample of a different distribution
(`rate_matrix.py:104`). With one adapter that is 2 draws; with two it is 3, so whatever is
wrong with the rate-space placement should be WORSE here. Every composition result reported
so far -- the 4-quadrant separation, the product-vs-mean PoE finding -- was produced in
rate space, so this is as much a check on those as it is a new number.
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


def draw_targets(split, n, seed):
    """(smiles, {logp, qed}) for n real molecules -- their own measured values."""
    from defog.data import zinc_reference as zref
    s = zref.load_reference_split()
    pool = {"validation": s.val_smiles, "test": s.test_smiles}[split]
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(pool)):
        smi = pool[int(i)]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            out.append((smi, {k: f(m) for k, f in PROPS.items()}))
        except Exception:                                  # noqa: BLE001
            continue
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--logp-adapter", default="molsmith/clogp@1.2.0")
    ap.add_argument("--qed-adapter", default="molsmith/qed@3.1.0")
    ap.add_argument("--split", required=True, choices=("validation", "test"))
    ap.add_argument("--blend-space", default="rate", choices=("rate", "prob"))
    ap.add_argument("--composite-mode", default="product", choices=("product", "mean"))
    ap.add_argument("--weight", type=float, default=1.0, help="applied to BOTH branches")
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--per-target", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from molsmith import sample as ms

    targets = draw_targets(args.split, args.n_targets, args.seed)
    lv = np.array([t["logp"] for _, t in targets])
    qv = np.array([t["qed"] for _, t in targets])
    print(f"{len(targets)} joint targets from {args.split} (seed {args.seed}); "
          f"logp [{lv.min():.2f}, {lv.max():.2f}]  qed [{qv.min():.3f}, {qv.max():.3f}]",
          flush=True)
    print(f"blend={args.blend_space}  composite={args.composite_mode}  w={args.weight}",
          flush=True)

    def cfg_for(tgt, seed):
        return ms.SamplingConfig(
            base=args.base, n=args.per_target, seed=seed, steps=args.steps,
            eta=args.eta, omega=args.omega, time_distortion=args.time_distortion,
            adapters=[
                ms.AdapterTarget(package=args.logp_adapter, target=tgt["logp"],
                                 weight=args.weight, property="logp"),
                ms.AdapterTarget(package=args.qed_adapter, target=tgt["qed"],
                                 weight=args.weight, property="qed"),
            ],
            composite_mode=args.composite_mode,
            blend_space=args.blend_space,
            method="none")

    if "blend_space" not in ms.SamplingConfig.__dataclass_fields__:
        sys.exit("REFUSING: this molsmith build has no SamplingConfig.blend_space, so "
                 "--blend-space would be silently ignored and the arms mislabelled.")

    loaded = ms.load(cfg_for(targets[0][1], args.seed))
    rows, err = [], {"logp": [], "qed": []}
    t0 = time.time()
    for k, (smi, tgt) in enumerate(targets):
        res = ms.sample(cfg_for(tgt, args.seed + k), loaded)
        got = {"logp": [], "qed": []}
        ok = []
        for s in [x for x in res.smiles if x]:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            try:
                vals = {p: f(m) for p, f in PROPS.items()}
            except Exception:                              # noqa: BLE001
                continue
            for p, v in vals.items():
                got[p].append(v)
            ok.append(s)
        row = {"target_smiles": smi, "target": tgt,
               "n_valid": len(ok), "validity": len(ok) / args.per_target,
               "uniqueness": (len(set(ok)) / len(ok)) if ok else float("nan")}
        for p in PROPS:
            a = np.array(got[p])
            e = np.abs(a - tgt[p]) if a.size else np.array([])
            err[p].extend(e.tolist())
            row[f"mae_{p}"] = float(e.mean()) if e.size else float("nan")
        rows.append(row)
        if (k + 1) % 10 == 0:
            fl = np.nanmean([r["mae_logp"] for r in rows])
            fq = np.nanmean([r["mae_qed"] for r in rows])
            print(f"  {k+1}/{len(targets)} targets  logp {fl:.4f}  qed {fq:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    mae_logp = float(np.mean(err["logp"])) if err["logp"] else float("nan")
    mae_qed = float(np.mean(err["qed"])) if err["qed"] else float("nan")
    summary = {
        "base": args.base, "logp_adapter": args.logp_adapter,
        "qed_adapter": args.qed_adapter, "split": args.split, "seed": args.seed,
        "n_targets": len(rows), "per_target": args.per_target,
        "sampling": {"weight": args.weight, "steps": args.steps, "eta": args.eta,
                     "omega": args.omega, "time_distortion": args.time_distortion,
                     "blend_space": args.blend_space,
                     "composite_mode": args.composite_mode},
        "mae_logp": mae_logp, "mae_qed": mae_qed,
        # FreeGress's joint column is the mean of the two per-property MAEs
        "mae_mean": float(np.mean([mae_logp, mae_qed])),
        "validity": float(np.mean([r["validity"] for r in rows])),
        "uniqueness": float(np.nanmean([r["uniqueness"] for r in rows])),
        "per_target": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print()
    print(f"=== E2 JOINT logP+QED / {args.split} / blend={args.blend_space} / "
          f"w={args.weight} ===")
    print(f"  logP MAE {mae_logp:.4f}   QED MAE {mae_qed:.4f}   "
          f"joint {summary['mae_mean']:.4f}")
    print(f"  validity {summary['validity']:.4f}   uniqueness {summary['uniqueness']:.4f}")
    print(f"  FreeGress joint reference: 0.12-0.16 at validity 0.73-0.83")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
