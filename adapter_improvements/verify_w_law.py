#!/usr/bin/env python
"""Test the guidance-gain law on an adapter it was NOT derived from.

THE LAW, derived on molsmith/qed@3.1.0 over five fixed weights:

    slope(w) ~= k * w          with k = slope(w=1)      [claim A: linearity]
    MAE is minimised at w* = 1/k, i.e. where slope = 1  [claim B: optimality]

Measured there: k = 0.477, w* = 2.10, and MAE did bottom out at w = 2.0.

WHY IT MIGHT NOT GENERALISE, stated before the run. Claim B is the weaker of the two.
MAE = bias + spread, and w moves both; on QED the spread barely moved (-9% from w=1 to
w=3) so the MAE optimum coincided with the bias-minimising point. On an adapter where
guidance sharpens the conditional more, the optimum would sit past slope = 1. There is
already a hint of this: the logP arm at 40 epochs posted its best MAE at w=3, where the
slope implies substantial overshoot.

TPSA is also the property most linearly readable from atom counts (R^2 = 0.852, against
0.536 for logP and 0.201 for QED), so if adapters steer it more efficiently, k should be
LARGER and w* correspondingly SMALLER than QED's 2.10. That is a directional prediction
this run can falsify.

CONDITIONS ARE COPIED FROM closed_loop_qed.py, not from the long-training evaluator:
eta=25, omega=0, 500 steps, polydec, 128 molecules per level, seed 42. The long-train
evals run at eta=5, so their numbers are NOT on this curve.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from rdkit import Chem, RDLogger                            # noqa: E402
from rdkit.Chem import Crippen, Descriptors, QED            # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROP_FNS = {
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "clogp": lambda m: float(Crippen.MolLogP(m)),
    "qed": lambda m: float(QED.qed(m)),
    "tpsa": lambda m: float(Descriptors.TPSA(m)),
}


def slope_of(targets, achieved):
    x, y = np.asarray(targets, float), np.asarray(achieved, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="molsmith/tpsa@2.0.0")
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--property", default="tpsa", choices=sorted(PROP_FNS))
    ap.add_argument("--weights", required=True, help="comma-separated guidance weights")
    ap.add_argument("--percentiles", default="5,25,50,75,95")
    ap.add_argument("--n-per-level", type=int, default=128)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--blend-space", default="prob", choices=("prob", "rate"))
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from molsmith import sample as ms
    from defog.core import AdapterComposition, ConditionBranch, AdaptedSampler
    from defog.data import zinc_reference as zref
    from experiments.utils import build_encoders, pyg_data_to_mol, mol_to_smiles

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prop_fn = PROP_FNS[args.property]

    # Load base AND adapter through molsmith, so the pair is exactly the shipped one.
    # Loading the adapter against a separately exported base would still run and would
    # silently evaluate a combination that was never trained together.
    cfg = ms.SamplingConfig(base=args.base, n=1,
                            adapters=[ms.AdapterTarget(package=args.adapter, target=0.0,
                                                       weight=1.0)])
    loaded = ms.load(cfg)
    base, adapter = loaded.base, loaded.adapters[args.adapter]
    base = base.to(device).eval()
    adapter = adapter.to(device).eval()
    adapter.check_compatible(base)
    # Decoders are LISTS, not inverted dicts: atom_decoder is ["C","N",...] and
    # bond_decoder is [None, SINGLE, ...] with a NO-BOND slot at index 0 that inverting
    # bond_encoder cannot produce. Getting this wrong decodes nothing and reports
    # validity 0.000 with no error -- which is exactly what the first smoke run did.
    # The order is taken from molsmith's own encoders rather than assumed, so the
    # vocabulary always matches the package that was loaded.
    atom_types = [k for k, _ in sorted(loaded.atom_encoder.items(), key=lambda kv: kv[1])]
    bond_types = [getattr(k, "name", str(k))
                  for k, _ in sorted(loaded.bond_encoder.items(), key=lambda kv: kv[1])]
    from defog.data import vocabulary as vocab
    print(vocab.check_model(base, atom_types, bond_types, what=f"base {args.base}"), flush=True)
    _, atom_dec, _, bond_dec = build_encoders(atom_types, bond_types)
    print(f"base {args.base}  adapter {args.adapter} "
          f"({sum(p.numel() for p in adapter.parameters()):,} params)", flush=True)

    pcts = [float(p) for p in args.percentiles.split(",")]
    pool = np.asarray([prop_fn(m) for m in
                       (Chem.MolFromSmiles(s) for s in zref.load_reference_split().train_smiles)
                       if m is not None], dtype=float)
    levels = {f"p{int(p):02d}": float(np.percentile(pool, p)) for p in pcts}
    print(f"levels: { {k: round(v, 3) for k, v in levels.items()} }", flush=True)

    out = {"adapter": args.adapter, "base": args.base, "property": args.property,
           "levels": levels, "seed": args.seed,
           "sampling": {"steps": args.steps, "eta": args.eta, "omega": args.omega,
                        "time_distortion": args.time_distortion,
                        "blend_space": args.blend_space,
                        "n_per_level": args.n_per_level},
           "per_w": {}}

    t0 = time.time()
    for w in [float(x) for x in args.weights.split(",")]:
        per_level = {}
        for name, tgt in levels.items():
            comp = AdapterComposition(
                [ConditionBranch(adapter, torch.tensor([float(tgt)]), float(w))],
                base=base, mode="product", blend_space=args.blend_space)
            samp = AdaptedSampler(base, comp, eta=args.eta, omega=args.omega,
                                  sample_steps=args.steps,
                                  time_distortion=args.time_distortion)
            torch.manual_seed(args.seed)
            got, rem = [], args.n_per_level
            while rem > 0:
                cur = min(args.chunk, rem)
                got += samp.sample(cur, device=device, show_progress=False)
                rem -= cur
            vals = []
            for s in got:
                mol = pyg_data_to_mol(s, atom_dec, bond_dec)
                smi = mol_to_smiles(mol) if mol is not None else None
                m = Chem.MolFromSmiles(smi) if smi else None
                if m is None:
                    continue
                try:
                    vals.append(float(prop_fn(m)))
                except Exception:                            # noqa: BLE001
                    continue
            v = np.asarray(vals, dtype=float)
            per_level[name] = {
                "target": tgt, "n_valid": int(v.size),
                "validity": float(v.size / args.n_per_level),
                "achieved_mean": float(v.mean()) if v.size else float("nan"),
                "achieved_sd": float(v.std()) if v.size else float("nan"),
                "mae": float(np.mean(np.abs(v - tgt))) if v.size else float("nan"),
            }
        s = slope_of([per_level[k]["target"] for k in per_level],
                     [per_level[k]["achieved_mean"] for k in per_level])
        row = {
            "per_level": per_level, "slope": s,
            "mean_mae": float(np.nanmean([per_level[k]["mae"] for k in per_level])),
            "mean_sd": float(np.nanmean([per_level[k]["achieved_sd"] for k in per_level])),
            "mean_abs_bias": float(np.nanmean(
                [abs(per_level[k]["achieved_mean"] - per_level[k]["target"]) for k in per_level])),
            "mean_validity": float(np.mean([per_level[k]["validity"] for k in per_level])),
        }
        out["per_w"][str(w)] = row
        print(f"w={w:<5} slope {s:7.3f}  MAE {row['mean_mae']:.4f}  "
              f"|bias| {row['mean_abs_bias']:.4f}  sd {row['mean_sd']:.4f}  "
              f"val {row['mean_validity']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
