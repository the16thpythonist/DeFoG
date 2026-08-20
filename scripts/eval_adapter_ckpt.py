#!/usr/bin/env python
"""
Evaluate a RAW adapter checkpoint -- both pre-registered metrics, one code path.

WHY THIS EXISTS RATHER THAN `e2_targeting.py`. That script resolves its adapter through
the molsmith store (`ms.load` on a `namespace/name@version` reference), which is right
for anything shipped and wrong for the five throwaway mid-training checkpoints this is
built to compare. Packaging each of them into the store to answer one question would put
five dead versions in a store that other people read.

WHY BOTH METRICS LIVE HERE. The "does training longer help?" question was pre-registered
on two numbers that answer different things, and they must come from the same weights,
the same seed and the same sampler or the pair is not a pair:

  * SLOPE, on the p05..p95 percentile grid at a FIXED weight -- the mechanism readout.
    Directly comparable to the capacity ladder's 0.369 / 0.323. Slope, not MAE, because
    an adapter that ignores its target and emits the dataset mean already scores QED
    MAE ~0.15; slope 0 says "ignored the request" and slope 1 says "tracked it".
  * E2 MAE, on 100 real molecules' own property values x 10 draws -- the shipping
    readout, and the number FreeGress's 0.16 is comparable to.

Neither alone is sufficient: slope can rise while MAE worsens (guidance overshoots --
measured: sliding-mode at w0=3 reaches slope 1.278 with WORSE MAE and validity), and MAE
can improve for reasons that have nothing to do with steering.

SPLIT DISCIPLINE. Defaults to validation. The test split is a one-shot resource and
nothing here should touch it.

Usage:
    python scripts/eval_adapter_ckpt.py \\
        --base ckpts/zinc_kek/best_model --adapter-ckpt run/qed_adapter_ep60.ckpt \\
        --property qed --weights 1.0,2.0,3.0 --slope-weight 2.0 \\
        --out results/qed_ep60.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
from rdkit import Chem, RDLogger                                # noqa: E402
from rdkit.Chem import Crippen, Descriptors, QED                # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROP_FNS = {
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "qed": lambda m: float(QED.qed(m)),
    "tpsa": lambda m: float(Descriptors.TPSA(m)),
}


class _TargetedSize:
    """A :class:`SizeDistribution` with one target value pinned into it.

    ``_prepare_generation`` calls ``size_dist.sample(n, condition=condition, ...)`` with
    whatever ``condition`` reached ``Sampler.sample`` -- which here is None, because the
    target rides in the adapter's ConditionBranch rather than in the sampler's own
    conditioning path. Without this wrapper the size model would be handed None and fall
    back to its unconditional marginal, silently producing "learned" results that are
    nothing of the kind. Same reason ``e2_targeting`` carries one.
    """

    def __init__(self, model, target: float, n: int):
        self.model = model
        self.condition = torch.full((max(int(n), 1), model.cond_dim), float(target))

    def sample(self, num_samples, condition=None, device=None, generator=None):
        c = (self.condition[:num_samples] if self.condition.size(0) >= num_samples
             else self.condition[:1].expand(num_samples, -1))
        return self.model.sample(num_samples, condition=c, device=device,
                                 generator=generator)

    @property
    def max_size(self) -> int:
        return self.model.max_size

    def log_prob(self, sizes, condition=None):
        return self.model.log_prob(sizes, condition=self.condition[:sizes.numel()])


def _vocabulary(name: str):
    """(atom_types, bond_types) for a named base vocabulary.

    Kept byte-identical to ``adapter_training__zinc._vocabulary`` rather than imported,
    because importing that module instantiates a pycomex Experiment as a side effect.
    The ORDER is the payload here, not just the membership.
    """
    if name == "legacy_aromatic":
        return (["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"],
                ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])
    if name == "e1_kekulized":
        from defog.data import zinc_reference as zref
        return (list(zref.ATOM_TYPES), list(zref.BOND_TYPES))
    raise ValueError(f"unknown vocabulary {name!r}; have 'legacy_aromatic', 'e1_kekulized'")


def draw_targets(split: str, n: int, seed: int, prop_fn):
    """The E2 targets: real molecules' OWN measured property, per FreeGress §4.2.

    Deliberately identical in construction to ``e2_targeting.draw_targets`` -- same
    reference split, same rng, same skip-on-unparseable rule -- so a number produced
    here and a number produced there are drawn from the same 100 molecules.
    """
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
            out.append((smi, float(prop_fn(m))))
        except Exception:                                       # noqa: BLE001
            continue
        if len(out) >= n:
            break
    return out


def _sample_at(base, adapter, target, n, w, *, steps, eta, omega, td, chunk,
               blend_space, device, seed, atom_decoder, bond_decoder, prop_fn,
               size_model=None):
    """`n` molecules steered at `target` with weight `w`; returns (values, smiles, n_asked)."""
    from defog.core import AdapterComposition, ConditionBranch, AdaptedSampler
    from experiments.utils import pyg_data_to_mol, mol_to_smiles

    comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([float(target)]), float(w))],
                              base=base, mode="product", blend_space=blend_space)
    samp = AdaptedSampler(base, comp, eta=eta, omega=omega, sample_steps=steps,
                          time_distortion=td)
    size_dist = None if size_model is None else _TargetedSize(size_model, target, n)
    torch.manual_seed(seed)
    out, rem = [], int(n)
    while rem > 0:
        cur = min(chunk, rem)
        out += samp.sample(cur, size_dist=size_dist, device=device, show_progress=False)
        rem -= cur
    vals, smis = [], []
    for s in out:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        try:
            vals.append(float(prop_fn(m)))
            smis.append(smi)
        except Exception:                                       # noqa: BLE001
            continue
    return np.asarray(vals, dtype=float), smis, int(n)


def _ols(x, y):
    """Slope with a standard error, so a trend is never read off a point estimate alone.

    The capacity ladder's verdict flipped from FLAT to RISING on one extra observation
    precisely because it compared a slope against a hard threshold with no error bar.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan"), float("nan")
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).sum()
    b = ((x - mx) * (y - my)).sum() / sxx
    a = my - b * mx
    if n < 3:
        return float(b), float("nan"), float("nan")
    resid = y - (a + b * x)
    se = float(np.sqrt((resid ** 2).sum() / (n - 2) / sxx))
    ss_tot = ((y - my) ** 2).sum()
    r2 = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    return float(b), se, r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to the frozen base checkpoint")
    ap.add_argument("--adapter-ckpt", required=True, help="path to a raw adapter .ckpt")
    ap.add_argument("--property", required=True, choices=sorted(PROP_FNS))
    ap.add_argument("--vocabulary", default="e1_kekulized",
                    choices=("e1_kekulized", "legacy_aromatic"),
                    help="MUST match the base; the two share 9 atoms in different orders "
                         "and mismatching them mis-decodes silently")
    ap.add_argument("--csv", default=None,
                    help="dataset CSV, for the percentile grid (defaults to the "
                         "reference split's own training smiles)")
    ap.add_argument("--split", default="validation", choices=("validation", "test"))
    ap.add_argument("--epoch", type=int, default=None, help="recorded, for the trend fit")
    # --- metric A: percentile grid -> slope ---
    ap.add_argument("--percentiles", default="5,25,50,75,95")
    ap.add_argument("--slope-weight", type=float, default=2.0,
                    help="the FIXED weight the slope is measured at")
    ap.add_argument("--n-per-level", type=int, default=128)
    # --- metric B: E2 protocol -> MAE ---
    ap.add_argument("--weights", default="1.0,2.0,3.0",
                    help="comma-separated w grid for the E2 sweep; best-w MAE is reported")
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--per-target", type=int, default=10)
    ap.add_argument("--skip-e2", action="store_true", help="metric A only (cheap probe)")
    # --- sampling ---
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=5.0)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--blend-space", default="prob", choices=("prob", "rate"))
    # DEFAULT marginal, so this stays comparable with the in-flight run and with every
    # E2 number before the size model existed. "learned" is an ABLATION AXIS, not a free
    # improvement: FreeGress Tab. 3 shows conditioned node inference alone moving MW MAE
    # by -70%, i.e. able to dominate the column it appears in. Run the pair, report both.
    ap.add_argument("--size-mode", default="marginal", choices=("marginal", "learned"))
    ap.add_argument("--size-model", default=None, help="LearnedSizeDistribution ckpt")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd
    from defog.core import DeFoGModel, AdaLNAdapter
    from experiments.utils import build_encoders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prop_fn = PROP_FNS[args.property]

    base = DeFoGModel.load(args.base).to(device).eval()
    adapter = AdaLNAdapter.load(args.adapter_ckpt, device=device).eval()
    adapter.check_compatible(base)
    print(f"base {args.base}\nadapter {args.adapter_ckpt} "
          f"({sum(p.numel() for p in adapter.parameters()):,} params)", flush=True)

    size_model = None
    if args.size_mode == "learned":
        if not args.size_model:
            sys.exit("--size-mode learned needs --size-model PATH")
        from defog.core import LearnedSizeDistribution
        size_model = LearnedSizeDistribution.load(args.size_model)
        # A size model fitted on a DIFFERENT property conditions graph size on the wrong
        # signal and still runs, producing plausible numbers labelled "learned". clogp is
        # an explicit alias of logp (same Crippen estimate), so it is accepted.
        alias = {"clogp": "logp"}
        got = alias.get(size_model.property_name, size_model.property_name)
        want = alias.get(args.property, args.property)
        if got != want:
            sys.exit(f"REFUSING: size model is for {size_model.property_name!r} but "
                     f"--property is {args.property!r}; sizes would be conditioned on the "
                     f"wrong signal and the run would still look fine.")
        print(f"size model: {args.size_model} property={size_model.property_name!r} "
              f"from={size_model.property_from!r} grid {size_model.min_size}.."
              f"{size_model.max_size}", flush=True)

    # The decoders must be the base's own. This CANNOT be read off the checkpoint: the
    # two ZINC vocabularies have the same 9 atoms in DIFFERENT orders (C N O S F Cl Br
    # I P vs C N O F P S Cl Br I), so a model loaded under the wrong one still has
    # matching class counts and decodes every graph to a different molecule in silence.
    # Hence an explicit flag plus check_model, which at least catches a count mismatch.
    from defog.data import vocabulary as vocab
    atom_types, bond_types = _vocabulary(args.vocabulary)
    print(vocab.check_model(base, atom_types, bond_types, what=f"base {args.base}"), flush=True)
    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, bond_types)

    # ---- metric A: percentile grid at a fixed weight -> slope ----------------
    pcts = [float(p) for p in args.percentiles.split(",")]
    if args.csv:
        vals_pool = np.asarray([prop_fn(m) for m in
                                (Chem.MolFromSmiles(s) for s in pd.read_csv(args.csv)["smiles"])
                                if m is not None], dtype=float)
    else:
        from defog.data import zinc_reference as zref
        s = zref.load_reference_split()
        vals_pool = np.asarray([prop_fn(m) for m in
                                (Chem.MolFromSmiles(x) for x in s.train_smiles)
                                if m is not None], dtype=float)
    levels = {f"p{int(p):02d}": float(np.percentile(vals_pool, p)) for p in pcts}
    print(f"levels: { {k: round(v, 4) for k, v in levels.items()} }", flush=True)

    grid, t0 = {}, time.time()
    for name, tgt in levels.items():
        v, _, asked = _sample_at(
            base, adapter, tgt, args.n_per_level, args.slope_weight,
            steps=args.steps, eta=args.eta, omega=args.omega, td=args.time_distortion,
            chunk=args.chunk, blend_space=args.blend_space, device=device,
            seed=args.seed, atom_decoder=atom_decoder, bond_decoder=bond_decoder,
            prop_fn=prop_fn, size_model=size_model)
        grid[name] = {
            "target": tgt, "n_valid": int(v.size), "validity": float(v.size / asked),
            "achieved_mean": float(v.mean()) if v.size else float("nan"),
            "achieved_sd": float(v.std()) if v.size else float("nan"),
            "mae": float(np.mean(np.abs(v - tgt))) if v.size else float("nan"),
        }
        print(f"  {name} target={tgt:.4f} -> mean={grid[name]['achieved_mean']:.4f} "
              f"mae={grid[name]['mae']:.4f} val={grid[name]['validity']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    xs = [grid[k]["target"] for k in grid]
    ys = [grid[k]["achieved_mean"] for k in grid]
    slope, slope_se, slope_r2 = _ols(xs, ys)
    metric_a = {
        "weight": args.slope_weight, "levels": grid,
        "slope": slope, "slope_se": slope_se, "slope_r2": slope_r2,
        "target_span": float(max(xs) - min(xs)),
        "achieved_span": float(np.nanmax(ys) - np.nanmin(ys)),
        "mean_mae": float(np.nanmean([grid[k]["mae"] for k in grid])),
        "mean_validity": float(np.mean([grid[k]["validity"] for k in grid])),
    }
    print(f"SLOPE @w={args.slope_weight}: {slope:.4f} (se {slope_se:.4f}, R2 {slope_r2:.3f})",
          flush=True)

    # ---- metric B: E2 protocol over a weight grid ---------------------------
    metric_b = None
    if not args.skip_e2:
        targets = draw_targets(args.split, args.n_targets, args.seed, prop_fn)
        tv = np.array([v for _, v in targets])
        print(f"{len(targets)} E2 targets from {args.split} (seed {args.seed}); "
              f"range [{tv.min():.3f}, {tv.max():.3f}]", flush=True)
        per_w = {}
        for w in [float(x) for x in args.weights.split(",")]:
            rows = []
            for k, (smi, tgt) in enumerate(targets):
                v, smis, asked = _sample_at(
                    base, adapter, tgt, args.per_target, w,
                    steps=args.steps, eta=args.eta, omega=args.omega,
                    td=args.time_distortion, chunk=args.chunk,
                    blend_space=args.blend_space, device=device,
                    seed=args.seed + k, atom_decoder=atom_decoder,
                    bond_decoder=bond_decoder, prop_fn=prop_fn,
                    size_model=size_model)
                rows.append({
                    "target": tgt, "n_valid": int(v.size),
                    "validity": v.size / asked,
                    "uniqueness": (len(set(smis)) / len(smis)) if smis else float("nan"),
                    "mae": float(np.mean(np.abs(v - tgt))) if v.size else float("nan"),
                })
                if (k + 1) % 25 == 0:
                    done = [r["mae"] for r in rows if np.isfinite(r["mae"])]
                    print(f"  w={w} {k+1}/{len(targets)} running MAE {np.mean(done):.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            # By target third, not only pooled: an adapter that holds the middle and
            # fails both ends posts a respectable pooled number, and the open-loop
            # diagnosis predicts improvement at the ENDS specifically.
            order = np.argsort([r["target"] for r in rows])
            thirds = [float(np.nanmean([rows[i]["mae"] for i in part]))
                      for part in np.array_split(order, 3)]
            finite = [r["mae"] for r in rows if np.isfinite(r["mae"])]
            per_w[str(w)] = {
                "mae": float(np.mean(finite)),
                "mae_by_third": {"low": thirds[0], "mid": thirds[1], "high": thirds[2]},
                "validity": float(np.mean([r["validity"] for r in rows])),
                "uniqueness": float(np.nanmean([r["uniqueness"] for r in rows])),
                "n_targets": len(rows),
            }
            print(f"E2 w={w}: MAE {per_w[str(w)]['mae']:.4f}  "
                  f"low {thirds[0]:.4f} mid {thirds[1]:.4f} high {thirds[2]:.4f}  "
                  f"val {per_w[str(w)]['validity']:.4f}", flush=True)
        best_w = min(per_w, key=lambda k: per_w[k]["mae"])
        metric_b = {"per_w": per_w, "best_w": float(best_w),
                    "best_mae": per_w[best_w]["mae"]}
        print(f"BEST w={best_w}: MAE {per_w[best_w]['mae']:.4f}", flush=True)

    out = {
        "adapter_ckpt": args.adapter_ckpt, "base": args.base,
        "property": args.property, "epoch": args.epoch, "split": args.split,
        "seed": args.seed,
        "sampling": {"steps": args.steps, "eta": args.eta, "omega": args.omega,
                     "time_distortion": args.time_distortion,
                     "blend_space": args.blend_space,
                     "size_mode": args.size_mode,
                     "size_model": args.size_model},
        "slope_grid": metric_a,
        "e2": metric_b,
        "elapsed_s": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
