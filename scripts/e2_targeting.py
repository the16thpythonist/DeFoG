#!/usr/bin/env python
"""
E2: property targeting under FreeGress's protocol.

Protocol (docs/targeting-protocol.md §1, matching FreeGress Tab. 2 on ZINC250k):

  1. take 100 molecules from the dataset; each molecule's own measured property
     value becomes a target y_i
  2. generate 10 molecules per target -> 1000 generated molecules
  3. measure the property on every generated molecule with RDKit
  4. MAE = mean |y_i - yhat_ij| over all 1000
  5. report chemical validity in the same row

WHY 10 PER TARGET AND NOT 1-OF-K
Under Feynman-Kac the ten come from ONE particle system of K=10, and all ten are
kept. It is tempting to instead run ten systems of K=8 and keep the best particle
from each -- the outputs would be independent, which superficially matches the
baseline better. It would also be best-of-8 selection: an 8x compute advantage
over a method that simply draws ten times. Keeping all ten spends the same budget
the baseline spends.

The cost is that resampling COUPLES the ten: it culls low-weight particles and
duplicates high-weight ones, so a badly tuned FK run can return ten copies of one
molecule and post an excellent MAE. That is why uniqueness is reported beside MAE
here rather than left to a follow-up, and why FK's warmup_frac / ess_frac / beta
are the knobs to tune -- they control how hard and how early the system collapses.

SPLIT DISCIPLINE. --split validation for anything that informs a choice; --split
test exactly once, with the configuration already frozen. The 100 targets are
drawn with an explicit --seed, and both split and seed are recorded in the output
so the caption can state them, as the protocol requires.

Usage:
    python scripts/e2_targeting.py --adapter molsmith/clogp@1.2.0 --property logp \\
        --split validation --method adapter --weight 1.0 --out e2_val_adapter.json
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


def draw_targets(split: str, n: int, seed: int, prop_fn):
    """The 100 target values, from real molecules in `split`.

    Targets are the molecules' OWN measured property, not quantiles of the
    distribution -- that is the difference between this and the percentile mode the
    training experiment uses, and it is what makes the number comparable to
    FreeGress's.
    """
    from defog.data import zinc_reference as zref
    s = zref.load_reference_split()
    pool = {"validation": s.val_smiles, "test": s.test_smiles}[split]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))
    out = []
    for i in idx:
        smi = pool[int(i)]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            out.append((smi, float(prop_fn(m))))
        except Exception:                                   # noqa: BLE001
            continue
        if len(out) >= n:
            break
    return out


class _TargetedSize:
    """A :class:`SizeDistribution` that pins one target value into a size model.

    ``molsmith`` hands ``size_dist.sample`` whatever condition the *adapter* pipeline
    computed, which is not the raw property value the size model wants. This wrapper
    carries the target instead, so the two cannot drift apart.
    """

    def __init__(self, model, target: float, n: int):
        import torch
        self.model = model
        self.condition = torch.full((n, model.cond_dim), float(target))

    def sample(self, num_samples, condition=None, device=None, generator=None):
        c = self.condition[:num_samples] if self.condition.size(0) >= num_samples else \
            self.condition[:1].expand(num_samples, -1)
        return self.model.sample(num_samples, condition=c, device=device,
                                 generator=generator)

    @property
    def max_size(self) -> int:
        return self.model.max_size

    def log_prob(self, sizes, condition=None):
        return self.model.log_prob(sizes, condition=self.condition[:sizes.numel()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--adapter", required=True, help="store ref, e.g. molsmith/clogp@1.2.0")
    ap.add_argument("--property", required=True, choices=sorted(PROP_FNS))
    ap.add_argument("--split", required=True, choices=("validation", "test"),
                    help="validation for anything that informs a choice; test once, frozen")
    ap.add_argument("--method", required=True, choices=("adapter", "fk"))
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--per-target", type=int, default=10)
    ap.add_argument("--weight", type=float, default=2.0, help="adapter guidance weight")
    # Kept reachable, not merely documented: every E2 number before 2026-08-17 was
    # measured in rate space, and re-deriving one means being able to ask for it.
    ap.add_argument("--blend-space", default="prob", choices=("prob", "rate"),
                    help="where CFG is applied: 'prob' blends clean-graph marginals "
                         "(FreeGress Eq. 10/11, the default), 'rate' blends rate matrices "
                         "(historical; breaks down above w=1)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec")
    # FK knobs. These carry two jobs at once: pull toward the target, and avoid
    # collapsing the ten particles into copies of one molecule.
    ap.add_argument("--fk-beta", type=float, default=2.5)
    ap.add_argument("--fk-warmup", type=float, default=0.6)
    ap.add_argument("--fk-ess", type=float, default=0.5,
                    help="resample only when effective sample size < ess*K; lower "
                         "means less culling and more surviving diversity")
    ap.add_argument("--fk-rejuvenate", action="store_true",
                    help="MCMC moves after each resample -- the standard SMC remedy "
                         "for particle impoverishment. Regenerates duplicated "
                         "particles rather than leaving copies, which is the failure "
                         "the uniqueness column exists to catch.")
    ap.add_argument("--fk-jump", type=int, default=10)
    # How many nodes each generated graph gets. DEFAULT IS `marginal`, which is what every
    # E2 number before this flag existed used -- so those numbers stay reproducible.
    #
    # This is an ABLATION AXIS, not a free improvement. FreeGress Tab. 3 shows conditioned
    # node inference alone moving MW MAE by -70%, i.e. capable of dominating the column it
    # appears in. Folded silently into one row it reads as "the adapter got better". Run
    # the pair.
    ap.add_argument("--size-mode", default="marginal",
                    choices=("marginal", "learned"),
                    help="marginal: the base's P(n) (the historical default). "
                         "learned: P(n|target) from --size-model.")
    ap.add_argument("--size-model", default=None,
                    help="Path to a LearnedSizeDistribution ckpt (--size-mode learned)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.size_mode == "learned" and not args.size_model:
        sys.exit("--size-mode learned needs --size-model PATH")

    from molsmith import sample as ms

    prop_fn = PROP_FNS[args.property]
    targets = draw_targets(args.split, args.n_targets, args.seed, prop_fn)
    tvals = np.array([v for _, v in targets])
    print(f"{len(targets)} targets from {args.split} (seed {args.seed}); "
          f"{args.property} range [{tvals.min():.3f}, {tvals.max():.3f}] "
          f"mean {tvals.mean():.3f}")

    size_model = None
    if args.size_mode == "learned":
        from defog.core import LearnedSizeDistribution
        size_model = LearnedSizeDistribution.load(args.size_model)
        print(f"size model: {args.size_model}  grid {size_model.min_size}.."
              f"{size_model.max_size}  property={size_model.property_name!r} "
              f"from={size_model.property_from!r}")

    # GUARD: SamplingConfig is a plain dataclass, so `config.size_dist = ...` silently
    # succeeds even against a molsmith build that never reads it -- the run then uses
    # the marginal while this script's own JSON records "learned". That happened once;
    # it is not allowed to happen quietly again.
    if size_model is not None and "size_dist" not in ms.SamplingConfig.__dataclass_fields__:
        sys.exit(
            "REFUSING: this molsmith build has no SamplingConfig.size_dist field, so a "
            "learned size distribution would be silently ignored and the results would "
            "be mislabelled as 'learned'. Update molsmith before re-running."
        )
    # The same failure mode, for the blend space: asking for "prob" against a molsmith
    # that predates the field would sample in rate space and record "prob".
    if "blend_space" not in ms.SamplingConfig.__dataclass_fields__:
        sys.exit(
            f"REFUSING: this molsmith build has no SamplingConfig.blend_space field, so "
            f"--blend-space {args.blend_space!r} would be ignored and the output would be "
            f"mislabelled. Update molsmith before re-running."
        )

    def cfg_for(target: float, seed: int):
        c = ms.SamplingConfig(
            base=args.base, n=args.per_target, seed=seed, steps=args.steps,
            eta=args.eta, omega=args.omega, time_distortion=args.time_distortion,
            adapters=[ms.AdapterTarget(package=args.adapter, target=target,
                                       weight=args.weight)],
            blend_space=args.blend_space,
            method="fk" if args.method == "fk" else "none")
        if size_model is not None:
            # A ready-made SizeDistribution, bypassing size_mode. The condition rides in
            # the branch here rather than through SamplingConfig, because the target is
            # per-call and the model is not.
            c.size_dist = _TargetedSize(size_model, target, args.per_target)
        if args.method == "fk":
            c.fk = ms.FeynmanKac(beta=args.fk_beta, warmup_frac=args.fk_warmup,
                                 ess_frac=args.fk_ess,
                                 rejuvenate=args.fk_rejuvenate,
                                 jump_length=args.fk_jump)
        return c

    loaded = ms.load(cfg_for(float(tvals[0]), args.seed))
    if args.method == "fk":
        h = loaded.heads.get(args.adapter)
        if h is None:
            sys.exit(f"REFUSING: --method fk but {args.adapter} bundles no property head. "
                     f"LearnedPropertyEnergy needs one; without it FK has nothing to score "
                     f"and would silently reduce to plain adapter sampling.")
        print(f"FK energy: head from {args.adapter}  "
              f"beta={args.fk_beta} warmup={args.fk_warmup} ess={args.fk_ess} "
              f"rejuvenate={args.fk_rejuvenate}")

    rows, all_err, t0 = [], [], time.time()
    for k, (smi, tgt) in enumerate(targets):
        cfg = cfg_for(tgt, args.seed + k)          # a different draw per target
        res = ms.sample(cfg, loaded)
        smis = [s for s in res.smiles if s]
        achieved, ok = [], []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            try:
                achieved.append(float(prop_fn(m)))
                ok.append(s)
            except Exception:                               # noqa: BLE001
                continue
        a = np.array(achieved)
        err = np.abs(a - tgt) if a.size else np.array([])
        all_err.extend(err.tolist())
        rows.append({
            "target_smiles": smi, "target": tgt,
            "n_requested": args.per_target, "n_valid": len(ok),
            # validity is over what was ASKED for, not over what parsed
            "validity": len(ok) / args.per_target,
            # uniqueness catches FK particle collapse: ten copies of one molecule
            # post an excellent MAE and are worthless
            "uniqueness": (len(set(ok)) / len(ok)) if ok else float("nan"),
            "achieved_mean": float(a.mean()) if a.size else float("nan"),
            "achieved_sd": float(a.std()) if a.size else float("nan"),
            "mae": float(err.mean()) if err.size else float("nan"),
        })
        if (k + 1) % 10 == 0:
            done = np.array([r["mae"] for r in rows if np.isfinite(r["mae"])])
            print(f"  {k+1}/{len(targets)} targets  running MAE {done.mean():.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    finite = np.array([r["mae"] for r in rows if np.isfinite(r["mae"])])
    val = np.array([r["validity"] for r in rows])
    uniq = np.array([r["uniqueness"] for r in rows if np.isfinite(r["uniqueness"])])
    # MAE across the RANGE, not only pooled: the protocol asks for it, and a model
    # that nails mid-range while failing the ends posts a fine pooled number.
    order = np.argsort([r["target"] for r in rows])
    thirds = np.array_split(order, 3)
    by_third = [float(np.nanmean([rows[i]["mae"] for i in part])) for part in thirds]

    summary = {
        "adapter": args.adapter, "base": args.base, "property": args.property,
        "split": args.split, "method": args.method, "seed": args.seed,
        "n_targets": len(rows), "per_target": args.per_target,
        "sampling": {"weight": args.weight, "steps": args.steps, "eta": args.eta,
                     "omega": args.omega, "time_distortion": args.time_distortion,
                     "blend_space": args.blend_space},
        # Recorded so a pair of runs can be read as an ablation without anyone having to
        # remember which was which.
        "size": {"mode": args.size_mode, "model": args.size_model,
                 "grid": ([size_model.min_size, size_model.max_size]
                          if size_model is not None else None)},
        "fk": ({"beta": args.fk_beta, "warmup_frac": args.fk_warmup,
                "ess_frac": args.fk_ess, "rejuvenate": args.fk_rejuvenate,
                "jump_length": args.fk_jump} if args.method == "fk" else None),
        "mae_pooled": float(np.mean(all_err)) if all_err else float("nan"),
        "mae_per_target_mean": float(finite.mean()),
        "mae_low_third": by_third[0], "mae_mid_third": by_third[1],
        "mae_high_third": by_third[2],
        "validity": float(val.mean()),
        "uniqueness": float(uniq.mean()) if uniq.size else float("nan"),
        "per_target": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print()
    print(f"=== E2 {args.property} / {args.method} / {args.split} / "
          f"size={args.size_mode} ===")
    print(f"  MAE (pooled over {len(all_err)} molecules) {summary['mae_pooled']:.4f}")
    print(f"  MAE by target third   low {by_third[0]:.4f}  mid {by_third[1]:.4f}  "
          f"high {by_third[2]:.4f}")
    print(f"  validity   {summary['validity']:.4f}")
    print(f"  uniqueness {summary['uniqueness']:.4f}"
          + ("   <- FK collapse check; well below 1.0 means duplicated particles"
             if args.method == "fk" else ""))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
