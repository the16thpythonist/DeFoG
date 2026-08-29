#!/usr/bin/env python
"""
E2 joint targeting: TWO properties at once, via product-of-experts adapter composition.

Mirrors FreeGress's multi-property setting (arXiv 2312.17397, appendix Tables 1-2), so
the numbers are comparable to their published row.

THE JOINT METRIC IS THE MEAN, NOT THE SUM. Their appendix reports LogP MAE, QED MAE and
"Total MAE" per row, and the Total column is the unweighted mean of the two -- verified
against their DiGress rows, where (0.83+0.14)/2 = 0.49, (0.67+0.13)/2 = 0.40 and
(0.55+0.14)/2 = 0.35 reproduce the printed Totals exactly. This script reports both
components and the mean, so nobody has to take that reconstruction on trust. It also
reports the std-normalised mean, because averaging logP units with QED units lets the
wider-ranged property dominate an otherwise scale-free comparison.

WHAT THE APPENDIX ALSO SHOWS, and the main table hides: FreeGress's best JOINT row is
logP ~0.18 / QED ~0.06, both worse than its single-property bests of 0.16 / 0.04.
Joint targeting costs them accuracy on each property, so a fair reading of our joint
number compares it against their joint row, not their single-property one.

A SEPARATE SCRIPT, deliberately. scripts/e2_targeting.py is load-bearing for every
single-property number measured so far; bending it into a two-adapter shape would put
those at risk for no benefit. Shared pieces (target drawing, the property table, the
per-target size wrapper) are imported from it rather than copied.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/ssd2/Programming/defog-web")

import numpy as np  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from e2_targeting import PROP_FNS, _TargetedSize  # noqa: E402


def draw_joint_targets(split: str, n: int, seed: int, props):
    """`n` molecules from `split`, each carrying ALL requested property values.

    Drawn jointly rather than per property: the point of the joint setting is that the
    two targets come from the SAME molecule and are therefore realisable together. Two
    independent draws would ask for combinations the data may never exhibit, which
    measures something else entirely.
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
            vals = [float(PROP_FNS[p](m)) for p in props]
        except Exception:                                   # noqa: BLE001
            continue
        out.append((smi, vals))
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--properties", required=True,
                    help="Comma-separated, e.g. logp,qed")
    ap.add_argument("--adapter-ckpts", required=True,
                    help="Comma-separated raw AdaLNAdapter .ckpt paths, same order")
    ap.add_argument("--head-ckpts", default=None,
                    help="Comma-separated PropertyHead .ckpts (required for --method fk)")
    ap.add_argument("--size-models", default=None,
                    help="Comma-separated LearnedSizeDistribution .ckpts; composed as a "
                         "product of experts over node count")
    ap.add_argument("--weights", required=True,
                    help="Comma-separated per-branch guidance weights, same order")
    ap.add_argument("--split", required=True, choices=("validation", "test"))
    ap.add_argument("--method", required=True, choices=("adapter", "fk"))
    ap.add_argument("--composite-mode", default="product", choices=("product", "mean"))
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--per-target", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--blend-space", default="prob", choices=("prob", "rate"))
    ap.add_argument("--fk-beta", type=float, default=2.5)
    ap.add_argument("--fk-warmup", type=float, default=0.6)
    ap.add_argument("--fk-ess", type=float, default=0.5)
    ap.add_argument("--fk-jump", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    props = [p.strip() for p in args.properties.split(",")]
    ckpts = [p.strip() for p in args.adapter_ckpts.split(",")]
    weights = [float(w) for w in args.weights.split(",")]
    heads = [p.strip() for p in args.head_ckpts.split(",")] if args.head_ckpts else []
    sizes = [p.strip() for p in args.size_models.split(",")] if args.size_models else []
    n = len(props)
    if n < 2:
        sys.exit("--properties needs at least two; use e2_targeting.py for one")
    for name, lst in (("--adapter-ckpts", ckpts), ("--weights", weights)):
        if len(lst) != n:
            sys.exit(f"{name} has {len(lst)} entries but --properties has {n}")
    for p in props:
        if p not in PROP_FNS:
            sys.exit(f"unknown property {p!r}; have {sorted(PROP_FNS)}")
    if args.method == "fk" and len(heads) != n:
        sys.exit("--method fk needs one --head-ckpts entry per property: the FK energy is "
                 "a sum of per-property terms and a missing head makes one term silently "
                 "absent while the run is still labelled fk")
    if sizes and len(sizes) != n:
        sys.exit(f"--size-models has {len(sizes)} entries but --properties has {n}")

    from molsmith import sample as ms
    from defog.core import AdaLNAdapter
    from defog.core.adapter import _base_token

    # ---- load every adapter up front, so nothing is discovered mid-run ----------
    modules, keys = [], []
    for p, c in zip(props, ckpts):
        a = AdaLNAdapter.load(c, device="cpu")
        modules.append(a)
        keys.append(f"ckpt:{Path(c).resolve()}")
        cfg = a._config()
        print(f"adapter[{p}]: {c}\n  params={sum(x.numel() for x in a.parameters()):,} "
              f"fourier={cfg.get('cond_fourier')} xattn={cfg.get('xattn_tokens')}/"
              f"{cfg.get('xattn_dim')}/{cfg.get('xattn_heads')} "
              f"cond=({float(a.cond_mean.reshape(-1)[0]):.6f},"
              f"{float(a.cond_std.reshape(-1)[0]):.6f})")
        if float(a.cond_std.reshape(-1)[0]) == 1.0:
            sys.exit(f"REFUSING: {p} adapter has cond_std exactly 1.0, which collides with "
                     f"molsmith's 'unfilled' sentinel for AdapterTarget.scale.")
    if len(set(keys)) != n:
        sys.exit("REFUSING: two branches resolve to the same checkpoint path, so the "
                 "composition would stack an adapter with itself.")

    targets = draw_joint_targets(args.split, args.n_targets, args.seed, props)
    arr = np.array([v for _, v in targets])
    for i, p in enumerate(props):
        print(f"{len(targets)} joint targets from {args.split} (seed {args.seed}); "
              f"{p} range [{arr[:, i].min():.3f}, {arr[:, i].max():.3f}] "
              f"mean {arr[:, i].mean():.3f}")

    raw_kw = [dict(property=p,
                   scale=float(m.cond_std.reshape(-1)[0]),
                   mean=float(m.cond_mean.reshape(-1)[0]))
              for p, m in zip(props, modules)]

    size_models = []
    if sizes:
        from defog.core import LearnedSizeDistribution
        for p, sm, mod in zip(props, sizes, modules):
            m = LearnedSizeDistribution.load(sm)
            # Refuses a size model fit under a different label convention than the
            # adapter it is paired with -- they agree everywhere except the extremes,
            # which is exactly where targeting is hardest and nobody is looking.
            m.check_compatible(mod)
            size_models.append(m)
            print(f"size model[{p}]: {sm}  grid {m.min_size}..{m.max_size}")

    def _compose_sizes(vals):
        """Product-of-experts over each property's P(n | target).

        Both adapters have an opinion about how many atoms the molecule needs and they
        will not agree; composing them is the size-draw counterpart of composing the
        conditioning itself, and uses the same mode so the two halves cannot disagree
        about what 'product' means.
        """
        from defog.core.size_distribution import ComposedSizeDistribution, SizeBranch
        return ComposedSizeDistribution(
            [SizeBranch(dist=m, condition=float(v), weight=1.0)
             for m, v in zip(size_models, vals)], mode=args.composite_mode)

    def cfg_for(vals, seed):
        c = ms.SamplingConfig(
            base=args.base, n=args.per_target, seed=seed, steps=args.steps,
            eta=args.eta, omega=args.omega, time_distortion=args.time_distortion,
            adapters=[ms.AdapterTarget(package=k, target=float(v), weight=w, **kw)
                      for k, v, w, kw in zip(keys, vals, weights, raw_kw)],
            blend_space=args.blend_space, composite_mode=args.composite_mode,
            method="fk" if args.method == "fk" else "none")
        if args.method == "fk":
            c.fk = ms.FeynmanKac(beta=args.fk_beta, warmup_frac=args.fk_warmup,
                                 ess_frac=args.fk_ess, jump_length=args.fk_jump)
        if size_models:
            c.size_dist = _compose_sizes(vals)
        return c

    probe = cfg_for(targets[0][1], args.seed)
    held, probe.adapters = probe.adapters, []
    loaded = ms.load(probe)
    probe.adapters = held
    dev = next(loaded.base.parameters()).device
    tok = _base_token(loaded.base)
    for p, m, k, c in zip(props, modules, keys, ckpts):
        m.to(dev).eval()
        m.check_compatible(loaded.base)
        if m.base_token is None or abs(tok - m.base_token) > 1e-6 * (1 + abs(m.base_token)):
            sys.exit(f"REFUSING: {c} was trained on a different base "
                     f"(token {m.base_token} != {tok}).")
        loaded.adapters[k] = m
        loaded.heads[k] = None
        loaded.size_models[k] = None
    print(f"base check: OK (token {tok:.8g}); {n} adapters injected")

    if args.method == "fk":
        from defog.core.property_head import PropertyHead
        for p, hc, k, m in zip(props, heads, keys, modules):
            h = PropertyHead.load(hc, device="cpu").to(dev).eval()
            hm, hs = float(h.prop_mean), float(h.prop_std)
            am = float(m.cond_mean.reshape(-1)[0]); asd = float(m.cond_std.reshape(-1)[0])
            if abs(hm - am) > 1e-3 * max(abs(am), 1.0) or abs(hs - asd) > 1e-3 * max(asd, 1.0):
                sys.exit(f"REFUSING: {p} head normalisation ({hm:.6f},{hs:.6f}) != adapter "
                         f"({am:.6f},{asd:.6f}); the energy would score a different value "
                         f"than the one being steered to.")
            loaded.heads[k] = h
            print(f"FK head[{p}]: {hc}  ({hm:.6f}, {hs:.6f}) matches the adapter")

    # The composition must actually be built with N branches; molsmith falling back to a
    # single branch or to none would still produce plausible molecules.
    import defog.core as _dc
    probe_state = {"n": None}
    _real = _dc.AdapterComposition

    def _probe_ctor(*a, **kw):
        br = a[0] if a else kw.get("branches", [])
        probe_state["n"] = len(br)
        return _real(*a, **kw)

    _dc.AdapterComposition = _probe_ctor

    rows, per_prop_err, t0 = [], [[] for _ in props], time.time()
    for k_i, (smi, vals) in enumerate(targets):
        res = ms.sample(cfg_for(vals, args.seed + k_i), loaded)
        if k_i == 0:
            _dc.AdapterComposition = _real
            if probe_state["n"] != n:
                sys.exit(f"REFUSING: the composition was built with {probe_state['n']} "
                         f"branches, not {n}. The run would be steered by fewer properties "
                         f"than it claims.")
            print(f"wiring check: AdapterComposition received all {n} branches", flush=True)
        smis = [s for s in res.smiles if s]
        ach, ok = [], []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            try:
                ach.append([float(PROP_FNS[p](m)) for p in props])
                ok.append(s)
            except Exception:                               # noqa: BLE001
                continue
        A = np.array(ach) if ach else np.zeros((0, n))
        row = {"target_smiles": smi, "targets": vals, "n_requested": args.per_target,
               "n_valid": len(ok), "validity": len(ok) / args.per_target,
               "uniqueness": (len(set(ok)) / len(ok)) if ok else float("nan")}
        for i, p in enumerate(props):
            e = np.abs(A[:, i] - vals[i]) if A.size else np.array([])
            per_prop_err[i].extend(e.tolist())
            row[f"mae_{p}"] = float(e.mean()) if e.size else float("nan")
        row["mae_total"] = float(np.mean([row[f"mae_{p}"] for p in props]))
        rows.append(row)
        if (k_i + 1) % 10 == 0:
            fin = [r["mae_total"] for r in rows if math.isfinite(r["mae_total"])]
            print(f"  {k_i+1}/{len(targets)}  running Total MAE "
                  f"{np.mean(fin):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    pooled = {p: (float(np.mean(e)) if e else float("nan"))
              for p, e in zip(props, per_prop_err)}
    total = float(np.mean(list(pooled.values())))
    stds = {p: float(arr[:, i].std()) for i, p in enumerate(props)}
    total_norm = float(np.mean([pooled[p] / stds[p] for p in props]))
    val = float(np.mean([r["validity"] for r in rows]))
    uq = [r["uniqueness"] for r in rows if math.isfinite(r["uniqueness"])]
    summary = {
        "properties": props, "adapters": ckpts, "heads": heads, "size_models": sizes,
        "base": args.base, "split": args.split, "method": args.method, "seed": args.seed,
        "weights": weights, "composite_mode": args.composite_mode,
        "n_targets": len(rows), "per_target": args.per_target,
        "sampling": {"steps": args.steps, "eta": args.eta,
                     "blend_space": args.blend_space},
        "mae_per_property": pooled, "target_std": stds,
        "mae_total": total, "mae_total_normalised": total_norm,
        "validity": val,
        "uniqueness": float(np.mean(uq)) if uq else float("nan"),
        "per_target": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print()
    print(f"=== E2 JOINT {'+'.join(props)} / {args.method} / {args.split} ===")
    for p in props:
        print(f"  {p:9s} MAE {pooled[p]:.4f}   (target std {stds[p]:.4f} -> "
              f"{pooled[p]/stds[p]:.3f} std)")
    print(f"  Total MAE (mean, FreeGress convention) {total:.4f}")
    print(f"  Total MAE (std-normalised)             {total_norm:.4f}")
    print(f"  validity {val:.4f}   uniqueness {summary['uniqueness']:.4f}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
