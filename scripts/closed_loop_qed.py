#!/usr/bin/env python
"""Wave 4a: a hand-coded CLOSED-LOOP controller over the QED adapter.

The open-loop diagnosis says the adapter emits the same push regardless of what it has
already built: its modulation is a function of (condition, time) only, never of the graph.
This tests that with no training at all -- read the property off the partially-built
molecule each step and adjust the guidance weight by the error:

    w_t = clip( w0 + k * (target - reading) / sigma , 0, W_MAX )

`sigma` is the property's own std (from the adapter's cond_std), so the gain `k` is in
units of "standard deviations of error" and is not silently mis-scaled by QED's narrow
range -- a gain tuned on logP (sigma ~1.16) would be ~9x too strong here (sigma 0.133).

WHAT THE READING IS. The head must see a DISCRETE, in-distribution graph. Feeding it the
noisy current state, or soft marginals, puts it far outside what it was fitted on -- that
is why an earlier soft-input self-consistency attempt failed, and why Feynman-Kac works:
it scores `predict_clean`'s argmax'd clean-graph prediction. This uses the same primitive.

PER-MOLECULE CONTROL. Each molecule carries its own error, so each gets its own weight;
`_blend_logp` accepts (N, bs) weights for exactly this. Averaging to one weight per batch
would smooth away the effect under test.

Warmup: before `warmup_frac` of the trajectory the clean-graph prediction is mostly noise,
so the controller is disabled and w stays at w0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from rdkit import Chem, RDLogger                            # noqa: E402
from rdkit.Chem import QED                                  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def qed_of(mol):
    return float(QED.qed(mol))



@torch.no_grad()
def _branch_marginals(model, comp, X, E, y, t, node_mask):
    """One batched forward giving BOTH the unconditional and conditional clean-graph
    marginals, mirroring what denoise_step does internally.

    Both controllers need something from this pass and neither needs its own: the
    task-error law wants the conditional MAP graph to read the property off, and the
    CFG-Ctrl law wants the cond-vs-uncond discrepancy. Sharing one forward keeps the two
    variants at the same cost, so a timing difference cannot masquerade as a quality one.
    """
    bs = X.size(0)
    rep = len(comp) + 1
    noisy = {"X_t": X, "E_t": E, "y_t": y, "t": t, "node_mask": node_mask}
    extra = model._compute_extra_data(noisy)
    from defog.core.model import PlaceHolder
    nd = {"X_t": X.repeat(rep, 1, 1), "E_t": E.repeat(rep, 1, 1, 1),
          "y_t": y.repeat(rep, 1), "t": t.repeat(rep, 1),
          "node_mask": node_mask.repeat(rep, 1)}
    eb = PlaceHolder(X=extra.X.repeat(rep, 1, 1), E=extra.E.repeat(rep, 1, 1, 1),
                     y=extra.y.repeat(rep, 1))
    mod = comp.build_modulation(bs, t)
    pred = model.forward(nd, eb, nd["node_mask"], cond_modulation=mod)
    pX = F.softmax(pred.X, dim=-1).view(rep, bs, *pred.X.shape[1:])
    pE = F.softmax(pred.E, dim=-1).view(rep, bs, *pred.E.shape[1:])
    X1 = F.one_hot(pX[1].argmax(-1), pX.size(-1)).float()
    E1 = F.one_hot(pE[1].argmax(-1), pE.size(-1)).float()
    # scalar cond-vs-uncond discrepancy per molecule: mean |log p_cond - log p_unc|
    m = node_mask.float().unsqueeze(-1)
    d = ((pX[1] + 1e-8).log() - (pX[0] + 1e-8).log()).abs() * m
    disc = d.sum((1, 2)) / m.sum((1, 2)).clamp_min(1.0)
    return X1, E1, disc


@torch.no_grad()
def sample_closed_loop(model, adapter, head, target, n, *, steps, eta, omega,
                       time_distortion, w0, gain, sigma, warmup_frac, w_max,
                       size_dist, device, seed, controller='p', lam=1.0,
                       w_post=None):
    """One batch of `n` molecules toward `target`, with per-molecule adaptive w.

    gain == 0 reproduces fixed-w sampling exactly (the control arm), so both arms run
    through identical code and differ only in the number they multiply the error by.
    """
    from defog.core import AdapterComposition, ConditionBranch
    from defog.core.feynman_kac import predict_clean
    from defog.core.noise import sample_noise

    torch.manual_seed(seed)
    cond = torch.full((n, adapter.cond_dim), float(target), device=device)
    comp = AdapterComposition([ConditionBranch(adapter, cond, w0)], base=model,
                              blend_space="prob")

    n_nodes = size_dist.sample(n, device=device).clamp(1, model.max_nodes)
    n_max = int(n_nodes.max())
    node_mask = torch.arange(n_max, device=device)[None, :] < n_nodes[:, None]
    z = sample_noise(model.limit_dist, node_mask)
    X, E, y = z.X.to(device), z.E.to(device), torch.zeros(n, 0, device=device)

    ts = torch.linspace(0, 1, steps + 1, device=device)
    w_trace, sat, spread = [], [], []
    prev_e = None
    disc_ema = None
    for i in range(steps):
        t = ts[i].repeat(n, 1)
        s = ts[i + 1].repeat(n, 1)
        if gain != 0.0 and float(ts[i]) >= warmup_frac:
            X1, E1, disc = _branch_marginals(model, comp, X, E, y, t, node_mask)
            if controller == "smc_cfg":
                # CFG-Ctrl as published: the error is the cond-vs-uncond DISCREPANCY,
                # no property head involved. s = edot + lam*e; a growing discrepancy
                # means the guidance is running away, so back off.
                #
                # The discrepancy itself is a MAGNITUDE and so always positive, which
                # pins sign(s) at +1 and turns the switching law into a constant -- the
                # first smoke test showed exactly that (w_spread 0.00). CFG-Ctrl's own
                # error is a signed velocity difference, so the signed analogue here is
                # the deviation from the trajectory's running level: is the discrepancy
                # above or below where it has been?
                ema = disc if disc_ema is None else 0.9 * disc_ema + 0.1 * disc
                err = disc - ema
                disc_ema = ema
            else:
                reading = head.predict(X1, E1, node_mask).reshape(-1)   # un-normalised QED
                err = (torch.as_tensor(float(target), device=device) - reading) / sigma
            if controller == "p":
                w = (w0 + gain * err).clamp(0.0, w_max)
            else:
                edot = err - prev_e if prev_e is not None else torch.zeros_like(err)
                surf = edot + lam * err
                sgn = torch.sign(surf) if controller == "smc_task" else -torch.sign(surf)
                w = (w0 + gain * sgn).clamp(0.0, w_max)
            prev_e = err
        elif w_post is not None and float(ts[i]) >= warmup_frac:
            # Two-phase FIXED control: w0 through warmup, then a constant. This exists to
            # match a switching arm's time profile exactly -- a flat w equal to the
            # switching arm's overall mean is NOT the same trajectory, because half of
            # that mean is warmup spent at w0.
            w = torch.full((n,), float(w_post), device=device)
        else:
            w = torch.full((n,), float(w0), device=device)
        w_trace.append(float(w.mean()))
        if gain != 0.0 and float(ts[i]) >= warmup_frac:
            # A controller pinned at its clip is not a controller: it is a constant.
            # Track how often that happens and how much w actually varies, so a slope
            # gain cannot be mistaken for feedback when it is really just a bigger push.
            sat.append(float(((w <= 1e-6) | (w >= w_max - 1e-6)).float().mean()))
            spread.append(float(w.std()))
        comp.set_weights([w])                       # (bs,) -> per-molecule for this branch
        X, E, y = model.denoise_step(t, s, X, E, y, node_mask, eta=eta, omega=omega,
                                     composition=comp)
    return X, E, node_mask, w_trace, sat, spread


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--adapter", default="molsmith/qed@3.1.0")
    ap.add_argument("--w0", type=float, default=1.0)
    ap.add_argument("--controller", default="p",
                    choices=("p", "smc_task", "smc_cfg"),
                    help="p: w = w0 + k*e.  smc_task: sliding mode on the TASK error.  "
                         "smc_cfg: CFG-Ctrl as published, on the cond-uncond discrepancy.")
    ap.add_argument("--lam", type=float, default=1.0, help="sliding-surface lambda")
    ap.add_argument("--w-post", type=float, default=None,
                    help="with --gain 0: constant weight AFTER warmup (two-phase fixed "
                         "control, time-matched to a switching arm)")
    ap.add_argument("--gain", type=float, required=True,
                    help="0 = fixed-w control arm; else P-gain in error-std units")
    ap.add_argument("--w-max", type=float, default=3.0)
    ap.add_argument("--warmup-frac", type=float, default=0.5)
    ap.add_argument("--n-per-level", type=int, default=128)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from molsmith import sample as ms
    from defog.core import EmpiricalSizeDistribution
    from defog.data import zinc_reference as zref

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ms.SamplingConfig(base=args.base, n=1,
                            adapters=[ms.AdapterTarget(package=args.adapter, target=0.7)])
    loaded = ms.load(cfg, device=device)
    model = loaded.base.to(device).eval()
    adapter = loaded.adapters[args.adapter]
    head = loaded.heads.get(args.adapter)
    if head is None:
        sys.exit(f"REFUSING: {args.adapter} bundles no property head, so the controller "
                 f"has nothing to read the current QED off. Closed loop is impossible.")
    head = head.to(device).eval()
    sigma = float(adapter.cond_std.reshape(-1)[0])
    print(f"adapter cond_mean {float(adapter.cond_mean.reshape(-1)[0]):.4f} sigma {sigma:.4f}")
    print(f"w0={args.w0} gain={args.gain} w_max={args.w_max} warmup={args.warmup_frac}")

    # same 5 levels as the capacity ladder / oracle
    smis = zref.load_reference_split().val_smiles[:5000]
    vals = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            vals.append(qed_of(m))
    levels = {f"p{p:02d}": float(np.percentile(vals, p)) for p in (5, 25, 50, 75, 95)}
    print(f"targets: { {k: round(v,4) for k,v in levels.items()} }")

    size_dist = EmpiricalSizeDistribution(model.node_dist.prob)
    out = {"adapter": args.adapter, "controller": args.controller, "lam": args.lam,
           "w_post": args.w_post,
           "w0": args.w0, "gain": args.gain,
           "w_max": args.w_max, "warmup_frac": args.warmup_frac, "sigma": sigma,
           "seed": args.seed, "steps": args.steps, "targets": levels, "per_level": {}}

    from defog.domains.molecule import mol_to_smiles, pyg_data_to_mol
    from defog.core.data import dense_to_pyg as to_pyg
    dom = loaded.domain
    t0 = time.time()
    for li, (name, tgt) in enumerate(levels.items()):
        X, E, mask, wtrace, sat, spread = sample_closed_loop(
            model, adapter, head, tgt, args.n_per_level, steps=args.steps, eta=args.eta,
            omega=args.omega, time_distortion=args.time_distortion, w0=args.w0,
            gain=args.gain, sigma=sigma, warmup_frac=args.warmup_frac, w_max=args.w_max,
            size_dist=size_dist, device=device, seed=args.seed + li,
            controller=args.controller, lam=args.lam, w_post=args.w_post)
        # Same decode path the training experiment uses (props_of): dense -> PyG ->
        # RDKit -> canonical SMILES -> re-parse, so "valid" means the same thing here
        # as everywhere else in this project.
        got = []
        for g in to_pyg(X, E, None, mask):
            m = pyg_data_to_mol(g, dom.atom_decoder, dom.bond_decoder)
            smi = mol_to_smiles(m) if m is not None else None
            if smi and Chem.MolFromSmiles(smi) is not None:
                try:
                    got.append(qed_of(m))
                except Exception:                            # noqa: BLE001
                    pass
        a = np.array(got)
        out["per_level"][name] = {
            "target": tgt, "n_valid": int(a.size),
            "validity": a.size / args.n_per_level,
            "achieved_mean": float(a.mean()) if a.size else float("nan"),
            "achieved_sd": float(a.std()) if a.size else float("nan"),
            "mae": float(np.abs(a - tgt).mean()) if a.size else float("nan"),
            "w_mean": float(np.mean(wtrace)), "w_final": float(wtrace[-1]),
            "w_saturated_frac": float(np.mean(sat)) if sat else 0.0,
            "w_spread": float(np.mean(spread)) if spread else 0.0,
        }
        r = out["per_level"][name]
        print(f"  {name} target={tgt:.3f}: achieved={r['achieved_mean']:.3f} "
              f"mae={r['mae']:.4f} valid={r['validity']:.3f} w_mean={r['w_mean']:.2f} "
              f"sat={r['w_saturated_frac']:.2f} wsd={r['w_spread']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    t = np.array([v["target"] for v in out["per_level"].values()])
    a = np.array([v["achieved_mean"] for v in out["per_level"].values()])
    ok = np.isfinite(a)
    out["slope"] = float(np.polyfit(t[ok], a[ok], 1)[0]) if ok.sum() > 1 else float("nan")
    out["mean_mae"] = float(np.nanmean([v["mae"] for v in out["per_level"].values()]))
    out["mean_validity"] = float(np.mean([v["validity"] for v in out["per_level"].values()]))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print()
    print(f"=== CLOSED LOOP QED  controller={args.controller} gain={args.gain} lam={args.lam} ===")
    print(f"  SLOPE {out['slope']:.4f}   mean MAE {out['mean_mae']:.4f}   "
          f"validity {out['mean_validity']:.4f}")
    print(f"  (frozen adapter reference from the capacity ladder: slope 0.369)")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
