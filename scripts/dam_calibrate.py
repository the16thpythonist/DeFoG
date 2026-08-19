#!/usr/bin/env python
"""
Step 8.5 of docs/dam_design.md: calibrate DAM on the REAL base before spending
cluster time.

The tabular gate in tests/test_dam_estimator.py pins the estimator's algebra, and it
is blind by construction to the three things most likely to kill DAM on a real model:

  * the PROJECTION GAP  -- the tabular policy is unconstrained; DeFoG's head is not,
                           so DAM's ideal target rate may be unreachable;
  * the HEAD SURROGATE  -- there is no network in the fixture, so the substitution of
                           a one-shot clean prediction for the CTMC's terminal law is
                           invisible there;
  * MIS-TEMPERING       -- the fixture's reward span is a free choice; the real one is
                           not.

This script measures the two diagnostics that can see them. Both are cheap.

(a) THE y = x CONTROL. When the jump target equals the current state the true adjoint
    is exactly 1.0, whatever the reward is doing. So E[a_hat] there is a pure readout
    of estimator bias, and it is the only signal that separates "lambda is too hot"
    from "the reward is working". Reported across lambda and K.

(b) THE PROJECTION GAP. Take the base rate, tilt it by a random directional factor,
    and ask how much of that tilt the head can actually express: fit the clean-graph
    simplex to the tilted target and report residual gKL / no-op gKL. 0 means the
    family reaches the target exactly; 1 means it cannot move at all.

Usage:
    python scripts/dam_calibrate.py [--device cpu|cuda] [--graphs 8] [--reps 32]
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defog.core import DeFoGModel                                    # noqa: E402
from defog.core.dam import (discrete_adjoint, estimate_neg_value,    # noqa: E402
                            gkl, marginal_rate, rate_basis)
from defog.core.noise import sample_from_probs                       # noqa: E402
from defog.core.renoise import renoise_states                        # noqa: E402
from defog.core.rl import _score_logprob                             # noqa: E402


DEFAULT_BASE = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")


# --------------------------------------------------------------------------- reward
_PROJ = {}


def _proj(key, numel, device):
    """A fixed random projection, cached, so the stand-in reward is DETERMINISTIC in
    the graph -- like a real reward -- rather than in the call."""
    if key not in _PROJ:
        g = torch.Generator().manual_seed(1234)
        _PROJ[key] = torch.randn(numel, generator=g)
    return _PROJ[key].to(device)


def tiered_reward(X1, E1, node_mask, invalid_frac=0.10, disc_frac=0.15):
    """A stand-in with PropertyMatchReward's SPAN and tiering -- [-10, 0], a ~10%
    invalid floor at -10, a ~15% disconnected tier at -4, the rest graded in [-3, 0].

    The span is what fixes the temperature, and the real reward fixes it: rl.py:835
    sets invalid=-10 and disconnect=-4, rl.py:859 gives -min(|dp|/scale, 3.0). Using
    RDKit here would make the script depend on decoding and add nothing.

    It MUST vary across samples. An earlier version keyed the tiers off a per-call RNG
    instead of the graph, which made g(Z) == g(X1_k) identically and collapsed the
    adjoint to exactly 1.0 -- a control that always passes. The self-check in main()
    exists so that cannot recur silently.
    """
    B = X1.shape[0]
    em = (node_mask[:, :, None] & node_mask[:, None, :]).float()
    fx = (X1 * node_mask[..., None]).reshape(B, -1)
    fe = (E1 * em[..., None]).reshape(B, -1)
    scale = math.sqrt(max(int(node_mask.sum(-1).float().mean()), 1))
    s1 = (fx @ _proj("x1", fx.shape[1], X1.device)
          + fe @ _proj("e1", fe.shape[1], X1.device)) / scale
    s2 = (fx @ _proj("x2", fx.shape[1], X1.device)
          + fe @ _proj("e2", fe.shape[1], X1.device)) / scale
    # Quantile thresholds so the tier fractions are what we ASK for, not whatever the
    # projection happens to produce. The invalid fraction is the knob that matters:
    # section 4.3 measures the head's one-shot clean predictions at 94-97% invalid at
    # low t, and that is precisely what drives the adjoint's tail.
    r = -3.0 * torch.sigmoid(s1)
    if invalid_frac > 0 or disc_frac > 0:
        q = torch.quantile(s2, torch.tensor(
            [invalid_frac, invalid_frac + disc_frac], device=s2.device).clamp(0, 1))
        r = torch.where(s2 < q[0], torch.full_like(r, -10.0), r)
        r = torch.where((s2 >= q[0]) & (s2 < q[1]), torch.full_like(r, -4.0), r)
    return r


# --------------------------------------------------------------------------- (a)
@torch.no_grad()
def y_equals_x_control(model, X_t, E_t, t, node_mask, *, lam, K, reps, invalid_frac):
    """E[a_hat] where the true adjoint is exactly 1.0."""
    noisy = {"X_t": X_t, "E_t": E_t, "y_t": torch.zeros(X_t.shape[0], 0, device=X_t.device),
             "t": t, "node_mask": node_mask}
    pred = model.forward(noisy, model._compute_extra_data(noisy), node_mask)
    lp = (F.log_softmax(pred.X, -1), F.log_softmax(pred.E, -1))

    def draw():
        s = sample_from_probs(lp[0].exp(), lp[1].exp(), node_mask)
        X1 = F.one_hot(s.X, lp[0].shape[-1]).float() * node_mask[..., None]
        em = (node_mask[:, :, None] & node_mask[:, None, :]).float()
        E1 = F.one_hot(s.E, lp[1].shape[-1]).float() * em[..., None]
        return X1, E1

    out, clamps = [], []
    for _ in range(reps):
        # p^base == p^theta at the pre-RL fixed point, so the log-ratio is 0 and the
        # control isolates the estimator, not the drift.
        Zx, Ze = draw()
        g_Z = -lam * tiered_reward(Zx, Ze, node_mask, invalid_frac=invalid_frac)
        lr_Z = torch.zeros_like(g_Z)
        gk, lrk = [], []
        for _ in range(K):
            Xk, Ek = draw()
            gk.append(-lam * tiered_reward(Xk, Ek, node_mask, invalid_frac=invalid_frac))
            lrk.append(torch.zeros_like(gk[-1]))
        log_a, frac = discrete_adjoint(lr_Z, g_Z, torch.stack(lrk, -1),
                                       torch.stack(gk, -1), clamp=10.0)
        out.append(log_a.exp())
        clamps.append(frac)
    a = torch.cat(out)
    return float(a.mean()), float(a.median()), float(sum(clamps) / len(clamps))


# --------------------------------------------------------------------------- (b)
def projection_gap(model, X_t, E_t, t, node_mask, *, eta, tilt_sd=0.5, steps=300):
    """How much of a random directional tilt the clean-graph head can express."""
    BX, BE = rate_basis(model, X_t, E_t, t, node_mask, eta=eta)
    with torch.no_grad():
        noisy = {"X_t": X_t, "E_t": E_t,
                 "y_t": torch.zeros(X_t.shape[0], 0, device=X_t.device),
                 "t": t, "node_mask": node_mask}
        pred = model.forward(noisy, model._compute_extra_data(noisy), node_mask)
        pX0, pE0 = F.softmax(pred.X, -1), F.softmax(pred.E, -1)
        uX0, uE0 = marginal_rate(pX0, pE0, BX, BE)
        torch.manual_seed(0)
        tX = (tilt_sd * torch.randn_like(uX0)).exp()
        tE = (tilt_sd * torch.randn_like(uE0)).exp()
        tgtX, tgtE = uX0 * tX, uE0 * tE
        noop = (gkl(uX0, tgtX).sum(), gkl(uE0, tgtE).sum())

    lX = torch.log(pX0.clamp_min(1e-8)).clone().requires_grad_(True)
    lE = torch.log(pE0.clamp_min(1e-8)).clone().requires_grad_(True)
    opt = torch.optim.Adam([lX, lE], lr=0.05)
    for _ in range(steps):
        uX, uE = marginal_rate(F.softmax(lX, -1), F.softmax(lE, -1), BX, BE)
        loss = gkl(uX, tgtX).sum() + gkl(uE, tgtE).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        uX, uE = marginal_rate(F.softmax(lX, -1), F.softmax(lE, -1), BX, BE)
        rX = float(gkl(uX, tgtX).sum() / noop[0].clamp_min(1e-12))
        rE = float(gkl(uE, tgtE).sum() / noop[1].clamp_min(1e-12))
    return rX, rE


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--graphs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--eta", type=float, default=1.0)
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(f"loading {args.base}")
    model = DeFoGModel.load(args.base, device="cpu").to(dev).eval()
    dx = model.limit_dist.num_node_classes
    de = model.limit_dist.num_edge_classes
    print(f"base: dx={dx} de={de} noise={model.limit_dist.noise_type} "
          f"rdb={model.rate_matrix_designer.rdb} device={dev}")

    # A batch of realistic clean graphs, then re-noise them: exactly the states the
    # trainer scores at.
    n = 24
    torch.manual_seed(0)
    node_mask = torch.zeros(args.graphs, n, dtype=torch.bool, device=dev)
    node_mask[:, :20] = True
    X1 = F.one_hot(torch.randint(0, dx, (args.graphs, n), device=dev), dx).float()
    idxE = torch.randint(0, de, (args.graphs, n, n), device=dev)
    idxE = torch.triu(idxE, 1); idxE = idxE + idxE.transpose(1, 2)
    E1 = F.one_hot(idxE, de).float()
    X1 = X1 * node_mask[..., None]
    em = (node_mask[:, :, None] & node_mask[:, None, :]).float()
    E1 = E1 * em[..., None]
    y0 = torch.zeros(args.graphs, 0, device=dev)

    # Self-check: a reward that does not vary across samples makes the adjoint
    # identically 1.0 and the control vacuous. Fail loudly rather than report 1.000.
    t_chk = torch.full((args.graphs, 1), 0.5, device=dev)
    (Xc, Ec, _), = renoise_states(model, X1, E1, y0, node_mask, [t_chk])
    _nz = {"X_t": Xc, "E_t": Ec, "y_t": y0, "t": t_chk, "node_mask": node_mask}
    with torch.no_grad():
        _p = model.forward(_nz, model._compute_extra_data(_nz), node_mask)
        _lx, _le = F.softmax(_p.X, -1), F.softmax(_p.E, -1)
        _rs = []
        for _ in range(64):
            _s = sample_from_probs(_lx, _le, node_mask)
            _X = F.one_hot(_s.X, dx).float() * node_mask[..., None]
            _E = F.one_hot(_s.E, de).float() * em[..., None]
            _rs.append(tiered_reward(_X, _E, node_mask))
        _rs = torch.cat(_rs)
    print(f"stand-in reward over 64 head draws: mean {float(_rs.mean()):+.3f} "
          f"sd {float(_rs.std()):.3f} min {float(_rs.min()):+.2f} max {float(_rs.max()):+.2f} "
          f"| floor {float((_rs <= -9.9).float().mean()):.2f} "
          f"disc {float((_rs == -4.0).float().mean()):.2f}")
    if float(_rs.std()) < 1e-6:
        raise SystemExit("stand-in reward does not vary across samples -- the y=x "
                         "control would be vacuous (adjoint identically 1.0)")

    print("\n(a) y = x CONTROL -- the true adjoint is exactly 1.0")
    print(f"{'invalid':>8} {'t':>5} {'lambda':>7} {'K':>4} {'E[a_hat]':>11} "
          f"{'median':>9} {'clamp':>7}")
    for inv in (0.10, 0.95):
        for t_val in (0.2, 0.9):
            t = torch.full((args.graphs, 1), t_val, device=dev)
            (X_t, E_t, _), = renoise_states(model, X1, E1, y0, node_mask, [t])
            for lam in (0.3, 1.0):
                for K in (12, 64):
                    mean, med, frac = y_equals_x_control(
                        model, X_t, E_t, t, node_mask, lam=lam, K=K,
                        reps=args.reps, invalid_frac=inv)
                    flag = "  <-- hot" if abs(math.log(max(mean, 1e-12))) > 0.5 else ""
                    print(f"{inv:8.2f} {t_val:5.1f} {lam:7.1f} {K:4d} {mean:11.3f} "
                          f"{med:9.3f} {frac:7.2f}{flag}")

    print("\n(b) PROJECTION GAP -- residual / no-op gKL after fitting the head")
    print(f"{'t':>6} {'nodes':>9} {'edges':>9}")
    for t_val in (0.2, 0.5, 0.9):
        t = torch.full((args.graphs, 1), t_val, device=dev)
        (X_t, E_t, _), = renoise_states(model, X1, E1, y0, node_mask, [t])
        rX, rE = projection_gap(model, X_t, E_t, t, node_mask, eta=args.eta)
        print(f"{t_val:6.1f} {rX:9.3f} {rE:9.3f}")

    print("\nread: (a) E[a_hat] far from 1.0 means lambda is too hot or K too small;")
    print("      (b) a ratio near 1.0 means the head cannot express the tilt at all.")


if __name__ == "__main__":
    main()
