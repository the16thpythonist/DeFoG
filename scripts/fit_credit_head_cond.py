"""Fit the credit head against the ACTUAL conditional expectation.

The first fitting harness (`fit_credit_head.py`) re-noised ONE endpoint per state, so
each x_t had exactly one (x_t, w) pair. At the noise levels that matters at, x_t nearly
determines its endpoint, so the head can satisfy the loss by reconstructing which
molecule the state came from and reporting its reward -- a per-STATE value function,
with no coordinate structure. That is consistent with everything round 1 and round 2
measured: Gate 1 passed (state value beats a per-class scalar), Gate 2 failed (no
per-coordinate content), Gate 3 null.

Here each state carries K completions SIMULATED FROM IT, so the same x_t maps to K
different endpoints and K different rewards. The regression target is then genuinely

    m_t(xt)[i, c] = E[ exp(-g(z)) | z^i = c, xt ]

over a real conditional distribution, and the reconstruct-the-endpoint shortcut does
not fit it.

Cost is LOWER than the original despite sounding heavier: completions start from x_t
rather than from noise, so they run only the remaining steps.

  python scripts/fit_credit_head_cond.py --states 96 --completions 8 --steps 100  # smoke
  python scripts/fit_credit_head_cond.py --states 1024 --completions 8 --eta 30   # cluster
"""
import argparse, json, math, os, time

import torch

from defog.core import DeFoGModel, AdaLNAdapter
from defog.core.adapter import AdapterComposition, ConditionBranch
from defog.core.credit import (CreditHead, constant_baseline, credit_gkl, edge_mask_of,
                               gather_log_m, pad_batch, per_class_baseline)
from defog.core.dam import simulate_to_end
from defog.core.renoise import draw_times, renoise_states
from defog.core.rl import PropertyMatchReward, RolloutSampler
from defog.domains.molecule import build_encoders
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

ATOMS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = ["SINGLE", "DOUBLE", "TRIPLE"]


def build_cond_pool(base, adapter, args, dev):
    _, adec, _, bdec = build_encoders(ATOMS, BONDS)
    rew = PropertyMatchReward(adec, bdec, lambda m: float(Crippen.MolLogP(m)), scale=1.0)
    K = args.completions
    batches, done, t0 = [], 0, time.time()
    part = (args.pool_cache + ".part") if args.pool_cache else ""
    if part and os.path.exists(part):
        batches = torch.load(part, weights_only=False)["batches"]
        done = sum(b["t"].shape[0] for b in batches)
        print(f"  resuming: {done}/{args.states} states already built", flush=True)
    while done < args.states:
        b = min(args.batch, args.states - done)
        cond = (torch.rand(b, 1, device=dev) * 4.0 - 1.0)
        comp = AdapterComposition([ConditionBranch(adapter, cond, 1.0)], base=base,
                                  mode="product")
        s = RolloutSampler(base, eta=args.eta, omega=0.0, sample_steps=args.steps,
                           time_distortion="polydec", record_trace=False)
        s.composition = comp
        s.sample(b, condition=cond, device=dev, show_progress=False)
        X1, E1 = s.endpoint
        nm = s.end_node_mask
        # One noise level per batch; varied ACROSS batches so the pool spans t. States
        # in a batch must share it because completions resume from a common grid index.
        # Bias toward LOWER step indices. polydec maps a uniform step fraction u to
        # t = 2u - u^2, which piles t up near 1 -- and the measured within-state reward
        # spread there is 0.21 against 2.19 at t=0.44, i.e. the conditional this head
        # regresses is nearly degenerate at high t. Learning is only possible where the
        # conditional is rich, so sample where it is.
        u = float(torch.rand(1)) ** args.t_bias
        t_int = int(min(max(u * args.steps, args.steps // 16), args.steps - 2))
        times = draw_times(base, b, dev, mode="match", n_draws=1, step_indices=[t_int],
                           sample_steps=args.steps, time_distortion="polydec")
        X_t, E_t, t = renoise_states(base, X1, E1,
                                     torch.zeros(b, 0, device=dev), nm, times)[0]
        zx, ze, rr = [], [], []
        with torch.no_grad():
            k_done = 0
            while k_done < K:
                r = min(max(1, args.chunk // max(b, 1)), K - k_done)
                cr = cond.repeat(r, 1); nr = nm.repeat(r, 1)
                sX, sE = simulate_to_end(
                    base, X_t.repeat(r, 1, 1), E_t.repeat(r, 1, 1, 1),
                    torch.zeros(r * b, 0, device=dev), nr, t_int,
                    sample_steps=args.steps, time_distortion="polydec",
                    eta=args.eta, omega=0.0,
                    composition=AdapterComposition(
                        [ConditionBranch(adapter, cr, 1.0)], base=base, mode="product"))
                Xi, Ei, _ = base.limit_dist.ignore_virtual_classes(sX.clone(), sE.clone())
                rv = rew(Xi, Ei, nr, cr).to(dev).float().reshape(r, b)
                zx.append(sX.view(r, b, *sX.shape[1:]).cpu())
                ze.append(sE.view(r, b, *sE.shape[1:]).cpu())
                rr.append(rv.cpu())
                k_done += r
        batches.append({"X_t": X_t.cpu(), "E_t": E_t.cpu(), "t": t.cpu(),
                        "cond": cond.cpu(), "node_mask": nm.cpu(),
                        "Z_X": torch.cat(zx), "Z_E": torch.cat(ze),
                        "reward": torch.cat(rr)})
        done += b
        spread = float(torch.cat(rr).std(0).mean())     # within-state reward spread
        print(f"  states {done}/{args.states}  t={float(t[0,0]):.3f}  n={X_t.shape[1]}"
              f"  within-state sd(r) {spread:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        if part and len(batches) % 2 == 0:
            torch.save({"batches": batches}, part + ".tmp")
            os.replace(part + ".tmp", part)
    n = max(x["X_t"].shape[1] for x in batches)
    out = {k: [] for k in batches[0]}
    for x in batches:
        cur = x["X_t"].shape[1]
        pX, pE, pM = pad_batch(x["X_t"], x["E_t"], x["node_mask"], n)
        out["X_t"].append(pX); out["E_t"].append(pE); out["node_mask"].append(pM)
        out["t"].append(x["t"]); out["cond"].append(x["cond"]); out["reward"].append(x["reward"])
        K_, B_ = x["Z_X"].shape[:2]
        # the ORIGINAL mask, not pM -- pM is already padded to n, and pad_batch expects
        # a mask matching the tensor it is padding
        zX, zE, _ = pad_batch(x["Z_X"].reshape(K_ * B_, cur, -1),
                              x["Z_E"].reshape(K_ * B_, cur, cur, -1),
                              x["node_mask"].repeat(K_, 1), n)
        out["Z_X"].append(zX.reshape(K_, B_, n, -1))
        out["Z_E"].append(zE.reshape(K_, B_, n, n, -1))
    cat = {k: (torch.cat(v, 1) if k.startswith("Z") or k == "reward" else torch.cat(v))
           for k, v in out.items()}
    return cat


def cond_loss(head, pool, idx, k, lam, dev):
    X_t = pool["X_t"][idx].to(dev); E_t = pool["E_t"][idx].to(dev)
    t = pool["t"][idx].to(dev); nm = pool["node_mask"][idx].to(dev)
    cond = pool["cond"][idx].to(dev)
    Z_X = pool["Z_X"][k, idx].to(dev); Z_E = pool["Z_E"][k, idx].to(dev)
    log_w = (lam * pool["reward"][k, idx].to(dev))
    lmX, lmE = head(X_t, E_t, t, nm, cond)
    gX, gE = gather_log_m(lmX, lmE, Z_X, Z_E)
    em = edge_mask_of(nm)
    ln = credit_gkl(gX, log_w[:, None].expand_as(gX))[nm].mean()
    le = credit_gkl(gE, log_w[:, None, None].expand_as(gE))[em].mean()
    return 0.5 * (ln + le), float(ln), float(le), (gX, Z_X, nm, log_w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=int, default=96)
    p.add_argument("--completions", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--batch-train", type=int, default=32)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eta", type=float, default=30.0)
    p.add_argument("--t-bias", type=float, default=1.6,
                   help="exponent on U(0,1) for the step index; >1 favours "
                        "lower t, where the conditional is not degenerate")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--readout", default="scaled", choices=["scaled", "gated"])
    p.add_argument("--readout-scale", type=float, default=0.3)
    p.add_argument("--base", default="ckpts/zinc_e1_seed42_kek.ckpt")
    p.add_argument("--adapter", default="ckpts/clogp_v11/clogp_adapter.ckpt")
    p.add_argument("--pool-cache", default="")
    p.add_argument("--out", default="ckpts/credit/credit_head_cond.ckpt")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    base = DeFoGModel.load(a.base, device="cpu").to(dev).eval()
    for q in base.parameters():
        q.requires_grad_(False)
    adapter = AdaLNAdapter.load(a.adapter, device=dev)

    if a.pool_cache and os.path.exists(a.pool_cache):
        pool = torch.load(a.pool_cache, weights_only=False)
        print(f"pool loaded: {pool['t'].shape[0]} states x {pool['reward'].shape[0]}",
              flush=True)
    else:
        pool = build_cond_pool(base, adapter, a, dev)
        if a.pool_cache:
            torch.save(pool, a.pool_cache + ".tmp")
            os.replace(a.pool_cache + ".tmp", a.pool_cache)
            if os.path.exists(a.pool_cache + ".part"):
                os.remove(a.pool_cache + ".part")
    S, K = pool["t"].shape[0], pool["reward"].shape[0]
    r = pool["reward"]
    # The whole point: within-state spread must be NON-ZERO, else the conditional is
    # degenerate and this pool is no better than the one-endpoint version.
    within = float(r.std(0).mean()); between = float(r.mean(0).std())
    print(f"pool: {S} states x {K} completions | reward {float(r.mean()):+.4f}"
          f" | within-state sd {within:.4f}  between-state sd {between:.4f}"
          f"  within/total {within/max((within**2+between**2)**0.5,1e-9):.3f}", flush=True)

    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(S, generator=g)
    nval = max(1, int(a.val_frac * S))
    va, tr = perm[:nval].tolist(), perm[nval:].tolist()

    head = CreditHead(base, cond_dim=1, readout=a.readout,
                      readout_scale=a.readout_scale,
                      cond_mean=[float(pool["cond"].mean())],
                      cond_std=[float(pool["cond"].std()) or 1.0]).to(dev)
    clsX = pool["Z_X"][:, tr].argmax(-1).reshape(-1, pool["Z_X"].shape[-2])
    mk = pool["node_mask"][tr].repeat(K, 1)
    lw = (a.lam * pool["reward"][:, tr]).reshape(-1)[:, None].expand_as(clsX)
    cst = constant_baseline(lw[mk])
    pc = per_class_baseline(lw, clsX, pool["Z_X"].shape[-1], mk)
    pc = torch.where(torch.isfinite(pc), pc, torch.full_like(pc, cst))
    head.init_bias(pc.to(dev), torch.full((pool["Z_E"].shape[-1],), cst).to(dev))
    print(f"credit head: {sum(q.numel() for q in head.parameters() if q.requires_grad):,}"
          f" trainable params (readout={a.readout})", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-5)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        idx = [tr[int(j)] for j in torch.randint(0, len(tr), (a.batch_train,), generator=g)]
        k = int(torch.randint(0, K, (1,), generator=g))
        loss, ln, le, _ = cond_loss(head, pool, idx, k, a.lam, dev)
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if it % max(1, a.iters // 12) == 0 or it == 1:
            print(f"  it {it:5d}  loss {float(loss):.5f}  |g| {float(gn):.3f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)

    head.eval()
    tot = {"head": [], "const": [], "class": []}
    pcd = pc.to(dev)
    with torch.no_grad():
        for kk in range(K):
            for i in range(0, len(va), a.batch_train):
                idx = va[i:i + a.batch_train]
                _, _, _, aux = cond_loss(head, pool, idx, kk, a.lam, dev)
                gX, Z_X, nm, log_w = aux
                lwb = log_w[:, None].expand_as(gX)
                cls = Z_X.argmax(-1)
                tot["head"].append(credit_gkl(gX, lwb)[nm].cpu())
                tot["const"].append(credit_gkl(torch.full_like(gX, cst), lwb)[nm].cpu())
                tot["class"].append(credit_gkl(pcd[cls], lwb)[nm].cpu())
    v = {kk: torch.cat(x) for kk, x in tot.items()}
    res = {kk: float(x.mean()) for kk, x in v.items()}
    print(f"\nGATE 1  held-out gKL ({len(va)} states x {K} completions)")
    for kk in ("head", "const", "class"):
        print(f"  {kk:6s} {res[kk]:.6f}")
    ok = True
    for ref, lab in (("const", "constant"), ("class", "per-class")):
        d = v["head"] - v[ref]
        se = float(d.std() / math.sqrt(d.numel()))
        t = float(d.mean()) / max(se, 1e-12)
        res[f"d_{ref}"], res[f"t_{ref}"] = float(d.mean()), t
        print(f"  vs {lab:9s}: {100*(1-res['head']/max(res[ref],1e-12)):+6.2f}%   "
              f"paired d {float(d.mean()):+.6f} +- {se:.6f}  t = {t:+7.2f}")
        ok = ok and d.mean() < 0 and t < -3.0
    print(f"  -> {'PASS' if ok else 'FAIL'}", flush=True)
    head.save(a.out, gate1=res, args=vars(a))
    json.dump({"gate1": res, "pass": bool(ok), "args": vars(a)},
              open(os.path.splitext(a.out)[0] + "_gate1.json", "w"), indent=1)
    print(f"\nwrote {a.out}\nFIT-DONE", flush=True)


if __name__ == "__main__":
    main()
