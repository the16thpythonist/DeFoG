"""Round 4: train the head on what guidance actually consumes.

Guidance renormalises each coordinate independently:

    q(x1^i = c)  ~  p_base(c) * m[i, c],   normalised over c

so multiplying m[i, .] by ANY constant that depends on the state and coordinate but not
the class leaves the sampler's output bit-for-bit unchanged. That is a gauge freedom:
only the variation ACROSS CLASSES at a coordinate does anything.

Rounds 1-3 minimised gkl(m[i, observed_class], w) -- the ABSOLUTE value of m against the
reward. That is not gauge-invariant, and the level is the large, easy part (between-state
reward sd 0.66-0.86 against a small per-coordinate contrast). So the optimiser spent its
capacity on E[w | x_t], a quantity guidance mathematically ignores, and treated the only
thing that matters as a residual.

It explains everything at once: Gate 1 passed (it scores the level), Gate 2 failed (it
strips per-state means, leaving the contrast), Gate 2 did not track Gate 1 (orthogonal
quantities), and fixing the readout and the target changed neither.

Here the target is the reward-tilted empirical class distribution over the K completions

    p*[i, c] = sum_k softmax(lam * r)_k * 1[z_k^i = c]

and the loss is cross-entropy against the NORMALISED guided marginal. Gauge-invariant by
construction, and it uses all K completions per state rather than one.

Reuses the round-3 pools unchanged, so this costs training only.
"""
import argparse, json, math, os, time

import torch
import torch.nn.functional as F

from defog.core import DeFoGModel, AdaLNAdapter
from defog.core.adapter import AdapterComposition, ConditionBranch
from defog.core.credit import CreditHead, edge_mask_of
from defog.core.rl import _base_uncond_softmax, _compose_logmarginals


@torch.no_grad()
def base_marginals(base, adapter, pool, dev, batch=32):
    """log p_base(x1 | x_t) for every state, under the SAME composed policy the sampler
    uses. Frozen, so computed once and cached rather than per iteration."""
    S = pool["t"].shape[0]
    outX, outE = [], []
    for i in range(0, S, batch):
        sl = slice(i, min(i + batch, S))
        X_t = pool["X_t"][sl].to(dev); E_t = pool["E_t"][sl].to(dev)
        t = pool["t"][sl].to(dev); nm = pool["node_mask"][sl].to(dev)
        cond = pool["cond"][sl].to(dev)
        puX, puE, noisy, extra = _base_uncond_softmax(base, X_t, E_t, t, nm)
        lX, lE = _compose_logmarginals(base, adapter, noisy, extra, nm, cond,
                                       puX, puE, 1.0, "product")
        outX.append(lX.cpu()); outE.append(lE.cpu())
    return torch.cat(outX), torch.cat(outE)


OXY = 2   # index of O in ["C","N","O","F","P","S","Cl","Br","I"]


def pool_reward(pool, kind, dev=None):
    """The pool stores COMPLETIONS, not rewards -- so a different reward costs nothing
    to evaluate. Oxygen count is read straight off the one-hot; no RDKit, no resampling.

    Standardised to unit sd so `lam` means the same thing across rewards, which is how
    the oxy-max vs logp-match contrast was measured at matched effective sample size.
    """
    if kind == "logp":
        r = pool["reward"].clone()
    elif kind == "oxy":
        nm = pool["node_mask"][None, :, :].float()
        r = (pool["Z_X"][..., OXY] * nm).sum(-1)
    else:
        raise ValueError(kind)
    r = (r - r.mean()) / r.std().clamp_min(1e-8)
    return r.to(dev) if dev is not None else r


def ce_loss(head, pool, lpX, lpE, idx, lam, dev, rw=None, base_mode="model"):
    X_t = pool["X_t"][idx].to(dev); E_t = pool["E_t"][idx].to(dev)
    t = pool["t"][idx].to(dev); nm = pool["node_mask"][idx].to(dev)
    cond = pool["cond"][idx].to(dev)
    Z_X = pool["Z_X"][:, idx].to(dev)          # (K, b, n, dx) one-hot completions
    Z_E = pool["Z_E"][:, idx].to(dev)
    r = (pool["reward"][:, idx].to(dev) if rw is None else rw[:, idx].to(dev))
    w = torch.softmax(lam * r, dim=0)                                   # (K, b)

    # the reward-tilted empirical class distribution -- the target guidance should hit
    tX = (w[:, :, None, None] * Z_X).sum(0)
    tE = (w[:, :, None, None, None] * Z_E).sum(0)

    if base_mode == "emp":
        # Target the TILT ONLY. p* and p_emp come from the SAME completions, so the
        # base model's miscalibration appears in both and cancels exactly; what is left
        # is the reward's effect and nothing else. Round 4 used the model's marginals
        # here, which bundled calibration with credit -- and the lambda=0 control showed
        # the entire gain was calibration. This also makes the loss and Gate 2 measure
        # the same object, which they did not before.
        bX = torch.log(Z_X.mean(0).clamp_min(1e-6))
        bE = torch.log(Z_E.mean(0).clamp_min(1e-6))
    else:
        bX, bE = lpX[idx].to(dev), lpE[idx].to(dev)

    lmX, lmE = head(X_t, E_t, t, nm, cond)
    qX = F.log_softmax(bX + lmX, dim=-1)                                # normalised =>
    qE = F.log_softmax(bE + lmE, dim=-1)                                # gauge-invariant
    em = edge_mask_of(nm)
    ceX = -(tX * qX).sum(-1)[nm].mean()
    ceE = -(tE * qE).sum(-1)[em].mean()
    return 0.5 * (ceX + ceE), float(ceX), float(ceE), (qX, tX, nm, bX)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--base", default="ckpts/zinc_e1_seed42_kek.ckpt")
    p.add_argument("--adapter", default="ckpts/clogp_v11/clogp_adapter.ckpt")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--batch-train", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--readout", default="scaled", choices=["scaled", "gated"])
    p.add_argument("--readout-scale", type=float, default=0.3)
    p.add_argument("--out", default="ckpts/credit/credit_head_ce.ckpt")
    p.add_argument("--use-k", type=int, default=0,
                   help="use only the first K completions; 0 = all. Lets the "
                        "K dose-response run on IDENTICAL states.")
    p.add_argument("--reward", default="logp", choices=["logp", "oxy"])
    p.add_argument("--base-mode", default="emp", choices=["emp", "model"],
                   help="emp targets the reward TILT only (calibration "
                        "cancels); model bundles calibration with credit")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    base = DeFoGModel.load(a.base, device="cpu").to(dev).eval()
    for q in base.parameters():
        q.requires_grad_(False)
    adapter = AdaLNAdapter.load(a.adapter, device=dev)
    pool = torch.load(a.pool, weights_only=False)
    if a.use_k and a.use_k < pool["reward"].shape[0]:
        for k in ("Z_X", "Z_E", "reward"):
            pool[k] = pool[k][:a.use_k]
    S, K = pool["t"].shape[0], pool["reward"].shape[0]
    rw = pool_reward(pool, a.reward)
    # Split-half reliability of the TARGET. Round 5 regressed a target whose reliability
    # was ~0.00 at K=6-8, which no amount of training can fix; printing it makes an
    # uninterpretable run obvious from its first line instead of its last.
    with torch.no_grad():
        h = K // 2
        def _tilt(ix):
            w = torch.softmax(rw[ix], 0)
            ps = (w[:, :, None, None] * pool["Z_X"][ix]).sum(0)
            pe = pool["Z_X"][ix].mean(0)
            return torch.log((ps + 1e-6) / (pe + 1e-6))
        m = pool["node_mask"]
        A, B = _tilt(list(range(h)))[m].flatten(), _tilt(list(range(h, 2*h)))[m].flatten()
        A, B = A - A.mean(), B - B.mean()
        rel = float((A*B).sum() / (A.norm()*B.norm()).clamp_min(1e-12))
    print(f"target split-half reliability at K={h}/half: r = {rel:+.4f}"
          f"   (Spearman-Brown at K={K}: {2*rel/(1+rel) if rel > -1 else float('nan'):+.4f})",
          flush=True)
    print(f"pool: {S} states x {K} completions | reward={a.reward} "
          f"base-mode={a.base_mode} | within-state sd {float(rw.std(0).mean()):.4f} "
          f"between {float(rw.mean(0).std()):.4f}", flush=True)
    t0 = time.time()
    lpX, lpE = base_marginals(base, adapter, pool, dev)
    print(f"base marginals cached [{time.time()-t0:.0f}s]", flush=True)

    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(S, generator=g)
    nval = max(1, int(a.val_frac * S))
    va, tr = perm[:nval].tolist(), perm[nval:].tolist()

    head = CreditHead(base, cond_dim=1, readout=a.readout,
                      readout_scale=a.readout_scale,
                      cond_mean=[float(pool["cond"].mean())],
                      cond_std=[float(pool["cond"].std()) or 1.0]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-5)

    t0 = time.time()
    for it in range(1, a.iters + 1):
        idx = [tr[int(j)] for j in torch.randint(0, len(tr), (a.batch_train,), generator=g)]
        loss, cx, ce, _ = ce_loss(head, pool, lpX, lpE, idx, a.lam, dev, rw, a.base_mode)
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if it % max(1, a.iters // 12) == 0 or it == 1:
            print(f"  it {it:5d}  CE {float(loss):.5f}  (node {cx:.5f} edge {ce:.5f})"
                  f"  |g| {float(gn):.3f}  [{time.time()-t0:.0f}s]", flush=True)

    # Gate 1, in the CE formulation: held-out cross-entropy of the GUIDED marginal
    # against the tilted target, versus (a) no guidance at all and (b) a per-class m.
    # Lower is better. Both references are gauge-equivalent to what they claim to be.
    head.eval()
    hs, us, ps = [], [], []
    with torch.no_grad():
        # per-class m fitted on the TRAINING split, in log space
        acc_n = torch.zeros(pool["Z_X"].shape[-1], device=dev)
        acc_d = torch.zeros_like(acc_n)
        for i in range(0, len(tr), 64):
            idx = tr[i:i + 64]
            nm = pool["node_mask"][idx].to(dev)
            Z = pool["Z_X"][:, idx].to(dev)
            w = torch.softmax(a.lam * rw[:, idx].to(dev), dim=0)
            tX = (w[:, :, None, None] * Z).sum(0)
            bX = Z.mean(0)
            acc_n += tX[nm].sum(0); acc_d += bX[nm].sum(0)
        pc = torch.log((acc_n / acc_d.clamp_min(1e-9)).clamp_min(1e-9))
        for i in range(0, len(va), 16):
            idx = va[i:i + 16]
            _, _, _, aux = ce_loss(head, pool, lpX, lpE, idx, a.lam, dev, rw, a.base_mode)
            qX, tX, nm, lp = aux
            hs.append((-(tX * qX).sum(-1))[nm].cpu())
            us.append((-(tX * F.log_softmax(lp, -1)).sum(-1))[nm].cpu())
            ps.append((-(tX * F.log_softmax(lp + pc, -1)).sum(-1))[nm].cpu())
    H, U, P = torch.cat(hs), torch.cat(us), torch.cat(ps)
    res = {"head": float(H.mean()), "unguided": float(U.mean()), "per_class": float(P.mean())}
    print(f"\nGATE 1 (CE)  held-out, node channel ({len(va)} states)")
    for k in ("head", "unguided", "per_class"):
        print(f"  {k:9s} {res[k]:.6f}")
    ok = True
    for ref, arr, lab in (("unguided", U, "unguided"), ("per_class", P, "per-class")):
        d = H - arr
        se = float(d.std() / math.sqrt(d.numel()))
        tt = float(d.mean()) / max(se, 1e-12)
        res[f"d_{ref}"], res[f"t_{ref}"] = float(d.mean()), tt
        print(f"  vs {lab:9s}: paired d {float(d.mean()):+.6f} +- {se:.6f}  t = {tt:+7.2f}")
        ok = ok and d.mean() < 0 and tt < -3.0
    print(f"  -> {'PASS' if ok else 'FAIL'}", flush=True)
    head.save(a.out, gate1_ce=res, args=vars(a))
    json.dump({"gate1_ce": res, "pass": bool(ok), "args": vars(a)},
              open(os.path.splitext(a.out)[0] + "_gate1.json", "w"), indent=1)
    print(f"\nwrote {a.out}\nFIT-DONE", flush=True)


if __name__ == "__main__":
    main()
