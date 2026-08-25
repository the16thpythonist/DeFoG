"""Fit the amortised credit head, and run Gate 1.

    m_t(xt)[i, c] = E_base[ exp(-g(x1)) | x1^i = c, xt ]

Two phases, because the cost is lopsided. Building a pool of endpoints needs full
rollouts (expensive, done once and cached); training then only needs a re-noise plus one
credit-head forward per step (cheap, done thousands of times). Re-noising is how DeFoG
itself was trained, so the state distribution matches what the base has seen.

Gate 1 (docs/credit_head_design.md): on HELD-OUT endpoints the fitted head must beat
  (a) the best global constant  -- log E[w]
  (b) the best per-class scalar -- log E[w | class]
by more than the seed spread. The earlier scoring head died exactly here, landing worse
than predicting the mean, so this is the gate and not a formality.

  python scripts/fit_credit_head.py --pool 512 --steps 100 --iters 400      # local smoke
  python scripts/fit_credit_head.py --pool 8192 --steps 500 --eta 30        # cluster
"""
import argparse, json, math, os, time

import torch
import torch.nn.functional as F

from defog.core import DeFoGModel, AdaLNAdapter
from defog.core.adapter import AdapterComposition, ConditionBranch
from defog.core.credit import (CreditHead, assemble, constant_baseline, credit_gkl,
                               edge_mask_of, gather_log_m, pad_batch,
                               per_class_baseline)
from defog.core.renoise import draw_times, renoise_states
from defog.core.rl import PropertyMatchReward, RolloutSampler
from defog.domains.molecule import build_encoders
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

ATOMS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = ["SINGLE", "DOUBLE", "TRIPLE"]


def build_pool(base, adapter, args, dev):
    """Rollout endpoints under the composed policy and score them. Cached: this is the
    only expensive part, and the training loop reuses it for every epoch."""
    _, adec, _, bdec = build_encoders(ATOMS, BONDS)
    rew = PropertyMatchReward(adec, bdec, lambda m: float(Crippen.MolLogP(m)), scale=1.0)
    batches, done, t0 = [], 0, time.time()
    part = (args.pool_cache + ".part") if args.pool_cache else ""
    if part and os.path.exists(part):
        batches = torch.load(part, weights_only=False)["batches"]
        done = sum(b["reward"].shape[0] for b in batches)
        print(f"  resuming from {part}: {done}/{args.pool} already sampled", flush=True)
    while done < args.pool:
        b = min(args.batch, args.pool - done)
        cond = (torch.rand(b, 1, device=dev) * 4.0 - 1.0)
        comp = AdapterComposition([ConditionBranch(adapter, cond, 1.0)],
                                  base=base, mode="product")
        s = RolloutSampler(base, eta=args.eta, omega=0.0, sample_steps=args.steps,
                           time_distortion="polydec", record_trace=False)
        s.composition = comp
        s.sample(b, condition=cond, device=dev, show_progress=False)
        X1, E1 = s.endpoint
        nm = s.end_node_mask
        Xr, Er, _ = base.limit_dist.ignore_virtual_classes(X1.clone(), E1.clone())
        r = rew(Xr, Er, nm, cond).to(dev).float().reshape(-1)
        batches.append({"X1": X1.cpu(), "E1": E1.cpu(), "node_mask": nm.cpu(),
                        "cond": cond.cpu(), "reward": r.cpu()})
        done += b
        mr = float(torch.cat([q["reward"] for q in batches]).mean())
        print(f"  pool {done}/{args.pool}  mean r {mr:+.4f}  n={X1.shape[1]}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        # Checkpoint the RAW batches, before any step that can fail. The first version
        # saved only after concatenating, and the concatenation was the thing that
        # crashed -- losing 100 minutes of sampling per job. Sampling is the expensive
        # part; it must never be at risk from a cheap step that follows it.
        if part and len(batches) % 4 == 0:
            torch.save({"batches": batches}, part + ".tmp")
            os.replace(part + ".tmp", part)
    return assemble(batches)


def batch_loss(head, base, pool, idx, lam, dev, t_int=None, sample_steps=100,
               time_distortion="polydec", grad=True):
    X1 = pool["X1"][idx].to(dev); E1 = pool["E1"][idx].to(dev)
    nm = pool["node_mask"][idx].to(dev); cond = pool["cond"][idx].to(dev)
    log_w = (lam * pool["reward"][idx].to(dev))                      # log exp(-g)
    y0 = torch.zeros(len(idx), 0, device=dev)
    steps = [t_int] if t_int is not None else \
        [int(torch.randint(1, sample_steps, (1,)))]
    times = draw_times(base, len(idx), dev, mode="match", n_draws=1, step_indices=steps,
                       sample_steps=sample_steps, time_distortion=time_distortion)
    X_t, E_t, t = renoise_states(base, X1, E1, y0, nm, times)[0]
    with torch.set_grad_enabled(grad):
        lmX, lmE = head(X_t, E_t, t, nm, cond)
    gX, gE = gather_log_m(lmX, lmE, X1, E1)
    em = edge_mask_of(nm)
    lw = log_w[:, None].expand_as(gX)
    ln = credit_gkl(gX, lw)[nm].mean()
    le = credit_gkl(gE, log_w[:, None, None].expand_as(gE))[em].mean()
    return 0.5 * (ln + le), float(ln), float(le), (lmX, lmE, X1, E1, nm, em, log_w)


def fit_baselines(pool, tr, lam):
    """Fit the two reference predictors ON THE TRAINING SPLIT.

    They must be fitted where the head was fitted. Computing them from the validation
    batch's own labels -- as an earlier version of this function did -- makes them
    ORACLES that peeked at the answers, and the head, which only ever saw the training
    split, then has to beat predictors that cannot be beaten fairly. The gate would
    read FAIL for a perfectly good head.

    The class a coordinate takes is read straight off the endpoint, so no re-noising is
    needed and these are exact over the whole split.
    """
    X1, E1 = pool["X1"][tr], pool["E1"][tr]
    nm = pool["node_mask"][tr]
    lw = lam * pool["reward"][tr]
    clsX = X1.argmax(-1)
    cst = constant_baseline(lw[:, None].expand_as(clsX)[nm])
    pc = per_class_baseline(lw[:, None].expand_as(clsX), clsX, X1.shape[-1], nm)
    pc = torch.where(torch.isfinite(pc), pc, torch.full_like(pc, cst))
    return cst, pc


def gate1(head, base, pool, va, args, dev, cst, pc):
    """Held-out gKL of the head against the two reference predictors.

    `cst` and `pc` come from :func:`fit_baselines` on the TRAINING split -- see the note
    there about why fitting them on the validation data invalidates the comparison.
    """
    per = {"head": [], "const": [], "class": []}
    pc = pc.to(dev)
    draws = max(1, int(getattr(args, "eval_t_draws", 4)))
    # Average over several t draws: the gate is an expectation over t, and a single
    # draw makes it a noisy statistic (measured ~0.02% wobble on the smoke pool).
    for _ in range(draws):
        for i in range(0, len(va), args.batch_train):
            idx = va[i:i + args.batch_train]
            with torch.no_grad():
                _, _, _, aux = batch_loss(head, base, pool, idx, args.lam, dev,
                                          sample_steps=args.steps, grad=False)
            lmX, lmE, X1, E1, nm, em, log_w = aux
            gX, _ = gather_log_m(lmX, lmE, X1, E1)
            lw = log_w[:, None].expand_as(gX)
            cls = X1.argmax(-1)
            per["head"].append(credit_gkl(gX, lw)[nm].detach().cpu())
            per["const"].append(credit_gkl(torch.full_like(gX, cst), lw)[nm].cpu())
            per["class"].append(credit_gkl(pc[cls], lw)[nm].cpu())
    v = {k: torch.cat(x) for k, x in per.items()}
    out = {k: float(x.mean()) for k, x in v.items()}
    # PAIRED differences. Head and baselines are scored on the SAME entries, so the
    # per-entry difference has far less variance than either mean -- and the reward is
    # a whole-molecule property, so most of the spread is shared and cancels. Comparing
    # two aggregate means with unknown variance cannot resolve the couple of percent
    # that is all a per-coordinate credit can buy here.
    for ref in ("const", "class"):
        d = v["head"] - v[ref]
        se = float(d.std() / math.sqrt(max(d.numel(), 1)))
        out[f"d_{ref}"] = float(d.mean())
        out[f"se_{ref}"] = se
        out[f"t_{ref}"] = float(d.mean()) / se if se > 0 else float("nan")
    out["n_entries"] = int(v["head"].numel())
    return out


def _report(res, res_tr=None):
    """Paired reporting. A bare mean comparison cannot resolve a 1-2% margin; the
    per-entry paired difference and its standard error can."""
    for k in ("head", "const", "class"):
        print(f"  {k:6s} {res[k]:.6f}")
    ok = True
    for ref, label in (("const", "constant"), ("class", "per-class")):
        d, se, t = res[f"d_{ref}"], res[f"se_{ref}"], res[f"t_{ref}"]
        rel = 100 * (1 - res["head"] / max(res[ref], 1e-12))
        extra = ""
        if res_tr is not None:
            extra = f"   [train d {res_tr[f'd_{ref}']:+.6f} t {res_tr[f't_{ref}']:+6.2f}]"
        print(f"  vs {label:9s}: {rel:+6.2f}%   paired d {d:+.6f} +- {se:.6f}"
              f"  t = {t:+7.2f}{extra}")
        ok = ok and d < 0 and t < -3.0
    print(f"  n entries: {res['n_entries']:,}")
    if res_tr is not None:
        # Diagnose from the HEAD's own train-vs-val gap, not from whether it beats the
        # baselines on train. The per-class reference is fitted on train, so it is
        # in-sample there and out-of-sample on val -- it looks strong on train for
        # reasons that have nothing to do with the head. (Observed on the smoke pool:
        # head loses to per-class on train at t=+20 while beating it on val at t=-13.6,
        # which is the baseline overfitting, not the head underfitting.)
        gap = (res["head"] - res_tr["head"]) / max(abs(res_tr["head"]), 1e-12)
        print(f"  head gKL  train {res_tr['head']:.6f}  val {res['head']:.6f}"
              f"   gap {100*gap:+.2f}%")
        if not ok:
            print("  diagnosis: " + (
                "val is much worse than train -> OVERFITTING; the fix is more "
                "endpoints or regularisation, not architecture."
                if gap > 0.02 else
                "train and val agree, so the head is not overfitting -- it simply does "
                "not separate the target. Capacity, optimisation, or no learnable "
                "signal at this t/lambda."))
    print(f"  -> {'PASS' if ok else 'FAIL'}  (need paired d < 0 and t < -3 vs BOTH)",
          flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--batch-train", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eta", type=float, default=30.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--backbone", default="copy", choices=["copy", "shared"])
    p.add_argument("--base", default="ckpts/zinc_e1_seed42_kek.ckpt")
    p.add_argument("--adapter", default="ckpts/clogp_v11/clogp_adapter.ckpt")
    p.add_argument("--pool-cache", default="")
    p.add_argument("--out", default="ckpts/credit_head.ckpt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-t-draws", type=int, default=4,
                   help="t draws to average the held-out gate over; the gate is an "
                        "expectation over t and one draw makes it noisy.")
    p.add_argument("--eval-only", default="",
                   help="Path to a fitted head: recompute Gate 1 from the CACHED pool "
                        "without redoing the rollouts. Used to re-score a run whose "
                        "gate was computed with the oracle-baseline bug.")
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    base = DeFoGModel.load(args.base, device="cpu").to(dev).eval()
    for q in base.parameters():
        q.requires_grad_(False)
    adapter = AdaLNAdapter.load(args.adapter, device=dev)

    if args.pool_cache and os.path.exists(args.pool_cache):
        pool = torch.load(args.pool_cache, weights_only=False)
        print(f"pool loaded: {len(pool['reward'])} endpoints", flush=True)
    else:
        pool = build_pool(base, adapter, args, dev)
        if args.pool_cache:
            torch.save(pool, args.pool_cache + ".tmp")
            os.replace(args.pool_cache + ".tmp", args.pool_cache)
            if os.path.exists(args.pool_cache + ".part"):
                os.remove(args.pool_cache + ".part")
    r = pool["reward"]
    print(f"pool: {len(r)} endpoints  reward {float(r.mean()):+.4f} +- {float(r.std()):.4f}"
          f"  log_w range [{float(args.lam*r.min()):+.2f}, {float(args.lam*r.max()):+.2f}]",
          flush=True)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(r), generator=g)
    nval = max(1, int(args.val_frac * len(r)))
    va, tr = perm[:nval].tolist(), perm[nval:].tolist()

    head = CreditHead(base, cond_dim=1, backbone=args.backbone,
                      cond_mean=[float(pool["cond"].mean())],
                      cond_std=[float(pool["cond"].std()) or 1.0]).to(dev)

    # Start AT the stronger Gate 1 reference. The class a coordinate takes is read
    # straight off the endpoint -- no re-noising needed -- so log E[w | class] is exact
    # over the whole training split. With the gate at zero the head then reproduces the
    # per-class baseline bit-for-bit at iteration 0, which is what makes any held-out
    # improvement attributable to learning rather than to initialisation.
    cst, pc = fit_baselines(pool, tr, args.lam)
    with torch.no_grad():
        trE = pool["E1"][tr].argmax(-1)
        trEM = edge_mask_of(pool["node_mask"][tr])
        lw_tr = (args.lam * pool["reward"][tr])
        be = per_class_baseline(lw_tr[:, None, None].expand_as(trE), trE,
                                pool["E1"].shape[-1], trEM)
        be = torch.where(torch.isfinite(be), be, torch.full_like(be, cst))
        head.init_bias(pc.to(dev), be.to(dev))
    print("  init bias (nodes, log E[w|class]): "
          + " ".join(f"{float(v):+.3f}" for v in head.bias_X), flush=True)
    if args.eval_only:
        head.load_state_dict(torch.load(args.eval_only, map_location=dev,
                                        weights_only=False)["state_dict"])
        res = gate1(head, base, pool, va, args, dev, cst, pc)
        tr_s = tr[:len(va)]
        res_tr = gate1(head, base, pool, tr_s, args, dev, cst, pc)
        print(f"\nGATE 1 (re-scored, training-split baselines)  {args.eval_only}")
        ok = _report(res, res_tr)
        print("FIT-DONE", flush=True)
        return

    npar = sum(q.numel() for q in head.parameters() if q.requires_grad)
    print(f"credit head: {npar:,} trainable params (backbone={args.backbone})", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        idx = [tr[int(j)] for j in torch.randint(0, len(tr), (args.batch_train,), generator=g)]
        loss, ln, le, _ = batch_loss(head, base, pool, idx, args.lam, dev,
                                     sample_steps=args.steps)
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if it % max(1, args.iters // 20) == 0 or it == 1:
            print(f"  it {it:5d}  loss {float(loss):.5f}  (node {ln:.5f} edge {le:.5f})"
                  f"  |g| {float(gn):.3f}  [{time.time()-t0:.0f}s]", flush=True)

    res = gate1(head, base, pool, va, args, dev, cst, pc)
    # Same-sized slice of the TRAINING split, so a failure can be diagnosed instead of
    # merely reported: beats the references on train but not val => overfitting, and
    # the fix is data or regularisation; fails on train too => the head cannot express
    # or reach the target at all, which is a capacity or optimisation problem. Without
    # this a FAIL is a dead end.
    res_tr = gate1(head, base, pool, tr[:len(va)], args, dev, cst, pc)
    print(f"\nGATE 1  held-out gKL, node channel ({len(va)} endpoints)")
    ok = _report(res, res_tr)

    head.save(args.out, gate1=res, args=vars(args))
    json.dump({"gate1": res, "pass": bool(ok), "args": vars(args)},
              open(os.path.splitext(args.out)[0] + "_gate1.json", "w"), indent=1)
    print(f"\nwrote {args.out}\nFIT-DONE", flush=True)


if __name__ == "__main__":
    main()
