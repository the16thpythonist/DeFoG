"""Gate 2: does the trained credit head reproduce the instruction we already measured?

The empirical per-coordinate shift at a state is what `exp(-g)`-reweighting the base's
own completions does to the marginals. The head-implied shift is what multiplying by
`m` does. If the head has learned `m = E[exp(-g) | x1^i=c, xt]` they should agree.

Both shifts are computed FROM THE SAME empirical base marginal, so any mismatch between
the model's one-shot head and the true `p(x1|xt)` cancels and only the head's
contribution is under test.

Three references, because a correlation with nothing to compare it to means little:
  * shuffled null      -- weights permuted; the finite-sample floor
  * per-class scalar   -- the predictor the head must beat to justify its existence
  * reliability ceiling-- nothing can correlate with the empirical shift better than
                          the empirical shift correlates with ITSELF (split-half 0.89)

PASS: resid-2 correlation >= 0.6 AND clearly above the per-class reference.
"""
import argparse, json, math, torch, torch.nn.functional as F

from defog.core import DeFoGModel, AdaLNAdapter
from defog.core.adapter import AdapterComposition, ConditionBranch
from defog.core.credit import CreditHead, edge_mask_of, per_class_baseline
from defog.core.renoise import draw_times, renoise_states
from defog.core.rl import PropertyMatchReward, RolloutSampler
from defog.core.dam import simulate_to_end
from defog.domains.molecule import build_encoders
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

ATOMS = ["C","N","O","F","P","S","Cl","Br","I"]; BONDS = ["SINGLE","DOUBLE","TRIPLE"]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def resid2(M, msk, bs):
    """Remove the per-state per-class mean: what survives is coordinate-specific."""
    M = M.clone()
    flat = M.reshape(bs, -1, M.shape[-1]); mk = msk.reshape(bs, -1)
    for b in range(bs):
        if int(mk[b].sum()):
            flat[b][mk[b]] -= flat[b][mk[b]].mean(0)
    return M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True)
    p.add_argument("--base", default="ckpts/zinc_e1_seed42_kek.ckpt")
    p.add_argument("--adapter", default="ckpts/clogp_v11/clogp_adapter.ckpt")
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--t-int", type=int, default=375)
    p.add_argument("--eta", type=float, default=30.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--ceiling", type=float, default=0.89)
    p.add_argument("--pass-at", type=float, default=0.6)
    p.add_argument("--out", default="")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(11)

    base = DeFoGModel.load(a.base, device="cpu").to(dev).eval()
    adapter = AdaLNAdapter.load(a.adapter, device=dev)
    head = CreditHead.load(a.head, base, device=dev,
                           cond_mean=[1.0], cond_std=[1.2]).eval()
    _, adec, _, bdec = build_encoders(ATOMS, BONDS)
    rew = PropertyMatchReward(adec, bdec, lambda m: float(Crippen.MolLogP(m)), scale=1.0)

    cond = (torch.rand(a.states, 1, device=dev) * 4.0 - 1.0)
    comp = AdapterComposition([ConditionBranch(adapter, cond, 1.0)], base=base,
                              mode="product")
    s = RolloutSampler(base, eta=a.eta, omega=0.0, sample_steps=a.steps,
                       time_distortion="polydec", record_trace=False)
    s.composition = comp
    s.sample(a.states, condition=cond, device=dev, show_progress=False)
    X1, E1 = s.endpoint; nm = s.end_node_mask
    times = draw_times(base, a.states, dev, mode="match", n_draws=1,
                       step_indices=[a.t_int], sample_steps=a.steps,
                       time_distortion="polydec")
    X_t, E_t, t = renoise_states(base, X1, E1,
                                 torch.zeros(a.states, 0, device=dev), nm, times)[0]

    zx, ze, rw = [], [], []
    with torch.no_grad():
        done = 0
        while done < a.k:
            r = min(max(1, a.chunk // a.states), a.k - done)
            Xr, Er = X_t.repeat(r,1,1), E_t.repeat(r,1,1,1)
            nr, cr = nm.repeat(r,1), cond.repeat(r,1)
            sX, sE = simulate_to_end(base, Xr, Er, torch.zeros(nr.shape[0],0,device=dev),
                                     nr, a.t_int, sample_steps=a.steps,
                                     time_distortion="polydec", eta=a.eta, omega=0.0,
                                     composition=AdapterComposition(
                                         [ConditionBranch(adapter, cr, 1.0)],
                                         base=base, mode="product"))
            Xi, Ei, _ = base.limit_dist.ignore_virtual_classes(sX.clone(), sE.clone())
            rr = rew(Xi, Ei, nr, cr).to(dev).float().reshape(r, a.states)
            zx.append(sX.view(r, a.states, *sX.shape[1:]))
            ze.append(sE.view(r, a.states, *sE.shape[1:])); rw.append(rr)
            done += r
    ZX, ZE, R = torch.cat(zx), torch.cat(ze), torch.cat(rw)
    em = edge_mask_of(nm)

    with torch.no_grad():
        lmX, lmE = head(X_t, E_t, t, nm, cond)
        clsX = X1.argmax(-1)
        pcX = per_class_baseline((a.lam*R.mean(0))[:,None].expand_as(clsX), clsX,
                                 X1.shape[-1], nm)
        pcX = torch.where(torch.isfinite(pcX), pcX, torch.zeros_like(pcX))

    def shift(logm, shuffle=False, w=None):
        base_m = ZX.mean(0)
        if w is None:
            w = torch.softmax(a.lam * R, dim=0)
            if shuffle:
                w = w[torch.randperm(w.shape[0], device=w.device)]
            return (w[:,:,None,None] * ZX).sum(0) - base_m
        q = base_m * logm.exp()
        return q / q.sum(-1, keepdim=True).clamp_min(1e-12) - base_m

    D_emp = shift(None)
    D_null = shift(None, shuffle=True)
    D_head = shift(lmX, w=1)
    D_pc = shift(pcX.expand_as(lmX), w=1)

    print(f"Gate 2  |  {a.states} states x {a.k} completions, eta={a.eta:g}, "
          f"t_int={a.t_int}/{a.steps}, t={float(t[0,0]):.3f}")
    print(f"  {'predictor':10s} | {'raw':>7s} {'resid-2':>8s}")
    res = {}
    for tag, D in (("head", D_head), ("per-class", D_pc), ("null", D_null)):
        r_raw = corr(D[nm], D_emp[nm])
        r_r2 = corr(resid2(D, nm, a.states)[nm], resid2(D_emp, nm, a.states)[nm])
        res[tag] = {"raw": r_raw, "resid2": r_r2}
        print(f"  {tag:10s} | {r_raw:+7.3f} {r_r2:+8.3f}")
    print(f"  ceiling (split-half reliability): {a.ceiling:+.3f}")
    ok = res["head"]["resid2"] >= a.pass_at and \
        res["head"]["resid2"] > res["per-class"]["resid2"] + 0.05
    print(f"  -> {'PASS' if ok else 'FAIL'}  "
          f"(need resid-2 >= {a.pass_at} and > per-class + 0.05)")
    if a.out:
        json.dump({"gate2": res, "pass": bool(ok), "args": vars(a)}, open(a.out,"w"), indent=1)
    print("GATE2-DONE", flush=True)


if __name__ == "__main__":
    main()
