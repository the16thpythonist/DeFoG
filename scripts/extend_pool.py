"""Add completions to an existing conditional pool.

Round 5 regressed a target whose split-half reliability is ~0.00 at K=6-8 -- two
independent halves of the completions give UNCORRELATED tilt targets (logp +0.002,
oxy +0.019). The head was asked to fit noise, so its failure says nothing about whether
credit is amortisable.

Reliability rises with K, so this extends the SAME states rather than building new ones.
Training at K = 8, 16, 32, 64 on identical states then gives a dose-response, which is
far more convincing than two independent runs -- and if Gate 1 and Gate 2 stay flat as
reliability climbs, amortisation is dead for a real reason.

The base rollouts are not repeated; only completions are added.
"""
import argparse, os, time

import torch

from defog.core import DeFoGModel, AdaLNAdapter
from defog.core.adapter import AdapterComposition, ConditionBranch
from defog.core.dam import simulate_to_end
from defog.core.rl import PropertyMatchReward
from defog.core.sampler import Sampler
from defog.domains.molecule import build_encoders
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Crippen

ATOMS = ["C","N","O","F","P","S","Cl","Br","I"]; BONDS = ["SINGLE","DOUBLE","TRIPLE"]


def recover_t_int(base, t_vals, steps, td):
    """The pool stores the continuous t but not the grid index the completions must
    resume from. Rebuild the grid and match -- draw_times(mode='match') emits exact
    grid values, so this is a lookup and not an approximation."""
    probe = Sampler(base, sample_steps=steps, time_distortion=td, eta=0.0, omega=0.0)
    grid = []
    for k in range(steps):
        tn, _ = probe._step_times(k, 1, t_vals.device)
        grid.append(float(tn.reshape(-1)[0]))
    g = torch.tensor(grid, device=t_vals.device)
    return (t_vals.reshape(-1, 1) - g.reshape(1, -1)).abs().argmin(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--target-k", type=int, default=64)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eta", type=float, default=30.0)
    p.add_argument("--base", default="ckpts/zinc_e1_seed42_kek.ckpt")
    p.add_argument("--adapter", default="ckpts/clogp_v11/clogp_adapter.ckpt")
    p.add_argument("--out", default="")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = a.out or a.pool.replace(".pt", f"_k{a.target_k}.pt")

    base = DeFoGModel.load(a.base, device="cpu").to(dev).eval()
    adapter = AdaLNAdapter.load(a.adapter, device=dev)
    _, adec, _, bdec = build_encoders(ATOMS, BONDS)
    rew = PropertyMatchReward(adec, bdec, lambda m: float(Crippen.MolLogP(m)), scale=1.0)

    pool = torch.load(a.pool, weights_only=False)
    if os.path.exists(out):
        pool = torch.load(out, weights_only=False)
        print(f"resuming from {out}", flush=True)
    S, K0 = pool["t"].shape[0], pool["Z_X"].shape[0]
    need = a.target_k - K0
    print(f"pool {S} states, K={K0} -> {a.target_k} ({need} to add)", flush=True)
    if need <= 0:
        print("nothing to do\nEXTEND-DONE", flush=True); return

    t_int = recover_t_int(base, pool["t"].to(dev), a.steps, "polydec").cpu()
    print(f"recovered grid indices: min {int(t_int.min())} max {int(t_int.max())}",
          flush=True)

    # Pre-allocate ONCE, before the loop. The first version grew the arrays inside the
    # loop under a conditional, which is unreadable and a resume would have re-grown
    # them. `filled` marks which states already have their extra completions, so a
    # resumed run skips them.
    if pool["Z_X"].shape[0] == K0:
        pool["Z_X"] = torch.cat([pool["Z_X"],
                                 torch.zeros(need, S, *pool["Z_X"].shape[2:])], 0)
        pool["Z_E"] = torch.cat([pool["Z_E"],
                                 torch.zeros(need, S, *pool["Z_E"].shape[2:])], 0)
        pool["reward"] = torch.cat([pool["reward"], torch.zeros(need, S)], 0)
        pool["filled"] = torch.zeros(S, dtype=torch.bool)
    filled = pool.setdefault("filled", torch.zeros(S, dtype=torch.bool))

    t0 = time.time()
    for s0 in range(0, S, a.batch):
        sl = slice(s0, min(s0 + a.batch, S))
        if bool(filled[sl].all()):
            continue
        b = sl.stop - sl.start
        ti = int(t_int[sl].mode().values)          # states in a batch share their index
        X_t = pool["X_t"][sl].to(dev); E_t = pool["E_t"][sl].to(dev)
        nm = pool["node_mask"][sl].to(dev); cond = pool["cond"][sl].to(dev)
        zx, ze, rr = [], [], []
        done = 0
        while done < need:
            r = min(max(1, a.chunk // max(b, 1)), need - done)
            cr, nr = cond.repeat(r, 1), nm.repeat(r, 1)
            with torch.no_grad():
                sX, sE = simulate_to_end(
                    base, X_t.repeat(r,1,1), E_t.repeat(r,1,1,1),
                    torch.zeros(r*b, 0, device=dev), nr, ti,
                    sample_steps=a.steps, time_distortion="polydec",
                    eta=a.eta, omega=0.0,
                    composition=AdapterComposition(
                        [ConditionBranch(adapter, cr, 1.0)], base=base, mode="product"))
                Xi, Ei, _ = base.limit_dist.ignore_virtual_classes(sX.clone(), sE.clone())
                rv = rew(Xi, Ei, nr, cr).to(dev).float().reshape(r, b)
            zx.append(sX.view(r, b, *sX.shape[1:]).cpu())
            ze.append(sE.view(r, b, *sE.shape[1:]).cpu()); rr.append(rv.cpu())
            done += r
        pool["Z_X"][K0:, sl] = torch.cat(zx)
        pool["Z_E"][K0:, sl] = torch.cat(ze)
        pool["reward"][K0:, sl] = torch.cat(rr)
        filled[sl] = True
        el = time.time() - t0
        print(f"  states {sl.stop}/{S}  t_int={ti}  [{el:.0f}s, eta "
              f"{el/max(sl.stop,1)*(S-sl.stop):.0f}s]", flush=True)
        if (s0 // a.batch) % 4 == 3:
            torch.save(pool, out + ".tmp"); os.replace(out + ".tmp", out)
    torch.save(pool, out + ".tmp"); os.replace(out + ".tmp", out)
    print(f"wrote {out}\nEXTEND-DONE", flush=True)


if __name__ == "__main__":
    main()
