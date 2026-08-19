#!/usr/bin/env python
"""
Fit a PropertyHead that scores the graphs DAM actually looks at.

WHY. DAM's adjoint compares the reward of candidate clean graphs drawn from the
base's own head. Measured on zinc-kek, 84-98% of those draws fail to decode, and
PropertyMatchReward assigns every failure the same floor (-10). With g(Z) == g(X1_k)
for nearly every sample the adjoint collapses to 1 and the update has no signal at
all -- which is what the in-situ residual showed (median 1.21-1.28, i.e. worse than
doing nothing).

PropertyHead.forward already takes dense tensors and needs no RDKit, so it CAN score
those graphs. What it cannot do is score them WELL: rl.py:922 notes the head is
trained on re-encoded valid molecules and mispredicts on raw generated graphs. This
script removes that mismatch by training on the draws themselves.

LABELS. Each drawn graph gets:
  * its OWN RDKit property when it parses -- ground truth, and what keeps the head
    honest on valid molecules (the acceptance criterion);
  * otherwise the property of the ROLLOUT ENDPOINT it was drawn from -- "the molecule
    you are a corruption of scored this", which is exactly the graded signal the
    floored region is missing.

The fraction of each is reported, because a head trained overwhelmingly on surrogate
labels is a different object from one trained on ground truth.

SCOPE. This head is for DAM's adjoint only -- g(Z) and g(X1_k). The RL reward on
rollout endpoints stays RDKit ground truth, so the GDPO and RAM arms are untouched
and stay comparable to the historical runs.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defog.core import AdaLNAdapter, DeFoGModel                       # noqa: E402
from defog.core.data import dense_to_pyg                              # noqa: E402
from defog.core.noise import sample_from_probs                        # noqa: E402
from defog.core.property_head import PropertyHead, fit_property_head  # noqa: E402
from defog.core.renoise import draw_times, renoise_states             # noqa: E402
from defog.core.rl import RolloutSampler                              # noqa: E402

KEK_ATOMS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
KEK_BONDS = ["SINGLE", "DOUBLE", "TRIPLE"]


def rdkit_prop(X1, E1, node_mask, adec, bdec, prop_fn):
    """Per-graph RDKit property, or None where the graph does not decode."""
    from rdkit import Chem

    from defog.domains.molecule import mol_to_smiles, pyg_data_to_mol
    n = node_mask.sum(-1)
    datas = dense_to_pyg(X1, E1, None, node_mask, n)
    out = []
    for d in datas:
        mol = pyg_data_to_mol(d, adec, bdec)
        smi = mol_to_smiles(mol) if mol is not None else None
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None or "." in (smi or "."):
            out.append(None)
            continue
        try:
            out.append(float(prop_fn(m)))
        except Exception:
            out.append(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--property", default="logp", choices=("logp", "qed", "tpsa"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoints", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rollout-steps", type=int, default=250)
    ap.add_argument("--t-per-endpoint", type=int, default=4)
    ap.add_argument("--draws-per-state", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from rdkit import RDLogger
    from rdkit.Chem import Crippen, Descriptors, QED
    RDLogger.DisableLog("rdApp.*")
    prop_fn = {"logp": lambda m: float(Crippen.MolLogP(m)),
               "qed": lambda m: float(QED.qed(m)),
               "tpsa": lambda m: float(Descriptors.TPSA(m))}[args.property]

    from defog.domains.molecule import build_encoders
    _, adec, _, bdec = build_encoders(KEK_ATOMS, KEK_BONDS)

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    base = DeFoGModel.load(args.base, device="cpu").to(dev).eval()
    dx, de = base.limit_dist.num_node_classes, base.limit_dist.num_edge_classes
    if (dx, de) != (len(KEK_ATOMS), len(KEK_BONDS) + 1):
        raise SystemExit(f"vocabulary mismatch: base has dx={dx} de={de}, "
                         f"expected {len(KEK_ATOMS)}/{len(KEK_BONDS) + 1} for kek")
    adapter = AdaLNAdapter.load(args.adapter, device=dev) if args.adapter else None
    print(f"base dx={dx} de={de} | adapter={'yes' if adapter else 'no'} | dev={dev}")

    graphs, own, surrogate, dropped = [], 0, 0, 0
    done = 0
    while done < args.endpoints:
        k = min(args.batch, args.endpoints - done)
        comp = None
        if adapter is not None:
            from defog.core.adapter import AdapterComposition, ConditionBranch
            cond = (torch.rand(k, adapter.cond_dim, device=dev) * 4.0 - 1.0)
            comp = AdapterComposition([ConditionBranch(adapter, cond, 1.0)],
                                      base=base, mode="product")
        s = RolloutSampler(base, eta=1.0, omega=0.0, sample_steps=args.rollout_steps,
                           time_distortion="polydec", record_trace=False)
        if comp is not None:
            s.composition = comp
        s.sample(k, condition=(cond if adapter is not None else None),
                 device=dev, show_progress=False)
        X1, E1 = s.endpoint
        nm = s.end_node_mask
        ep_prop = rdkit_prop(X1, E1, nm, adec, bdec, prop_fn)

        keep = [i for i, v in enumerate(ep_prop) if v is not None]
        dropped += k - len(keep)
        if not keep:
            done += k
            continue
        idx = torch.tensor(keep, device=dev)
        X1, E1, nm = X1[idx], E1[idx], nm[idx]
        lab_ep = torch.tensor([ep_prop[i] for i in keep], device=dev)
        y0 = torch.zeros(len(keep), 0, device=dev)

        # score at the t values the trainer scores at: late-weighted, polydec grid
        w = torch.linspace(0.2, 1.0, args.rollout_steps) ** 2
        picks = torch.multinomial(w, args.t_per_endpoint, replacement=False).tolist()
        times = draw_times(base, len(keep), dev, mode="match", step_indices=picks,
                           sample_steps=args.rollout_steps, time_distortion="polydec")
        for (X_t, E_t, t) in renoise_states(base, X1, E1, y0, nm, times):
            nz = {"X_t": X_t, "E_t": E_t, "y_t": y0, "t": t, "node_mask": nm}
            with torch.no_grad():
                pr = base.forward(nz, base._compute_extra_data(nz), nm)
                px, pe = F.softmax(pr.X, -1), F.softmax(pr.E, -1)
            em = (nm[:, :, None] & nm[:, None, :]).float()
            for _ in range(args.draws_per_state):
                sd = sample_from_probs(px, pe, nm)
                Xd = F.one_hot(sd.X, dx).float() * nm[..., None]
                Ed = F.one_hot(sd.E, de).float() * em[..., None]
                p_own = rdkit_prop(Xd, Ed, nm, adec, bdec, prop_fn)
                n_nodes = nm.sum(-1)
                datas = dense_to_pyg(Xd, Ed, None, nm, n_nodes)
                for j, d in enumerate(datas):
                    v = p_own[j]
                    if v is None:
                        v = float(lab_ep[j]); surrogate += 1
                    else:
                        own += 1
                    d.cond = torch.tensor([[float(v)]])
                    graphs.append(d)
        done += k
        print(f"  endpoints {done}/{args.endpoints} | graphs {len(graphs)} "
              f"| own-labelled {own} surrogate {surrogate} | undecodable endpoints {dropped}")

    tot = own + surrogate
    print(f"\ndataset: {len(graphs)} graphs | own-labelled {own / tot:.1%} "
          f"surrogate {surrogate / tot:.1%} | endpoints dropped {dropped}")
    if surrogate / tot < 0.2:
        print("  NOTE: few surrogate labels -- the floored region this head exists to "
              "grade is barely represented. Check the t distribution.")

    vals = np.array([float(g.cond) for g in graphs])
    head = PropertyHead(dx, de, hid=args.hid, layers=args.layers,
                        prop_mean=float(vals.mean()), prop_std=float(vals.std()) or 1.0)
    print(f"label stats: mean {vals.mean():.3f} sd {vals.std():.3f} "
          f"[{vals.min():.2f}, {vals.max():.2f}]")
    fit_property_head(head, graphs, epochs=args.epochs, seed=args.seed,
                      device=str(dev), progress=lambda *a, **k: None)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "na": dx, "nb": de,
                "hid": args.hid, "layers": args.layers,
                "prop_mean": float(vals.mean()), "prop_std": float(vals.std()) or 1.0,
                "property": args.property, "base": args.base, "adapter": args.adapter,
                "n_graphs": len(graphs), "own_frac": own / tot}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
