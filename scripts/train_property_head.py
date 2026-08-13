#!/usr/bin/env python
"""
Fit a standalone PropertyHead against a frozen base's vocabulary.

Why this exists separately from adapter training
------------------------------------------------
A head is a graph -> scalar regressor. It does NOT need the adapter it will be
bundled with: `fit_property_head` trains it by grounding regression on the same
(graph, label) pairs the adapter sees, with no adapter in the loop. So a head can
be added to an adapter that already exists, which is exactly the situation on
molsmith/zinc-kek -- clogp@1.1.0 and the fingerprint adapters all shipped without
one, and FK steering needs it.

The head is what `LearnedPropertyEnergy` turns into a Feynman-Kac `energy_fn`:
each predicted-clean particle is scored by the squared error of the head's
prediction to the target. FK feeds it the DISCRETE one-hot predicted-clean graph,
which is why a head works there where soft-input coupling did not.

THE LABELS MUST MATCH THE ADAPTER'S. This mirrors adapter_training__zinc.py's
pipeline exactly -- same vocabulary resolution, same kekulize flag, same
`property_from` convention. A head trained on source-SMILES labels and paired with
an adapter trained on decoded-graph labels would disagree about what the target
means at precisely the end of the range where the charge loss bites, and the FK
energy would then pull against the adapter rather than with it.

Usage:
    python scripts/train_property_head.py --base ckpts/zinc_rl2_seed42/best_model \\
        --vocabulary e1_kekulized --property qed --property-from decoded \\
        --out ckpts/heads/qed_head
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

from defog.core import DeFoGModel  # noqa: E402
from defog.core.property_head import PropertyHead, fit_property_head  # noqa: E402
from defog.data import vocabulary  # noqa: E402
from defog.domains.molecule import pyg_data_to_mol, mol_to_smiles  # noqa: E402
from experiments.utils import build_encoders, smiles_to_pyg_data  # noqa: E402

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_training_module():
    """Reuse adapter_training__zinc's vocabulary table and property functions rather
    than restating them. A second copy is a place for them to drift, and a drifted
    property function produces a head that is confidently wrong."""
    spec = importlib.util.spec_from_file_location(
        "_at", _PROJECT_DIR / "experiments" / "adapter_training__zinc.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["_at"] = m
    spec.loader.exec_module(m)
    return m


def _fit_size(args, vals, node_counts, fit_idx, base, device, out):
    """Fit and save the learned ``P(n | property)`` beside the head.

    Uses the same rows, the same RAW property values, and therefore the same
    ``--property-from`` convention as the head -- which is why this lives here rather
    than in its own script.

    The grid ceiling is the base's ``max_nodes``: a size model must not put mass on a
    graph the base cannot build.
    """
    from defog.core.property_head import fit_size_model

    y = vals[fit_idx].astype("float32")
    n = node_counts[fit_idx]
    ceiling = int(getattr(base, "max_nodes", int(n.max())))
    lo, hi = int(n.min()), min(int(n.max()), ceiling)
    keep = n <= hi
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} molecules above the base's max_nodes={ceiling}")
    t0 = time.time()
    model, m = fit_size_model(
        torch.from_numpy(y[keep]), torch.from_numpy(n[keep]),
        min_size=lo, max_size=hi, hidden=args.size_hidden, layers=args.size_layers,
        epochs=args.size_epochs, lr=args.size_lr, seed=args.seed, device=device,
        property_name=args.property, property_from=args.property_from,
    )
    size_out = out.replace(".ckpt", "_size.ckpt")
    model.save(size_out)
    print(f"size model: grid {lo}..{hi}, "
          f"{sum(p.numel() for p in model.parameters()):,} params, "
          f"{time.time()-t0:.0f}s")
    print(f"  held-out NLL {m['nll_val']:.4f} vs marginal {m['nll_marginal']:.4f}  "
          f"-> gain {m['gain_nats']:+.4f} nats   shrink {m['shrink']:.3f}")
    # The whole reason gain_nats is computed. A size model that does not beat the
    # marginal is not a neutral addition -- it is a second thing to keep in sync that
    # buys nothing, and shipping it makes the E2 ablation row unreadable.
    if m["gain_nats"] < 0.02:
        print("WARNING: this property carries essentially no information about graph "
              "size. Do NOT use the learned size draw for it; the marginal is as good "
              "and has no moving parts.")
    print(f"saved {size_out}")
    return {k: v for k, v in m.items() if k != "history"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Frozen base, WITHOUT .ckpt")
    ap.add_argument("--vocabulary", default="e1_kekulized")
    ap.add_argument("--property", required=True)
    ap.add_argument("--property-from", default="decoded", choices=("source", "decoded"))
    ap.add_argument("--csv", default=str(_PROJECT_DIR / "data" / "zinc_250k_rdkit.csv"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=5000,
                    help="Molecules held out to report an honest MAE. The fit itself "
                         "reports training loss, which says nothing about whether the "
                         "head generalises -- and a head that does not is an FK energy "
                         "that steers toward its own errors.")
    ap.add_argument("--limit", type=int, default=None, help="Cap molecules (smoke runs)")
    ap.add_argument("--out", required=True, help="Output checkpoint, WITHOUT .ckpt")
    # -- learned P(n | property), written alongside the head as <out>_size.ckpt --------
    # Fit from the SAME (property, molecule) pairs the head is fit from, which is the
    # point of putting it here: the size model inherits --property-from for free, and a
    # size model fit under one label convention paired with an adapter trained under
    # another is exactly the mismatch that bites only at the extremes.
    ap.add_argument("--with-size-model", action="store_true",
                    help="Also fit a LearnedSizeDistribution -> <out>_size.ckpt")
    ap.add_argument("--size-only", action="store_true",
                    help="Fit ONLY the size model (retrofit for an already-shipped head)")
    ap.add_argument("--size-hidden", type=int, default=512)
    ap.add_argument("--size-layers", type=int, default=2)
    ap.add_argument("--size-epochs", type=int, default=200)
    ap.add_argument("--size-lr", type=float, default=1e-3)
    args = ap.parse_args()

    at = _load_training_module()
    if args.property not in at.PROP_FNS:
        sys.exit(f"unknown property {args.property!r}; have {sorted(at.PROP_FNS)}")
    prop_fn = at.PROP_FNS[args.property]
    atom_types, bond_types, kekulize, source = at._vocabulary(args.vocabulary)
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, bond_types)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = DeFoGModel.load(args.base, device="cpu")
    # The head's input width IS the base's class counts. A mismatch here trains a
    # head on channels that mean different elements -- it converges and is useless.
    print(vocabulary.check_model(base, atom_types, bond_types, what=f"base {args.base}"))
    na = int(base.output_dims["X"])
    nb = int(base.output_dims["E"])
    print(f"vocabulary '{args.vocabulary}': {na} atom / {nb} edge classes; "
          f"property={args.property} from={args.property_from}")

    if source == "reference_split":
        from defog.data import zinc_reference as zref
        smiles = list(zref.load_reference_split().train_smiles)
    else:
        import pandas as pd
        smiles = pd.read_csv(args.csv)["smiles"].tolist()
    if args.limit:
        smiles = smiles[:args.limit]
    print(f"source molecules: {len(smiles)}")

    graphs, vals, node_counts, n_skipped = [], [], [], 0
    t0 = time.time()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_skipped += 1
            continue
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder, kekulize=kekulize)
        if data is None:
            n_skipped += 1
            continue
        target_mol = mol
        if args.property_from == "decoded":
            dec = pyg_data_to_mol(data, atom_decoder, bond_decoder)
            back = mol_to_smiles(dec) if dec is not None else None
            target_mol = Chem.MolFromSmiles(back) if back else None
            if target_mol is None:
                n_skipped += 1
                continue
        try:
            v = prop_fn(target_mol)
        except Exception:                                   # noqa: BLE001
            n_skipped += 1
            continue
        data.cond = torch.tensor([[v]], dtype=torch.float)  # RAW target, as the pipeline builds it
        graphs.append(data)
        vals.append(v)
        node_counts.append(int(data.x.size(0)))
    vals = np.asarray(vals)
    node_counts = np.asarray(node_counts)
    print(f"{len(graphs)} graphs (skipped {n_skipped}) in {time.time()-t0:.0f}s; "
          f"{args.property} mean={vals.mean():.4f} std={vals.std():.4f} "
          f"range=[{vals.min():.4f}, {vals.max():.4f}]")
    if n_skipped > 0.02 * max(1, n_skipped + len(graphs)):
        sys.exit(f"REFUSING: {n_skipped} molecules failed to encode. Under a matching "
                 f"vocabulary this is near zero -- check --vocabulary.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(graphs))
    n_hold = min(args.holdout, len(graphs) // 5)
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]
    fit_graphs = [graphs[i] for i in fit_idx]
    hold_graphs = [graphs[i] for i in hold_idx]
    print(f"fit on {len(fit_graphs)}, held out {len(hold_graphs)}")

    out = args.out if args.out.endswith(".ckpt") else args.out + ".ckpt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    size_meta = None
    if args.with_size_model or args.size_only:
        size_meta = _fit_size(args, vals, node_counts, fit_idx, base, device, out)

    if args.size_only:
        print("--size-only: skipping the head.")
        return 0

    head = PropertyHead(na, nb, hid=args.hidden, layers=args.layers,
                        prop_mean=float(vals[fit_idx].mean()),
                        prop_std=float(vals[fit_idx].std())).to(device)
    print(f"head: {sum(p.numel() for p in head.parameters()):,} params "
          f"(hid={args.hidden}, layers={args.layers})")

    t0 = time.time()
    fit_property_head(head, fit_graphs, epochs=args.epochs, lr=args.lr,
                      batch_size=args.batch_size, seed=args.seed, device=device)
    print(f"fit done in {(time.time()-t0)/60:.1f} min")

    # Held-out MAE, in the property's own units. This is the number that says whether
    # the head is usable as an FK energy; training loss does not.
    head.eval()
    from torch_geometric.loader import DataLoader
    from defog.core.data import to_dense
    preds, trues = [], []
    with torch.no_grad():
        for batch in DataLoader(hold_graphs, batch_size=256, shuffle=False):
            dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            dense = dense.mask(mask)
            p = head.predict(dense.X.to(device), dense.E.to(device), mask.to(device))
            preds.append(p.detach().cpu().numpy().reshape(-1))
            trues.append(batch.cond.numpy().reshape(-1))
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    mae = float(np.abs(preds - trues).mean())
    corr = float(np.corrcoef(preds, trues)[0, 1])
    spread = float(vals.std())
    print(f"held-out MAE {mae:.4f}   corr {corr:.4f}   (property std {spread:.4f}, "
          f"so MAE/std = {mae/spread:.3f})")
    # A head no better than predicting the mean is not a ruler. Say so loudly rather
    # than shipping it and wondering later why FK steering does nothing.
    if mae >= spread:
        print("WARNING: MAE is not better than predicting the dataset mean. This head "
              "carries no signal and would make LearnedPropertyEnergy a no-op.")

    # PropertyHead.save/load own the format; do not hand-roll a dict here or the
    # head will not load through PropertyHead.load, which is what molsmith uses.
    head.save(out)
    meta = {"property": args.property, "property_from": args.property_from,
            "vocabulary": args.vocabulary, "base": args.base, "seed": args.seed,
            "n_fit": len(fit_graphs), "n_holdout": len(hold_graphs),
            "holdout_mae": mae, "holdout_corr": corr, "property_std": spread,
            "epochs": args.epochs, "lr": args.lr, "hidden": args.hidden,
            "layers": args.layers, "size_model": size_meta}
    Path(out.replace(".ckpt", ".json")).write_text(json.dumps(meta, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
