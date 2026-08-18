#!/usr/bin/env python
"""Fit a LearnedSizeDistribution matched to an already-shipped adapter.

`train_property_head.py --size-only` wants a bare DeFoGModel checkpoint for `--base`;
a shipped base lives in the molsmith store as safetensors instead. This loads the base
through molsmith the same way `e2_targeting.py` does, so the vocabulary, the encoders and
the node-count ceiling all come from the package the adapter is actually bound to.

THE POINT OF THIS SCRIPT is the label convention. An adapter conditioned on decoded logP
paired with a size model fit on source logP disagree by the ~0.37 source-vs-decoded offset,
and they disagree WORST at the extremes -- which is exactly where the size draw matters.
So the fit ends with a hard check that the fitted cond_mean/cond_std match the adapter's,
and refuses to save if they do not.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
from rdkit import Chem, RDLogger                          # noqa: E402
from rdkit.Chem import Crippen, Descriptors, QED          # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROP_FNS = {
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "qed": lambda m: float(QED.qed(m)),
    "tpsa": lambda m: float(Descriptors.TPSA(m)),
}


def _measure(smiles, args, loaded, prop_fn):
    """(property, node count) per molecule, through the SAME encode/decode round trip
    the adapter's own training used. This is the expensive part -- six RDKit operations
    per molecule -- which is why it is worth caching."""
    from defog.domains.molecule import (mol_to_smiles, pyg_data_to_mol,
                                        smiles_to_pyg_data)
    atom_enc, atom_dec = loaded.atom_encoder, loaded.domain.atom_decoder
    bond_enc, bond_dec = loaded.bond_encoder, loaded.domain.bond_decoder
    kekulize = not any(str(b).upper() == "AROMATIC" for b in bond_enc)

    vals, sizes, skipped = [], [], 0
    t0 = time.time()
    for i, smi in enumerate(smiles):
        if i and i % 25000 == 0:
            print(f"  {i}/{len(smiles)} measured ({time.time()-t0:.0f}s)", flush=True)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue
        data = smiles_to_pyg_data(smi, atom_enc, bond_enc, kekulize=kekulize)
        if data is None:
            skipped += 1
            continue
        target_mol = mol
        if args.property_from == "decoded":
            dec = pyg_data_to_mol(data, atom_dec, bond_dec)
            back = mol_to_smiles(dec) if dec is not None else None
            target_mol = Chem.MolFromSmiles(back) if back else None
            if target_mol is None:
                skipped += 1
                continue
        try:
            vals.append(prop_fn(target_mol))
        except Exception:                                  # noqa: BLE001
            skipped += 1
            continue
        sizes.append(int(data.x.size(0)))
    print(f"  measured {len(vals)} in {time.time()-t0:.0f}s", flush=True)
    return np.asarray(vals, dtype="float32"), np.asarray(sizes), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--property", required=True, choices=sorted(PROP_FNS))
    ap.add_argument("--property-from", default="decoded", choices=("source", "decoded"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="The model is a 2x512 MLP over a scalar; at batch 256 the "
                         "per-step overhead dwarfs the arithmetic.")
    ap.add_argument("--cache", default=None,
                    help="npz of (vals, sizes). Written if absent, reused if present -- "
                         "the RDKit encode/decode round trip is the expensive part and "
                         "does not change between fits.")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max allowed |fitted - adapter| on cond_mean/cond_std")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from molsmith import sample as ms
    from defog.core.property_head import fit_size_model
    from defog.data import zinc_reference as zref

    cfg = ms.SamplingConfig(base=args.base, n=1,
                            adapters=[ms.AdapterTarget(package=args.adapter, target=0.0)])
    loaded = ms.load(cfg)
    adapter = loaded.adapters[args.adapter]
    base = loaded.base
    a_mean = float(adapter.cond_mean.reshape(-1)[0])
    a_std = float(adapter.cond_std.reshape(-1)[0])
    print(f"adapter {args.adapter}: cond_mean {a_mean:.6f}  cond_std {a_std:.6f}")
    print(f"base max_nodes {base.max_nodes}")

    prop_fn = PROP_FNS[args.property]
    smiles = zref.load_reference_split().train_smiles
    if args.limit:
        smiles = smiles[:args.limit]

    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        z = np.load(cache)
        vals, sizes, skipped = z["vals"], z["sizes"], 0
        print(f"reusing cached labels from {cache} ({len(vals)} molecules)")
    else:
        vals, sizes, skipped = _measure(smiles, args, loaded, prop_fn)
        if cache is not None:
            np.savez(cache, vals=vals, sizes=sizes)
            print(f"cached labels -> {cache}")

    vals = np.asarray(vals, dtype="float32")
    sizes = np.asarray(sizes)
    print(f"{len(vals)} molecules (skipped {skipped}); "
          f"{args.property}[{args.property_from}] mean {vals.mean():.6f} std {vals.std():.6f}")

    # The check that makes this script worth having: does the label convention MATCH?
    d_mean, d_std = abs(vals.mean() - a_mean), abs(vals.std() - a_std)
    print(f"vs adapter: d(mean) {d_mean:.6f}  d(std) {d_std:.6f}  (tol {args.tol})")
    if d_mean > args.tol or d_std > args.tol:
        sys.exit(
            f"REFUSING to save. The labels this fit produced do not match what the adapter "
            f"was conditioned on, so the size draw and the steering would disagree -- worst "
            f"at the extremes, where the size draw matters most. Try "
            f"--property-from {'source' if args.property_from == 'decoded' else 'decoded'}."
        )

    hi = min(int(sizes.max()), int(base.max_nodes))
    keep = sizes <= hi
    model, m = fit_size_model(
        torch.from_numpy(vals[keep]), torch.from_numpy(sizes[keep]),
        min_size=int(sizes.min()), max_size=hi, hidden=args.hidden, layers=args.layers,
        epochs=args.epochs, seed=args.seed, device=args.device,
        batch_size=args.batch_size,
        cond_mean=[a_mean], cond_std=[a_std],      # adopt the adapter's, do not re-derive
        property_name=args.property, property_from=args.property_from,
    )
    print(f"grid {m['min_size']}..{m['max_size']}  held-out NLL {m['nll_val']:.4f} vs "
          f"marginal {m['nll_marginal']:.4f}  -> gain {m['gain_nats']:+.4f} nats  "
          f"shrink {m['shrink']:.3f}")
    if m["gain_nats"] < 0.02:
        print("WARNING: this property carries essentially no size information.")

    model.check_compatible(adapter)          # the pairing guard, as a hard assert
    print("check_compatible(adapter): OK")
    out = model.save(args.out)
    Path(str(out).replace(".ckpt", ".json")).write_text(json.dumps(
        {k: v for k, v in m.items() if k != "history"} |
        {"adapter": args.adapter, "property_from": args.property_from}, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
