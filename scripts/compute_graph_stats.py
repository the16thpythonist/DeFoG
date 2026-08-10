#!/usr/bin/env python
"""
Node/edge marginals and size histogram for a SMILES file under a representation.

The marginals are not bookkeeping -- with ``noise_type="marginal"`` they *are*
the prior the model denoises from, so they have to describe the graphs the model
actually sees. Changing the bond vocabulary changes them: dropping the AROMATIC
class moves ~half the real bond mass into single/double, so a kekulized run
started from the aromatic ``chembl_stats.json`` would be denoising from the
wrong prior. (Node marginals and the size histogram are unaffected -- same atoms,
same molecule sizes -- but they are recomputed here anyway so one file describes
one representation completely.)

Counts come from ``smiles_to_pyg_data``, the same encoder training uses, rather
than from a parallel RDKit pass. Two reasons: the two cannot drift, and a
molecule the encoder rejects is automatically excluded from the marginals -- which
is correct, because it is excluded from training too.

Usage:
    python scripts/compute_graph_stats.py --dataset chembl \
        --representation kekulized_v2 \
        --smiles data/chembl/chembl_train.smiles \
        --out data/chembl/chembl_kek_stats.json
"""
import argparse
import collections
import gc
import importlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import RDLogger  # noqa: E402

from defog.domains.molecule import build_encoders, smiles_to_pyg_data  # noqa: E402

REFERENCES = {"zinc": "zinc_reference", "guacamol": "guacamol_reference",
              "moses": "moses_reference", "chembl": "chembl_reference"}

_W = {}


def _init_worker(atom_types, bond_types, kekulize):
    RDLogger.DisableLog("rdApp.*")
    ae, _, be, _ = build_encoders(list(atom_types), list(bond_types))
    _W.update(atom_encoder=ae, bond_encoder=be, kekulize=kekulize,
              n_atom=len(atom_types), n_bond=len(bond_types))


def count_one(smiles):
    """(n_heavy, node_counts, bond_counts) or None if the encoder rejects it."""
    data = smiles_to_pyg_data(smiles, _W["atom_encoder"], _W["bond_encoder"],
                              kekulize=_W["kekulize"])
    if data is None:
        return None
    node = data.x.sum(0).long().tolist()
    # edge_index carries both directions of every bond, and class 0 (no edge) is
    # reserved and never emitted -- so drop it and halve.
    per_class = data.edge_attr.sum(0).long().tolist()[1:]
    bonds = [c // 2 for c in per_class]
    return (int(data.x.shape[0]), node, bonds)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(REFERENCES))
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--representation", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=2000)
    args = ap.parse_args()

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    mod = importlib.import_module(f"defog.data.{REFERENCES[args.dataset]}")
    if hasattr(mod, "get_representation"):
        rep = mod.get_representation(args.representation)
        atom_types, bond_types, kekulize = rep.atom_types, rep.bond_types, rep.kekulize
        rep_name = rep.name
    else:
        atom_types, bond_types = list(mod.ATOM_TYPES), list(mod.BOND_TYPES)
        kekulize = "AROMATIC" not in bond_types
        rep_name = "default"

    smiles = [ln.strip() for ln in open(args.smiles) if ln.strip()]
    print(f"{args.dataset}/{rep_name}: {len(smiles):,} molecules, {workers} workers")
    print(f"  atoms={atom_types}")
    print(f"  bonds={bond_types} kekulize={kekulize}")

    node_tot = [0] * len(atom_types)
    bond_tot = [0] * len(bond_types)
    size_hist = collections.Counter()
    total_pairs = total_bonds = n_ok = n_skip = 0

    # Before forking: the workers inherit the whole SMILES list copy-on-write and
    # never read it (they only get pickled chunks), but a child's cyclic GC writes
    # to every inherited object header and copies the page. At union scale (~100M
    # strings, ~10 GB) times 16 workers that is fatal; freeze() moves the list into
    # a generation the collector never visits. See prepare_smiles_union.py.
    gc.freeze()
    with Pool(workers, initializer=_init_worker,
              initargs=(atom_types, bond_types, kekulize)) as pool:
        for res in pool.imap_unordered(count_one, smiles, chunksize=args.chunksize):
            if res is None:
                n_skip += 1
                continue
            n_ok += 1
            nh, node, bonds = res
            for i, c in enumerate(node):
                node_tot[i] += c
            for i, c in enumerate(bonds):
                bond_tot[i] += c
            size_hist[nh] += 1
            total_pairs += nh * (nh - 1) // 2
            total_bonds += sum(bonds)

    node_sum = sum(node_tot) or 1
    node_marginals = [c / node_sum for c in node_tot]
    edge_full = [total_pairs - total_bonds] + bond_tot
    edge_sum = sum(edge_full) or 1
    edge_marginals = [c / edge_sum for c in edge_full]
    max_nodes = max(size_hist) if size_hist else 0

    stats = {
        "representation": rep_name,
        "num_node_classes": len(atom_types),
        "num_edge_classes": len(bond_types) + 1,
        "atom_decoder": list(atom_types),
        "bond_decoder": ["none"] + [b.lower() for b in bond_types],
        "kekulize": kekulize,
        "max_nodes": max_nodes,
        "min_heavy": min(size_hist) if size_hist else 0,
        "node_marginals": node_marginals,
        "edge_marginals": edge_marginals,
        "node_counts": node_tot,
        "bond_counts": bond_tot,
        "size_histogram": {str(k): size_hist[k] for k in sorted(size_hist)},
        "n_encoded": n_ok,
        "n_skipped": n_skip,
        "source_smiles": args.smiles,
    }
    with open(args.out, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"  encoded {n_ok:,}   skipped {n_skip:,} ({n_skip / max(1, len(smiles)):.5f})")
    print("  node marginals: "
          + ", ".join(f"{s}={m:.4f}" for s, m in zip(atom_types, node_marginals)))
    print("  edge marginals: "
          + ", ".join(f"{s}={m:.5f}"
                      for s, m in zip(["none"] + list(bond_types), edge_marginals)))
    print(f"  max_nodes={max_nodes}; wrote {args.out}")


if __name__ == "__main__":
    main()
