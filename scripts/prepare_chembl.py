#!/usr/bin/env python
"""ChEMBL 37 standardization pipeline for the DeFoG molecular foundation model.

Reads the raw chemical-representations dump
``data/chembl/raw/chembl_37_chemreps.txt.gz`` (TSV:
``chembl_id  canonical_smiles  standard_inchi  standard_inchi_key``), applies a
*structural-sanity* filter, and writes clean, deduplicated, stereo-free SMILES
split 98/1/1 into train/val/test, plus a report and a frozen-schema stats file.

Frozen schema (the public contract every downstream adapter/guidance binds to):
  - node classes = 12 organic elements ``[C,N,O,F,B,Br,Cl,I,P,S,Se,Si]``
    (GuacaMol vocab; ZINC's 9 are a subset), element-only (charges are recovered
    at decode time via ``build_molecule_with_partial_charges``);
  - edge classes = ``[no-edge, single, double, triple, aromatic]`` (aromaticity
    is kept, not kekulized -> matches ``smiles_to_pyg_data``);
  - heavy atoms only, 3 <= N <= 48; largest SSSR ring <= 8.

Filtering policy (agreed 2026-07-20):
  - REJECT any multi-fragment entry outright (no desalting / largest-fragment).
  - Drop molecules with any ring of size >= 9 ("wonky" macrocycles).
  - Structural sanity only -- NO drug-likeness (MW/logP/PAINS) filters.
  - Strip stereochemistry and isotopes; keep formal charges as-is.
  - Deduplicate on the stereo-free canonical SMILES.

Usage:
    .venv/bin/python scripts/prepare_chembl.py               # full run
    .venv/bin/python scripts/prepare_chembl.py --limit 5000  # quick smoke test
    .venv/bin/python scripts/prepare_chembl.py --workers 20
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
from collections import Counter
from multiprocessing import Pool

from rdkit import Chem, RDLogger

# --------------------------------------------------------------------- schema
ATOM_DECODER = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]
ATOM_IDX = {sym: i for i, sym in enumerate(ATOM_DECODER)}
ALLOWED_ELEMENTS = set(ATOM_DECODER)

# edge class order: index 0 = no-edge, then these bond orders 1..4
BOND_ORDER = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}
BOND_NAMES = ["single", "double", "triple", "aromatic"]  # for the stats report

MIN_HEAVY = 3
MAX_HEAVY = 48
MAX_RING = 8          # largest allowed SSSR ring; >= 9 is dropped as "wonky"
SPLIT_SEED = 42
VAL_FRACTION = 0.01
TEST_FRACTION = 0.01

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "chembl")
RAW_PATH = os.path.join(DATA_DIR, "raw", "chembl_37_chemreps.txt.gz")


# ---------------------------------------------------------------- worker: clean
def _init_worker():
    RDLogger.DisableLog("rdApp.*")


def clean_one(smiles: str):
    """Return ("keep", canonical_smiles) or ("drop", reason) for one SMILES."""
    if not smiles:
        return ("drop", "empty")
    if "." in smiles:                       # fast multi-fragment reject
        return ("drop", "multifragment")
    try:
        mol = Chem.MolFromSmiles(smiles)    # parses + sanitizes (aromaticity)
    except Exception:
        return ("drop", "parse_error")
    if mol is None:
        return ("drop", "unparsable")
    try:
        if len(Chem.GetMolFrags(mol)) > 1:  # belt-and-suspenders
            return ("drop", "multifragment")

        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in ALLOWED_ELEMENTS:
                return ("drop", "element")
            if atom.GetNumRadicalElectrons() != 0:
                return ("drop", "radical")

        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < MIN_HEAVY:
            return ("drop", "too_small")
        if n_heavy > MAX_HEAVY:
            return ("drop", "too_large")

        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > MAX_RING:
                return ("drop", "wonky_ring")

        for bond in mol.GetBonds():
            if bond.GetBondType() not in BOND_ORDER:
                return ("drop", "bond_type")

        # strip stereochemistry + isotopes (not representable in the graph)
        Chem.RemoveStereochemistry(mol)
        for atom in mol.GetAtoms():
            if atom.GetIsotope() != 0:
                atom.SetIsotope(0)

        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        if not canonical:
            return ("drop", "canonicalize")
        # guard: must still have at least one bond (smiles_to_pyg_data rejects
        # bond-less graphs); a single connected fragment with >=3 atoms always does
        if "." in canonical:
            return ("drop", "multifragment")
        return ("keep", canonical)
    except Exception:
        return ("drop", "exception")


# --------------------------------------------------------------- worker: stats
def stats_one(smiles: str):
    """Per-molecule counts over an already-clean SMILES, for frozen marginals."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node = [0] * len(ATOM_DECODER)
    for atom in mol.GetAtoms():
        node[ATOM_IDX[atom.GetSymbol()]] += 1
    edge = [0, 0, 0, 0]  # single, double, triple, aromatic
    for bond in mol.GetBonds():
        edge[BOND_ORDER[bond.GetBondType()] - 1] += 1
    n_heavy = mol.GetNumHeavyAtoms()
    return (n_heavy, node, edge)


# ---------------------------------------------------------------------- io util
def iter_raw_smiles(path: str, limit: int | None):
    """Yield the canonical_smiles column from the gzipped TSV (skips header)."""
    with gzip.open(path, "rt") as fh:
        next(fh, None)  # header
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            yield parts[1] if len(parts) > 1 else ""


def write_lines(path: str, lines):
    with open(path, "w") as fh:
        for s in lines:
            fh.write(s + "\n")


# --------------------------------------------------------------------- pipeline
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=RAW_PATH)
    ap.add_argument("--out-dir", default=DATA_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N raw rows (smoke test)")
    ap.add_argument("--chunksize", type=int, default=2000)
    args = ap.parse_args()

    RDLogger.DisableLog("rdApp.*")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[prepare_chembl] raw={args.raw}")
    print(f"[prepare_chembl] workers={args.workers} limit={args.limit}")

    # ---- Pass 1: clean + dedup ------------------------------------------------
    reasons = Counter()
    seen = set()
    n_in = 0
    with Pool(args.workers, initializer=_init_worker) as pool:
        for status, val in pool.imap_unordered(
            clean_one, iter_raw_smiles(args.raw, args.limit), chunksize=args.chunksize
        ):
            n_in += 1
            if status == "keep":
                if val in seen:
                    reasons["duplicate"] += 1
                else:
                    seen.add(val)
            else:
                reasons[val] += 1
            if n_in % 200_000 == 0:
                print(f"  processed {n_in:,} | kept {len(seen):,}")

    kept = sorted(seen)  # deterministic order regardless of worker scheduling
    n_kept = len(kept)
    print(f"[prepare_chembl] pass1 done: {n_in:,} in -> {n_kept:,} unique kept")
    for reason, count in reasons.most_common():
        print(f"    dropped[{reason}] = {count:,}")

    # ---- Split 98/1/1 (fixed seed) -------------------------------------------
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(kept)
    n_val = round(n_kept * VAL_FRACTION)
    n_test = round(n_kept * TEST_FRACTION)
    n_train = n_kept - n_val - n_test
    train = kept[:n_train]
    val = kept[n_train:n_train + n_val]
    test = kept[n_train + n_val:]

    write_lines(os.path.join(args.out_dir, "chembl_train.smiles"), train)
    write_lines(os.path.join(args.out_dir, "chembl_val.smiles"), val)
    write_lines(os.path.join(args.out_dir, "chembl_test.smiles"), test)
    print(f"[prepare_chembl] split: train={len(train):,} "
          f"val={len(val):,} test={len(test):,}")

    # ---- Pass 2: frozen-schema stats over the kept set ------------------------
    node_tot = [0] * len(ATOM_DECODER)
    edge_tot = [0, 0, 0, 0]
    size_hist = Counter()
    total_pairs = 0  # undirected atom pairs, for the no-edge marginal
    total_bonds = 0
    with Pool(args.workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(stats_one, kept, chunksize=args.chunksize):
            if res is None:
                continue
            n_heavy, node, edge = res
            for i in range(len(node_tot)):
                node_tot[i] += node[i]
            for i in range(4):
                edge_tot[i] += edge[i]
            size_hist[n_heavy] += 1
            total_pairs += n_heavy * (n_heavy - 1) // 2
            total_bonds += sum(edge)

    node_sum = sum(node_tot) or 1
    node_marginals = [c / node_sum for c in node_tot]
    no_edge = total_pairs - total_bonds
    edge_counts_full = [no_edge] + edge_tot
    edge_sum = sum(edge_counts_full) or 1
    edge_marginals = [c / edge_sum for c in edge_counts_full]
    max_nodes = max(size_hist) if size_hist else 0

    stats = {
        "num_node_classes": len(ATOM_DECODER),
        "num_edge_classes": 5,
        "atom_decoder": ATOM_DECODER,
        "bond_decoder": ["none"] + BOND_NAMES,
        "max_nodes": max_nodes,
        "min_heavy": MIN_HEAVY,
        "node_marginals": node_marginals,
        "edge_marginals": edge_marginals,
        "node_counts": [node_tot[i] for i in range(len(ATOM_DECODER))],
        "size_histogram": {str(k): size_hist[k] for k in sorted(size_hist)},
    }
    with open(os.path.join(args.out_dir, "chembl_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    report = {
        "source": os.path.basename(args.raw),
        "n_input": n_in,
        "n_kept_unique": n_kept,
        "n_duplicates_removed": reasons.get("duplicate", 0),
        "drop_reasons": dict(reasons.most_common()),
        "split": {"train": len(train), "val": len(val), "test": len(test),
                  "seed": SPLIT_SEED},
        "schema": {"atom_decoder": ATOM_DECODER, "max_heavy": MAX_HEAVY,
                   "min_heavy": MIN_HEAVY, "max_ring": MAX_RING},
        "max_nodes": max_nodes,
    }
    with open(os.path.join(args.out_dir, "prep_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[prepare_chembl] node marginals: "
          + ", ".join(f"{s}={m:.4f}" for s, m in zip(ATOM_DECODER, node_marginals)))
    print(f"[prepare_chembl] edge marginals (none/S/D/T/arom): "
          + ", ".join(f"{m:.4f}" for m in edge_marginals))
    print(f"[prepare_chembl] max_nodes={max_nodes}")
    print(f"[prepare_chembl] wrote splits + chembl_stats.json + prep_report.json "
          f"to {args.out_dir}")


if __name__ == "__main__":
    main()
