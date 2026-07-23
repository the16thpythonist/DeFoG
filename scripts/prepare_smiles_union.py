#!/usr/bin/env python
"""Source-agnostic SMILES standardization + union assembly for the scaled
ZINC ∪ ChEMBL foundation dataset.

Streams a large SMILES source (the ZINC20-druglike ``.tar.xz``, or plain
``.smi``/``.txt`` files), uniformly **reservoir-samples** a target number of raw
molecules (the source is tranche-ordered, so first-N would be MW/logP-biased),
applies the SAME structural-sanity filter as ``prepare_chembl.py`` (single
fragment, 12-element organic vocab, 3<=heavy<=48, largest ring<=8, stereo/isotope
stripped, no radicals), deduplicates on stereo-free canonical SMILES, and unions
with an already-cleaned set (ChEMBL), dropping cross-duplicates. Writes 98/1/1
splits + a frozen-schema stats file + KL reference descriptors.

Design for scale (target up to ~100M): the source is streamed (the ~70 GB
uncompressed tarball is never materialized); reservoir sampling is one pass with
O(reservoir) memory; RDKit filtering is parallelized over the sampled subset only.

Usage:
    .venv/bin/python scripts/prepare_smiles_union.py \
        --source data/zinc/raw/zinc-druglike-cano.tar.xz \
        --union-smiles data/chembl/chembl_train.smiles \
        --target-clean 100000000 --oversample 1.18 \
        --out-dir data/zinc_chembl_union
    # smoke:
    .venv/bin/python scripts/prepare_smiles_union.py --source <f> --target-clean 5000 --limit-scan 50000 ...
"""
from __future__ import annotations

import argparse
import json
import lzma
import os
import random
import tarfile
from collections import Counter
from multiprocessing import Pool

from rdkit import Chem, RDLogger

# ---- Frozen schema (identical to scripts/prepare_chembl.py) -----------------
ATOM_DECODER = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]
ATOM_IDX = {s: i for i, s in enumerate(ATOM_DECODER)}
ALLOWED_ELEMENTS = set(ATOM_DECODER)
BOND_ORDER = {
    Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4,
}
BOND_NAMES = ["single", "double", "triple", "aromatic"]
MIN_HEAVY, MAX_HEAVY, MAX_RING = 3, 48, 8
SPLIT_SEED, VAL_FRACTION, TEST_FRACTION = 42, 0.01, 0.01


def _init_worker():
    RDLogger.DisableLog("rdApp.*")


def clean_one(smiles: str):
    """('keep', canonical) or ('drop', reason). Mirrors prepare_chembl.clean_one."""
    if not smiles:
        return ("drop", "empty")
    if "." in smiles:
        return ("drop", "multifragment")
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return ("drop", "parse_error")
    if mol is None:
        return ("drop", "unparsable")
    try:
        if len(Chem.GetMolFrags(mol)) > 1:
            return ("drop", "multifragment")
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in ALLOWED_ELEMENTS:
                return ("drop", "element")
            if atom.GetNumRadicalElectrons() != 0:
                return ("drop", "radical")
        n = mol.GetNumHeavyAtoms()
        if n < MIN_HEAVY:
            return ("drop", "too_small")
        if n > MAX_HEAVY:
            return ("drop", "too_large")
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > MAX_RING:
                return ("drop", "wonky_ring")
        for bond in mol.GetBonds():
            if bond.GetBondType() not in BOND_ORDER:
                return ("drop", "bond_type")
        Chem.RemoveStereochemistry(mol)
        for atom in mol.GetAtoms():
            if atom.GetIsotope() != 0:
                atom.SetIsotope(0)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        if not canonical or "." in canonical:
            return ("drop", "canonicalize")
        return ("keep", canonical)
    except Exception:
        return ("drop", "exception")


def stats_one(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node = [0] * len(ATOM_DECODER)
    for atom in mol.GetAtoms():
        node[ATOM_IDX[atom.GetSymbol()]] += 1
    edge = [0, 0, 0, 0]
    for bond in mol.GetBonds():
        edge[BOND_ORDER[bond.GetBondType()] - 1] += 1
    return (mol.GetNumHeavyAtoms(), node, edge)


# ---- streaming source readers ----------------------------------------------
def _clean_smiles_token(line: str) -> str:
    """First whitespace token of a line (SMILES; drops any ID/property columns)."""
    line = line.strip()
    return line.split()[0] if line else ""


def iter_source_smiles(path: str, limit_scan=None):
    """Yield raw SMILES from a .tar.xz (of .smi/.txt members), a bare .xz, or a
    plain text file. Streamed line-by-line; never materializes the whole source."""
    n = 0
    if path.endswith((".tar.xz", ".tar.gz", ".tgz")):
        mode = "r:xz" if path.endswith(".tar.xz") else "r:gz"
        with tarfile.open(path, mode) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                first = True
                for raw in f:
                    s = _clean_smiles_token(raw.decode("utf-8", "ignore"))
                    # skip a header line like "smiles ..." / "smiles zinc_id"
                    if first:
                        first = False
                        if s.lower() in ("smiles", "smi", "canonical_smiles"):
                            continue
                    if s:
                        yield s
                        n += 1
                        if limit_scan and n >= limit_scan:
                            return
    else:
        opener = (lambda p: lzma.open(p, "rt")) if path.endswith(".xz") else (lambda p: open(p))
        with opener(path) as fh:
            for raw in fh:
                s = _clean_smiles_token(raw)
                if s:
                    yield s
                    n += 1
                    if limit_scan and n >= limit_scan:
                        return


def reservoir_sample(iterable, k: int, seed: int = 0):
    """Uniform k-sample via Algorithm R (one pass, no total count needed)."""
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(iterable):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
        if (i + 1) % 20_000_000 == 0:
            print(f"  scanned {i + 1:,} raw molecules...", flush=True)
    return reservoir


def read_lines(path, limit=None):
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            s = line.strip()
            if s:
                out.append(s)
    return out


def write_lines(path, lines):
    with open(path, "w") as fh:
        for s in lines:
            fh.write(s + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="ZINC .tar.xz / .xz / .smi / .txt")
    ap.add_argument("--union-smiles", default=None,
                    help="cleaned SMILES file to union with (e.g. ChEMBL); its "
                         "molecules are added and used to drop cross-duplicates")
    ap.add_argument("--target-clean", type=int, default=100_000_000,
                    help="target number of CLEAN, unique source molecules to keep")
    ap.add_argument("--oversample", type=float, default=1.18,
                    help="reservoir raw size = target-clean * oversample (covers "
                         "filter drop-rate so we net ~target-clean)")
    ap.add_argument("--limit-scan", type=int, default=None, help="cap raw rows scanned (smoke)")
    ap.add_argument("--out-dir", default="data/zinc_chembl_union")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--name", default="union", help="output file prefix")
    args = ap.parse_args()

    RDLogger.DisableLog("rdApp.*")
    os.makedirs(args.out_dir, exist_ok=True)
    reservoir_k = int(args.target_clean * args.oversample)
    print(f"[prep-union] source={args.source}")
    print(f"[prep-union] target_clean={args.target_clean:,} reservoir={reservoir_k:,} "
          f"workers={args.workers}")

    # ---- Pass 1: reservoir-sample raw SMILES (streaming) --------------------
    raw = reservoir_sample(iter_source_smiles(args.source, args.limit_scan),
                           reservoir_k, seed=SPLIT_SEED)
    print(f"[prep-union] reservoir holds {len(raw):,} raw molecules")

    # ---- Pass 2: filter + dedup the sampled subset -------------------------
    reasons = Counter()
    seen = set()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for status, val in pool.imap_unordered(clean_one, raw, chunksize=args.chunksize):
            if status == "keep":
                if val in seen:
                    reasons["duplicate"] += 1
                else:
                    seen.add(val)
            else:
                reasons[val] += 1
    del raw
    print(f"[prep-union] source kept unique: {len(seen):,}")
    for reason, count in reasons.most_common():
        print(f"    dropped[{reason}] = {count:,}")

    # ---- Cross-dedup + union with the existing cleaned set ------------------
    union_only = []
    n_cross_dup = 0
    if args.union_smiles:
        existing = read_lines(args.union_smiles)
        existing_set = set(existing)
        print(f"[prep-union] union set: {len(existing_set):,} molecules from "
              f"{os.path.basename(args.union_smiles)}")
        before = len(seen)
        seen -= existing_set  # source molecules already in the union set
        n_cross_dup = before - len(seen)
        print(f"[prep-union] dropped {n_cross_dup:,} source molecules already in the union set")
        union_only = existing  # keep the existing set in the union
    all_smiles = sorted(seen) + list(union_only)
    n_total = len(all_smiles)
    print(f"[prep-union] UNION total: {n_total:,} "
          f"(source-unique {len(seen):,} + union-set {len(union_only):,})")

    # ---- Split 98/1/1 ------------------------------------------------------
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(all_smiles)
    n_val = round(n_total * VAL_FRACTION)
    n_test = round(n_total * TEST_FRACTION)
    n_train = n_total - n_val - n_test
    write_lines(os.path.join(args.out_dir, f"{args.name}_train.smiles"), all_smiles[:n_train])
    write_lines(os.path.join(args.out_dir, f"{args.name}_val.smiles"),
                all_smiles[n_train:n_train + n_val])
    write_lines(os.path.join(args.out_dir, f"{args.name}_test.smiles"),
                all_smiles[n_train + n_val:])
    print(f"[prep-union] split: train={n_train:,} val={n_val:,} test={n_test:,}")

    # ---- Frozen-schema stats over the union --------------------------------
    node_tot = [0] * len(ATOM_DECODER)
    edge_tot = [0, 0, 0, 0]
    size_hist = Counter()
    total_pairs = total_bonds = 0
    with Pool(args.workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(stats_one, all_smiles, chunksize=args.chunksize):
            if res is None:
                continue
            nh, node, edge = res
            for i in range(len(node_tot)):
                node_tot[i] += node[i]
            for i in range(4):
                edge_tot[i] += edge[i]
            size_hist[nh] += 1
            total_pairs += nh * (nh - 1) // 2
            total_bonds += sum(edge)
    node_sum = sum(node_tot) or 1
    node_marginals = [c / node_sum for c in node_tot]
    no_edge = total_pairs - total_bonds
    edge_full = [no_edge] + edge_tot
    edge_sum = sum(edge_full) or 1
    edge_marginals = [c / edge_sum for c in edge_full]
    max_nodes = max(size_hist) if size_hist else 0
    stats = {
        "num_node_classes": len(ATOM_DECODER), "num_edge_classes": 5,
        "atom_decoder": ATOM_DECODER, "bond_decoder": ["none"] + BOND_NAMES,
        "max_nodes": max_nodes, "min_heavy": MIN_HEAVY,
        "node_marginals": node_marginals, "edge_marginals": edge_marginals,
        "node_counts": node_tot,
        "size_histogram": {str(k): size_hist[k] for k in sorted(size_hist)},
    }
    with open(os.path.join(args.out_dir, f"{args.name}_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    report = {
        "source": os.path.basename(args.source),
        "target_clean": args.target_clean, "reservoir": reservoir_k,
        "source_kept_unique": len(seen) + n_cross_dup, "cross_duplicates_removed": n_cross_dup,
        "union_total": n_total, "drop_reasons": dict(reasons.most_common()),
        "split": {"train": n_train, "val": n_val, "test": n_test, "seed": SPLIT_SEED},
        "max_nodes": max_nodes,
    }
    with open(os.path.join(args.out_dir, f"{args.name}_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("[prep-union] node marginals: "
          + ", ".join(f"{s}={m:.4f}" for s, m in zip(ATOM_DECODER, node_marginals)))
    print(f"[prep-union] max_nodes={max_nodes}; wrote splits + {args.name}_stats.json + "
          f"{args.name}_report.json to {args.out_dir}")


if __name__ == "__main__":
    main()
