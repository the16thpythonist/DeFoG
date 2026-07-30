"""
GuacaMol under its official split, as Brown et al. define it.

Unlike ZINC250k, GuacaMol ships an evaluation package that fixes the split *and*
the metric definitions, so E1's job here is to use it rather than reinvent it
(``docs/unconditional-protocol.md`` section 1). This module covers the split
half; the metrics half is the official ``guacamol`` package and is not
reimplemented here.

Three things this module exists to prevent:

1. **Training on test.** ``data/guacamol/guacamol_all.smiles`` is the *combined*
   1,591,378-molecule release -- exactly train + valid + test. Any run that
   splits that file randomly has trained on the official test set.
   ``experiments/training__guacamol_uncond.py`` does precisely this, which is
   why its numbers cannot appear in an E1 table.

2. **The wrong download URL.** ``src/datasets/guacamol_dataset.py`` fetches from
   ``figshare.com/ndownloader/files/<id>``, which answers HTTP 202 with an empty
   body -- so the download "succeeds" and writes a zero-byte file. The working
   host is ``ndownloader.figshare.com/files/<id>``, used below.

3. **An unrecorded ``filter`` flag.** See :func:`build_graphs`. It is not a
   detail: it removes about 12% of the training set.

Unlike ZINC, GuacaMol is used in its **aromatic** form (4 bond types, no
kekulization), matching ``src/datasets/guacamol_dataset.py``. That difference
between the two datasets is deliberate and has to be recorded per run.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ===========================================================================
# Pinned authority
# ===========================================================================
# NOTE the host. See point 2 in the module docstring.
_BASE_URL = "https://ndownloader.figshare.com/files/"
GUACAMOL_URLS = {
    "train": _BASE_URL + "13612760",
    "valid": _BASE_URL + "13612766",
    "test": _BASE_URL + "13612757",
}

#: MD5, not SHA256, and deliberately so: these are the hashes the DiGress /
#: DeFoG lineage publishes (``src/datasets/guacamol_dataset.py``). Matching them
#: proves we hold the same bytes those papers trained on, which a hash we
#: invented ourselves could not establish. Verified against a fresh download
#: 2026-07-30.
GUACAMOL_MD5 = {
    "train": "05ad85d871958a05c02ab51a4fde8530",
    "valid": "e53db4bff7dc4784123ae6df72e3b1f0",
    "test": "677b757ccec4809febd83850b43e1616",
}

GUACAMOL_COUNTS = {"train": 1273104, "valid": 79568, "test": 238706}
N_TOTAL = sum(GUACAMOL_COUNTS.values())  # 1591378 == len(guacamol_all.smiles)

#: Frozen 12-element vocabulary in the order used by
#: ``src/datasets/guacamol_dataset.py``, so channel indices agree between the two
#: implementations. Se and B are rare, which is exactly why deriving the
#: vocabulary from a sample (as the current experiment does) is a hazard.
ATOM_TYPES: List[str] = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]

#: Aromatic, NOT kekulized -- the opposite of the ZINC E1 choice.
BOND_TYPES: List[str] = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

AROMATIC = True
KEKULIZE = False

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _PROJECT_ROOT / "data" / "guacamol"


class ReferenceDataError(RuntimeError):
    """The reference data is missing, altered, or the wrong shape."""


def md5_file(path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_paths(root=None) -> Dict[str, Path]:
    root = Path(root) if root is not None else DEFAULT_ROOT
    return {s: root / f"guacamol_v1_{s}.smiles" for s in GUACAMOL_URLS}


def download_reference(root=None, *, force: bool = False) -> Dict[str, Path]:
    """Fetch the three official split files if absent.

    figshare answers the *other* URL form with HTTP 202 and an empty body, so a
    naive fetch produces a zero-byte file that looks like a success. The size
    check below refuses to accept that outcome regardless of status code.
    """
    root = Path(root) if root is not None else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    paths = reference_paths(root)

    for split, url in GUACAMOL_URLS.items():
        path = paths[split]
        if path.exists() and not force:
            continue
        tmp = path.with_suffix(path.suffix + ".partial")
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 1_000_000:
                raise ReferenceDataError(
                    f"{url} returned only {tmp.stat().st_size} bytes -- figshare "
                    f"was still staging the file. Retry in a moment."
                )
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            raise ReferenceDataError(f"could not download {url} to {path}: {exc}") from exc
        os.replace(tmp, path)

    return paths


def verify_reference(root=None, *, allow_hash_mismatch: bool = False) -> Dict[str, str]:
    """Check all three files against the lineage's published MD5s."""
    paths = reference_paths(root)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise ReferenceDataError(
            f"reference files not found: {', '.join(missing)}. Call "
            f"download_reference() first."
        )

    hashes, mismatched = {}, []
    for split, path in paths.items():
        got = md5_file(path)
        hashes[f"{split}_md5"] = got
        hashes[f"{split}_sha256"] = sha256_file(path)
        if got != GUACAMOL_MD5[split]:
            mismatched.append(f"{split}: got {got}, pinned {GUACAMOL_MD5[split]}")

    if mismatched and not allow_hash_mismatch:
        raise ReferenceDataError(
            f"GuacaMol reference data does not match the published hashes "
            f"({'; '.join(mismatched)}). Do not proceed with a comparison against "
            f"published numbers until this is resolved."
        )
    hashes["hash_verified"] = not mismatched
    return hashes


# ===========================================================================
# The split
# ===========================================================================
@dataclass(frozen=True)
class GuacamolReferenceSplit:
    """The official train / valid / test SMILES, plus provenance.

    GuacaMol ships a real validation split, so unlike ZINC nothing is carved out
    of train. ``test_smiles`` is here to be hashed and counted, not trained on.
    """

    train_smiles: List[str]
    val_smiles: List[str]
    test_smiles: List[str]
    provenance: Dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.train_smiles)

    @property
    def n_val(self) -> int:
        return len(self.val_smiles)

    @property
    def n_test(self) -> int:
        return len(self.test_smiles)

    def summary(self) -> str:
        p = self.provenance
        return (
            f"GuacaMol official split: train {self.n_train} | val {self.n_val} | "
            f"test {self.n_test} (sealed)\n"
            f"  train md5 {p.get('train_md5', '?')}\n"
            f"  valid md5 {p.get('valid_md5', '?')}\n"
            f"  test  md5 {p.get('test_md5', '?')}\n"
            f"  aromatic={p.get('aromatic')} kekulized={p.get('kekulized')}"
        )


def _read_smiles(path) -> List[str]:
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def load_reference_split(
    root=None,
    *,
    download: bool = True,
    allow_hash_mismatch: bool = False,
) -> GuacamolReferenceSplit:
    """Load GuacaMol's official train / valid / test split.

    There is no ``split_seed`` parameter and no shuffling, by design: the split
    is Brown et al.'s, not ours.
    """
    root = Path(root) if root is not None else DEFAULT_ROOT
    if download:
        download_reference(root)
    hashes = verify_reference(root, allow_hash_mismatch=allow_hash_mismatch)
    paths = reference_paths(root)

    splits = {}
    for split, path in paths.items():
        smiles = _read_smiles(path)
        expected = GUACAMOL_COUNTS[split]
        if len(smiles) != expected:
            raise ReferenceDataError(
                f"{path} holds {len(smiles)} molecules, expected {expected}."
            )
        splits[split] = smiles

    return GuacamolReferenceSplit(
        train_smiles=splits["train"],
        val_smiles=splits["valid"],
        test_smiles=splits["test"],
        provenance={
            "dataset": "guacamol_v1",
            "urls": dict(GUACAMOL_URLS),
            **hashes,
            "n_total": N_TOTAL,
            "n_train": len(splits["train"]),
            "n_val": len(splits["valid"]),
            "n_test": len(splits["test"]),
            "split_source": "official (Brown et al. 2019), not regenerated",
            "aromatic": AROMATIC,
            "kekulized": KEKULIZE,
            "atom_types": list(ATOM_TYPES),
            "bond_types": list(BOND_TYPES),
        },
    )


# ===========================================================================
# Encoding, and the filter that is not a detail
# ===========================================================================
def build_graphs(
    smiles_list,
    *,
    atom_types: Optional[List[str]] = None,
    bond_types: Optional[List[str]] = None,
    filter_roundtrip: bool = True,
    progress: bool = False,
):
    """SMILES -> graphs, optionally applying the lineage's round-trip filter.

    ``filter_roundtrip`` is ``configs/dataset/guacamol.yaml``'s ``filter: True``,
    reimplemented against the packaged encoder. It encodes each molecule, decodes
    it back with charge correction, and keeps it only if the result sanitizes
    **and** is a single fragment.

    Measured on 50,000 molecules of the official train split, this drops
    **12.2%** -- roughly 155,000 of 1,273,104. Filtered and raw are therefore
    different datasets, and a published number assumes one of them, so the choice
    has to be recorded (protocol section 2 and trap 6). The loss is dominated by
    aromatic-ring reconstruction rather than charges: charged molecules fail at
    22.7% against 11.6% for neutral ones, but neutral molecules are ~89% of the
    failures by count.

    Returns ``(graphs, source_smiles, decoded_smiles, stats)``.

    Both SMILES lists are returned on purpose. The lineage stores the *decoded*
    strings as its training set and therefore as its novelty reference, but for
    ~5% of kept molecules the decoded string differs from the source molecule's
    canonical form -- the same class of mismatch as ZINC's flattened CSV. Pick
    one explicitly and record which; do not let the default decide silently.
    """
    from rdkit import Chem, RDLogger

    from defog.domains.molecule import (
        build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles,
    )

    RDLogger.DisableLog("rdApp.*")
    atom_types = list(atom_types if atom_types is not None else ATOM_TYPES)
    bond_types = list(bond_types if bond_types is not None else BOND_TYPES)
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(
        atom_types, bond_types
    )

    iterator = smiles_list
    if progress:
        from tqdm import tqdm

        iterator = tqdm(smiles_list, desc="encoding")

    graphs, source, decoded = [], [], []
    stats = {"n_input": len(smiles_list), "encode_failed": 0,
             "decode_failed": 0, "multi_fragment": 0}

    for smi in iterator:
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder, kekulize=KEKULIZE)
        if data is None:
            stats["encode_failed"] += 1
            continue

        if not filter_roundtrip:
            graphs.append(data)
            source.append(smi)
            decoded.append(None)
            continue

        mol = pyg_data_to_mol(data, atom_decoder, bond_decoder, charge_correction=True)
        dec = mol_to_smiles(mol) if mol is not None else None
        if dec is None:
            stats["decode_failed"] += 1
            continue
        try:
            frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException,
                Chem.rdchem.MolSanitizeException, ValueError):
            stats["decode_failed"] += 1
            continue
        if len(frags) != 1:
            stats["multi_fragment"] += 1
            continue

        graphs.append(data)
        source.append(smi)
        decoded.append(dec)

    stats["n_kept"] = len(graphs)
    stats["kept_fraction"] = len(graphs) / max(1, len(smiles_list))
    stats["filter_roundtrip"] = filter_roundtrip
    return graphs, source, decoded, stats
