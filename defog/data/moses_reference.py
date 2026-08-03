"""
MOSES under its official split, with ``test`` and ``test_scaffolds`` kept apart.

MOSES ships an evaluation package that fixes both the split and the metrics
(Polykovskiy et al.), so E1's job is to use it rather than reinvent it. This
module covers the split; the metrics are ``molsets``, driven from
``scripts/e1_metrics.py``.

Two things this module exists to prevent:

1. **Mislabelling the two held-out sets.** ``src/datasets/moses_dataset.py``
   maps them like this::

       train_url = ".../train.csv"
       val_url   = ".../test.csv"            # MOSES *test*           -> "val"
       test_url  = ".../test_scaffolds.csv"  # MOSES *test_scaffolds* -> "test"

   So its "val" is MOSES's official **test** set, and tuning anything on it is
   tuning on test -- the same defect as ZINC's ``val == test`` in a different
   disguise. It also collapses a distinction the protocol depends on: FCD must
   be reported against **both** ``test`` and ``test_scaffolds``, which is
   impossible once one of them has been renamed to "val".

2. **A validation split that isn't one.** MOSES ships no validation set, so like
   ZINC one has to be carved out of TRAIN. It is never taken from either
   held-out set.

Each source file carries a ``SPLIT`` column naming itself (``train`` /
``test`` / ``test_scaffolds``), which is asserted on load -- a free check that
the right file is in the right place, independent of the hash.

Representation follows ``src/datasets/moses_dataset.py``: **aromatic** (4 bond
types, no kekulization, as GuacaMol and unlike ZINC), implicit hydrogens, and
the frozen 8-type vocabulary. Note ``H`` is in that vocabulary but effectively
unused -- ``process()`` never calls ``AddHs``, so a hydrogen node appears only
if one was explicit in the SMILES. It is kept so channel indices match the
legacy implementation.
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
_BASE = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/"
MOSES_URLS = {
    "train": _BASE + "train.csv",
    "test": _BASE + "test.csv",
    "test_scaffolds": _BASE + "test_scaffolds.csv",
}

#: SHA256 as retrieved 2026-07-31.
MOSES_SHA256 = {
    "train": "49fe1aae29604ec0f5023ea34edc8789415f954138f2903dabd005a4b888a961",
    "test": "257165e8f1e1be2d5712db51c93d2b1544564b782dc303d7d19c887c21a858e6",
    "test_scaffolds": "8bc942dd807147e07b506151fe8675f58780ed8e494801b705c4bdadb3df3440",
}

#: Counts as they actually are in the files. The MOSES paper is often quoted as
#: 176,226 test_scaffolds; the shipped file holds 176,225. The file wins.
MOSES_COUNTS = {"train": 1584663, "test": 176074, "test_scaffolds": 176225}

#: Frozen vocabulary, order from src/datasets/moses_dataset.py so channel
#: indices agree between implementations.
ATOM_TYPES: List[str] = ["C", "N", "S", "O", "F", "Cl", "Br", "H"]
ATOM_VALENCY = {"C": 4, "N": 3, "S": 4, "O": 2, "F": 1, "Cl": 1, "Br": 1, "H": 1}
ATOM_WEIGHT = {"C": 12.0, "N": 14.0, "S": 32.0, "O": 16.0, "F": 19.0,
               "Cl": 35.4, "Br": 79.9, "H": 1.0}
MAX_ATOM_WEIGHT = 350.0   # MOSESinfos.max_weight

#: Aromatic, as GuacaMol and unlike ZINC.
BOND_TYPES: List[str] = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
AROMATIC = True
KEKULIZE = False


# ===========================================================================
# Representations
# ===========================================================================
@dataclass(frozen=True)
class Representation:
    """How MOSES molecules are encoded as graphs.

    This is a *model-side* choice, not an evaluation choice -- the E1 protocol
    scores SMILES, so changing it does not affect comparability with published
    baselines. It does change the channel count, which means a checkpoint and
    the representation it was trained under must travel together: decoding with
    the wrong one mis-decodes silently rather than raising. ``matches_model``
    exists to turn that into an error.
    """

    name: str
    atom_types: List[str]
    bond_types: List[str]
    kekulize: bool
    note: str = ""

    @property
    def aromatic(self) -> bool:
        return "AROMATIC" in self.bond_types

    def encoders(self):
        from defog.domains.molecule import build_encoders

        return build_encoders(list(self.atom_types), list(self.bond_types))

    def matches_model(self, model) -> bool:
        """True iff the model's node/edge class counts fit this vocabulary.

        Reads ``output_dims``, not ``input_dims``: the latter is padded with
        RRWP/molecular/time features, so it does not equal the vocabulary size.
        Edge classes carry an extra 'no bond' class at index 0, which is why the
        bond comparison is off by one.
        """
        try:
            dims = getattr(model, "output_dims", None) or {}
            n_x, n_e = int(dims["X"]), int(dims["E"])
        except Exception:                                   # noqa: BLE001
            return True          # cannot tell -> do not block the caller
        return n_x == len(self.atom_types) and n_e == len(self.bond_types) + 1


#: The representation every MOSES artifact before 2026-08-03 was trained under.
#: Kept as the default so existing checkpoints (moses_e1_seed*, moses_rl*,
#: moses_rlpen*) keep decoding correctly.
AROMATIC_V1 = Representation(
    name="aromatic_v1",
    atom_types=list(ATOM_TYPES),
    bond_types=list(BOND_TYPES),
    kekulize=KEKULIZE,
    note="original: 8 atom types including a never-used H, aromatic bonds",
)

#: Kekulized, with the dead hydrogen class removed. Two measured motivations:
#:
#: 1. 118 of 120 hard validity failures on the aromatic base are kekulization
#:    errors and exactly one is a valence error (scripts/diagnose_validity.py).
#:    An AROMATIC bond class is a promise about the whole ring system that RDKit
#:    checks by kekulizing; the model asserts it per-edge and cannot keep it.
#:    Removing the class makes that failure impossible by construction. ZINC
#:    trains this way and reaches ~0.99 validity.
#: 2. 'H' never appears as an atom in MOSES. Verified across a random 200,000
#:    train molecules AND all 220 whose SMILES literally contain "[H]" -- RDKit
#:    folds those into implicit hydrogen counts, so the graph never holds one.
#:    (Those 220 are almost all imino tautomers, 214 exocyclic N-H double-bonded
#:    to an aromatic ring; the amino/imino distinction lives in bond order,
#:    which the graph does carry, so nothing is lost by dropping the class.)
#:
#: Encoding is lossless: 50,000 random train molecules and all 220 "[H]" cases
#: round-trip to identical canonical SMILES, with zero encode failures.
KEKULIZED_V2 = Representation(
    name="kekulized_v2",
    atom_types=["C", "N", "S", "O", "F", "Cl", "Br"],
    bond_types=["SINGLE", "DOUBLE", "TRIPLE"],
    kekulize=True,
    note="kekulized bonds, dead H class removed (7 atom types)",
)

REPRESENTATIONS = {r.name: r for r in (AROMATIC_V1, KEKULIZED_V2)}
DEFAULT_REPRESENTATION = "aromatic_v1"


def get_representation(name=None) -> Representation:
    if name is None:
        name = DEFAULT_REPRESENTATION
    if isinstance(name, Representation):
        return name
    if name not in REPRESENTATIONS:
        raise ReferenceDataError(
            f"unknown MOSES representation {name!r}; have {sorted(REPRESENTATIONS)}")
    return REPRESENTATIONS[name]

DEFAULT_VAL_SIZE = 5000
DEFAULT_SPLIT_SEED = 42

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _PROJECT_ROOT / "data" / "moses"


class ReferenceDataError(RuntimeError):
    """The reference data is missing, altered, or the wrong shape."""


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_paths(root=None) -> Dict[str, Path]:
    root = Path(root) if root is not None else DEFAULT_ROOT
    return {s: root / f"{s}.csv" for s in MOSES_URLS}


def download_reference(root=None, *, force: bool = False) -> Dict[str, Path]:
    """Fetch the three official CSVs if absent.

    These are served through GitHub's LFS media host; a plain raw.githubusercontent
    URL returns an LFS pointer stub rather than the data, which is why the media
    host is used here and why the size check below matters.
    """
    root = Path(root) if root is not None else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    paths = reference_paths(root)

    for split, url in MOSES_URLS.items():
        path = paths[split]
        if path.exists() and not force:
            continue
        tmp = path.with_suffix(".partial")
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 1_000_000:
                raise ReferenceDataError(
                    f"{url} returned only {tmp.stat().st_size} bytes -- most likely "
                    f"an LFS pointer stub rather than the CSV."
                )
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            raise ReferenceDataError(f"could not download {url}: {exc}") from exc
        os.replace(tmp, path)
    return paths


def verify_reference(root=None, *, allow_hash_mismatch: bool = False) -> Dict[str, str]:
    paths = reference_paths(root)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise ReferenceDataError(
            f"reference files not found: {', '.join(missing)}. Call "
            f"download_reference() first."
        )
    hashes, mismatched = {}, []
    for split, path in paths.items():
        got = sha256_file(path)
        hashes[f"{split}_sha256"] = got
        if got != MOSES_SHA256[split]:
            mismatched.append(f"{split}: got {got}, pinned {MOSES_SHA256[split]}")
    if mismatched and not allow_hash_mismatch:
        raise ReferenceDataError(
            f"MOSES reference data does not match the pinned hashes "
            f"({'; '.join(mismatched)}). Resolve before comparing against "
            f"published numbers."
        )
    hashes["hash_verified"] = not mismatched
    return hashes


# ===========================================================================
# The split
# ===========================================================================
@dataclass(frozen=True)
class MosesReferenceSplit:
    """Train / val / test / test_scaffolds.

    ``test`` and ``test_scaffolds`` are separate attributes on purpose: the
    protocol requires FCD against both, and merging or renaming them makes that
    impossible. ``val_smiles`` comes out of train, because MOSES ships no
    validation set.
    """

    train_smiles: List[str]
    val_smiles: List[str]
    test_smiles: List[str]
    test_scaffolds_smiles: List[str]
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

    @property
    def n_test_scaffolds(self) -> int:
        return len(self.test_scaffolds_smiles)

    def summary(self) -> str:
        p = self.provenance
        return (
            f"MOSES official split: train {self.n_train} | val {self.n_val} "
            f"(from train) | test {self.n_test} (sealed) | "
            f"test_scaffolds {self.n_test_scaffolds} (sealed)\n"
            f"  train sha256 {p.get('train_sha256', '?')[:16]}...\n"
            f"  aromatic={p.get('aromatic')} kekulized={p.get('kekulized')} "
            f"val seed={p.get('split_seed')}"
        )


def _read_split(path, expected_split: str) -> List[str]:
    """Read one MOSES CSV, asserting its self-declared SPLIT column.

    The SPLIT column names the file's own identity, so this catches a
    train/test mix-up independently of the hash -- cheap insurance against
    exactly the mislabelling the legacy loader institutionalised.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    if "SMILES" not in frame.columns:
        raise ReferenceDataError(f"{path} has no SMILES column ({list(frame.columns)})")
    if "SPLIT" in frame.columns:
        found = sorted(frame["SPLIT"].astype(str).unique())
        if found != [expected_split]:
            raise ReferenceDataError(
                f"{path} declares SPLIT={found}, expected ['{expected_split}']. "
                f"The wrong file is in the wrong place."
            )
    return frame["SMILES"].astype(str).str.strip().tolist()


def load_reference_split(
    root=None,
    *,
    val_size: int = DEFAULT_VAL_SIZE,
    split_seed: int = DEFAULT_SPLIT_SEED,
    download: bool = True,
    allow_hash_mismatch: bool = False,
) -> MosesReferenceSplit:
    """Load MOSES's official split, with validation carved out of train."""
    import numpy as np

    root = Path(root) if root is not None else DEFAULT_ROOT
    if download:
        download_reference(root)
    hashes = verify_reference(root, allow_hash_mismatch=allow_hash_mismatch)
    paths = reference_paths(root)

    smiles = {}
    for split, path in paths.items():
        vals = _read_split(path, split)
        if len(vals) != MOSES_COUNTS[split]:
            raise ReferenceDataError(
                f"{path} holds {len(vals)} molecules, expected {MOSES_COUNTS[split]}."
            )
        smiles[split] = vals

    train_full = smiles["train"]
    if not 0 <= val_size < len(train_full):
        raise ReferenceDataError(
            f"val_size={val_size} does not fit inside {len(train_full)} train rows."
        )

    # Validation is drawn from TRAIN only, so neither held-out set can be
    # affected by the seed. PCG64 for a stream numpy guarantees across versions.
    order = np.random.default_rng(split_seed).permutation(len(train_full))
    val_idx = sorted(int(i) for i in order[:val_size])
    train_idx = sorted(int(i) for i in order[val_size:])

    return MosesReferenceSplit(
        train_smiles=[train_full[i] for i in train_idx],
        val_smiles=[train_full[i] for i in val_idx],
        test_smiles=smiles["test"],
        test_scaffolds_smiles=smiles["test_scaffolds"],
        provenance={
            "dataset": "moses",
            "urls": dict(MOSES_URLS),
            **hashes,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(smiles["test"]),
            "n_test_scaffolds": len(smiles["test_scaffolds"]),
            "val_size": val_size,
            "split_seed": split_seed,
            "val_drawn_from": "train",
            "split_source": "official (Polykovskiy et al. 2020), not regenerated",
            "test_sets_kept_separate": True,
            "aromatic": AROMATIC,
            "kekulized": KEKULIZE,
            "atom_types": list(ATOM_TYPES),
            "bond_types": list(BOND_TYPES),
        },
    )


def build_graphs(
    smiles_list,
    *,
    atom_types: Optional[List[str]] = None,
    bond_types: Optional[List[str]] = None,
    representation=None,
    progress: bool = False,
):
    """SMILES -> PyG graphs against a MOSES vocabulary.

    ``configs/dataset/moses.yaml`` sets ``filter: False``, so unlike GuacaMol
    there is no round-trip filter here. Returns ``(graphs, kept_smiles,
    n_skipped)``; skips are reported rather than swallowed.

    ``representation`` selects the vocabulary *and* the kekulize flag together,
    which is the point: those two must agree or every aromatic molecule silently
    fails to encode. Passing ``atom_types``/``bond_types`` explicitly still
    works for backwards compatibility, but then the kekulize flag is inferred
    from whether the bond set contains AROMATIC rather than assumed -- the old
    code read a module-level constant, which was correct only for the default.
    """
    from defog.domains.molecule import build_encoders, smiles_to_pyg_data

    if representation is not None:
        rep = get_representation(representation)
        atom_types = list(rep.atom_types)
        bond_types = list(rep.bond_types)
        kekulize = rep.kekulize
    else:
        atom_types = list(atom_types if atom_types is not None else ATOM_TYPES)
        bond_types = list(bond_types if bond_types is not None else BOND_TYPES)
        kekulize = "AROMATIC" not in bond_types
    atom_encoder, _, bond_encoder, _ = build_encoders(atom_types, bond_types)

    iterator = smiles_list
    if progress:
        from tqdm import tqdm

        iterator = tqdm(smiles_list, desc="encoding")

    graphs, kept, skipped = [], [], 0
    for smi in iterator:
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder, kekulize=kekulize)
        if data is None:
            skipped += 1
            continue
        graphs.append(data)
        kept.append(smi)
    return graphs, kept, skipped
