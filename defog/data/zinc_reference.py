"""
ZINC250k under the split the published numbers were actually produced with.

E1 (unconditional distribution quality, ``docs/unconditional-protocol.md``)
quotes its baselines from the literature instead of rerunning them, so every
number produced here has to come from the *same* split those papers used.
ZINC250k ships no canonical split; the GDSS/GruM lineage DeFoG follows -- "the
same setting of previous works (Jo et al., 2024)" -- fixes one with an index
file, and that file is the authority.

So this module retrieves a split, it never makes one. There is deliberately no
code path that shuffles rows into train/test: the protocol calls a regenerated
split "the single most likely place to lose comparability", and the cheapest way
to honour that is to not offer the option.

What *is* drawn locally is the validation slice, and it comes **out of train**
(protocol section 2). It exists so the sampling sweep over steps/eta/omega has
somewhere to run that is not the test set -- the mistake baked into the legacy
path, where ``src/datasets/zinc_dataset.py`` hands back the same indices for
both ``val`` and ``test``.

Provenance is not optional. Every load returns a ``provenance`` dict with the
SHA256 of both source files and the counts they produced, and the loader
hard-fails when a hash or a count disagrees with what is pinned below. With
published-only baselines there is no local control run, so a split that quietly
drifted would surface as an unexplained performance gap rather than as a bug
(protocol section 8).

Representation follows ``configs/dataset/zinc.yaml``: implicit hydrogens,
**kekulized** (``aromatic: False``), frozen 9-element vocabulary. The atom order
matches ``ZINCinfos.atom_encoder`` in the legacy path so channel indices agree
between the two implementations.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ===========================================================================
# Pinned authority
# ===========================================================================
# Both files come from the GruM reference implementation, which is where the
# ZINC250k rows this repo compares against were read from.
ZINC250K_CSV_URL = (
    "https://raw.githubusercontent.com/harryjo97/GruM/master/GruM_2D/data/zinc250k.csv"
)
ZINC250K_TEST_IDX_URL = (
    "https://raw.githubusercontent.com/harryjo97/GruM/master/GruM_2D/data/"
    "valid_idx_zinc250k.json"
)

#: SHA256 of the files as retrieved 2026-07-30. A mismatch means the upstream
#: file moved under us; it does NOT mean "download again and carry on".
ZINC250K_CSV_SHA256 = "8dfc7f364bdd0c6f89d9dbaf9d1812f58d5f8b55dac9b970b26bbe597e5115f2"
ZINC250K_TEST_IDX_SHA256 = (
    "cf1af3a8588c947493ff0a2b60178db6726e9a4dbb445235b149cd3bbe6831df"
)

#: Counts the protocol says to report and check. N_TEST is the length of the
#: index file -- note the file is named "valid_idx" but the lineage uses it as
#: the held-out TEST set, which is why nothing here calls it validation.
N_TOTAL = 249455
N_TEST = 24887
N_TRAIN_FULL = N_TOTAL - N_TEST  # 224568

#: Frozen vocabulary. Derived-from-data ordering (as
#: ``experiments/training__zinc_uncond.py`` does) makes a checkpoint's channel
#: order depend on the CSV, which is a reproducibility hazard for no benefit.
ATOM_TYPES: List[str] = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

#: Kekulized: no AROMATIC class. Aromatic vs. kekulized changes what counts as a
#: valid molecule, so this is one of the flags every run has to record
#: (protocol trap 6).
BOND_TYPES: List[str] = ["SINGLE", "DOUBLE", "TRIPLE"]

REMOVE_H = True
AROMATIC = False

DEFAULT_VAL_SIZE = 5000
DEFAULT_SPLIT_SEED = 42

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _PROJECT_ROOT / "data" / "zinc250k"


class ReferenceDataError(RuntimeError):
    """The reference data is missing, altered, or the wrong shape."""


# ===========================================================================
# Retrieval and verification
# ===========================================================================
def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a file, streamed so the 23 MB CSV never lands in memory twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_paths(root=None) -> Dict[str, Path]:
    """Where the two source files live. Does not check that they exist."""
    root = Path(root) if root is not None else DEFAULT_ROOT
    return {
        "csv": root / "zinc250k.csv",
        "test_idx": root / "valid_idx_zinc250k.json",
    }


def download_reference(root=None, *, force: bool = False) -> Dict[str, Path]:
    """Fetch the reference CSV and test-index file if they are not already local.

    Returns the paths either way. ``data/`` is gitignored, so these are cached on
    disk rather than committed -- but they are pinned by hash, so a cached copy
    is as trustworthy as a fresh download and considerably faster.
    """
    root = Path(root) if root is not None else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    paths = reference_paths(root)

    for key, url in (("csv", ZINC250K_CSV_URL), ("test_idx", ZINC250K_TEST_IDX_URL)):
        path = paths[key]
        if path.exists() and not force:
            continue
        tmp = path.with_suffix(path.suffix + ".partial")
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as exc:  # noqa: BLE001 - network failure is the user's problem to see
            if tmp.exists():
                tmp.unlink()
            raise ReferenceDataError(
                f"could not download {url} to {path}: {exc}. Fetch it manually and "
                f"place it there; the loader only needs the file, not the network."
            ) from exc
        # Rename only after a complete download, so an interrupted fetch can
        # never masquerade as a valid cached file on the next run.
        os.replace(tmp, path)

    return paths


def verify_reference(root=None, *, allow_hash_mismatch: bool = False) -> Dict[str, str]:
    """Hash both source files and compare against the pinned values.

    Returns ``{'csv_sha256': ..., 'test_idx_sha256': ...}`` for the run log.

    ``allow_hash_mismatch`` downgrades the failure to a recorded warning. Use it
    only when you have consciously decided to compare against something other
    than the pinned files, and say so in the paper -- the hashes are the only
    thing standing between a silently-changed upstream file and a table of
    numbers that are no longer comparable to anything.
    """
    paths = reference_paths(root)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise ReferenceDataError(
            f"reference files not found: {', '.join(missing)}. Call "
            f"download_reference() first, or pass download=True to load_reference_split()."
        )

    hashes = {
        "csv_sha256": sha256_file(paths["csv"]),
        "test_idx_sha256": sha256_file(paths["test_idx"]),
    }
    expected = {
        "csv_sha256": ZINC250K_CSV_SHA256,
        "test_idx_sha256": ZINC250K_TEST_IDX_SHA256,
    }
    mismatched = [k for k in expected if hashes[k] != expected[k]]
    if mismatched and not allow_hash_mismatch:
        detail = "; ".join(
            f"{k}: got {hashes[k]}, pinned {expected[k]}" for k in mismatched
        )
        raise ReferenceDataError(
            f"reference data does not match the pinned hashes ({detail}). The "
            f"upstream file changed, or the local copy is corrupt. Do not proceed "
            f"with a comparison against published numbers until this is resolved."
        )
    hashes["hash_verified"] = not mismatched
    return hashes


# ===========================================================================
# The split
# ===========================================================================
@dataclass(frozen=True)
class ZincReferenceSplit:
    """Train / validation / test SMILES plus everything needed to reproduce them.

    ``test_smiles`` is present so it can be hashed and counted, not so it can be
    trained or swept on. The protocol allows exactly one evaluation pass over it,
    with a sampling configuration already frozen on ``val_smiles``.
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
            f"ZINC250k reference split: train {self.n_train} | val {self.n_val} | "
            f"test {self.n_test} (sealed)\n"
            f"  csv      {p.get('csv_sha256', '?')}\n"
            f"  test_idx {p.get('test_idx_sha256', '?')}\n"
            f"  val drawn from train, seed={p.get('split_seed')}, "
            f"canonicalized={p.get('canonicalized')}, kekulized={p.get('kekulized')}"
        )


def _read_reference_smiles(csv_path) -> List[str]:
    """The SMILES column of the reference CSV, in file order, whitespace stripped.

    Row order is load-bearing: the test-index file indexes positionally into this
    exact file, so anything that reorders or filters rows before this point
    silently selects the wrong molecules.
    """
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if "smiles" not in frame.columns:
        raise ReferenceDataError(
            f"{csv_path} has no 'smiles' column (found {list(frame.columns)})"
        )
    # The original ZINC export carries trailing newlines inside the field.
    return frame["smiles"].astype(str).str.strip().tolist()


def _canonicalize(smiles_list: List[str]) -> List[str]:
    """RDKit-canonical SMILES, stereochemistry and formal charges preserved.

    This matches what the reference implementations do to their evaluation sets.
    It is emphatically NOT the same as ``data/zinc_250k_rdkit.csv``, which has
    been stereo-stripped and charge-neutralized -- fine as a source of training
    *graphs* (the encoding drops both anyway), wrong as the reference set for
    FCD, scaffold similarity or novelty, all of which compare SMILES strings.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        out.append(Chem.MolToSmiles(mol) if mol is not None else smi)
    return out


def load_reference_split(
    root=None,
    *,
    val_size: int = DEFAULT_VAL_SIZE,
    split_seed: int = DEFAULT_SPLIT_SEED,
    download: bool = True,
    canonicalize: bool = True,
    allow_hash_mismatch: bool = False,
) -> ZincReferenceSplit:
    """Load ZINC250k split exactly as the published lineage splits it.

    Args:
        root: Directory holding ``zinc250k.csv`` and ``valid_idx_zinc250k.json``.
            Defaults to ``data/zinc250k/``.
        val_size: Molecules carved out of train for the sampling sweep and
            checkpoint selection. Never taken from test.
        split_seed: Seed for the validation draw. Uses numpy's PCG64, whose
            stream numpy guarantees stable across versions -- ``random.sample``
            makes no such promise, and this split has to survive an interpreter
            upgrade.
        download: Fetch the source files if absent.
        canonicalize: Return RDKit-canonical SMILES. Leave on unless you have a
            specific reason; downstream string comparisons assume it.
        allow_hash_mismatch: See :func:`verify_reference`.

    Returns:
        :class:`ZincReferenceSplit`.
    """
    import numpy as np

    root = Path(root) if root is not None else DEFAULT_ROOT
    if download:
        download_reference(root)
    hashes = verify_reference(root, allow_hash_mismatch=allow_hash_mismatch)
    paths = reference_paths(root)

    all_smiles = _read_reference_smiles(paths["csv"])
    if len(all_smiles) != N_TOTAL:
        raise ReferenceDataError(
            f"{paths['csv']} has {len(all_smiles)} rows, expected {N_TOTAL}. This is "
            f"not the reference ZINC250k file; resolve before generating anything."
        )

    with open(paths["test_idx"]) as fh:
        test_idx = json.load(fh)
    if isinstance(test_idx, dict):  # tolerate {"0": idx, ...} shaped exports
        test_idx = list(test_idx.values())
    test_idx = sorted(int(i) for i in test_idx)
    if len(test_idx) != N_TEST:
        raise ReferenceDataError(
            f"{paths['test_idx']} holds {len(test_idx)} indices, expected {N_TEST}."
        )
    if len(set(test_idx)) != len(test_idx):
        raise ReferenceDataError(f"{paths['test_idx']} contains duplicate indices.")
    if test_idx[0] < 0 or test_idx[-1] >= N_TOTAL:
        raise ReferenceDataError(
            f"test indices fall outside [0, {N_TOTAL}): "
            f"min {test_idx[0]}, max {test_idx[-1]}."
        )

    test_set = set(test_idx)
    train_full_idx = [i for i in range(N_TOTAL) if i not in test_set]
    if len(train_full_idx) != N_TRAIN_FULL:
        raise ReferenceDataError(
            f"train complement is {len(train_full_idx)}, expected {N_TRAIN_FULL}."
        )

    if not 0 <= val_size < len(train_full_idx):
        raise ReferenceDataError(
            f"val_size={val_size} does not fit inside {len(train_full_idx)} train rows."
        )

    # Validation comes out of TRAIN. Permuting the train complement (rather than
    # the whole dataset) means test membership cannot be affected by the seed.
    order = np.random.default_rng(split_seed).permutation(len(train_full_idx))
    val_positions = sorted(int(p) for p in order[:val_size])
    train_positions = sorted(int(p) for p in order[val_size:])
    val_idx = [train_full_idx[p] for p in val_positions]
    train_idx = [train_full_idx[p] for p in train_positions]

    if canonicalize:
        all_smiles = _canonicalize(all_smiles)

    split = ZincReferenceSplit(
        train_smiles=[all_smiles[i] for i in train_idx],
        val_smiles=[all_smiles[i] for i in val_idx],
        test_smiles=[all_smiles[i] for i in test_idx],
        provenance={
            "dataset": "zinc250k",
            "csv_path": str(paths["csv"]),
            "test_idx_path": str(paths["test_idx"]),
            "csv_url": ZINC250K_CSV_URL,
            "test_idx_url": ZINC250K_TEST_IDX_URL,
            **hashes,
            "n_total": N_TOTAL,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "val_size": val_size,
            "split_seed": split_seed,
            "val_drawn_from": "train",
            "canonicalized": canonicalize,
            # Representation flags the protocol requires on every run (trap 6).
            "remove_h": REMOVE_H,
            "aromatic": AROMATIC,
            "kekulized": True,
            "atom_types": list(ATOM_TYPES),
            "bond_types": list(BOND_TYPES),
        },
    )

    # Cross-check: the three sets must partition the file with no overlap. Cheap
    # relative to a training run, and catches an off-by-one in the index logic
    # that counts alone would not.
    if len(set(train_idx) & set(val_idx)) or len(set(train_idx) & test_set):
        raise ReferenceDataError("split overlap detected; this is a bug in the loader.")
    if len(set(val_idx) & test_set):
        raise ReferenceDataError("validation overlaps test; this is a bug in the loader.")

    return split


def build_graphs(
    smiles_list,
    *,
    atom_types: Optional[List[str]] = None,
    bond_types: Optional[List[str]] = None,
    kekulize: bool = True,
    progress: bool = False,
):
    """SMILES -> PyG ``Data`` graphs against the frozen ZINC vocabulary.

    Returns ``(graphs, kept_smiles, n_skipped)``. ``kept_smiles`` tracks the
    graphs one-for-one, so a novelty reference built from it can never drift from
    what was actually trained on.

    Skips are reported rather than swallowed: with ``kekulize=True`` and the
    3-bond vocabulary, a molecule that fails to kekulize produces no graph, and a
    non-trivial skip count means the vocabulary and the data disagree.
    """
    from defog.domains.molecule import build_encoders, smiles_to_pyg_data

    atom_types = list(atom_types if atom_types is not None else ATOM_TYPES)
    bond_types = list(bond_types if bond_types is not None else BOND_TYPES)
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
