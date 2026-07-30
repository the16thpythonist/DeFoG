"""Reference datasets whose splits are fixed by an external authority.

Modules here do not *make* splits, they *retrieve* them. See
:mod:`defog.data.zinc_reference` for the reasoning.
"""

from .zinc_reference import (
    ATOM_TYPES,
    BOND_TYPES,
    N_TEST,
    N_TOTAL,
    N_TRAIN_FULL,
    ZincReferenceSplit,
    download_reference,
    load_reference_split,
    sha256_file,
)

__all__ = [
    "ATOM_TYPES",
    "BOND_TYPES",
    "N_TEST",
    "N_TOTAL",
    "N_TRAIN_FULL",
    "ZincReferenceSplit",
    "download_reference",
    "load_reference_split",
    "sha256_file",
]
