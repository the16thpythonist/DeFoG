"""Reference datasets whose splits are fixed by an external authority.

Modules here do not *make* splits, they *retrieve* them. See
:mod:`defog.data.zinc_reference` for the reasoning.
"""

from . import guacamol_reference, moses_reference, zinc_reference
from .zinc_reference import (
    N_TEST,
    N_TOTAL,
    N_TRAIN_FULL,
    ZincReferenceSplit,
    sha256_file,
)
from .guacamol_reference import GuacamolReferenceSplit
from .moses_reference import MosesReferenceSplit

# Deliberately no top-level ATOM_TYPES / BOND_TYPES / load_reference_split
# re-export. The three datasets disagree on every one of them:
#
#   zinc      9 atoms   SINGLE/DOUBLE/TRIPLE            KEKULIZED
#   guacamol  12 atoms  + AROMATIC                      aromatic
#   moses     8 atoms   + AROMATIC                      aromatic
#
# so a bare `from defog.data import ATOM_TYPES` would silently pick one. Import
# through the dataset module -- `zinc_reference.ATOM_TYPES` -- so the choice is
# visible at the call site.

__all__ = [
    "zinc_reference",
    "guacamol_reference",
    "moses_reference",
    "N_TEST",
    "N_TOTAL",
    "N_TRAIN_FULL",
    "ZincReferenceSplit",
    "GuacamolReferenceSplit",
    "MosesReferenceSplit",
    "sha256_file",
]
