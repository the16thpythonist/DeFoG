"""
The frozen graph vocabulary of the ChEMBL foundation-model lineage.

Unlike its neighbours in this package, this module does **not** retrieve an
externally fixed split. ChEMBL ships no canonical train/test partition, so
``scripts/prepare_chembl.py`` makes one (98/1/1, seed 42) and freezes it in
``data/chembl/chembl_stats.json``. What this module owns is the other half of
the contract -- the **vocabulary** -- which the released model card
(``docs/CHEMBL_FOUNDATION_MODEL.md``) publishes as a public interface that
adapters, guidance, RL and inpainting all bind to.

It exists because that vocabulary was, until now, written out by hand in three
places (``scripts/prepare_chembl.py``, ``scripts/train_chembl_ddp.py``,
``experiments/training__chembl_uncond.py``) and nowhere that
``defog.data.vocabulary`` could see. So the one tool whose entire job is to ask
"why are this model's samples invalid?" -- ``scripts/diagnose_validity.py`` --
could be pointed at ZINC, GuacaMol and MOSES but not at the foundation model.

The **union lineage reuses this vocabulary unchanged**: the ZINC-union ChEMBL
model is the same frozen schema trained on more data, which is the point of
freezing it. A union checkpoint resolves through this module too.

Representations
---------------
``aromatic_v1`` is what the released v1 and v2 checkpoints were trained under
and stays the default so they keep decoding. ``kekulized_v2`` removes the
AROMATIC bond class, mirroring the change that recovered 11.5 validity points
on MOSES (0.884 -> 0.991, with kekulization failures going 0.108 -> 0).

Whether that transfers to ChEMBL is **an open question at the time of writing,
not an established one.** GuacaMol is aromatic over a near-identical 12-element
vocabulary and reaches ~0.98, while this model sits at 0.845 -- so aromaticity
alone does not explain the deficit, and ChEMBL's could have another cause
(the 48-atom cap, charge reconstruction, or the deliberately unfiltered
chemistry). ``scripts/diagnose_validity.py --dataset chembl`` is what settles
it; ``kekulized_v2`` is defined here so that diagnosis can be run and acted on,
not because the answer is known.

Two facts measured from ``chembl_stats.json`` before defining ``kekulized_v2``:

1. **No atom class is dead**, so unlike MOSES (which dropped a never-used ``H``)
   all 12 elements stay. The rarest are genuinely present across 2.44M training
   molecules: Si 3,353 atoms, Se 4,051, B 5,607, I 13,600.
2. **Aromatic bonds are not a rounding detail.** Of all atom pairs, 3.55% are
   aromatic against 3.50% single -- aromatic slightly *outnumbers* single among
   real edges. Dropping the class redistributes ~half the bond mass into
   single/double, so the marginal noise prior must be recomputed with it; the
   cleaned ``.smiles`` files need no change, since SMILES is representation-
   neutral and kekulization happens at encode time.
"""

from __future__ import annotations

import os
from typing import Dict, List

from .vocabulary import Representation, VocabularyMismatch  # noqa: F401

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(_PROJECT_ROOT, "data", "chembl")

# ===========================================================================
# Frozen schema
# ===========================================================================
#: The 12 elements of the public contract, in the channel order the released
#: checkpoints were trained with. GuacaMol's vocabulary, chosen so that ZINC's
#: 9 elements are a subset and existing molecular work stays compatible.
#: **Order is load-bearing** -- it is the one-hot channel index.
ATOM_TYPES: List[str] = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]

#: Released (aromatic) bond vocabulary. A 5th class, 'no bond', is prepended at
#: encode time, so this is 4 bond types but 5 edge channels.
BOND_TYPES: List[str] = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

#: Per-element valency and weight for the ``molecular_features`` inputs, which
#: must be listed in ``ATOM_TYPES`` order. Duplicated by the two training
#: entrypoints today; they should import from here when the kekulized training
#: path lands, so the three copies cannot drift apart.
ATOM_VALENCY: Dict[str, int] = {
    "C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Br": 1, "Cl": 1, "I": 1,
    "P": 5, "S": 6, "Se": 2, "Si": 4,
}
ATOM_WEIGHT: Dict[str, float] = {
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "B": 10.81, "Br": 79.904,
    "Cl": 35.45, "I": 126.904, "P": 30.974, "S": 32.06, "Se": 78.971, "Si": 28.085,
}
#: Normalizer for total molecular weight (at most 48 heavy atoms).
MAX_ATOM_WEIGHT: float = 700.0

#: Structural bounds enforced by ``scripts/prepare_chembl.py``.
MIN_HEAVY, MAX_HEAVY, MAX_RING = 3, 48, 8


# ===========================================================================
# Representations
# ===========================================================================
#: What v1 (``chembl_foundation_lr3e-4/best_model.ckpt``) and v2
#: (``chembl_foundation_v2_kl0.2.ckpt``) were trained under: 12 atom / 5 edge
#: channels. Default, so every released artifact keeps decoding correctly.
AROMATIC_V1 = Representation(
    name="aromatic_v1",
    atom_types=list(ATOM_TYPES),
    bond_types=list(BOND_TYPES),
    kekulize=False,
    note="released v1/v2 schema: 12 atom types, aromatic bonds (5 edge classes)",
)

#: Candidate successor: same 12 atom types, no AROMATIC class, so 12 atom / 4
#: edge channels. An AROMATIC bond class is a promise about a whole ring system
#: that RDKit checks by kekulizing; the model asserts it per-edge and cannot
#: keep it, which is why removing the class made that failure mode unreachable
#: on MOSES. All 12 atom classes are retained -- none is dead here (see the
#: module docstring for counts), so there is no ``H``-equivalent to drop.
#:
#: NOT yet validated on ChEMBL. Two gates before anything is trained with it:
#: (1) kekulization must be the dominant failure category in
#:     ``diagnose_validity.py --dataset chembl``;
#: (2) the encode/decode round trip must be lossless over the 2.44M training
#:     molecules -- the exposure being charged aromatics (pyridinium, N-oxides,
#:     azolium, nitro), because formal charge is not a generated channel and is
#:     reconstructed at decode time.
KEKULIZED_V2 = Representation(
    name="kekulized_v2",
    atom_types=list(ATOM_TYPES),
    bond_types=["SINGLE", "DOUBLE", "TRIPLE"],
    kekulize=True,
    note="kekulized bonds, all 12 atom types kept (4 edge classes)",
)

REPRESENTATIONS = {r.name: r for r in (AROMATIC_V1, KEKULIZED_V2)}
DEFAULT_REPRESENTATION = "aromatic_v1"


def get_representation(name=None) -> Representation:
    """Resolve a representation by name. Unknown names raise rather than
    falling back, because the fallback would decode to plausible molecules made
    of the wrong elements instead of failing."""
    if name is None:
        name = DEFAULT_REPRESENTATION
    if isinstance(name, Representation):
        return name
    if name not in REPRESENTATIONS:
        raise VocabularyMismatch(
            f"unknown ChEMBL representation {name!r}; have {sorted(REPRESENTATIONS)}")
    return REPRESENTATIONS[name]
