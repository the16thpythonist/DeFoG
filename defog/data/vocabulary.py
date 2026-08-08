"""
Resolve a dataset's graph vocabulary, and refuse to decode with the wrong one.

Every script that turns a checkpoint back into molecules needs the same three
lines: pick the atom/bond types, build the decoders, and -- the part that kept
getting left out -- check that they actually match the checkpoint.

That check matters more than it looks. Decoding with the wrong vocabulary does
not raise. It produces molecules: plausible ones, with the wrong atoms, which
then flow into validity, FCD and every table downstream. MOSES now ships two
representations whose channel counts differ (8 atom / 5 edge against 7 / 4), so
this stopped being hypothetical the moment ``kekulized_v2`` existed.

Written once here rather than copied into diagnose_validity.py,
sweep_sampling.py, final_eval.py and gdpo_sanity.py, because four copies of a
guard is three chances for one to drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


class VocabularyMismatch(RuntimeError):
    """A checkpoint's channel counts disagree with the requested vocabulary."""


@dataclass(frozen=True)
class Representation:
    """How a dataset's molecules are encoded as graphs.

    Atom types, bond types and the kekulize flag travel together because they
    must agree: encoding an aromatic molecule against a bond set with no
    AROMATIC class and ``kekulize=False`` silently discards nearly every
    drug-like molecule instead of raising.

    This is a *model-side* choice, not an evaluation choice -- the E1 protocol
    scores SMILES, so changing it does not affect comparability with published
    baselines. It does change the channel count, which means a checkpoint and
    the representation it was trained under must travel together: decoding with
    the wrong one mis-decodes silently rather than raising. ``matches_model``
    exists to turn that into an error.

    Lives here rather than in one dataset module because more than one dataset
    now defines named representations (MOSES since 2026-08-03, ChEMBL since
    2026-08-07), and two copies of this class is one chance for them to drift.
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


def resolve(dataset_module, representation: Optional[str] = None):
    """(atom_types, bond_types, representation_or_None) for a dataset module.

    ``representation`` is only meaningful for datasets that define named ones
    (currently MOSES). Passing it to a dataset without them is an error rather
    than a silent no-op -- a caller who asks for ``kekulized_v2`` on ZINC has a
    misconception worth surfacing.
    """
    if representation is None:
        return list(dataset_module.ATOM_TYPES), list(dataset_module.BOND_TYPES), None
    if not hasattr(dataset_module, "get_representation"):
        raise VocabularyMismatch(
            f"{dataset_module.__name__} defines no named representations, so "
            f"--representation {representation!r} cannot be honoured")
    rep = dataset_module.get_representation(representation)
    return list(rep.atom_types), list(rep.bond_types), rep


def check_model(model, atom_types, bond_types, *, what: str = "checkpoint") -> str:
    """Raise unless the model's class counts match this vocabulary.

    Reads ``output_dims`` (the raw class counts); ``input_dims`` is padded with
    RRWP/molecular/time features and is not the vocabulary size. Edge classes
    carry an extra 'no bond' class at index 0, hence the off-by-one.

    Returns a human-readable confirmation. A model that cannot report its dims
    is passed rather than blocked -- absence of evidence is not a mismatch.
    """
    dims = getattr(model, "output_dims", None) or {}
    try:
        n_x, n_e = int(dims["X"]), int(dims["E"])
    except Exception:                                       # noqa: BLE001
        return "vocabulary check skipped (model does not report output_dims)"

    want_x, want_e = len(atom_types), len(bond_types) + 1
    if n_x != want_x or n_e != want_e:
        raise VocabularyMismatch(
            f"{what} has {n_x} atom / {n_e} edge classes, but the selected "
            f"vocabulary implies {want_x} / {want_e} "
            f"(atoms={atom_types}, bonds={bond_types}). Decoding would not "
            f"fail -- it would silently produce the wrong molecules. Pass the "
            f"representation this checkpoint was trained with.")
    return f"vocabulary check OK: {n_x} atom classes, {n_e} edge classes"


def resolve_and_check(dataset_module, model, representation: Optional[str] = None,
                      *, what: str = "checkpoint") -> Tuple:
    """resolve() + build_encoders() + check_model(), the usual call sequence.

    Returns ``(atom_types, bond_types, atom_decoder, bond_decoder, rep, message)``.
    """
    from defog.domains.molecule import build_encoders

    atom_types, bond_types, rep = resolve(dataset_module, representation)
    message = check_model(model, atom_types, bond_types, what=what)
    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, bond_types)
    return atom_types, bond_types, atom_decoder, bond_decoder, rep, message
