"""
Tests for defog.data.vocabulary -- the guard against decoding a checkpoint with
the wrong graph vocabulary.

This exists because the failure it prevents is invisible. Decoding with a
mismatched vocabulary does not raise; it yields plausible molecules with the
wrong atoms, which then flow into validity, FCD and every downstream table. Now
that MOSES ships two representations with different channel counts (8 atom / 5
edge against 7 / 4), that is a live hazard rather than a hypothetical one.
"""

import pytest

from defog.data import chembl_reference as cref
from defog.data import guacamol_reference as gmref
from defog.data import moses_reference as mref
from defog.data import vocabulary
from defog.data import zinc_reference as zref


class FakeModel:
    """Stands in for DeFoGModel: only output_dims matters to the guard."""

    def __init__(self, n_atom, n_edge):
        self.output_dims = {"X": n_atom, "E": n_edge, "y": 0}
        # input_dims is padded with RRWP/molecular/time features and must NOT
        # be what the guard reads -- deliberately wrong here to catch that.
        self.input_dims = {"X": n_atom + 40, "E": n_edge + 40, "y": 12}


# ===========================================================================
# resolve
# ===========================================================================
def test_resolve_defaults_to_the_module_vocabulary():
    for mod in (zref, gmref, mref):
        atoms, bonds, rep = vocabulary.resolve(mod)
        assert atoms == list(mod.ATOM_TYPES)
        assert bonds == list(mod.BOND_TYPES)
        assert rep is None


def test_resolve_named_representation_for_moses():
    atoms, bonds, rep = vocabulary.resolve(mref, "kekulized_v2")
    assert rep.name == "kekulized_v2"
    assert atoms == ["C", "N", "S", "O", "F", "Cl", "Br"]
    assert bonds == ["SINGLE", "DOUBLE", "TRIPLE"]


def test_representation_on_a_dataset_without_them_is_an_error():
    """A caller asking for kekulized_v2 on ZINC holds a misconception; a silent
    no-op would let them believe it took effect."""
    with pytest.raises(vocabulary.VocabularyMismatch):
        vocabulary.resolve(zref, "kekulized_v2")


def test_unknown_representation_name_is_rejected():
    with pytest.raises(mref.ReferenceDataError):
        vocabulary.resolve(mref, "no_such_representation")


# ===========================================================================
# check_model -- the actual guard
# ===========================================================================
def test_matching_model_passes():
    msg = vocabulary.check_model(FakeModel(8, 5), list(mref.ATOM_TYPES),
                                 list(mref.BOND_TYPES))
    assert "OK" in msg


def test_the_exact_mixup_this_guard_exists_for():
    """A kekulized checkpoint (7/4) read with the aromatic vocabulary (8/5)."""
    with pytest.raises(vocabulary.VocabularyMismatch) as exc:
        vocabulary.check_model(FakeModel(7, 4), list(mref.ATOM_TYPES),
                               list(mref.BOND_TYPES))
    assert "7 atom" in str(exc.value) and "8" in str(exc.value)

    # ...and the reverse.
    kek = mref.get_representation("kekulized_v2")
    with pytest.raises(vocabulary.VocabularyMismatch):
        vocabulary.check_model(FakeModel(8, 5), list(kek.atom_types),
                               list(kek.bond_types))


def test_guard_reads_output_dims_not_input_dims():
    """input_dims is padded with extra features. Reading it would make the
    guard compare the wrong numbers and pass everything."""
    model = FakeModel(7, 4)
    assert model.input_dims["X"] != model.output_dims["X"]
    kek = mref.get_representation("kekulized_v2")
    assert "OK" in vocabulary.check_model(model, list(kek.atom_types),
                                          list(kek.bond_types))


def test_edge_classes_account_for_the_no_bond_class():
    """Edge channels are len(bond_types) + 1; forgetting the +1 would reject
    every correct checkpoint."""
    kek = mref.get_representation("kekulized_v2")
    assert "OK" in vocabulary.check_model(FakeModel(7, 4), list(kek.atom_types),
                                          list(kek.bond_types))
    with pytest.raises(vocabulary.VocabularyMismatch):
        vocabulary.check_model(FakeModel(7, 3), list(kek.atom_types),
                               list(kek.bond_types))


def test_model_without_output_dims_is_not_blocked():
    """Absence of evidence is not a mismatch -- do not break callers holding a
    model type that cannot report its dims."""
    msg = vocabulary.check_model(object(), ["C"], ["SINGLE"])
    assert "skipped" in msg


def test_error_message_names_the_fix():
    with pytest.raises(vocabulary.VocabularyMismatch) as exc:
        vocabulary.check_model(FakeModel(7, 4), list(mref.ATOM_TYPES),
                               list(mref.BOND_TYPES), what="my.ckpt")
    text = str(exc.value)
    assert "my.ckpt" in text
    assert "representation" in text
    # The message must say WHY this matters, not just that dims differ.
    assert "silently" in text


# ===========================================================================
# resolve_and_check
# ===========================================================================
def test_resolve_and_check_returns_usable_decoders():
    kek = mref.get_representation("kekulized_v2")
    atoms, bonds, adec, bdec, rep, msg = vocabulary.resolve_and_check(
        mref, FakeModel(7, 4), "kekulized_v2")
    assert atoms == list(kek.atom_types)
    assert rep.name == "kekulized_v2"
    assert adec[0] == "C"
    assert bdec[0] is None          # index 0 is the no-bond class
    assert len(bdec) == len(bonds) + 1
    assert "OK" in msg


def test_resolve_and_check_raises_before_building_decoders():
    with pytest.raises(vocabulary.VocabularyMismatch):
        vocabulary.resolve_and_check(mref, FakeModel(7, 4), None)


def test_every_dataset_passes_with_its_own_dims():
    for mod in (zref, gmref, mref, cref):
        atoms, bonds, _ = vocabulary.resolve(mod)
        model = FakeModel(len(atoms), len(bonds) + 1)
        assert "OK" in vocabulary.check_model(model, atoms, bonds)


# ===========================================================================
# ChEMBL / foundation lineage
# ===========================================================================
def test_chembl_defaults_to_the_released_aromatic_schema():
    """The released v1/v2 checkpoints are 12 atom / 5 edge. Defaulting to
    anything else would break every artifact built on the model card."""
    atoms, bonds, rep = vocabulary.resolve(cref)
    assert atoms == ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]
    assert bonds == ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    assert cref.get_representation().name == "aromatic_v1"
    assert rep is None


def test_chembl_kekulized_keeps_all_twelve_atom_types():
    """Unlike MOSES, which dropped a never-used H class, no ChEMBL element is
    dead: the rarest (Si) still appears 3,353 times across 2.44M molecules."""
    atoms, bonds, rep = vocabulary.resolve(cref, "kekulized_v2")
    assert rep.name == "kekulized_v2"
    assert atoms == list(cref.ATOM_TYPES)
    assert bonds == ["SINGLE", "DOUBLE", "TRIPLE"]
    assert rep.kekulize is True and rep.aromatic is False


def test_kekulized_v2_means_different_things_per_dataset():
    """The name is scoped to its dataset module: 7 atom types on MOSES, 12 on
    ChEMBL. Sharing a name across modules is fine precisely because the guard
    below catches a checkpoint meeting the wrong one."""
    moses_atoms, _, _ = vocabulary.resolve(mref, "kekulized_v2")
    chembl_atoms, _, _ = vocabulary.resolve(cref, "kekulized_v2")
    assert len(moses_atoms) == 7 and len(chembl_atoms) == 12


def test_released_chembl_dims_are_refused_by_the_kekulized_vocabulary():
    """The mixup this guard exists for, on the lineage about to acquire a
    second representation: a 12/5 checkpoint decoded as 12/4 would silently
    read AROMATIC edges as 'no bond'."""
    released = FakeModel(12, 5)
    assert cref.get_representation("aromatic_v1").matches_model(released)
    assert not cref.get_representation("kekulized_v2").matches_model(released)
    with pytest.raises(vocabulary.VocabularyMismatch):
        vocabulary.resolve_and_check(cref, released, "kekulized_v2")


def test_chembl_unknown_representation_is_rejected():
    with pytest.raises(vocabulary.VocabularyMismatch):
        vocabulary.resolve(cref, "no_such_representation")


def test_molecular_feature_tables_cover_the_vocabulary():
    """atom_valencies/atom_weights are passed to the model as lists indexed by
    channel, so a missing element would silently shift every later element's
    valency by one."""
    for table in (cref.ATOM_VALENCY, cref.ATOM_WEIGHT):
        assert set(table) == set(cref.ATOM_TYPES)
