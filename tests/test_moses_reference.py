"""
Tests for the E1 MOSES official split.

The tests that matter here pin down the mislabelling in
``src/datasets/moses_dataset.py``, which loads MOSES's *test* as "val" and its
*test_scaffolds* as "test". That both hides an official test set behind a name
that invites tuning, and destroys the ``test`` vs ``test_scaffolds`` distinction
the protocol needs for its two FCD columns.
"""
import pytest

from defog.data import moses_reference as mr
from defog.domains.molecule import build_encoders, smiles_to_pyg_data

ATOM_ENC, ATOM_DEC, BOND_ENC, BOND_DEC = build_encoders(mr.ATOM_TYPES, mr.BOND_TYPES)

_AVAILABLE = all(p.exists() for p in mr.reference_paths().values())
requires_reference = pytest.mark.skipif(
    not _AVAILABLE,
    reason=("MOSES splits absent; run `python -c 'from defog.data import "
            "moses_reference as m; m.download_reference()'`"),
)


# ===========================================================================
# Constants and representation
# ===========================================================================
def test_three_splits_not_two():
    """test and test_scaffolds are separate sets with separate counts."""
    assert set(mr.MOSES_COUNTS) == {"train", "test", "test_scaffolds"}
    assert mr.MOSES_COUNTS["test"] != mr.MOSES_COUNTS["test_scaffolds"]


def test_moses_is_aromatic_like_guacamol_not_kekulized_like_zinc():
    from defog.data import guacamol_reference as gm
    from defog.data import zinc_reference as zr

    assert mr.BOND_TYPES == gm.BOND_TYPES == ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    assert mr.AROMATIC is True and mr.KEKULIZE is False
    assert "AROMATIC" not in zr.BOND_TYPES


def test_vocabulary_matches_the_legacy_channel_order():
    """Order is load-bearing: it fixes the checkpoint's channel indices."""
    assert mr.ATOM_TYPES == ["C", "N", "S", "O", "F", "Cl", "Br", "H"]
    assert [mr.ATOM_VALENCY[a] for a in mr.ATOM_TYPES] == [4, 3, 4, 2, 1, 1, 1, 1]


def test_aromatic_molecule_encodes_without_kekulization():
    assert smiles_to_pyg_data("c1ccccc1", ATOM_ENC, BOND_ENC) is not None


def test_molecule_outside_vocabulary_is_skipped():
    # Iodine is in ZINC/GuacaMol but NOT in the MOSES vocabulary.
    assert smiles_to_pyg_data("ICCI", ATOM_ENC, BOND_ENC) is None


# ===========================================================================
# Representations (aromatic_v1 vs kekulized_v2)
# ===========================================================================
def test_default_representation_is_the_one_existing_checkpoints_used():
    """Load-bearing. Every MOSES checkpoint before 2026-08-03 (moses_e1_seed*,
    moses_rl*, moses_rlpen*) was trained on aromatic_v1. If the default moves,
    those decode against the wrong vocabulary -- silently, since nothing in the
    decode path validates it."""
    assert mr.DEFAULT_REPRESENTATION == "aromatic_v1"
    rep = mr.get_representation()
    assert rep.atom_types == mr.ATOM_TYPES
    assert rep.bond_types == mr.BOND_TYPES
    assert rep.kekulize is mr.KEKULIZE


def test_kekulized_v2_drops_aromatic_and_the_dead_hydrogen():
    rep = mr.get_representation("kekulized_v2")
    assert "AROMATIC" not in rep.bond_types
    assert rep.bond_types == ["SINGLE", "DOUBLE", "TRIPLE"]
    assert "H" not in rep.atom_types
    assert rep.atom_types == ["C", "N", "S", "O", "F", "Cl", "Br"]
    assert rep.kekulize is True
    assert rep.aromatic is False


def test_unknown_representation_is_rejected():
    with pytest.raises(mr.ReferenceDataError):
        mr.get_representation("kekulised_v2")      # plausible misspelling


def test_kekulized_representation_encodes_aromatic_input_losslessly():
    """The premise of the whole change: MOSES survives kekulization.

    Measured across 50,000 train molecules at 100% exact round-trip; this pins
    a few representative cases so a regression shows up without the dataset.
    """
    from defog.domains.molecule import mol_to_smiles, pyg_data_to_mol
    from rdkit import Chem

    rep = mr.get_representation("kekulized_v2")
    ae, ad, be, bd = rep.encoders()
    for smi in ["c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "c1ccc2ccccc2c1",
                "Clc1ccccc1CN1CCOCC1", "O=C(N)c1cccnc1",
                # imino tautomer: the exocyclic N=c that needs [H] in SMILES
                "[H]N=c1c(C#N)c(C)c(C(=O)OCC)nn1-c1ccc(C)cc1"]:
        data = smiles_to_pyg_data(smi, ae, be, kekulize=True)
        assert data is not None, smi
        back = mol_to_smiles(pyg_data_to_mol(data, ad, bd))
        assert back == Chem.CanonSmiles(smi), smi


def test_aromatic_encoder_cannot_take_kekulized_input_and_vice_versa():
    """Why vocabulary and kekulize flag must be selected together.

    Encoding an aromatic molecule against a bond set with no AROMATIC class,
    without kekulizing first, hits the unknown-bond branch and returns None --
    for a drug-like set that silently discards nearly everything.
    """
    rep = mr.get_representation("kekulized_v2")
    ae, _, be, _ = rep.encoders()
    assert smiles_to_pyg_data("c1ccccc1O", ae, be, kekulize=False) is None
    assert smiles_to_pyg_data("c1ccccc1O", ae, be, kekulize=True) is not None


def test_build_graphs_infers_kekulize_from_the_bond_set():
    """Passing the kekulized vocabulary must not silently use the module-level
    KEKULIZE=False, which is what the previous implementation did."""
    graphs, kept, skipped = mr.build_graphs(
        ["c1ccccc1O", "CC(=O)Nc1ccc(O)cc1"],
        atom_types=["C", "N", "S", "O", "F", "Cl", "Br"],
        bond_types=["SINGLE", "DOUBLE", "TRIPLE"])
    assert skipped == 0 and len(graphs) == 2 and len(kept) == 2


def test_build_graphs_by_representation_matches_explicit_vocabulary():
    a, _, _ = mr.build_graphs(["c1ccccc1O"], representation="kekulized_v2")
    b, _, _ = mr.build_graphs(["c1ccccc1O"],
                              atom_types=["C", "N", "S", "O", "F", "Cl", "Br"],
                              bond_types=["SINGLE", "DOUBLE", "TRIPLE"])
    assert a[0].x.shape == b[0].x.shape
    assert a[0].edge_attr.shape == b[0].edge_attr.shape


def test_representation_channel_counts_differ_between_versions():
    """The reason a checkpoint and its representation must travel together."""
    v1, v2 = mr.get_representation("aromatic_v1"), mr.get_representation("kekulized_v2")
    assert len(v1.atom_types) == 8 and len(v1.bond_types) + 1 == 5
    assert len(v2.atom_types) == 7 and len(v2.bond_types) + 1 == 4


def test_matches_model_catches_a_vocabulary_mismatch():
    class FakeModel:
        def __init__(self, x, e):
            self.output_dims = {"X": x, "E": e, "y": 0}

    v1, v2 = mr.get_representation("aromatic_v1"), mr.get_representation("kekulized_v2")
    assert v1.matches_model(FakeModel(8, 5))
    assert not v1.matches_model(FakeModel(7, 4))
    assert v2.matches_model(FakeModel(7, 4))
    assert not v2.matches_model(FakeModel(8, 5))
    # A model that cannot report its dims must not block the caller.
    assert v2.matches_model(object())


# ===========================================================================
# The SPLIT self-declaration
# ===========================================================================
def test_split_column_mismatch_is_fatal(tmp_path):
    """Each MOSES CSV names its own split, so a swapped file is catchable
    independently of the hash -- exactly the mistake the legacy loader bakes in."""
    p = tmp_path / "train.csv"
    p.write_text("SMILES,SPLIT\nCCO,test\n")
    with pytest.raises(mr.ReferenceDataError, match="declares SPLIT"):
        mr._read_split(p, "train")


def test_split_column_match_is_accepted(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("SMILES,SPLIT\nCCO,train\nCCC,train\n")
    assert mr._read_split(p, "train") == ["CCO", "CCC"]


def test_missing_smiles_column_is_fatal(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("mol,SPLIT\nCCO,train\n")
    with pytest.raises(mr.ReferenceDataError, match="no SMILES column"):
        mr._read_split(p, "train")


def test_verify_reports_missing_files(tmp_path):
    with pytest.raises(mr.ReferenceDataError, match="not found"):
        mr.verify_reference(tmp_path)


# ===========================================================================
# The split
# ===========================================================================
@requires_reference
def test_hashes_match_pinned():
    h = mr.verify_reference()
    assert h["hash_verified"] is True
    for split, expected in mr.MOSES_SHA256.items():
        assert h[f"{split}_sha256"] == expected


@requires_reference
def test_counts_match_the_shipped_files():
    s = mr.load_reference_split(download=False)
    assert s.n_test == 176074
    assert s.n_test_scaffolds == 176225
    assert s.n_val == mr.DEFAULT_VAL_SIZE
    assert s.n_train + s.n_val == mr.MOSES_COUNTS["train"] == 1584663


@requires_reference
def test_the_two_held_out_sets_are_actually_different():
    """If these ever collapse into one, the protocol's two FCD columns are a lie."""
    s = mr.load_reference_split(download=False)
    assert s.test_smiles != s.test_scaffolds_smiles
    overlap = set(s.test_smiles) & set(s.test_scaffolds_smiles)
    # Scaffold-split by construction: the sets should share little or nothing.
    assert len(overlap) < 0.01 * s.n_test


@requires_reference
def test_validation_comes_out_of_train_and_never_from_a_held_out_set():
    """The defect in src/datasets/moses_dataset.py, asserted against."""
    s = mr.load_reference_split(download=False)
    val = set(s.val_smiles)
    assert not (val & set(s.test_smiles)), "validation leaked from test"
    assert not (val & set(s.test_scaffolds_smiles)), "validation leaked from test_scaffolds"
    assert not (val & set(s.train_smiles)), "validation overlaps train"


@requires_reference
def test_val_size_change_does_not_move_the_held_out_sets():
    a = mr.load_reference_split(download=False, val_size=1000)
    b = mr.load_reference_split(download=False, val_size=9000)
    assert a.test_smiles == b.test_smiles
    assert a.test_scaffolds_smiles == b.test_scaffolds_smiles


@requires_reference
def test_split_is_deterministic_for_a_seed():
    a = mr.load_reference_split(download=False, split_seed=42)
    b = mr.load_reference_split(download=False, split_seed=42)
    c = mr.load_reference_split(download=False, split_seed=7)
    assert a.val_smiles == b.val_smiles
    assert a.val_smiles != c.val_smiles
    assert a.test_smiles == c.test_smiles


@requires_reference
def test_provenance_keeps_the_test_sets_flagged_separate():
    p = mr.load_reference_split(download=False).provenance
    for key in ("train_sha256", "n_test", "n_test_scaffolds", "val_drawn_from",
                "aromatic", "kekulized", "atom_types", "test_sets_kept_separate"):
        assert key in p, f"provenance is missing {key}"
    assert p["val_drawn_from"] == "train"
    assert p["test_sets_kept_separate"] is True


@requires_reference
def test_frozen_vocabulary_encodes_the_reference_data():
    s = mr.load_reference_split(download=False)
    graphs, kept, skipped = mr.build_graphs(s.train_smiles[:2000])
    assert skipped == 0, f"{skipped}/2000 MOSES molecules failed to encode"
    assert len(graphs) == len(kept) == 2000


@requires_reference
def test_molecule_sizes_match_the_moses_specification():
    """MOSES is 8-27 heavy atoms; much smaller than ZINC (38) or GuacaMol (72)."""
    s = mr.load_reference_split(download=False)
    graphs, _, _ = mr.build_graphs(s.train_smiles[:5000])
    sizes = [g.x.size(0) for g in graphs]
    assert min(sizes) >= 8
    assert max(sizes) <= 27
