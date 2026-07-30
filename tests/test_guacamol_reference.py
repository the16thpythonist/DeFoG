"""
Tests for the E1 GuacaMol official split.

Same two-group structure as ``test_zinc_reference.py``: logic that needs no data
runs always, and the split checks skip when ``data/guacamol/`` is unpopulated.

The tests worth reading are the ones that pin down mistakes already made in this
repo -- the combined-file leak, the broken download host, and the size of the
``filter`` flag's effect.
"""
import pytest
from rdkit import Chem

from defog.data import guacamol_reference as gm
from defog.domains.molecule import build_encoders, smiles_to_pyg_data

ATOM_ENC, ATOM_DEC, BOND_ENC, BOND_DEC = build_encoders(gm.ATOM_TYPES, gm.BOND_TYPES)

_AVAILABLE = all(p.exists() for p in gm.reference_paths().values())
requires_reference = pytest.mark.skipif(
    not _AVAILABLE,
    reason=("GuacaMol splits absent; run `python -c 'from defog.data import "
            "guacamol_reference as g; g.download_reference()'`"),
)


# ===========================================================================
# Constants and vocabulary
# ===========================================================================
def test_counts_sum_to_the_combined_release():
    """The arithmetic behind the leak.

    guacamol_all.smiles has exactly 1,591,378 lines, which is train + valid +
    test. Anything that splits that file randomly has trained on test.
    """
    assert gm.N_TOTAL == 1591378
    assert sum(gm.GUACAMOL_COUNTS.values()) == gm.N_TOTAL
    assert gm.GUACAMOL_COUNTS == {"train": 1273104, "valid": 79568, "test": 238706}


def test_download_host_is_the_working_one():
    """figshare.com/ndownloader/ answers 202 with an empty body; the download
    then 'succeeds' and writes a zero-byte file. src/datasets/guacamol_dataset.py
    still uses that form."""
    for url in gm.GUACAMOL_URLS.values():
        assert url.startswith("https://ndownloader.figshare.com/files/")


def test_guacamol_is_aromatic_unlike_zinc():
    """The two E1 datasets deliberately disagree, so the flag has to be recorded."""
    from defog.data import zinc_reference

    assert gm.BOND_TYPES == ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    assert gm.AROMATIC is True and gm.KEKULIZE is False
    assert "AROMATIC" not in zinc_reference.BOND_TYPES
    assert zinc_reference.AROMATIC is False


def test_vocabulary_is_frozen_and_includes_the_rare_elements():
    """Se and B are rare enough that a sampled vocabulary can miss them, which
    would change the checkpoint's channel layout between runs."""
    assert gm.ATOM_TYPES[:4] == ["C", "N", "O", "F"]
    assert "Se" in gm.ATOM_TYPES and "B" in gm.ATOM_TYPES and "Si" in gm.ATOM_TYPES
    assert len(gm.ATOM_TYPES) == 12


def test_data_module_does_not_export_an_ambiguous_vocabulary():
    """`from defog.data import ATOM_TYPES` must not resolve, since the two
    datasets disagree and the winner would be silent."""
    import defog.data as d

    assert not hasattr(d, "ATOM_TYPES")
    assert not hasattr(d, "load_reference_split")


# ===========================================================================
# Encoding
# ===========================================================================
def test_aromatic_molecule_encodes_without_kekulization():
    """The inverse of the ZINC case: here AROMATIC is in the vocabulary, so no
    kekulization is needed or wanted."""
    assert smiles_to_pyg_data("c1ccccc1", ATOM_ENC, BOND_ENC) is not None


@pytest.mark.parametrize("smiles", ["[Se]1cccc1", "B(O)(O)c1ccccc1", "[SiH3]CCC"])
def test_rare_elements_encode(smiles):
    if Chem.MolFromSmiles(smiles) is None:
        pytest.skip(f"RDKit cannot parse {smiles}")
    assert smiles_to_pyg_data(smiles, ATOM_ENC, BOND_ENC) is not None


def test_filter_off_keeps_everything_encodable():
    smis = ["c1ccccc1", "CCO", "CC(=O)Nc1ccccc1"]
    graphs, source, decoded, stats = gm.build_graphs(smis, filter_roundtrip=False)
    assert len(graphs) == len(source) == 3
    assert stats["n_kept"] == 3
    assert stats["filter_roundtrip"] is False
    assert all(d is None for d in decoded)  # nothing was decoded


def test_filter_on_returns_decoded_smiles_alongside_source():
    """Both lists come back because the lineage stores the decoded string as its
    novelty reference, and for some molecules it differs from the source."""
    smis = ["c1ccccc1", "CCO", "CC(=O)Nc1ccccc1"]
    graphs, source, decoded, stats = gm.build_graphs(smis, filter_roundtrip=True)
    assert len(graphs) == len(source) == len(decoded)
    assert all(isinstance(d, str) for d in decoded)
    assert stats["filter_roundtrip"] is True


def test_filter_drops_a_disconnected_molecule():
    """The filter requires a single fragment, not merely a valid molecule."""
    _, _, _, stats = gm.build_graphs(["CCO.CCO"], filter_roundtrip=True)
    assert stats["n_kept"] == 0
    assert stats["multi_fragment"] == 1


def test_filter_stats_account_for_every_input():
    smis = ["c1ccccc1", "CCO.CCO", "[Xe]CC", "CC(=O)Nc1ccccc1"]
    _, _, _, s = gm.build_graphs(smis, filter_roundtrip=True)
    assert (s["n_kept"] + s["encode_failed"] + s["decode_failed"]
            + s["multi_fragment"]) == s["n_input"] == 4
    assert s["encode_failed"] == 1  # xenon is outside the vocabulary


# ===========================================================================
# The split
# ===========================================================================
@requires_reference
def test_hashes_match_the_published_lineage_values():
    """These MD5s are DiGress/DeFoG's own, so a match proves we hold the same
    bytes the published numbers were produced from."""
    h = gm.verify_reference()
    assert h["hash_verified"] is True
    for split, expected in gm.GUACAMOL_MD5.items():
        assert h[f"{split}_md5"] == expected


@requires_reference
def test_split_counts_match():
    s = gm.load_reference_split()
    assert (s.n_train, s.n_val, s.n_test) == (1273104, 79568, 238706)
    assert s.n_train + s.n_val + s.n_test == gm.N_TOTAL


@requires_reference
def test_splits_do_not_overlap():
    s = gm.load_reference_split()
    train, val, test = set(s.train_smiles), set(s.val_smiles), set(s.test_smiles)
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


@requires_reference
def test_combined_file_is_exactly_the_three_splits():
    """Pins the leak in place so it cannot quietly come back.

    If this passes, then training on guacamol_all.smiles IS training on test --
    which is what experiments/training__guacamol_uncond.py does.
    """
    import pathlib

    combined = gm.DEFAULT_ROOT / "guacamol_all.smiles"
    if not combined.exists():
        pytest.skip("guacamol_all.smiles not present")
    all_smiles = {l.strip() for l in pathlib.Path(combined).read_text().splitlines() if l.strip()}
    s = gm.load_reference_split()
    assert len(all_smiles) == gm.N_TOTAL
    # Every held-out test molecule is inside the combined file.
    assert set(s.test_smiles).issubset(all_smiles)
    assert set(s.val_smiles).issubset(all_smiles)


@requires_reference
def test_provenance_records_the_representation_flags():
    p = gm.load_reference_split().provenance
    for key in ("train_md5", "valid_md5", "test_md5", "n_train", "n_val", "n_test",
                "aromatic", "kekulized", "atom_types", "bond_types", "split_source"):
        assert key in p, f"provenance is missing {key}"
    assert p["aromatic"] is True
    assert p["kekulized"] is False
    assert "not regenerated" in p["split_source"]
