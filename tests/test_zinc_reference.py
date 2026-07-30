"""
Tests for the E1 ZINC250k reference split and the protocol-conforming
representation/validity machinery it depends on.

Split into two groups:

* Everything that can be checked without the 23 MB reference CSV runs always --
  the kekulize option, the three validity conventions, both metric denominators.
* The split itself (counts, hashes, partitioning, determinism) is skipped when
  ``data/zinc250k/`` has not been populated, since a unit-test run should not
  depend on the network. Those are the tests that matter most, so the skip
  message says how to make them run.
"""
import json

import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Data

from defog.data import zinc_reference
from defog.domains.molecule import (
    build_encoders,
    smiles_to_pyg_data,
    pyg_data_to_mol,
    mol_to_smiles,
    largest_fragment_smiles,
    validity_report,
)

# Kekulized ZINC vocabulary -- the E1 representation.
KEK_ATOM_ENC, KEK_ATOM_DEC, KEK_BOND_ENC, KEK_BOND_DEC = build_encoders(
    zinc_reference.ATOM_TYPES, zinc_reference.BOND_TYPES
)
# Aromatic vocabulary -- what the sweep script uses, kept here to demonstrate the
# difference rather than to endorse it.
AR_ATOM_ENC, AR_ATOM_DEC, AR_BOND_ENC, AR_BOND_DEC = build_encoders(
    zinc_reference.ATOM_TYPES, ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
)

_REFERENCE_AVAILABLE = all(p.exists() for p in zinc_reference.reference_paths().values())
requires_reference = pytest.mark.skipif(
    not _REFERENCE_AVAILABLE,
    reason=(
        "reference files absent; run "
        "`python -c 'from defog.data import zinc_reference; "
        "zinc_reference.download_reference()'` to enable"
    ),
)


# ===========================================================================
# Kekulization
# ===========================================================================
def test_aromatic_molecule_needs_kekulize_under_three_bond_vocabulary():
    """The failure this option exists to prevent.

    Without kekulization a 3-bond vocabulary silently rejects every aromatic
    molecule -- which on a drug-like set is most of the data, discarded with no
    error anywhere.
    """
    assert smiles_to_pyg_data("c1ccccc1", KEK_ATOM_ENC, KEK_BOND_ENC) is None
    assert smiles_to_pyg_data("c1ccccc1", KEK_ATOM_ENC, KEK_BOND_ENC, kekulize=True) is not None


def test_kekulize_default_is_off():
    """Existing aromatic-vocabulary callers must be bit-for-bit unaffected."""
    a = smiles_to_pyg_data("c1ccccc1", AR_ATOM_ENC, AR_BOND_ENC)
    b = smiles_to_pyg_data("c1ccccc1", AR_ATOM_ENC, AR_BOND_ENC, kekulize=False)
    assert a is not None and b is not None
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_attr, b.edge_attr)
    # ...and with kekulize on, the same molecule encodes differently, which is
    # exactly why the two vocabularies are not interchangeable.
    c = smiles_to_pyg_data("c1ccccc1", AR_ATOM_ENC, AR_BOND_ENC, kekulize=True)
    assert not torch.equal(a.edge_attr, c.edge_attr)


@pytest.mark.parametrize("smiles", [
    "c1ccccc1",
    "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
    "CC1CC(C)CC(Nc2cncc(-c3nncn3C)c2)C1",
    "O=C(Nc1ccccc1)c1ccc(S(=O)(=O)N2CCOCC2)cc1",
])
def test_kekulized_roundtrip_recovers_the_molecule(smiles):
    """Encode kekulized, decode, and the canonical SMILES must come back.

    RDKit re-perceives aromaticity during sanitization, so a kekulized graph
    decodes to the same aromatic molecule it came from.
    """
    data = smiles_to_pyg_data(smiles, KEK_ATOM_ENC, KEK_BOND_ENC, kekulize=True)
    assert data is not None
    mol = pyg_data_to_mol(data, KEK_ATOM_DEC, KEK_BOND_DEC)
    assert mol_to_smiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("bad", [
    "",                      # empty
    "not_a_smiles",          # unparseable
    "C1CC",                  # unclosed ring
    "[Se]c1ccccc1",          # selenium: outside the frozen ZINC vocabulary
    "[Na+].[Cl-]",           # no bonds after fragmentation
])
def test_bad_input_returns_none_rather_than_raising(bad):
    """Every rejection path returns None; nothing escapes as an exception.

    Kekulization itself cannot fail here -- ``MolFromSmiles`` sanitizes, and
    sanitization already kekulizes, so anything that parses is kekulizable. The
    try/except around ``Chem.Kekulize`` is defensive, not a live branch.
    """
    assert smiles_to_pyg_data(bad, KEK_ATOM_ENC, KEK_BOND_ENC, kekulize=True) is None


# ===========================================================================
# Validity conventions
# ===========================================================================
def _graph_from_smiles(smiles):
    return smiles_to_pyg_data(smiles, KEK_ATOM_ENC, KEK_BOND_ENC, kekulize=True)


def test_largest_fragment_picks_the_bigger_component():
    mol = Chem.MolFromSmiles("CCCCCC.C")
    assert largest_fragment_smiles(mol) == "CCCCCC"


def test_largest_fragment_requires_whole_molecule_to_sanitize():
    """Fragment-first would be more permissive than the published convention.

    A graph with one broken component and one good one must not score valid just
    because the good fragment survives on its own.
    """
    # Five-bonded carbon: the whole molecule fails to sanitize.
    broken = Chem.RWMol()
    for _ in range(6):
        broken.AddAtom(Chem.Atom("C"))
    for j in range(1, 6):
        broken.AddBond(0, j, Chem.rdchem.BondType.SINGLE)
    assert largest_fragment_smiles(broken) is None


def test_three_validity_conventions_are_ordered_and_present():
    samples = [_graph_from_smiles(s) for s in [
        "c1ccccc1", "CCO", "CC(=O)Nc1ccccc1", "CCCCCC",
    ]]
    rep = validity_report(samples, KEK_ATOM_DEC, KEK_BOND_DEC)
    for key in ("validity_relaxed_largest_frag", "validity_strict_largest_frag",
                "validity_whole_molecule"):
        assert key in rep
    # All four are clean neutral molecules, so every convention agrees at 1.0.
    assert rep["validity_relaxed_largest_frag"] == pytest.approx(1.0)
    assert rep["validity_strict_largest_frag"] == pytest.approx(1.0)
    assert rep["validity_whole_molecule"] == pytest.approx(1.0)
    assert rep["validity_convention"] == "relaxed_largest_frag"


def test_charge_correction_changes_the_strict_number():
    """The relaxed reading must be >= the strict one, and actually differ.

    An over-valent nitrogen is the canonical case: relaxed repairs it with a +1
    formal charge, strict rejects it. If these two numbers were always equal the
    'with/without valency correction' distinction would be decorative.
    """
    # Four single bonds on N -- valid only as N+.
    n = 5
    x = torch.zeros(n, len(KEK_ATOM_DEC))
    x[0, KEK_ATOM_ENC["N"]] = 1
    for i in range(1, n):
        x[i, KEK_ATOM_ENC["C"]] = 1
    rows, cols = [], []
    for j in range(1, n):
        rows += [0, j]
        cols += [j, 0]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    single = KEK_BOND_ENC[Chem.rdchem.BondType.SINGLE] + 1
    edge_attr = torch.zeros(edge_index.size(1), len(KEK_BOND_DEC))
    edge_attr[:, single] = 1
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    relaxed = pyg_data_to_mol(data, KEK_ATOM_DEC, KEK_BOND_DEC, charge_correction=True)
    strict = pyg_data_to_mol(data, KEK_ATOM_DEC, KEK_BOND_DEC, charge_correction=False)
    assert mol_to_smiles(relaxed) is not None
    assert mol_to_smiles(strict) is None

    rep = validity_report([data], KEK_ATOM_DEC, KEK_BOND_DEC)
    assert rep["validity_relaxed_largest_frag"] == pytest.approx(1.0)
    assert rep["validity_strict_largest_frag"] == pytest.approx(0.0)


def test_both_metric_conventions_are_emitted_and_consistent():
    """V.U. and V.U.N. must equal the per-valid chain, per protocol trap 1."""
    samples = [_graph_from_smiles(s) for s in [
        "c1ccccc1", "c1ccccc1", "CCO", "CC(=O)Nc1ccccc1",
    ]]
    reference = {"CCO"}  # one of the unique molecules is "known"
    rep = validity_report(samples, KEK_ATOM_DEC, KEK_BOND_DEC, reference_smiles=reference)

    assert rep["num_valid"] == 4
    assert rep["num_unique"] == 3          # benzene appears twice
    assert rep["uniqueness"] == pytest.approx(3 / 4)
    assert rep["novelty"] == pytest.approx(2 / 3)

    assert rep["v"] == pytest.approx(rep["validity_relaxed_largest_frag"])
    assert rep["v_u"] == pytest.approx(rep["v"] * rep["uniqueness"])
    assert rep["v_u_n"] == pytest.approx(rep["v"] * rep["uniqueness"] * rep["novelty"])


def test_validity_report_handles_empty_input():
    rep = validity_report([], KEK_ATOM_DEC, KEK_BOND_DEC, reference_smiles=set())
    assert rep["num_samples"] == 0
    assert rep["validity_relaxed_largest_frag"] == 0.0
    assert rep["v_u_n"] == 0.0


# ===========================================================================
# Hashing
# ===========================================================================
def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"defog" * 100000  # larger than one read chunk
    path.write_bytes(payload)
    assert zinc_reference.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_verify_reference_reports_missing_files(tmp_path):
    with pytest.raises(zinc_reference.ReferenceDataError, match="not found"):
        zinc_reference.verify_reference(tmp_path)


def test_hash_mismatch_is_fatal_by_default(tmp_path):
    """A tampered file must stop the run, not warn."""
    paths = zinc_reference.reference_paths(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths["csv"].write_text("smiles\nCCO\n")
    paths["test_idx"].write_text("[0]")

    with pytest.raises(zinc_reference.ReferenceDataError, match="pinned hashes"):
        zinc_reference.verify_reference(tmp_path)

    hashes = zinc_reference.verify_reference(tmp_path, allow_hash_mismatch=True)
    assert hashes["hash_verified"] is False


def test_wrong_row_count_is_fatal(tmp_path):
    """Counts are checked independently of hashes, so a same-size edit is caught."""
    paths = zinc_reference.reference_paths(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths["csv"].write_text("smiles\nCCO\nCCC\n")
    paths["test_idx"].write_text(json.dumps([0]))

    with pytest.raises(zinc_reference.ReferenceDataError, match="expected 249455"):
        zinc_reference.load_reference_split(
            tmp_path, download=False, allow_hash_mismatch=True
        )


# ===========================================================================
# The split itself
# ===========================================================================
@requires_reference
def test_reference_hashes_match_pinned():
    hashes = zinc_reference.verify_reference()
    assert hashes["hash_verified"] is True
    assert hashes["csv_sha256"] == zinc_reference.ZINC250K_CSV_SHA256
    assert hashes["test_idx_sha256"] == zinc_reference.ZINC250K_TEST_IDX_SHA256


@requires_reference
def test_split_counts_match_the_protocol():
    split = zinc_reference.load_reference_split(canonicalize=False)
    assert split.n_test == zinc_reference.N_TEST == 24887
    assert split.n_val == zinc_reference.DEFAULT_VAL_SIZE == 5000
    assert split.n_train == zinc_reference.N_TRAIN_FULL - split.n_val == 219568
    assert split.n_train + split.n_val + split.n_test == zinc_reference.N_TOTAL


@requires_reference
def test_splits_do_not_overlap():
    split = zinc_reference.load_reference_split(canonicalize=False)
    train, val, test = (set(split.train_smiles), set(split.val_smiles),
                        set(split.test_smiles))
    # Disjointness is asserted on indices inside the loader; here it is checked
    # on the molecules, which is what leakage would actually mean.
    assert not (train & val)
    assert not (val & test)
    assert not (train & test)


@requires_reference
def test_validation_comes_out_of_train_not_test():
    """The bug in the legacy path, asserted against.

    ``src/datasets/zinc_dataset.py`` hands back the same indices for ``val`` and
    ``test``, which makes any sampling sweep a sweep on test.
    """
    a = zinc_reference.load_reference_split(canonicalize=False, val_size=1000)
    b = zinc_reference.load_reference_split(canonicalize=False, val_size=9000)
    # Changing the validation size must not move a single test molecule.
    assert a.test_smiles == b.test_smiles
    # And the extra validation molecules must have come out of train.
    assert set(b.val_smiles) & set(a.train_smiles)
    assert not (set(b.val_smiles) & set(a.test_smiles))


@requires_reference
def test_split_is_deterministic_for_a_seed():
    a = zinc_reference.load_reference_split(canonicalize=False, split_seed=42)
    b = zinc_reference.load_reference_split(canonicalize=False, split_seed=42)
    c = zinc_reference.load_reference_split(canonicalize=False, split_seed=7)
    assert a.val_smiles == b.val_smiles
    assert a.val_smiles != c.val_smiles
    assert a.test_smiles == c.test_smiles  # seed never touches test


@requires_reference
def test_provenance_records_what_the_protocol_asks_for():
    split = zinc_reference.load_reference_split(canonicalize=False)
    p = split.provenance
    for key in ("csv_sha256", "test_idx_sha256", "n_train", "n_val", "n_test",
                "split_seed", "remove_h", "aromatic", "kekulized", "atom_types",
                "bond_types", "val_drawn_from"):
        assert key in p, f"provenance is missing {key}"
    assert p["val_drawn_from"] == "train"
    assert p["aromatic"] is False
    assert p["kekulized"] is True


@requires_reference
def test_frozen_vocabulary_encodes_the_reference_data():
    """A frozen vocabulary that drops molecules would train on a filtered set."""
    split = zinc_reference.load_reference_split(canonicalize=True)
    sample = split.train_smiles[:2000]
    graphs, kept, skipped = zinc_reference.build_graphs(sample)
    assert skipped == 0, f"{skipped}/2000 reference molecules failed to encode"
    assert len(graphs) == len(kept) == len(sample)


@requires_reference
def test_canonicalization_preserves_stereo_and_charge():
    """The reference set must NOT be flattened the way data/zinc_250k_rdkit.csv is.

    FCD, scaffold similarity and novelty all compare SMILES strings, so a
    stereo-stripped reference silently changes those numbers.
    """
    split = zinc_reference.load_reference_split(canonicalize=True)
    head = split.train_smiles[:20000]
    assert any("@" in s for s in head), "stereochemistry was stripped"
    assert any("+" in s or "-" in s for s in head), "formal charges were stripped"
