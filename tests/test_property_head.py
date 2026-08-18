import torch

from defog.core import PropertyHead, LearnedPropertyEnergy


def _toy_graph(bs=4, n=5, na=6, nb=4):
    X = torch.zeros(bs, n, na); X[..., 0] = 1.0          # one-hot node type 0
    E = torch.zeros(bs, n, n, nb); E[..., 0] = 1.0       # one-hot "no bond"
    mask = torch.ones(bs, n, dtype=torch.bool)
    return X, E, mask


def test_forward_shape():
    head = PropertyHead(6, 4, hid=16, layers=2)
    X, E, m = _toy_graph()
    assert head(X, E, m).shape == (4,)


def test_predict_unnormalizes():
    head = PropertyHead(6, 4, hid=16, layers=2, prop_mean=2.0, prop_std=3.0)
    X, E, m = _toy_graph()
    assert torch.allclose(head.predict(X, E, m), head(X, E, m) * 3.0 + 2.0, atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    head = PropertyHead(6, 4, hid=16, layers=2, prop_mean=1.5, prop_std=2.5)
    X, E, m = _toy_graph()
    before = head.predict(X, E, m)
    head2 = PropertyHead.load(head.save(str(tmp_path / "h.ckpt")))
    assert torch.allclose(before, head2.predict(X, E, m), atol=1e-5)
    assert abs(float(head2.prop_mean) - 1.5) < 1e-4 and abs(float(head2.prop_std) - 2.5) < 1e-4


def test_load_experiment_format(tmp_path):
    """The training experiment saves a state_dict WITHOUT the prop_mean/std buffers plus
    separate scalar keys — PropertyHead.load must reconstruct those from the scalars."""
    ref = PropertyHead(6, 4, hid=16, layers=2, prop_mean=3.3, prop_std=1.1)
    sd = {k: v for k, v in ref.state_dict().items() if k not in ("prop_mean", "prop_std")}
    ck = {"state_dict": sd, "na": 6, "nb": 4, "hid": 16, "layers": 2, "prop_mean": 3.3, "prop_std": 1.1}
    p = str(tmp_path / "exp.ckpt"); torch.save(ck, p)
    head = PropertyHead.load(p)
    X, E, m = _toy_graph()
    assert torch.allclose(head.predict(X, E, m), ref.predict(X, E, m), atol=1e-5)
    assert abs(float(head.prop_mean) - 3.3) < 1e-4 and abs(float(head.prop_std) - 1.1) < 1e-4


def test_learned_property_energy_constructs():
    """LearnedPropertyEnergy decodes/re-encodes via the domain before the head, so its full
    __call__ is exercised in the integration validation (validate_head_fk); here just smoke
    the construction + descriptor with a stub domain."""
    head = PropertyHead(6, 4, hid=16, layers=2)
    e = LearnedPropertyEnergy(head, 1.5, domain=object(), atom_encoder={}, bond_encoder={})
    assert e.target == 1.5 and "1.5" in e._desc()


# ---------------------------------------------------------------- re-encode vocabulary
# LearnedPropertyEnergy decodes each predicted-clean graph to a molecule and re-encodes it
# from `Chem.MolToSmiles`, which emits AROMATIC bonds. On a kekulized base (zinc-kek,
# union-kek) the bond vocabulary is {SINGLE, DOUBLE, TRIPLE} with no aromatic class, so that
# round trip used to reject ~94% of real drug-like molecules and fall through to
# `invalid_energy=1e3`. The head was then never consulted and FK resampled on "did the
# re-encode happen to succeed" instead of on the property.

AROMATIC_SMILES = "Cc1cccn2c(=O)c(C(=O)NCC3CCOC3)cnc12"


def _mol_dense(smiles, atom_encoder, bond_encoder):
    from torch_geometric.data import Batch

    from defog.core.data import to_dense
    from defog.domains.molecule import needs_kekulize, smiles_to_pyg_data
    d = smiles_to_pyg_data(smiles, atom_encoder, bond_encoder,
                           kekulize=needs_kekulize(bond_encoder))
    batch = Batch.from_data_list([d])
    dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    return dense.mask(mask), mask


def test_needs_kekulize_reads_the_vocabulary():
    from defog.domains.molecule import build_encoders, needs_kekulize
    _, _, kek_be, _ = build_encoders(list("CNOF"), ["SINGLE", "DOUBLE", "TRIPLE"])
    _, _, aro_be, _ = build_encoders(list("CNOF"), ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])
    assert needs_kekulize(kek_be) is True
    assert needs_kekulize(aro_be) is False


def test_energy_scores_aromatic_molecule_on_kekulized_vocabulary():
    """The regression: a real aromatic molecule must get a REAL energy, not invalid_energy."""
    from defog.domains.molecule import MoleculeDomain, build_encoders
    atoms = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    ae, adec, be, bdec = build_encoders(atoms, ["SINGLE", "DOUBLE", "TRIPLE"])
    head = PropertyHead(len(atoms), 4, hid=16, layers=2)
    energy = LearnedPropertyEnergy(head, 2.5, MoleculeDomain(adec, bdec), ae, be,
                                   invalid_energy=1e3)
    dense, mask = _mol_dense(AROMATIC_SMILES, ae, be)
    out = energy(dense.X, dense.E, mask)
    assert out.shape == (1,)
    assert float(out[0]) < 1e3 - 1, (
        "aromatic molecule scored invalid_energy on a kekulized vocabulary: the re-encode "
        "is not kekulizing, so the head is never consulted"
    )
