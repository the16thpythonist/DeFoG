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
