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


def test_learned_property_energy():
    head = PropertyHead(6, 4, hid=16, layers=2, prop_mean=0.0, prop_std=1.0)
    X, E, m = _toy_graph()
    target = 1.234
    energy = LearnedPropertyEnergy(head, target)(X, E, m)
    assert energy.shape == (4,)
    assert (energy >= 0).all()
    assert torch.allclose(energy, (head.predict(X, E, m).reshape(-1) - target) ** 2, atol=1e-5)


def test_energy_minimized_at_target():
    """Energy drops toward 0 as the target approaches the head's own prediction."""
    head = PropertyHead(6, 4, hid=16, layers=2)
    X, E, m = _toy_graph()
    pred0 = float(head.predict(X, E, m)[0])
    at_pred = float(LearnedPropertyEnergy(head, pred0)(X, E, m)[0])
    far = float(LearnedPropertyEnergy(head, pred0 + 10.0)(X, E, m)[0])
    assert at_pred < 1e-6 < far
