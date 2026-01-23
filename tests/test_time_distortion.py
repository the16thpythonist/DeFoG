"""
Tests for src/core/time_distortion.py

Tests cover:
- TimeDistorter creation
- Training time sampling
- Sampling time distortion
- Different distortion functions
"""

import pytest
import torch

from defog.core.time_distortion import TimeDistorter


class TestTimeDistorter:
    """Tests for TimeDistorter class."""

    def test_creation_defaults(self):
        """Test creation with default parameters."""
        distorter = TimeDistorter()

        assert distorter.train_distortion == "identity"
        assert distorter.sample_distortion == "identity"

    def test_creation_custom(self):
        """Test creation with custom parameters."""
        distorter = TimeDistorter(
            train_distortion="polydec",
            sample_distortion="cos",
        )

        assert distorter.train_distortion == "polydec"
        assert distorter.sample_distortion == "cos"

    def test_train_ft_shape(self):
        """Test train_ft produces correct shape."""
        distorter = TimeDistorter()
        bs = 16
        device = torch.device("cpu")

        t = distorter.train_ft(bs, device)

        assert t.shape == (bs, 1)

    def test_train_ft_range(self):
        """Test train_ft produces values in [0, 1]."""
        distorter = TimeDistorter()

        # Sample many times to check range
        for _ in range(10):
            t = distorter.train_ft(100, torch.device("cpu"))
            assert (t >= 0).all()
            assert (t <= 1).all()

    def test_train_ft_device(self):
        """Test train_ft respects device."""
        distorter = TimeDistorter()
        device = torch.device("cpu")

        t = distorter.train_ft(16, device)

        assert t.device == device

    def test_sample_ft_identity(self):
        """Test sample_ft with identity distortion."""
        distorter = TimeDistorter()
        t_input = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])

        t_output = distorter.sample_ft(t_input, "identity")

        assert torch.allclose(t_input, t_output)

    def test_sample_ft_polydec(self):
        """Test sample_ft with polydec distortion."""
        distorter = TimeDistorter()
        t_input = torch.tensor([[0.0], [0.5], [1.0]])

        t_output = distorter.sample_ft(t_input, "polydec")

        # polydec: f(t) = 2t - t^2
        # f(0) = 0, f(0.5) = 0.75, f(1) = 1
        expected = torch.tensor([[0.0], [0.75], [1.0]])
        assert torch.allclose(t_output, expected)

    def test_sample_ft_preserves_endpoints(self):
        """Test all distortions preserve endpoints 0 and 1."""
        distorter = TimeDistorter()
        t_endpoints = torch.tensor([[0.0], [1.0]])

        for distortion in ["identity", "polydec", "polyinc", "cos", "revcos"]:
            t_output = distorter.sample_ft(t_endpoints, distortion)
            assert torch.allclose(t_output[0], torch.tensor([0.0]), atol=1e-6)
            assert torch.allclose(t_output[1], torch.tensor([1.0]), atol=1e-6)

    def test_sample_ft_monotonic(self):
        """Test distorted time is monotonically increasing."""
        distorter = TimeDistorter()
        t_input = torch.linspace(0, 1, 100).unsqueeze(1)

        for distortion in ["identity", "polydec", "polyinc", "cos", "revcos"]:
            t_output = distorter.sample_ft(t_input, distortion)
            # Check monotonicity
            diffs = t_output[1:] - t_output[:-1]
            assert (diffs >= -1e-6).all(), f"{distortion} is not monotonic"

    def test_sample_ft_range(self):
        """Test distorted time stays in [0, 1]."""
        distorter = TimeDistorter()
        t_input = torch.linspace(0, 1, 100).unsqueeze(1)

        for distortion in ["identity", "polydec", "polyinc", "cos", "revcos"]:
            t_output = distorter.sample_ft(t_input, distortion)
            assert (t_output >= -1e-6).all()
            assert (t_output <= 1 + 1e-6).all()

    def test_polydec_vs_polyinc(self):
        """Test polydec and polyinc are different."""
        distorter = TimeDistorter()
        t_input = torch.tensor([[0.3], [0.5], [0.7]])

        t_polydec = distorter.sample_ft(t_input, "polydec")
        t_polyinc = distorter.sample_ft(t_input, "polyinc")

        # polydec should map to higher values (faster early)
        # polyinc should map to lower values (slower early)
        assert (t_polydec > t_input).all()
        assert (t_polyinc < t_input).all()

    def test_cos_vs_revcos(self):
        """Test cos and revcos are different."""
        distorter = TimeDistorter()
        t_input = torch.tensor([[0.25], [0.5], [0.75]])

        t_cos = distorter.sample_ft(t_input, "cos")
        t_revcos = distorter.sample_ft(t_input, "revcos")

        # They should be different
        assert not torch.allclose(t_cos, t_revcos)

    def test_different_batch_sizes(self):
        """Test works with different batch sizes."""
        distorter = TimeDistorter()

        for bs in [1, 5, 32, 128]:
            t = distorter.train_ft(bs, torch.device("cpu"))
            assert t.shape == (bs, 1)
