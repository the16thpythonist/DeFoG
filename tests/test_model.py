"""
Tests for src/core/model.py

Tests cover:
- DeFoGModel creation and configuration
- Training step (forward pass, loss computation)
- Sampling with various parameters
- Save and load functionality
- from_dataloader factory method
"""

import pytest
import torch
import tempfile
import os
from torch_geometric.data import Data, Batch

from defog.core import DeFoGModel


class TestDeFoGModelCreation:
    """Tests for model creation and initialization."""

    def test_basic_creation(self, small_model_config, node_counts_distribution):
        """Test basic model creation with minimal config."""
        model = DeFoGModel(
            num_node_classes=4,
            num_edge_classes=2,
            noise_type="uniform",
            node_counts=node_counts_distribution,
        )

        assert model.num_node_classes == 4
        assert model.num_edge_classes == 2

    def test_full_config_creation(self, small_model_config, node_counts_distribution):
        """Test model creation with full configuration."""
        model = DeFoGModel(
            **small_model_config,
            noise_type="uniform",
            node_counts=node_counts_distribution,
        )

        assert model.num_node_classes == small_model_config["num_node_classes"]
        assert model.num_edge_classes == small_model_config["num_edge_classes"]

    def test_marginal_noise_type(self, small_model_config, node_counts_distribution,
                                  node_marginals, edge_marginals):
        """Test model creation with marginal noise type."""
        model = DeFoGModel(
            **small_model_config,
            noise_type="marginal",
            node_marginals=node_marginals,
            edge_marginals=edge_marginals,
            node_counts=node_counts_distribution,
        )

        assert model.limit_dist.noise_type == "marginal"

    def test_input_output_dims(self, small_model, small_model_config):
        """Test input and output dimensions are set correctly."""
        # Input dims should include extra features
        assert small_model.input_dims["X"] >= small_model_config["num_node_classes"]
        assert small_model.input_dims["E"] >= small_model_config["num_edge_classes"]

        # Output dims should match number of classes
        assert small_model.output_dims["X"] == small_model_config["num_node_classes"]
        assert small_model.output_dims["E"] == small_model_config["num_edge_classes"]

    def test_repr(self, small_model):
        """Test string representation."""
        repr_str = repr(small_model)
        assert "DeFoGModel" in repr_str
        assert "node_classes" in repr_str
        assert "edge_classes" in repr_str


class TestDeFoGModelTraining:
    """Tests for training functionality."""

    def test_training_step_runs(self, small_model, graph_batch):
        """Test training step executes without errors."""
        small_model.train()
        result = small_model.training_step(graph_batch, 0)

        assert result is not None
        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)

    def test_training_step_loss_finite(self, small_model, graph_batch):
        """Test training step produces finite loss."""
        small_model.train()
        result = small_model.training_step(graph_batch, 0)

        assert torch.isfinite(result["loss"])

    def test_training_step_gradients(self, small_model, graph_batch):
        """Test training step enables gradient computation."""
        small_model.train()
        result = small_model.training_step(graph_batch, 0)

        # Loss should have grad_fn
        assert result["loss"].requires_grad

        # Should be able to backprop
        result["loss"].backward()

        # At least some parameters should have gradients
        has_grad = any(p.grad is not None for p in small_model.parameters())
        assert has_grad

    def test_configure_optimizers(self, small_model):
        """Test optimizer configuration."""
        optimizer = small_model.configure_optimizers()

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)


class TestDeFoGModelSampling:
    """Tests for sampling functionality."""

    def test_sample_basic(self, small_model):
        """Test basic sampling."""
        small_model.eval()
        samples = small_model.sample(
            num_samples=3,
            sample_steps=3,
            show_progress=False,
        )

        assert len(samples) == 3
        for s in samples:
            assert isinstance(s, Data)
            assert hasattr(s, "x")
            assert hasattr(s, "edge_index")
            assert hasattr(s, "edge_attr")

    def test_sample_fixed_nodes(self, small_model):
        """Test sampling with fixed number of nodes."""
        small_model.eval()
        samples = small_model.sample(
            num_samples=3,
            num_nodes=5,
            sample_steps=3,
            show_progress=False,
        )

        for s in samples:
            assert s.x.shape[0] == 5

    def test_sample_variable_nodes(self, small_model):
        """Test sampling with variable node counts."""
        small_model.eval()
        node_counts = torch.tensor([3, 5, 7])
        samples = small_model.sample(
            num_samples=3,
            num_nodes=node_counts,
            sample_steps=3,
            show_progress=False,
        )

        for i, s in enumerate(samples):
            assert s.x.shape[0] == node_counts[i].item()

    def test_sample_with_eta(self, small_model):
        """Test sampling with stochasticity parameter."""
        small_model.eval()
        samples = small_model.sample(
            num_samples=2,
            eta=10.0,
            sample_steps=3,
            show_progress=False,
        )

        assert len(samples) == 2

    def test_sample_with_omega(self, small_model):
        """Test sampling with target guidance."""
        small_model.eval()
        samples = small_model.sample(
            num_samples=2,
            omega=0.1,
            sample_steps=3,
            show_progress=False,
        )

        assert len(samples) == 2

    def test_sample_with_time_distortion(self, small_model):
        """Test sampling with different time distortions."""
        small_model.eval()

        for distortion in ["identity", "polydec", "cos"]:
            samples = small_model.sample(
                num_samples=2,
                time_distortion=distortion,
                sample_steps=3,
                show_progress=False,
            )
            assert len(samples) == 2

    def test_sample_output_validity(self, small_model):
        """Test sampled graphs have valid structure."""
        small_model.eval()
        samples = small_model.sample(
            num_samples=5,
            sample_steps=5,
            show_progress=False,
        )

        for s in samples:
            # Node features should be one-hot
            assert s.x.dim() == 2
            assert s.x.shape[1] == small_model.num_node_classes
            assert torch.allclose(s.x.sum(dim=1), torch.ones(s.x.shape[0]))

            # Edge indices should be valid
            if s.edge_index.numel() > 0:
                assert s.edge_index.max() < s.x.shape[0]
                assert s.edge_index.min() >= 0


class TestDeFoGModelSaveLoad:
    """Tests for save and load functionality."""

    def test_save_creates_file(self, small_model):
        """Test save creates checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            saved_path = small_model.save(path)

            assert os.path.exists(saved_path)
            assert saved_path.endswith(".ckpt")

    def test_save_appends_extension(self, small_model):
        """Test save auto-appends .ckpt extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_no_ext")
            saved_path = small_model.save(path)

            assert saved_path == path + ".ckpt"

    def test_save_with_extension(self, small_model):
        """Test save with existing .ckpt extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.ckpt")
            saved_path = small_model.save(path)

            assert saved_path == path

    def test_load_basic(self, small_model):
        """Test basic model loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_model.save(path)

            loaded = DeFoGModel.load(path)

            assert isinstance(loaded, DeFoGModel)
            assert loaded.num_node_classes == small_model.num_node_classes
            assert loaded.num_edge_classes == small_model.num_edge_classes

    def test_load_appends_extension(self, small_model):
        """Test load auto-appends .ckpt extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_model.save(path)

            # Load without extension
            loaded = DeFoGModel.load(path)
            assert loaded is not None

    def test_load_eval_mode(self, small_model):
        """Test loaded model is in eval mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_model.save(path)

            loaded = DeFoGModel.load(path)

            assert not loaded.training

    def test_load_to_device(self, small_model):
        """Test loading to specific device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_model.save(path)

            loaded = DeFoGModel.load(path, device="cpu")

            param_device = next(loaded.parameters()).device
            assert param_device == torch.device("cpu")

    def test_save_load_preserves_weights(self, small_model):
        """Test save/load preserves model weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_model.save(path)

            loaded = DeFoGModel.load(path)

            # Compare state dicts
            for (name1, param1), (name2, param2) in zip(
                small_model.state_dict().items(),
                loaded.state_dict().items()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2)

    def test_loaded_model_can_sample(self, small_model):
        """Test loaded model can generate samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_model.save(path)

            loaded = DeFoGModel.load(path)
            samples = loaded.sample(num_samples=2, sample_steps=3, show_progress=False)

            assert len(samples) == 2


class TestDeFoGModelFromDataloader:
    """Tests for from_dataloader factory method."""

    def test_from_dataloader_basic(self, small_dataloader):
        """Test creating model from dataloader."""
        model = DeFoGModel.from_dataloader(
            small_dataloader,
            n_layers=2,
            hidden_dim=32,
            noise_type="uniform",
        )

        assert isinstance(model, DeFoGModel)
        assert model.num_node_classes == 4
        assert model.num_edge_classes == 2

    def test_from_dataloader_marginal(self, small_dataloader):
        """Test creating model with marginal noise from dataloader."""
        model = DeFoGModel.from_dataloader(
            small_dataloader,
            n_layers=2,
            hidden_dim=32,
            noise_type="marginal",
        )

        assert model.limit_dist.noise_type == "marginal"

    def test_from_dataloader_can_train(self, small_dataloader):
        """Test model from dataloader can be trained."""
        model = DeFoGModel.from_dataloader(
            small_dataloader,
            n_layers=2,
            hidden_dim=32,
            noise_type="uniform",
        )

        batch = next(iter(small_dataloader))
        model.train()
        result = model.training_step(batch, 0)

        assert torch.isfinite(result["loss"])

    def test_from_dataloader_can_sample(self, small_dataloader):
        """Test model from dataloader can sample."""
        model = DeFoGModel.from_dataloader(
            small_dataloader,
            n_layers=2,
            hidden_dim=32,
            noise_type="uniform",
        )

        model.eval()
        samples = model.sample(num_samples=2, sample_steps=3, show_progress=False)

        assert len(samples) == 2
