"""
Tests for conditioning and classifier-free guidance in DeFoGModel.

Tests cover:
- Conditional model creation and dimension handling
- CFG condition dropout during training
- Conditional sampling with guidance scale
- Unconditional sampling on conditional models
- Save/load preservation of conditioning parameters
- Backward compatibility (cond_dim=0 behaves as before)
"""

import pytest
import torch
import tempfile
import os
from torch_geometric.data import Data, Batch

from defog.core import DeFoGModel


class TestConditionalModelCreation:
    """Tests for conditional model initialization."""

    def test_cond_dim_stored(self, small_cond_model, cond_dim):
        """Test cond_dim is stored correctly."""
        assert small_cond_model.cond_dim == cond_dim

    def test_cond_drop_prob_stored(self, small_cond_model):
        """Test cond_drop_prob is stored correctly."""
        assert small_cond_model.cond_drop_prob == 0.1

    def test_guidance_scale_stored(self, small_cond_model):
        """Test guidance_scale is stored correctly."""
        assert small_cond_model.guidance_scale == 2.0

    def test_input_dims_include_condition(self, small_cond_model):
        """Test input_dims['y'] accounts for the condition embedding, not cond_dim."""
        # y dimension = cond_emb_dim + 1 (time) + extra_features_y
        assert small_cond_model.input_dims["y"] >= small_cond_model.cond_emb_dim + 1

    def test_conditioning_modules_exist(self, small_cond_model):
        """Test the learned encoder, norm, and null embedding are created."""
        assert hasattr(small_cond_model, "cond_encoder")
        assert hasattr(small_cond_model, "cond_norm")
        assert isinstance(small_cond_model.null_embedding, torch.nn.Parameter)
        assert small_cond_model.null_embedding.shape == (small_cond_model.cond_emb_dim,)

    def test_repr_includes_cond_info(self, small_cond_model):
        """Test repr includes conditioning info for conditional models."""
        repr_str = repr(small_cond_model)
        assert "cond_dim" in repr_str
        assert "cond_drop_prob" in repr_str
        assert "guidance_scale" in repr_str

    def test_repr_excludes_cond_info_when_unconditional(self, small_model):
        """Test repr excludes conditioning info for unconditional models."""
        repr_str = repr(small_model)
        assert "cond_dim" not in repr_str


class TestCFGTraining:
    """Tests for classifier-free guidance during training."""

    def test_training_step_with_condition(self, small_cond_model, cond_graph_batch):
        """Test training step runs with conditioned batch."""
        small_cond_model.train()
        result = small_cond_model.training_step(cond_graph_batch, 0)

        assert result is not None
        assert "loss" in result
        assert torch.isfinite(result["loss"])

    def test_training_step_loss_has_gradient(self, small_cond_model, cond_graph_batch):
        """Test loss from conditional training step supports backprop."""
        small_cond_model.train()
        result = small_cond_model.training_step(cond_graph_batch, 0)

        assert result["loss"].requires_grad
        result["loss"].backward()

        has_grad = any(p.grad is not None for p in small_cond_model.parameters())
        assert has_grad

    def test_cfg_dropout_replaces_with_null_embedding(self, small_model_config,
                                                       node_counts_distribution):
        """Dropped rows map to the (LayerNorm'd) learned null embedding."""
        model = DeFoGModel(
            **small_model_config,
            noise_type="uniform",
            node_counts=node_counts_distribution,
            cond_dim=2,
        )
        model.eval()

        bs = 4
        cond = torch.randn(bs, 2)
        drop_mask = torch.tensor([True, False, True, False])
        emb = model._embed_condition(cond, drop_mask=drop_mask)

        null = model.cond_norm(model.null_embedding)
        # Dropped rows equal the null embedding, kept rows do not.
        assert torch.allclose(emb[0], null, atol=1e-6)
        assert torch.allclose(emb[2], null, atol=1e-6)
        assert not torch.allclose(emb[1], null, atol=1e-6)
        assert not torch.allclose(emb[3], null, atol=1e-6)

    def test_cfg_no_dropout_uses_encoder_output(self, small_model_config,
                                                node_counts_distribution):
        """With no dropout, embeddings are pure encoder outputs (no null)."""
        model = DeFoGModel(
            **small_model_config,
            noise_type="uniform",
            node_counts=node_counts_distribution,
            cond_dim=2,
        )
        model.eval()

        cond = torch.randn(4, 2)
        drop_mask = torch.zeros(4, dtype=torch.bool)
        emb = model._embed_condition(cond, drop_mask=drop_mask)
        expected = model.cond_norm(model.cond_encoder(cond))

        assert torch.allclose(emb, expected, atol=1e-6)

    def test_null_embedding_receives_gradient(self, small_model_config,
                                              node_counts_distribution,
                                              cond_graph_batch):
        """With 100% dropout, the null embedding is trained and the encoder is not."""
        model = DeFoGModel(
            **small_model_config,
            noise_type="uniform",
            node_counts=node_counts_distribution,
            cond_dim=2,
            cond_drop_prob=1.0,
        )
        model.train()
        result = model.training_step(cond_graph_batch, 0)
        result["loss"].backward()

        assert model.null_embedding.grad is not None
        assert model.null_embedding.grad.abs().sum() > 0
        # Encoder is never used when every row is dropped.
        enc_grads = [p.grad for p in model.cond_encoder.parameters() if p.grad is not None]
        assert all(g.abs().sum() == 0 for g in enc_grads)

    def test_encoder_receives_gradient_without_dropout(self, small_model_config,
                                                       node_counts_distribution,
                                                       cond_graph_batch):
        """With 0% dropout, the condition encoder receives gradient."""
        model = DeFoGModel(
            **small_model_config,
            noise_type="uniform",
            node_counts=node_counts_distribution,
            cond_dim=2,
            cond_drop_prob=0.0,
        )
        model.train()
        result = model.training_step(cond_graph_batch, 0)
        result["loss"].backward()

        enc_grad = sum(
            p.grad.abs().sum() for p in model.cond_encoder.parameters()
            if p.grad is not None
        )
        assert enc_grad > 0

    def test_training_without_batch_y(self, small_cond_model, graph_batch):
        """Test training step handles missing batch.y gracefully (uses zeros)."""
        small_cond_model.train()
        # graph_batch has no y attribute
        result = small_cond_model.training_step(graph_batch, 0)

        assert result is not None
        assert torch.isfinite(result["loss"])


class TestConditionalSampling:
    """Tests for conditional sampling with CFG."""

    def test_sample_with_condition(self, small_cond_model, cond_dim):
        """Test sampling with a provided condition vector."""
        small_cond_model.eval()
        num_samples = 3
        condition = torch.randn(num_samples, cond_dim)

        samples = small_cond_model.sample(
            num_samples=num_samples,
            num_nodes=4,
            condition=condition,
            sample_steps=3,
            show_progress=False,
        )

        assert len(samples) == num_samples
        for s in samples:
            assert isinstance(s, Data)
            assert s.x.shape[0] == 4

    def test_sample_with_guidance_scale(self, small_cond_model, cond_dim):
        """Test sampling with explicit guidance scale."""
        small_cond_model.eval()
        num_samples = 2
        condition = torch.randn(num_samples, cond_dim)

        samples = small_cond_model.sample(
            num_samples=num_samples,
            num_nodes=4,
            condition=condition,
            guidance_scale=3.0,
            sample_steps=3,
            show_progress=False,
        )

        assert len(samples) == num_samples

    def test_sample_guidance_scale_1_no_cfg(self, small_cond_model, cond_dim):
        """Test guidance_scale=1.0 runs single forward pass (no CFG blending)."""
        small_cond_model.eval()
        num_samples = 2
        condition = torch.randn(num_samples, cond_dim)

        # Should work without error (single forward pass path)
        samples = small_cond_model.sample(
            num_samples=num_samples,
            num_nodes=4,
            condition=condition,
            guidance_scale=1.0,
            sample_steps=3,
            show_progress=False,
        )

        assert len(samples) == num_samples

    def test_sample_without_condition_uses_null(self, small_cond_model):
        """Test sampling without condition on conditional model uses null embedding."""
        small_cond_model.eval()

        # Should not raise, generates unconditionally via the null embedding.
        samples = small_cond_model.sample(
            num_samples=2,
            num_nodes=4,
            sample_steps=3,
            show_progress=False,
        )

        assert len(samples) == 2
        # No condition provided -> no condition attached to the outputs.
        for s in samples:
            assert not hasattr(s, "y") or s.y is None

    def test_sample_attaches_raw_condition(self, small_cond_model, cond_dim):
        """Sampled graphs carry the raw condition (not the internal embedding)."""
        small_cond_model.eval()
        condition = torch.randn(3, cond_dim)

        samples = small_cond_model.sample(
            num_samples=3,
            num_nodes=4,
            condition=condition,
            sample_steps=3,
            show_progress=False,
        )

        for i, s in enumerate(samples):
            assert s.y.shape == (1, cond_dim)
            assert torch.allclose(s.y[0], condition[i])

    def test_sample_1d_condition_unsqueezed(self, small_model_config,
                                             node_counts_distribution):
        """Test that a 1D condition tensor is handled correctly."""
        model = DeFoGModel(
            **small_model_config,
            noise_type="uniform",
            node_counts=node_counts_distribution,
            cond_dim=1,
        )
        model.eval()

        # 1D tensor should be unsqueezed to (num_samples, 1)
        condition = torch.randn(3)
        samples = model.sample(
            num_samples=3,
            num_nodes=4,
            condition=condition,
            sample_steps=3,
            show_progress=False,
        )

        assert len(samples) == 3

    def test_sample_output_validity(self, small_cond_model, cond_dim):
        """Test sampled graphs from conditional model have valid structure."""
        small_cond_model.eval()
        condition = torch.randn(4, cond_dim)

        samples = small_cond_model.sample(
            num_samples=4,
            num_nodes=5,
            condition=condition,
            sample_steps=5,
            show_progress=False,
        )

        for s in samples:
            # Node features should be one-hot
            assert s.x.dim() == 2
            assert s.x.shape[1] == small_cond_model.num_node_classes
            assert torch.allclose(s.x.sum(dim=1), torch.ones(s.x.shape[0]))

            # Edge indices should be valid
            if s.edge_index.numel() > 0:
                assert s.edge_index.max() < s.x.shape[0]
                assert s.edge_index.min() >= 0


class TestConditionalSaveLoad:
    """Tests for save/load with conditional models."""

    def test_save_load_preserves_cond_dim(self, small_cond_model, cond_dim):
        """Test cond_dim is preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_cond_model.save(path)

            loaded = DeFoGModel.load(path)

            assert loaded.cond_dim == cond_dim

    def test_save_load_preserves_cond_drop_prob(self, small_cond_model):
        """Test cond_drop_prob is preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_cond_model.save(path)

            loaded = DeFoGModel.load(path)

            assert loaded.cond_drop_prob == small_cond_model.cond_drop_prob

    def test_save_load_preserves_guidance_scale(self, small_cond_model):
        """Test guidance_scale is preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_cond_model.save(path)

            loaded = DeFoGModel.load(path)

            assert loaded.guidance_scale == small_cond_model.guidance_scale

    def test_loaded_cond_model_can_sample(self, small_cond_model, cond_dim):
        """Test loaded conditional model can generate samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_cond_model.save(path)

            loaded = DeFoGModel.load(path)
            condition = torch.randn(2, cond_dim)
            samples = loaded.sample(
                num_samples=2,
                num_nodes=4,
                condition=condition,
                sample_steps=3,
                show_progress=False,
            )

            assert len(samples) == 2

    def test_save_load_preserves_weights(self, small_cond_model):
        """Test save/load preserves model weights for conditional model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            small_cond_model.save(path)

            loaded = DeFoGModel.load(path)

            for (name1, param1), (name2, param2) in zip(
                small_cond_model.state_dict().items(),
                loaded.state_dict().items(),
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2)


class TestBackwardCompatibility:
    """Tests to ensure cond_dim=0 (default) preserves original behavior."""

    def test_default_cond_dim_is_zero(self, small_model):
        """Test default model has cond_dim=0."""
        assert small_model.cond_dim == 0

    def test_uncond_model_has_no_conditioning_modules(self, small_model):
        """Test cond_dim=0 model does not create conditioning parameters."""
        assert not hasattr(small_model, "cond_encoder")
        assert not hasattr(small_model, "null_embedding")
        assert not hasattr(small_model, "cond_norm")

    def test_uncond_model_input_dims_unchanged(self, small_model_config,
                                                node_counts_distribution):
        """Test input_dims['y'] is unchanged for unconditional model."""
        model = DeFoGModel(
            **small_model_config,
            noise_type="uniform",
            node_counts=node_counts_distribution,
            extra_features_type="none",
        )

        # Without conditioning or extra features: y = time only
        assert model.input_dims["y"] == 1

    def test_uncond_training_step(self, small_model, graph_batch):
        """Test unconditional training still works."""
        small_model.train()
        result = small_model.training_step(graph_batch, 0)

        assert result is not None
        assert torch.isfinite(result["loss"])

    def test_uncond_sampling(self, small_model):
        """Test unconditional sampling still works."""
        small_model.eval()
        samples = small_model.sample(
            num_samples=2,
            sample_steps=3,
            show_progress=False,
        )

        assert len(samples) == 2
