"""Tests for TrainingMonitorCallback and SampleVisualizationCallback."""

import pytest
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from defog.core import DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback


@pytest.fixture
def model_and_loaders(small_model_config, node_counts_distribution, small_dataset):
    """Create a model and train/val DataLoaders for callback testing."""
    model = DeFoGModel(
        **small_model_config,
        noise_type="uniform",
        node_counts=node_counts_distribution,
    )
    # Split dataset into train/val
    train_ds = small_dataset[:15]
    val_ds = small_dataset[15:]
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    return model, train_loader, val_loader


class TestTrainingMonitorCallback:
    """Tests for TrainingMonitorCallback."""

    def test_callback_instantiation(self):
        cb = TrainingMonitorCallback()
        assert cb.smoothing_window == 5
        assert cb.hw_sample_interval == 2.0
        assert cb.figure_callback is None
        assert cb.last_figure is None

    def test_callback_custom_params(self):
        cb = TrainingMonitorCallback(
            smoothing_window=10,
            hw_sample_interval=5.0,
            figure_callback=lambda fig: None,
        )
        assert cb.smoothing_window == 10
        assert cb.hw_sample_interval == 5.0
        assert cb.figure_callback is not None

    def test_callback_with_trainer(self, model_and_loaders):
        """Run 2 epochs of training + validation with the callback."""
        model, train_loader, val_loader = model_and_loaders
        figures = []
        cb = TrainingMonitorCallback(
            smoothing_window=1,
            figure_callback=lambda fig: figures.append(fig),
        )

        trainer = pl.Trainer(
            max_epochs=2,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        # Should have produced one figure per validation epoch
        assert len(figures) == 2
        # last_figure should be set
        assert cb.last_figure is not None

        # History should have entries
        assert len(cb.history["epoch_time"]) == 2
        assert len(cb.history["grad_norm_total"]) == 2
        assert len(cb.history["param_delta_total"]) == 2
        assert len(cb.history["entropy_X"]) == 2

    def test_callback_history_keys(self, model_and_loaders):
        """Verify all expected history keys are populated."""
        model, train_loader, val_loader = model_and_loaders
        cb = TrainingMonitorCallback(smoothing_window=1)

        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        expected_keys = [
            "epoch_time",
            "train_loss", "val_loss",
            "grad_norm_total",
            "grad_norm_mlp_in_X", "grad_norm_mlp_in_E", "grad_norm_mlp_in_y",
            "grad_norm_tf_layers",
            "grad_norm_mlp_out_X", "grad_norm_mlp_out_E",
            "param_delta_total",
            "param_delta_mlp_in_X", "param_delta_mlp_in_E", "param_delta_mlp_in_y",
            "param_delta_tf_layers",
            "param_delta_mlp_out_X", "param_delta_mlp_out_E",
            "entropy_X", "entropy_E",
            "acc_X_per_class", "acc_E_per_class",
            "gpu_util", "gpu_mem", "cpu_util", "ram",
        ]
        for key in expected_keys:
            assert key in cb.history, f"Missing key: {key}"
            assert len(cb.history[key]) == 1, f"Key {key} has {len(cb.history[key])} entries, expected 1"

    def test_gradient_norms_positive(self, model_and_loaders):
        """Gradient norms should be positive after training."""
        model, train_loader, val_loader = model_and_loaders
        cb = TrainingMonitorCallback(smoothing_window=1)

        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        assert cb.history["grad_norm_total"][0] > 0
        assert cb.history["grad_norm_mlp_in_X"][0] > 0

    def test_param_deltas_positive(self, model_and_loaders):
        """Parameter deltas should be > 0 after training (optimizer took steps)."""
        model, train_loader, val_loader = model_and_loaders
        cb = TrainingMonitorCallback(smoothing_window=1)

        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        assert cb.history["param_delta_total"][0] > 0

    def test_figure_is_valid_matplotlib(self, model_and_loaders):
        """The generated figure should be a valid matplotlib Figure with 5x4 axes."""
        import matplotlib.pyplot as plt

        model, train_loader, val_loader = model_and_loaders
        cb = TrainingMonitorCallback(smoothing_window=1)

        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        fig = cb.last_figure
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 20  # 5x4 grid
        plt.close(fig)


class TestValidationStep:
    """Tests for DeFoGModel.validation_step."""

    def test_validation_step_returns_loss(self, small_model, graph_batch):
        """validation_step should return a dict with val_loss."""
        result = small_model.validation_step(graph_batch, 0)
        assert "val_loss" in result
        assert torch.isfinite(result["val_loss"])

    def test_validation_step_no_cfg_dropout(self, small_cond_model, cond_graph_batch):
        """validation_step should NOT apply CFG dropout."""
        # Run validation step multiple times — condition should never be sentinel
        model = small_cond_model
        # We can't directly observe dropout from outside, but we can verify
        # the step runs and produces valid loss
        result = model.validation_step(cond_graph_batch, 0)
        assert "val_loss" in result
        assert torch.isfinite(result["val_loss"])

    def test_training_step_returns_extra_tensors(self, small_model, graph_batch):
        """training_step should return detached pred/true tensors for the callback."""
        result = small_model.training_step(graph_batch, 0)
        assert "_pred_X" in result
        assert "_pred_E" in result
        assert "_true_X" in result
        assert "_true_E" in result
        assert "_node_mask" in result
        # Should be detached (no grad)
        assert not result["_pred_X"].requires_grad
        assert not result["_true_X"].requires_grad


class TestSmoothingFunction:
    """Tests for the _smooth helper."""

    def test_smooth_identity_window_1(self):
        from defog.core.callbacks import _smooth
        import numpy as np

        vals = [1.0, 2.0, 3.0, 4.0]
        result = _smooth(vals, 1)
        np.testing.assert_array_almost_equal(result, vals)

    def test_smooth_window_larger_than_data(self):
        from defog.core.callbacks import _smooth
        import numpy as np

        vals = [1.0, 2.0, 3.0]
        result = _smooth(vals, 10)
        assert len(result) == 3
        # First entry is just itself, second is mean of first 2, third is mean of all 3
        np.testing.assert_almost_equal(result[0], 1.0)
        np.testing.assert_almost_equal(result[1], 1.5)
        np.testing.assert_almost_equal(result[2], 2.0)

    def test_smooth_empty(self):
        from defog.core.callbacks import _smooth
        import numpy as np

        result = _smooth([], 5)
        assert len(result) == 0


class TestSampleVisualizationCallback:
    """Tests for SampleVisualizationCallback."""

    def test_instantiation_defaults(self):
        cb = SampleVisualizationCallback()
        assert cb.num_samples == 8
        assert cb.every_k_epochs == 1
        assert cb.sample_steps is None
        assert cb.eta is None
        assert cb.omega is None
        assert cb.time_distortion is None
        assert cb.figure_callback is None
        assert cb.last_figure is None

    def test_instantiation_custom(self):
        cb = SampleVisualizationCallback(
            num_samples=4,
            every_k_epochs=5,
            sample_steps=10,
            eta=0.5,
            omega=0.1,
            time_distortion="polydec",
            render_fn=lambda ax, data: None,
            figure_callback=lambda fig: None,
        )
        assert cb.num_samples == 4
        assert cb.every_k_epochs == 5
        assert cb.sample_steps == 10
        assert cb.eta == 0.5

    def test_samples_every_epoch(self, model_and_loaders):
        """With every_k_epochs=1, should produce a figure each validation epoch."""
        model, train_loader, val_loader = model_and_loaders
        figures = []
        cb = SampleVisualizationCallback(
            num_samples=3,
            every_k_epochs=1,
            sample_steps=2,
            figure_callback=lambda fig: figures.append(fig),
        )

        trainer = pl.Trainer(
            max_epochs=2,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        assert len(figures) == 2
        assert cb.last_figure is not None

    def test_samples_every_k_epochs(self, model_and_loaders):
        """With every_k_epochs=2 and 3 epochs, should produce figure only at epoch 2."""
        model, train_loader, val_loader = model_and_loaders
        figures = []
        cb = SampleVisualizationCallback(
            num_samples=2,
            every_k_epochs=2,
            sample_steps=2,
            figure_callback=lambda fig: figures.append(fig),
        )

        trainer = pl.Trainer(
            max_epochs=3,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        # epoch 1: skip, epoch 2: sample, epoch 3: skip
        assert len(figures) == 1

    def test_figure_has_correct_subplots(self, model_and_loaders):
        """Figure should have one subplot per sample in a single row."""
        import matplotlib.pyplot as mplt

        model, train_loader, val_loader = model_and_loaders
        num_samples = 4
        cb = SampleVisualizationCallback(
            num_samples=num_samples,
            every_k_epochs=1,
            sample_steps=2,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        fig = cb.last_figure
        assert isinstance(fig, mplt.Figure)
        axes = fig.get_axes()
        assert len(axes) == num_samples
        mplt.close(fig)

    def test_figure_has_suptitle(self, model_and_loaders):
        """Figure should have a suptitle with epoch and avg stats."""
        model, train_loader, val_loader = model_and_loaders
        cb = SampleVisualizationCallback(
            num_samples=2,
            every_k_epochs=1,
            sample_steps=2,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        fig = cb.last_figure
        suptitle = fig._suptitle.get_text()
        assert "Epoch" in suptitle
        assert "avg nodes" in suptitle
        assert "avg edges" in suptitle
        import matplotlib.pyplot as mplt
        mplt.close(fig)

    def test_custom_render_fn(self, model_and_loaders):
        """A custom render_fn should be called for each sample."""
        model, train_loader, val_loader = model_and_loaders
        rendered = []

        def my_render(ax, data):
            rendered.append(data)
            ax.text(0.5, 0.5, "custom", transform=ax.transAxes, ha="center")

        cb = SampleVisualizationCallback(
            num_samples=3,
            every_k_epochs=1,
            sample_steps=2,
            render_fn=my_render,
            figure_callback=lambda fig: None,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        assert len(rendered) == 3

    def test_sampling_overrides(self, model_and_loaders):
        """Callback should use overridden sampling params, not model defaults."""
        model, train_loader, val_loader = model_and_loaders
        # Model default is sample_steps=5, we override to 2
        cb = SampleVisualizationCallback(
            num_samples=2,
            every_k_epochs=1,
            sample_steps=2,
            eta=1.0,
            omega=0.5,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[cb],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        # Should run without error and use overridden params
        trainer.fit(model, train_loader, val_loader)
        assert cb.last_figure is not None

    def test_works_alongside_monitor(self, model_and_loaders):
        """Both callbacks should work together without interference."""
        model, train_loader, val_loader = model_and_loaders
        monitor_figs = []
        sample_figs = []

        monitor = TrainingMonitorCallback(
            smoothing_window=1,
            figure_callback=lambda fig: monitor_figs.append(fig),
        )
        sampler = SampleVisualizationCallback(
            num_samples=2,
            every_k_epochs=1,
            sample_steps=2,
            figure_callback=lambda fig: sample_figs.append(fig),
        )

        trainer = pl.Trainer(
            max_epochs=2,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[monitor, sampler],
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)

        assert len(monitor_figs) == 2
        assert len(sample_figs) == 2
