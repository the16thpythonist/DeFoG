"""
Training monitoring callbacks for DeFoG.

Provides PyTorch Lightning callbacks for:
- Training diagnostics (loss, gradients, weight deltas, predictions, hardware)
- Periodic sample visualization during training

Example:
    >>> from defog.core import DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback
    >>> monitor = TrainingMonitorCallback(
    ...     smoothing_window=5,
    ...     figure_callback=lambda fig: e.track("training_progress", fig),
    ... )
    >>> sampler = SampleVisualizationCallback(
    ...     num_samples=8,
    ...     every_k_epochs=10,
    ...     figure_callback=lambda fig: e.track("samples", fig),
    ... )
    >>> trainer = pl.Trainer(callbacks=[monitor, sampler])
    >>> trainer.fit(model, train_loader, val_loader)
"""

import os
import time
import threading
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Module groups on the GraphTransformer (accessed via pl_module.model)
_MODULE_GROUPS = {
    "mlp_in_X": "mlp_in_X",
    "mlp_in_E": "mlp_in_E",
    "mlp_in_y": "mlp_in_y",
    "tf_layers": "tf_layers",
    "mlp_out_X": "mlp_out_X",
    "mlp_out_E": "mlp_out_E",
}


def _grad_norm(params) -> float:
    """Compute the total L2 gradient norm across parameters."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.sqrt(sum(g.norm() ** 2 for g in grads)).item()


def _smooth(values: List[float], window: int) -> np.ndarray:
    """Simple moving average. Returns array of same length (partial windows at start)."""
    if len(values) == 0:
        return np.array([])
    arr = np.array(values, dtype=np.float64)
    if window <= 1 or len(arr) <= 1:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="full")[:len(arr)]
    # Fix the first (window-1) entries to use partial windows
    for i in range(min(window - 1, len(arr))):
        smoothed[i] = arr[: i + 1].mean()
    return smoothed


class _HardwareMonitor:
    """Lightweight daemon thread that samples hardware stats periodically."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._samples: Dict[str, List[float]] = {
            "gpu_util": [],
            "gpu_mem": [],
            "cpu_util": [],
            "ram": [],
        }
        self._has_psutil = False
        self._has_pynvml = False

        try:
            import psutil  # noqa: F401

            self._has_psutil = True
        except ImportError:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            self._has_pynvml = True
        except Exception:
            pass

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            self._sample()
            time.sleep(self.interval)

    def _sample(self):
        with self._lock:
            if self._has_psutil:
                import psutil

                self._samples["cpu_util"].append(psutil.cpu_percent())
                self._samples["ram"].append(psutil.virtual_memory().used / 1e9)

            if self._has_pynvml:
                try:
                    import pynvml

                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self._samples["gpu_util"].append(util.gpu)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self._samples["gpu_mem"].append(mem_info.used / 1e9)
                except Exception:
                    pass
            elif torch.cuda.is_available():
                try:
                    mem_info = torch.cuda.mem_get_info(0)
                    used_gb = (mem_info[1] - mem_info[0]) / 1e9
                    self._samples["gpu_mem"].append(used_gb)
                except Exception:
                    pass

    def get_and_reset(self) -> Dict[str, float]:
        """Return averages since last call and reset buffers."""
        with self._lock:
            result = {}
            for key, vals in self._samples.items():
                result[key] = float(np.mean(vals)) if vals else float("nan")
            self._samples = {k: [] for k in self._samples}
        return result

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


class TrainingMonitorCallback(pl.Callback):
    """
    Collects training diagnostics and generates a 5x4 monitoring figure
    at every validation epoch.

    The figure tracks:
      Row 1 - Loss: train/val (linear & log), ratio, epoch time
      Row 2 - Gradients: total and per module group
      Row 3 - Weights: total and per module group parameter deltas
      Row 4 - Predictions: entropy and per-class accuracy for nodes/edges
      Row 5 - Hardware: GPU util/mem, CPU util, RAM

    Args:
        smoothing_window: Window size for simple moving average on plots.
        hw_sample_interval: Seconds between hardware stat samples.
        figure_callback: Optional callable(fig) invoked with the matplotlib
            figure each validation epoch (e.g. for ``e.track()``).
        generation_metrics_fn: Optional callable(list_of_Data) -> dict with
            keys "validity"/"uniqueness"/"novelty". When provided, the model is
            sampled unconditionally every ``gen_every_k`` epochs and these
            metrics are computed and added as an extra row of panels to the grid.
        gen_every_k: Sample-and-evaluate the generation metrics every k epochs.
        gen_num_samples: Number of graphs to sample for the metrics.
        gen_sample_steps: Denoising steps for the metric sampling (defaults to
            the model's configured ``sample_steps``).
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        hw_sample_interval: float = 2.0,
        figure_callback: Optional[Callable] = None,
        generation_metrics_fn: Optional[Callable] = None,
        gen_every_k: int = 10,
        gen_num_samples: int = 64,
        gen_sample_steps: Optional[int] = None,
        gen_eta: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        super().__init__()
        self.smoothing_window = smoothing_window
        self.hw_sample_interval = hw_sample_interval
        self.figure_callback = figure_callback
        self.generation_metrics_fn = generation_metrics_fn
        self.gen_every_k = gen_every_k
        self.gen_num_samples = gen_num_samples
        self.gen_sample_steps = gen_sample_steps
        self.gen_eta = gen_eta
        # When set (together with generation_metrics_fn), the model is saved to
        # <checkpoint_dir>/best_model whenever validity reaches a new maximum.
        self.checkpoint_dir = checkpoint_dir
        self.best_validity = -1.0

        # Per-epoch history (appended at each validation epoch end)
        self.history: Dict[str, List] = defaultdict(list)

        # Per-epoch accumulators (reset each epoch)
        self._epoch_accum: Dict[str, List[float]] = defaultdict(list)
        self._epoch_pred_accum: Dict[str, List] = defaultdict(list)

        # State
        self._prev_params: Dict[str, torch.Tensor] = {}
        self._hw_monitor: Optional[_HardwareMonitor] = None
        self._epoch_start_time: float = 0.0
        self.last_figure: Optional[plt.Figure] = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Snapshot parameters for delta computation
        self._prev_params = {
            name: p.detach().clone()
            for name, p in pl_module.model.named_parameters()
        }
        # Start hardware monitor
        self._hw_monitor = _HardwareMonitor(interval=self.hw_sample_interval)
        self._hw_monitor.start()
        self._epoch_start_time = time.time()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._hw_monitor is not None:
            self._hw_monitor.stop()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._epoch_accum = defaultdict(list)
        self._epoch_pred_accum = defaultdict(list)
        self._epoch_start_time = time.time()

    # ------------------------------------------------------------------
    # Per-batch collection
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        # outputs may be wrapped by Lightning; unwrap if needed
        if isinstance(outputs, list):
            outputs = outputs[0]
        if not isinstance(outputs, dict):
            return

        # --- Gradient norms ---
        model = pl_module.model
        total_grads = []
        for group_key, attr_name in _MODULE_GROUPS.items():
            module = getattr(model, attr_name, None)
            if module is None:
                continue
            gn = _grad_norm(module.parameters())
            self._epoch_accum[f"grad_norm_{group_key}"].append(gn)
            total_grads.append(gn)

        if total_grads:
            self._epoch_accum["grad_norm_total"].append(
                float(np.sqrt(sum(g ** 2 for g in total_grads)))
            )

        # --- Prediction entropy & accuracy ---
        pred_X = outputs.get("_pred_X")
        pred_E = outputs.get("_pred_E")
        true_X = outputs.get("_true_X")
        true_E = outputs.get("_true_E")
        node_mask = outputs.get("_node_mask")

        if pred_X is not None and true_X is not None and node_mask is not None:
            self._collect_prediction_stats(pred_X, pred_E, true_X, true_E, node_mask)

    def _collect_prediction_stats(
        self,
        pred_X: torch.Tensor,
        pred_E: torch.Tensor,
        true_X: torch.Tensor,
        true_E: torch.Tensor,
        node_mask: torch.Tensor,
    ):
        """Compute per-batch prediction entropy and per-class accuracy."""
        # Node entropy
        probs_X = F.softmax(pred_X, dim=-1)
        ent_X = -(probs_X * torch.log(probs_X + 1e-8)).sum(dim=-1)
        if node_mask.any():
            self._epoch_accum["entropy_X"].append(ent_X[node_mask].mean().item())

        # Edge entropy (upper triangle only, masked)
        if pred_E is not None and true_E is not None:
            bs, n = node_mask.shape
            edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
            # Upper triangle
            triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=pred_E.device), diagonal=1)
            edge_mask = edge_mask & triu.unsqueeze(0)

            probs_E = F.softmax(pred_E, dim=-1)
            ent_E = -(probs_E * torch.log(probs_E + 1e-8)).sum(dim=-1)
            if edge_mask.any():
                self._epoch_accum["entropy_E"].append(ent_E[edge_mask].mean().item())

        # Per-class accuracy (nodes)
        pred_classes_X = pred_X.argmax(dim=-1)
        true_classes_X = true_X.argmax(dim=-1)
        num_node_classes = pred_X.shape[-1]
        acc_X = {}
        for c in range(num_node_classes):
            mask_c = (true_classes_X == c) & node_mask
            if mask_c.any():
                acc_X[c] = (pred_classes_X[mask_c] == c).float().mean().item()
        if acc_X:
            self._epoch_pred_accum["acc_X_per_class"].append(acc_X)

        # Per-class accuracy (edges)
        if pred_E is not None and true_E is not None:
            bs, n = node_mask.shape
            edge_mask_full = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
            triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=pred_E.device), diagonal=1)
            edge_mask_full = edge_mask_full & triu.unsqueeze(0)

            pred_classes_E = pred_E.argmax(dim=-1)
            true_classes_E = true_E.argmax(dim=-1)
            num_edge_classes = pred_E.shape[-1]
            acc_E = {}
            for c in range(num_edge_classes):
                mask_c = (true_classes_E == c) & edge_mask_full
                if mask_c.any():
                    acc_E[c] = (pred_classes_E[mask_c] == c).float().mean().item()
            if acc_E:
                self._epoch_pred_accum["acc_E_per_class"].append(acc_E)

    # ------------------------------------------------------------------
    # Epoch-level aggregation
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch_time = time.time() - self._epoch_start_time
        self.history["epoch_time"].append(epoch_time)

        # --- Loss ---
        # Retrieve logged scalars from trainer's callback_metrics
        metrics = trainer.callback_metrics
        if "train/loss" in metrics:
            self.history["train_loss"].append(metrics["train/loss"].item())
        elif "train/loss_epoch" in metrics:
            self.history["train_loss"].append(metrics["train/loss_epoch"].item())

        if "val/loss" in metrics:
            self.history["val_loss"].append(metrics["val/loss"].item())

        # --- Gradient norms (epoch averages) ---
        for key in ["grad_norm_total"] + [f"grad_norm_{g}" for g in _MODULE_GROUPS]:
            vals = self._epoch_accum.get(key, [])
            self.history[key].append(float(np.mean(vals)) if vals else float("nan"))

        # --- Parameter deltas ---
        model = pl_module.model
        total_delta_sq = 0.0
        for group_key, attr_name in _MODULE_GROUPS.items():
            module = getattr(model, attr_name, None)
            if module is None:
                self.history[f"param_delta_{group_key}"].append(float("nan"))
                continue
            delta_sq = 0.0
            for name, p in module.named_parameters():
                full_name = f"{attr_name}.{name}"
                if full_name in self._prev_params:
                    delta_sq += (p.detach() - self._prev_params[full_name]).norm().item() ** 2
            delta = float(np.sqrt(delta_sq))
            self.history[f"param_delta_{group_key}"].append(delta)
            total_delta_sq += delta_sq

        self.history["param_delta_total"].append(float(np.sqrt(total_delta_sq)))

        # Update parameter snapshot
        self._prev_params = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
        }

        # --- Prediction stats (epoch averages) ---
        for key in ["entropy_X", "entropy_E"]:
            vals = self._epoch_accum.get(key, [])
            self.history[key].append(float(np.mean(vals)) if vals else float("nan"))

        # Per-class accuracy: average across batches
        for key in ["acc_X_per_class", "acc_E_per_class"]:
            batch_dicts = self._epoch_pred_accum.get(key, [])
            if batch_dicts:
                merged: Dict[int, List[float]] = defaultdict(list)
                for d in batch_dicts:
                    for c, v in d.items():
                        merged[c].append(v)
                avg = {c: float(np.mean(vs)) for c, vs in merged.items()}
                self.history[key].append(avg)
            else:
                self.history[key].append({})

        # --- Hardware ---
        if self._hw_monitor is not None:
            hw = self._hw_monitor.get_and_reset()
            for key in ["gpu_util", "gpu_mem", "cpu_util", "ram"]:
                self.history[key].append(hw.get(key, float("nan")))
        else:
            for key in ["gpu_util", "gpu_mem", "cpu_util", "ram"]:
                self.history[key].append(float("nan"))

        # --- Generation metrics (sample the model every k epochs) ---
        if self.generation_metrics_fn is not None and not trainer.sanity_checking:
            current_idx = len(self.history["epoch_time"])  # 1-based epoch index
            is_last = trainer.current_epoch >= trainer.max_epochs - 1
            if current_idx % self.gen_every_k == 0 or is_last:
                self._collect_generation_metrics(pl_module, current_idx)

        # --- Generate figure ---
        fig = self._generate_figure()
        self.last_figure = fig

        if self.figure_callback is not None:
            self.figure_callback(fig)
        else:
            plt.close(fig)

    def _collect_generation_metrics(self, pl_module: pl.LightningModule, epoch_idx: int):
        """Sample the model unconditionally and record validity/uniqueness/novelty."""
        was_training = pl_module.training
        try:
            pl_module.eval()
            steps = self.gen_sample_steps or getattr(pl_module, "sample_steps", 100)
            samples = pl_module.sample(
                num_samples=self.gen_num_samples,
                sample_steps=steps,
                eta=self.gen_eta,   # None -> model default; low eta keeps the probe unsaturated
                show_progress=False,
            )
            metrics = self.generation_metrics_fn(samples)
        except Exception as exc:  # never let metric sampling break training
            warnings.warn(f"generation metrics sampling failed: {exc}")
            metrics = {}
        finally:
            if was_training:
                pl_module.train()

        self.history["gen_epochs"].append(epoch_idx)
        for key in ("validity", "uniqueness", "novelty"):
            self.history[f"gen_{key}"].append(float(metrics.get(key, float("nan"))))

        # Save the model whenever validity reaches a new best (captures the peak
        # even if late-training generation quality oscillates).
        validity = float(metrics.get("validity", float("nan")))
        if (self.checkpoint_dir is not None and validity == validity  # not NaN
                and validity > self.best_validity):
            self.best_validity = validity
            try:
                pl_module.save(os.path.join(self.checkpoint_dir, "best_model"))
                warnings.warn(
                    f"[epoch {epoch_idx}] new best validity {validity:.3f} -> saved best_model"
                )
            except Exception as exc:
                warnings.warn(f"best-checkpoint save failed: {exc}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _generate_figure(self) -> plt.Figure:
        n_rows = 6 if self.generation_metrics_fn is not None else 5
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
        sw = self.smoothing_window
        epochs = list(range(1, len(self.history["epoch_time"]) + 1))

        # ===== Row 1: Loss =====
        self._plot_loss_linear(axes[0, 0], epochs, sw)
        self._plot_loss_log(axes[0, 1], epochs, sw)
        self._plot_loss_ratio(axes[0, 2], epochs, sw)
        self._plot_epoch_time(axes[0, 3], epochs, sw)

        # ===== Row 2: Gradients =====
        self._plot_single_metric(axes[1, 0], epochs, "grad_norm_total", "Total Gradient Norm", sw)
        self._plot_multi_metric(
            axes[1, 1], epochs,
            ["grad_norm_mlp_in_X", "grad_norm_mlp_in_E", "grad_norm_mlp_in_y"],
            ["mlp_in_X", "mlp_in_E", "mlp_in_y"],
            "Input MLPs Grad Norm", sw,
        )
        self._plot_multi_metric(
            axes[1, 2], epochs,
            ["grad_norm_mlp_out_X", "grad_norm_mlp_out_E"],
            ["mlp_out_X", "mlp_out_E"],
            "Output MLPs Grad Norm", sw,
        )
        self._plot_single_metric(axes[1, 3], epochs, "grad_norm_tf_layers", "TF Layers Grad Norm", sw)

        # ===== Row 3: Weight deltas =====
        self._plot_single_metric(axes[2, 0], epochs, "param_delta_total", "Total Param Delta", sw)
        self._plot_multi_metric(
            axes[2, 1], epochs,
            ["param_delta_mlp_in_X", "param_delta_mlp_in_E", "param_delta_mlp_in_y"],
            ["mlp_in_X", "mlp_in_E", "mlp_in_y"],
            "Input MLPs Param Delta", sw,
        )
        self._plot_multi_metric(
            axes[2, 2], epochs,
            ["param_delta_mlp_out_X", "param_delta_mlp_out_E"],
            ["mlp_out_X", "mlp_out_E"],
            "Output MLPs Param Delta", sw,
        )
        self._plot_single_metric(axes[2, 3], epochs, "param_delta_tf_layers", "TF Layers Param Delta", sw)

        # ===== Row 4: Predictions =====
        self._plot_single_metric(axes[3, 0], epochs, "entropy_X", "Node Prediction Entropy", sw)
        self._plot_single_metric(axes[3, 1], epochs, "entropy_E", "Edge Prediction Entropy", sw)
        self._plot_per_class_accuracy(axes[3, 2], epochs, "acc_X_per_class", "Node Accuracy (per class)", sw)
        self._plot_per_class_accuracy(axes[3, 3], epochs, "acc_E_per_class", "Edge Accuracy (per class)", sw)

        # ===== Row 5: Hardware =====
        self._plot_single_metric(axes[4, 0], epochs, "gpu_util", "GPU Utilization (%)", sw, ylabel="%")
        self._plot_single_metric(axes[4, 1], epochs, "gpu_mem", "GPU Memory (GB)", sw, ylabel="GB")
        self._plot_single_metric(axes[4, 2], epochs, "cpu_util", "CPU Utilization (%)", sw, ylabel="%")
        self._plot_single_metric(axes[4, 3], epochs, "ram", "RAM Usage (GB)", sw, ylabel="GB")

        # ===== Row 6: Generation metrics (unconditional samples) =====
        if self.generation_metrics_fn is not None:
            self._plot_gen_metric(axes[5, 0], ["gen_validity"], ["validity"], "Validity")
            self._plot_gen_metric(axes[5, 1], ["gen_uniqueness"], ["uniqueness"], "Uniqueness (of valid)")
            self._plot_gen_metric(axes[5, 2], ["gen_novelty"], ["novelty"], "Novelty (of unique)")
            self._plot_gen_metric(
                axes[5, 3],
                ["gen_validity", "gen_uniqueness", "gen_novelty"],
                ["validity", "uniqueness", "novelty"],
                "Generation Metrics",
            )

        fig.tight_layout()
        return fig

    def _plot_gen_metric(self, ax, keys, labels, title):
        """Plot generation metrics against the epochs at which they were sampled."""
        ax.set_title(title)
        gen_epochs = self.history.get("gen_epochs", [])
        plotted = False
        for key, label in zip(keys, labels):
            vals = self.history.get(key, [])
            n = min(len(gen_epochs), len(vals))
            if n > 0:
                ax.plot(gen_epochs[:n], vals[:n], marker="o", markersize=3, label=label)
                plotted = True
        ax.set_xlabel("Epoch")
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.02, 1.02)
        if plotted and len(labels) > 1:
            ax.legend(fontsize="small")

    # ------ individual subplot helpers ------

    def _plot_loss_linear(self, ax: plt.Axes, epochs: List[int], sw: int):
        ax.set_title("Train + Val Loss")
        train = self.history.get("train_loss", [])
        val = self.history.get("val_loss", [])
        if train:
            e = epochs[: len(train)]
            ax.plot(e, train, alpha=0.3, color="C0")
            ax.plot(e, _smooth(train, sw), color="C0", label="train")
        if val:
            e = epochs[: len(val)]
            ax.plot(e, val, alpha=0.3, color="C1")
            ax.plot(e, _smooth(val, sw), color="C1", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize="small")

    def _plot_loss_log(self, ax: plt.Axes, epochs: List[int], sw: int):
        ax.set_title("Train + Val Loss (log)")
        train = self.history.get("train_loss", [])
        val = self.history.get("val_loss", [])
        if train:
            e = epochs[: len(train)]
            ax.plot(e, train, alpha=0.3, color="C0")
            ax.plot(e, _smooth(train, sw), color="C0", label="train")
        if val:
            e = epochs[: len(val)]
            ax.plot(e, val, alpha=0.3, color="C1")
            ax.plot(e, _smooth(val, sw), color="C1", label="val")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log)")
        ax.legend(fontsize="small")

    def _plot_loss_ratio(self, ax: plt.Axes, epochs: List[int], sw: int):
        ax.set_title("Train / Val Loss Ratio")
        train = self.history.get("train_loss", [])
        val = self.history.get("val_loss", [])
        n = min(len(train), len(val))
        if n > 0:
            ratio = [t / v if v > 0 else float("nan") for t, v in zip(train[:n], val[:n])]
            e = epochs[:n]
            ax.plot(e, ratio, alpha=0.3, color="C2")
            ax.plot(e, _smooth(ratio, sw), color="C2")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Ratio")

    def _plot_epoch_time(self, ax: plt.Axes, epochs: List[int], sw: int):
        ax.set_title("Epoch Wall-Clock Time")
        vals = self.history.get("epoch_time", [])
        if vals:
            e = epochs[: len(vals)]
            ax.plot(e, vals, alpha=0.3, color="C3")
            ax.plot(e, _smooth(vals, sw), color="C3")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Seconds")

    def _plot_single_metric(
        self,
        ax: plt.Axes,
        epochs: List[int],
        key: str,
        title: str,
        sw: int,
        ylabel: str = "",
        color: str = "C0",
    ):
        ax.set_title(title)
        vals = self.history.get(key, [])
        # Filter out nan for checking emptiness
        valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if valid:
            e = epochs[: len(vals)]
            ax.plot(e, vals, alpha=0.3, color=color)
            ax.plot(e, _smooth(vals, sw), color=color)
        else:
            ax.text(
                0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="gray",
            )
        ax.set_xlabel("Epoch")
        if ylabel:
            ax.set_ylabel(ylabel)

    def _plot_multi_metric(
        self,
        ax: plt.Axes,
        epochs: List[int],
        keys: List[str],
        labels: List[str],
        title: str,
        sw: int,
    ):
        ax.set_title(title)
        has_data = False
        for i, (key, label) in enumerate(zip(keys, labels)):
            vals = self.history.get(key, [])
            valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if valid:
                has_data = True
                e = epochs[: len(vals)]
                color = f"C{i}"
                ax.plot(e, vals, alpha=0.3, color=color)
                ax.plot(e, _smooth(vals, sw), color=color, label=label)
        if has_data:
            ax.legend(fontsize="small")
        else:
            ax.text(
                0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="gray",
            )
        ax.set_xlabel("Epoch")

    def _plot_per_class_accuracy(
        self,
        ax: plt.Axes,
        epochs: List[int],
        key: str,
        title: str,
        sw: int,
    ):
        ax.set_title(title)
        dicts_list = self.history.get(key, [])
        if not dicts_list or all(not d for d in dicts_list):
            ax.text(
                0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="gray",
            )
            ax.set_xlabel("Epoch")
            return

        # Collect all class indices that appear
        all_classes = sorted({c for d in dicts_list for c in d})
        for i, c in enumerate(all_classes):
            vals = [d.get(c, float("nan")) for d in dicts_list]
            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                e = epochs[: len(vals)]
                color = f"C{i % 10}"
                ax.plot(e, vals, alpha=0.3, color=color)
                ax.plot(e, _smooth(vals, sw), color=color, label=f"class {c}")

        ax.legend(fontsize="small", ncol=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)


# ======================================================================
# Default graph renderer (networkx spring layout)
# ======================================================================

def _default_render_graph(ax: plt.Axes, data) -> None:
    """
    Draw a PyG Data object on the given axes using networkx spring layout.

    Node classes are shown as distinct colors, edges are drawn in gray.
    """
    try:
        import networkx as nx
    except ImportError:
        ax.text(
            0.5, 0.5, "networkx\nnot installed",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    n = data.x.shape[0] if data.x is not None else 0
    if n == 0:
        ax.text(
            0.5, 0.5, "empty", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    G = nx.Graph()
    G.add_nodes_from(range(n))

    if data.edge_index is not None and data.edge_index.numel() > 0:
        ei = data.edge_index
        for i in range(ei.shape[1]):
            u, v = ei[0, i].item(), ei[1, i].item()
            if u < v:
                G.add_edge(u, v)

    # Node colors from class index
    cmap = plt.cm.Set3
    if data.x.dim() == 2 and data.x.shape[1] > 1:
        node_classes = data.x.argmax(dim=-1).tolist()
    else:
        node_classes = [0] * n
    num_classes = max(node_classes) + 1
    colors = [cmap(c / max(num_classes, 1)) for c in node_classes]

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(
        G, pos, ax=ax,
        node_color=colors,
        node_size=80,
        width=0.8,
        edge_color="#888888",
        with_labels=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])


# ======================================================================
# SampleVisualizationCallback
# ======================================================================

class SampleVisualizationCallback(pl.Callback):
    """
    Periodically samples graphs from the model during training and
    produces a visualization figure.

    At every *k*-th validation epoch the callback calls ``model.sample()``
    to generate a small batch of graphs, renders each one into a subplot
    (single row), and optionally forwards the figure via *figure_callback*
    (e.g. for ``e.track()``).

    Args:
        num_samples: Number of graphs to generate each time.
        every_k_epochs: Run sampling every k validation epochs (1 = every epoch).
        sample_steps: Override model's default number of denoising steps.
            Use fewer steps for faster preview samples.
        eta: Override stochasticity parameter.
        omega: Override target guidance strength.
        time_distortion: Override time distortion type.
        render_fn: Callable ``(ax, data) -> None`` that draws a single PyG
            Data object onto a matplotlib Axes. If *None*, uses a default
            networkx spring-layout renderer with node-class coloring.
        figure_callback: Optional ``callable(fig)`` invoked with the figure.
        show_progress: Show tqdm progress bar during sampling.

    Example:
        >>> sampler = SampleVisualizationCallback(
        ...     num_samples=8,
        ...     every_k_epochs=10,
        ...     sample_steps=20,  # fewer steps for quick previews
        ...     figure_callback=lambda fig: e.track("samples", fig),
        ... )
        >>> trainer = pl.Trainer(callbacks=[sampler])
    """

    def __init__(
        self,
        num_samples: int = 8,
        every_k_epochs: int = 1,
        sample_steps: Optional[int] = None,
        eta: Optional[float] = None,
        omega: Optional[float] = None,
        time_distortion: Optional[str] = None,
        render_fn: Optional[Callable] = None,
        figure_callback: Optional[Callable] = None,
        show_progress: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.every_k_epochs = max(1, every_k_epochs)
        self.sample_steps = sample_steps
        self.eta = eta
        self.omega = omega
        self.time_distortion = time_distortion
        self.render_fn = render_fn if render_fn is not None else _default_render_graph
        self.figure_callback = figure_callback
        self.show_progress = show_progress

        self._val_epoch_count = 0
        self.last_figure: Optional[plt.Figure] = None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self._val_epoch_count += 1
        if self._val_epoch_count % self.every_k_epochs != 0:
            return

        # Build sampling kwargs (only pass overrides that are set)
        sample_kwargs: Dict[str, Any] = {
            "num_samples": self.num_samples,
            "show_progress": self.show_progress,
        }
        if self.sample_steps is not None:
            sample_kwargs["sample_steps"] = self.sample_steps
        if self.eta is not None:
            sample_kwargs["eta"] = self.eta
        if self.omega is not None:
            sample_kwargs["omega"] = self.omega
        if self.time_distortion is not None:
            sample_kwargs["time_distortion"] = self.time_distortion

        # Sample
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            samples = pl_module.sample(**sample_kwargs)
        if was_training:
            pl_module.train()

        # Compute summary stats
        total_nodes = 0
        total_edges = 0
        for s in samples:
            total_nodes += s.x.shape[0] if s.x is not None else 0
            total_edges += s.edge_index.shape[1] // 2 if (s.edge_index is not None and s.edge_index.numel() > 0) else 0
        avg_nodes = total_nodes / len(samples) if samples else 0
        avg_edges = total_edges / len(samples) if samples else 0

        # Build figure: single row
        n = len(samples)
        fig_width = max(3 * n, 6)
        fig, axes = plt.subplots(1, n, figsize=(fig_width, 3.5))
        if n == 1:
            axes = [axes]

        for ax, data in zip(axes, samples):
            self.render_fn(ax, data)

        current_epoch = trainer.current_epoch
        fig.suptitle(
            f"Epoch {current_epoch}  |  "
            f"avg nodes: {avg_nodes:.1f}  |  avg edges: {avg_edges:.1f}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        self.last_figure = fig

        if self.figure_callback is not None:
            self.figure_callback(fig)
        else:
            plt.close(fig)


class EMACallback(pl.Callback):
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of the trainable parameters, updated after every
    optimizer step as ``shadow = decay * shadow + (1 - decay) * param``. During
    validation the EMA weights are swapped into the model (and restored after),
    so that the validation loss, the in-training generation-metric sampling, and
    any best-checkpoint saved during validation all reflect the smoothed EMA
    weights. At fit end the EMA weights are baked into the model so the final
    saved model is the EMA model.

    You MUST evaluate/sample from the EMA weights for EMA to help -- this
    callback ensures that by swapping during validation and at fit end.
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.shadow:  # don't clobber a resumed shadow
            self.shadow = {
                n: p.detach().clone()
                for n, p in pl_module.named_parameters()
                if p.requires_grad
            }

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for n, p in pl_module.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module):
        if not self.shadow:
            return
        self._backup = {}
        for n, p in pl_module.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        for n, p in pl_module.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}

    @torch.no_grad()
    def on_fit_end(self, trainer, pl_module):
        # Bake EMA weights into the model so the final saved model is the EMA one.
        for n, p in pl_module.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
