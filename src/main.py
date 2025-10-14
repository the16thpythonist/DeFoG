import os
import pathlib
import warnings
import random
import string

# import graph_tool  # Only needed for non-molecular datasets, imported in their modules
import torch

torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf

# Fix for PyTorch 2.6+: Allow OmegaConf and custom objects in checkpoint loading
# PyTorch 2.6 changed torch.load() default to weights_only=True for security
# Our checkpoints contain OmegaConf config objects and model classes, so we need to allowlist them
# These are trusted checkpoints created by this codebase, so it's safe to add these globals

# Import all required classes for safe_globals
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode
import numpy as np
from collections import defaultdict
from typing import Any

# Add all required classes to safe globals
torch.serialization.add_safe_globals([
    # OmegaConf classes
    DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode,
    # NumPy classes
    np.dtype, type(np.array(0).item()),  # numpy.core.multiarray.scalar
    # Python builtins and typing
    dict, list, int, defaultdict, Any,
    # PyTorch classes
    torch.distributions.categorical.Categorical,
])

# Project-specific classes will be added dynamically after they're imported
# This function will be called before loading checkpoints
def add_project_safe_globals():
    """Add project-specific classes to safe globals for checkpoint loading"""
    import sys
    safe_classes = []

    # Collect all classes from project modules that might be in checkpoints
    for module_name in list(sys.modules.keys()):
        if any(module_name.startswith(prefix) for prefix in ['models.', 'datasets.', 'analysis.', 'metrics.']):
            try:
                module = sys.modules[module_name]
                # Get all classes from the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):  # It's a class
                        safe_classes.append(attr)
            except (AttributeError, ImportError):
                pass

    if safe_classes:
        torch.serialization.add_safe_globals(safe_classes)
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from graph_discrete_flow_model import GraphDiscreteFlowModel
from models.extra_features import DummyExtraFeatures, ExtraFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning)

# Register custom resolver for unique random ID to prevent output directory collisions
OmegaConf.register_new_resolver(
    "random_id",
    lambda: ''.join(random.choices(string.ascii_lowercase + string.digits, k=4)),
    use_cache=True  # Ensures same ID is used throughout config
)


def print_config_summary(cfg: DictConfig):
    """Print a formatted summary of all configuration parameters"""
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY".center(80))
    print("=" * 80)

    # General Settings
    print("\n[GENERAL SETTINGS]")
    print(f"  Experiment name:        {cfg.general.name}")
    print(f"  Wandb mode:             {cfg.general.wandb}")
    print(f"  GPUs:                   {cfg.general.gpus}")
    print(f"  Seed:                   {cfg.train.seed}")
    print(f"  Test only:              {cfg.general.test_only}")
    print(f"  Resume from:            {cfg.general.resume}")

    # Dataset Settings
    print("\n[DATASET SETTINGS]")
    print(f"  Dataset name:           {cfg.dataset.name}")
    if hasattr(cfg.dataset, 'datadir'):
        print(f"  Data directory:         {cfg.dataset.datadir}")
    print(f"  Batch size:             {cfg.train.batch_size}")
    print(f"  Num workers:            {cfg.train.num_workers}")

    # Training Settings
    print("\n[TRAINING SETTINGS]")
    print(f"  Number of epochs:       {cfg.train.n_epochs}")
    print(f"  Learning rate:          {cfg.train.lr}")
    print(f"  Optimizer:              {cfg.train.optimizer}")
    print(f"  Weight decay:           {cfg.train.weight_decay}")
    print(f"  Gradient clipping:      {cfg.train.clip_grad}")
    print(f"  EMA decay:              {cfg.train.ema_decay}")
    print(f"  Time distortion:        {cfg.train.time_distortion}")
    print(f"  Save model:             {cfg.train.save_model}")

    # Model Settings
    print("\n[MODEL SETTINGS]")
    print(f"  Model type:             {cfg.model.model}")
    print(f"  Transition:             {cfg.model.transition}")
    print(f"  Number of layers:       {cfg.model.n_layers}")
    print(f"  Hidden dims:            {cfg.model.hidden_dims}")
    print(f"  Hidden MLP dims:        {cfg.model.hidden_mlp_dims}")
    print(f"  Extra features:         {cfg.model.extra_features}")
    print(f"  RRWP steps:             {cfg.model.rrwp_steps}")
    print(f"  Lambda train [E, y]:    {cfg.model.lambda_train}")

    # Sampling Settings
    print("\n[SAMPLING SETTINGS]")
    print(f"  Eta:                    {cfg.sample.eta}")
    print(f"  Omega:                  {cfg.sample.omega}")
    print(f"  Sample steps:           {cfg.sample.sample_steps}")
    print(f"  Time distortion:        {cfg.sample.time_distortion}")
    print(f"  Search mode:            {cfg.sample.search}")
    print(f"  RDB type:               {cfg.sample.rdb}")

    # Validation Settings
    print("\n[VALIDATION SETTINGS]")
    print(f"  Check val every:        {cfg.general.check_val_every_n_epochs} epochs")
    print(f"  Sample every val:       {cfg.general.sample_every_val}")
    print(f"  Samples to generate:    {cfg.general.samples_to_generate}")
    print(f"  Samples to save:        {cfg.general.samples_to_save}")
    print(f"  Chains to save:         {cfg.general.chains_to_save}")

    # Conditional Generation (if applicable)
    if cfg.general.conditional:
        print("\n[CONDITIONAL GENERATION]")
        print(f"  Conditional:            {cfg.general.conditional}")
        print(f"  Target:                 {cfg.general.target}")
        print(f"  Guidance weight:        {cfg.general.guidance_weight}")

    print("\n" + "=" * 80 + "\n")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)

    # Print configuration summary
    print_config_summary(cfg)

    dataset_config = cfg["dataset"]

    if dataset_config["name"] in [
        "sbm",
        "comm20",
        "planar",
        "tree",
    ]:
        from analysis.visualization import NonMolecularVisualization
        from datasets.spectre_dataset import (
            SpectreGraphDataModule,
            SpectreDatasetInfos,
        )
        from analysis.spectre_utils import (
            PlanarSamplingMetrics,
            SBMSamplingMetrics,
            Comm20SamplingMetrics,
            TreeSamplingMetrics,
        )

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config["name"] == "sbm":
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config["name"] == "comm20":
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config["name"] == "planar":
            sampling_metrics = PlanarSamplingMetrics(datamodule)
        elif dataset_config["name"] == "tree":
            sampling_metrics = TreeSamplingMetrics(datamodule)
        else:
            raise NotImplementedError(
                f"Dataset {dataset_config['name']} not implemented"
            )

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)

        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )


    elif dataset_config["name"] in ["qm9", "guacamol", "moses", "zinc", "aqsoldb"]:
        from metrics.molecular_metrics import (
            TrainMolecularMetrics,
            SamplingMolecularMetrics,
        )
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from models.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if "qm9" in dataset_config["name"]:
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = qm9_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        elif dataset_config["name"] == "guacamol":
            from datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            dataset_smiles = guacamol_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )

        elif dataset_config.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            dataset_smiles = moses_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )
        elif "zinc" in dataset_config["name"]:
            from datasets import zinc_dataset

            datamodule = zinc_dataset.ZINCDataModule(cfg)
            dataset_infos = zinc_dataset.ZINCinfos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = zinc_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        elif dataset_config["name"] == "aqsoldb":
            from datasets import aqsoldb_dataset

            datamodule = aqsoldb_dataset.AqSolDBDataModule(cfg)
            dataset_infos = aqsoldb_dataset.AqSolDBinfos(datamodule, cfg)
            dataset_smiles = aqsoldb_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )
        else:
            raise ValueError("Dataset not implemented")

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

        # We do not evaluate novelty during training
        add_virtual_states = "absorbing" == cfg.model.transition
        sampling_metrics = SamplingMolecularMetrics(
            dataset_infos, dataset_smiles, cfg, add_virtual_states=add_virtual_states
        )
        visualization_tools = MolecularVisualization(
            cfg.dataset.remove_h, dataset_infos=dataset_infos
        )

    elif dataset_config["name"] == "tls":
        from datasets import tls_dataset
        from metrics.tls_metrics import TLSSamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = tls_dataset.TLSDataModule(cfg)
        dataset_infos = tls_dataset.TLSInfos(datamodule=datamodule)

        train_metrics = TrainAbstractMetricsDiscrete()
        extra_features = (
            ExtraFeatures(
                cfg.model.extra_features,
                cfg.model.rrwp_steps,
                dataset_info=dataset_infos,
            )
            if cfg.model.extra_features is not None
            else DummyExtraFeatures()
        )
        domain_features = DummyExtraFeatures()

        sampling_metrics = TLSSamplingMetrics(datamodule)

        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    dataset_infos.compute_reference_metrics(
        datamodule=datamodule,
        sampling_metrics=sampling_metrics,
    )

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "test_labels": (
            datamodule.test_labels
            if ("qm9" in cfg.dataset.name and cfg.general.conditional)
            else None
        ),
    }

    utils.create_folders(cfg)
    model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="{epoch}",
            save_top_k=-1,
            every_n_epochs=cfg.general.sample_every_val
            * cfg.general.check_val_every_n_epochs,
        )
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=name == "debug",
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        logger=[],
    )

    # Add project-specific classes to safe globals before loading any checkpoints
    add_project_safe_globals()

    if not cfg.general.test_only and cfg.general.generated_path is None:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if ".ckpt" in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
