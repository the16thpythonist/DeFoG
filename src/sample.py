#!/usr/bin/env python
"""
Standalone sampling script for conditional and unconditional graph generation.

This script allows generating molecular graphs with user-specified conditions,
validating conditions against the dataset distribution, and producing
distribution plots comparing generated samples to the training data.

Usage:
    python sample.py --checkpoint path/to/model.ckpt --condition 2.5 -0.25 --num-samples 100
"""

import os
import sys
import json
import pickle
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

import rich_click as click
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Add src to path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from omegaconf import DictConfig, OmegaConf

# Configure rich_click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"

console = Console()

# Property information for supported datasets
PROPERTY_INFO = {
    "mu": {
        "name": "Dipole Moment",
        "unit": "Debye",
        "index": 3,
    },
    "homo": {
        "name": "HOMO Energy",
        "unit": "Hartree",
        "index": 5,
    },
}

# Mapping from target config to property keys
TARGET_TO_PROPERTIES = {
    "mu": ["mu"],
    "homo": ["homo"],
    "both": ["mu", "homo"],
}


def get_device(device_override: Optional[str] = None) -> torch.device:
    """Auto-detect or use specified device."""
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Any, DictConfig]:
    """
    Load model checkpoint and return model with config.

    Returns:
        Tuple of (model, config)
    """
    from graph_discrete_flow_model import GraphDiscreteFlowModel

    console.print(f"Loading checkpoint from [cyan]{checkpoint_path}[/cyan]...")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["hyper_parameters"]["cfg"]

    return cfg, checkpoint


def setup_model_and_data(
    cfg: DictConfig,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[Any, Any, Any, Any]:
    """
    Set up model, datamodule, and related components.

    Returns:
        Tuple of (model, datamodule, dataset_infos, property_values)
    """
    import pytorch_lightning as pl
    from models.extra_features import DummyExtraFeatures, ExtraFeatures
    from metrics.abstract_metrics import TrainAbstractMetricsDiscrete

    dataset_name = cfg.dataset.name

    # Import dataset-specific modules
    if "qm9" in dataset_name:
        from datasets import qm9_dataset
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from models.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        datamodule = qm9_dataset.QM9DataModule(cfg)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

    elif dataset_name == "guacamol":
        from datasets import guacamol_dataset
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from models.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        datamodule = guacamol_dataset.GuacamolDataModule(cfg)
        dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)

    elif dataset_name == "moses":
        from datasets import moses_dataset
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from models.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        datamodule = moses_dataset.MosesDataModule(cfg)
        dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)

    elif "zinc" in dataset_name:
        from datasets import zinc_dataset
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from models.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        datamodule = zinc_dataset.ZINCDataModule(cfg)
        dataset_infos = zinc_dataset.ZINCinfos(datamodule=datamodule, cfg=cfg)

    elif dataset_name == "aqsoldb":
        from datasets import aqsoldb_dataset
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from models.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        datamodule = aqsoldb_dataset.AqSolDBDataModule(cfg)
        dataset_infos = aqsoldb_dataset.AqSolDBinfos(datamodule, cfg)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Only molecular datasets are supported.")

    # Set up extra features
    extra_features = ExtraFeatures(
        cfg.model.extra_features,
        cfg.model.rrwp_steps,
        dataset_info=dataset_infos,
    )

    from models.extra_features_molecular import ExtraMolecularFeatures
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    # Set up metrics (minimal, just for model initialization)
    from metrics.molecular_metrics import SamplingMolecularMetrics
    from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    from analysis.visualization import MolecularVisualization

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    # Get smiles for metrics
    if "qm9" in dataset_name:
        dataset_smiles = qm9_dataset.get_smiles(
            cfg=cfg,
            datamodule=datamodule,
            dataset_infos=dataset_infos,
            evaluate_datasets=False,
        )
    elif dataset_name == "guacamol":
        dataset_smiles = guacamol_dataset.get_smiles(
            raw_dir=datamodule.train_dataset.raw_dir,
            filter_dataset=cfg.dataset.filter,
        )
    elif dataset_name == "moses":
        dataset_smiles = moses_dataset.get_smiles(
            raw_dir=datamodule.train_dataset.raw_dir,
            filter_dataset=cfg.dataset.filter,
        )
    elif "zinc" in dataset_name:
        dataset_smiles = zinc_dataset.get_smiles(
            cfg=cfg,
            datamodule=datamodule,
            dataset_infos=dataset_infos,
            evaluate_datasets=False,
        )
    elif dataset_name == "aqsoldb":
        dataset_smiles = aqsoldb_dataset.get_smiles(
            raw_dir=datamodule.train_dataset.raw_dir,
            filter_dataset=cfg.dataset.filter,
        )

    add_virtual_states = "absorbing" == cfg.model.transition
    sampling_metrics = SamplingMolecularMetrics(
        dataset_infos, dataset_smiles, cfg, add_virtual_states=add_virtual_states
    )
    visualization_tools = MolecularVisualization(
        cfg.dataset.remove_h, dataset_infos=dataset_infos
    )

    # Create model
    from graph_discrete_flow_model import GraphDiscreteFlowModel

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "test_labels": None,
    }

    model = GraphDiscreteFlowModel.load_from_checkpoint(
        checkpoint_path,
        **model_kwargs,
    )
    model.eval()
    model.to(device)

    # Extract property values from training data for distribution
    property_values = extract_property_values(datamodule, cfg)

    return model, datamodule, dataset_infos, property_values


def extract_property_values(datamodule, cfg) -> Dict[str, np.ndarray]:
    """
    Extract property values from the training dataset.

    Returns:
        Dictionary mapping property names to arrays of values
    """
    property_values = {}

    target = getattr(cfg.general, "target", None)
    if not target or not cfg.general.conditional:
        return property_values

    # Get raw property values from train dataset
    train_dataset = datamodule.train_dataloader().dataset

    # For QM9-like datasets, extract from the raw data before transform
    if hasattr(train_dataset, 'data') and hasattr(train_dataset.data, 'y'):
        # We need to get the original y values, not the transformed ones
        # Load raw data again without transform
        if "qm9" in cfg.dataset.name:
            import pathlib
            from torch_geometric.data import InMemoryDataset
            from datasets.qm9_dataset import QM9Dataset

            base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
            root_path = os.path.join(base_path, cfg.dataset.datadir)

            # Load without transform to get raw properties
            raw_dataset = QM9Dataset(
                stage="train",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                target_prop=None,
                transform=None,  # No transform
            )

            all_y = raw_dataset.data.y.numpy()

            properties = TARGET_TO_PROPERTIES.get(target, [])
            for prop in properties:
                if prop in PROPERTY_INFO:
                    idx = PROPERTY_INFO[prop]["index"]
                    property_values[prop] = all_y[:, idx]

    return property_values


def validate_conditions(
    conditions: List[float],
    cfg: DictConfig,
    property_values: Dict[str, np.ndarray],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate conditions against dataset distribution.

    Returns:
        Tuple of (is_valid, condition_info_list)
    """
    target = getattr(cfg.general, "target", None)
    is_conditional = cfg.general.conditional

    if not is_conditional:
        if conditions:
            return False, [{"error": "Model is unconditional but conditions were provided"}]
        return True, []

    properties = TARGET_TO_PROPERTIES.get(target, [])

    if len(conditions) != len(properties):
        return False, [{
            "error": f"Expected {len(properties)} condition(s) for target '{target}', got {len(conditions)}"
        }]

    condition_info = []
    for i, (prop, value) in enumerate(zip(properties, conditions)):
        info = {"property": prop, "value": value}

        if prop in PROPERTY_INFO:
            info["name"] = PROPERTY_INFO[prop]["name"]
            info["unit"] = PROPERTY_INFO[prop]["unit"]
        else:
            info["name"] = f"Property {i+1}"
            info["unit"] = ""

        if prop in property_values:
            values = property_values[prop]
            info["min"] = float(np.min(values))
            info["max"] = float(np.max(values))
            info["mean"] = float(np.mean(values))
            info["std"] = float(np.std(values))
            info["percentile"] = float(stats.percentileofscore(values, value))

            # Warning if outside range
            if value < info["min"] or value > info["max"]:
                info["warning"] = "Value is outside dataset range!"

        condition_info.append(info)

    return True, condition_info


def print_condition_info(condition_info: List[Dict[str, Any]]) -> None:
    """Print formatted condition information."""
    if not condition_info:
        console.print("[yellow]Running in unconditional mode[/yellow]")
        return

    table = Table(title="Condition Validation", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Target Value", style="green")
    table.add_column("Dataset Range", style="blue")
    table.add_column("Percentile", style="magenta")

    for info in condition_info:
        if "error" in info:
            console.print(f"[red]Error: {info['error']}[/red]")
            continue

        name = info.get("name", "Unknown")
        unit = info.get("unit", "")
        value = info["value"]

        value_str = f"{value:.4f} {unit}".strip()

        if "min" in info:
            range_str = f"[{info['min']:.4f}, {info['max']:.4f}]"
            percentile_str = f"{info['percentile']:.1f}%"
        else:
            range_str = "N/A"
            percentile_str = "N/A"

        table.add_row(name, value_str, range_str, percentile_str)

        if "warning" in info:
            console.print(f"  [yellow]⚠ {info['warning']}[/yellow]")

    console.print(table)


@torch.no_grad()
def sample_molecules(
    model,
    num_samples: int,
    conditions: Optional[torch.Tensor],
    device: torch.device,
    eta: float,
    omega: float,
    steps: int,
    time_distortion: str,
    guidance_weight: float,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate molecular samples.

    Returns:
        List of (node_types, edge_types) tuples
    """
    from flow_matching import flow_matching_utils
    import utils

    model.eval()

    # Override sampling parameters
    original_eta = model.cfg.sample.eta
    original_omega = model.cfg.sample.omega
    original_steps = model.cfg.sample.sample_steps
    original_distortion = model.cfg.sample.time_distortion
    original_guidance = model.cfg.general.guidance_weight if hasattr(model.cfg.general, 'guidance_weight') else 1.0

    model.cfg.sample.eta = eta
    model.cfg.sample.omega = omega
    model.cfg.sample.sample_steps = steps
    model.cfg.sample.time_distortion = time_distortion
    model.cfg.general.guidance_weight = guidance_weight
    model.rate_matrix_designer.eta = eta
    model.rate_matrix_designer.omega = omega

    samples = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        # Sample in batches
        batch_size = min(100, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size

        task = progress.add_task("Generating samples...", total=num_samples)

        samples_generated = 0
        while samples_generated < num_samples:
            current_batch_size = min(batch_size, num_samples - samples_generated)

            # Sample number of nodes
            n_nodes = model.node_dist.sample_n(current_batch_size, device)
            n_max = torch.max(n_nodes).item()

            # Build node mask
            arange = torch.arange(n_max, device=device).unsqueeze(0).expand(current_batch_size, -1)
            node_mask = arange < n_nodes.unsqueeze(1)

            # Sample initial noise
            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=model.noise_dist.get_limit_dist(),
                node_mask=node_mask
            )

            # Set conditions
            if conditions is not None:
                cond = conditions.unsqueeze(0).expand(current_batch_size, -1).to(device)
                z_T.y = cond
            else:
                z_T.y = torch.zeros(current_batch_size, 0, device=device)

            X, E, y = z_T.X, z_T.E, z_T.y

            # Sampling loop
            for t_int in range(steps):
                t_norm = t_int / steps
                s_norm = (t_int + 1) / steps

                t_tensor = torch.full((current_batch_size, 1), t_norm, device=device)
                s_tensor = torch.full((current_batch_size, 1), s_norm, device=device)

                # Handle absorbing state edge case
                if ("absorb" in model.cfg.model.transition) and (t_int == 0):
                    t_tensor = t_tensor + 1e-6

                # Apply time distortion
                t_tensor = model.time_distorter.sample_ft(t_tensor, time_distortion)
                s_tensor = model.time_distorter.sample_ft(s_tensor, time_distortion)

                # Sample next state
                sampled_s, _ = model.sample_p_zs_given_zt(
                    t_tensor, s_tensor, X, E, y, node_mask
                )
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Final processing
            sampled = utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse=True)
            X_final, E_final, _ = model.noise_dist.ignore_virtual_classes(
                sampled.X, sampled.E, sampled.y
            )

            # Extract individual samples
            for i in range(current_batch_size):
                n = n_nodes[i].item()
                atom_types = X_final[i, :n].cpu()
                edge_types = E_final[i, :n, :n].cpu()
                samples.append((atom_types, edge_types))

            samples_generated += current_batch_size
            progress.update(task, completed=samples_generated)

    # Restore original parameters
    model.cfg.sample.eta = original_eta
    model.cfg.sample.omega = original_omega
    model.cfg.sample.sample_steps = original_steps
    model.cfg.sample.time_distortion = original_distortion
    model.cfg.general.guidance_weight = original_guidance
    model.rate_matrix_designer.eta = original_eta
    model.rate_matrix_designer.omega = original_omega

    return samples


def convert_to_smiles(
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    dataset_infos,
) -> List[Optional[str]]:
    """Convert samples to SMILES strings."""
    from analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges

    smiles_list = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting to SMILES...", total=len(samples))

        for atom_types, edge_types in samples:
            try:
                mol = build_molecule_with_partial_charges(
                    atom_types, edge_types, dataset_infos.atom_decoder
                )
                smiles = mol2smiles(mol)
                smiles_list.append(smiles)
            except Exception:
                smiles_list.append(None)

            progress.advance(task)

    return smiles_list


def compute_properties_psi4(
    smiles_list: List[Optional[str]],
    properties: List[str],
) -> Dict[str, List[Optional[float]]]:
    """
    Compute molecular properties using PSI4.

    This is expensive - requires quantum chemistry calculations.
    """
    try:
        import psi4
        psi4.set_memory('2 GB')
        psi4.core.set_output_file('/dev/null', False)  # Suppress output
    except ImportError:
        raise ImportError(
            "PSI4 is required for computing molecular properties. "
            "Please install it or run in unconditional mode."
        )

    from rdkit import Chem
    from rdkit.Chem import AllChem

    results = {prop: [] for prop in properties}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Computing properties (PSI4)...", total=len(smiles_list))

        for smiles in smiles_list:
            if smiles is None:
                for prop in properties:
                    results[prop].append(None)
                progress.advance(task)
                continue

            try:
                # Generate 3D structure
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)

                # Generate conformer
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                AllChem.EmbedMolecule(mol, params)
                AllChem.MMFFOptimizeMolecule(mol)

                # Get coordinates for PSI4
                conf = mol.GetConformer()
                xyz_string = f"{mol.GetNumAtoms()}\n\n"
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    symbol = atom.GetSymbol()
                    xyz_string += f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"

                # PSI4 calculation
                psi4.geometry(xyz_string)
                psi4.set_options({'basis': '6-31g*', 'reference': 'rhf'})

                energy, wfn = psi4.energy('b3lyp', return_wfn=True)

                for prop in properties:
                    if prop == "mu":
                        # Dipole moment
                        dip = wfn.variable("SCF DIPOLE")
                        dipole_moment = math.sqrt(dip[0]**2 + dip[1]**2 + dip[2]**2) * 2.5417464519
                        results[prop].append(dipole_moment)
                    elif prop == "homo":
                        # HOMO energy
                        LUMO_idx = wfn.nalpha()
                        HOMO_idx = LUMO_idx - 1
                        homo = wfn.epsilon_a_subset("AO", "ALL").np[HOMO_idx]
                        results[prop].append(homo)
                    else:
                        results[prop].append(None)

            except Exception as e:
                for prop in properties:
                    results[prop].append(None)

            progress.advance(task)
            psi4.core.clean()

    return results


def create_distribution_plot(
    property_name: str,
    property_unit: str,
    dataset_values: np.ndarray,
    target_value: float,
    generated_values: List[Optional[float]],
    output_path: str,
) -> None:
    """Create distribution comparison plot."""

    # Filter None values from generated
    valid_generated = [v for v in generated_values if v is not None]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Dataset distribution (faint background)
    ax.hist(
        dataset_values,
        bins=50,
        density=True,
        alpha=0.3,
        color='gray',
        label='Dataset Distribution'
    )

    # KDE for dataset
    if len(dataset_values) > 1:
        kde_dataset = stats.gaussian_kde(dataset_values)
        x_range = np.linspace(
            min(dataset_values.min(), target_value) - 0.5,
            max(dataset_values.max(), target_value) + 0.5,
            200
        )
        ax.plot(x_range, kde_dataset(x_range), color='gray', linestyle='--', alpha=0.7)

    # Generated distribution
    if valid_generated:
        ax.hist(
            valid_generated,
            bins=30,
            density=True,
            alpha=0.6,
            color='steelblue',
            label='Generated Distribution'
        )

        # KDE for generated
        if len(valid_generated) > 1:
            kde_generated = stats.gaussian_kde(valid_generated)
            ax.plot(x_range, kde_generated(x_range), color='steelblue', linewidth=2)

    # Target condition line
    ax.axvline(
        x=target_value,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Target: {target_value:.4f}'
    )

    # Labels and title
    xlabel = f"{property_name}"
    if property_unit:
        xlabel += f" ({property_unit})"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{property_name} Distribution: Dataset vs Generated', fontsize=14)
    ax.legend(loc='upper right')

    # Add statistics text box
    if valid_generated:
        stats_text = f"Generated (n={len(valid_generated)}):\n"
        stats_text += f"  Mean: {np.mean(valid_generated):.4f}\n"
        stats_text += f"  Std: {np.std(valid_generated):.4f}\n"
        stats_text += f"  MAE from target: {np.mean(np.abs(np.array(valid_generated) - target_value)):.4f}"

        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_outputs(
    output_dir: Path,
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    smiles_list: List[Optional[str]],
    condition_info: List[Dict[str, Any]],
    property_values: Dict[str, np.ndarray],
    generated_properties: Dict[str, List[Optional[float]]],
    cfg: DictConfig,
    cli_args: Dict[str, Any],
) -> None:
    """Save all outputs to the specified directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save summary.json
    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": cli_args.get("checkpoint"),
        "num_samples": len(samples),
        "conditions": cli_args.get("condition"),
        "condition_info": condition_info,
        "sampling_params": {
            "eta": cli_args.get("eta"),
            "omega": cli_args.get("omega"),
            "steps": cli_args.get("steps"),
            "time_distortion": cli_args.get("time_distortion"),
            "guidance_weight": cli_args.get("guidance_weight"),
            "seed": cli_args.get("seed"),
        },
        "dataset": cfg.dataset.name,
        "results": {
            "valid": sum(1 for s in smiles_list if s is not None),
            "total": len(smiles_list),
            "validity_rate": sum(1 for s in smiles_list if s is not None) / len(smiles_list) if smiles_list else 0,
        }
    }

    # Add unique count
    valid_smiles = [s for s in smiles_list if s is not None]
    unique_smiles = set(valid_smiles)
    summary["results"]["unique"] = len(unique_smiles)
    summary["results"]["uniqueness_rate"] = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 2. Save samples.csv
    df_data = {
        "sample_id": list(range(len(samples))),
        "smiles": smiles_list,
        "valid": [s is not None for s in smiles_list],
    }

    # Add generated properties
    for prop, values in generated_properties.items():
        if prop in PROPERTY_INFO:
            col_name = f"{PROPERTY_INFO[prop]['name']} ({PROPERTY_INFO[prop]['unit']})"
        else:
            col_name = prop
        df_data[col_name] = values

    df = pd.DataFrame(df_data)
    df.to_csv(output_dir / "samples.csv", index=False)

    # 3. Save samples.pkl
    with open(output_dir / "samples.pkl", "wb") as f:
        pickle.dump(samples, f)

    # 4. Save smiles.txt
    with open(output_dir / "smiles.txt", "w") as f:
        for smiles in smiles_list:
            f.write(f"{smiles if smiles else 'INVALID'}\n")

    # 5. Create distribution plots for each condition
    if condition_info:
        properties = [info["property"] for info in condition_info if "property" in info]

        for i, (prop, info) in enumerate(zip(properties, condition_info)):
            if prop in property_values and prop in generated_properties:
                condition_dir = output_dir / f"condition_{i+1}"
                condition_dir.mkdir(exist_ok=True)

                # Save property info
                with open(condition_dir / "info.json", "w") as f:
                    json.dump(info, f, indent=2)

                # Create plot
                create_distribution_plot(
                    property_name=info.get("name", prop),
                    property_unit=info.get("unit", ""),
                    dataset_values=property_values[prop],
                    target_value=info["value"],
                    generated_values=generated_properties[prop],
                    output_path=str(condition_dir / "distribution_plot.png"),
                )

    console.print(f"\n[green]Outputs saved to {output_dir}[/green]")


def print_results_summary(
    smiles_list: List[Optional[str]],
    generated_properties: Dict[str, List[Optional[float]]],
    condition_info: List[Dict[str, Any]],
) -> None:
    """Print results summary table."""

    valid_count = sum(1 for s in smiles_list if s is not None)
    total_count = len(smiles_list)
    valid_smiles = [s for s in smiles_list if s is not None]
    unique_count = len(set(valid_smiles))

    console.print("\n")

    # Results table
    table = Table(title="Generation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Valid", f"{valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
    table.add_row("Unique", f"{unique_count}/{valid_count} ({100*unique_count/valid_count:.1f}%)" if valid_count > 0 else "N/A")

    # Property statistics
    if condition_info and generated_properties:
        for info in condition_info:
            if "property" not in info:
                continue
            prop = info["property"]
            if prop in generated_properties:
                valid_props = [v for v in generated_properties[prop] if v is not None]
                if valid_props:
                    target = info["value"]
                    mae = np.mean(np.abs(np.array(valid_props) - target))
                    name = info.get("name", prop)
                    table.add_row(f"{name} MAE", f"{mae:.4f}")

    console.print(table)


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model checkpoint file (.ckpt)",
)
@click.option(
    "--condition",
    type=float,
    multiple=True,
    default=None,
    help="Condition value(s) for conditional generation. "
         "Omit for unconditional generation. "
         "For multi-property conditioning, provide multiple values in order.",
)
@click.option(
    "--num-samples",
    type=int,
    default=100,
    show_default=True,
    help="Number of samples to generate",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./generated",
    show_default=True,
    help="Output directory for generated samples and plots",
)
@click.option(
    "--eta",
    type=float,
    default=None,
    help="Stochasticity parameter (eta). If not set, uses checkpoint default.",
)
@click.option(
    "--omega",
    type=float,
    default=None,
    help="Target guidance parameter (omega). If not set, uses checkpoint default.",
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="Number of sampling steps. If not set, uses checkpoint default.",
)
@click.option(
    "--time-distortion",
    type=click.Choice(["identity", "polydec", "cos", "revcos", "polyinc"]),
    default=None,
    help="Time distortion type. If not set, uses checkpoint default.",
)
@click.option(
    "--guidance-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Classifier-free guidance weight (only for conditional generation)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu'). Auto-detects if not set.",
)
def main(
    checkpoint: str,
    condition: Tuple[float, ...],
    num_samples: int,
    output_dir: str,
    eta: Optional[float],
    omega: Optional[float],
    steps: Optional[int],
    time_distortion: Optional[str],
    guidance_weight: float,
    seed: Optional[int],
    device: Optional[str],
):
    """
    Generate molecular graphs with optional conditioning.

    This script loads a trained DeFoG model and generates molecular samples,
    optionally conditioned on specified property values. For conditional
    generation, it validates conditions against the training distribution
    and produces comparison plots.

    Examples:

        # Unconditional generation
        python sample.py --checkpoint model.ckpt --num-samples 100

        # Conditional on dipole moment (mu)
        python sample.py --checkpoint model.ckpt --condition 2.5 --num-samples 100

        # Conditional on both mu and homo
        python sample.py --checkpoint model.ckpt --condition 2.5 -0.25 --num-samples 100

        # With custom sampling parameters
        python sample.py --checkpoint model.ckpt --condition 2.5 \\
            --eta 50 --omega 0.1 --steps 500 --time-distortion polydec
    """

    console.print(Panel.fit(
        "[bold blue]DeFoG Molecular Generator[/bold blue]\n"
        "Discrete Flow Matching for Graph Generation",
        border_style="blue"
    ))

    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        console.print(f"Random seed set to: [cyan]{seed}[/cyan]")

    # Get device
    device_obj = get_device(device)
    console.print(f"Using device: [cyan]{device_obj}[/cyan]")

    # Load config from checkpoint
    cfg, checkpoint_data = load_checkpoint(checkpoint, device_obj)

    # Convert condition tuple to list
    conditions_list = list(condition) if condition else []

    # Check if model is conditional
    is_conditional = getattr(cfg.general, "conditional", False)
    target = getattr(cfg.general, "target", None)

    if is_conditional:
        console.print(f"Model type: [green]Conditional[/green] (target: {target})")
    else:
        console.print(f"Model type: [yellow]Unconditional[/yellow]")

    # Validate early: unconditional model with conditions
    if not is_conditional and conditions_list:
        console.print("[red]Error: Model is unconditional but conditions were provided.[/red]")
        raise click.Abort()

    # Validate early: conditional model without conditions
    if is_conditional and not conditions_list:
        console.print("[yellow]Warning: Conditional model running in unconditional mode.[/yellow]")

    # Load model and data
    console.print("\nLoading model and dataset...")
    model, datamodule, dataset_infos, property_values = setup_model_and_data(
        cfg, checkpoint, device_obj
    )

    # Validate conditions
    is_valid, condition_info = validate_conditions(conditions_list, cfg, property_values)
    if not is_valid:
        for info in condition_info:
            if "error" in info:
                console.print(f"[red]Error: {info['error']}[/red]")
        raise click.Abort()

    print_condition_info(condition_info)

    # Set up sampling parameters (use checkpoint defaults if not specified)
    eta_val = eta if eta is not None else cfg.sample.eta
    omega_val = omega if omega is not None else cfg.sample.omega
    steps_val = steps if steps is not None else cfg.sample.sample_steps
    distortion_val = time_distortion if time_distortion is not None else cfg.sample.time_distortion

    console.print(f"\nSampling parameters:")
    console.print(f"  eta: [cyan]{eta_val}[/cyan]")
    console.print(f"  omega: [cyan]{omega_val}[/cyan]")
    console.print(f"  steps: [cyan]{steps_val}[/cyan]")
    console.print(f"  time_distortion: [cyan]{distortion_val}[/cyan]")
    console.print(f"  guidance_weight: [cyan]{guidance_weight}[/cyan]")

    # Prepare conditions tensor
    if conditions_list:
        conditions_tensor = torch.tensor(conditions_list, dtype=torch.float32)
    else:
        conditions_tensor = None

    # Generate samples
    console.print(f"\nGenerating {num_samples} samples...")
    samples = sample_molecules(
        model=model,
        num_samples=num_samples,
        conditions=conditions_tensor,
        device=device_obj,
        eta=eta_val,
        omega=omega_val,
        steps=steps_val,
        time_distortion=distortion_val,
        guidance_weight=guidance_weight,
    )

    # Convert to SMILES
    console.print("\nConverting to SMILES...")
    smiles_list = convert_to_smiles(samples, dataset_infos)

    # Compute properties for generated molecules (if conditional)
    generated_properties = {}
    if condition_info:
        properties = [info["property"] for info in condition_info if "property" in info]
        if properties:
            console.print("\nComputing molecular properties...")
            try:
                generated_properties = compute_properties_psi4(smiles_list, properties)
            except ImportError as e:
                console.print(f"[yellow]Warning: {e}[/yellow]")
                console.print("[yellow]Skipping property computation. Plots will only show target values.[/yellow]")
                generated_properties = {prop: [None] * len(smiles_list) for prop in properties}

    # Print results summary
    print_results_summary(smiles_list, generated_properties, condition_info)

    # Save outputs
    output_path = Path(output_dir)
    cli_args = {
        "checkpoint": checkpoint,
        "condition": conditions_list if conditions_list else None,
        "num_samples": num_samples,
        "eta": eta_val,
        "omega": omega_val,
        "steps": steps_val,
        "time_distortion": distortion_val,
        "guidance_weight": guidance_weight,
        "seed": seed,
    }

    save_outputs(
        output_dir=output_path,
        samples=samples,
        smiles_list=smiles_list,
        condition_info=condition_info,
        property_values=property_values,
        generated_properties=generated_properties,
        cfg=cfg,
        cli_args=cli_args,
    )

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
