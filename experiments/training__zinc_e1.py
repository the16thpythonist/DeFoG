"""
Unconditional DeFoG on ZINC250k, under the E1 evaluation protocol.

This is the protocol-conforming sibling of ``training__zinc_uncond.py``. That
script stays as it is -- it is the record of the LR sweep and every adapter/RL
run built on top of it. This one exists because E1 compares against numbers
taken from the literature rather than rerun locally, which makes protocol
fidelity the single point of failure (``docs/unconditional-protocol.md`` §8).

What differs from ``training__zinc_uncond.py``, and why:

  split            the GDSS/GruM reference split (24,887 held-out test rows
                   fixed by ``valid_idx_zinc250k.json``), not a fresh 90/10
                   shuffle. Validation is carved out of TRAIN and is the only
                   split the sampling sweep and checkpoint selection may see.
  source           the reference ``zinc250k.csv``. ``data/zinc_250k_rdkit.csv``
                   has been stereo-stripped and charge-neutralized -- harmless
                   for training graphs, wrong for any reference set that FCD,
                   scaffold similarity or novelty compares SMILES against.
  representation   KEKULIZED, 3 bond types, matching ``configs/dataset/zinc.yaml``
                   (``aromatic: False``). The sweep script uses the aromatic
                   4-bond vocabulary, which changes what counts as valid.
  vocabulary       frozen to the 9 ZINC elements in a fixed order, not derived
                   from whatever the CSV happens to contain.
  validity         three conventions reported side by side; the headline is the
                   relaxed + largest-fragment reading the published rows use.
  metrics          uniqueness/novelty emitted under BOTH denominators.
  recipe           published scale: 300 epochs at batch 256, lr 2e-4.

Still outstanding before an E1 table row can be produced (protocol §6), and
deliberately NOT done here:
  - FCD, NSPDK and scaffold similarity are not computed (items 6-8).
  - the sampling configuration has not been swept on validation, so
    ``sampling_config_frozen`` is recorded as False and the eval block below is
    a diagnostic, not a table row (item 10).

Usage:
    python experiments/training__zinc_e1.py
    python experiments/training__zinc_e1.py --LEARNING_RATE 3e-4
    python experiments/training__zinc_e1.py --__TESTING__ True
"""
import os
import sys

import numpy as np
import torch
import pytorch_lightning as pl
from rdkit import RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog.core import (  # noqa: E402
    DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback, EMACallback,
    BestValLossCheckpoint,
)
from defog.data import zinc_reference  # noqa: E402
from defog.domains import MoleculeDomain  # noqa: E402
from defog.domains.molecule import validity_report  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================

# --- Chaining across walltime windows ---
# :param CKPT_DIR:
#     When set, full training state (weights + optimizer + LR schedule + EMA
#     shadow + epoch) is written to <CKPT_DIR>/last.ckpt and auto-resumed from
#     there. This is what makes a 300-epoch run survive JUPITER's 12 h ceiling:
#     each chained job re-runs this same script and picks up where the last one
#     stopped. Give every parallel arm its own CKPT_DIR or they will resume from
#     each other's state.
CKPT_DIR: str = None
RESUME_CKPT: str = None
CKPT_EVERY_N_STEPS: int = 2000
# :param MAX_TIME_HOURS:
#     Wall-clock budget for THIS link, set below the SLURM limit so the job stops
#     itself and writes a clean checkpoint instead of being killed mid-write.
#     EPOCHS stays fixed across links -- it is the cosine horizon, so cutting it
#     per link would restart the LR schedule on every resume.
MAX_TIME_HOURS: float = None

# --- Reference data (protocol §2) ---
# :param DATA_ROOT: Where zinc250k.csv and valid_idx_zinc250k.json live. Both are
#     downloaded on first use and checked against pinned SHA256s; data/ is
#     gitignored so they are cached, not committed.
DATA_ROOT: str = os.path.join(_PROJECT_DIR, "data", "zinc250k")
VAL_SIZE: int = 5000
SPLIT_SEED: int = 42
# :param ALLOW_HASH_MISMATCH: Escape hatch for a changed upstream file. Leaving
#     this True would defeat the point of pinning; a run with it set must say so.
ALLOW_HASH_MISMATCH: bool = False

# --- Representation (protocol trap 6) ---
ATOM_TYPES: list = list(zinc_reference.ATOM_TYPES)   # C N O F P S Cl Br I
BOND_TYPES: list = list(zinc_reference.BOND_TYPES)   # SINGLE DOUBLE TRIPLE
KEKULIZE: bool = True

# --- Model architecture (settled defog recipe) ---
N_LAYERS: int = 9
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 20

# --- Training (published scale: configs/experiment/zinc.yaml) ---
EPOCHS: int = 300
BATCH_SIZE: int = 256
LEARNING_RATE: float = 2e-4
LR_SCHEDULER: str = "cosine"
LR_MIN: float = 1e-6
WEIGHT_DECAY: float = 1e-5
LAMBDA_EDGE: float = 5.0
TRAIN_TIME_DISTORTION: str = "polydec"
EMA_DECAY: float = 0.9999
NUM_WORKERS: int = 4

MOLECULAR_FEATURES: bool = True
# :param ATOM_VALENCY: Maximum valencies, matching ZINCinfos in the legacy path
#     (src/datasets/zinc_dataset.py) rather than the reduced P=3/S=2 table used
#     by the sweep script, so the molecular features agree between the two
#     implementations.
ATOM_VALENCY: dict = {
    "C": 4, "N": 3, "O": 2, "F": 1, "P": 5, "S": 6, "Cl": 1, "Br": 1, "I": 1,
}
ATOM_WEIGHT_TABLE: dict = {
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Br": 79.904, "I": 126.904,
}
MAX_ATOM_WEIGHT: float = 500.0   # ZINCinfos.max_weight

# --- Sampling / evaluation ---
# :param SAMPLE_STEPS / ETA / OMEGA:
#     PLACEHOLDERS. The protocol requires these to be swept on the validation
#     split and frozen before the single test pass; that sweep is item 10 and has
#     not run. Until it does, SAMPLING_CONFIG_FROZEN stays False and no number
#     produced here belongs in a paper table.
SAMPLE_STEPS: int = 500
ETA: float = 0.0
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "polydec"
SAMPLING_CONFIG_FROZEN: bool = False

# :param NUM_EVAL_SAMPLES: 10,000 is the ZINC convention and matches
#     configs/experiment/zinc.yaml's final_model_samples_to_generate.
NUM_EVAL_SAMPLES: int = 10000
EVAL_CHUNK: int = 250
# :param SKIP_FINAL_EVAL:
#     Training links set this. Sampling 10,000 molecules at an eta/omega that has
#     not been swept yet costs about an hour of GPU time and produces a number
#     that cannot go in a table, so the production chain trains only and
#     evaluation happens later as its own job.
SKIP_FINAL_EVAL: bool = False

SAMPLE_VIS_EVERY_K: int = 10
GEN_PROBE_EVERY_K: int = 10
GEN_PROBE_SAMPLES: int = 64
GEN_PROBE_STEPS: int = 100

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log("ZINC250k UNCONDITIONAL -- E1 protocol")
    pl.seed_everything(e.SEED, workers=True)

    # -- Reference split ----------------------------------------------------
    # Hard-fails on a hash or count mismatch. That is the intended behaviour:
    # with published-only baselines a drifted split shows up as an unexplained
    # performance gap, never as an error message, unless it is checked here.
    split = zinc_reference.load_reference_split(
        e.DATA_ROOT,
        val_size=e.VAL_SIZE,
        split_seed=e.SPLIT_SEED,
        allow_hash_mismatch=e.ALLOW_HASH_MISMATCH,
    )
    e.log(split.summary())
    e["provenance/split"] = split.provenance

    # -- Encode -------------------------------------------------------------
    # The test split is hashed and counted above and is NOT encoded, loaded or
    # touched again in this script. It exists for one evaluation pass, later,
    # with a sampling configuration already frozen on validation.
    train_graphs, train_smiles, n_skip_train = zinc_reference.build_graphs(
        split.train_smiles, atom_types=e.ATOM_TYPES, bond_types=e.BOND_TYPES,
        kekulize=e.KEKULIZE, progress=True,
    )
    val_graphs, val_smiles, n_skip_val = zinc_reference.build_graphs(
        split.val_smiles, atom_types=e.ATOM_TYPES, bond_types=e.BOND_TYPES,
        kekulize=e.KEKULIZE,
    )
    e.log(f"encoded train {len(train_graphs)} (skipped {n_skip_train}) | "
          f"val {len(val_graphs)} (skipped {n_skip_val})")
    e["provenance/encoding"] = {
        "atom_types": list(e.ATOM_TYPES),
        "bond_types": list(e.BOND_TYPES),
        "kekulize": e.KEKULIZE,
        "remove_h": zinc_reference.REMOVE_H,
        "aromatic": zinc_reference.AROMATIC,
        "n_train_graphs": len(train_graphs),
        "n_val_graphs": len(val_graphs),
        "n_skipped_train": n_skip_train,
        "n_skipped_val": n_skip_val,
    }
    # A frozen vocabulary should fit the reference data exactly. Anything more
    # than a rounding error means the vocabulary and the data disagree, and the
    # model would be trained on a quietly filtered dataset.
    skip_frac = n_skip_train / max(1, len(split.train_smiles))
    if skip_frac > 0.001:
        e.log(f"WARNING: {skip_frac:.2%} of train molecules did not encode. "
              f"Expected ~0 for ZINC250k under the frozen 9-element vocabulary.")

    from torch_geometric.loader import DataLoader
    # persistent_workers stays OFF deliberately. At 858 steps/epoch the respawn
    # cost is noise, and a lingering worker pool is the suspected reason a failed
    # arm of the 978228 sweep sat on a GPU for ~5 h instead of exiting.
    train_loader = DataLoader(train_graphs, batch_size=e.BATCH_SIZE, shuffle=True,
                              num_workers=e.NUM_WORKERS, persistent_workers=False)
    val_loader = DataLoader(val_graphs, batch_size=e.BATCH_SIZE,
                            num_workers=e.NUM_WORKERS, persistent_workers=False)

    atom_valencies = [e.ATOM_VALENCY[a] for a in e.ATOM_TYPES]
    atom_weights_list = [e.ATOM_WEIGHT_TABLE[a] for a in e.ATOM_TYPES]

    # -- Model --------------------------------------------------------------
    model = DeFoGModel.from_dataloader(
        train_loader,
        n_layers=e.N_LAYERS, hidden_dim=e.HIDDEN_DIM, hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS, dropout=e.DROPOUT, noise_type=e.NOISE_TYPE,
        extra_features_type=e.EXTRA_FEATURES_TYPE, rrwp_steps=e.RRWP_STEPS,
        molecular_features=e.MOLECULAR_FEATURES, atom_valencies=atom_valencies,
        atom_weights=atom_weights_list, max_atom_weight=e.MAX_ATOM_WEIGHT,
        lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY,
        lambda_edge=e.LAMBDA_EDGE, train_time_distortion=e.TRAIN_TIME_DISTORTION,
        lr_scheduler=e.LR_SCHEDULER, lr_min=e.LR_MIN,
        sample_steps=e.SAMPLE_STEPS, eta=e.ETA, omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
    )
    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"Model params: {e['model/num_params']:,}")

    # Node counts come from the training loader, so graph sizes are drawn from
    # the training distribution rather than fixed or uniform (protocol trap 5).
    size_dist = model.default_size_dist
    e["provenance/size_distribution"] = {
        "type": type(size_dist).__name__,
        "source": "train split node-count histogram",
        "max_size": int(size_dist.max_size),
    }

    # -- Train --------------------------------------------------------------
    atom_decoder = list(e.ATOM_TYPES)
    from defog.domains.molecule import build_encoders
    _, atom_decoder, _, bond_decoder = build_encoders(e.ATOM_TYPES, e.BOND_TYPES)

    def probe_metrics(samples):
        rep = validity_report(samples, atom_decoder, bond_decoder,
                              reference_smiles=train_smiles)
        return {"validity": rep["validity_relaxed_largest_frag"],
                "uniqueness": rep["uniqueness"], "novelty": rep.get("novelty", 0.0)}

    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig),
        generation_metrics_fn=probe_metrics, gen_every_k=e.GEN_PROBE_EVERY_K,
        gen_num_samples=e.GEN_PROBE_SAMPLES, gen_sample_steps=e.GEN_PROBE_STEPS,
        # No checkpoint_dir: selection is on val/loss, not on a 64-sample probe.
    )
    mol_domain = MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_smiles)
    sampler = SampleVisualizationCallback(
        num_samples=8, every_k_epochs=e.SAMPLE_VIS_EVERY_K,
        sample_steps=e.GEN_PROBE_STEPS, domain=mol_domain,
        figure_callback=lambda fig: e.track("samples", fig),
    )
    # The best-val-loss model goes in CKPT_DIR, not the pycomex archive: each
    # chained link gets a fresh timestamped archive, so leaving it there would
    # scatter one run's best checkpoint across N directories.
    best_dir = e.CKPT_DIR if e.CKPT_DIR else e.path
    best_ckpt = BestValLossCheckpoint(checkpoint_dir=best_dir, monitor="val/loss")

    callbacks = [monitor, sampler, best_ckpt]
    if e.EMA_DECAY and e.EMA_DECAY > 0:
        callbacks = [EMACallback(decay=e.EMA_DECAY)] + callbacks
        e.log(f"EMA enabled (decay={e.EMA_DECAY})")

    max_time = None
    if e.MAX_TIME_HOURS:
        hrs = int(e.MAX_TIME_HOURS)
        mins = int(round((e.MAX_TIME_HOURS - hrs) * 60))
        max_time = {"hours": hrs, "minutes": mins}
        e.log(f"Trainer max_time = {hrs}h{mins:02d}m (stops cleanly before the SLURM kill)")

    resume_path, enable_ckpt = e.RESUME_CKPT, False
    if e.CKPT_DIR:
        os.makedirs(e.CKPT_DIR, exist_ok=True)
        from pytorch_lightning.callbacks import ModelCheckpoint
        callbacks.append(ModelCheckpoint(
            dirpath=e.CKPT_DIR, save_last=True, save_top_k=0,
            every_n_train_steps=e.CKPT_EVERY_N_STEPS,
        ))
        enable_ckpt = True
        auto = os.path.join(e.CKPT_DIR, "last.ckpt")
        if resume_path is None and os.path.exists(auto):
            resume_path = auto
        e.log(f"Resumable checkpointing -> {e.CKPT_DIR} "
              f"(every {e.CKPT_EVERY_N_STEPS} steps); "
              + (f"RESUMING from {resume_path}" if resume_path else "fresh start"))

    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, max_time=max_time, accelerator="auto", devices=1,
        enable_progress_bar=True, enable_checkpointing=enable_ckpt, logger=False,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=resume_path)
    e.log(f"Saved final model to {model.save(os.path.join(e.path, 'model'))}")
    e["training/best_val_loss"] = best_ckpt.best
    e["training/best_val_epoch"] = best_ckpt.best_epoch
    e["training/epochs_completed"] = int(trainer.current_epoch)
    e["training/finished_horizon"] = int(trainer.current_epoch) >= e.EPOCHS
    e.log(f"epochs completed: {trainer.current_epoch}/{e.EPOCHS}")

    if e.SKIP_FINAL_EVAL:
        e.log("SKIP_FINAL_EVAL: training link only, no sampling. Evaluate after "
              "the chain completes and the sampling config has been swept on val.")
        return

    # -- Diagnostic evaluation ----------------------------------------------
    # NOT a table row: the sampling configuration has not been swept on
    # validation yet, and FCD / NSPDK / scaffold similarity are not implemented.
    e.log("=" * 60)
    e.log(f"DIAGNOSTIC EVAL: {e.NUM_EVAL_SAMPLES} samples "
          f"(sampling_config_frozen={e.SAMPLING_CONFIG_FROZEN})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if best_ckpt.saved_path and os.path.exists(best_ckpt.saved_path):
        e.log(f"loading best-val-loss checkpoint (val/loss={best_ckpt.best:.5f}, "
              f"epoch {best_ckpt.best_epoch})")
        model = DeFoGModel.load(os.path.join(e.path, "best_model"))
    model = model.to(device).eval()

    samples, remaining = [], e.NUM_EVAL_SAMPLES
    while remaining > 0:
        cur = min(e.EVAL_CHUNK, remaining)
        samples += model.sample(num_samples=cur, sample_steps=e.SAMPLE_STEPS,
                                device=device, show_progress=False)
        remaining -= cur
    e.log(f"generated {len(samples)} samples")

    report = validity_report(samples, atom_decoder, bond_decoder,
                             reference_smiles=train_smiles)
    gen_smiles = report.pop("smiles")
    e.commit_json("generated_smiles.json", gen_smiles)

    e.log(f"validity (relaxed, largest frag) = {report['validity_relaxed_largest_frag']:.4f}"
          f"   <- headline, comparable to published ZINC rows")
    e.log(f"validity (strict, no correction) = {report['validity_strict_largest_frag']:.4f}")
    e.log(f"validity (whole molecule)        = {report['validity_whole_molecule']:.4f}")
    e.log(f"per-valid:  uniqueness={report['uniqueness']:.4f} "
          f"novelty={report.get('novelty', float('nan')):.4f}")
    e.log(f"cumulative: V={report['v']:.4f} V.U.={report['v_u']:.4f} "
          f"V.U.N.={report.get('v_u_n', float('nan')):.4f}")

    e.commit_json("e1_report.json", {
        "metrics": report,
        "provenance": {
            "split": split.provenance,
            "encoding": e["provenance/encoding"],
            "size_distribution": e["provenance/size_distribution"],
            "sampling": {
                "steps": e.SAMPLE_STEPS, "eta": e.ETA, "omega": e.OMEGA,
                "time_distortion": e.SAMPLE_TIME_DISTORTION,
                "n_samples": len(samples),
                "seed": e.SEED,
                "frozen": e.SAMPLING_CONFIG_FROZEN,
            },
            "recipe": {
                "n_layers": e.N_LAYERS, "hidden_dim": e.HIDDEN_DIM,
                "epochs": e.EPOCHS, "batch_size": e.BATCH_SIZE,
                "lr": e.LEARNING_RATE, "ema_decay": e.EMA_DECAY,
            },
            "not_yet_implemented": ["fcd", "nspdk", "scaffold_similarity"],
            "novelty_reference": "train split",
        },
    })
    e.log("Done. NOT a table row: sweep the sampling config on validation and "
          "implement FCD/NSPDK/scaffold similarity first.")


@experiment.testing
def testing(e: Experiment):
    """Smoke configuration: proves the pipeline executes without a real run.

    Subsamples the reference split AFTER it has been loaded and verified, so the
    hash/count checks still run for real -- a smoke test that skipped them would
    not exercise the part most likely to break.
    """
    # Only shrink the horizon if the caller did not set one. The testing hook
    # runs after CLI parsing, so an unconditional assignment would silently
    # discard `--EPOCHS N` -- which is exactly what a chain-resume smoke test
    # needs to vary in order to prove the second link advances past the first.
    if e.EPOCHS == 300:
        e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.NUM_WORKERS = 0
    e.VAL_SIZE = 200
    e.SAMPLE_STEPS = 5
    e.GEN_PROBE_STEPS = 5
    e.GEN_PROBE_EVERY_K = 1
    e.GEN_PROBE_SAMPLES = 8
    e.NUM_EVAL_SAMPLES = 32
    e.EVAL_CHUNK = 8
    e.SAMPLE_VIS_EVERY_K = 1
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2

    # Truncate the train split only; the loader's own count assertions have
    # already passed against the full reference files by the time this runs.
    _real_loader = zinc_reference.load_reference_split

    def _small_loader(*args, **kwargs):
        split = _real_loader(*args, **kwargs)
        return zinc_reference.ZincReferenceSplit(
            train_smiles=split.train_smiles[:400],
            val_smiles=split.val_smiles,
            test_smiles=split.test_smiles,
            provenance={**split.provenance, "SMOKE_TEST_TRUNCATED": True},
        )

    zinc_reference.load_reference_split = _small_loader


experiment.run_if_main()
