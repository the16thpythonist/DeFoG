"""
Unconditional DeFoG on GuacaMol, under the E1 evaluation protocol.

Sibling of ``training__zinc_e1.py``. ``training__guacamol_uncond.py`` stays as
the record of the earlier 4-seed run, but its numbers cannot be used for E1:
it trains on ``data/guacamol/guacamol_all.smiles``, which is the COMBINED
release (1,591,378 = train + valid + test), split randomly. That means the
official test set was in its training data.

What differs from ``training__guacamol_uncond.py``:

  split         Brown et al.'s official train/valid/test, hash-verified. No
                shuffle and no split seed -- the split is not ours to make.
                GuacaMol ships a real ``valid``, so unlike ZINC nothing is
                carved out of train.
  vocabulary    frozen 12 elements incl. Se/B/Si, not derived from the first
                50k rows (Se and B are rare enough for a sample to miss them).
  filter        the ``filter: True`` round-trip is applied and RECORDED. It
                drops ~12.2% of train, so filtered and raw are different
                datasets and a published number assumes one of them.
  metrics       validity under all three conventions, uniqueness/novelty under
                both denominators, novelty against both reference conventions.

Representation is AROMATIC (4 bond types, no kekulization), matching
``src/datasets/guacamol_dataset.py`` -- deliberately the opposite of the ZINC E1
choice, and recorded per run because it changes what counts as valid.

On the epoch count: ``configs/experiment/guacamol.yaml`` specifies 1000 epochs
at batch 64, which is ~19.9M steps and not reachable -- at an optimistic 10 it/s
that is ~23 days. This experiment is therefore WALL-CLOCK bounded and reports
the epochs actually reached, as an explicit deviation. EPOCHS is the cosine
horizon, held fixed across chained links; MAX_TIME_HOURS cuts each link.

Usage:
    python experiments/training__guacamol_e1.py --__TESTING__ True
    python experiments/training__guacamol_e1.py --BATCH_SIZE 128 --MAX_TIME_HOURS 10.5 \
        --CKPT_DIR "'ckpts/guacamol_e1_seed42'" --SKIP_FINAL_EVAL True
"""
import hashlib
import os
import sys

import torch
import pytorch_lightning as pl
from rdkit import RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog.core import (  # noqa: E402
    DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback, EMACallback,
    BestValLossCheckpoint,
)
from defog.data import guacamol_reference as gmref  # noqa: E402
from defog.domains import MoleculeDomain  # noqa: E402
from defog.domains.molecule import build_encoders, validity_report  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================
DATA_ROOT: str = os.path.join(_PROJECT_DIR, "data", "guacamol")
ALLOW_HASH_MISMATCH: bool = False

# :param FILTER_ROUNDTRIP:
#     configs/dataset/guacamol.yaml's `filter: True`. Drops ~12.2% of train
#     (~155k of 1,273,104), so this flag selects between two different datasets
#     and must appear in the run record either way.
FILTER_ROUNDTRIP: bool = True

# :param NOVELTY_REFERENCE:
#     "decoded" matches the DiGress/DeFoG lineage, which stores the round-tripped
#     SMILES as its training set and therefore as its novelty reference.
#     "source" uses the actual dataset molecules. They differ for ~5% of kept
#     molecules. Both are computed and reported; this only picks the headline.
NOVELTY_REFERENCE: str = "decoded"

# --- Representation (protocol trap 6) ---
ATOM_TYPES: list = list(gmref.ATOM_TYPES)   # C N O F B Br Cl I P S Se Si
BOND_TYPES: list = list(gmref.BOND_TYPES)   # SINGLE DOUBLE TRIPLE AROMATIC

# --- Graph cache ---
# :param GRAPH_CACHE_DIR:
#     Encoding 1.27M molecules costs real minutes and a chained run would pay it
#     on EVERY link. The cache key hashes the source md5, vocabulary and filter
#     setting, so a stale cache cannot be silently reused after any of those
#     change. Set to None to disable.
GRAPH_CACHE_DIR: str = os.path.join(_PROJECT_DIR, "data", "guacamol", "_graph_cache")

# --- Model architecture ---
N_LAYERS: int = 9
HIDDEN_DIM: int = 256
HIDDEN_MLP_DIM: int = 512
N_HEADS: int = 8
DROPOUT: float = 0.1
NOISE_TYPE: str = "marginal"
EXTRA_FEATURES_TYPE: str = "rrwp"
RRWP_STEPS: int = 20

# --- Training ---
# :param EPOCHS:
#     The cosine horizon, NOT a promise to run that many. Held fixed across
#     links so the LR schedule is continuous; MAX_TIME_HOURS ends each link.
EPOCHS: int = 100
BATCH_SIZE: int = 128           # set from the throughput probe
LEARNING_RATE: float = 2e-4
LR_SCHEDULER: str = "cosine"
LR_MIN: float = 1e-6
WEIGHT_DECAY: float = 1e-5
LAMBDA_EDGE: float = 5.0
TRAIN_TIME_DISTORTION: str = "polydec"
EMA_DECAY: float = 0.9999
NUM_WORKERS: int = 8

MOLECULAR_FEATURES: bool = True
ATOM_VALENCY: dict = {
    "C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Br": 1,
    "Cl": 1, "I": 1, "P": 5, "S": 6, "Se": 2, "Si": 4,
}
ATOM_WEIGHT_TABLE: dict = {
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "B": 10.81,
    "Br": 79.904, "Cl": 35.45, "I": 126.904, "P": 30.974, "S": 32.06,
    "Se": 78.971, "Si": 28.085,
}
MAX_ATOM_WEIGHT: float = 1000.0   # GuacaMol reaches ~72 heavy atoms

# --- Chaining ---
CKPT_DIR: str = None
RESUME_CKPT: str = None
CKPT_EVERY_N_STEPS: int = 2000
MAX_TIME_HOURS: float = None

# --- Sampling / evaluation ---
# PLACEHOLDERS until the validation sweep runs (protocol section 5).
SAMPLE_STEPS: int = 500
ETA: float = 0.0
OMEGA: float = 0.0
SAMPLE_TIME_DISTORTION: str = "polydec"
SAMPLING_CONFIG_FROZEN: bool = False

NUM_EVAL_SAMPLES: int = 10000
EVAL_CHUNK: int = 100
SKIP_FINAL_EVAL: bool = False

SAMPLE_VIS_EVERY_K: int = 5
GEN_PROBE_EVERY_K: int = 5
GEN_PROBE_SAMPLES: int = 64
GEN_PROBE_STEPS: int = 100

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


def _cache_path(cache_dir, provenance, atom_types, bond_types, filter_roundtrip):
    """Cache filename keyed by everything that changes the encoded graphs."""
    key = "|".join([
        provenance.get("train_md5", "?"),
        ",".join(atom_types), ",".join(bond_types),
        f"filter={filter_roundtrip}", f"kekulize={gmref.KEKULIZE}",
    ])
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"guacamol_graphs_{digest}.pt")


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log("GuacaMol UNCONDITIONAL -- E1 protocol")
    pl.seed_everything(e.SEED, workers=True)

    # -- Official split -----------------------------------------------------
    split = gmref.load_reference_split(
        e.DATA_ROOT, allow_hash_mismatch=e.ALLOW_HASH_MISMATCH
    )
    e.log(split.summary())
    e["provenance/split"] = split.provenance

    # -- Encode (cached) ----------------------------------------------------
    # The test split is hashed and counted by the loader above and is NOT
    # encoded or touched here.
    cache_file = None
    if e.GRAPH_CACHE_DIR:
        os.makedirs(e.GRAPH_CACHE_DIR, exist_ok=True)
        cache_file = _cache_path(e.GRAPH_CACHE_DIR, split.provenance,
                                 e.ATOM_TYPES, e.BOND_TYPES, e.FILTER_ROUNDTRIP)

    if cache_file and os.path.exists(cache_file):
        e.log(f"loading encoded graphs from cache {os.path.basename(cache_file)}")
        blob = torch.load(cache_file, weights_only=False)
        train_graphs, train_src, train_dec, tr_stats = (
            blob["train_graphs"], blob["train_src"], blob["train_dec"], blob["train_stats"])
        val_graphs, val_src = blob["val_graphs"], blob["val_src"]
    else:
        e.log("encoding train split (this is the slow part; it will be cached)")
        train_graphs, train_src, train_dec, tr_stats = gmref.build_graphs(
            split.train_smiles, atom_types=e.ATOM_TYPES, bond_types=e.BOND_TYPES,
            filter_roundtrip=e.FILTER_ROUNDTRIP, progress=True,
        )
        val_graphs, val_src, _, _ = gmref.build_graphs(
            split.val_smiles, atom_types=e.ATOM_TYPES, bond_types=e.BOND_TYPES,
            filter_roundtrip=e.FILTER_ROUNDTRIP,
        )
        if cache_file:
            tmp = cache_file + ".partial"
            torch.save({"train_graphs": train_graphs, "train_src": train_src,
                        "train_dec": train_dec, "train_stats": tr_stats,
                        "val_graphs": val_graphs, "val_src": val_src}, tmp)
            os.replace(tmp, cache_file)  # rename only when complete
            e.log(f"cached encoded graphs -> {os.path.basename(cache_file)}")

    e.log(f"train graphs {len(train_graphs)} (kept {tr_stats['kept_fraction']:.4f} "
          f"of {tr_stats['n_input']}) | val graphs {len(val_graphs)}")
    e["provenance/encoding"] = {
        "atom_types": list(e.ATOM_TYPES), "bond_types": list(e.BOND_TYPES),
        "aromatic": gmref.AROMATIC, "kekulize": gmref.KEKULIZE,
        "filter_roundtrip": e.FILTER_ROUNDTRIP,
        "filter_stats": tr_stats,
        "n_train_graphs": len(train_graphs), "n_val_graphs": len(val_graphs),
    }

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=e.BATCH_SIZE, shuffle=True,
                              num_workers=e.NUM_WORKERS, persistent_workers=False)
    val_loader = DataLoader(val_graphs, batch_size=e.BATCH_SIZE,
                            num_workers=e.NUM_WORKERS, persistent_workers=False)

    # -- Model --------------------------------------------------------------
    model = DeFoGModel.from_dataloader(
        train_loader,
        n_layers=e.N_LAYERS, hidden_dim=e.HIDDEN_DIM, hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS, dropout=e.DROPOUT, noise_type=e.NOISE_TYPE,
        extra_features_type=e.EXTRA_FEATURES_TYPE, rrwp_steps=e.RRWP_STEPS,
        molecular_features=e.MOLECULAR_FEATURES,
        atom_valencies=[e.ATOM_VALENCY[a] for a in e.ATOM_TYPES],
        atom_weights=[e.ATOM_WEIGHT_TABLE[a] for a in e.ATOM_TYPES],
        max_atom_weight=e.MAX_ATOM_WEIGHT,
        lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY,
        lambda_edge=e.LAMBDA_EDGE, train_time_distortion=e.TRAIN_TIME_DISTORTION,
        lr_scheduler=e.LR_SCHEDULER, lr_min=e.LR_MIN,
        sample_steps=e.SAMPLE_STEPS, eta=e.ETA, omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
    )
    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"Model params: {e['model/num_params']:,}")
    e["provenance/size_distribution"] = {
        "type": type(model.default_size_dist).__name__,
        "source": "train split node-count histogram",
        "max_size": int(model.default_size_dist.max_size),
    }

    # -- Train --------------------------------------------------------------
    _, atom_decoder, _, bond_decoder = build_encoders(e.ATOM_TYPES, e.BOND_TYPES)
    novelty_ref = train_dec if e.NOVELTY_REFERENCE == "decoded" else train_src
    novelty_ref = [s for s in novelty_ref if s]

    def probe_metrics(samples):
        rep = validity_report(samples, atom_decoder, bond_decoder,
                              reference_smiles=novelty_ref)
        return {"validity": rep["validity_relaxed_largest_frag"],
                "uniqueness": rep["uniqueness"], "novelty": rep.get("novelty", 0.0)}

    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig),
        generation_metrics_fn=probe_metrics, gen_every_k=e.GEN_PROBE_EVERY_K,
        gen_num_samples=e.GEN_PROBE_SAMPLES, gen_sample_steps=e.GEN_PROBE_STEPS,
    )
    sampler = SampleVisualizationCallback(
        num_samples=8, every_k_epochs=e.SAMPLE_VIS_EVERY_K,
        sample_steps=e.GEN_PROBE_STEPS,
        domain=MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=novelty_ref),
        figure_callback=lambda fig: e.track("samples", fig),
    )
    best_ckpt = BestValLossCheckpoint(
        checkpoint_dir=e.CKPT_DIR if e.CKPT_DIR else e.path, monitor="val/loss")

    callbacks = [monitor, sampler, best_ckpt]
    if e.EMA_DECAY and e.EMA_DECAY > 0:
        callbacks = [EMACallback(decay=e.EMA_DECAY)] + callbacks
        e.log(f"EMA enabled (decay={e.EMA_DECAY})")

    max_time = None
    if e.MAX_TIME_HOURS:
        hrs = int(e.MAX_TIME_HOURS)
        mins = int(round((e.MAX_TIME_HOURS - hrs) * 60))
        max_time = {"hours": hrs, "minutes": mins}
        e.log(f"Trainer max_time = {hrs}h{mins:02d}m")

    resume_path, enable_ckpt = e.RESUME_CKPT, False
    if e.CKPT_DIR:
        os.makedirs(e.CKPT_DIR, exist_ok=True)
        from pytorch_lightning.callbacks import ModelCheckpoint
        callbacks.append(ModelCheckpoint(dirpath=e.CKPT_DIR, save_last=True,
                                         save_top_k=0,
                                         every_n_train_steps=e.CKPT_EVERY_N_STEPS))
        enable_ckpt = True
        auto = os.path.join(e.CKPT_DIR, "last.ckpt")
        if resume_path is None and os.path.exists(auto):
            resume_path = auto
        e.log(f"Resumable checkpointing -> {e.CKPT_DIR}; "
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
    # The honest headline: GuacaMol is wall-clock bounded, so the epoch count is
    # a result to report, not a target that was met.
    e.log(f"epochs completed: {trainer.current_epoch}/{e.EPOCHS} (cosine horizon)")

    if e.SKIP_FINAL_EVAL:
        e.log("SKIP_FINAL_EVAL: training link only.")
        return

    # -- Diagnostic evaluation ----------------------------------------------
    e.log("=" * 60)
    e.log(f"DIAGNOSTIC EVAL: {e.NUM_EVAL_SAMPLES} samples "
          f"(sampling_config_frozen={e.SAMPLING_CONFIG_FROZEN})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if best_ckpt.saved_path and os.path.exists(best_ckpt.saved_path):
        model = DeFoGModel.load(os.path.join(
            e.CKPT_DIR if e.CKPT_DIR else e.path, "best_model"))
    model = model.to(device).eval()

    samples, remaining = [], e.NUM_EVAL_SAMPLES
    while remaining > 0:
        cur = min(e.EVAL_CHUNK, remaining)
        samples += model.sample(num_samples=cur, sample_steps=e.SAMPLE_STEPS,
                                device=device, show_progress=False)
        remaining -= cur

    rep_dec = validity_report(samples, atom_decoder, bond_decoder,
                              reference_smiles=[s for s in train_dec if s])
    rep_src = validity_report(samples, atom_decoder, bond_decoder,
                              reference_smiles=train_src)
    gen_smiles = rep_dec.pop("smiles")
    rep_src.pop("smiles", None)
    e.commit_json("generated_smiles.json", gen_smiles)

    e.log(f"validity (relaxed, largest frag) = {rep_dec['validity_relaxed_largest_frag']:.4f}")
    e.log(f"validity (strict, no correction) = {rep_dec['validity_strict_largest_frag']:.4f}")
    e.log(f"validity (whole molecule)        = {rep_dec['validity_whole_molecule']:.4f}")
    e.log(f"cumulative: V={rep_dec['v']:.4f} V.U.={rep_dec['v_u']:.4f} "
          f"V.U.N.(decoded ref)={rep_dec.get('v_u_n', float('nan')):.4f} "
          f"V.U.N.(source ref)={rep_src.get('v_u_n', float('nan')):.4f}")

    e.commit_json("e1_report.json", {
        "metrics": rep_dec,
        "metrics_novelty_source_reference": {
            "novelty": rep_src.get("novelty"), "v_u_n": rep_src.get("v_u_n")},
        "provenance": {
            "split": split.provenance,
            "encoding": e["provenance/encoding"],
            "size_distribution": e["provenance/size_distribution"],
            "novelty_reference_headline": e.NOVELTY_REFERENCE,
            "sampling": {"steps": e.SAMPLE_STEPS, "eta": e.ETA, "omega": e.OMEGA,
                         "time_distortion": e.SAMPLE_TIME_DISTORTION,
                         "n_samples": len(samples), "seed": e.SEED,
                         "frozen": e.SAMPLING_CONFIG_FROZEN},
            "recipe": {"n_layers": e.N_LAYERS, "hidden_dim": e.HIDDEN_DIM,
                       "epochs_horizon": e.EPOCHS,
                       "epochs_completed": e["training/epochs_completed"],
                       "batch_size": e.BATCH_SIZE, "lr": e.LEARNING_RATE},
            "not_yet_implemented": ["guacamol KL/FCD are computed separately via "
                                    "scripts/e1_metrics.py under .venv_metrics"],
        },
    })
    e.log("Done. NOT a table row until the sampling config is swept on validation.")


@experiment.testing
def testing(e: Experiment):
    if e.EPOCHS == 100:
        e.EPOCHS = 2
    e.BATCH_SIZE = 8
    e.NUM_WORKERS = 0
    e.SAMPLE_STEPS = 5
    e.GEN_PROBE_STEPS = 5
    e.GEN_PROBE_EVERY_K = 1
    e.GEN_PROBE_SAMPLES = 4
    e.NUM_EVAL_SAMPLES = 16
    e.EVAL_CHUNK = 8
    e.SAMPLE_VIS_EVERY_K = 1
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2
    e.GRAPH_CACHE_DIR = None   # never cache a truncated smoke dataset

    _real = gmref.load_reference_split

    def _small(*a, **kw):
        s = _real(*a, **kw)
        return gmref.GuacamolReferenceSplit(
            train_smiles=s.train_smiles[:300],
            val_smiles=s.val_smiles[:100],
            test_smiles=s.test_smiles,
            provenance={**s.provenance, "SMOKE_TEST_TRUNCATED": True},
        )

    gmref.load_reference_split = _small


experiment.run_if_main()
