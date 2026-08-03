"""
Unconditional DeFoG on MOSES, under the E1 evaluation protocol.

Third sibling of ``training__zinc_e1.py`` / ``training__guacamol_e1.py``. Unlike
those two there is no prior MOSES run to supersede -- ``data/moses/`` did not
exist and no experiment referenced it, so this is the dataset's first
protocol-conforming path.

What it fixes relative to ``src/datasets/moses_dataset.py``:

  split naming  that loader maps MOSES *test* to "val" and *test_scaffolds* to
                "test". So its "val" is an official held-out set (tuning on it
                is tuning on test), and the two held-out sets lose their
                identities -- which the protocol needs, because FCD is reported
                against BOTH. Here they stay named and separate, and validation
                is carved out of TRAIN.
  vocabulary    frozen 8 types in the legacy channel order.

Representation is AROMATIC (4 bond types, no kekulization) as GuacaMol and
unlike ZINC, with implicit hydrogens. ``configs/dataset/moses.yaml`` sets
``filter: False``, so unlike GuacaMol no round-trip filter is applied -- also
recorded.

MOSES molecules are 8-27 heavy atoms, far smaller than ZINC (38) or GuacaMol
(72). Since dense batches pad to the largest molecule and the edge tensor goes
as n^2, MOSES should tolerate a much larger batch than GuacaMol's 128 -- but
that is for a throughput probe to establish, not for this docstring to assume.

EPOCHS is the cosine horizon, fixed across chained links; MAX_TIME_HOURS cuts
each link via PerLinkTimer (per-link, NOT cumulative -- see that class).

Usage:
    python experiments/training__moses_e1.py --__TESTING__ True
    python experiments/training__moses_e1.py --BATCH_SIZE 256 --MAX_TIME_HOURS 10.0 \
        --CKPT_DIR "'ckpts/moses_e1_seed42'" --SKIP_FINAL_EVAL True
"""
import hashlib
import os

import torch
import pytorch_lightning as pl
from rdkit import RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog.core import (  # noqa: E402
    DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback, EMACallback,
    BestValLossCheckpoint, PerLinkTimer,
)
from defog.data import moses_reference as mref  # noqa: E402
from defog.domains import MoleculeDomain  # noqa: E402
from defog.domains.molecule import build_encoders, validity_report  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================
DATA_ROOT: str = os.path.join(_PROJECT_DIR, "data", "moses")
ALLOW_HASH_MISMATCH: bool = False
VAL_SIZE: int = 5000
SPLIT_SEED: int = 42

# --- Representation (protocol trap 6) ---
# :param REPRESENTATION: Selects the graph vocabulary and the kekulize flag
#     together, from defog.data.moses_reference.REPRESENTATIONS.
#
#     "aromatic_v1"  -- 8 atom types (incl. a never-used H), aromatic bonds.
#                       What every MOSES artifact before 2026-08-03 used.
#                       Default, so those checkpoints stay reproducible.
#     "kekulized_v2" -- 7 atom types (H dropped), kekulized bonds.
#
#     The case for v2 is measured, not aesthetic. On the v1 base, 118 of 120
#     hard validity failures are kekulization errors and exactly one is a
#     valence error. An AROMATIC bond class is a promise about the whole ring
#     system that RDKit checks by kekulizing, and the model makes it per-edge
#     without being able to keep it. Dropping the class makes that failure
#     impossible by construction; ZINC trains kekulized and reaches ~0.99
#     validity against MOSES's ~0.90.
#
#     Changing this changes the channel count, so a checkpoint only decodes
#     correctly under the representation it was trained with. The value is
#     recorded in provenance and the model dims are asserted against it below.
REPRESENTATION: str = "aromatic_v1"

# :param GRAPH_CACHE_DIR:
#     1.58M molecules is more than GuacaMol; re-encoding on every chained link
#     would be pure waste. Keyed by source hash + vocabulary so a stale cache
#     cannot survive a change to either.
GRAPH_CACHE_DIR: str = os.path.join(_PROJECT_DIR, "data", "moses", "_graph_cache")

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
EPOCHS: int = 100               # cosine horizon; set from the probe + budget
BATCH_SIZE: int = 256           # provisional until the MOSES probe runs
LEARNING_RATE: float = 2e-4
LR_SCHEDULER: str = "cosine"
LR_MIN: float = 1e-6
WEIGHT_DECAY: float = 1e-5
LAMBDA_EDGE: float = 5.0
TRAIN_TIME_DISTORTION: str = "polydec"
EMA_DECAY: float = 0.9999
NUM_WORKERS: int = 8

MOLECULAR_FEATURES: bool = True
ATOM_VALENCY: dict = dict(mref.ATOM_VALENCY)
ATOM_WEIGHT_TABLE: dict = dict(mref.ATOM_WEIGHT)
MAX_ATOM_WEIGHT: float = mref.MAX_ATOM_WEIGHT   # 350, per MOSESinfos

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

# :param NUM_EVAL_SAMPLES:
#     configs/experiment/moses.yaml uses 25,000 -- a third distinct convention
#     (ZINC 10,000, GuacaMol 18,000). FCD scales roughly as 1/n, so these are
#     NOT interchangeable and the number must be reported (trap 4).
NUM_EVAL_SAMPLES: int = 25000
EVAL_CHUNK: int = 250
SKIP_FINAL_EVAL: bool = False

SAMPLE_VIS_EVERY_K: int = 5
GEN_PROBE_EVERY_K: int = 5
GEN_PROBE_SAMPLES: int = 64
GEN_PROBE_STEPS: int = 100

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


def _cache_path(cache_dir, provenance, rep):
    """Cache key covers the vocabulary AND the kekulize flag.

    Both belong in the key: a kekulized cache and an aromatic cache of the same
    molecules are different tensors, and reusing one for the other would train
    on silently mis-encoded data.
    """
    key = "|".join([provenance.get("train_sha256", "?"),
                    ",".join(rep.atom_types), ",".join(rep.bond_types),
                    f"kekulize={rep.kekulize}"])
    return os.path.join(cache_dir,
                        "moses_graphs_%s.pt" % hashlib.sha256(key.encode()).hexdigest()[:16])


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log("MOSES UNCONDITIONAL -- E1 protocol")
    pl.seed_everything(e.SEED, workers=True)

    rep = mref.get_representation(e.REPRESENTATION)
    e.log(f"representation '{rep.name}': {len(rep.atom_types)} atom types "
          f"{rep.atom_types}, bonds {rep.bond_types}, kekulize={rep.kekulize}")
    if rep.note:
        e.log(f"  ({rep.note})")

    # -- Official split -----------------------------------------------------
    split = mref.load_reference_split(
        e.DATA_ROOT, val_size=e.VAL_SIZE, split_seed=e.SPLIT_SEED,
        allow_hash_mismatch=e.ALLOW_HASH_MISMATCH,
    )
    e.log(split.summary())
    e["provenance/split"] = split.provenance

    # -- Encode (cached) ----------------------------------------------------
    # Neither test nor test_scaffolds is encoded or touched here; both are
    # hashed and counted by the loader and reserved for one evaluation pass.
    cache_file = None
    if e.GRAPH_CACHE_DIR:
        os.makedirs(e.GRAPH_CACHE_DIR, exist_ok=True)
        cache_file = _cache_path(e.GRAPH_CACHE_DIR, split.provenance, rep)

    if cache_file and os.path.exists(cache_file):
        e.log(f"loading encoded graphs from cache {os.path.basename(cache_file)}")
        blob = torch.load(cache_file, weights_only=False)
        train_graphs, train_smiles = blob["train_graphs"], blob["train_smiles"]
        val_graphs, n_skip = blob["val_graphs"], blob["n_skipped"]
    else:
        e.log("encoding train split (slow; will be cached)")
        train_graphs, train_smiles, n_skip = mref.build_graphs(
            split.train_smiles, representation=rep, progress=True)
        val_graphs, _, _ = mref.build_graphs(split.val_smiles, representation=rep)
        if cache_file:
            tmp = cache_file + ".partial"
            torch.save({"train_graphs": train_graphs, "train_smiles": train_smiles,
                        "val_graphs": val_graphs, "n_skipped": n_skip}, tmp)
            os.replace(tmp, cache_file)  # rename only when complete
            e.log(f"cached -> {os.path.basename(cache_file)}")

    e.log(f"train graphs {len(train_graphs)} (skipped {n_skip}) | val {len(val_graphs)}")
    if n_skip > 0.001 * max(1, len(split.train_smiles)):
        e.log(f"WARNING: {n_skip} molecules failed to encode; expected ~0 under the "
              f"'{rep.name}' vocabulary.")
    e["provenance/encoding"] = {
        "representation": rep.name, "representation_note": rep.note,
        "atom_types": list(rep.atom_types), "bond_types": list(rep.bond_types),
        "aromatic": rep.aromatic, "kekulize": rep.kekulize,
        "filter_roundtrip": False,   # configs/dataset/moses.yaml: filter: False
        "n_train_graphs": len(train_graphs), "n_val_graphs": len(val_graphs),
        "n_skipped": n_skip,
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
        atom_valencies=[e.ATOM_VALENCY[a] for a in rep.atom_types],
        atom_weights=[e.ATOM_WEIGHT_TABLE[a] for a in rep.atom_types],
        max_atom_weight=e.MAX_ATOM_WEIGHT,
        lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY,
        lambda_edge=e.LAMBDA_EDGE, train_time_distortion=e.TRAIN_TIME_DISTORTION,
        lr_scheduler=e.LR_SCHEDULER, lr_min=e.LR_MIN,
        sample_steps=e.SAMPLE_STEPS, eta=e.ETA, omega=e.OMEGA,
        sample_time_distortion=e.SAMPLE_TIME_DISTORTION,
    )
    # The model infers its channel counts from the data, so a mismatch here
    # means the encoded graphs do not match the declared vocabulary. That would
    # not raise anywhere downstream -- it would just decode to the wrong atoms
    # for the rest of the checkpoint's life -- so assert it while it is cheap.
    if not rep.matches_model(model):
        raise RuntimeError(
            f"model classes {getattr(model, 'output_dims', {})} do not match "
            f"representation '{rep.name}' ({len(rep.atom_types)} atom types, "
            f"{len(rep.bond_types)} bond types + 1 no-bond). A stale graph cache "
            f"is the usual cause.")
    e.log(f"representation check OK: model dims agree with '{rep.name}'")

    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"Model params: {e['model/num_params']:,}")
    e["provenance/size_distribution"] = {
        "type": type(model.default_size_dist).__name__,
        "source": "train split node-count histogram",
        "max_size": int(model.default_size_dist.max_size),
    }

    # -- Train --------------------------------------------------------------
    _, atom_decoder, _, bond_decoder = rep.encoders()

    def probe_metrics(samples):
        rep = validity_report(samples, atom_decoder, bond_decoder,
                              reference_smiles=train_smiles)
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
        domain=MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_smiles),
        figure_callback=lambda fig: e.track("samples", fig),
    )
    best_ckpt = BestValLossCheckpoint(
        checkpoint_dir=e.CKPT_DIR if e.CKPT_DIR else e.path, monitor="val/loss")

    callbacks = [monitor, sampler, best_ckpt]
    if e.EMA_DECAY and e.EMA_DECAY > 0:
        callbacks = [EMACallback(decay=e.EMA_DECAY)] + callbacks
        e.log(f"EMA enabled (decay={e.EMA_DECAY})")

    # PerLinkTimer, NOT Trainer(max_time=...): Lightning's Timer restores its
    # elapsed time from the checkpoint, making the budget cumulative over the
    # chain and leaving later links with nothing.
    if e.MAX_TIME_HOURS:
        hrs = int(e.MAX_TIME_HOURS)
        mins = int(round((e.MAX_TIME_HOURS - hrs) * 60))
        callbacks.append(PerLinkTimer(duration={"hours": hrs, "minutes": mins}))
        e.log(f"PerLinkTimer = {hrs}h{mins:02d}m for THIS link "
              f"(per-link, not cumulative across the chain)")

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
        max_epochs=e.EPOCHS, accelerator="auto", devices=1,
        enable_progress_bar=True, enable_checkpointing=enable_ckpt, logger=False,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=resume_path)
    e.log(f"Saved final model to {model.save(os.path.join(e.path, 'model'))}")
    e["training/best_val_loss"] = best_ckpt.best
    e["training/best_val_epoch"] = best_ckpt.best_epoch
    e["training/epochs_completed"] = int(trainer.current_epoch)
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

    report = validity_report(samples, atom_decoder, bond_decoder,
                             reference_smiles=train_smiles)
    gen_smiles = report.pop("smiles")
    e.commit_json("generated_smiles.json", gen_smiles)

    e.log(f"validity (relaxed, largest frag) = {report['validity_relaxed_largest_frag']:.4f}")
    e.log(f"validity (strict, no correction) = {report['validity_strict_largest_frag']:.4f}")
    e.log(f"cumulative: V={report['v']:.4f} V.U.={report['v_u']:.4f} "
          f"V.U.N.={report.get('v_u_n', float('nan')):.4f}")
    e.log("MOSES Filters/SNN/Frag/Scaf and FCD-vs-test / FCD-vs-test_scaffolds "
          "are computed separately: scripts/e1_metrics.py --dataset moses "
          "--reference <test> --reference-scaffolds <test_scaffolds>")

    e.commit_json("e1_report.json", {
        "metrics": report,
        "provenance": {
            "split": split.provenance,
            "encoding": e["provenance/encoding"],
            "size_distribution": e["provenance/size_distribution"],
            "sampling": {"steps": e.SAMPLE_STEPS, "eta": e.ETA, "omega": e.OMEGA,
                         "time_distortion": e.SAMPLE_TIME_DISTORTION,
                         "n_samples": len(samples), "seed": e.SEED,
                         "frozen": e.SAMPLING_CONFIG_FROZEN},
            "recipe": {"n_layers": e.N_LAYERS, "hidden_dim": e.HIDDEN_DIM,
                       "epochs_horizon": e.EPOCHS,
                       "epochs_completed": e["training/epochs_completed"],
                       "batch_size": e.BATCH_SIZE, "lr": e.LEARNING_RATE},
        },
    })
    e.log("Done. NOT a table row until the sampling config is swept on validation.")


@experiment.testing
def testing(e: Experiment):
    if e.EPOCHS == 100:
        e.EPOCHS = 2
    e.BATCH_SIZE = 8
    e.NUM_WORKERS = 0
    e.VAL_SIZE = 100
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

    _real = mref.load_reference_split

    def _small(*a, **kw):
        s = _real(*a, **kw)
        return mref.MosesReferenceSplit(
            train_smiles=s.train_smiles[:300],
            val_smiles=s.val_smiles[:100],
            test_smiles=s.test_smiles,
            test_scaffolds_smiles=s.test_scaffolds_smiles,
            provenance={**s.provenance, "SMOKE_TEST_TRUNCATED": True},
        )

    mref.load_reference_split = _small


experiment.run_if_main()
