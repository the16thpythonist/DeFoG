"""
Train ONE frozen-base CFG-adapter for a single scalar molecular property on ZINC
250k. The base (unconditional ZINC model) is FROZEN; only the AdaLN/FiLM adapter
trains, with the base's own denoising CE loss (a conditional denoiser p(x1|x_t,c)).

Used to validate the adapter mechanism on properties that traditional direct-CFG
already nails (logP, TPSA). Two arms per property (different LRs) run on JUPITER;
the two trained adapters are then COMPOSED in adapter_compose_2d__zinc.py.

End-of-run eval (single-property steering, before composition): steer to the 5th
and 95th percentile of the property; measure achieved-vs-target and MAE over a
guidance-scale sweep, plot the property distribution (gray dataset + generated).

Usage:
    python experiments/adapter_training__zinc.py --PROPERTY logp --__TESTING__ True
    python experiments/adapter_training__zinc.py --PROPERTY tpsa --LEARNING_RATE 3e-4
"""
import os
import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, AdaLNAdapter, AdapterModule, AdapterComposition, ConditionBranch,
    AdaptedSampler, Sampler,
)

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROP_FNS = {"logp": lambda m: float(Crippen.MolLogP(m)),
            "tpsa": lambda m: float(Descriptors.TPSA(m))}

# ============================================================================
# Parameters
# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
# MUST match the frozen base's node-class order (frequency-derived on full ZINC);
# do NOT derive from a CSV subset (a smoke subset can miss rare atoms -> class-count
# mismatch with the base). This is the order the ZINC uncond/connectivity base used.
ATOM_TYPES: list = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")

PROPERTY: str = "logp"          # logp | tpsa

# --- Adapter architecture ---
H_HIDDEN: int = 256
TIME_CONDITIONED: bool = True
STREAMS: list = ["X", "E", "y"]

# --- Training ---
EPOCHS: int = 20
BATCH_SIZE: int = 24
LEARNING_RATE: float = 2e-4      # swept per-arm (2 LRs per property)
COND_DROP_PROB: float = 0.0      # uncond branch IS the frozen base -> dropout not needed
MAX_TIME_HOURS: float = 5.0

# --- Sampling / evaluation ---
EVAL_STEPS: int = 250
ETA: float = 5.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
TARGET_PERCENTILES: list = [5, 95]
LEVEL_NAMES: list = ["low", "high"]
GUIDANCE_WEIGHTS: list = [1.0, 2.0, 4.0]
N_PER_TARGET: int = 128
N_BASELINE: int = 256
EVAL_CHUNK: int = 32
COMPOSE_MODE: str = "product"    # single branch: product == mean

# --- mid-training probe ---
PROBE_EVERY_K: int = 5
PROBE_N: int = 32
PROBE_STEPS: int = 100
PROBE_WEIGHT: float = 2.0

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


def derive_atom_types(smiles_list) -> list:
    counts = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for a in mol.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def props_of(samples, atom_decoder, bond_decoder, prop_fn):
    """Decode graph samples -> list of the property for valid molecules."""
    vals = []
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi is not None and Chem.MolFromSmiles(smi) is not None:
            try:
                vals.append(prop_fn(mol))
            except Exception:
                pass
    return np.asarray(vals, dtype=float)


class AdapterPropProbe(pl.Callback):
    """Per-epoch loss log + every-K-epoch steering probe (achieved property mean
    when steering to low/high targets), so training is visible."""

    def __init__(self, e, base, adapter, atom_decoder, bond_decoder, prop_fn,
                 targets, every_k, n, steps, weight, mode, eta, omega, td, chunk):
        super().__init__()
        self.e, self.base, self.adapter = e, base, adapter
        self.ad, self.bd, self.prop_fn = atom_decoder, bond_decoder, prop_fn
        self.targets = targets       # {"low": val, "high": val}
        self.every_k, self.n, self.steps, self.weight = every_k, n, steps, weight
        self.mode, self.eta, self.omega, self.td, self.chunk = mode, eta, omega, td, chunk

    def on_train_epoch_end(self, trainer, pl_module):
        ep = int(trainer.current_epoch)
        loss = trainer.callback_metrics.get("adapter/loss_epoch", trainer.callback_metrics.get("adapter/loss"))
        self.e.log(f"[epoch {ep}] adapter/loss={float(loss):.4f}" if loss is not None else f"[epoch {ep}] done")
        if not self.every_k or (ep + 1) % self.every_k != 0:
            return
        try:
            self._probe(pl_module, ep)
        except Exception as ex:
            self.e.log(f"[epoch {ep}] PROBE failed (non-fatal): {ex}")

    @torch.no_grad()
    def _probe(self, pl_module, ep):
        device = pl_module.device
        out = {}
        for lvl, tgt in self.targets.items():
            comp = AdapterComposition([ConditionBranch(pl_module.adapter, torch.tensor([tgt]), self.weight)],
                                      base=pl_module.base, mode=self.mode)
            samp = AdaptedSampler(pl_module.base, comp, eta=self.eta, omega=self.omega,
                                  sample_steps=self.steps, time_distortion=self.td)
            samples, rem = [], self.n
            while rem > 0:
                cur = min(self.chunk, rem)
                samples += samp.sample(cur, device=device, show_progress=False)
                rem -= cur
            vals = props_of(samples, self.ad, self.bd, self.prop_fn)
            out[lvl] = (float(vals.mean()) if vals.size else float("nan"), int(vals.size))
        self.e.log(f"[epoch {ep}] PROBE(w={self.weight}) " +
                   "  ".join(f"{lvl}->{tgt:.1f}: achieved={out[lvl][0]:.2f} (n={out[lvl][1]})"
                            for lvl, tgt in self.targets.items()))


# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"ZINC frozen-base CFG-ADAPTER training for property={e.PROPERTY}")
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prop_fn = PROP_FNS[e.PROPERTY]

    df = pd.read_csv(e.CSV_PATH)
    atom_types = e.ATOM_TYPES   # fixed to match the frozen base's node classes
    e.log(f"Atom vocabulary ({len(atom_types)}): {atom_types}")
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, e.BOND_TYPES)

    graphs, vals = [], []
    for smi in df[e.SMILES_COLUMN]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder)
        if data is None:
            continue
        try:
            v = prop_fn(mol)
        except Exception:
            continue
        data.cond = torch.tensor([[v]], dtype=torch.float)   # (1,1) RAW scalar condition
        graphs.append(data)
        vals.append(v)
    vals = np.asarray(vals)
    cond_mean, cond_std = float(vals.mean()), float(vals.std())
    e.log(f"{len(graphs)} graphs; {e.PROPERTY} mean={cond_mean:.2f} std={cond_std:.2f}")

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(graphs, batch_size=e.BATCH_SIZE, shuffle=True)

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0, f"expected unconditional base, cond_dim={base.cond_dim}"
    adapter = AdaLNAdapter.for_base(
        base, cond_dim=1, hidden=e.H_HIDDEN, time_conditioned=e.TIME_CONDITIONED,
        streams=tuple(e.STREAMS), cond_mean=[cond_mean], cond_std=[cond_std],
        name=f"{e.PROPERTY}_adapter", cond_type=e.PROPERTY)
    e["adapter/num_params"] = sum(p.numel() for p in adapter.parameters())
    e.log(f"adapter: {e['adapter/num_params']:,} params (base {sum(p.numel() for p in base.parameters()):,} frozen)")

    module = AdapterModule(base, adapter, cond_attr="cond", cond_drop_prob=e.COND_DROP_PROB, lr=e.LEARNING_RATE)

    targets = dict(zip(e.LEVEL_NAMES, [float(x) for x in np.percentile(vals, e.TARGET_PERCENTILES)]))
    e["eval/targets"] = targets
    e.log(f"targets ({e.PROPERTY}): {targets}")
    probe = AdapterPropProbe(e, base, adapter, atom_decoder, bond_decoder, prop_fn, targets,
                             e.PROBE_EVERY_K, e.PROBE_N, e.PROBE_STEPS, e.PROBE_WEIGHT, e.COMPOSE_MODE,
                             e.ETA, e.OMEGA, e.TIME_DISTORTION, e.EVAL_CHUNK)
    trainer = pl.Trainer(max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS}, accelerator="auto",
                         devices=1, enable_progress_bar=True, enable_checkpointing=False, logger=False,
                         gradient_clip_val=1.0, callbacks=[probe])
    e.log(f"Training adapter: epochs<={e.EPOCHS} max_time={e.MAX_TIME_HOURS}h batch={e.BATCH_SIZE} LR={e.LEARNING_RATE}")
    trainer.fit(module, train_dataloaders=train_loader)

    ckpt = adapter.save(os.path.join(e.path, f"{e.PROPERTY}_adapter"))
    with open(os.path.join(e.path, f"{e.PROPERTY}_adapter_stats.json"), "w") as f:
        json.dump({"property": e.PROPERTY, "mean": cond_mean, "std": cond_std,
                   "targets": targets, "atom_types": atom_types,
                   "percentiles": {str(p): float(np.percentile(vals, p)) for p in [5, 25, 50, 75, 95]}}, f)
    e.log(f"Saved adapter -> {ckpt}")

    # -- Eval: single-property steering (before composition) ---------------
    e.log("=" * 60)
    base = base.to(device).eval()
    adapter = adapter.to(device).eval()

    e.log(f"baseline: sampling {e.N_BASELINE} unconditional")
    base_sampler = Sampler(base, eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS, time_distortion=e.TIME_DISTORTION)
    bsamp, rem = [], e.N_BASELINE
    while rem > 0:
        cur = min(e.EVAL_CHUNK, rem)
        bsamp += base_sampler.sample(cur, device=device, show_progress=False)
        rem -= cur
    base_vals = props_of(bsamp, atom_decoder, bond_decoder, prop_fn)
    e.log(f"baseline {e.PROPERTY} mean={base_vals.mean():.2f} (n={base_vals.size})")

    results = {"property": e.PROPERTY, "baseline_mean": float(base_vals.mean()), "targets": targets, "per_level": {}}
    gen_by = {}
    for lvl, tgt in targets.items():
        results["per_level"][lvl] = {"target": tgt, "per_w": {}}
        for w in e.GUIDANCE_WEIGHTS:
            comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([tgt]), w)], base=base, mode=e.COMPOSE_MODE)
            samp = AdaptedSampler(base, comp, eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS,
                                  time_distortion=e.TIME_DISTORTION)
            gsamp, rem = [], e.N_PER_TARGET
            while rem > 0:
                cur = min(e.EVAL_CHUNK, rem)
                gsamp += samp.sample(cur, device=device, show_progress=False)
                rem -= cur
            gv = props_of(gsamp, atom_decoder, bond_decoder, prop_fn)
            results["per_level"][lvl]["per_w"][str(w)] = {
                "n_valid": int(gv.size), "mean": float(gv.mean()) if gv.size else None,
                "mae": float(np.mean(np.abs(gv - tgt))) if gv.size else None,
            }
            if abs(w - e.PROBE_WEIGHT) < 1e-9:
                gen_by[lvl] = gv
            e.log(f"  {lvl} target={tgt:.1f} w={w}: n={gv.size} mean={gv.mean() if gv.size else float('nan'):.2f} "
                  f"mae={results['per_level'][lvl]['per_w'][str(w)]['mae']}")
    e.commit_json("adapter_steering_metrics.json", results)

    # distribution plot: dataset gray + generated per level (at PROBE_WEIGHT) + target lines
    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = np.percentile(vals, [1, 99])
    bins = np.linspace(lo, hi, 50)
    ax.hist(vals, bins=bins, density=True, color="0.7", label="dataset", zorder=1)
    colors = {"low": "#2c7fb8", "high": "#d95f0e"}
    for lvl, tgt in targets.items():
        gv = gen_by.get(lvl, np.array([]))
        if gv.size:
            ax.hist(gv, bins=bins, density=True, histtype="stepfilled", alpha=0.5,
                    color=colors.get(lvl), label=f"gen {lvl} (mean {gv.mean():.1f})", zorder=2)
        ax.axvline(tgt, ls="--", color=colors.get(lvl), lw=2, label=f"target {lvl}={tgt:.1f}")
    ax.set_xlabel(e.PROPERTY); ax.set_ylabel("density")
    ax.set_title(f"Adapter steering: {e.PROPERTY} (LR={e.LEARNING_RATE}, w={e.PROBE_WEIGHT})")
    ax.legend(fontsize=8); fig.tight_layout()
    e.commit_fig(f"steering_{e.PROPERTY}.png", fig)
    e.log("Done.")


@experiment.testing
def testing(e: Experiment):
    e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.MAX_TIME_HOURS = 0.2
    e.H_HIDDEN = 32
    e.EVAL_STEPS = 5
    e.PROBE_STEPS = 5
    e.PROBE_EVERY_K = 1
    e.PROBE_N = 6
    e.N_PER_TARGET = 8
    e.N_BASELINE = 8
    e.EVAL_CHUNK = 8
    e.GUIDANCE_WEIGHTS = [2.0]
    df = pd.read_csv(e.CSV_PATH).head(300)
    smoke = os.path.join(folder_path(__file__), "_adapter_smoke.csv")
    df.to_csv(smoke, index=False)
    e.CSV_PATH = smoke
    e.BASE_CKPT = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")


experiment.run_if_main()
