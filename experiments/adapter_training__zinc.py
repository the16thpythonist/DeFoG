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
from rdkit.Chem import Crippen, Descriptors, QED
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, AdaLNAdapter, AdapterModule, AdapterComposition, ConditionBranch,
    AdaptedSampler, Sampler,
)

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Synthetic accessibility, from RDKit's contrib SA_Score. Not importable by default --
# it lives outside the installed package tree -- so the path is added explicitly rather
# than left to fail at first use, hours into a training run.
def _load_sascorer():
    import os, sys
    from rdkit import RDConfig
    p = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if p not in sys.path:
        sys.path.append(p)
    import sascorer
    return sascorer


_SASCORER = None


def _sa_score(m):
    """SA score, roughly 1 (easy) to 10 (hard). Unlike logp/tpsa it is a heuristic over
    fragment frequencies plus complexity penalties, so it is bounded in practice to about
    [1, 8] on drug-like molecules and is NOT symmetric -- most of ZINC sits near 2-3."""
    global _SASCORER
    if _SASCORER is None:
        _SASCORER = _load_sascorer()
    return float(_SASCORER.calculateScore(m))


PROP_FNS = {"logp": lambda m: float(Crippen.MolLogP(m)),
            "clogp": lambda m: float(Crippen.MolLogP(m)),   # alias; same Crippen estimate
            "tpsa": lambda m: float(Descriptors.TPSA(m)),
            "sascore": _sa_score,
            # QED is bounded in [0, 1] and its mass sits mid-range, unlike logp and
            # tpsa which are unbounded and roughly symmetric. That matters twice: the
            # conditioning normaliser sees a much narrower spread, and a low MAE can be
            # earned by regressing to the middle. Judge it across the range, not on the
            # mean. For scale, FreeGress reports an UNCONDITIONAL MAE of 0.15 on this
            # property and its own best at 0.04 -- the useful band is narrow.
            "qed": lambda m: float(QED.qed(m))}


def _vocabulary(name: str):
    """(atom_types, bond_types, kekulize, source) for a named base vocabulary."""
    if name == "legacy_aromatic":
        return (["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"],
                ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"], False, "csv")
    if name == "e1_kekulized":
        from defog.data import zinc_reference as zref

        return (list(zref.ATOM_TYPES), list(zref.BOND_TYPES), True, "reference_split")
    raise ValueError(f"unknown VOCABULARY {name!r}; "
                     f"have 'legacy_aromatic', 'e1_kekulized'")

# ============================================================================
# Parameters
# ============================================================================
# :param VOCABULARY: Which frozen base this adapter is being trained for. It
#     bundles the atom order, the bond set, the kekulize flag and the SMILES
#     source, because those four must agree -- an adapter trained against a
#     vocabulary the base does not use is not wrong-ish, it is meaningless, and
#     nothing in the training loop would say so.
#
#     "legacy_aromatic"  the original ZINC base (zinc_uncond_4e-4_connectivity),
#                        9 atoms in frequency order, AROMATIC bonds, loose CSV.
#                        The default, so every adapter shipped before
#                        2026-08-04 (logp, tpsa, qed, sascore, fingerprint,
#                        logd) stays reproducible.
#     "e1_kekulized"     the E1 / RL lineage: zinc_reference's 9 atoms in ITS
#                        order, kekulized bonds, SMILES from the hash-pinned
#                        reference split. Note the atom ORDER differs between
#                        the two (C N O S F Cl Br I P against C N O F P S Cl Br
#                        I), so this is not merely a bond-set change.
VOCABULARY: str = "legacy_aromatic"

# :param PROPERTY_FROM: Which molecule the conditioning label describes.
#
#     "source"   the input SMILES, exactly as it appears in the dataset.
#     "decoded"  the molecule the GRAPH actually represents, i.e. encode ->
#                decode -> measure.
#
#     These differ because a DeFoG graph stores atoms and bonds but NOT formal
#     charges or stereochemistry. 33% of ZINC carries a formal charge, and
#     protonated amines and carboxylates are precisely what make a molecule
#     low-logP -- so stripping the charge moves clogP up, and only at the low
#     end. Measured on ZINC train:
#
#         5th pct:   source -0.90   graph +0.73   error +1.64   92.5% charged
#         95th pct:  source +4.90   graph +4.90   error -0.00    7.0% charged
#
#     Under "source" the adapter is trained on (graph, label) pairs where the
#     low-end labels describe something the graph is not, so asking it for -0.1
#     yields ~1.4. That is not a broken adapter -- the training graphs labelled
#     -0.1 genuinely are ~+0.7 -- it is a miscalibrated target scale.
#
#     "decoded" does NOT give the model new reach. It cannot generate clogP
#     -0.1; that region needs charges the representation cannot express. What
#     it does is make the declared range honest (~0.8 to 4.9 rather than -0.1
#     to 4.5), so a requested target means what it says.
#
#     Default is "source" so the six already-shipped adapters stay reproducible.
PROPERTY_FROM: str = "source"

# Only used by VOCABULARY="legacy_aromatic"; "e1_kekulized" reads the
# hash-pinned reference split instead.
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"

# ATOM_TYPES / BOND_TYPES are deliberately NOT parameters. They must match the
# frozen base's node-class order exactly -- a mismatch trains the adapter against
# classes that decode to different elements, which converges fine and produces a
# useless adapter. Set VOCABULARY instead; it supplies both, together with the
# kekulize flag and SMILES source that have to agree with them. Leaving them here
# as overridable values would let a caller set one and silently contradict the
# base.

BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")

PROPERTY: str = "logp"          # logp | clogp | tpsa | qed | sascore

# --- Adapter architecture ---
H_HIDDEN: int = 256
TIME_CONDITIONED: bool = True
STREAMS: list = ["X", "E", "y"]
INTERIOR_FF: bool = False        # L4: pre-FFN adaLN-Zero FiLM on X,E
INTERIOR_ATTN: bool = False      # L10: condition e_mul (edge->attention logits)
L10_LR_SCALE: float = 0.3        # smaller LR on the L10 heads (validity guard)

# --- Condition path: Fourier bands on the target ---
# Without this the trunk sees the property as ONE raw float while the flow-time gets a
# 64-dim sinusoidal embedding. See AdaLNAdapter for the measured bandwidth table; 3 is
# the ceiling that keeps neighbouring targets correlated enough to interpolate between.
# Set 0 for the pre-2026-08-28 FiLM-only adapter.
COND_FOURIER: int = 3

# --- Node -> condition cross-attention (0 tokens = off) ---
# Each atom queries the condition instead of every atom receiving the same broadcast
# FiLM correction. Nodes only: edges and the global vector keep the FiLM path. This is
# the half that carries the effect: attribution at 60 epochs was -0.153 MAE for
# cross-attention against -0.024 for the Fourier bands.
XATTN_TOKENS: int = 64
XATTN_DIM: int = 128
XATTN_HEADS: int = 16

# --- Training ---
# THESE THREE DEFAULTS TOGETHER ARE THE MEASURED RECIPE (logP MAE 0.3250). Changing any
# one of them in isolation does not reproduce it:
#   * EPOCHS. The old default of 20 was badly under-trained for this architecture --
#     0.4299 at 20 epochs against 0.3596 at 40. ep40->60 bought only -0.0095, so 60-80 is
#     the saturated region and 80 is what the shipped arms ran.
#   * LEARNING_RATE. 4e-4, NOT the 2e-4 this module used to default to. Every measured
#     number, including the FiLM baseline the architecture is compared against, is a
#     4e-4 arm; at 2e-4 a bigger adapter trains at half the LR of the thing it is
#     supposed to beat, which is a comparison of two changes at once.
#   * MAX_TIME_HOURS bounds trainer.fit and NOTHING else, so it must clear the epoch
#     count or training stops early and the run still reports a checkpoint.
EPOCHS: int = 80
BATCH_SIZE: int = 24
LEARNING_RATE: float = 4e-4
COND_DROP_PROB: float = 0.0      # uncond branch IS the frozen base -> dropout not needed
MAX_TIME_HOURS: float = 20.0

# --- Sampling / evaluation ---
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
TARGET_PERCENTILES: list = [5, 95]
LEVEL_NAMES: list = ["low", "high"]
# CORRECTED 2026-08-17. This list used to stop at 1.0, justified by "with w > 1 the
# unconditional coefficient goes negative, the blend extrapolates past the conditional,
# and _stabilize's clamp silently drops rates -- empirically w=2 degrades rather than
# steers harder." Every word of that is true OF RATE-SPACE BLENDING, which was the only
# blend space at the time. It is not a property of the task: the composition now blends
# clean-graph marginals by default (AdapterComposition.blend_space="prob"), where w=2 is
# the measured OPTIMUM -- logP MAE 0.6410 at w=1 vs 0.5420 at w=2, and the degradation
# only resumes past w~2.5. The grid therefore has to straddle 2.0 or it cannot find the
# operating point the adapter actually ships at.
GUIDANCE_WEIGHTS: list = [1.0, 1.5, 2.0, 2.5, 3.0]
N_PER_TARGET: int = 128
N_BASELINE: int = 256
EVAL_CHUNK: int = 32
COMPOSE_MODE: str = "product"    # single branch: product == mean

# --- mid-training probe ---
PROBE_EVERY_K: int = 5
PROBE_N: int = 32
PROBE_STEPS: int = 100
# Probe at the weight the adapter actually SHIPS at. It was 1.0 because rate-space
# blending made anything above that unusable; probing at 1.0 while shipping at 2.0 is
# how the capacity ladder ended up comparing an adapter at w=1 against a jointly
# trained model at w=2 and reading the difference as a capacity gap.
PROBE_WEIGHT: float = 2.0

# --- periodic checkpointing ---
# 0 disables. WHY THIS EXISTS: the C_long / D_attn arms of the capacity ladder were
# killed by the SLURM wall after ~50 epochs and the adapter is only written after
# `trainer.fit` returns, so both arms were lost entirely -- no final eval, no weights,
# and the one hypothesis they were testing ("more epochs") stayed open. With this set,
# a kill costs the tail since the last multiple of K rather than everything.
CKPT_EVERY_K: int = 0

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


class AdapterEpochCheckpoint(pl.Callback):
    """Write the adapter to ``{property}_adapter_ep{N}`` every K epochs.

    Two jobs, and the second is the one that matters. It makes a wall-clock kill cost
    only the tail -- but it also turns "does training longer help?" into a question that
    can be answered by EVALUATION rather than by a trend line through mid-training
    probes. That distinction is load-bearing: on the two capacity-ladder arms that had
    both, the probe-vs-final-eval offsets went in OPPOSITE directions (+0.112 and
    -0.131, on a quantity of ~0.35), so a probe cannot stand in for a real eval and a
    trend fitted through probes cannot be believed. Saved checkpoints can be evaluated
    properly after the fact, at whatever weight and sample count the protocol demands.
    """

    def __init__(self, e, adapter, every_k: int, prop: str):
        super().__init__()
        self.e, self.adapter, self.every_k, self.prop = e, adapter, int(every_k), prop
        self.saved = []

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.every_k:
            return
        ep = int(trainer.current_epoch) + 1          # 1-based: "after N epochs of training"
        if ep % self.every_k:
            return
        try:
            path = self.adapter.save(os.path.join(self.e.path, f"{self.prop}_adapter_ep{ep}"))
            self.saved.append({"epoch": ep, "path": path})
            self.e["checkpoints"] = self.saved
            self.e.log(f"[epoch {ep}] checkpoint -> {path}")
        except Exception as ex:                       # never let a save kill the run
            self.e.log(f"[epoch {ep}] CHECKPOINT FAILED (non-fatal): {ex}")


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

    if e.PROPERTY_FROM not in ("source", "decoded"):
        raise ValueError(f"PROPERTY_FROM must be 'source' or 'decoded', "
                         f"got {e.PROPERTY_FROM!r}")
    atom_types, bond_types, kekulize, source = _vocabulary(e.VOCABULARY)
    e.log(f"vocabulary '{e.VOCABULARY}': {len(atom_types)} atoms {atom_types}")
    e.log(f"  bonds={bond_types} kekulize={kekulize} smiles_source={source}")
    e.log(f"  property_from={e.PROPERTY_FROM}"
          + ("  (label describes the GRAPH, charges dropped)"
             if e.PROPERTY_FROM == "decoded"
             else "  (label describes the SOURCE SMILES, charges included)"))
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, bond_types)

    if source == "reference_split":
        # The hash-pinned split, not a loose CSV: an adapter trained on molecules
        # the base never saw would be learning to steer a distribution shift as
        # well as the property.
        from defog.data import zinc_reference as zref

        smiles_iter = zref.load_reference_split().train_smiles
    else:
        smiles_iter = pd.read_csv(e.CSV_PATH)[e.SMILES_COLUMN]
    e.log(f"source molecules: {len(smiles_iter)}")

    graphs, vals = [], []
    n_skipped = 0
    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_skipped += 1
            continue
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder, kekulize=kekulize)
        if data is None:
            n_skipped += 1
            continue
        # Label the SOURCE molecule or the one the graph actually represents.
        # See PROPERTY_FROM: the two differ by up to 1.6 log units at the low
        # end of clogP, because the graph drops formal charges.
        target_mol = mol
        if e.PROPERTY_FROM == "decoded":
            decoded = pyg_data_to_mol(data, atom_decoder, bond_decoder)
            smi_back = mol_to_smiles(decoded) if decoded is not None else None
            target_mol = Chem.MolFromSmiles(smi_back) if smi_back else None
            if target_mol is None:
                # Cannot measure what the graph is, so cannot label it honestly.
                n_skipped += 1
                continue
        try:
            v = prop_fn(target_mol)
        except Exception:
            continue
        data.cond = torch.tensor([[v]], dtype=torch.float)   # (1,1) RAW scalar condition
        graphs.append(data)
        vals.append(v)
    vals = np.asarray(vals)
    cond_mean, cond_std = float(vals.mean()), float(vals.std())
    e.log(f"{len(graphs)} graphs (skipped {n_skipped}); "
          f"{e.PROPERTY} mean={cond_mean:.2f} std={cond_std:.2f}")
    if n_skipped > 0.02 * max(1, n_skipped + len(graphs)):
        e.log(f"WARNING: {n_skipped} molecules failed to encode. Under a matching "
              f"vocabulary this should be near zero -- check VOCABULARY.")
    e["encoding"] = {"vocabulary": e.VOCABULARY, "atom_types": atom_types,
                     "bond_types": bond_types, "kekulize": kekulize,
                     "smiles_source": source, "n_graphs": len(graphs),
                     "n_skipped": n_skipped, "property_from": e.PROPERTY_FROM}

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(graphs, batch_size=e.BATCH_SIZE, shuffle=True)

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0, f"expected unconditional base, cond_dim={base.cond_dim}"

    # The adapter modulates the base's own channels, so a vocabulary mismatch
    # would train it against classes that mean something else. That produces a
    # converging loss and a useless adapter, with nothing in between to notice.
    from defog.data import vocabulary as vocab_check

    e.log(vocab_check.check_model(base, atom_types, bond_types,
                                  what=f"base {e.BASE_CKPT}"))
    adapter = AdaLNAdapter.for_base(
        base, cond_dim=1, hidden=e.H_HIDDEN, time_conditioned=e.TIME_CONDITIONED,
        streams=tuple(e.STREAMS), cond_mean=[cond_mean], cond_std=[cond_std],
        interior_ff=e.INTERIOR_FF, interior_attn=e.INTERIOR_ATTN,
        cond_fourier=e.COND_FOURIER, xattn_tokens=e.XATTN_TOKENS,
        xattn_dim=e.XATTN_DIM, xattn_heads=e.XATTN_HEADS,
        name=f"{e.PROPERTY}_adapter", cond_type=e.PROPERTY)
    e["adapter/num_params"] = sum(p.numel() for p in adapter.parameters())
    e.log(f"adapter: {e['adapter/num_params']:,} params (interior_ff={e.INTERIOR_FF} interior_attn={e.INTERIOR_ATTN} "
          f"cond_fourier={e.COND_FOURIER} xattn_tokens={e.XATTN_TOKENS}; "
          f"base {sum(p.numel() for p in base.parameters()):,} frozen)")
    # Condition dropout with cross-attention needs Modulation.bypass_rows to silence the
    # cross-attention delta as well as the FiLM gates. It does -- but say so out loud,
    # because "unconditional branch is still conditioned" is invisible in the loss.
    if e.XATTN_TOKENS and e.COND_DROP_PROB:
        e.log(f"note: cond_drop_prob={e.COND_DROP_PROB} with cross-attention; "
              f"bypass_rows scales the xattn delta to zero on dropped rows "
              f"(tests/test_adapter.py::test_xattn_bypass_rows_silences_it)")

    module = AdapterModule(base, adapter, cond_attr="cond", cond_drop_prob=e.COND_DROP_PROB,
                           lr=e.LEARNING_RATE, l10_lr_scale=e.L10_LR_SCALE)

    targets = dict(zip(e.LEVEL_NAMES, [float(x) for x in np.percentile(vals, e.TARGET_PERCENTILES)]))
    e["eval/targets"] = targets
    e.log(f"targets ({e.PROPERTY}): {targets}")
    probe = AdapterPropProbe(e, base, adapter, atom_decoder, bond_decoder, prop_fn, targets,
                             e.PROBE_EVERY_K, e.PROBE_N, e.PROBE_STEPS, e.PROBE_WEIGHT, e.COMPOSE_MODE,
                             e.ETA, e.OMEGA, e.TIME_DISTORTION, e.EVAL_CHUNK)
    callbacks = [probe]
    if e.CKPT_EVERY_K:
        callbacks.append(AdapterEpochCheckpoint(e, adapter, e.CKPT_EVERY_K, e.PROPERTY))
    trainer = pl.Trainer(max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS}, accelerator="auto",
                         devices=1, enable_progress_bar=True, enable_checkpointing=False, logger=False,
                         gradient_clip_val=1.0, callbacks=callbacks)
    e.log(f"Training adapter: epochs<={e.EPOCHS} max_time={e.MAX_TIME_HOURS}h batch={e.BATCH_SIZE} "
          f"LR={e.LEARNING_RATE} ckpt_every={e.CKPT_EVERY_K or 'off'}")
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
    e.CKPT_EVERY_K = 1        # exercise the checkpoint path; it is the fix for a real loss
    e.PROBE_N = 6
    e.N_PER_TARGET = 8
    e.N_BASELINE = 8
    e.EVAL_CHUNK = 8
    e.GUIDANCE_WEIGHTS = [2.0]
    df = pd.read_csv(e.CSV_PATH).head(300)
    smoke = os.path.join(folder_path(__file__), "_adapter_smoke.csv")
    df.to_csv(smoke, index=False)
    e.CSV_PATH = smoke
    # BASE_CKPT is left alone: a smoke test must exercise the base the caller
    # actually passed, since the vocabulary/base agreement check is one of the
    # things worth smoke-testing.
    if e.VOCABULARY == "legacy_aromatic" and not os.path.exists(e.BASE_CKPT):
        e.BASE_CKPT = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")

    # The reference-split path ignores CSV_PATH, so truncate the split too or a
    # "smoke" run encodes all 224k training molecules.
    from defog.data import zinc_reference as _zref

    _real = _zref.load_reference_split

    def _small(*a, **kw):
        s = _real(*a, **kw)
        return _zref.ZincReferenceSplit(
            train_smiles=s.train_smiles[:300],
            val_smiles=s.val_smiles[:50],
            test_smiles=s.test_smiles[:50],
            provenance={**s.provenance, "TRUNCATED_FOR_SMOKE_TEST": True},
        )

    _zref.load_reference_split = _small


experiment.run_if_main()
