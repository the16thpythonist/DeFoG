"""
Train ONE frozen-base AdaLN/FiLM CFG-adapter conditioned on a 128-bit Morgan
FINGERPRINT, ZINC 250k. The base (connectivity-improved unconditional ZINC model)
is FROZEN; only the adapter trains, with the base's own denoising CE loss (a
conditional denoiser p(x1|x_t, fp)).

This is the high-dimensional / holistic test of the frozen-base adapter: the
per-coordinate GUIDANCE adapter and FK were ~null on fingerprints, while the full
CFG-conditional model steered (+0.12 Tanimoto lift). This asks whether the FROZEN
base + adapter matches that (it injects the same conditional-denoiser signal via
FiLM, so it should).

Eval (mirrors cfg_fingerprint / fingerprint_guidance): per held-out target
molecule, condition on its FP, sample via AdaptedSampler at w in {1,2,4}, and
measure Tanimoto(generated, target) vs an unconditional baseline. Size-independent
generation (global size prior), matching those runs.

Usage:
    python experiments/adapter_fingerprint__zinc.py --__TESTING__ True
    python experiments/adapter_fingerprint__zinc.py --LEARNING_RATE 3e-4
"""
import os
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Draw
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles
from defog.core import (
    DeFoGModel, AdaLNAdapter, AdapterModule, AdapterComposition, ConditionBranch,
    AdaptedSampler, Sampler,
)

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================
# :param VOCABULARY: Which frozen base this adapter targets. Bundles atom order,
#     bond set, kekulize flag and SMILES source, because those four must agree
#     -- see adapter_training__zinc.py for the same selector.
#     "legacy_aromatic" is the original ZINC base; "e1_kekulized" is the E1/RL
#     lineage behind molsmith/zinc-kek. The atom ORDER differs between them, not
#     just the bond set, so this is not a cosmetic switch.
VOCABULARY: str = "legacy_aromatic"

# :param FP_FROM: Whether the conditioning fingerprint describes the SOURCE
#     SMILES or the molecule the graph actually is.
#
#     Morgan bits encode formal charges. A DeFoG graph does not store them, and
#     32% of ZINC molecules carry one. Measured over ZINC train, 512-bit r=2,
#     Tanimoto( FP(source), FP(decoded) ):
#
#         neutral molecules (68%)   1.0000   <- identical, no loss whatsoever
#         charged molecules (32%)   0.6813
#         overall                   0.8990
#
#     Stereochemistry does not affect Morgan bits, so unlike clogP the damage is
#     confined entirely to charged molecules -- and for those it is severe.
#     "decoded" makes the fingerprint-to-graph mapping self-consistent at no
#     cost to the 68% that were already exact.
#
#     That 0.899 is also a CEILING on the reported metric: a model reproducing a
#     target graph perfectly still scores only 0.899 against a source-derived
#     target fingerprint. The evaluation below reports against BOTH conventions
#     so the ceiling is visible rather than silently absorbed into the score.
FP_FROM: str = "source"

# Only used by VOCABULARY="legacy_aromatic"; "e1_kekulized" reads the
# hash-pinned reference split.
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"

# ATOM_TYPES / BOND_TYPES are deliberately NOT parameters: they must match the
# frozen base exactly, and a mismatch trains against classes that decode to
# different elements -- which converges fine and produces a useless adapter.
# Set VOCABULARY instead.

BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")

FP_BITS: int = 512      # 512-bit Morgan/ECFP4: far fewer bit collisions than 128 -> more
FP_RADIUS: int = 2      # discriminative fingerprint + a cleaner Tanimoto signal

# :param FP_COUNTS: Condition on how MANY times each substructure occurs, not
#     merely whether it occurs.
#
#     A binary Morgan vector records presence only, which is why it calls hexane
#     and eicosane 0.875 similar -- the same environments, different amounts.
#     The v2.0.0 adapter inherited that blind spot, and it shows: steering
#     quality falls off sharply with target size (corr(heavy atoms, lift) =
#     -0.92 over six held-out targets), and its own analogues include molecules
#     with the reference motif repeated twice scoring 0.705.
#
#     True uses GetHashedMorganFingerprint with **log1p** applied. The transform
#     is not cosmetic: counts are small integers with a heavy tail, the adapter
#     normalises per-bit by mean/std, and on a rare bit a raw count of 3 becomes
#     roughly a ten-sigma input -- which is where FiLM conditioning destabilises.
#
#     MUST match molsmith's FingerprintSpec.counts for the shipped package.
#     Serving an adapter the encoding it was not trained on raises nothing; it
#     just steers badly. molsmith reads the flag from the package for that
#     reason, so the two cannot drift apart once shipped.
FP_COUNTS: bool = False

# --- Adapter architecture ---
# :param H_HIDDEN: Width of the shared trunk that feeds EVERY FiLM head. Without a
#     cond_encoder this is also the narrowest point in the whole conditioning path:
#     the trunk's first layer is Linear(FP_BITS + time_emb, H_HIDDEN), so widening
#     FP_BITS alone does not widen the path. 512 and 1024 bits were both squeezed to
#     256 here, which is the likeliest reason that doubling the fingerprint bought
#     only +0.015 Tanimoto lift.
H_HIDDEN: int = 256
# :param COND_ENCODER: Optional module between the normalized condition and the
#     trunk. None keeps the historical behaviour (condition goes straight in).
#     A dict is a spec resolved through AdaLNAdapter's registry, e.g.
#         {"kind": "mlp", "out_dim": 512, "hidden": 1024, "n_blocks": 2}
#     `in_dim` is filled in from FP_BITS automatically -- passing it here would let
#     it silently disagree with the fingerprint width.
COND_ENCODER: dict = None
TIME_CONDITIONED: bool = True
STREAMS: list = ["X", "E", "y"]
INTERIOR_FF: bool = False        # L4: pre-FFN adaLN-Zero FiLM on X,E
INTERIOR_ATTN: bool = False      # L10: condition e_mul (edge->attention logits)
L10_LR_SCALE: float = 0.3        # smaller LR on the L10 heads (validity guard)

# --- Training (8h wall on JUPITER; base frozen, only the adapter trains) ---
EPOCHS: int = 50
BATCH_SIZE: int = 24
LEARNING_RATE: float = 2e-4      # swept per-arm (4 LRs)
COND_DROP_PROB: float = 0.0      # uncond branch IS the frozen base -> dropout not needed
MAX_TIME_HOURS: float = 8.0
N_HOLDOUT: int = 2000            # held out of training; eval targets drawn from here (unseen FPs)

# --- Sampling / evaluation ---
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
N_TARGETS: int = 6
# :param N_TARGETS_EXTRA: Additional held-out targets drawn AFTER the first
#     N_TARGETS, from the remaining pool. Kept as a separate draw on purpose: the
#     original six have now driven selection across four runs and are no longer a
#     clean generalization estimate, but re-drawing them would break comparability
#     with every number already measured. This way the run reports both -- the old
#     six for continuity, the fresh set for an honest estimate.
N_TARGETS_EXTRA: int = 10
# :param N_TARGETS_TRAIN: Targets drawn from the TRAINING pool. The regime
#     diagnostic: if lift on molecules the adapter trained on is no better than on
#     held-out ones, the adapter is not failing to generalize -- it is failing to
#     fit at all, and capacity (wider trunk, encoder, more bits) cannot be the
#     binding constraint. A large train-vs-holdout gap means the opposite, and more
#     capacity would make it worse.
N_TARGETS_TRAIN: int = 6
N_PER_TARGET: int = 64
N_BASELINE: int = 256
EVAL_CHUNK: int = 32
# Steering weights must stay <= 1. The composition blends rate matrices as
# log R_uncond + sum_i w_i (log R_cond_i - log R_uncond); with w > 1 the
# unconditional coefficient 1 - sum(w_i) goes NEGATIVE, so the blend
# extrapolates past the conditional instead of interpolating toward it, and
# _stabilize's >1e5 clamp then silently drops rates. Empirically w=2 does not
# work -- it degrades rather than steers harder.
GUIDANCE_WEIGHTS: list = [0.25, 0.5, 1.0]
GRID_N: int = 24
GRID_SCALE: float = 2.0

# --- mid-training probe ---
PROBE_EVERY_K: int = 5
PROBE_N_TARGETS: int = 2
PROBE_N: int = 24
PROBE_STEPS: int = 100
PROBE_WEIGHT: float = 1.0
PROBE_BASELINE_N: int = 48

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ============================================================================
# Helpers
# ============================================================================
def mol_morgan_bits(mol, radius, n_bits, counts: bool = False) -> np.ndarray:
    """Condition vector for one molecule.

    ``counts=False`` is a binary bit vector. ``counts=True`` is a hashed count
    fingerprint with ``log1p``. This MUST stay identical to
    ``molsmith.sample.morgan_bits``: the adapter is served through that function,
    and feeding it the other encoding raises nothing while steering badly.
    """
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    if not counts:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr
    cv = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(cv, arr)
    return np.log1p(arr).astype(np.float32)


def mol_morgan_binary(mol, radius, n_bits) -> np.ndarray:
    """Always-binary fingerprint, for the Tanimoto METRIC.

    The reported similarity stays binary whatever the conditioning encoding is,
    so a number from a count-conditioned adapter is directly comparable to
    fingerprint@2.0.0's. Changing the metric alongside the conditioning would
    make 'did counts help' unanswerable.
    """
    return mol_morgan_bits(mol, radius, n_bits, counts=False)


def tanimoto_to_target(fp_mat, target):
    if fp_mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    inter = fp_mat @ target
    union = fp_mat.sum(1) + target.sum() - inter
    return inter / np.clip(union, 1e-8, None)


def decode_and_fp(samples, atom_decoder, bond_decoder, radius, n_bits):
    """(mols, smiles, binary fingerprints, disconnected fraction).

    "Valid" here means "decodes to a parseable SMILES", which a multi-fragment
    result satisfies while being useless as a molecule. That distinction was
    invisible in this experiment for a long time and it hid a real effect: the
    frozen base emits fragments ~1.6% of the time unconditionally, and fingerprint
    conditioning pushes that to ~15%. Both report validity ~0.98.

    Disconnected molecules stay in the scored pool so the lifts remain comparable
    with everything already measured; the fraction is reported alongside so the
    validity column can no longer flatter an adapter that is quietly fragmenting.
    """
    mols, smis, n_disc = [], [], 0
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi is not None and Chem.MolFromSmiles(smi) is not None:
            mols.append(mol)
            smis.append(smi)
            n_disc += "." in smi
    # BINARY: this feeds the Tanimoto metric, which stays binary whatever the
    # conditioning encoding is, so numbers remain comparable across generations.
    fp = np.stack([mol_morgan_binary(m, radius, n_bits) for m in mols]) if mols else \
        np.zeros((0, n_bits), dtype=np.float32)
    return mols, smis, fp, (n_disc / len(mols) if mols else float("nan"))


def guided_sample(base, adapter, target_fp, weight, n, steps, eta, omega, td, chunk, device):
    comp = AdapterComposition([ConditionBranch(adapter, torch.as_tensor(target_fp, dtype=torch.float32), weight)],
                              base=base, mode="product")
    samp = AdaptedSampler(base, comp, eta=eta, omega=omega, sample_steps=steps, time_distortion=td)
    out, rem = [], n
    while rem > 0:
        cur = min(chunk, rem)
        out += samp.sample(cur, device=device, show_progress=False)
        rem -= cur
    return out


class FPAdapterProbe(pl.Callback):
    """Per-epoch loss log + every-K-epoch Tanimoto-steering probe vs a cached
    unconditional baseline, so training is visible."""

    def __init__(self, e, atom_decoder, bond_decoder, radius, n_bits, targets,
                 baseline_tan, every_k, n, steps, weight, eta, omega, td, chunk):
        super().__init__()
        self.e = e
        self.ad, self.bd, self.radius, self.n_bits = atom_decoder, bond_decoder, radius, n_bits
        # (condition, metric) per target. They differ once the adapter is
        # conditioned on counts while Tanimoto is still reported on binary bits,
        # and conflating them would silently score against the wrong vector.
        self.targets = targets
        self.baseline_tan = baseline_tan
        self.every_k, self.n, self.steps, self.weight = every_k, n, steps, weight
        self.eta, self.omega, self.td, self.chunk = eta, omega, td, chunk

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
        per = []
        for tcond, tmetric in self.targets:
            samples = guided_sample(pl_module.base, pl_module.adapter, tcond, self.weight, self.n,
                                    self.steps, self.eta, self.omega, self.td, self.chunk, device)
            _, _, gfp, _ = decode_and_fp(samples, self.ad, self.bd, self.radius, self.n_bits)
            sims = tanimoto_to_target(gfp, tmetric)
            per.append(float(sims.mean()) if sims.size else float("nan"))
        guided = float(np.nanmean(per)) if per else float("nan")
        base = float(np.nanmean(self.baseline_tan)) if self.baseline_tan else float("nan")
        self.e.log(f"[epoch {ep}] PROBE(w={self.weight}) guided<T>={guided:.3f} baseline<T>={base:.3f} "
                   f"lift={guided - base:+.3f} per_target={[round(x, 3) for x in per]}")


# ============================================================================
@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"ZINC frozen-base AdaLN CFG-ADAPTER on {e.FP_BITS}-bit Morgan fingerprint")
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if e.FP_FROM not in ("source", "decoded"):
        raise ValueError(f"FP_FROM must be 'source' or 'decoded', got {e.FP_FROM!r}")

    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_advoc", os.path.join(_PROJECT_DIR, "experiments", "adapter_training__zinc.py"))
    _m = _ilu.module_from_spec(_spec)
    sys.modules["_advoc"] = _m           # pycomex reads annotations off the module
    _spec.loader.exec_module(_m)
    atom_types, bond_types, kekulize, source = _m._vocabulary(e.VOCABULARY)

    e.log(f"vocabulary '{e.VOCABULARY}': {len(atom_types)} atoms {atom_types}")
    e.log(f"  bonds={bond_types} kekulize={kekulize} smiles_source={source}")
    e.log(f"  fp_from={e.FP_FROM}"
          + ("  (fingerprint describes the GRAPH)" if e.FP_FROM == "decoded"
             else "  (fingerprint describes the SOURCE SMILES)"))
    atom_encoder, atom_decoder, bond_encoder, bond_decoder = build_encoders(atom_types, bond_types)

    if source == "reference_split":
        from defog.data import zinc_reference as _zref
        smiles_iter = _zref.load_reference_split().train_smiles
    else:
        smiles_iter = pd.read_csv(e.CSV_PATH)[e.SMILES_COLUMN]
    e.log(f"source molecules: {len(smiles_iter)}")

    # ONE full-size array only. At 1024 bits over 224k molecules a float32 array
    # is ~0.9 GB and 2048 bits is ~1.8 GB, so keeping both conventions and both
    # encodings would be many gigabytes for no benefit: the evaluation needs only
    # a couple of dozen target molecules, whose fingerprints are recomputed on
    # demand. The decoded SMILES are kept (cheap, strings) so those targets can be
    # built in either convention later.
    # Peak is ~2x the final array, briefly: np.stack allocates the result while
    # fp_list is still alive. ~3.6 GB at 2048 bits, which the node absorbs, but it
    # is the reason `del fp_list` follows immediately.
    CEIL_SAMPLE = 5000          # the ceiling is a mean; 5k pins it to ~0.003
    graphs, smiles_kept, dec_smiles, fp_list = [], [], [], []
    ceil_num = []
    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder, kekulize=kekulize)
        if data is None:
            continue
        dec_mol = pyg_data_to_mol(data, atom_decoder, bond_decoder)
        back = mol_to_smiles(dec_mol) if dec_mol is not None else None
        dec_mol = Chem.MolFromSmiles(back) if back else None
        if dec_mol is None:
            continue          # cannot describe the graph, so cannot label it
        graphs.append(data)
        smiles_kept.append(smi)
        dec_smiles.append(back)
        # The CONDITION: FP_FROM picks which molecule, FP_COUNTS picks the encoding.
        cond_mol = dec_mol if e.FP_FROM == "decoded" else mol
        fp_list.append(mol_morgan_bits(cond_mol, e.FP_RADIUS, e.FP_BITS,
                                       counts=e.FP_COUNTS))
        if len(ceil_num) < CEIL_SAMPLE:
            # Ceiling is a property of the BINARY metric, so measure it there
            # regardless of how the adapter is conditioned.
            a = mol_morgan_binary(mol, e.FP_RADIUS, e.FP_BITS)
            b = mol_morgan_binary(dec_mol, e.FP_RADIUS, e.FP_BITS)
            i = float((a * b).sum()); u = float(a.sum() + b.sum() - i)
            ceil_num.append(i / u if u > 0 else 1.0)
    M = len(graphs)
    fp = np.stack(fp_list); del fp_list
    ceil = float(np.mean(ceil_num))
    e.log(f"{M} graphs; source-vs-decoded BINARY Tanimoto = {ceil:.4f} "
          f"over {len(ceil_num)} (the ceiling when scoring against SOURCE targets)")
    e.log(f"condition: {e.FP_BITS} bits, radius {e.FP_RADIUS}, "
          f"counts={e.FP_COUNTS}, from={e.FP_FROM}; "
          f"nonzero/molecule ~{float((fp > 0).sum(1).mean()):.1f}, max value "
          f"{float(fp.max()):.3f}")
    e["fp/ceiling_source_vs_decoded"] = ceil
    e["fp/from"] = e.FP_FROM
    e["fp/counts"] = e.FP_COUNTS
    e["fp/bits"] = e.FP_BITS
    cond_mean = fp.mean(0)
    cond_std = np.clip(fp.std(0), 1e-6, None)
    fp_t = torch.from_numpy(fp)
    for i, g in enumerate(graphs):
        g.cond = fp_t[i].unsqueeze(0)   # (1, 128) RAW fingerprint (adapter normalizes internally)

    perm = torch.randperm(M).tolist()
    n_hold = min(e.N_HOLDOUT, M // 5)
    holdout_idx, train_idx = perm[:n_hold], perm[n_hold:]
    train_graphs = [graphs[i] for i in train_idx]
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_graphs, batch_size=e.BATCH_SIZE, shuffle=True)
    e.log(f"Adapter train: {len(train_graphs)}   held-out target pool: {len(holdout_idx)}")

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0, f"expected unconditional base, cond_dim={base.cond_dim}"

    # A vocabulary mismatch trains the adapter against classes that decode to
    # different elements. It converges normally and produces a useless adapter,
    # with nothing in the loop to notice.
    from defog.data import vocabulary as _vocab
    e.log(_vocab.check_model(base, atom_types, bond_types,
                             what=f"base {e.BASE_CKPT}"))
    # in_dim is derived from FP_BITS, never taken from the spec: a hand-written
    # in_dim that disagreed with the fingerprint width would raise at the first
    # forward, but only after the run had already loaded the dataset and the base.
    enc_spec = None
    if e.COND_ENCODER:
        enc_spec = dict(e.COND_ENCODER)
        enc_spec.setdefault("kind", "mlp")
        enc_spec["in_dim"] = e.FP_BITS
    adapter = AdaLNAdapter.for_base(
        base, cond_dim=e.FP_BITS, hidden=e.H_HIDDEN, time_conditioned=e.TIME_CONDITIONED,
        streams=tuple(e.STREAMS), cond_mean=cond_mean, cond_std=cond_std,
        interior_ff=e.INTERIOR_FF, interior_attn=e.INTERIOR_ATTN,
        cond_encoder=enc_spec,
        name="fp_adapter", cond_type=f"morgan{e.FP_BITS}")
    e["adapter/num_params"] = sum(p.numel() for p in adapter.parameters())
    e["adapter/cond_encoder"] = enc_spec
    e["adapter/hidden"] = e.H_HIDDEN
    # The narrowest point in the conditioning path -- the thing this run is varying.
    bottleneck = e.H_HIDDEN if enc_spec is None else min(e.H_HIDDEN, enc_spec.get("out_dim", 512))
    e["adapter/bottleneck"] = bottleneck
    e.log(f"adapter: {e['adapter/num_params']:,} params (interior_ff={e.INTERIOR_FF} interior_attn={e.INTERIOR_ATTN}; "
          f"base {sum(p.numel() for p in base.parameters()):,} frozen)")
    e.log(f"conditioning path: {e.FP_BITS} bits -> "
          + (f"encoder[{enc_spec['kind']} h={enc_spec.get('hidden')} x{enc_spec.get('n_blocks')}]"
             f" -> {enc_spec.get('out_dim')} -> " if enc_spec else "")
          + f"trunk({e.H_HIDDEN}) -> FiLM heads   [narrowest = {bottleneck}]")
    module = AdapterModule(base, adapter, cond_attr="cond", cond_drop_prob=e.COND_DROP_PROB,
                           lr=e.LEARNING_RATE, l10_lr_scale=e.L10_LR_SCALE)

    # probe targets (held out) + cached unconditional baseline
    probe_idx = random.sample(holdout_idx, min(e.PROBE_N_TARGETS, len(holdout_idx)))
    probe_raw = [fp[i] for i in probe_idx]        # conditioning, trained encoding
    probe_metric = [mol_morgan_binary(Chem.MolFromSmiles(dec_smiles[i]),
                                      e.FP_RADIUS, e.FP_BITS) for i in probe_idx]
    pb = []
    pbs = Sampler(base, eta=e.ETA, omega=e.OMEGA, sample_steps=e.PROBE_STEPS, time_distortion=e.TIME_DISTORTION)
    rem = max(32, e.PROBE_N)
    while rem > 0:
        cur = min(e.EVAL_CHUNK, rem)
        pb += pbs.sample(cur, device=device, show_progress=False)
        rem -= cur
    _, _, pb_fp, _ = decode_and_fp(pb, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
    probe_baseline = [float(tanimoto_to_target(pb_fp, t).mean()) if pb_fp.shape[0] else float("nan")
                      for t in probe_metric]
    e.log(f"probe baseline <T>: {[round(x, 3) for x in probe_baseline]}")
    probe = FPAdapterProbe(e, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS,
                           list(zip(probe_raw, probe_metric)), probe_baseline,
                           e.PROBE_EVERY_K, e.PROBE_N, e.PROBE_STEPS, e.PROBE_WEIGHT,
                           e.ETA, e.OMEGA, e.TIME_DISTORTION, e.EVAL_CHUNK)

    trainer = pl.Trainer(max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS}, accelerator="auto",
                         devices=1, enable_progress_bar=True, enable_checkpointing=False, logger=False,
                         gradient_clip_val=1.0, callbacks=[probe])
    e.log(f"Training adapter: epochs<={e.EPOCHS} max_time={e.MAX_TIME_HOURS}h batch={e.BATCH_SIZE} LR={e.LEARNING_RATE}")
    trainer.fit(module, train_dataloaders=train_loader)

    ckpt = adapter.save(os.path.join(e.path, "fp_adapter"))
    with open(os.path.join(e.path, "fp_adapter_stats.json"), "w") as f:
        json.dump({"fp_bits": e.FP_BITS, "fp_radius": e.FP_RADIUS, "atom_types": atom_types,
                   "cond_mean": cond_mean.tolist(), "cond_std": cond_std.tolist(),
                   "learning_rate": e.LEARNING_RATE}, f)
    e.log(f"Saved adapter -> {ckpt}")

    # -- Evaluation: Tanimoto lift vs baseline, guidance-weight sweep ----------
    e.log("=" * 60)
    base = base.to(device).eval()
    adapter = adapter.to(device).eval()
    # This first draw is deliberately the ORIGINAL call -- same population, same k,
    # same RNG state -- so it reproduces the exact six targets earlier runs were
    # scored on and the new numbers stay comparable to them. Everything else is
    # drawn afterwards, from what is left, so it cannot perturb that draw.
    tgt_idx = random.sample(holdout_idx, min(e.N_TARGETS, len(holdout_idx)))
    tgt_split = ["holdout_orig"] * len(tgt_idx)
    _rest = [i for i in holdout_idx if i not in set(tgt_idx)]
    _extra = random.sample(_rest, min(e.N_TARGETS_EXTRA, len(_rest)))
    tgt_idx += _extra
    tgt_split += ["holdout_fresh"] * len(_extra)
    _tr = random.sample(train_idx, min(e.N_TARGETS_TRAIN, len(train_idx)))
    tgt_idx += _tr
    tgt_split += ["train"] * len(_tr)
    e.log(f"targets: {tgt_split.count('holdout_orig')} original held-out (comparability) "
          f"+ {tgt_split.count('holdout_fresh')} fresh held-out (honest estimate) "
          f"+ {tgt_split.count('train')} training (regime diagnostic)")
    # Condition on the TRAINING convention (that is what the adapter speaks),
    # then score the same generated molecules against BOTH conventions. One
    # sampling run, two measurements: the gap between them is exactly the
    # charge ceiling, shown rather than absorbed.
    tgt_raw = [fp[i] for i in tgt_idx]          # conditioning, trained encoding
    # Measurement targets are always BINARY, in both conventions.
    tgt_by_conv = {
        "decoded": [mol_morgan_binary(Chem.MolFromSmiles(dec_smiles[i]),
                                      e.FP_RADIUS, e.FP_BITS) for i in tgt_idx],
        "source": [mol_morgan_binary(Chem.MolFromSmiles(smiles_kept[i]),
                                     e.FP_RADIUS, e.FP_BITS) for i in tgt_idx],
    }
    tgt_mols = [Chem.MolFromSmiles(smiles_kept[i]) for i in tgt_idx]
    e.log(f"{len(tgt_idx)} target molecules; conditioning on '{e.FP_FROM}' FPs, "
          f"scoring against both conventions")

    base_sampler = Sampler(base, eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS, time_distortion=e.TIME_DISTORTION)
    bsamp, rem = [], e.N_BASELINE
    while rem > 0:
        cur = min(e.EVAL_CHUNK, rem)
        bsamp += base_sampler.sample(cur, device=device, show_progress=False)
        rem -= cur
    _, _, base_fp, base_disc = decode_and_fp(bsamp, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
    e.log(f"baseline valid: {base_fp.shape[0]}/{e.N_BASELINE}  disconnected {base_disc:.3f}"
          f"  <- the frozen base's own fragment rate; steering is measured against it")
    e["fp/baseline_disconnected"] = base_disc

    methods = ["baseline"] + [f"w={w}" for w in e.GUIDANCE_WEIGHTS]
    CONVS = ("decoded", "source")
    SPLITS = ("holdout_orig", "holdout_fresh", "train")
    agg = {c: {m: [] for m in methods} for c in CONVS}
    # Same numbers, sliced by split. Pooling them would let the training targets
    # inflate the held-out figure, which is the one that means anything.
    agg_split = {s: {c: {m: [] for m in methods} for c in CONVS} for s in SPLITS}
    per_target = []
    for ti, (traw, tmol) in enumerate(zip(tgt_raw, tgt_mols)):
        split = tgt_split[ti]
        rec = {"index": int(tgt_idx[ti]), "smiles": smiles_kept[tgt_idx[ti]],
               "split": split, "n_heavy": tmol.GetNumHeavyAtoms(),
               "baseline_mean_tanimoto": {}, "per_w": {}}
        for c in CONVS:
            bs = tanimoto_to_target(base_fp, tgt_by_conv[c][ti])
            agg[c]["baseline"].extend(bs.tolist())
            agg_split[split][c]["baseline"].extend(bs.tolist())
            rec["baseline_mean_tanimoto"][c] = float(bs.mean()) if bs.size else None
        best_grid = None
        for w in e.GUIDANCE_WEIGHTS:
            samples = guided_sample(base, adapter, traw, w, e.N_PER_TARGET, e.EVAL_STEPS,
                                    e.ETA, e.OMEGA, e.TIME_DISTORTION, e.EVAL_CHUNK, device)
            mols, smis, gfp, gdisc = decode_and_fp(samples, atom_decoder, bond_decoder, e.FP_RADIUS, e.FP_BITS)
            entry = {"n_valid": len(mols), "n_unique": len(set(smis)),
                     "validity": len(mols) / len(samples) if samples else 0.0,
                     "disconnected": gdisc}
            sims = None
            for c in CONVS:
                s = tanimoto_to_target(gfp, tgt_by_conv[c][ti])
                agg[c][f"w={w}"].extend(s.tolist())
                agg_split[split][c][f"w={w}"].extend(s.tolist())
                entry[f"mean_tanimoto_{c}"] = float(s.mean()) if s.size else None
                entry[f"max_tanimoto_{c}"] = float(s.max()) if s.size else None
                if c == "decoded":
                    sims = s
            rec["per_w"][str(w)] = entry
            e.log(f"  [t{ti} {split}] w={w}: valid={len(mols)}/{len(samples)} uniq={len(set(smis))} "
                  f"<T>decoded={entry['mean_tanimoto_decoded']} "
                  f"<T>source={entry['mean_tanimoto_source']} disc={gdisc:.3f}")
            if abs(w - e.GRID_SCALE) < 1e-9 and mols:
                order = np.argsort(-sims)[:e.GRID_N]
                best_grid = ([tmol] + [mols[j] for j in order], ["TARGET"] + [f"T={sims[j]:.2f}" for j in order])
        if best_grid:
            Draw.MolsToGridImage(best_grid[0], molsPerRow=5, subImgSize=(220, 220),
                                 legends=best_grid[1]).save(os.path.join(e.path, f"grid_target{ti}.png"))
        per_target.append(rec)

    summary = {"methods": methods, "n_targets": len(tgt_idx), "learning_rate": e.LEARNING_RATE,
               "eval_steps": e.EVAL_STEPS, "baseline_valid": int(base_fp.shape[0]),
               "vocabulary": e.VOCABULARY, "fp_from": e.FP_FROM,
               "ceiling_source_vs_decoded": ceil,
               "per_target": per_target, "aggregate": {}}
    e.log("=" * 60)
    e.log(f"{'convention':12s}{'method':12s}{'<T>':>9s}{'lift':>9s}")
    for c in CONVS:
        base_mean = float(np.mean(agg[c]["baseline"])) if agg[c]["baseline"] else float("nan")
        summary["aggregate"][c] = {}
        for m in methods:
            mean = float(np.mean(agg[c][m])) if agg[c][m] else float("nan")
            summary["aggregate"][c][m] = {"mean_tanimoto": mean,
                                          "lift_over_baseline": mean - base_mean}
            e.log(f"{c:12s}{m:12s}{mean:>9.4f}{mean - base_mean:>+9.4f}")
    e.log(f"ceiling when scoring against SOURCE targets: {ceil:.4f} "
          f"(a perfect reproduction of the target GRAPH cannot beat this)")

    # --- by split: the regime diagnostic -------------------------------------
    # `train` here means the adapter SAW these molecules during training. If its
    # lift on them is no higher than on held-out targets, the limit is not
    # generalization and no amount of extra conditioning capacity will move it.
    summary["by_split"] = {}
    e.log("-" * 60)
    e.log(f"{'split':16s}{'n':>4s}  {'method':10s}{'<T>dec':>9s}{'lift':>9s}{'validity':>10s}")
    for s in SPLITS:
        rows = [r for r in per_target if r["split"] == s]
        if not rows:
            continue
        bl = agg_split[s]["decoded"]["baseline"]
        base_mean = float(np.mean(bl)) if bl else float("nan")
        summary["by_split"][s] = {"n_targets": len(rows), "aggregate": {}}
        for m in methods:
            vals = agg_split[s]["decoded"][m]
            mean = float(np.mean(vals)) if vals else float("nan")
            val = float("nan")
            if m != "baseline":
                w = m.split("=", 1)[1]
                vv = [r["per_w"][w]["validity"] for r in rows if w in r["per_w"]]
                val = float(np.mean(vv)) if vv else float("nan")
            summary["by_split"][s]["aggregate"][m] = {
                "mean_tanimoto": mean, "lift_over_baseline": mean - base_mean,
                "validity": val}
            e.log(f"{s:16s}{len(rows):>4d}  {m:10s}{mean:>9.4f}{mean - base_mean:>+9.4f}{val:>10.4f}")
    ho = summary["by_split"].get("holdout_fresh", {}).get("aggregate", {})
    tr = summary["by_split"].get("train", {}).get("aggregate", {})
    top = f"w={e.GUIDANCE_WEIGHTS[-1]}"
    if top in ho and top in tr:
        gap = tr[top]["lift_over_baseline"] - ho[top]["lift_over_baseline"]
        # A nan gap means a split produced no valid molecules at all. Reading a
        # verdict off that would state a conclusion drawn from missing data -- the
        # exact failure this diagnostic exists to avoid -- so say so instead.
        if not np.isfinite(gap):
            e.log(f"train-minus-heldout lift gap at {top}: UNDEFINED "
                  f"(a split produced no valid molecules; no conclusion available)")
        else:
            e["fp/train_holdout_gap"] = gap
            e.log(f"train-minus-heldout lift gap at {top}: {gap:+.4f}  "
                  + ("-> memorization-limited: extra capacity would widen this, not help"
                     if gap > 0.05 else
                     "-> NOT generalization-limited: the adapter fits its own training "
                     "targets no better, so capacity is not the binding constraint"))
    e.commit_json("adapter_fingerprint_metrics.json", summary)

    # Both conventions side by side, with the source-target ceiling drawn in --
    # otherwise the source bars look like a failure rather than a known limit.
    dec_mean = summary["aggregate"]["decoded"]["baseline"]["mean_tanimoto"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods)); wdt = 0.38
    for k, (c, col) in enumerate((("decoded", "#55a868"), ("source", "#4c72b0"))):
        vals = [summary["aggregate"][c][m]["mean_tanimoto"] for m in methods]
        bars = ax.bar(x + (k - 0.5) * wdt, vals, wdt, color=col, label=f"{c} targets")
        for b, mn in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, mn, f"{mn:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(ceil, ls=":", color="#c44e52", lw=1.4,
               label=f"source-target ceiling {ceil:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("mean Tanimoto to target"); ax.legend(fontsize=8)
    ax.set_title(f"FP adapter steering (LR={e.LEARNING_RATE}, {e.EVAL_STEPS} steps)")
    fig.tight_layout()
    e.commit_fig("method_comparison.png", fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5.2))
    bins = np.linspace(0, 1, 41)
    for m in methods:
        if agg["decoded"][m]:
            ax2.hist(agg["decoded"][m], bins=bins, density=True, histtype="stepfilled",
                     alpha=0.45, label=f"{m} (<T>={np.mean(agg['decoded'][m]):.3f})")
    ax2.set_xlabel("Tanimoto to target"); ax2.set_ylabel("density")
    ax2.set_title("FP adapter steering: Tanimoto to DECODED target by guidance weight")
    ax2.legend(fontsize=9); fig2.tight_layout()
    e.commit_fig("tanimoto_distributions.png", fig2)

    e.log("=" * 60)
    for c in ("decoded", "source"):
        for m in methods:
            a = summary["aggregate"][c][m]
            e.log(f"{c:8s} {m:10s} <T>={a['mean_tanimoto']:.3f}  "
                  f"lift={a['lift_over_baseline']:+.3f}")
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
    e.PROBE_N_TARGETS = 2
    e.N_HOLDOUT = 40
    e.N_TARGETS = 2
    e.N_TARGETS_EXTRA = 1
    e.N_TARGETS_TRAIN = 1
    e.N_PER_TARGET = 8
    e.N_BASELINE = 8
    e.EVAL_CHUNK = 8
    e.GUIDANCE_WEIGHTS = [2.0]
    e.GRID_N = 4
    df = pd.read_csv(e.CSV_PATH).head(300)
    smoke = os.path.join(folder_path(__file__), "_adapter_fp_smoke.csv")
    df.to_csv(smoke, index=False)
    e.CSV_PATH = smoke
    if e.VOCABULARY == "legacy_aromatic" and not os.path.exists(e.BASE_CKPT):
        e.BASE_CKPT = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")

    # The reference-split path ignores CSV_PATH, so truncate it too or a "smoke"
    # run encodes and double-fingerprints all 224k training molecules.
    from defog.data import zinc_reference as _zr
    _real = _zr.load_reference_split

    def _small(*a, **kw):
        s0 = _real(*a, **kw)
        return _zr.ZincReferenceSplit(
            train_smiles=s0.train_smiles[:300], val_smiles=s0.val_smiles[:50],
            test_smiles=s0.test_smiles[:50],
            provenance={**s0.provenance, "TRUNCATED_FOR_SMOKE_TEST": True})

    _zr.load_reference_split = _small


experiment.run_if_main()
