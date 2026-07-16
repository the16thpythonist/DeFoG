"""
GDPO structural-sanity fine-tune -- a SINGLE configuration, as a pycomex experiment.

Discourages "weird artefact shapes" (odd ring sizes, oversized fused/spiro ring
systems, too many rings) that a factorized generator hallucinates off the data
manifold, by GDPO's eager policy gradient with a BINARY structural-sanity reward:

    reward(mol) = 1.0  iff  mol is valid  AND  connected ('.'-free SMILES)  AND
                            every structural feature is inside the training-set
                            envelope (ring sizes in the observed set; #rings,
                            largest fused/spiro ring-system, spiro & bridgehead
                            counts all within the observed support);
                = 0.0  otherwise.

The envelope is DERIVED FROM the reference SMILES (the model's training data), so
by construction every real molecule scores 1.0 -- the reward rewards *fidelity to
the manifold*, not *typicality within it*, and therefore does NOT collapse diversity
(unlike a "how average are your rings" reward, whose optimum is benzene-everywhere).
See docs / the design discussion for the fidelity-vs-typicality distinction.

This STACKS on the connectivity fine-tune: the default base is a ``*_connectivity``
checkpoint and the reward keeps connectivity in the objective (``valid & connected
& sane``), so the earlier gain cannot regress. Best-snapshot selection is input-
inclusive, so a stacking run can never ship a model worse than what it started from.

Headline metric: structural-violation rate down (= 1 - E[reward]); we also report
the artifact rate among otherwise-good (valid&connected) molecules -- the literal
"weird shape" rate. Fidelity cross-checks: ring-size histogram divergence to the
dataset and FCD (Frechet ChemNet Distance, via ``fcd_torch``) should drop. Guard-
rails: validity / uniqueness / novelty must not regress.

Model-agnostic: the atom vocabulary is reconstructed from the checkpoint's
atom_weights, so the same experiment runs on ZINC / GuacaMol / AqSolDB by changing
CKPT_PATH + REFERENCE_SMILES. One run = one config; the SWEEP / fan-out is many
submissions (one per dataset), not done here.

Usage:
    python experiments/gdpo_sanity.py \
        --CKPT_PATH "'~/Downloads/zinc_uncond_4e-4_connectivity.ckpt'" \
        --REFERENCE_SMILES "'data/zinc_250k_rdkit.csv'"
    python experiments/gdpo_sanity.py --__TESTING__ True
"""
import os
import json
import time
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, GetPeriodicTable, rdMolDescriptors
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog.core import DeFoGModel, GDPOTrainer
from defog.core.data import dense_to_pyg
from defog.domains import MoleculeDomain
from defog.domains.molecule import build_encoders
from experiments.guided_logp_demo import BOND_TYPES

RDLogger.DisableLog("rdApp.*")

# == configuration ==========================================================
# :param CKPT_PATH: pretrained DeFoG molecular checkpoint to fine-tune. Default is the
#     ZINC connectivity-improved model -- this experiment STACKS sanity on top of it.
CKPT_PATH: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")
# :param REFERENCE_SMILES: CSV (or .txt/.smi, one per line) of the model's TRAINING
#     SMILES. Defines the structural envelope (support/quantile of ring & topology
#     features) AND the FCD / ring-histogram / novelty reference. Must match the model.
REFERENCE_SMILES: str = "data/zinc_250k_rdkit.csv"
# :param SMILES_COLUMN: column name holding SMILES if REFERENCE_SMILES is a CSV.
SMILES_COLUMN: str = "smiles"
SEED: int = 0

# --- sanity envelope (the reward definition) ---
# :param ENVELOPE_QUANTILE: quantile of each COUNT feature (#rings, largest ring-
#     system, spiro, bridgehead) used as its upper bound. 1.0 -> strict max = every
#     real molecule passes (honors the "fidelity, no diversity cost" guarantee).
#     <1.0 trims the tail to be stricter, at the cost of rejecting a small fraction
#     of real molecules.
ENVELOPE_QUANTILE: float = 1.0
# :param RING_MIN_COUNT: a ring size is admitted to the allowed set only if it occurs
#     at least this many times in the reference. 50 on the 249k ZINC set trims the
#     size-9..24 macrocycle tail (each only a handful of molecules) to the drug-like
#     set {3..8} at ~0.02% real-molecule rejection ("moderate" envelope). NOTE: this
#     is an ABSOLUTE occurrence count, so it is dataset-size dependent -- revisit it
#     on fan-out (GuacaMol ~1.4M, AqSolDB ~10k) to keep the same ~0.1%-of-molecules
#     floor. 1 -> full observed support (permits the exotic tail).
RING_MIN_COUNT: int = 50
# :param REQUIRE_CONNECTED: include connectivity in the reward (valid & connected &
#     sane). True keeps the stacked connectivity gain in the objective so it can't
#     regress; False rewards sanity alone.
REQUIRE_CONNECTED: bool = True
# :param REFERENCE_LIMIT: cap on reference molecules parsed for the envelope/hist/
#     novelty (None = all). Full set makes the max-bounds exact; subsample only to
#     speed up quick tests.
REFERENCE_LIMIT: int = None

# --- fidelity cross-checks ---
# :param COMPUTE_FCD: compute Frechet ChemNet Distance (needs ``fcd_torch``; if absent
#     the experiment logs a skip and reports fcd=None instead of crashing).
COMPUTE_FCD: bool = True
# :param FCD_REF_SAMPLES: reference molecules sampled from REFERENCE_SMILES for FCD /
#     ring-histogram / novelty comparison.
FCD_REF_SAMPLES: int = 10000
# :param RING_HIST_MAX: ring sizes 3..RING_HIST_MAX are binned individually; larger
#     rings fall into one macrocycle overflow bin, for the ring-histogram divergence.
RING_HIST_MAX: int = 12

# --- training budget / rollout policy (tuned config from the connectivity work) ---
# :param ROUNDS: re-anchoring rounds (1 = a plain single fine-tune).
ROUNDS: int = 1
# :param ITERATIONS: GDPO updates per round.
ITERATIONS: int = 100
# :param ROLLOUT_SIZE: K rollout molecules per iteration.
ROLLOUT_SIZE: int = 128
# :param SAMPLE_STEPS / ETA / OMEGA / TIME_DISTORTION: rollout (and matched eval) policy.
SAMPLE_STEPS: int = 100
ETA: float = 0.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
# :param SUBSAMPLE_STEPS: noisy states per trajectory that enter the gradient.
SUBSAMPLE_STEPS: int = 12
# :param MINIBATCH_SIZE: trajectories per grad forward (bounds autograd memory).
MINIBATCH_SIZE: int = 16

# --- eager gradient / advantage ---
# :param REDUCTION: "sum" (true joint LL) | "mean".
REDUCTION: str = "sum"
# :param ADVANTAGE_MODE: "grpo" | "mean" | "none".
ADVANTAGE_MODE: str = "grpo"
# :param POSITIVE_ONLY: RAFT-style clamp advantage>=0 (never push down bad endpoints).
POSITIVE_ONLY: bool = False
# :param LAMBDA_EDGE: weight of the edge (bond) term vs the node (atom-type) term.
LAMBDA_EDGE: float = 1.0
# :param LR: AdamW learning rate.
LR: float = 2e-5

# --- KL to reference (over-optimization guard) ---
# :param KL_COEF: KL-to-reference strength (0 -> no reference, no KL). 0.2 = tuned value.
KL_COEF: float = 0.2
# :param KL_ANCHOR: "fixed" | "moving".
KL_ANCHOR: str = "fixed"
# :param ANCHOR_DECAY: EMA decay of the moving anchor.
ANCHOR_DECAY: float = 0.99
# :param KL_TARGET: adaptive KL target (None -> fixed KL_COEF).
KL_TARGET: float = None
# :param EMA_DECAY: deployment-weights EMA.
EMA_DECAY: float = 0.9

# --- evaluation / checkpointing ---
# :param EVAL_SAMPLES / EVAL_STEPS: fresh molecules for the COMPARED BEFORE/AFTER
#     measurement. 500 steps = the default deploy point.
EVAL_SAMPLES: int = 2048
EVAL_STEPS: int = 500
# :param ROUND_EVAL_SAMPLES: cheaper eval after each intermediate round.
ROUND_EVAL_SAMPLES: int = 512
# :param SELECT_EVAL_STEPS: cheap step count for RANKING snapshots (ordering holds).
SELECT_EVAL_STEPS: int = 100
# :param CKPT_EVERY: save a snapshot every N iters (0=off).
CKPT_EVERY: int = 20
# :param SELECT_BEST: after training, rank every snapshot + final (+ input) by the
#     structural-violation rate (validity- and uniqueness-gated) and keep the best.
SELECT_BEST: bool = True
# :param SELECT_EVAL_SAMPLES: samples per snapshot during best-snapshot ranking.
SELECT_EVAL_SAMPLES: int = 1024
# :param SELECT_INCLUDE_INPUT: also rank the INPUT checkpoint, so a stacking run can
#     never ship worse than what it started from. On by default (stacking).
SELECT_INCLUDE_INPUT: bool = True

# NOTE: __DEBUG__ must be False when submitting a sweep -- debug mode writes to a
# single overwriteable folder, so parallel runs would clobber each other.
__DEBUG__: bool = False
__TESTING__: bool = False


def atom_decoder_from_ckpt(model):
    """Reconstruct the exact atom_decoder (class idx -> symbol) from the checkpoint's
    atom_weights, so decoding matches the model regardless of dataset."""
    weights = model.hparams.get("atom_weights")
    if not weights:
        raise ValueError("checkpoint has no atom_weights (molecular_features off)")
    pt = GetPeriodicTable()
    cand = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Na", "Si", "B", "Se", "K"]
    tab = [(pt.GetAtomicWeight(pt.GetAtomicNumber(s)), s) for s in cand]
    return [min(tab, key=lambda t: abs(t[0] - w))[1] for w in weights]


# ===========================================================================
# Structural features + envelope
# ===========================================================================
def largest_ring_system(mol) -> int:
    """Number of rings in the largest group of rings connected by shared atoms (a
    fused OR spiro ring system). Detects oversized polycyclic 'blobs'. 0 if acyclic."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    n = len(rings)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if rings[i] & rings[j]:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    comp = Counter(find(i) for i in range(n))
    return max(comp.values())


def mol_features(mol) -> dict:
    """Structural fingerprint of a (sanitized) RDKit mol used by the sanity envelope."""
    sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    return {
        "ring_sizes": sizes,
        "num_rings": len(sizes),
        "largest_ring_system": largest_ring_system(mol),
        "num_spiro": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "num_bridgehead": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
    }


class SanityEnvelope:
    """Structural envelope derived from reference molecules. ``check(mol)`` returns
    ``(ok, reason)``; a molecule is sane iff every feature is inside the envelope."""

    def __init__(self, allowed_ring_sizes, max_num_rings, max_ring_system,
                 max_spiro, max_bridgehead):
        self.allowed_ring_sizes = set(int(s) for s in allowed_ring_sizes)
        self.max_num_rings = int(max_num_rings)
        self.max_ring_system = int(max_ring_system)
        self.max_spiro = int(max_spiro)
        self.max_bridgehead = int(max_bridgehead)

    def check(self, mol):
        f = mol_features(mol)
        for s in f["ring_sizes"]:
            if s not in self.allowed_ring_sizes:
                return False, f"ring_size:{s}"
        if f["num_rings"] > self.max_num_rings:
            return False, "num_rings"
        if f["largest_ring_system"] > self.max_ring_system:
            return False, "ring_system"
        if f["num_spiro"] > self.max_spiro:
            return False, "spiro"
        if f["num_bridgehead"] > self.max_bridgehead:
            return False, "bridgehead"
        return True, None

    def to_dict(self):
        return {
            "allowed_ring_sizes": sorted(self.allowed_ring_sizes),
            "max_num_rings": self.max_num_rings,
            "max_ring_system": self.max_ring_system,
            "max_spiro": self.max_spiro,
            "max_bridgehead": self.max_bridgehead,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["allowed_ring_sizes"], d["max_num_rings"], d["max_ring_system"],
                   d["max_spiro"], d["max_bridgehead"])


def _upper_bound(values, quantile):
    """Upper bound for a count feature: strict max at quantile 1.0 (every reference
    molecule passes), else the ceil of the requested quantile."""
    if not values:
        return 0
    return int(np.ceil(np.quantile(np.asarray(values, dtype=float), quantile)))


def build_envelope(mols_features, quantile=1.0, ring_min_count=1):
    """Build a :class:`SanityEnvelope` from a list of per-molecule feature dicts."""
    ring_counter = Counter()
    for f in mols_features:
        ring_counter.update(f["ring_sizes"])
    allowed = {s for s, c in ring_counter.items() if c >= ring_min_count}
    return SanityEnvelope(
        allowed_ring_sizes=allowed,
        max_num_rings=_upper_bound([f["num_rings"] for f in mols_features], quantile),
        max_ring_system=_upper_bound([f["largest_ring_system"] for f in mols_features], quantile),
        max_spiro=_upper_bound([f["num_spiro"] for f in mols_features], quantile),
        max_bridgehead=_upper_bound([f["num_bridgehead"] for f in mols_features], quantile),
    )


def ring_size_histogram(smiles_iter, ring_hist_max=12):
    """Normalized histogram of ring sizes aggregated over all rings in all molecules.
    Bins are 3..ring_hist_max individually + one overflow bin for larger rings."""
    bins = list(range(3, ring_hist_max + 1)) + ["overflow"]
    idx = {b: i for i, b in enumerate(bins)}
    h = np.zeros(len(bins), dtype=float)
    for smi in smiles_iter:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi
        if m is None:
            continue
        for r in m.GetRingInfo().AtomRings():
            s = len(r)
            h[idx[s] if s <= ring_hist_max else idx["overflow"]] += 1.0
    total = h.sum()
    return (h / total) if total > 0 else h, bins


def hist_divergence(p, q):
    """Total-variation distance and mean-absolute error between two histograms."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    tv = 0.5 * float(np.abs(p - q).sum())
    mae = float(np.abs(p - q).mean())
    return tv, mae


def compute_fcd_score(ref_smiles, gen_smiles, device):
    """FCD via fcd_torch (pure-PyTorch, MOSES standard). Returns None if unavailable
    or if there are too few valid generated molecules."""
    if not gen_smiles or len(gen_smiles) < 2:
        return None
    try:
        from fcd_torch import FCD
    except ImportError:
        return None
    try:
        dev = "cuda" if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        fcd = FCD(device=dev, n_jobs=min(8, os.cpu_count() or 1))
        return float(fcd(list(ref_smiles), list(gen_smiles)))
    except Exception:
        return None


def _read_reference_smiles(path, smiles_column, limit=None):
    path = os.path.expanduser(path)
    if path.lower().endswith(".csv"):
        import csv
        out = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            col = smiles_column if smiles_column in (reader.fieldnames or []) else (reader.fieldnames or [None])[0]
            for row in reader:
                out.append(row[col])
                if limit is not None and len(out) >= limit:
                    break
        return out
    with open(path) as fh:
        out = [ln.strip().split()[0] for ln in fh if ln.strip()]
    return out[:limit] if limit is not None else out


def prepare_reference(path, smiles_column, quantile, ring_min_count, limit,
                      fcd_ref_samples, ring_hist_max, seed=0, cache_dir="data"):
    """Parse the reference SMILES once and return
    ``(envelope, ref_canonical_set, ref_ring_hist, ring_bins, fcd_ref_smiles)``.

    The full bundle is cached to disk keyed on (path, mtime, quantile, ring_min_count,
    limit) so repeat runs (sweeps / re-anchoring) skip the ~minutes-long parse.
    """
    rp = os.path.expanduser(path)
    key = f"{os.path.abspath(rp)}|{os.path.getmtime(rp)}|q{quantile}|rmc{ring_min_count}|lim{limit}|fcd{fcd_ref_samples}|rh{ring_hist_max}"
    cache_path = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f".sanity_ref_{abs(hash(key))}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                blob = json.load(fh)
            if blob.get("key") == key:
                env = SanityEnvelope.from_dict(blob["envelope"])
                return (env, set(blob["ref_canonical"]), np.asarray(blob["ring_hist"], float),
                        blob["ring_bins"], blob["fcd_ref"])
    except Exception:
        cache_path = None

    raw = _read_reference_smiles(rp, smiles_column, limit)
    feats, canon = [], []
    for smi in raw:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        feats.append(mol_features(m))
        canon.append(Chem.MolToSmiles(m))
    if not feats:
        raise ValueError(f"no parseable reference molecules in {path}")
    envelope = build_envelope(feats, quantile=quantile, ring_min_count=ring_min_count)
    ref_ring_hist, ring_bins = ring_size_histogram(canon, ring_hist_max)
    rng = np.random.RandomState(seed)
    fcd_ref = list(rng.choice(canon, size=min(fcd_ref_samples, len(canon)), replace=False))
    ref_set = set(canon)

    if cache_path is not None:
        try:
            with open(cache_path, "w") as fh:
                json.dump({"key": key, "envelope": envelope.to_dict(),
                           "ref_canonical": sorted(ref_set), "ring_hist": ref_ring_hist.tolist(),
                           "ring_bins": [str(b) for b in ring_bins], "fcd_ref": fcd_ref}, fh)
        except Exception:
            pass
    return envelope, ref_set, ref_ring_hist, [str(b) for b in ring_bins], fcd_ref


# ===========================================================================
# Reward
# ===========================================================================
class StructuralSanityReward:
    """valid & (connected) & sane = 1.0; else 0.0. ``sane`` = every structural feature
    inside ``envelope``. Tracks the last batch's category / per-axis fractions in
    ``self.last`` for training-curve logging."""

    def __init__(self, domain, envelope, require_connected=True):
        self.domain = domain
        self.envelope = envelope
        self.require_connected = bool(require_connected)
        self.last = {}

    def __call__(self, X1, E1, node_mask):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = torch.zeros(len(datas))
        n_valid = n_disc = n_sane = 0
        viol = Counter()
        for i, d in enumerate(datas):
            smi = self.domain.identity(d)  # canonical SMILES iff genuinely valid, else None
            if smi is None:
                viol["invalid"] += 1
                continue
            n_valid += 1
            if self.require_connected and "." in smi:
                viol["disconnected"] += 1
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:  # defensive: identity already round-trips, so this is rare
                viol["invalid"] += 1
                continue
            ok, reason = self.envelope.check(mol)
            if ok:
                out[i] = 1.0
                n_sane += 1
            else:
                axis = reason.split(":")[0]
                viol[axis] += 1
        k = max(1, len(datas))
        conn_valid = n_valid - viol["disconnected"] if self.require_connected else n_valid
        self.last = {
            "sane_frac": n_sane / k,
            "valid_frac": n_valid / k,
            "disconnected_frac": viol["disconnected"] / k,
            "invalid_frac": viol["invalid"] / k,
            # artifact = valid&connected but structurally out-of-envelope (the "weird shape" rate)
            "artifact_frac_of_conn_valid": ((conn_valid - n_sane) / conn_valid) if conn_valid else 0.0,
            "viol_ring_size": viol["ring_size"] / k,
            "viol_num_rings": viol["num_rings"] / k,
            "viol_ring_system": viol["ring_system"] / k,
            "viol_spiro": viol["spiro"] / k,
            "viol_bridgehead": viol["bridgehead"] / k,
        }
        return out


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate_sanity(model, domain, envelope, ref_set, ref_ring_hist, fcd_ref,
                    n_samples, sample_steps, size_dist, device, eta, omega,
                    time_distortion, require_connected=True, ring_hist_max=12,
                    compute_fcd=False, seed=0):
    """Sample n_samples fresh molecules under the rollout policy and report the
    structural-violation rate (headline), the artifact rate among valid&connected
    molecules, validity / disconnected / uniqueness / novelty guardrails, ring-size
    histogram divergence, and (optionally) FCD -- plus example mols for a grid."""
    torch.manual_seed(seed)
    mols, smis, tags = [], [], []
    all_smis = []            # all valid SMILES (for uniqueness / novelty / FCD / ring-hist)
    n_valid = n_disc = n_sane = 0
    viol = Counter()
    remaining, chunk = n_samples, 64
    while remaining > 0:
        k = min(chunk, remaining)
        samples = model.sample(k, size_dist=size_dist, eta=eta, omega=omega,
                               sample_steps=sample_steps, time_distortion=time_distortion,
                               device=device, show_progress=False)
        for d in samples:
            smi = domain.identity(d)
            if smi is None:
                viol["invalid"] += 1
                continue
            n_valid += 1
            all_smis.append(smi)
            disconnected = "." in smi
            if disconnected:
                n_disc += 1
            mol = Chem.MolFromSmiles(smi)
            ok, reason = envelope.check(mol) if mol is not None else (False, "invalid")
            sane = ok and not (require_connected and disconnected)
            if sane:
                n_sane += 1
            else:
                if require_connected and disconnected:
                    viol["disconnected"] += 1
                elif not ok:
                    viol[reason.split(":")[0]] += 1
            if len(mols) < 25:
                mols.append(mol); smis.append(smi)
                tags.append("" if sane else ("disc" if disconnected else (reason or "")))
        remaining -= k

    f = lambda x: x / n_samples
    conn_valid = n_valid - n_disc if require_connected else n_valid
    uniq = set(all_smis)
    gen_ring_hist, _ = ring_size_histogram(all_smis, ring_hist_max)
    ring_tv, ring_mae = hist_divergence(gen_ring_hist, ref_ring_hist)
    fcd = None
    if compute_fcd:
        fcd = compute_fcd_score(fcd_ref, all_smis, device)

    result = {
        "n": n_samples,
        "valid_frac": f(n_valid),
        "sane_frac_all": f(n_sane),                       # = E[reward]
        "violation_frac_all": 1.0 - f(n_sane),            # HEADLINE (lower is better)
        "disconnected_frac_all": f(n_disc),
        "disconnected_frac_of_valid": (n_disc / n_valid) if n_valid else 0.0,
        "artifact_frac_of_conn_valid": ((conn_valid - n_sane) / conn_valid) if conn_valid else 0.0,
        "unique_frac_of_valid": (len(uniq) / n_valid) if n_valid else 0.0,
        "novel_frac_of_valid": (len(uniq - ref_set) / n_valid) if (n_valid and ref_set) else 0.0,
        "ring_tv": ring_tv, "ring_mae": ring_mae,
        "fcd": fcd,
        "viol": {ax: f(c) for ax, c in viol.items()},
        "gen_ring_hist": gen_ring_hist.tolist(),
    }
    return result, mols, smis, tags


def save_grid(mols, smis, tags, path):
    if not mols:
        return
    legends = [(f"[{t}] " if t else "") + s[:22] for s, t in zip(smis, tags)]
    Draw.MolsToGridImage([m for m in mols[:25] if m is not None][:len(mols)],
                         molsPerRow=5, subImgSize=(240, 240),
                         legends=legends[:25]).save(path)


def save_curves(history, before_viol, path):
    def smooth(x, w=7):
        x = np.asarray(x, float)
        return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x
    rm = [h["reward_mean"] for h in history]
    art = [h.get("artifact_frac_of_conn_valid", np.nan) for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(rm, color="#bbb", lw=1, alpha=0.6)
    a1.plot(range(len(smooth(rm))), smooth(rm), color="#2c7fb8", lw=2)
    a1.set_xlabel("iteration"); a1.set_ylabel("rollout mean reward (1=valid&connected&sane)")
    a1.set_title("reward (sane fraction)"); a1.grid(alpha=0.3)
    a2.plot(art, color="#f4a582", lw=1, alpha=0.6)
    a2.plot(range(len(smooth(art))), smooth(art), color="#d95f0e", lw=2)
    a2.axhline(before_viol, color="k", ls="--", lw=1, label=f"before eval violation ({before_viol:.1%})")
    a2.set_xlabel("iteration"); a2.set_ylabel("rollout artifact fraction (of valid&connected)")
    a2.set_title("weird-shape fraction (rollout)"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.suptitle("GDPO structural-sanity fine-tune")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"GDPO sanity: ckpt={os.path.basename(e.CKPT_PATH)} ref={os.path.basename(e.REFERENCE_SMILES)} "
          f"K={e.ROLLOUT_SIZE} iters={e.ITERATIONS} kl_coef={e.KL_COEF} require_connected={e.REQUIRE_CONNECTED}")
    from pytorch_lightning import seed_everything
    seed_everything(e.SEED, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeFoGModel.load(e.CKPT_PATH, device="cpu").to(device)
    atom_types = atom_decoder_from_ckpt(model)
    ae, ad, be, bd = build_encoders(atom_types, BOND_TYPES)
    assert len(ad) == model.num_node_classes, \
        f"atom decoder {len(ad)} != model node classes {model.num_node_classes}"
    domain = MoleculeDomain(ad, bd)
    size_dist = model.default_size_dist
    e.log(f"atoms ({len(ad)}): {ad}")

    # --- build the structural envelope from the training reference ---
    t0 = time.time()
    envelope, ref_set, ref_ring_hist, ring_bins, fcd_ref = prepare_reference(
        e.REFERENCE_SMILES, e.SMILES_COLUMN, e.ENVELOPE_QUANTILE, e.RING_MIN_COUNT,
        e.REFERENCE_LIMIT, e.FCD_REF_SAMPLES, e.RING_HIST_MAX, seed=e.SEED)
    e["envelope"] = envelope.to_dict()
    e["ring_bins"] = ring_bins
    e["results/ref_ring_hist"] = ref_ring_hist.tolist()
    e.log(f"reference: {len(ref_set)} mols, fcd_ref={len(fcd_ref)}, built in {time.time()-t0:.0f}s")
    e.log(f"envelope: ring_sizes={sorted(envelope.allowed_ring_sizes)} max_rings={envelope.max_num_rings} "
          f"max_ring_system={envelope.max_ring_system} max_spiro={envelope.max_spiro} "
          f"max_bridgehead={envelope.max_bridgehead}")

    eval_kw = dict(eta=e.ETA, omega=e.OMEGA, time_distortion=e.TIME_DISTORTION,
                   require_connected=e.REQUIRE_CONNECTED, ring_hist_max=e.RING_HIST_MAX)

    def full_eval(m, n, steps, seed, fcd):
        return evaluate_sanity(m, domain, envelope, ref_set, ref_ring_hist, fcd_ref,
                               n, steps, size_dist, device, compute_fcd=fcd, seed=seed, **eval_kw)

    before, mols_b, smis_b, tags_b = full_eval(model, e.EVAL_SAMPLES, e.EVAL_STEPS, e.SEED, e.COMPUTE_FCD)
    e["results/before"] = before
    e.log(f"BEFORE: violation={before['violation_frac_all']:.1%} artifact(of conn&valid)="
          f"{before['artifact_frac_of_conn_valid']:.1%} valid={before['valid_frac']:.1%} "
          f"disc={before['disconnected_frac_all']:.1%} unique={before['unique_frac_of_valid']:.1%} "
          f"novel={before['novel_frac_of_valid']:.1%} ring_tv={before['ring_tv']:.3f} fcd={before['fcd']}")
    e.log(f"BEFORE violations by axis: {before['viol']}")
    save_grid(mols_b, smis_b, tags_b, os.path.join(e.path, "grid_before.png"))

    reward = StructuralSanityReward(domain, envelope, require_connected=e.REQUIRE_CONNECTED)
    ckpt_dir = os.path.join(e.path, "ckpts")
    if e.CKPT_EVERY > 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    round_results = []
    mols_a, smis_a, tags_a = mols_b, smis_b, tags_b

    for r in range(e.ROUNDS):
        trainer = GDPOTrainer(
            model, reward, rollout_size=e.ROLLOUT_SIZE, sample_steps=e.SAMPLE_STEPS,
            subsample_steps=e.SUBSAMPLE_STEPS, minibatch_size=e.MINIBATCH_SIZE,
            eta=e.ETA, omega=e.OMEGA, time_distortion=e.TIME_DISTORTION, size_dist=size_dist,
            advantage_mode=e.ADVANTAGE_MODE, reduction=e.REDUCTION, lambda_edge=e.LAMBDA_EDGE,
            positive_only=e.POSITIVE_ONLY,
            kl_coef=e.KL_COEF, kl_anchor=e.KL_ANCHOR, anchor_decay=e.ANCHOR_DECAY, kl_target=e.KL_TARGET,
            lr=e.LR, ema_decay=e.EMA_DECAY, device=device, seed=e.SEED + r,
        )

        def on_iter(it, m, _r=r, _tr=trainer):
            m = {**m, "round": _r, "iter": it, **reward.last}
            history.append(m)
            for key in ("reward_mean", "sane_frac", "artifact_frac_of_conn_valid", "valid_frac",
                        "disconnected_frac", "invalid_frac", "kl", "kl_coef", "grad_norm"):
                if key in m:
                    e.track(key, float(m[key]))
            if it % 5 == 0 or it == e.ITERATIONS - 1:
                e.log(f"  r{_r} iter {it:3d} reward={m['reward_mean']:+.3f} sane={m.get('sane_frac',0):.2f} "
                      f"artifact={m.get('artifact_frac_of_conn_valid',0):.2f} valid={m.get('valid_frac',0):.2f} "
                      f"disc={m.get('disconnected_frac',0):.2f} gnorm={m['grad_norm']:.1f} klc={m.get('kl_coef',0):.3f}")
            if e.CKPT_EVERY > 0 and (it + 1) % e.CKPT_EVERY == 0:
                _tr.save(os.path.join(ckpt_dir, f"round{_r}_iter{it + 1:04d}.ckpt"))
                e.commit_json("history.json", history)

        trainer.fit(e.ITERATIONS, on_iter=on_iter)
        if trainer.ema is not None:
            trainer.ema.copy_to(model)

        last = (r == e.ROUNDS - 1)
        n_eval = e.EVAL_SAMPLES if last else e.ROUND_EVAL_SAMPLES
        rev, mols_a, smis_a, tags_a = full_eval(model, n_eval, e.EVAL_STEPS, e.SEED + 100 + r, e.COMPUTE_FCD and last)
        round_results.append(rev)
        e[f"results/round_{r}"] = rev
        e.track("round_violation", float(rev["violation_frac_all"]))
        e.track("round_valid", float(rev["valid_frac"]))
        e.log(f"ROUND {r}: violation={rev['violation_frac_all']:.1%} artifact={rev['artifact_frac_of_conn_valid']:.1%} "
              f"valid={rev['valid_frac']:.1%} unique={rev['unique_frac_of_valid']:.1%} (n={n_eval})")

    after = round_results[-1]

    # --- best-snapshot selection: minimize the structural-violation rate, gated on
    #     validity AND uniqueness (reject any collapse). Input-inclusive so a stacking
    #     run can never ship worse than what it started from. ---
    best_snapshot = None
    if e.SELECT_BEST and e.CKPT_EVERY > 0:
        import glob
        val_floor = before["valid_frac"] - 0.02
        uniq_floor = before["unique_frac_of_valid"] - 0.02
        cands = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        e.log(f"best-snapshot: ranking {len(cands)} snapshots + final at {e.SELECT_EVAL_STEPS} steps "
              f"(val_floor={val_floor:.1%} uniq_floor={uniq_floor:.1%})")

        def rank_eval(m):
            rr, _, _, _ = full_eval(m, e.SELECT_EVAL_SAMPLES, e.SELECT_EVAL_STEPS, e.SEED + 200, False)
            return rr

        def gated_better(rev, best):
            if best is None:
                return True
            return (rev["valid_frac"] >= val_floor and rev["unique_frac_of_valid"] >= uniq_floor
                    and rev["violation_frac_all"] < best[0])

        best = None
        if e.SELECT_INCLUDE_INPUT:
            inp = DeFoGModel.load(e.CKPT_PATH, device="cpu").to(device)
            ir = rank_eval(inp)
            e.log(f"  input: violation={ir['violation_frac_all']:.1%} valid={ir['valid_frac']:.1%} "
                  f"unique={ir['unique_frac_of_valid']:.1%}")
            best = (ir["violation_frac_all"], ir["valid_frac"], e.CKPT_PATH)
            del inp
            torch.cuda.empty_cache()
        for tag, path in [("final", None)] + [(os.path.basename(sp), sp) for sp in cands]:
            if path is None:
                rev = rank_eval(model)
            else:
                cm = DeFoGModel.load(path, device="cpu").to(device)
                rev = rank_eval(cm)
                del cm
                torch.cuda.empty_cache()
            e.log(f"  {tag}: violation={rev['violation_frac_all']:.1%} valid={rev['valid_frac']:.1%} "
                  f"unique={rev['unique_frac_of_valid']:.1%}")
            if gated_better(rev, best):
                best = (rev["violation_frac_all"], rev["valid_frac"], path)
        best_snapshot = os.path.basename(best[2]) if best[2] else "final"
        e.log(f"best-snapshot: WINNER = {best_snapshot} (rank violation {best[0]:.1%}, valid {best[1]:.1%})")
        if best[2] is not None:
            model = DeFoGModel.load(best[2], device="cpu").to(device)
        after, mols_a, smis_a, tags_a = full_eval(model, e.EVAL_SAMPLES, e.EVAL_STEPS, e.SEED + 1, e.COMPUTE_FCD)

    e["results/after"] = after
    save_grid(mols_a, smis_a, tags_a, os.path.join(e.path, "grid_after.png"))
    save_curves(history, before["artifact_frac_of_conn_valid"], os.path.join(e.path, "reward_curve.png"))

    summary = {
        "violation_before": before["violation_frac_all"], "violation_after": after["violation_frac_all"],
        "artifact_before": before["artifact_frac_of_conn_valid"], "artifact_after": after["artifact_frac_of_conn_valid"],
        "valid_before": before["valid_frac"], "valid_after": after["valid_frac"],
        "disc_before": before["disconnected_frac_all"], "disc_after": after["disconnected_frac_all"],
        "unique_after": after["unique_frac_of_valid"], "novel_after": after["novel_frac_of_valid"],
        "ring_tv_before": before["ring_tv"], "ring_tv_after": after["ring_tv"],
        "fcd_before": before["fcd"], "fcd_after": after["fcd"],
        "ratchet": [round(rr["violation_frac_all"], 4) for rr in round_results],
        "best_snapshot": best_snapshot,
        "envelope": envelope.to_dict(),
        "ROUNDS": e.ROUNDS, "ITERATIONS": e.ITERATIONS, "KL_COEF": e.KL_COEF, "EVAL_STEPS": e.EVAL_STEPS,
        "REQUIRE_CONNECTED": e.REQUIRE_CONNECTED, "ROLLOUT_SIZE": e.ROLLOUT_SIZE,
    }
    e["results/summary"] = summary
    e.commit_json("summary.json", summary)
    e.commit_json("history.json", history)
    model.save(os.path.join(e.path, "gdpo_sane.ckpt"))
    e.log(f"SUMMARY violation {summary['violation_before']:.1%} -> {summary['violation_after']:.1%} "
          f"(best={summary['best_snapshot']}) | artifact {summary['artifact_before']:.1%} -> {summary['artifact_after']:.1%} "
          f"| valid {summary['valid_before']:.1%} -> {summary['valid_after']:.1%} "
          f"| disc {summary['disc_before']:.1%} -> {summary['disc_after']:.1%} "
          f"| unique {summary['unique_after']:.1%} | ring_tv {summary['ring_tv_before']:.3f} -> {summary['ring_tv_after']:.3f} "
          f"| fcd {summary['fcd_before']} -> {summary['fcd_after']}")


@experiment.testing
def testing(e: Experiment) -> None:
    e.ROUNDS = 1
    e.ITERATIONS = 4
    e.ROLLOUT_SIZE = 8
    e.SAMPLE_STEPS = 20
    e.SUBSAMPLE_STEPS = 2
    e.MINIBATCH_SIZE = 4
    e.CKPT_EVERY = 2
    e.SELECT_BEST = True
    e.SELECT_EVAL_SAMPLES = 16
    e.EVAL_SAMPLES = 16
    e.ROUND_EVAL_SAMPLES = 16
    e.EVAL_STEPS = 20
    e.SELECT_EVAL_STEPS = 20
    e.REFERENCE_LIMIT = 2000
    e.RING_MIN_COUNT = 1          # tiny reference -> keep full support so the smoke envelope is sensible
    e.FCD_REF_SAMPLES = 256
    e.COMPUTE_FCD = False


experiment.run_if_main()
