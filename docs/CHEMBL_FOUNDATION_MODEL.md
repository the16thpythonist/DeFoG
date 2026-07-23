# DeFoG-ChEMBL — Unconditional Molecular Foundation Model

A general-purpose **unconditional** DeFoG (Discrete Flow Matching) model for
drug-like molecular graph generation, trained on cleaned ChEMBL 37. It is meant
as a *frozen base* that downstream work (property adapters, guidance, RL
fine-tuning, inpainting) builds on top of — so its categorical schema is a fixed
public contract.

- **Merged to:** `main` (developed on `feat/chembl-foundation`)
- **Checkpoints:**
  - **v2 (recommended):** `ckpts/chembl_foundation_v2_kl0.2.ckpt` — v1 + GDPO
    structural-sanity RL (see "v2" below). validity 0.926 / sanity 0.908.
  - **v1 (base):** `ckpts/chembl_foundation_lr3e-4/best_model.ckpt` (EMA, ~epoch 39, 104 MB).
- **Architecture:** DeFoG graph transformer, 25,922,897 parameters

---

## Frozen schema (the public contract)

Everything built on this model must use the **same** categorical encoding:

| Field | Value |
|---|---|
| Node classes (atoms) | `[C, N, O, F, B, Br, Cl, I, P, S, Se, Si]` (12) |
| Edge classes (bonds) | `[none, single, double, triple, aromatic]` (5; class 0 = no edge) |
| Hydrogens | Implicit (heavy-atom graphs only) |
| Formal charges | Recovered at decode time (not a generated channel) |
| Max graph size | ≤ 48 heavy atoms (min 3) |
| Noise prior | Marginal (empirical node/edge marginals) |

Node marginals (train): C 0.736, N 0.120, O 0.102, F 0.018, S 0.013, Cl 0.008,
Br 0.002 (B, I, P, Se, Si each < 0.001). Edge marginals: none 0.925, single 0.035,
aromatic 0.036, double 0.004, triple 0.0002.

Encoders come from `experiments.utils.build_encoders(ATOM_DECODER, BOND_TYPES)`;
graph ↔ molecule conversion is `smiles_to_pyg_data` / `pyg_data_to_mol` in
`defog.domains.molecule`.

---

## Training data — ChEMBL 37 (cleaned)

Source: `chembl_37_chemreps.txt.gz` (EBI, release 37). Standardized by
`scripts/prepare_chembl.py` (structural-sanity filter only — no drug-likeness /
PAINS filtering):

- reject multi-fragment entries (salts/mixtures) outright — no desalting;
- keep only the 12 in-vocabulary elements; 3 ≤ heavy atoms ≤ 48;
- drop any ring of size ≥ 9 ("wonky" macrocycles);
- strip stereochemistry + isotopes; keep formal charges; drop radicals;
- deduplicate on stereo-free canonical SMILES.

**2,897,819 raw → 2,485,401 unique kept (85.8%).** Drops: too_large 146,397 ·
multifragment 120,725 · duplicate 117,762 · wonky_ring 25,116 · element 1,794 ·
radical 582. Split (seed 42): **train 2,435,693 / val 24,854 / test 24,854** (98/1/1).

---

## Model & training

| | |
|---|---|
| Layers / hidden / MLP / heads | 12 / 384 / 768 / 12 |
| Extra features | RRWP (20 steps) + molecular features (per-atom charge/valency, MW) |
| Parameters | 25,922,897 |
| Noise | Marginal; train time-distortion `polydec` |
| Optimizer | AdamW, lr 3e-4 (cosine, horizon 60 ep), weight decay 1e-5, λ_edge 5.0 |
| EMA | 0.9999 (released weights are the EMA weights) |
| Batch | Effective 256 (per-rank 64 × 4-GPU DDP) |
| Hardware | 1 node × 4 NVIDIA GH200 (JUPITER/JSC), DDP |
| Trained | ~39 epochs (3 chained ~9.5h links; training in diminishing returns) |

The learning rate (3e-4) was chosen by a 4-way single-GPU ablation
{1e-4, 2e-4, 3e-4, 4e-4} on the extended metric suite; 3e-4 won on sanity /
connectivity / KL.

---

## Evaluation

1000-/2000-sample unconditional generation, 500 sampling steps. Extended suite
(`defog.domains.molecule.molecular_metrics`): beyond validity/uniqueness/novelty
it reports **sanity** (valid AND single-fragment AND all rings ∈ [3,8]),
**connected** / disconnected, wonky-ring fraction, and **KL divergence** of
logP / TPSA / QED vs the training distribution (Gaussian-KDE, GuacaMol-style;
`kl_score = exp(−mean KL)`).

**Final metrics** (release config below, 2000 samples):

| Metric | Value |
|---|---|
| validity | 0.845 |
| uniqueness | 1.000 |
| novelty | 0.998 |
| sanity | 0.825 |
| connected | 0.983 |
| wonky-ring frac | 0.009 |
| KL logP / TPSA / QED | 0.011 / 0.024 / 0.006 |
| kl_score | 0.986 |

Training progression (η=0 eval): sanity 0.579 (~ep5, ablation) → 0.772 (~ep13) →
0.823 (~ep26) → 0.825 (~ep39); connectivity 0.823 → 0.938 → 0.975 → 0.983. Raw
validity/sanity had largely converged by ~ep26; further training mostly improved
connectivity and distribution match.

---

## Recommended sampling config

```python
from defog.core import DeFoGModel
model = DeFoGModel.load("ckpts/chembl_foundation_lr3e-4/best_model")  # no .ckpt suffix
samples = model.sample(
    num_samples=1000,
    eta=0.0,            # error-correction stochasticity (η); molecular data needs little
    omega=0.05,         # target guidance; light guidance marginally improves KL/novelty
    sample_steps=500,
    time_distortion="polydec",
)
```

An η × ω grid (η ∈ {0,5,25,50,100} × ω ∈ {0,0.05,0.1}) was swept
(`scripts/train_chembl_ddp.py --sweep`). The configs are within sampling noise;
**η=0, ω=0.05, polydec** is shipped as the best all-around (no KL/novelty cost).
Higher ω (0.1) buys a hair more raw validity but degrades KL; η barely helps
(unlike planar graphs). Decode samples to molecules with
`pyg_data_to_mol(sample, atom_decoder, bond_decoder)`.

---

## Reproduce

All on branch `feat/chembl-foundation`:

```bash
# 1. Prepare data (download chembl_37_chemreps.txt.gz to data/chembl/raw/ first)
python scripts/prepare_chembl.py

# 2. Train (4-GPU DDP, chained 12h links; auto-resumes from last.ckpt)
sbatch run_chembl_ddp_chain_jupiter.sh          # repeat per link

# 3. Extended eval / sampling sweep (single GPU)
sbatch run_chembl_eval_jupiter.sh  ckpts/chembl_foundation_lr3e-4/best_model.ckpt
sbatch run_chembl_sweep_jupiter.sh ckpts/chembl_foundation_lr3e-4/best_model.ckpt
```

Key code: `scripts/prepare_chembl.py` (standardization), `scripts/train_chembl_ddp.py`
(DDP train + resumable checkpointing + `--eval-only` / `--sweep`),
`defog.domains.molecule.molecular_metrics` (extended metrics). Tests:
`tests/test_molecular_metrics.py`, `tests/test_resume.py`.

Preserved milestone checkpoints on JUPITER:
`ckpts/chembl_foundation_snapshots/{link1_ep12,link2_final,link3_ep39}/`.

---

## v2 — GDPO structural-sanity RL improvement

The v1 base leaves ~15% of generations invalid/non-sane. v2 closes most of that
gap by **reward-aligning the base weights** with GDPO (Graph Diffusion Policy
Optimization, eager policy gradient) toward a **binary structural-sanity reward**:

```
reward(mol) = 1.0  iff  valid  AND  connected ('.'-free)  AND  rings within a
              ChEMBL-derived envelope (sizes ∈ [3,8], ring-system/spiro/bridgehead
              bounds); else 0.0
```

The reward's optimum is fidelity to real molecules (every genuine ChEMBL molecule
scores 1.0), so it has no degenerate "benzene-everywhere" optimum. Guards against
reward-hacking / diversity collapse: a **KL-to-frozen-v1 penalty**, monitored
**validity/uniqueness/novelty rails**, and **best-snapshot selection that can
never ship worse than the input** (`experiments/gdpo_sanity.py`).

**KL_COEF sweep** (4-GPU, one arm per GPU; K=128 rollouts, 100 iters, before/after
on 2048 samples). Structural-violation rate (= 1 − sanity, GDPO envelope):

| KL_COEF | violation 15.7% → | validity 86.8% → | outcome |
|---|---|---|---|
| 0.05 | 9.1% | 92.9% | collapsed mid-run; best-snapshot fell back to an early (iter-40) checkpoint |
| 0.10 | 8.2% | 93.8% | strongest gain, but largest property drift |
| **0.20** | **8.7%** | **93.2%** | **shipped — Pareto-best (gain with ~zero drift)** |
| 0.50 | 13.2% | 89.1% | over-anchored, least gain |

**v1 → v2 (KL=0.2), extended suite, η=0/ω=0, 1000 samples, 500 steps:**

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| validity | 0.845 | **0.926** | +0.081 |
| sanity | 0.825 | **0.908** | +0.083 |
| connected | 0.981 | 0.984 | +0.003 |
| uniqueness | 1.000 | 1.000 | — |
| novelty | 0.991 | 0.991 | — (held) |
| kl_logp / tpsa / qed | 0.013 / 0.031 / 0.003 | 0.018 / 0.027 / 0.004 | ~flat |
| kl_score | 0.985 | 0.984 | −0.001 (held) |

v2 lifts validity/sanity ~+8 points while **preserving diversity (uniqueness/
novelty) and the property-distribution match** (kl_score essentially unchanged).
KL=0.1 reaches slightly higher validity (0.943) but at 4× the property drift
(kl_score 0.974), so KL=0.2 is shipped. Same schema and sampling config as v1
(η=0, ω=0.05, polydec, 500 steps) — v2 is a drop-in replacement.

Reproduce: `run_chembl_gdpo_sweep_jupiter.sh` (4-GPU KL sweep) then
`run_chembl_eval_jupiter.sh ckpts/chembl_foundation_v2_kl0.2.ckpt`.
