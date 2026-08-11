# DeFoG-Union — Unconditional Molecular Foundation Model (kekulized)

A general-purpose **unconditional** DeFoG model for drug-like molecular graph
generation, trained on ~102M molecules drawn from ZINC20-druglike ∪ ChEMBL 37.
It supersedes [DeFoG-ChEMBL](CHEMBL_FOUNDATION_MODEL.md) v1/v2 and is intended as
the frozen base that downstream work (property adapters, guidance, RL
fine-tuning, inpainting) builds on.

> **Status: complete pending distribution.** Training, the η/ω sweep and the
> frozen n=10,000 evaluation are all done. The only open item is deciding how the
> checkpoint is distributed — it currently exists only on JUPITER.

- **Developed on:** `feat/kekulized-foundation`
- **Checkpoint:** `ckpts/foundation_union_kek_snapshots/link6_final/foundation_model.ckpt`
  (EMA weights, 2 epochs, ~104 MB, on JUPITER)
- **Architecture:** DeFoG graph transformer, **25,922,128** parameters

---

## Why this model exists: the aromatic representation was lossy

The predecessor's vocabulary carried an `AROMATIC` bond class. That class is a
promise about a whole ring system which RDKit verifies by kekulizing, and which
a model asserting bonds independently cannot keep. The cost was measured three
ways before anything was retrained:

**1. It could not round-trip its own training data.** Encoding real molecules to
graphs and decoding them back through the shipped `pyg_data_to_mol` path
(`scripts/check_representation_roundtrip.py`, which runs every declared
representation side by side so the shipped one acts as a control):

| would score INVALID | aromatic | kekulized |
|---|---|---|
| ChEMBL (n=50,000) | **14.07%** | 0.02% |
| ZINC cleaned (n=299,794) | **8.07%** | 1 molecule |
| union train (n=99,830,353) | — | **0 encode failures** |

The 0.02% residue is 10 `AtomValenceException`s that are *identical* under both
vocabularies — representation-independent. Formal-charge notation loss (nitro
groups) is likewise present in both and is a property of charge not being a
generated channel.

**2. It caused most of the model's failures.** `scripts/diagnose_validity.py` on
the released v2 checkpoint (n=2048): **120 of 129 hard failures were
kekulization errors — 93%**, against 98% measured on MOSES.

**3. Removing it dominated everything else tried.** A controlled A/B holding lr,
cosine horizon, batch, architecture and data fixed, changing only the
vocabulary — one 9.5 h link on ChEMBL:

| | aromatic (~ep13) | kekulized (ep12) |
|---|---|---|
| validity | 0.838 | **0.9840** |
| sanity | 0.772 | **0.9400** |

One kekulized link beat the aromatic v1 (39 epochs, 3 links) *and* v2 (v1 plus a
GDPO RL round). Note the model's kekulize failure rate (5.9%) was *below* the
data's round-trip loss (14.1%): it was not reproducing the un-decodable
chemistry, it was avoiding it — paying in distribution match rather than
validity. Kekulizing dissolves that trade-off instead of pricing it.

---

## Frozen schema (the public contract)

**This is a new contract and is NOT compatible with DeFoG-ChEMBL v1/v2**, which
are 12 atom / 5 edge. Decoding either with the other's vocabulary does not raise
— it yields plausible molecules made of the wrong elements. `defog/data/vocabulary.py`
turns that into an error; pass `--representation kekulized_v2` to anything
touching these checkpoints.

| Field | Value |
|---|---|
| Node classes (atoms) | `[C, N, O, F, B, Br, Cl, I, P, S, Se, Si]` (12) |
| Edge classes (bonds) | `[none, single, double, triple]` (4; class 0 = no edge) |
| Aromaticity | **Kekulized** — no AROMATIC class |
| Hydrogens | Implicit (heavy-atom graphs only) |
| Formal charges | Recovered at decode time (not a generated channel) |
| Graph size | 3 ≤ heavy atoms ≤ 48 |
| Noise prior | Marginal (empirical node/edge marginals) |

Node marginals (train): C 0.7292, N 0.1293, O 0.1048, F 0.0177, S 0.0116,
Cl 0.0047, Br 0.0023, I 0.0002, B 0.0001, P 0.0001 (Se, Si < 0.0001).
Edge marginals: none 0.90947, single 0.07260, double 0.01771, triple 0.00022.

Relative to ChEMBL alone the union is more nitrogen-rich and more densely bonded
per atom pair — which is why the prior is recomputed per dataset *and* per
representation rather than inherited (`scripts/compute_graph_stats.py`).

---

## Training data — ZINC20-druglike ∪ ChEMBL 37

Built by `scripts/prepare_smiles_union.py`. The source tarball is streamed and
never unpacked; the ~1.8 B rows are **uniformly reservoir-sampled** because ZINC
is tranche-ordered by molecular weight, so a first-N slice would be badly biased.

| | |
|---|---|
| Source scanned | ~1,800,000,000 rows (`zinc-druglike-cano.tar.xz`, 11.9 GiB → 80.8 GiB) |
| Reservoir | 105,000,000 (oversample 1.05) |
| ZINC kept unique | 99,544,563 |
| Already in ChEMBL (cross-dedup) | 112,549 |
| **Union total** | **101,867,707** (99,432,014 ZINC + 2,435,693 ChEMBL train) |
| Split (seed 42, 98/1/1) | train 99,830,353 / val 1,018,677 / test 1,018,677 |

Drops from the reservoir: duplicate 5,405,683 (5.15%) · wonky_ring 43,710 ·
radical 5,963 · element 63 · too_large 18 · **kekulize 0**.

Two things worth recording about that table. **Zero of 105 M molecules failed to
kekulize**, and only 63 fell outside the 12-element vocabulary — the frozen
schema covers ZINC essentially completely. And the duplicate rate is *not* a
constant: duplicates here are stereoisomers collapsing onto one stereo-free
canonical SMILES, so the rate scales with sampling density (0.016% at 300 k of
1.8 B; 19.3% at 5.05 M of 20 M). Fitting both points gives ~2.65 source rows per
distinct molecule, which predicted 4.6% at the real density against 5.15%
observed. Do not read a small-sample keep rate as a fixed filter yield.

Filtering (identical to `prepare_chembl.py`, plus the kekulize check): reject
multi-fragment outright; 12 in-vocabulary elements only; 3 ≤ heavy ≤ 48; drop any
ring ≥ 9; strip stereochemistry and isotopes; keep formal charges; drop radicals;
deduplicate on stereo-free canonical SMILES.

---

## Model & training

| | |
|---|---|
| Layers / hidden / MLP / heads | 12 / 384 / 768 / 12 |
| Extra features | RRWP (20 steps) + molecular features (charge/valency, MW) |
| Parameters | 25,922,128 |
| Noise | Marginal; train time-distortion `polydec` |
| Optimizer | AdamW, lr 3e-4 (cosine, horizon 2 epochs), weight decay 1e-5, λ_edge 5.0 |
| EMA | 0.9999 (released weights are the EMA weights) |
| Batch | Effective 256 (per-rank 64 × 4-GPU DDP) |
| Hardware | 1 node × 4 NVIDIA GH200 (JUPITER/JSC) |
| Trained | **2 epochs, 778,000 steps**, 6 chained ~9.5 h links (~57 GPU-hours) |

Model size was deliberately held at the predecessor's 25.9 M so that the 40×
increase in data is the only variable.

One operational note for anyone resuming this recipe: **`gen_every_k` counts
epochs**, and at ~30 h/epoch no in-training probe fires at all — the first five
links produced no validation pass and no `best_model.ckpt`. Use the end-of-link
`foundation_model.ckpt`. Separately, on the ChEMBL A/B the 64-sample η=5 probe
selected a checkpoint that the η=0 eval rated *worse* on every metric, so probe-
based best-selection needs a far larger sample before it is worth acting on.

---

## Evaluation

Unconditional generation at 500 steps, scored with
`defog.domains.molecule.molecular_metrics`: validity / uniqueness / novelty,
**sanity** (valid AND single-fragment AND all rings ∈ [3,8]), connectivity,
wonky-ring fraction, and KDE KL of logP / TPSA / QED against the training
distribution (`kl_score = exp(−mean KL)`).

**RELEASE FIGURES — n=10,000, η=25 / ω=0.1, 500 steps, polydec:**

| Metric | Value |
|---|---|
| validity | **0.9967** |
| uniqueness | 1.0000 |
| novelty | 0.9754 |
| sanity | **0.9897** |
| connected | 0.9950 |
| disconnected | 0.0050 |
| wonky-ring frac | 0.0028 |
| KL logP / TPSA / QED | 0.0097 / 0.0015 / 0.0066 |
| kl_score | **0.9941** |

At n=10,000 the standard error on validity is ±0.00055. This is a single run, not
replicated; the MOSES-derived noise floors (FCD ±0.0084, sanity ±0.0020 at
n=10,000) are the reference for judging differences of this size.

**The same model at η=0 / ω=0** (n=5000), for the contribution of the sampling
config: validity 0.9946, sanity 0.9824, connected 0.9922, novelty 0.9813,
kl_score 0.9807. So the swept config buys +0.007 sanity, +0.002 validity and
**+0.013 kl_score** — the TPSA divergence alone improves nearly tenfold
(0.0143 → 0.0015) — at a cost of ~0.006 novelty.

**Versus the predecessors** (their figures are against ChEMBL's reference; KL and
novelty are therefore *not* directly comparable across lineages — each model is
scored against its own training distribution):

| | v1 aromatic (39 ep) | v2 (v1 + GDPO RL) | ChEMBL-kek (12 ep) | **union-kek (release)** |
|---|---|---|---|---|
| validity | 0.845 | 0.926 | 0.984 | **0.9967** |
| sanity | 0.825 | 0.908 | 0.940 | **0.9897** |
| connected | 0.983 | 0.984 | 0.963 | **0.9950** |
| wonky_ring | 0.009 | — | 0.021 | **0.0028** |
| kl_score | 0.986 | 0.984 | 0.984 | **0.9941** |
| novelty | 0.998 | 0.991 | 0.997 | 0.9754 |

**Read the novelty row with care.** Novelty is the fraction of valid samples
absent from the *training set*, and this model's training set is 99.8M molecules
against v1's 2.4M — a 41× larger denominator makes a collision far more likely, so
0.9754 here is not the same measurement as 0.998 there and is arguably the harder
number. Part of the remainder is a genuine trade from η>0, which buys structural
quality with a little diversity.

**Training progression** (all n=5000, η=0):

| | link 3 (1 ep) | link 4 (1⅓ ep) | link 5 (1⅔ ep) | link 6 (2 ep) |
|---|---|---|---|---|
| validity | 0.9904 | 0.9938 | 0.9934 | 0.9946 |
| sanity | 0.9746 | 0.9842 | 0.9840 | 0.9824 |
| connected | 0.9895 | 0.9934 | 0.9932 | 0.9922 |
| kl_score | 0.9825 | 0.9839 | 0.9836 | 0.9807 |

The model converged by ~1⅓ epochs: link 3 → 4 moved sanity +0.0096 (3.4σ), and
every change from link 4 onward is inside ±0.003, i.e. noise at n=5000. The
cosine decay to LR floor added nothing measurable. **This looks like a capacity
limit rather than a data limit** — 100 M molecules were exhausted by a 25.9 M
parameter model after roughly one and a third passes — which makes model scale,
not more data, the obvious next experiment.

> **On measurement precision.** At validity ≈0.99 the n=1000 standard error is
> ±0.0026, which is larger than the per-link differences. An n=1000 comparison of
> links 3 and 4 showed link 4 *behind*; at n=5000 it was ahead by 0.0096 sanity
> (3.4σ). Quote n≥5000 for anything in this range, and n=10,000 with replicates
> for release figures.

---

## Recommended sampling config

**η = 25, ω = 0.1, polydec, 500 steps.**

```python
from defog.core import DeFoGModel
model = DeFoGModel.load("ckpts/foundation_union_kek_snapshots/link6_final/foundation_model")
samples = model.sample(
    num_samples=1000,
    eta=25.0,           # error-correction stochasticity -- NOT optional here
    omega=0.1,          # target guidance
    sample_steps=500,
    time_distortion="polydec",
)
```

### η matters on this model, contradicting the ChEMBL-era conclusion

The full 5×3 grid (`run_chembl_sweep_jupiter.sh`, 1500 samples/config), sanity:

| η | ω=0 | ω=0.05 | ω=0.1 |
|---|---|---|---|
| **0** | 0.982 | 0.982 | 0.982 |
| **5** | 0.985 | 0.991 | 0.983 |
| **25** | 0.992 | 0.991 | **0.994** |
| **50** | 0.989 | **0.995** | 0.990 |
| **100** | 0.991 | 0.993 | 0.992 |

All three η=0 configs sit at 0.982; every η≥25 config lands in 0.989–0.995, with
connectivity moving 0.991 → 0.997–0.999 alongside. That is a uniform pattern
across nine configs against three, not one lucky cell.

The predecessor's card concluded that η "barely helps (unlike planar graphs)" and
shipped η=0. **This lineage inherited that conclusion without ever testing it.**
The kekulized model's error profile is different — kekulization failures are gone
by construction, so what remains is the kind of local error that CTMC
error-correction can actually repair — and η buys real structural quality on it.
The general lesson is that a sampling optimum belongs to the model that was swept,
not to the architecture or the domain.

**Why η=25/ω=0.1 rather than the sanity-ranked top cell** (η=50/ω=0.05, sanity
0.995): the two differ by 0.001 sanity, roughly 0.4σ at n=1500, i.e. tied, while
η=25/ω=0.1 was better on both kl_score (0.992 vs 0.990) and novelty (0.981 vs
0.977). For a frozen base that adapters, guidance and RL bind to, distribution
fidelity and diversity are worth more than a structural decimal. The n=10,000 run
above confirms the choice delivers (kl_score 0.9941), but note it validates *what
shipping this config gives*, not that it was the best of the six — the top
configs were statistically tied at the precision they were ranked with.

Decode with `pyg_data_to_mol(sample, atom_decoder, bond_decoder)` using the
**kekulized** decoders — `defog.data.chembl_reference.get_representation("kekulized_v2").encoders()`.

---

## Reproduce

```bash
# 1. Build the union (streams the ZINC tarball; ~2h, peak RSS ~25GB)
python scripts/prepare_smiles_union.py \
    --source data/zinc/raw/zinc-druglike-cano.tar.xz \
    --union-smiles data/chembl/chembl_train.smiles \
    --target-clean 100000000 --oversample 1.05 \
    --out-dir data/zinc_chembl_union

# 2. Kekulized prior (prepare_smiles_union writes an AROMATIC stats file which
#    the trainer will refuse; on a 288-core node this takes ~5 min)
sbatch run_union_stats_jupiter.sh

# 3. Train (4-GPU DDP, chained ~9.5h links, auto-resumes from last.ckpt)
sbatch run_union_ddp_chain_jupiter.sh      # repeat per link, or chain with --dependency=afterany

# 4. Evaluate / sweep (single GPU) -- note the reference arguments
sbatch run_chembl_eval_jupiter.sh  <ckpt> kekulized_v2 data/zinc_chembl_union union 5000
sbatch run_chembl_sweep_jupiter.sh <ckpt> kekulized_v2 data/zinc_chembl_union union 1500
```

Passing `data/zinc_chembl_union union` is not optional: those arguments select
both the KL reference and the novelty denominator. Omitted, a union checkpoint is
scored against ChEMBL's property distribution and against a 2.4 M subset of its
own training set — which on the first link 1 eval read `kl_score` 0.607 and
`novelty` 1.000, neither of which described the model.

Key code: `scripts/prepare_smiles_union.py` (dataset), `scripts/compute_graph_stats.py`
(per-representation prior), `scripts/train_chembl_ddp.py` (DDP train + resumable
checkpointing + `--eval-only` / `--sweep`), `defog/data/chembl_reference.py`
(vocabulary + named representations), `defog/data/vocabulary.py` (the guard).

Per-link snapshots: `ckpts/foundation_union_kek_snapshots/link{1..5,6_final}/`.

---

## Release checklist

- [x] η/ω sweep — full 5×3 grid at 1500 samples/config
- [x] Frozen evaluation at n=10,000 (single run, not replicated)
- [x] Sampling config and release figures filled in
- [x] DeFoG-ChEMBL v1/v2 marked superseded in `docs/CHEMBL_FOUNDATION_MODEL.md`
- [ ] **Decide checkpoint distribution** — the file exists only on JUPITER
- [ ] Optional: replicate the n=10,000 eval 2–3× to establish this lineage's own
      noise floors rather than borrowing MOSES's
- [ ] Optional: GDPO structural-sanity RL, re-scoped — on the kekulized MOSES base
      the same reward bought distribution match rather than validity, and at
      0.9967 validity / 0.9897 sanity there is very little left for it to recover.
      The remaining 1% of non-sane samples is now the least valuable place to spend
      compute on this model.

## What is worth doing next

Not more data, and probably not more RL. The model converged after ~1⅓ epochs of
a 100M-molecule set and every metric from link 4 onward moved inside noise, which
points at **capacity** rather than data or sampling as the binding constraint.
Model size was deliberately held at 25.9M so the data scale-up would be the only
variable; that experiment is finished and the answer is that 40× more data lifted
a kekulized ChEMBL model from 0.984/0.940 to 0.997/0.990. The open question this
run did not touch is what a larger model does with the same 100M molecules.
