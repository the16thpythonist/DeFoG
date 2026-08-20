# MOSES E1 row — kekulized lineage, frozen test pass

**Status: complete.** Sweep on validation → frozen config → one evaluation pass on
test, per `docs/unconditional-protocol.md` §5. This is the first MOSES result from
this lineage that is a quotable row rather than a lower bound.

## The rows

Two models, each measured once under the same frozen configuration:

- **base** — `ckpts/moses_kek_seed44/best_model`
- **+ sanity RL** — `ckpts/moses_kekrl_b0s43/best_model` (round-1 GDPO, β=0, RL-seed 43,
  from that same base; the `s43` in the name is the RL seed, **not** the base seed)

The two checkpoints have byte-identical architecture and training hyperparameters —
verified by diffing their stored `hyper_parameters` — so they differ in weights alone.

| metric | base | + sanity RL | direction |
|---|---:|---:|---|
| **Validity** | 0.9975 | **0.9985** | ↑ |
| Uniqueness @30k | 0.9994 | 0.9994 | ↑ |
| Novelty (vs train) | **0.9478** | 0.9455 | ↑ |
| V×U×N | **0.9449** | 0.9434 | ↑ |
| Connected | 0.9952 | **0.9961** | ↑ |
| Filters | **0.9827** | 0.9807 | ↑ |
| **FCD/TestSF** | 0.8744 | **0.8158** | ↓ |
| **SNN/TestSF** | 0.5230 | **0.5248** | ↑ |
| **Frag/TestSF** | 0.9964 | **0.9967** | ↑ |
| **Scaf/TestSF** | 0.1358 | **0.1402** | ↑ |

Test-split (non-SF) variants, recorded but **not** the ones to quote — see below:
base FCD/Test 0.3287, SNN 0.5487, Frag 0.9986, Scaf 0.9083; RL FCD/Test 0.2668,
SNN 0.5508, Frag 0.9988, Scaf 0.9095.

**What the RL round bought.** FCD 0.8744 → 0.8158 and Scaf 0.1358 → 0.1402, against a
small cost in novelty (0.9478 → 0.9455) and filters (0.9827 → 0.9807). Validity moved
+0.10 points, which is inside the seed spread. That is **distribution match, not
validity** — the same conclusion the independent n=10,000 comparison reached
(FCD 0.5928 vs 0.6313), now reproduced at n=30,000 against the full test split.
Most of the margin over published baselines is in the **base**, not the RL stage.

## Frozen configuration — reproduce with this

```
steps            500
eta              25
omega            0
time_distortion  polydec
n                30,000
seed             42
representation   kekulized_v2  (7 atom / 3 bond; remove_h=True, aromatic=False)
validity          relaxed_largest_frag
metrics impl     molsets 0.3.1 (official)
test reference   176,074   test_scaffolds reference  176,225
```

Provenance:

| | sampling | md5 of `seed42.smi` | scoring |
|---|---|---|---|
| base | KCIST job 43017, 5875 s | in `final_mosesbase_43017/` | KCIST job 43021 |
| + sanity RL | JUPITER job 1318337, 2555 s | `12808c49b9d195cb00fc1c084dffd1cd` | KCIST job 42993 |

Config chosen on validation in jobs 1313054 (stage 1) and 1317727 (stage 2), both on
the **RL** model, and applied unchanged to the base. That transfer is deliberate:
re-sweeping per model would be more tuning, not less, and the sweep separated nothing
beyond its noise floor anyway. Disclosed in the manuscript caption.

The base ran on KCIST because JUPITER went into a cluster-wide maintenance (5,713
nodes drained, zero jobs running) partway through. KCIST needed the whole
`defog/data` module uploaded on top of an older checkout, so the job gates on a
32-sample generate-and-decode smoke test before the real run — version skew between
an old core and a new data module would not crash, it would produce plausible
molecules made of the wrong elements.

## Three things that are easy to get wrong here

**1. `moses_validity` in the JSON is 1.0000 and must NOT be reported.** The SMILES
file contains only molecules that already decoded, so the suite measures validity on
a pre-filtered list and returns 1.0 by construction. The model's validity is
**0.9985** — 29,954 valid of 30,000 generated — from the sampling record in
`seed42.json`. Quoting 1.0000 against baselines' ~0.93 would be a fabricated perfect
score, and the field is sitting in the output under an inviting name.

**2. Quote the TestSF columns.** MOSES computes every metric against both `test` and
`test_scaffolds`, and they are far apart. Calibrated against DeFoG's own published
samples, Scaf/Test reads 0.868 where the paper reports 0.144, while Scaf/TestSF reads
0.107. Reporting Scaf/Test would look like a 6× improvement that does not exist.

**3. All three validity conventions agree here** (relaxed 0.99847, strict 0.99803,
whole-molecule 0.99847), so this particular number does not depend on the convention.
State it anyway — the protocol warns that quoting a valency-corrected number against
someone else's uncorrected one inflates the result, and that only stays checkable if
the convention is written down.

**4. Read the JSON, not the job's printed summary.** The base-model scoring job
(43021) printed the **RL model's** numbers. Its summary block was copied from the RL
scoring script and contained a hard-coded path; the heredoc is single-quoted, so the
`$D` substitution that fixed every other reference never reached inside it. The job
scored the correct samples and wrote the correct JSON — only the display was wrong,
which is the dangerous kind: nothing downstream would have flagged the RL numbers
sitting in the base row. What caught it was three metrics agreeing to four decimals
between two different models. Both scoring scripts now take the directory as an
argument. **Check `n_generated` in the JSON against the sampling record** (29,924 base
vs 29,954 RL) as the cheap confirmation that a metrics file describes the model you
think it does.

## Against DeFoG's published MOSES row

| metric | published | ours | |
|---|---:|---:|---|
| Validity | 0.928 | **0.9985** | +7.1 points |
| FCD ↓ | 1.95 | **0.8158** | 2.4× better |
| SNN | 0.55 | 0.5248 | slightly lower |
| Scaf | 0.144 | 0.1402 | comparable |

**Where the gain comes from, and where it does not.** The sampling sweep found
*nothing*: all 32 configurations landed within noise of each other, and the frozen
config is the one the model was already being sampled at. The gain is the
**representation fix** — kekulizing was worth 11.4 validity points on its own, because
118 of 120 hard failures under the aromatic encoding were kekulization errors.

The sweep's contribution is not a number but a status change: these figures stop being
"produced at settings inherited from a different model and never checked" and become
"produced at settings confirmed within noise of the best of 32 alternatives tuned on
validation".

**Caveat for the manuscript.** Our model is kekulized; DeFoG's published MOSES model is
aromatic. Validity is therefore being compared across representations, which changes
what "valid" can mean — an aromatic model can fail kekulization in ways a kekulized one
structurally cannot. This belongs in the caption, not a footnote.

SNN being marginally below the published value (0.525 vs 0.55) while FCD is far better
is a real trade, not noise to be smoothed over: the model matches the distribution more
closely while individual molecules are slightly less similar to their nearest test
neighbour.

## Why the sweep is worth reporting despite finding nothing

Stage 1 (32 points, n=2000) settled that 500 steps beats 50 decisively (mean FCD 1.27
vs 2.66) and that ω=0.25 is bad everywhere, but its top five spanned 0.080 FCD against
a ±0.046 noise floor — unusable for picking a winner.

Stage 2 (n=10,000) re-ran the leaders plus the inherited config **plus a seed
duplicate**. The duplicate decided it: one configuration run at seeds 42 and 777 scored
0.5611 and 0.5919 — a gap of **0.0308** — and those two identical draws ranked 3rd and
**last**. The entire spread among the top five was 0.0268, smaller than that gap.

Two consequences worth carrying forward:

- **The published ±0.0084 FCD floor does not transfer.** Against this 5,000-molecule
  validation reference it is 0.0308, 3.7× larger. Noise floors are properties of a
  measurement setup, not of a metric.
- **Always include a seed duplicate in a sweep.** Without it, this sweep would have
  crowned `η=25/ω=0.05` on a difference smaller than the spread between two runs of
  that same configuration.
