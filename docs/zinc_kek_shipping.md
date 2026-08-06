# Shipping the kekulized ZINC model + clogP adapter to molsmith

Two packages installed in `~/.molsmith`, alongside (not replacing) the existing
aromatic base, so its eight adapters keep working.

## What shipped

| package | kind | base | content hash |
|---|---|---|---|
| `molsmith/zinc-kek@1.0.0` | base | — | `4633ac4fa321` |
| `molsmith/clogp@1.0.0` | adapter | `molsmith/zinc-kek` | `11184acb369d` |
| `molsmith/clogp@1.1.0` | adapter | `molsmith/zinc-kek` | `5b5fb1539dd4` |

Source checkpoint: `ckpts/zinc_rl2_seed42/best_model` -- the 2-round sanity-RL
model, chosen over the E1 base for validity 0.9959 vs 0.9929, disconnected
0.0180 vs 0.0267 and FCD 1.346 vs 1.401, at a scaffold-diversity cost of
0.5464 vs 0.6035.

    atoms   C N O F P S Cl Br I    (9 classes, order significant)
    bonds   none single double triple      (4 -- no AROMATIC)
    size    6 .. 38 heavy atoms
    schema  4e39344a5067

**Why a new id rather than replacing `molsmith/zinc-base`.** The old base is a
different checkpoint (`zinc_uncond_4e-4_connectivity`) with a different bond set
AND a different atom order (`C N O S F Cl Br I P`). Its schema hash therefore
differs, and all eight installed adapters -- logp, tpsa, qed, sascore,
fingerprint, logd x3 -- are bound to it. Replacing in place would have taken
them all offline until retrained.

## Sampling defaults came from a sweep, not inheritance

`zinc_rl2_seed42` had only ever been evaluated at the E1 *base's* frozen config.
That is right for isolating an RL comparison and wrong for shipping: eta is
error-correction stochasticity whose optimum depends on where the policy's
errors are, and RL moved the policy. Job 1237968 swept 48 points
(steps x eta x omega) on validation, FCD-scored afterwards:

| steps | eta | omega | FCD | validity | ms/molecule |
|---:|---:|---:|---:|---:|---:|
| 500 | 5 | 0 | 2.582 | 0.9940 | 154 |
| 500 | 25 | 0 | 2.797 | 0.9980 | 156 |
| **250** | **25** | **0** | **2.739** | **0.9970** | **81** |
| 100 | 25 | 0 | 2.995 | 0.9960 | 32 |
| 50 | 25 | 0 | 3.479 | 0.9820 | 17 |

**Shipped: steps=250, eta=25, omega=0, polydec.** The 500-step configs cost 2x
the latency for FCD differences of 0.06-0.2, and the spread *among*
statistically equivalent configs is already ~0.1 -- so that is noise. eta=25
beats 0 and 5 at every step count; omega is inert. The step ladder was widened
past the E1 sweep's two points specifically because this model generates
interactively, where 81 ms/molecule against 156 is a real difference and a
table-oriented sweep would never have surfaced it.

FCD here is ~2.7 rather than the 1.346 quoted for the same model on test: this
sweep is n=1000, the test pass was n=10,000, and FCD is strongly n-biased. The
rows are comparable to each other, not to the test number.

## The clogP adapter, and a defect in how it was labelled

Four arms (LR x width), 20 epochs. Steering at guidance weight 1.0, n~128:

| arm | low (target -0.1) | high (target 4.5) | MAE low+high |
|---|---:|---:|---:|
| **lr2h256 (shipped)** | 1.48 | **4.15** | 2.332 |
| lr4h256 | 1.43 | 4.18 | 2.301 |
| lr2h512 | 1.54 | 4.15 | 2.383 |
| lr4h512 | 1.27 | 3.99 | 2.269 |

All four are within noise of each other; lr2h256 has the best high-end MAE and
is the smaller adapter. End-to-end through `molsmith sample`, 40 molecules per
target, 40/40 valid at each:

    target 1.0 -> 2.09      target 2.5 -> 2.75      target 4.5 -> 4.38

**The asymmetry is a labelling defect, not a model limitation.** Every arm
steers up well (MAE 0.65-0.76) and down poorly (1.51-1.73). The cause is
measurable: 33% of ZINC molecules carry a formal charge, which the graph
representation does not encode, and the property label was computed from the
SOURCE molecule.

| | source clogP | decoded clogP (what the graph IS) | shift | charged |
|---|---:|---:|---:|---:|
| low tail (5th pct) | -0.903 | +0.733 | **+1.636** | 92.5% |
| high tail (95th pct) | +4.901 | +4.896 | -0.005 | 7.0% |

Low-end labels are wrong by 1.64 log units; high-end labels are exact. So the
adapter is performing *better* than its MAE suggests -- asked for -0.1 it
produces ~1.3-1.5, while the graphs actually labelled -0.1 decode to ~+0.73.
It is close to what it was taught.

This was a deliberate deferral: the label source was left matching the validated
legacy adapters so that a vocabulary change and a label change would not be
confounded in one run. The vocabulary change is now validated, and the label
defect is diagnosed, so the two are separable.

**The fix** is to compute the property from the decoded molecule. Expected to
recover most of the 1.64-unit low-end deficit, at the cost of one 3-hour 4-GPU
run. Not applied yet.

## Verification performed

- Base loads and generates through molsmith: 20/20 valid, 20 unique.
- Adapter reports `compatible`, schema `4e39344a5067 matches`.
- All eight old adapters still report compatible against `molsmith/zinc-base`.
- ZINC encodes under the new vocabulary with zero failures; round-trip fidelity
  modulo stereo/charge is 0.882 against the legacy vocabulary's 0.883.


## clogP v1.1.0 — the label defect, fixed

v1.0 computed its conditioning label from the SOURCE SMILES. A DeFoG graph
stores atoms and bonds but not formal charges, and 33% of ZINC carries one, so
the low-end labels described molecules the graphs were not. v1.1.0 labels the
DECODED molecule instead (`PROPERTY_FROM="decoded"`), changing that one variable
and nothing else -- same base, vocabulary, LR x width grid and epochs.

Steering at w=1.0, n~128, four arms:

| arm | low MAE | high MAE |
|---|---:|---:|
| lr2h256 | 0.635 | 0.679 |
| **lr4h256 (shipped)** | **0.631** | **0.663** |
| lr2h512 | 0.710 | 0.788 |
| lr4h512 | 0.649 | 0.683 |
| *v1.0, all four* | *1.51 - 1.73* | *0.648 - 0.759* |

**Low-end MAE improved 2.4-2.7x; the high end did not move.** That is exactly
the pre-registered signature, and it matters that the high end held: the
high-end labels were already correct (95th percentile is only 7% charged), so a
large change there would have meant something other than the relabelling moved
and the run was suspect rather than successful. An intermediate epoch-4 reading
showed the high end apparently 25% better and was flagged as a possible
confound; at the proper n=128 evaluation it was unchanged, so that was
small-sample noise during early training.

End-to-end through `molsmith sample`, 40 molecules per target, all valid:

| target | v1.1.0 | v1.0.0 |
|---:|---:|---:|
| 1.0 | **1.04** | 2.09 |
| 2.5 | **2.40** | 2.75 |
| 4.5 | 4.09 | 4.38 |

The declared range moves from [-1.84, 6.75] to [-0.65, 6.30]. That is the fix,
not a regression: the model cannot generate clogP -0.1, because that region
needs charges the representation cannot express, and v1.0 advertised reach it
did not have. v1.0.0 stays installed for reproducibility.
