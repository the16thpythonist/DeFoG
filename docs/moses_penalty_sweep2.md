# MOSES RL penalty sweep, round 2 (job 1223435) + base-model validity diagnosis

Two questions: can MOSES validity go higher, and should the descriptor set be
widened. Round 2 answers both, and the diagnostic answers a third that matters
more than either.

## The base model's validity deficit is a kekulization problem

`scripts/diagnose_validity.py` on the MOSES E1 base, n=1024 at the frozen deploy
config, decoding to an unsanitized molecule and classifying the sanitization
failure:

| category | count | share |
|---|---:|---:|
| ok | 896 | 0.8750 |
| **kekulize** | **118** | **0.1152** |
| disconnected | 8 | 0.0078 |
| other_sanitize | 1 | 0.0010 |
| valence | 1 | 0.0010 |

**118 of 120 hard failures are kekulization. One is a valence error.** The model
is not producing impossible valences; it is producing aromatic ring systems
RDKit cannot kekulize.

ZINC trains on a kekulized representation (no AROMATIC bond class) and reaches
~0.99 validity. MOSES trains with an AROMATIC class and loses 11.5 points to
exactly the failure that class makes possible. Removing the class removes the
failure *by construction*, at no FCD cost -- which is a bigger prize than
anything RL has produced here (+3 to +6 points, always paid for in FCD).

It also explains the original hack in one line: the policy's cheapest route to
validity was "emit fewer aromatic rings", because aromatic rings are where
essentially all the failures live.

**Caveat, unresolved:** GuacaMol is also aromatic and reaches ~0.98, so
aromaticity alone does not explain why MOSES specifically is worse. A second
factor exists and has not been identified. This does not weaken the fix -- the
failure mode becomes impossible either way -- but it does mean the recovery
cannot be promised precisely. Running the same diagnostic on the GuacaMol and
ZINC checkpoints would settle the mechanism cheaply.

## The full frontier

All rows: seed 42, n=2048, deploy config (500 steps, eta=25), FCD against the
same 5000-molecule validation reference.

| arm | beta | iters | validity | d val | FCD | d FCD | d val/d FCD | uniq | novelty | MMD^2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | — | — | 0.8999 | — | 1.465 | — | — | 1.000 | 0.9474 | 0.00198 |
| b0 | 0 | 25 | 0.9478 | +0.0479 | 2.307 | +0.842 | 0.057 | 1.000 | 0.9464 | 0.01342 |
| b7 | 7 | 25 | 0.9409 | +0.0410 | 2.081 | +0.616 | 0.067 | 1.000 | 0.9564 | 0.00968 |
| b14 | 14 | 25 | 0.9292 | +0.0293 | 1.800 | +0.335 | 0.087 | 1.000 | 0.9369 | 0.00736 |
| b28 | 28 | 25 | 0.9185 | +0.0186 | 1.810 | +0.345 | 0.054 | 1.000 | 0.9288 | 0.00433 |
| b56 | 56 | 25 | 0.8989 | -0.0010 | 1.546 | +0.081 | — | 1.000 | 0.9408 | 0.00218 |
| **b14i50** | **14** | **50** | **0.9604** | **+0.0605** | **2.056** | **+0.591** | **0.102** | 0.9995 | 0.9079 | 0.00797 |
| b28i50 | 28 | 50 | 0.9526 | +0.0527 | 2.615 | +1.150 | 0.046 | 1.000 | 0.8903 | 0.01030 |

### More iterations was the right lever

`b14i50` beats the **unpenalised control on both axes at once**: validity
+0.0605 against +0.0479, and FCD +0.591 against +0.842. Best exchange rate in
the study, 0.102 against the control's 0.057. The prediction that drove this
arm -- beta=14 selected the final iteration of 25 and had not converged -- held.

The mechanism is that the penalty blocks the cheap route, so the policy needs
longer to find sanity gains that do not come from dropping aromatic rings.
`b14i50` ends at nArom 1.958, *above* the base's 1.864 and closest of any arm to
real data's 1.998.

### But it is not free: novelty pays

Novelty falls 0.9474 (base) -> 0.9079 (b14i50) -> 0.8903 (b28i50). Uniqueness
stays at ~1.0 throughout, so this is not duplicate collapse -- the longer runs
drift toward the *training* distribution. A mid-run worry that rising
`sim_sibling` meant mode collapse was **not** borne out by uniqueness; novelty
is where the cost actually landed.

Any claim about `b14i50` has to carry that: it wins validity and FCD, and loses
novelty. Under V.U.N. accounting that trade is not automatically favourable.

### beta has a right-hand edge

`b56` selected iteration 4 and gained nothing (-0.001 validity). Past roughly
beta=28 the MMD term dominates the reward and there is no longer enough
incentive to improve sanity at all. The frontier is bracketed on both sides.

### More beta is NOT simply better

Round 1's monotone FCD-vs-beta trend does not survive the extension.
`b28i50` (beta=28) has the **worst FCD of any arm**, +1.150, worse than the
unpenalised control. beta and iterations interact: more iterations at moderate
beta is a Pareto win, more iterations at high beta is worse than not penalising
at all. Do not extrapolate along one axis.

## Does the proxy track the target?

Across all 8 policies:

    corr(MMD^2, FCD)      pearson +0.879   spearman +0.952
    corr(validity, FCD)   pearson +0.866
    corr(novelty, FCD)    pearson -0.554

Spearman 0.95 is the number that justifies the design: the descriptor MMD ranks
policies by FCD almost perfectly, without ever computing FCD, at n=128 batches.
Pearson is lower because the relationship is not linear -- `b28` has much better
MMD^2 than `b14` (0.00433 vs 0.00736) at essentially the same FCD (1.810 vs
1.800), so the proxy is a good ranker and a poor calibrator. Use it to choose,
not to quote.

## Recommendation

- **b14i50** if validity is the priority and a ~4-point novelty cost is
  acceptable. It dominates the unpenalised control on both validity and FCD.
- **b14 at 25 iterations** if fidelity is the priority: +0.029 validity for only
  +0.335 FCD with novelty essentially intact (0.9369 vs 0.9474).
- **Not b28i50 and not b56.** The first is worse than no penalty at all on FCD;
  the second does not train.

**But the RL frontier is second-order.** The diagnostic says 11.5 points of
validity are sitting in a representation choice. Retraining MOSES kekulized
plausibly recovers most of it at zero FCD and zero novelty cost, against RL's
best-case +6 validity for +0.59 FCD and -0.04 novelty. Confirm the mechanism on
the GuacaMol and ZINC checkpoints first -- that is a few minutes of GPU.

## Caveats

- One seed per arm. b28's inversion against b14 in FCD (+0.345 vs +0.335) is
  within noise; the large effects (b14i50, b28i50, b56) are not.
- FCD absolutes depend on n and reference; every row here shares both.
- The `b56` arm effectively did not train, so its good FCD is trivially the base
  model's, not evidence that high beta preserves fidelity.
