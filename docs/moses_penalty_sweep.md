# MOSES RL distribution-penalty sweep (job 1219218)

Result of adding a descriptor-space MMD penalty to the MOSES sanity-RL reward,
to stop the reward hack documented in `docs/penalty_gate_moses.json`.

Base `ckpts/moses_e1_seed42/best_model`, seed 42, 25 iterations, 4 arms on one
node. All arms scored at the frozen deploy config (500 steps, eta=25) on n=2048,
FCD against the same 5000-molecule validation reference.

    r = sanity(0..3) - BETA * (sim_sibling - 2 * sim_reference)

## The frontier

| beta | validity | d validity | FCD | d FCD | d val / d FCD | MMD^2 | nArom |
|-----:|---------:|-----------:|----:|------:|--------------:|------:|------:|
| base |   0.8999 |          — | 1.465 |     — |             — | 0.00198 | 1.864 |
|    0 |   0.9478 |    +0.0479 | 2.307 | +0.842 |         0.057 | 0.01342 | 1.609 |
|  3.5 |   0.9370 |    +0.0371 | 2.218 | +0.753 |         0.049 | 0.01255 | 1.633 |
|    7 |   0.9409 |    +0.0410 | 2.081 | +0.616 |         0.067 | 0.00968 | 1.656 |
|   14 |   0.9292 |    +0.0293 | 1.800 | +0.334 |         0.088 | 0.00736 | 1.848 |
| real |        — |          — |     — |      — |             — | 0.00011 | 1.998 |

**The penalty works, monotonically.** FCD damage falls 0.842 -> 0.753 -> 0.616
-> 0.334 as beta rises, and MMD^2 falls with it. At beta=14 the FCD cost is cut
by 60% while 61% of the validity gain survives.

**It improves the trade, it does not merely slide along it.** Validity gained
per unit of FCD damage rises from 0.057 at beta=0 to 0.088 at beta=14, a 54%
better exchange rate. A term that only bought fidelity by giving back validity
would hold that ratio constant.

## Mechanism: it fixes what it measures

| set  | nArom | fCSP3 | nN |
|------|------:|------:|-----:|
| real | 1.998 | 0.350 | 3.018 |
| base | 1.864 | 0.350 | 2.881 |
| b0   | 1.609 | 0.413 | 2.631 |
| b14  | 1.848 | 0.348 | 2.649 |

The hack's two headline axes are **fully restored** at beta=14: aromatic rings
1.609 -> 1.848 against a base of 1.864, and sp3 fraction 0.413 -> 0.348 against
0.350. Nitrogen content is **not** restored (2.649 vs base 2.881).

That asymmetry explains the residual FCD gap rather than leaving it mysterious.
The penalty repairs the descriptors it weights and leaves the ones it does not;
MMD^2 at beta=14 is still 3.7x the base value, and FCD is still +0.334. If the
residual matters, the lever is the descriptor set, not a larger beta.

## Control validity

The beta=0 arm reproduced the original unpenalised run **byte-identically** --
`before.smi` and `after.smi` both `cmp`-equal to the 2026-08-02 seed-42 run, and
FCD equal to 12 significant figures. So the control really is the original code
path, and every difference across arms is attributable to beta.

## On the earlier 0.863 -> 1.706 figures

The FCD pair carried through earlier notes (0.863 -> 1.706) came from a
different scoring configuration than the 1.465 -> 2.307 measured here on the
same files. The **delta** is what transfers: +0.843 then against +0.842 now,
near-identical. beta* was derived from the delta, which is why it held up. Quote
deltas, not absolutes, when comparing across evaluation configurations.

## Verdict against the pre-registered criterion

The launcher fixed success as "keeps a real part of the validity gain while
holding FCD near the base". beta=14 keeps 61% of the gain but holds FCD at
+0.334, which is not "near base". **Partial success, honestly scored.**

No arm delivers validity for free. The frontier is real and the penalty shifts
it favourably, but on MOSES some sanity headroom is genuinely paid for in
distribution fidelity.

## Caveats

- One seed per arm. The beta=3.5 / beta=7 inversion in validity (+0.0371 vs
  +0.0410) is within seed noise; the FCD trend is monotone across all four and
  is the more trustworthy signal.
- 25 iterations only. Nothing here says where these curves go with a longer run.
- FCD absolutes depend on n and reference; all rows above share both.
