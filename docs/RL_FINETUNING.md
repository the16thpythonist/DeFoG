# RL fine-tuning of property adapters — changes & best practices

Practical notes from making GDPO adapter-RL actually *work* on the ZINC property
adapters. This is the operational companion to the theory in
[`gdpo_design.md`](gdpo_design.md) and the adapter design in
[`ADAPTER_PLAN.md`](ADAPTER_PLAN.md).

**Code:** `defog/core/rl.py` (trainers, EMA, CRN sampler, scoring),
`defog/core/sampler.py` (`_init_state` CRN hook),
`experiments/adapter_rl_finetune__zinc.py` (runnable experiment),
`tests/test_rl.py`. **Launcher:** `run_adapter_rl_head_jupiter.sh <prop> [adapter_ckpt]`.

## What it does (30 seconds)

The base model is frozen. A small AdaLN CFG-**adapter** is the only thing trained.
Each iteration we roll out `K` full generations under the current adapter, score
the decoded molecules with a reward, turn rewards into per-group advantages, and
take one eager policy-gradient step (REINFORCE) on the adapter. Rate matrices,
composition, and step count are all sampling-time constructs, so the *scored*
policy has to be reconstructed to match the *rollout* policy exactly (see below).

## What changed in this work

- **Removed the moving KL anchor.** It re-anchored the reference to the *live*
  policy every step, so there was no real trust region — it did nothing useful.
  Replaced by a **fixed** KL to a frozen reference, plus a multi-round *ratchet*.
- **`behavior == scored` fix.** The scored log-prob now reproduces the rollout's
  product-of-experts blend at the rollout weight, routed through the model's own
  `_blend_logp` (`_compose_logmarginals` / `_score_logprob` in `rl.py`). If
  scoring composes differently from the rollout, the gradient is for a *different*
  policy and RL silently under-performs or diverges. There are tests pinning
  `behavior == scored` at any weight.
- **Dr. GRPO by default** (`advantage_mode="mean"`): mean baseline, **no**
  per-group std-normalization. Std-norm (`"grpo"`) injects a difficulty bias
  (easy groups get inflated gradients). Literature-backed (Dr. GRPO).
- **Under-fitting fixes** (the "RL ran but the adapter didn't move" class of bug):
  LR `1e-4`, fast EMA `0.9`, paired eval, deploy-step probe, and a weight-diff
  sanity assert. See the table.
- **Head-based reward mode** (`REWARD_SOURCE="head"`): use a learned
  `PropertyHead` as the reward + selection signal for properties with no
  closed-form truth function.

## Best-practice config (and why)

| Knob | Default | Why |
|------|---------|-----|
| `advantage_mode` | `"mean"` (Dr. GRPO) | Mean baseline; avoids the difficulty bias that per-group std-normalization adds. |
| `LR` | `1e-4` | `1e-5` was ~20× below the adapter's own training LR → too weak to move the weights at all. |
| `EMA_DECAY` | `0.9` | `0.999` lagged near the original on short runs; a fast EMA tracks the live weights. |
| `ROLLOUT_SIZE` (K) | `128` | More rollouts per iteration → lower-variance advantage estimate. |
| `N_GROUPS` | `8` | 8 distinct targets × 16 rollouts each per iteration. |
| `CRN` | `True` | Common random numbers: every member of a group starts from the **same** noised graph and size, so within-group advantages compare like-with-like. |
| `ROLLOUT_ETA` | `1.0` | Stochastic rollouts. Under CRN this is the *sole* source of within-group diversity, so it can't be 0. Low-eta won the sweep. |
| `KL_COEF` | `0.1` | Fixed KL to the **pre-RL** adapter (trust region). The ratchet re-anchors this each round. |
| `EVAL_SEED` | `1234` | **Paired eval**: pre/post/probe draw the *same* graphs, so the measured Δ is signal, not sampling noise. |
| `VALIDITY_FLOOR_MARGIN` | `0.05` gate / `1.0` monitor-only | A snapshot must keep validity ≥ `pre-RL − margin`. Head runs set `1.0` to disable the gate (monitor + report only). |

Two invariants worth stating explicitly:

- **`reduction` must match** between the policy-gradient term and the KL term.
  A mean PG term makes `kl_coef` scale with `n²` and become untunable across graph
  sizes (`rl.py` docstring).
- **Weight-diff sanity assert.** After training, assert the deployed adapter's
  weights actually differ from the input. This is the guard that caught an earlier
  *false* "eta=1 wins" result where the deployed adapter was byte-for-byte the
  input (max|Δw| ≈ 1e-5).
- **Deploy-step probe.** Run the early-stop/selection probe at the *same* sampling
  step count you deploy at — a probe at a different step count scores a different
  policy.

## Multi-round ratchet

Fine-tune, pick the best seed by the eval metric, re-anchor the KL reference to
that winner, and go again. It **compounds for ~2 rounds, then plateaus or
overshoots**. Guard with *keep-the-better-round* (by the eval metric): e.g.
SA-score regressed in round-2 on every seed, and the head correctly reported it,
so we kept the round-1 adapter. The reward signal itself flags the overshoot.

## Reward design

- **Connectivity-first.** Gate the property term behind connectivity — a
  disconnected graph scores low regardless of its property value. Without this the
  policy games the property on fragments. (This alone took disconnected-rate from
  ~14% → ~8% on the connectivity adapter.)
- **Head as reward, when there's no truth fn.** A grounded `PropertyHead` can be
  the reward *and* the selection signal. Validated on all 4 ZINC properties
  (head-only reward + selection, RDKit only used to check at the end):

  | property | true (RDKit) MAE | Δ |
  |----------|------------------|---|
  | logP | 0.58 → 0.50 | −14% |
  | TPSA | 9.67 → 7.69 | −20% |
  | QED | 0.165 → 0.130 | −21% |
  | SA-score | 0.545 → 0.444 | −19% |

  Across all 32 arms the head-MAE Δ and RDKit-MAE Δ shared sign and magnitude — no
  head-gaming. Still: validate against a true function where one exists, and keep
  the better-round guard.
- **Head decode→re-encode gotcha.** The head was trained on
  `to_dense(smiles_to_pyg_data(mol))`. Feed it *that*, not the raw argmax graph, or
  it mispredicts systematically.
- **Validity: monitor, don't hard-gate.** Gating too aggressively throws away
  otherwise-good improvements; report it and use it as a keep/discard tiebreak.

## How to run

```bash
# Per-property head-reward run (JUPITER): 4 seeds, best seed → ratchet round.
sbatch run_adapter_rl_head_jupiter.sh logp ckpts/logp_adapter_sc.ckpt

# Or the pycomex experiment directly, RDKit truth reward:
python experiments/adapter_rl_finetune__zinc.py \
  --ADAPTER_CKPT 'ckpts/logp_adapter_sc.ckpt' --REWARD_SOURCE 'rdkit'

# Head reward instead of a closed-form fn:
python experiments/adapter_rl_finetune__zinc.py \
  --ADAPTER_CKPT 'ckpts/qed_adapter_sc.ckpt' \
  --REWARD_SOURCE 'head' --HEAD_CKPT 'ckpts/qed_head.ckpt'
```

## Deliverables

Final head-refined adapters: `ckpts/{logp,tpsa,qed,sascore}_head_rl_final.ckpt`
(logP/TPSA/QED = round-2 winners, SA-score = round-1). These are **deployed** in
the `defog-web` interface — its `manifest.yaml` repoints each ZINC property
adapter's `ckpt_path` at the `_head_rl_final.ckpt` (heads and `prop_mean`
unchanged, since RL leaves the adapter's `cond_mean`/`cond_std` buffers untouched).
