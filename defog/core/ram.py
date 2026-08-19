"""
Reinforce Adjoint Matching (RAM, arXiv:2605.10759) for DeFoG -- the ablation arm.

RAM keeps GDPO's advantage-weighted cross-entropy and changes exactly one thing:
where the scored states come from. GDPO scores at the states the rollout trajectory
occupied; RAM scores at states redrawn from the pretraining kernel
``p_{t|1}(. | G1)``, conditioned on the rollout's endpoint.

That is the whole arm, and it is deliberately the whole arm. In the three-way
comparison it isolates one variable:

    GDPO -> RAM   does re-noising beat trajectory states?
    RAM  -> DAM   is the derived discrete adjoint worth ~2.3x the forwards?

Without it, a DAM win could be either effect and there would be no way to tell.

RAM is NOT derived for discrete state. Its Theorem 3.1 -- the half that licenses
re-noised states -- is a disintegration argument that ports to a CTMC given
memorylessness. Its Theorem 3.2 and Prop 4.1 -- the half that gives the LOSS -- rely
on a Gaussian noising kernel for the spatial gradient of log p_{0|t}, and do not port.
This arm substitutes GDPO's advantage-weighted CE for that half, which is a
transposition, not a derivation. DAM is the derived alternative; that is why this is
the ablation and not the headline.

Two guards the plan's review made mandatory:

* ``kl_coef > 0``. Unlike DAM, where the anchor is structural (u^base multiplies the
  target and cannot be switched off), RAM's anchor is the optional KL term GDPO
  carries. At kl_coef = 0 there is no target distribution at all -- the fixed point is
  a point mass on argmax r -- so a "RAM" run at kl_coef = 0 is not RAM.
* ``advantage_mode != 'grpo'``. Per-group standard-deviation whitening removes the
  reward scale entirely, and with it the temperature that makes the tilt well defined.
"""

from typing import Optional

import torch

from .renoise import draw_times, renoise_states
from .rl import AdapterGDPOTrainer, RolloutBuffer

__all__ = ["AdapterRAMTrainer"]


class AdapterRAMTrainer(AdapterGDPOTrainer):
    """GDPO's estimator, scored at re-noised states instead of trajectory states.

    Overrides ``__init__`` (guards and the extra knobs) and ``update`` (swap the
    states, then defer to the parent). The rollout, the reward, the advantage
    machinery, the KL term, the checkpoint writer and the composed-policy scoring are
    all inherited unchanged -- which is what makes the GDPO comparison clean.
    """

    record_trace = False

    def __init__(self, base, adapter, cond_reward, *, kl_coef: float = 0.1,
                 renoise_draws: int = 16, t_sampler: str = "match", **gdpo_kw):
        if kl_coef <= 0:
            raise ValueError(
                f"RAM requires kl_coef > 0, got {kl_coef}. The KL-to-pre-RL term is "
                "not a guard here, it is what defines the target distribution: at "
                "kl_coef = 0 the objective is unregularised reward maximisation, whose "
                "fixed point is a point mass on argmax r, and the run is not RAM. "
                "(DAM needs no such flag -- there the anchor is structural.)"
            )
        if gdpo_kw.get("advantage_mode") == "grpo":
            raise ValueError(
                "RAM refuses advantage_mode='grpo': per-group std whitening removes "
                "the reward SCALE, and with it the temperature that makes the tilted "
                "target well defined. Use 'mean' (Dr. GRPO) or 'none'."
            )
        super().__init__(base, adapter, cond_reward, kl_coef=kl_coef, **gdpo_kw)
        # A literal default, NOT a mirror of subsample_steps: mirroring it while the
        # trace is off would give zero re-noised states, i.e. an update that does
        # nothing, in exactly the configuration this arm needs.
        self.renoise_draws = int(renoise_draws)
        self.t_sampler = t_sampler
        # `_choose_subsample()` counts from self.subsample_steps, and in "match" mode
        # that is what decides how many noise levels we get -- so renoise_draws would
        # be silently ignored there, making the RENOISE_DRAWS sweep in the experiment
        # plan a no-op in the DEFAULT configuration. Point them at the same knob.
        # subsample_steps has no other effect in this arm: it only feeds
        # RolloutSampler's subsample_idx, which record_trace=False ignores.
        self.subsample_steps = self.renoise_draws
        # In "match" mode the levels are distinct grid indices, so the effective
        # count is min(renoise_draws, sample_steps). That binds only on toy fixtures;
        # production runs at sample_steps=250.


    def _renoised_states(self, buf):
        idx = None
        if self.t_sampler == "match":
            # None means "every step" for the subsampler, which here is the full grid.
            idx = self._choose_subsample() or list(range(self.sample_steps))
        K = buf.X1.shape[0]
        times = draw_times(self.base, K, self.device, mode=self.t_sampler,
                           n_draws=self.renoise_draws, step_indices=idx,
                           sample_steps=self.sample_steps,
                           time_distortion=self.time_distortion)
        y0 = torch.zeros(K, 0, device=self.device)
        return renoise_states(self.base, buf.X1, buf.E1, y0, buf.node_mask, times)

    def update(self, buf: RolloutBuffer) -> dict:
        """Swap the states, then run GDPO's update verbatim.

        Rebuilding the buffer rather than mutating it keeps the substitution visible:
        every other field is passed straight through, so the diff against the GDPO arm
        is exactly one argument.
        """
        swapped = RolloutBuffer(self._renoised_states(buf), buf.X1, buf.E1, buf.y,
                                buf.node_mask, buf.reward, buf.advantage)
        return super().update(swapped)
