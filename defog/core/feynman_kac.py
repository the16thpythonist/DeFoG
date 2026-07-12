"""
Feynman-Kac / Sequential-Monte-Carlo steering for DeFoG (training-free).

Implements inference-time, reward-tilted sampling in the spirit of
*Discrete Feynman-Kac Correctors* (arXiv:2601.10403) and Feynman-Kac steering:
run a population of ``K`` particles through the (optionally guided) denoising
process, score each particle by a **potential** derived from an external reward,
and periodically **resample** (kill low-weight particles, clone high-weight ones).

Unlike single-trajectory guidance (``GuidedSampler``), the resampling acts on
whole trajectories, so the ensemble self-corrects globally: a particle that
happens to be heading toward the target "takes over", which attacks both failure
modes of plug-in guidance (undershoot / within-mode reward-hacking, and
mode-selection). No training or fine-tuning; the reward is evaluated on each
particle's *predicted clean* graph.

Two proposal modes:
  * base proposal (``proposal_transform=None``): the frozen model's own dynamics;
    the potential+resampling implement (approximate) reward-tilted sampling.
  * guided proposal (pass an ``ExactGuidance.reweight``): the learned guidance
    proposes target-ish molecules and the FK resampling concentrates them further
    (a stronger, heuristic search -- not an exact importance sampler).

Cost: ``K`` forward passes per step (the batch) + one extra forward per particle
at each resample checkpoint (to predict the clean graph for the reward), + reward
evaluation (e.g. RDKit decode) on ``K`` graphs per checkpoint.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import PlaceHolder, dense_to_pyg
from .sampler import Sampler


@torch.no_grad()
def predict_clean(model, X_t, E_t, y_t, t, node_mask):
    """MAP prediction of the clean graph from the current noisy state: run the
    model, softmax, argmax per node/edge. Returns masked one-hot (X1, E1)."""
    noisy = {"X_t": X_t, "E_t": E_t, "y_t": y_t, "t": t, "node_mask": node_mask}
    extra = model._compute_extra_data(noisy)
    pred = model.forward(noisy, extra, node_mask)
    pX = F.softmax(pred.X, dim=-1)
    pE = F.softmax(pred.E, dim=-1)
    X1 = F.one_hot(pX.argmax(-1), pX.size(-1)).float()
    E1 = F.one_hot(pE.argmax(-1), pE.size(-1)).float()
    ph = PlaceHolder(X=X1, E=E1, y=y_t).mask(node_mask)
    return ph.X, ph.E


class FeynmanKacSampler(Sampler):
    """
    Reward-tilted SMC steering over a (frozen) DeFoG model.

    Args:
        model:   the base DeFoGModel.
        energy_fn: callable ``(X1, E1, node_mask) -> (K,)`` energy per graph
            (lower = better), evaluated on the predicted CLEAN graph. For molecular
            property steering use :class:`MoleculePropertyEnergy` (returns
            ``(prop - target)^2``; invalid graphs get a large energy so they are
            culled). The FK potential is ``phi = -beta * energy``.
        beta:    tilt strength (inverse temperature of the reward).
        resample_interval: resample every this many steps (default ~steps/8).
        warmup_frac: fraction of steps to run before the FIRST resample. This
            matters a lot: early in denoising the predicted clean graph is
            unreliable AND size-biased -- large graphs are transiently invalid or
            overshoot the target, so early resampling culls them prematurely and
            collapses the sizes toward small "reliable" molecules. Resample only
            LATE (default 0.7; 0.8+ is tighter), once predictions are valid and
            shaped, so selection reflects final quality, not transient states.
        proposal_transform: optional ``posterior_transform`` (e.g. a trained
            ``ExactGuidance.reweight``) to use guided dynamics as the proposal.
        ess_frac: if set, only resample when the effective sample size drops below
            ``ess_frac * K`` (adaptive resampling; preserves diversity). If None,
            resample at every checkpoint.
    """

    def __init__(self, model, energy_fn: Callable, *, beta: float = 1.0,
                 resample_interval: Optional[int] = None, warmup_frac: float = 0.7,
                 proposal_transform: Optional[Callable] = None,
                 ess_frac: Optional[float] = None, rejuvenate: bool = False,
                 jump_length: Optional[int] = None, **kwargs):
        super().__init__(model, posterior_transform=proposal_transform, **kwargs)
        self.energy_fn = energy_fn
        self.beta = float(beta)
        self.resample_interval = resample_interval
        self.warmup_frac = warmup_frac
        self.ess_frac = ess_frac
        # resample-move: after resampling, jump back `jump_length` steps and
        # re-noise/re-denoise so cloned (duplicated) particles diverge again --
        # the standard SMC remedy for particle impoverishment / diversity collapse.
        # IMPORTANT: the re-denoise re-steers only through the proposal, so
        # rejuvenation must be paired with a GUIDED proposal (proposal_transform);
        # on a bare base proposal it re-noises toward the base and washes out the
        # reward tilt. Keep `jump_length` small (a few % of steps); too large a
        # jump re-collapses under a strong proposal.
        self.rejuvenate = rejuvenate
        self.jump_length = jump_length
        if rejuvenate and proposal_transform is None:
            import warnings
            warnings.warn(
                "FeynmanKacSampler(rejuvenate=True) without a guided proposal_transform "
                "will wash out the reward tilt (the re-denoise re-steers only via the "
                "proposal). Pass proposal_transform=guidance.reweight.",
                RuntimeWarning,
            )

    def _desc(self) -> str:
        tag = "+guided" if self.posterior_transform is not None else ""
        return "FK-SMC" + tag + ("+rejuv" if self.rejuvenate else "")

    def _jump_back_time(self, t_back_int):
        """Distorted normalized time (scalar) for integer step ``t_back_int``."""
        model = self.model
        t_norm = t_back_int / self.sample_steps
        if model.limit_dist.noise_type == "absorbing" and t_back_int == 0:
            t_norm = t_norm + 1e-6
        t_norm = model.time_distorter.sample_ft(
            torch.tensor([[t_norm]], dtype=torch.float32), self.time_distortion
        )
        return float(t_norm.reshape(-1)[0].item())

    @torch.no_grad()
    def _renoise_toward_current(self, X, E, tau, node_mask):
        """Re-noise the whole graph toward its own current one-hot at level tau:
        sample from tau*onehot(current) + (1-tau)*limit, INDEPENDENTLY per particle,
        so cloned particles split. Edges sampled upper-triangular then mirrored."""
        bs, n, dx = X.shape
        de = E.shape[-1]
        device = X.device
        limX = self.model.limit_dist.X.to(device).float()
        limE = self.model.limit_dist.E.to(device).float()

        probX = tau * X + (1.0 - tau) * limX.view(1, 1, dx)
        idxX = probX.clamp_min(0).reshape(bs * n, dx).multinomial(1).reshape(bs, n)
        Xn = F.one_hot(idxX, dx).float()

        probE = tau * E + (1.0 - tau) * limE.view(1, 1, 1, de)
        idxE = probE.clamp_min(0).reshape(bs * n * n, de).multinomial(1).reshape(bs, n, n)
        iu = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
        idxE_sym = torch.zeros(bs, n, n, dtype=idxE.dtype, device=device)
        idxE_sym[:, iu] = idxE[:, iu]
        idxE_sym = idxE_sym + idxE_sym.transpose(1, 2)
        En = F.one_hot(idxE_sym.long(), de).float()

        masked = PlaceHolder(X=Xn, E=En, y=None).mask(node_mask)
        return masked.X, masked.E

    @torch.no_grad()
    def _potential(self, X, E, y, t_norm, node_mask):
        X1, E1 = predict_clean(self.model, X, E, y, t_norm, node_mask)
        energy = self.energy_fn(X1, E1, node_mask).to(X.device).reshape(-1)
        return -self.beta * energy

    @torch.no_grad()
    def sample(self, num_samples, num_nodes=None, size_dist=None, condition=None,
               device=None, show_progress=True):
        model = self.model
        was_training = model.training
        model.eval()
        device = device if device is not None else model.device

        node_mask, n_nodes, X, E, y, y_raw, use_cfg = model._prepare_generation(
            num_samples, num_nodes, size_dist, condition, self.guidance_scale, device
        )
        K = num_samples
        logw = torch.zeros(K, device=device)
        prev_phi = None
        ri = self.resample_interval or max(1, self.sample_steps // 8)
        warmup = int(self.warmup_frac * self.sample_steps)

        iterator = range(self.sample_steps)
        if show_progress:
            iterator = tqdm(iterator, desc=self._desc())

        for t_int in iterator:
            X, E, y = self._advance(t_int, X, E, y, node_mask, use_cfg)

            is_checkpoint = (
                t_int >= warmup
                and (t_int - warmup) % ri == 0
                and t_int < self.sample_steps - 1
            )
            if not is_checkpoint:
                continue

            t_norm = ((t_int + 1) / self.sample_steps) * torch.ones(K, 1, device=device)
            phi = self._potential(X, E, y, t_norm, node_mask)
            # incremental weight (telescoping): reward *gained* since last checkpoint
            logw = logw + (phi if prev_phi is None else phi - prev_phi)
            prev_phi = phi

            w = torch.softmax(logw, dim=0)
            ess = 1.0 / w.pow(2).sum().clamp_min(1e-12)
            if (self.ess_frac is None) or (ess < self.ess_frac * K):
                idx = torch.multinomial(w, K, replacement=True)
                X, E, node_mask, y = X[idx], E[idx], node_mask[idx], y[idx]
                prev_phi = prev_phi[idx]
                if y_raw is not None:
                    y_raw = y_raw[idx]
                logw = torch.zeros(K, device=device)

                # resample-move: split cloned particles by jumping back and
                # re-noising/re-denoising, so the ensemble regains diversity.
                if self.rejuvenate:
                    j = self.jump_length or max(1, self.sample_steps // 25)
                    t_back = max(0, t_int + 1 - j)
                    tau = self._jump_back_time(t_back)
                    X, E = self._renoise_toward_current(X, E, tau, node_mask)
                    for tt in range(t_back, t_int + 1):
                        X, E, y = self._advance(tt, X, E, y, node_mask, use_cfg)
                    # particles changed -> refresh the telescoping reference
                    t_now = ((t_int + 1) / self.sample_steps) * torch.ones(K, 1, device=device)
                    prev_phi = self._potential(X, E, y, t_now, node_mask)

        X, E, _ = model.limit_dist.ignore_virtual_classes(X, E)
        n_final = node_mask.sum(-1)
        samples = dense_to_pyg(X, E, y_raw, node_mask, n_final)

        if was_training:
            model.train()
        return samples


class JointGuidanceSampler(FeynmanKacSampler):
    """FK-SMC steering toward MULTIPLE properties at once.

    Constructed from a list of trained :class:`~defog.core.guidance.ExactGuidance`
    modules plus a joint ``energy_fn`` (e.g.
    :class:`~defog.core.guidance.MultiPropertyEnergy`), and the usual FK
    hyperparameters (``beta``, ``warmup_frac``, ``resample_interval``, ``ess_frac``,
    ``eta``, ``omega``, ``sample_steps``, ...).

    The guidance modules are combined in the PROPOSAL via
    :class:`~defog.core.guidance.CompositeGuidance` (``mode`` = "product" | "mean" |
    "none"); the joint ``energy_fn`` is the FK reward and is fully decoupled from the
    guidance list. Set each guidance's target (``set_target``) before sampling, or
    use ``self.composite.set_targets([...])``.

    Args:
        guidances: list of ExactGuidance (each amortized on its own property).
        energy_fn: joint reward (lower = better), evaluated on the predicted clean graph.
        mode: proposal composition, "product" (default) | "mean" | "none".
        guidance_weights: per-guidance strengths (default = each guidance's ``.weight``).
    """

    def __init__(self, model, guidances, energy_fn, *, mode: str = "product",
                 guidance_weights=None, **fk_kwargs):
        from .guidance import CompositeGuidance
        self.composite = CompositeGuidance(guidances, mode=mode, weights=guidance_weights)
        proposal = self.composite.reweight if mode != "none" else None
        super().__init__(model, energy_fn, proposal_transform=proposal, **fk_kwargs)
        self.guidances = list(guidances)
        self.mode = mode

    def _desc(self) -> str:
        return f"JointFK[{len(self.guidances)}x,{self.mode}]"
