"""
The DAM estimator gate (docs/dam_design.md sections 4 and 10 step 5).

This is the go/no-go for the estimator's ALGEBRA. It is deliberately model-free: a
tabular one-jump CTMC where the optimum is known in closed form, so a failure here is
a mistranscribed adjoint and nothing else. It cannot see the projection gap (the
policy is unconstrained), the head surrogate (there is no head), or lambda/K
mis-tempering on real rewards -- those need the real base and are section 8.5's job.

The one-jump chain. From `start` you jump once to a terminal state j:

    V(j)        = g(j)                                  (terminal)
    V(start)    = -log sum_j p_base(j) e^{-g(j)}
    u*(j|start) = u_base(j) e^{-g(j) + V(start)}

so with u_base = p_base the optimal rate is proportional to p_base * e^{-g}: exactly
the tilted distribution. From j the clean-graph law is delta_j, so the adjoint's first
factor has log-ratio 0 and terminal loss g(j) -- the second factor is where the
importance weighting does its work, and where the reading of Eq. (13) matters.

Because the second factor depends on p_theta, the target MOVES: this exercises the
fixed point, not one-shot arithmetic.
"""

import math

import pytest
import torch

from defog.core.dam import discrete_adjoint, estimate_neg_value, gkl


S = 40
K = 512
SEED = 17


def _problem(lam, seed=SEED, span=10.0):
    """A base law and a reward tiered like this repo's PropertyMatchReward: a 10%
    invalid floor at -span, a 15% disconnected tier, the rest graded."""
    g = torch.Generator().manual_seed(seed)
    p_base = torch.rand(S, generator=g) + 0.05
    p_base /= p_base.sum()
    r = -torch.rand(S, generator=g) * 3.0                    # graded tier, [-3, 0]
    r[: int(0.10 * S)] = -span                               # invalid floor
    r[int(0.10 * S): int(0.25 * S)] = -0.4 * span            # disconnected tier
    return p_base, -lam * r                                  # (p_base, g)


def _target(p_base, gvec):
    t = p_base * torch.exp(-gvec)
    return t / t.sum()


def _kl(p, q):
    return float((p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum())


def _fit(p_base, gvec, *, steps=2000, lr=0.05, k=K, seed=SEED, adjoint=discrete_adjoint):
    """Run the DAM update to convergence on the tabular problem."""
    torch.manual_seed(seed)
    logu = torch.zeros(S, requires_grad=True)
    opt = torch.optim.Adam([logu], lr=lr)
    log_ratio_Z = torch.zeros(S)                             # p_{1|t}(.|j) = delta_j
    for _ in range(steps):
        u = logu.exp()
        p_theta = (u / u.sum()).detach()
        idx = torch.multinomial(p_theta, k, replacement=True)
        log_ratio_X1 = (p_base[idx].log() - p_theta[idx].log()).unsqueeze(0)
        with torch.no_grad():
            log_a, _ = adjoint(log_ratio_Z, gvec, log_ratio_X1, gvec[idx].unsqueeze(0))
            target = p_base * log_a.exp()
        opt.zero_grad()
        gkl(u, target).sum().backward()
        opt.step()
    u = logu.detach().exp()
    return u / u.sum(), u


# ================================================================ the second factor
def test_estimate_neg_value_matches_enumeration():
    """The second factor must estimate -V_t(x) = log E_{p_base}[e^{-g}], and it must
    do so when p_theta != p_base -- which is the only regime where the reading of
    Eq. (13) is observable at all."""
    p_base, gvec = _problem(lam=1.0)
    torch.manual_seed(SEED)
    logits = torch.randn(S)                                  # a genuinely different policy
    p_theta = torch.softmax(logits, 0)

    exact = float((p_base * torch.exp(-gvec)).sum().log())
    idx = torch.multinomial(p_theta, 200_000, replacement=True)
    est = float(estimate_neg_value(p_base[idx].log() - p_theta[idx].log(), gvec[idx]))
    assert abs(est - exact) < 0.05, f"estimate {est:.4f} != exact {exact:.4f}"


def test_bare_density_ratio_reading_is_wrong_on_and_off_policy():
    """Pins the amendment. The importance weight is the density ratio TIMES e^{-g};
    using the bare ratio estimates E[p_base/p_theta] = 1 instead of
    E_{p_base}[e^{-g}], so it is wrong by roughly 1/E[e^{-g}] whatever the policy.

    The magnitude is fixture-dependent -- the plan's review measured ~360x with a more
    divergent policy and a wider reward span than this fixture uses -- so the assertion
    is on the qualitative fact, which must not regress.
    """
    p_base, gvec = _problem(lam=1.0)
    exact = float((p_base * torch.exp(-gvec)).sum())

    def readings(p_theta):
        torch.manual_seed(SEED)
        idx = torch.multinomial(p_theta, 200_000, replacement=True)
        ratio = p_base[idx].log() - p_theta[idx].log()
        correct = math.exp(float(estimate_neg_value(ratio, gvec[idx])))
        bare = math.exp(float(torch.logsumexp(ratio, 0) - math.log(idx.numel())))
        return correct, bare

    torch.manual_seed(SEED)
    for label, p_theta in (("on-policy", p_base.clone()),
                           ("off-policy", torch.softmax(torch.randn(S), 0))):
        correct, bare = readings(p_theta)
        assert abs(correct / exact - 1) < 0.10, \
            f"{label}: correct reading drifted to {correct / exact:.3f}x"
        assert bare / correct > 2.0, \
            f"{label}: bare-ratio reading only {bare / correct:.2f}x off -- guard has no teeth"
        assert abs(bare - 1.0) < 0.10, \
            f"{label}: the bare reading should collapse to ~1 (it estimates E[p_base/p_theta])"


# ================================================================ THE GATE
# Tolerances are set from measurement, not taste, and they differ by lambda BECAUSE
# lambda is the inverse temperature: measured KL at 2000 steps is 0.0000 / 0.0001 /
# 0.0156 / 0.1182 / 0.3225 at lambda = 0.1 / 0.3 / 1 / 3 / 6. The recommended default
# (DAM_LAMBDA = 0.3) is the regime where the estimator is essentially exact.
@pytest.mark.parametrize("lam,tol", [(0.3, 0.005), (1.0, 0.05)])
def test_dam_reaches_kl_optimum(lam, tol):
    """Run the shipped estimator to convergence and require it to land on
    p_base * e^{-g} / Z."""
    p_base, gvec = _problem(lam)
    got, u = _fit(p_base, gvec)
    tgt = _target(p_base, gvec)
    kl = _kl(got, tgt)
    base_kl = _kl(p_base, tgt)
    assert kl < tol, f"lambda={lam}: KL to the tilted optimum {kl:.4f} (base {base_kl:.3f})"
    assert kl < base_kl / 20, "barely moved off the base distribution"
    # The loss fits RATES, not a normalised law, so the scale is part of the claim --
    # and it is the only part that can see a scalar error in the adjoint's second
    # factor, which cancels under normalisation. The target rate is
    # p_base * e^{-g} / E_{p_base}[e^{-g}], which sums to 1 by construction.
    assert abs(float(u.sum()) - 1.0) < 0.05, \
        f"lambda={lam}: total rate {float(u.sum()):.4f} != 1 -- the adjoint is mis-scaled"


def test_temperature_bias_grows_with_lambda():
    """lambda is the inverse temperature and the estimator degrades as it rises --
    which is why docs/dam_design.md defaults DAM_LAMBDA to 0.3 for this reward span
    rather than 1.0. Pinned so the default has a reason attached to it."""
    kls = []
    for lam in (0.3, 3.0):
        p_base, gvec = _problem(lam)
        kls.append(_kl(_fit(p_base, gvec)[0], _target(p_base, gvec)))
    assert kls[1] > kls[0], f"KL did not grow with lambda: {kls}"


def test_dam_concentrates_at_very_low_temperature():
    """The counterpart: as lambda grows the tilted optimum approaches a point mass,
    and the estimator must follow it rather than stalling at p_base."""
    p_base, gvec = _problem(lam=6.0)
    got, _ = _fit(p_base, gvec, steps=2000)
    tgt = _target(p_base, gvec)
    # At lambda=6 the target concentrates 3.5x above the base's own mode. The claim is
    # that the estimator FOLLOWS it -- finds the same mode and matches its mass -- not
    # that it reaches a point mass, which this reward span does not produce.
    assert float(tgt.max()) > 3 * float(p_base.max()), "fixture is not concentrated enough"
    assert got.argmax() == tgt.argmax(), "did not find the mode"
    assert abs(float(got.max()) - float(tgt.max())) < 0.02, \
        f"mode mass {float(got.max()):.3f} vs target {float(tgt.max()):.3f}"


# ================================================================ the gate can fail
@pytest.mark.parametrize("bug", ["drop_exp_g", "sum_normalise", "no_importance"])
def test_gate_rejects_mistranscribed_adjoints(bug):
    """A gate that cannot fail is not a gate. Each of these is a real misreading of
    Eq. (13) that the plan's review caught or nearly missed, and each must move the
    fixed point far enough that test_dam_reaches_kl_optimum would go red."""
    def broken(log_ratio_Z, g_Z, log_ratio_X1, g_X1, *, clamp=10.0):
        k = log_ratio_X1.shape[-1]
        if bug == "drop_exp_g":               # bare density ratio in the weight
            second = torch.logsumexp(log_ratio_X1, -1) - math.log(k)
        elif bug == "sum_normalise":          # normalise weights to sum 1, not mean 1
            second = torch.logsumexp(log_ratio_X1 - g_X1, -1)
        else:                                 # forget the importance ratio entirely
            second = torch.logsumexp(-g_X1, -1) - math.log(k)
        return ((log_ratio_Z - g_Z) - second).clamp(-clamp, clamp), 0.0

    p_base, gvec = _problem(lam=0.3)          # the regime the gate actually runs in
    tgt = _target(p_base, gvec)
    good_p, good_u = _fit(p_base, gvec)
    bad_p, bad_u = _fit(p_base, gvec, adjoint=broken)
    good, bad = _kl(good_p, tgt), _kl(bad_p, tgt)
    exact_u = p_base * torch.exp(-gvec) / (p_base * torch.exp(-gvec)).sum()
    assert good < 0.005 and abs(float(good_u.sum()) - 1.0) < 0.05

    # A mistranscribed adjoint must break the SHAPE or the SCALE. Note "no_importance"
    # and "drop_exp_g" leave the shape almost intact here, because in a one-jump chain
    # the second factor is a single scalar and cancels under normalisation -- which is
    # precisely why the gate asserts the total rate as well.
    shape_broken = bad > 0.05 and bad > 10 * good
    scale_broken = abs(float(bad_u.sum()) - 1.0) > 0.05
    assert shape_broken or scale_broken, (
        f"bug {bug!r} gave KL {bad:.4f} (correct {good:.4f}) and total rate "
        f"{float(bad_u.sum()):.4f} -- the gate would not have caught it"
    )


# ================================================================ the y = x control
def test_adjoint_is_one_at_y_equals_x():
    """The health metric section 8.5 calls for: when y = x the true adjoint is exactly
    1, so log a_hat must sit at 0 up to estimator noise. This is what separates
    'lambda is too hot' from 'the reward is doing work', and unlike the clamp fraction
    it is not zero by construction."""
    p_base, gvec = _problem(lam=0.3)
    torch.manual_seed(SEED)
    idx = torch.multinomial(p_base, 100_000, replacement=True)
    ratio = torch.zeros_like(gvec[idx])                      # p_theta = p_base
    # y = x: the first factor is the same expectation as the second
    log_a, frac = discrete_adjoint(
        estimate_neg_value(ratio, gvec[idx]).reshape(1),
        torch.zeros(1),
        ratio.unsqueeze(0),
        gvec[idx].unsqueeze(0),
    )
    assert abs(float(log_a)) < 0.05, f"y=x control gave log E[a_hat] = {float(log_a):+.4f}"
    assert frac == 0.0


def test_clamp_fraction_is_zero_when_lambda_times_span_fits():
    """The clamp is a saturating envelope, not a health metric: at lambda=1 over a
    span of 10 it provably never fires. Pinned so nobody reports it as a diagnostic."""
    p_base, gvec = _problem(lam=1.0, span=10.0)
    torch.manual_seed(SEED)
    idx = torch.multinomial(p_base, 4096, replacement=True)
    _, frac = discrete_adjoint(torch.zeros(S), gvec,
                               torch.zeros_like(gvec[idx]).unsqueeze(0),
                               gvec[idx].unsqueeze(0), clamp=10.0)
    assert frac == 0.0, f"clamp fired at {frac:.3f} -- it is meant to be inert here"
