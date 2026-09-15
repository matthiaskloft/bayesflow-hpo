"""TARP coverage: the port, and Algorithm 2's defining properties.

``compute_tarp_coverage`` is ported from ``bayesflow-irt`` rather than
depended upon, because the two packages are siblings and a generic HPO
helper must not import a domain extension. A port is only as good as its
verification, so this checks it two ways:

1. against the *properties* Algorithm 2 guarantees -- an exact posterior
   gives the diagonal, a degenerate one does not -- which is what would
   catch a transcription error;
2. against the source revision numerically, when that checkout is reachable
   (``BF_HPO_IRT_SBC``), which is what would catch a silent drift.

Design: ``docs/plans/plan-joint-metric-path.md`` D6.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from bayesflow_hpo.validation.tarp import compute_tarp_coverage

SEED = 20260914
# A DIFFERENT seed for the reference draw than for the data. The spawn key
# in `tarp.py` makes sharing one safe, but these tests should not depend on
# that fix to be meaningful -- `test_the_reference_stream_deliberately_
# diverges_from_the_source` tests it head-on instead.
REF_SEED = 771


def _gaussian_case(
    n_sims: int = 400,
    n_draws: int = 200,
    n_params: int = 3,
    *,
    correct: bool,
    seed: int = SEED,
):
    """The paper's Gaussian toy model (Section 4.1), in miniature.

    ``correct=True`` draws the posterior mean from ``N(theta*, Sigma)``, so
    the coverage probabilities are uniform. ``correct=False`` returns draws
    that ignore the truth entirely -- the ``p_hat(theta|x) = p(theta)``
    estimator Section 3.1 shows HPD coverage cannot see.

    The scales follow the paper: ``theta* ~ U(-5, 5)`` and
    ``log sigma ~ U(-5, -1)``, which is the detail that makes the "correct"
    case actually correct. A first version of this helper used a flat
    ``sigma = 0.5``, and that is NOT calibrated under a U(-5, 5) prior: the
    estimator ``N(mean, Sigma)`` is only the true posterior under a flat
    prior, so at 0.5 the truncation at the prior's edges is felt by half the
    simulations. The coverage curve bowed a systematic +0.07 above the
    diagonal and a KS test rejected uniformity at p = 4e-19 -- which read
    exactly like a broken implementation. At the paper's sigmas the
    truncation is negligible and the same code gives tarp_error ~ 0.015.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-5.0, 5.0, size=(n_sims, n_params))
    sigma = np.exp(rng.uniform(-5.0, -1.0, size=(n_sims, n_params)))
    if correct:
        mean = truth + rng.normal(size=(n_sims, n_params)) * sigma
    else:
        mean = rng.uniform(-5.0, 5.0, size=(n_sims, n_params))
    draws = (
        mean[:, None, :]
        + rng.normal(size=(n_sims, n_draws, n_params)) * sigma[:, None, :]
    )
    return draws, truth


# ---------------------------------------------------------------------------
# Algorithm 2's defining properties
# ---------------------------------------------------------------------------


def test_an_exact_posterior_gives_the_diagonal():
    """For an exact posterior the ``f_i`` are uniform, so ECP(c) = c."""
    draws, truth = _gaussian_case(correct=True)
    out = compute_tarp_coverage(draws, truth, seed=REF_SEED)

    # Measured 0.006-0.025 over five seeds at this size; the residual is
    # the discreteness staircase of f_i, not a bias.
    assert out["tarp_error"] < 0.04, out["tarp_error"]
    np.testing.assert_allclose(
        out["expected_coverage"], out["credibility_levels"], atol=0.06
    )


def test_a_data_ignoring_posterior_is_caught_by_a_data_dependent_reference():
    """The failure the whole two-key split exists for.

    Section 4.3: TARP with an x-INDEPENDENT reference point is blind to
    ``p_hat(theta|x) = p(theta)`` in the same way HPD coverage is. A
    reference derived from the data is what gives the test teeth. Both
    numbers are computed here on the same draws, so the contrast is the
    assertion rather than a claim about one of them.
    """
    draws, truth = _gaussian_case(correct=False)

    random_ref = compute_tarp_coverage(draws, truth, seed=REF_SEED)
    # A reference built from the data: here, the truth itself plus noise
    # stands in for an x-derived coordinate.
    rng = np.random.default_rng(SEED)
    data_ref = compute_tarp_coverage(
        draws,
        truth,
        reference_points=truth + rng.normal(0.0, 0.1, size=truth.shape),
    )

    assert data_ref["tarp_error"] > random_ref["tarp_error"], (
        "the data-dependent reference did not detect a posterior that "
        "ignores the data, which is the property that justifies reporting "
        "tarp_error and tarp_error_random as different metrics"
    )
    assert data_ref["reference_mode"] == "provided"
    assert random_ref["reference_mode"] == "random"


def test_the_coverage_curve_is_the_ecdf_of_the_fractions():
    """ECP(c) = mean_i 1[f_i < c] -- Algorithm 2's final line, directly."""
    draws, truth = _gaussian_case(n_sims=120, n_draws=50, correct=True)
    out = compute_tarp_coverage(draws, truth, seed=REF_SEED)

    f = out["coverage_fractions"]
    for level, ecp in zip(
        out["credibility_levels"], out["expected_coverage"], strict=True
    ):
        assert ecp == pytest.approx(float(np.mean(f < level)))


def test_the_fractions_count_draws_closer_than_the_truth():
    """f_i is the fraction of draws nearer the reference than the truth is.

    Computed here the slow, literal way from Algorithm 2, against a supplied
    reference so there is no randomness to reconcile.
    """
    rng = np.random.default_rng(SEED)
    n_sims, n_draws, n_params = 6, 40, 2
    draws = rng.normal(size=(n_sims, n_draws, n_params))
    truth = rng.normal(size=(n_sims, n_params))
    refs = rng.normal(size=(n_sims, n_params))

    out = compute_tarp_coverage(
        draws, truth, reference_points=refs, standardize=False
    )

    expected = []
    for i in range(n_sims):
        d_truth = np.sqrt(np.sum((truth[i] - refs[i]) ** 2))
        d_draws = np.sqrt(np.sum((draws[i] - refs[i]) ** 2, axis=-1))
        expected.append(np.mean(d_draws < d_truth))
    np.testing.assert_allclose(out["coverage_fractions"], expected)


def test_reference_mode_never_claims_the_reference_is_data_dependent():
    """It cannot know, and the distinction is load-bearing.

    An externally generated *random* reference, supplied only so a run
    reproduces, arrives by the same route as an x-derived one.
    """
    draws, truth = _gaussian_case(n_sims=50, n_draws=20, correct=True)
    rng = np.random.default_rng(SEED)
    out = compute_tarp_coverage(
        draws, truth, reference_points=rng.normal(size=truth.shape)
    )
    assert out["reference_mode"] == "provided"


def test_the_metric_choice_does_not_change_the_verdict():
    """Section 4.2 reports robustness to the distance metric."""
    draws, truth = _gaussian_case(correct=True)
    euc = compute_tarp_coverage(draws, truth, metric="euclidean", seed=REF_SEED)
    man = compute_tarp_coverage(draws, truth, metric="manhattan", seed=REF_SEED)
    assert euc["tarp_error"] < 0.04
    assert man["tarp_error"] < 0.04


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------


def test_two_dimensional_draws_raise():
    """The joint path normalizes to 3-D precisely because this raises."""
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="must be 3D"):
        compute_tarp_coverage(rng.normal(size=(10, 20)), rng.normal(size=(10,)))


def test_non_finite_draws_raise_rather_than_biasing_the_fractions():
    """`d_draws < d_truth` is False for NaN, so a NaN counts as 'not closer'."""
    draws, truth = _gaussian_case(n_sims=20, n_draws=10, correct=True)
    draws[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite inputs"):
        compute_tarp_coverage(draws, truth, seed=SEED)


def test_a_constant_dimension_raises_rather_than_being_dropped():
    """Dropping it would report a genuinely miscalibrated posterior as clean."""
    draws, truth = _gaussian_case(n_sims=30, n_draws=10, correct=True)
    truth[:, 1] = 2.0
    with pytest.raises(ValueError, match="constant across"):
        compute_tarp_coverage(draws, truth, seed=SEED)


def test_a_constant_dimension_is_fine_without_standardization():
    """Nothing is divided by the zero scale, so the dominance problem is moot."""
    draws, truth = _gaussian_case(n_sims=30, n_draws=10, correct=True)
    truth[:, 1] = 2.0
    out = compute_tarp_coverage(draws, truth, standardize=False, seed=SEED)
    assert np.isfinite(out["tarp_error"])


def test_the_seed_makes_the_random_reference_reproducible():
    draws, truth = _gaussian_case(n_sims=50, n_draws=20, correct=True)
    a = compute_tarp_coverage(draws, truth, seed=7)
    b = compute_tarp_coverage(draws, truth, seed=7)
    c = compute_tarp_coverage(draws, truth, seed=8)
    assert a["tarp_error"] == b["tarp_error"]
    assert a["tarp_error"] != c["tarp_error"]


# ---------------------------------------------------------------------------
# The port matches its source
# ---------------------------------------------------------------------------


def _load_reference_implementation():
    """Import the bayesflow-irt revision this module was ported from.

    Located via ``$BF_HPO_IRT_SBC`` (a path to ``sbc.py``, or to a
    bayesflow-irt checkout whose pinned commit is read with git). Skipped
    when unavailable, because a sibling repository is not a test dependency
    -- but when it IS reachable, drift between the two must fail rather than
    go unnoticed.
    """
    pinned = "ffc68d59ae311d1aeaa5e066f39f7e0badc6c853"
    location = os.environ.get("BF_HPO_IRT_SBC")
    if not location:
        pytest.skip("set BF_HPO_IRT_SBC to compare against bayesflow-irt")

    path = Path(location)
    if path.is_dir():
        src = subprocess.run(
            ["git", "-C", str(path), "show", f"{pinned}:src/bayesflow_irt/sbc.py"],
            capture_output=True,
        )
        if src.returncode != 0:
            detail = src.stderr.decode("utf-8", "replace").strip()
            pytest.skip(f"commit {pinned[:7]} not in {path}: {detail}")
        # Decoded as UTF-8 explicitly, matching the `path.is_file()` branch
        # below. `text=True` decodes with the LOCALE codec, so on a Windows
        # box every non-ASCII character in the reference implementation is
        # replaced and the module compared against is not the source it
        # claims to be -- silently, since the mangling lands in comments.
        source = src.stdout.decode("utf-8")
    elif path.is_file():
        source = path.read_text(encoding="utf-8")
    else:
        pytest.skip(f"BF_HPO_IRT_SBC={location} does not exist")

    module = types.ModuleType("_irt_sbc_reference")
    sys.modules["_irt_sbc_reference"] = module
    try:
        exec(compile(source, "_irt_sbc_reference.py", "exec"), module.__dict__)
    except ImportError as exc:  # its own optional dependencies
        pytest.skip(f"reference module needs {exc.name}")
    return module.compute_tarp_coverage


@pytest.mark.parametrize("correct", [True, False], ids=["exact", "degenerate"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"metric": "manhattan"},
        {"resolution": 50},
        {"standardize": False},
    ],
    ids=["default", "manhattan", "resolution50", "raw"],
)
def test_the_port_reproduces_the_source_revision(correct, kwargs):
    """Bit-for-bit on everything except the reference draw.

    Run against SUPPLIED reference points, so the comparison covers the
    whole computation -- standardization, both metrics, the distance
    chunking, the level grid, the aggregation -- with no RNG in it. The
    random-reference branch deliberately diverges; see the test below.
    """
    reference = _load_reference_implementation()
    draws, truth = _gaussian_case(n_sims=150, n_draws=60, correct=correct)
    refs = np.random.default_rng(99).normal(size=truth.shape)
    kwargs = {**kwargs, "reference_points": refs}

    ours = compute_tarp_coverage(draws, truth, **kwargs)
    theirs = reference(draws, truth, **kwargs)

    assert ours["reference_mode"] == theirs["reference_mode"]
    assert ours["tarp_error"] == pytest.approx(theirs["tarp_error"])
    assert ours["max_deviation"] == pytest.approx(theirs["max_deviation"])
    np.testing.assert_allclose(
        ours["coverage_fractions"], theirs["coverage_fractions"]
    )
    np.testing.assert_allclose(
        ours["expected_coverage"], theirs["expected_coverage"]
    )


def test_the_reference_stream_deliberately_diverges_from_the_source():
    """The one intentional difference, and the reason for it.

    Drawing the references from ``default_rng(seed)`` consumes the same
    uniform stream a caller's simulator does. Seeding both alike -- the
    obvious thing to do for a reproducible study -- then makes every
    reference point an affine image of its own truth, so the truth is the
    closest point to its own reference and a perfectly calibrated posterior
    scores the worst value the statistic can take. Silently.
    """
    reference = _load_reference_implementation()

    seed = 1
    rng = np.random.default_rng(seed)
    n_sims, n_draws, n_params, sigma = 400, 200, 3, 0.5
    # Truth drawn FIRST from this seed, at the shape the references will use
    # -- the collision the spawn key exists to prevent.
    truth = rng.uniform(-5.0, 5.0, size=(n_sims, n_params))
    mean = truth + rng.normal(0.0, sigma, size=(n_sims, n_params))
    draws = mean[:, None, :] + rng.normal(
        0.0, sigma, size=(n_sims, n_draws, n_params)
    )

    ours = compute_tarp_coverage(draws, truth, seed=seed)
    theirs = reference(draws, truth, seed=seed)

    assert theirs["tarp_error"] == pytest.approx(0.5), (
        "the collision this guards against no longer reproduces in the "
        "source revision; recheck whether the spawn key is still needed"
    )
    assert ours["tarp_error"] < 0.15, ours["tarp_error"]


def test_seeding_tarp_like_the_simulator_is_safe():
    """The property the spawn key buys, without needing the source repo."""
    seed = 3
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-5.0, 5.0, size=(300, 2))
    mean = truth + rng.normal(0.0, 0.5, size=(300, 2))
    draws = mean[:, None, :] + rng.normal(0.0, 0.5, size=(300, 150, 2))

    same = compute_tarp_coverage(draws, truth, seed=seed)
    other = compute_tarp_coverage(draws, truth, seed=seed + 1000)

    # Sharing the seed must not change the verdict.
    assert same["tarp_error"] < 0.15
    assert other["tarp_error"] < 0.15


# ---------------------------------------------------------------------------
# D6 -- the two-key split and the reference contract
# ---------------------------------------------------------------------------


def _joint_inputs(n_sims=60, n_draws=40, n_params=2, cond_id=0, n_conditions=3):
    from bayesflow_hpo.validation.registry import JointMetricInputs

    draws, truth = _gaussian_case(
        n_sims=n_sims, n_draws=n_draws, n_params=n_params, correct=True
    )
    return JointMetricInputs(
        draws=draws,
        true_values=truth,
        param_keys=tuple(f"p{i}" for i in range(n_params)),
        sim_batch={f"p{i}": truth[:, i] for i in range(n_params)}
        | {"x": np.zeros((n_sims, 2))},
        data_keys=("x",),
        approximator=object(),
        cond_id=cond_id,
        n_conditions=n_conditions,
    )


def test_the_reference_mode_decides_the_key():
    """Two numbers that cannot be compared must not be comparable by name."""
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    inputs = _joint_inputs()

    random_out = make_tarp_joint_metric()(inputs)
    provided_out = make_tarp_joint_metric(
        reference_points=lambda i: np.zeros((i.draws.shape[0], i.draws.shape[2]))
    )(inputs)

    assert set(random_out) == {"tarp_error_random"}
    assert set(provided_out) == {"tarp_error"}


def test_tarp_error_is_an_objective_and_the_random_key_is_a_diagnostic():
    """A diagnostic cannot be optimized, which is the guard's whole job."""
    from bayesflow_hpo.validation.registry import validate_objective_metric_kinds

    validate_objective_metric_kinds(["tarp_error"])  # must not raise
    with pytest.raises(ValueError, match="Diagnostic metric"):
        validate_objective_metric_kinds(["tarp_error_random"])


def test_a_single_array_reused_across_conditions_is_rejected():
    """The form a caller reaches for first, and it is a different metric
    on every condition."""
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    inputs = _joint_inputs()
    shared = np.zeros((inputs.draws.shape[0], inputs.draws.shape[2]))
    fn = make_tarp_joint_metric(reference_points=shared)

    with pytest.raises(TypeError, match="PER CONDITION"):
        fn(inputs)


def test_a_sequence_of_arrays_is_indexed_by_condition():
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    per_condition = [
        np.full((60, 2), float(c)) for c in range(3)
    ]
    fn = make_tarp_joint_metric(reference_points=per_condition)

    values = [fn(_joint_inputs(cond_id=c))["tarp_error"] for c in range(3)]
    assert len({round(v, 12) for v in values}) == 3, (
        "every condition produced the same value, so the sequence was not "
        "actually indexed by cond_id"
    )


def test_a_short_sequence_says_which_condition_is_missing():
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    fn = make_tarp_joint_metric(reference_points=[np.zeros((60, 2))])
    with pytest.raises(ValueError, match="condition 2"):
        fn(_joint_inputs(cond_id=2))


def test_a_misshapen_reference_is_rejected():
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    fn = make_tarp_joint_metric(
        reference_points=lambda i: np.zeros((5, 5))
    )
    with pytest.raises(ValueError, match="expected"):
        fn(_joint_inputs())


def test_the_provider_receives_the_data_it_needs():
    """A real reference is derived from sim_batch, so it must be there."""
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    seen = {}

    def provider(inputs):
        seen["keys"] = set(inputs.sim_batch)
        seen["data_keys"] = inputs.data_keys
        seen["cond"] = inputs.cond_id
        return np.zeros((inputs.draws.shape[0], inputs.draws.shape[2]))

    make_tarp_joint_metric(reference_points=provider)(_joint_inputs(cond_id=1))
    assert "x" in seen["keys"]
    assert seen["data_keys"] == ("x",)
    assert seen["cond"] == 1


def test_conditions_do_not_share_a_reference_draw():
    """Sharing one would correlate the per-condition noise."""
    from bayesflow_hpo.validation.tarp import make_tarp_joint_metric

    fn = make_tarp_joint_metric(seed=5)
    a = fn(_joint_inputs(cond_id=0))["tarp_error_random"]
    b = fn(_joint_inputs(cond_id=1))["tarp_error_random"]
    assert a != b


def test_tarp_error_cannot_run_at_its_registered_default():
    """Registered so the routing surface knows it; refused before inference.

    It has to be in the registry, because `_metric_names_for_pipeline` drops
    names `producer_for_key` returns None for. It cannot be computed without
    a reference. Raising per condition instead would route it through the
    joint guard, which invalidates the metric and substitutes its worst
    case -- correct for a numerical failure, far too quiet for a
    configuration one.
    """
    from bayesflow_hpo.validation.registry import (
        is_joint_metric,
        producer_for_key,
        resolve_joint_metrics,
    )

    assert is_joint_metric("tarp_error")
    assert producer_for_key("tarp_error") == "tarp_error"
    with pytest.raises(ValueError, match="requires reference points"):
        resolve_joint_metrics(["tarp_error"])


def test_the_random_key_runs_at_its_registered_default():
    from bayesflow_hpo.validation.registry import resolve_joint_metrics

    fn = resolve_joint_metrics(["tarp_error_random"])["tarp_error_random"]
    out = fn(_joint_inputs())
    assert set(out) == {"tarp_error_random"}
    assert np.isfinite(out["tarp_error_random"])


def test_both_keys_have_a_bounded_penalty():
    """A median of |ECP - level| is bounded by 1; no infinite penalty needed."""
    from bayesflow_hpo.objectives import worst_objective_value

    assert worst_objective_value("tarp_error") == 1.0
    assert worst_objective_value("tarp_error_random") == 1.0
