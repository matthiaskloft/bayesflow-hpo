"""``calibration_error`` and ``mean_calibration_error`` differ, on purpose.

``calibration_error`` calls
:func:`bayesflow.diagnostics.calibration_error` with all defaults, which
aggregates the 20 per-level absolute coverage deviations with
``np.median`` (``calibration_error.py:15``, ``aggregation: Callable =
np.median``).  It was documented as an Expected Calibration Error in
six places for several releases, which it is not: an ECE is a mean over
bins of predicted probability, and this is a median.  See
`issue #83 <https://github.com/matthiaskloft/bayesflow-hpo/issues/83>`_.

The computation of ``calibration_error`` is deliberately frozen -- its
remaining purpose is comparability with stored records -- so the two
metrics are expected to disagree, and this file pins the disagreement
rather than letting a later "cleanup" quietly reconcile them.

The mean-aggregated metric is *not* called ``ece``:
``bf.diagnostics.expected_calibration_error`` already exists and is a
different statistic (bin-weighted, over one-hot model indices, after
Naeini et al. 2015).

The disagreement is not uniform, which is why the construction below is
specific.  Rescaling a whole posterior bends the deviation curve into a
hump peaking at mid-alpha, and there the median reads *higher* than the
mean.  The median's actual defect shows up only when deviation is
concentrated in the tails while the centre behaves, so that is what
:func:`_tail_miscalibrated_posterior` builds.
"""

from __future__ import annotations

import numpy as np

from bayesflow_hpo.objectives import METRIC_DIRECTIONS
from bayesflow_hpo.validation.registry import (
    DEFAULT_METRICS,
    get_metric,
    list_metrics,
    resolve_metrics,
)

# Conjugate Gaussian model: theta ~ N(0, 1), y | theta ~ N(theta, 1).
# The exact posterior is N(y / (1 + sigma^2), sigma^2 / (1 + sigma^2)),
# so drawing from it gives a perfectly calibrated reference against
# which a deliberate defect can be introduced.
_SIGMA = 1.0
_N_SIMS = 4000
_N_SAMPLES = 4000


def _exact_posterior(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draws from the exact posterior, plus the priors that generated it."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, _N_SIMS)
    y = theta + rng.normal(0.0, _SIGMA, _N_SIMS)

    post_mean = y / (1.0 + _SIGMA**2)
    post_sd = np.sqrt(_SIGMA**2 / (1.0 + _SIGMA**2))
    draws = post_mean[:, None] + rng.normal(
        0.0, post_sd, (_N_SIMS, _N_SAMPLES),
    )
    return draws, theta


def _calibrated_posterior(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    return _exact_posterior(seed)


def _tail_miscalibrated_posterior(
    seed: int = 7, clip_sd: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """The exact posterior with its tails truncated at ``clip_sd``.

    Clipping leaves every central quantile untouched -- the middle of
    the calibration curve stays as calibrated as it was -- while the
    extreme quantiles collapse inward, so the nominal 99% interval
    under-covers badly.  Deviation is therefore concentrated at large
    alpha, which is exactly the half of the curve a median over the 20
    levels discards.
    """
    draws, theta = _exact_posterior(seed)
    post_sd = np.sqrt(_SIGMA**2 / (1.0 + _SIGMA**2))
    centre = np.median(draws, axis=1, keepdims=True)
    return (
        np.clip(draws, centre - clip_sd * post_sd, centre + clip_sd * post_sd),
        theta,
    )


def _score(draws: np.ndarray, theta: np.ndarray) -> tuple[float, float]:
    cal = get_metric("calibration_error")(draws, theta)["calibration_error"]
    mean_ce = get_metric("mean_calibration_error")(
        draws, theta,
    )["mean_calibration_error"]
    return cal, mean_ce


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_mean_calibration_error_is_registered_and_resolvable():
    assert "mean_calibration_error" in list_metrics()
    fns = resolve_metrics(["mean_calibration_error"])
    assert callable(fns["mean_calibration_error"])


def test_no_alias_shadows_the_coverage_output_key():
    """``mean_cal_error`` is a ``coverage`` output key, not an alias here.

    Aliasing it to this metric would make the name ambiguous between a
    coverage output and a metric, which ``objective_scalar`` documents as
    a fallback key.
    """
    from bayesflow_hpo.validation.registry import _ALIASES

    assert _ALIASES.get("mean_cal_error") != "mean_calibration_error"


def test_mean_calibration_error_emits_its_own_key():
    draws, theta = _calibrated_posterior()
    assert set(get_metric("mean_calibration_error")(draws, theta)) == {
        "mean_calibration_error",
    }


def test_mean_calibration_error_has_a_direction_entry():
    """Without one, a failed trial's penalty falls back to "unknown scale"."""
    direction = METRIC_DIRECTIONS["mean_calibration_error"]
    assert direction.higher_is_better is False
    assert direction.worst_raw == 1.0
    assert direction.to_minimize(0.25) == 0.25


def test_mean_calibration_error_is_not_in_default_metrics():
    """Adding it to the defaults would change every stored summary's columns."""
    assert "mean_calibration_error" not in DEFAULT_METRICS
    assert "calibration_error" in DEFAULT_METRICS


# ---------------------------------------------------------------------------
# The two metrics are different numbers
# ---------------------------------------------------------------------------


def test_both_are_near_zero_for_a_calibrated_posterior():
    """Neither aggregation invents miscalibration that is not there."""
    cal, mean_ce = _score(*_calibrated_posterior())
    assert cal < 0.02
    assert mean_ce < 0.02


def test_median_hides_tail_miscalibration_that_the_mean_reports():
    """The load-bearing claim of issue #83.

    Truncating the posterior's tails leaves the median aggregation close
    to its calibrated baseline while the mean departs from it sharply.
    A study that searched on ``calibration_error`` would barely notice a
    posterior whose 99% interval is badly too narrow.
    """
    base_cal, base_mean_ce = _score(*_calibrated_posterior())
    bad_cal, bad_mean_ce = _score(*_tail_miscalibrated_posterior())

    # The mean moves much further off its own baseline than the median
    # moves off its own -- compared as ratios, so the two metrics'
    # different baselines are not doing the work.
    assert (bad_mean_ce / base_mean_ce) > 2.0 * (bad_cal / base_cal)

    # And in absolute terms the two disagree by more than a rounding.
    assert bad_mean_ce > 2.0 * bad_cal

    # The absolute bands are the stable claim: swept over seeds 1-20 the
    # mean-aggregated value stays in 0.031-0.041 while the median-
    # aggregated one ranges over 0.001-0.017, i.e. the median can read as
    # barely distinguishable from a perfectly calibrated posterior. The
    # ratios above are noisier -- their denominators are Monte Carlo
    # noise -- so they are asserted loosely and the bands tightly.
    assert 0.025 < bad_mean_ce < 0.050
    assert bad_cal < 0.020


def test_the_two_metrics_are_not_interchangeable():
    """A guard against a later refactor collapsing one into the other."""
    draws, theta = _tail_miscalibrated_posterior()
    cal, mean_ce = _score(draws, theta)
    assert not np.isclose(cal, mean_ce, rtol=0.05, atol=1e-4)
