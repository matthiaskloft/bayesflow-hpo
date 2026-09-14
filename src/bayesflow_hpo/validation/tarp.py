"""TARP: expected coverage of Tests of Accuracy with Random Points.

A *joint*, optionally data-dependent accuracy test, in contrast with the
per-parameter metrics in :mod:`~bayesflow_hpo.validation.registry`. It is
exposed through the joint metric path (see
:class:`~bayesflow_hpo.validation.registry.JointMetricInputs`) because it
needs every parameter at once and, to be worth running, a reference point
derived from the conditioning data.

Implements Algorithm 2 of Lemos et al. (2023), transcribed from the paper:

    for i in 1..N_sims:
        theta_r ~ p~(theta_r | x)                      # reference point
        f_i = (1/n) sum_j 1[ d(theta_ij, theta_r) < d(theta*_i, theta_r) ]
    ECP(p_hat, alpha, D) = (1/N_sims) sum_i 1(f_i < 1 - alpha)

so the coverage curve is the ECDF of the ``f_i`` and, for an exact
posterior, the diagonal.

The implementation is ported from the author's own ``bayesflow-irt`` (commit
``ffc68d5``, ``src/bayesflow_irt/sbc.py``) rather than depended upon: the two
packages are siblings in the dependency graph, and a generic HPO helper must
not import a domain extension. ``tests/test_validation/test_tarp.py`` checks
this port against that revision numerically, not by inspection.

References
----------
Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023).
    Sampling-based accuracy testing of posterior estimators for general
    inference. In *Proceedings of the 40th International Conference on
    Machine Learning* (pp. 19256-19273). PMLR.
    https://doi.org/10.48550/arXiv.2302.03026
    Algorithm 2 (the estimator implemented here); Theorem 3 (positionable
    regions); Section 3.1 (HPD coverage is blind to
    ``p_hat(theta|x) = p(theta)``); Section 4.2 (robustness to the distance
    metric); Section 4.3 (an x-independent reference point shares that
    blindness).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Cap on the elements materialized per chunk of the distance computation.
_TARP_CHUNK_ELEMENTS = 2e7

# Spawn key isolating the reference-point stream from any stream a caller
# derives from the same seed. See the comment at its use site: without it,
# seeding a simulator and TARP alike makes each reference point an affine
# image of its own truth.
_TARP_REFERENCE_STREAM = 0x7A89


def compute_tarp_coverage(
    posterior_draws: np.ndarray,
    true_values: np.ndarray,
    *,
    reference_points: np.ndarray | None = None,
    resolution: int = 20,
    metric: str = "euclidean",
    standardize: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Expected coverage of TARP regions, a *joint* accuracy test.

    Implements Algorithm 2 of Lemos, Coogan, Hezaveh and Perreault-Levasseur
    (2023), "Sampling-Based Accuracy Testing of Posterior Estimators for
    General Inference" (see ``references.md``). For each simulation ``i`` with
    true parameters ``theta*_i`` and posterior draws ``theta_ij``, a reference
    point ``theta_r`` is drawn and

    ``f_i = (1/n) * sum_j 1[d(theta_ij, theta_r) < d(theta*_i, theta_r)]``

    is the fraction of draws lying closer to the reference point than the
    truth does. The expected coverage probability at credibility level
    ``1 - alpha`` is ``mean_i 1[f_i < 1 - alpha]`` -- the ECDF of the ``f_i``.
    For an exact posterior the ``f_i`` are uniform, so the curve is the
    diagonal.

    **Why this and not marginal coverage.** TARP regions are balls centred on
    a freely chosen reference point, which makes the region generator
    *positionable*. Lemos et al.'s Theorem 3 shows that correct expected
    coverage over every position function implies the estimator equals the
    true posterior -- a conclusion HPD coverage cannot reach, because HPD
    regions are not positionable. Concretely, an estimator that ignores the
    data entirely (``p_hat(theta|x) = p(theta)``) has *perfect* HPD coverage
    at every level, and marginal per-parameter checks share the blind spot
    (Modrak et al. 2025). TARP also tests the joint rather than the
    marginals, so it can see wrong correlation structure.

    **The reference distribution is not a free choice.** Section 4.3 of the
    paper is explicit that TARP with an ``x``-*independent* reference point is
    **also blind** to the ``p_hat(theta|x) = p(theta)`` case: the authors
    compute that estimator's coverage three ways and state they "expect the
    first two methods" -- HPD, and TARP with ``theta_r ~ U(0, 1)`` -- "to have
    ECP equal to ``1 - alpha``, but not for the third", the third being a
    reference point that depends on the data. So a TARP run with the default
    random reference points returns a *clean* result for exactly the failure
    this project suspects in its bad-basin checkpoints. Pass
    ``reference_points`` computed from the data to get the version with teeth.
    Running both and comparing is what makes the diagnosis; a clean
    random-reference curve on its own must never be reported as an all-clear.

    Parameters
    ----------
    posterior_draws : np.ndarray
        Draws with shape ``(n_simulations, n_draws, n_parameters)``.
    true_values : np.ndarray
        True parameters with shape ``(n_simulations, n_parameters)``.
    reference_points : np.ndarray, optional
        One reference point per simulation, shape
        ``(n_simulations, n_parameters)``, in the same units as
        ``true_values``. When ``None``, points are drawn uniformly over the
        hypercube spanned by the standardized truths, reproducing the paper's
        default -- which carries the blind spot described above.
    resolution : int
        Number of credibility levels at which the curve is evaluated.
    metric : {"euclidean", "manhattan"}
        Distance on parameter space. Section 4.2 reports the method is robust
        to this choice.
    standardize : bool
        Standardize each parameter dimension before computing distances.
        Required whenever dimensions have different scales -- ``log a`` and
        ``b`` do -- because otherwise the balls are dominated by whichever
        parameter has the larger variance and the distance is not meaningful.
        Raises if any dimension's truth is constant, since that cannot be
        standardized; slice such dimensions out yourself rather than letting
        them be silently ignored. Centring and scaling use the **true** values
        only, so the metric does
        not depend on the estimator under test and two checkpoints stay
        comparable. Taking the scale from the draws instead would let a wider
        (worse) posterior score better by shrinking its own distances.
    seed : int, optional
        Seed for the reference-point draw. Ignored when ``reference_points``
        is supplied. The stream is domain-separated from the seed, so
        passing the same number here as to a simulator is safe; drawing the
        references from ``default_rng(seed)`` directly is not, and the
        comment at the draw site says why.

    Returns
    -------
    dict
        ``credibility_levels``, ``expected_coverage``, the raw ``f_i`` as
        ``coverage_fractions``, ``tarp_error`` (median over levels of
        ``|ECP - level|``, the same *aggregation* as
        :func:`compute_calibration_error` though on a different level grid, so
        the two read side by side but are not the same statistic),
        ``max_deviation`` and ``reference_mode`` (``"provided"`` when
        ``reference_points`` was passed, ``"random"`` when drawn here).

        ``reference_mode`` deliberately does not claim ``"data_dependent"``.
        The function cannot tell how a supplied array was built, and the
        distinction is load-bearing: only an x-dependent reference detects a
        posterior that ignores the data. Record that property alongside the
        result yourself if you need it asserted.

    Notes
    -----
    ``f_i`` is supported on ``{0, 1/n_draws, ..., 1}``, so its ECDF is a
    staircase and ``tarp_error`` cannot reach zero. At ``n_draws = 5`` the
    floor is exactly ``1 / (resolution + 1)`` and dominates everything else;
    by ``n_draws = 100`` it has fallen below the Monte Carlo noise from a few
    thousand simulations, so beyond that point the residual is set by
    ``n_simulations`` rather than by discreteness. Use at least 100 draws. Any
    quoted floor figure is only meaningful alongside the ``n_simulations`` and
    ``resolution`` it was measured at, since those fix the noise it is being
    compared against.

    With ``reference_points=None`` and ``seed=None`` the reference draw is
    non-deterministic, so two identical calls return slightly different
    numbers. Pass ``seed`` for anything quoted in a report or committed.
    """
    posterior_draws = np.asarray(posterior_draws, dtype=float)
    true_values = np.asarray(true_values, dtype=float)
    if posterior_draws.ndim != 3:
        raise ValueError(
            "posterior_draws must be 3D (n_simulations, n_draws, n_parameters); "
            f"got shape {posterior_draws.shape}."
        )
    n_sims, n_draws, n_params = posterior_draws.shape
    if true_values.shape != (n_sims, n_params):
        raise ValueError(
            "true_values must have shape (n_simulations, n_parameters) matching "
            f"posterior_draws: {true_values.shape} vs {(n_sims, n_params)}."
        )
    if n_sims < 1:
        raise ValueError("posterior_draws must contain at least one simulation.")
    if n_draws < 2:
        raise ValueError("posterior_draws needs at least 2 draws per simulation.")
    if metric not in {"euclidean", "manhattan"}:
        raise ValueError("metric must be 'euclidean' or 'manhattan'.")
    if resolution < 1:
        raise ValueError("resolution must be >= 1.")

    # Non-finite values do not raise here by accident: `d_draws < d_truth` is
    # False for NaN, so NaN draws would be silently counted as "not closer",
    # biasing f_i downward with no error and no warning. A partially diverged
    # checkpoint would then produce a mildly-off but entirely plausible curve
    # instead of announcing that it diverged.
    n_bad_draws = int(np.count_nonzero(~np.isfinite(posterior_draws)))
    n_bad_truth = int(np.count_nonzero(~np.isfinite(true_values)))
    if n_bad_draws or n_bad_truth:
        raise ValueError(
            "compute_tarp_coverage requires finite inputs: "
            f"{n_bad_draws} non-finite posterior draw(s) and {n_bad_truth} "
            "non-finite true value(s). Drop or impute the affected "
            "simulations deliberately rather than letting them bias the "
            "coverage fractions."
        )

    # A parameter whose truth is constant across simulations cannot be
    # standardized, and the caller has to say what it means. An earlier
    # version silently dropped such dimensions, on the reasoning that a
    # constant dimension left in raw units would dominate every distance and
    # wreck a calibrated posterior's score. That reasoning was wrong, and the
    # drop actively hid real failures:
    #
    #   * Keeping the dimension costs a *correct* posterior nothing. If the
    #     draws are also constant there, the same term is added to both sides
    #     of ``d(draw, ref) < d(truth, ref)`` -- ``sqrt(S + c^2)`` for
    #     euclidean, ``+|c|`` for manhattan -- so every ``f_i`` is unchanged.
    #     Verified numerically for both metrics and several reference values.
    #   * Dropping it hides an *incorrect* one. A posterior placing sd-50
    #     mass on a point-mass parameter scored a clean 0.006 with the drop
    #     and 0.48 without it. The 0.5 that motivated the drop was a true
    #     positive being suppressed, not a scaling artifact.
    #
    # So there is no upside and a serious downside. Raise, and let the caller
    # slice out dimensions that are fixed settings rather than parameters
    # under test. Only ``standardize=True`` is affected: without
    # standardization nothing is divided by the zero scale, every dimension
    # stays in the same raw units, and the dominance problem cannot arise.
    if standardize:
        centre = true_values.mean(axis=0)
        scale = true_values.std(axis=0)
        if not (np.all(np.isfinite(scale)) and np.all(np.isfinite(centre))):
            bad = [
                int(i)
                for i in np.flatnonzero(
                    ~(np.isfinite(scale) & np.isfinite(centre))
                )
            ]
            raise ValueError(
                f"Parameter dimension(s) {bad} have a non-finite mean or "
                "standard deviation, so they cannot be standardized. "
                "np.std squares before averaging, so magnitudes above about "
                "1e154 overflow to inf; the z-scores would collapse to zero "
                "and the dimension would drop out of every distance without "
                "any sign in the result. Rescale the parameters first."
            )
        constant = np.flatnonzero(scale <= 0)
        if constant.size:
            raise ValueError(
                "Parameter dimension(s) "
                f"{[int(i) for i in constant]} are constant across "
                "simulations and cannot be standardized. Slice them out "
                "if they are fixed settings rather than parameters under "
                "test, or pass standardize=False if every dimension is "
                "already on a common scale. They are not dropped "
                "automatically: a posterior with spread on a point-mass "
                "parameter is genuinely miscalibrated, and dropping the "
                "dimension would report it as clean."
            )
    else:
        centre = np.zeros(n_params)
        scale = np.ones(n_params)

    draws_z = (posterior_draws - centre) / scale
    truth_z = (true_values - centre) / scale
    n_kept = n_params

    if reference_points is None:
        # Domain-separated, NOT `default_rng(seed)`. This is a deliberate
        # divergence from the bayesflow-irt revision this file is ported
        # from, and it fixes a silent failure found while porting:
        #
        # `default_rng(s).uniform(lo, hi, size=(n_sims, n_params))` consumes
        # the same underlying uniform stream as a caller's
        # `default_rng(s).uniform(a, b, size=(n_sims, n_params))`. Seed a
        # simulator and TARP with the same number -- the obvious thing to do
        # for a reproducible study, and what `seed=42` defaults invite --
        # and every reference point becomes an AFFINE IMAGE of its own
        # simulation's truth: measured correlation exactly 1.0. The truth is
        # then the closest point to its own reference, no draw beats it,
        # every f_i collapses to ~0, and a perfectly calibrated posterior
        # scores `tarp_error = 0.5`, the worst value the statistic can take.
        # Nothing raises, and the number is not obviously nonsense.
        #
        # A spawn key makes the collision impossible instead of documented:
        # the reference stream is a different child of the same seed, so it
        # cannot coincide with any stream the caller derives from it.
        rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(_TARP_REFERENCE_STREAM,))
            if seed is not None
            else None
        )
        # Percentiles, not min/max: the extremes of n_simulations draws grow
        # like sqrt(2 log n_simulations), so a min/max box would silently
        # widen with sample size and make tarp_error magnitudes incomparable
        # between runs of different length.
        lo = np.percentile(truth_z, 1.0, axis=0)
        hi = np.percentile(truth_z, 99.0, axis=0)
        hi = np.where(hi > lo, hi, lo + 1.0)
        refs_z = rng.uniform(lo, hi, size=(n_sims, n_kept))
        reference_mode = "random"
    else:
        reference_points = np.asarray(reference_points, dtype=float)
        if reference_points.shape != (n_sims, n_params):
            raise ValueError(
                "reference_points must have shape (n_simulations, n_parameters): "
                f"{reference_points.shape} vs {(n_sims, n_params)}."
            )
        if not np.all(np.isfinite(reference_points)):
            raise ValueError("reference_points must be finite.")
        refs_z = (reference_points - centre) / scale
        # "provided", NOT "data_dependent". Whether the reference depends on x
        # is exactly what decides if this diagnostic has teeth (Lemos et al.
        # 2023, Sec. 4.3: an x-independent reference reports perfect coverage
        # for a posterior that ignores the data), and passing an array here
        # says nothing about how it was built -- externally generated random
        # references, supplied only for reproducibility, arrive by the same
        # route. Claiming the safer mode on that basis would let a persisted
        # result assert a guarantee it does not have.
        reference_mode = "provided"

    if metric == "euclidean":
        # einsum here too, matching the draw-distance branch below: different
        # summation orders leave the two sides of the strict `<` inconsistent
        # at the 1e-16 level, which only matters for a draw sitting exactly on
        # the truth but costs nothing to avoid.
        _dt = truth_z - refs_z
        dist_truth = np.sqrt(np.einsum("ij,ij->i", _dt, _dt))
    else:
        dist_truth = np.sum(np.abs(truth_z - refs_z), axis=-1)

    # Chunked over simulations: the difference tensor is
    # (n_simulations, n_draws, n_parameters) float64 and the squaring
    # allocates a second one before reducing, so the peak is twice its size.
    # At 5000 x 2000 x 200 that is ~32 GB materialized all at once.
    fractions = np.empty(n_sims, dtype=float)
    chunk = max(1, int(_TARP_CHUNK_ELEMENTS // max(1, n_draws * n_kept)))
    for start in range(0, n_sims, chunk):
        stop = min(start + chunk, n_sims)
        diff = draws_z[start:stop] - refs_z[start:stop, None, :]
        if metric == "euclidean":
            dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        else:
            dist = np.sum(np.abs(diff), axis=-1)
        fractions[start:stop] = np.mean(
            dist < dist_truth[start:stop, None], axis=1
        )

    levels = np.linspace(0.0, 1.0, num=resolution + 2)[1:-1]
    expected = np.array([np.mean(fractions < level) for level in levels])
    deviation = np.abs(expected - levels)
    return {
        "credibility_levels": levels,
        "expected_coverage": expected,
        "coverage_fractions": fractions,
        "tarp_error": float(np.median(deviation)),
        "max_deviation": float(np.max(deviation)),
        "reference_mode": reference_mode,
        "n_simulations": int(n_sims),
        "n_draws": int(n_draws),
        "n_parameters": int(n_params),
    }
