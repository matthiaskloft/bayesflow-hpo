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

from bayesflow_hpo.objectives import register_metric_direction
from bayesflow_hpo.validation.registry import (
    JointMetricFn,
    JointMetricInputs,
    register_joint_metric,
)

# Cap on the elements materialized per chunk of the distance computation.
_TARP_CHUNK_ELEMENTS = 2e7

# Spawn key isolating the reference-point stream from any stream a caller
# derives from the same seed. See the comment at its use site: without it,
# seeding a simulator and TARP alike makes each reference point an affine
# image of its own truth.
_TARP_REFERENCE_STREAM = 0x7A89

#: Distance metrics `compute_tarp_coverage` accepts. Named here so the
#: factory can reject a bad one before any data is touched.
_TARP_METRICS = frozenset({"euclidean", "manhattan"})

#: Reference distributions `compute_tarp_coverage` can draw from when the
#: caller supplies none. Both are ``x``-independent and so share the blind
#: spot of Lemos et al. (2023) Sec. 4.3; the choice is one of power, not of
#: what the statistic can detect. See the `reference` parameter.
_TARP_REFERENCES = frozenset({"uniform_box", "prior_derangement"})

# Retries before the derangement sampler gives up. With DISTINCT truths a
# uniformly drawn permutation is a derangement with probability -> 1/e ~
# 0.368 for every n_sims >= 2, so the expected number of draws is e ~ 2.7 and
# the chance of exhausting this cap is (1 - 1/e)^64 ~ 1e-13. Duplicated
# truths lower that probability -- the sampler rejects on values, not indices
# -- so exhaustion IS reachable for a prior with dense atoms, and the failure
# raises with that explanation rather than being treated as a bug.
_TARP_DERANGEMENT_RETRIES = 64


def compute_tarp_coverage(
    posterior_draws: np.ndarray,
    true_values: np.ndarray,
    *,
    reference_points: np.ndarray | None = None,
    reference: str = "uniform_box",
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
        ``true_values``. When ``None``, points are drawn according to
        *reference*, which carries the blind spot described above.
    reference : {"uniform_box", "prior_derangement"}
        Which reference distribution to draw from when *reference_points* is
        ``None``. Ignored -- and rejected if set to anything but the default
        -- when *reference_points* is given, since a supplied array is the
        reference.

        ``"uniform_box"`` (default) draws uniformly over the box spanned by
        the 1st and 99th percentiles of the standardized truths.

        ``"prior_derangement"`` gives each simulation a reference point taken
        from another simulation's truth, under a uniformly drawn permutation
        with no fixed points. Because the truths are themselves prior draws,
        this samples ``theta_r ~ p(theta)`` -- the choice Lemos et al. (2023)
        make in Sec. 4.1 ("To pick the TARP reference points, we use the
        prior"), and the convention BayesFlow's own
        ``accuracy_random_points`` follows. Prefer it when the prior is
        correlated or far from box-shaped, where the box puts reference mass
        in corners no truth or draw ever occupies and the distance
        comparison loses its edge. The derangement is what keeps a
        simulation from referencing itself, which would make ``d_truth = 0``
        and pin ``f_i`` at 0.

        Both are ``x``-independent, so neither detects a posterior that
        ignores its data. Sec. 4.2 finds the coverage curve robust to this
        choice across uniform, normal and fixed reference distributions, so
        switching is not expected to move a verdict -- only the precision it
        is reached with. Recorded in the result as ``reference_mode``.
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
        ``max_deviation`` and ``reference_mode``: ``"provided"`` when
        ``reference_points`` was passed, otherwise ``"random"`` for
        ``reference="uniform_box"`` and ``"prior_derangement"`` for
        ``reference="prior_derangement"``.

        The box mode reports ``"random"`` rather than ``"uniform_box"``
        because it predates the choice and persisted results and study pins
        carry the old spelling. Renaming it would make every stored result
        read as a changed configuration on resume, which is exactly the
        signal the pin exists to give truthfully.

        **This field is not the one a study pin carries under that name.**
        ``make_tarp_joint_metric``'s ``joint_metric_settings`` also has a
        ``reference_mode``, and it answers a different question: whether the
        reference was *supplied* (``"provided"``) or drawn here
        (``"random"``), never which distribution was drawn from. So a
        ``prior_derangement`` run reports ``reference_mode="prior_derangement"``
        here while its pin reads ``reference_mode="random"`` plus
        ``reference="prior_derangement"``. Read the pin's ``reference`` key,
        not its ``reference_mode``, to recover the distribution. The names
        collide because the pin's spelling is frozen by studies already on
        disk.

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
    if metric not in _TARP_METRICS:
        raise ValueError("metric must be 'euclidean' or 'manhattan'.")
    if resolution < 1:
        raise ValueError("resolution must be >= 1.")
    if reference not in _TARP_REFERENCES:
        raise ValueError(
            f"reference must be one of {sorted(_TARP_REFERENCES)}, got "
            f"{reference!r}."
        )
    # Rejected rather than ignored. Both arguments name the reference, so a
    # call supplying each says two different things; honouring the array
    # silently would run a different statistic than the one the caller spelled
    # out, under the key that claims the other.
    if reference_points is not None and reference != "uniform_box":
        raise ValueError(
            "reference_points and reference= both specify the reference, and "
            f"{reference!r} cannot apply to a supplied array. Pass "
            "reference_points alone, or reference= alone to draw them here."
        )

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
        if reference == "uniform_box":
            # Percentiles, not min/max: the extremes of n_simulations draws
            # grow like sqrt(2 log n_simulations), so a min/max box would
            # silently widen with sample size and make tarp_error magnitudes
            # incomparable between runs of different length.
            lo = np.percentile(truth_z, 1.0, axis=0)
            hi = np.percentile(truth_z, 99.0, axis=0)
            hi = np.where(hi > lo, hi, lo + 1.0)
            refs_z = rng.uniform(lo, hi, size=(n_sims, n_kept))
            reference_mode = "random"
        else:
            # The truths are prior draws, so permuting them samples the prior
            # exactly -- no density, no bounds, no box to guess.
            #
            # A *derangement*, not any permutation: a fixed point would make
            # simulation i its own reference, so d_truth = 0, no draw is
            # strictly closer, and f_i = 0 regardless of how good the
            # posterior is. One fixed point in n_sims is a small bias; the
            # point is that it is a silent one, biasing f_i downward exactly
            # like the seed collision this module already guards against.
            #
            # Rejection-sampled rather than BayesFlow's single `np.roll`
            # shift. One shift offset determines the entire reference set, so
            # the whole set carries log2(n_sims) bits -- about 9 at 500
            # simulations -- and the references are perfectly dependent on one
            # another. Marginally each is still a prior draw, so the curve
            # stays valid either way, but an HPO objective is compared across
            # trials and wants its Monte Carlo error small rather than
            # concentrated in one integer.
            if n_sims < 2:
                raise ValueError(
                    "reference='prior_derangement' needs at least 2 "
                    "simulations: with one, the only reference available is "
                    "that simulation's own truth, which forces f_i = 0."
                )
            # Rejected on VALUES, not on indices. An index derangement
            # (`perm != arange`) is not enough: if two simulations happen to
            # share a truth -- which a discrete, ordinal or otherwise
            # atom-carrying prior makes ordinary rather than exotic --
            # then `perm[i] != i` can still land a reference numerically
            # equal to truth i, and d_truth is 0 again. Checking the rows
            # closes the gap the index check only appears to close.
            for _ in range(_TARP_DERANGEMENT_RETRIES):
                perm = rng.permutation(n_sims)
                if not np.any(np.all(truth_z[perm] == truth_z, axis=1)):
                    break
            else:
                # Not "treat it as a bug": with duplicated truths this is
                # reachable, and for a prior concentrated on few atoms it is
                # the normal outcome. Raising beats both a silent bias and a
                # loop that cannot terminate.
                n_dup = n_sims - len(np.unique(truth_z, axis=0))
                raise ValueError(
                    "Could not draw a reference assignment in which no "
                    f"simulation references its own truth value, after "
                    f"{_TARP_DERANGEMENT_RETRIES} attempts "
                    f"({n_dup} of {n_sims} truths are duplicates of another "
                    "simulation's). Any such collision makes d_truth = 0 and "
                    "pins that simulation's f_i at 0, biasing the coverage "
                    "curve downward. Use reference='uniform_box', or pass "
                    "reference_points derived from the data, for a prior "
                    "with atoms this dense."
                )
            refs_z = truth_z[perm]
            reference_mode = "prior_derangement"
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


# ---------------------------------------------------------------------------
# TARP as a joint metric
# ---------------------------------------------------------------------------

#: Marks a registered callable that cannot run at its default configuration.
#:
#: ``tarp_error`` has to be *registered* so that `producer_for_key` knows it,
#: because `_pipeline_metrics` drops names that lookup returns None
#: for -- a metric absent from the registry is requested, computed and
#: reported by nothing, with no error anywhere. But it cannot be *computed*
#: without a reference provider, and a default that merely raised per
#: condition would be caught by the joint guard, penalized, and reported as
#: a failed metric: loud enough to find afterwards, too late to act on. This
#: attribute lets `resolve_joint_metrics` refuse at resolve time, before any
#: inference is paid for.
REQUIRES_CONFIGURATION = "_bf_hpo_requires_configuration"


def _references_for_condition(
    reference_points: Any, inputs: JointMetricInputs
) -> np.ndarray:
    """Resolve the caller's reference specification for one condition.

    Accepts a callable evaluated per condition, or a sequence of arrays
    indexed by ``cond_id``. A single ``(n_sims, n_params)`` array is
    **rejected**, and that is the point of this function: the truths differ
    from condition to condition, so one array reused across all of them is a
    different metric on each -- reported under one name, averaged, and
    indistinguishable afterwards from a real value. It is also the form a
    caller reaches for first.
    """
    if callable(reference_points):
        refs = reference_points(inputs)
    elif isinstance(reference_points, np.ndarray) and reference_points.ndim == 3:
        # The natural spelling of "one array per condition" once someone
        # reaches for `np.array(list_of_arrays)`. Accepted rather than
        # rejected with a message about a single shared array, which is a
        # different mistake and quotes a shape that is not
        # (n_sims, n_params).
        if reference_points.shape[0] != inputs.n_conditions:
            raise ValueError(
                f"reference_points has {reference_points.shape[0]} "
                f"conditions, expected {inputs.n_conditions}."
            )
        refs = reference_points[inputs.cond_id]
    elif isinstance(reference_points, np.ndarray):
        raise TypeError(
            "reference_points must be a callable or one array PER CONDITION "
            f"(a sequence of {inputs.n_conditions} arrays), not a single "
            f"array of shape {reference_points.shape}. Reference points are "
            "compared against each condition's own true values, so one array "
            "reused across conditions measures something different on every "
            "condition while reporting a single averaged number. Pass "
            "`lambda inputs: ...` to derive them from "
            "`inputs.sim_batch`, or a list indexed by `inputs.cond_id`."
        )
    else:
        try:
            refs = reference_points[inputs.cond_id]
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"No reference points for condition {inputs.cond_id}: the "
                f"sequence must cover all {inputs.n_conditions} conditions."
            ) from exc
        except TypeError as exc:
            # A generator, most likely. Its own message ("not subscriptable")
            # would be recorded by the joint guard as the metric's reason
            # for failing, which sends the reader looking at the metric
            # rather than at what they passed.
            raise TypeError(
                "reference_points must be a callable or an indexable "
                "sequence of one array per condition; "
                f"{type(reference_points).__name__} cannot be indexed by "
                "condition. A generator is consumed once and cannot be "
                "replayed per condition -- pass a list."
            ) from exc

    refs = np.asarray(refs, dtype=float)
    expected = (inputs.draws.shape[0], inputs.draws.shape[2])
    if refs.shape != expected:
        raise ValueError(
            f"Reference points for condition {inputs.cond_id} have shape "
            f"{refs.shape}, expected {expected} "
            "(n_simulations, n_parameters)."
        )
    return refs


def make_tarp_joint_metric(
    reference_points: Any = None,
    *,
    reference: str = "uniform_box",
    resolution: int = 20,
    metric: str = "euclidean",
    standardize: bool = True,
    seed: int = 42,
    reference_id: str | None = None,
) -> JointMetricFn:
    """Create a TARP metric for the validation pipeline's joint dispatch.

    Which key it emits depends on the reference, and deliberately so:

    - ``reference_points=None`` draws them at random and emits
      **``tarp_error_random``**, registered as a *diagnostic*.
    - a provider emits **``tarp_error``**, registered as an *objective*.

    Two numbers that cannot be compared must not be comparable by name.
    Section 4.3 of Lemos et al. (2023) is explicit that a TARP run with an
    ``x``-independent reference point is blind to
    ``p_hat(theta|x) = p(theta)`` in the same way HPD coverage is, so the
    two keys are different statistics with different sensitivity -- one of
    which cannot see the failure the test is most worth running for. A
    single key carrying a mode field would be averaged across modes by
    something eventually, and a stored trial value would be uninterpretable
    six months later from `trials_to_dataframe()` alone.

    Parameters
    ----------
    reference_points
        ``None`` for the random-reference diagnostic. Otherwise either a
        ``Callable[[JointMetricInputs], np.ndarray]`` returning
        ``(n_sims, n_params)`` for that condition, or a sequence of such
        arrays indexed by ``cond_id``. A single array reused across
        conditions is rejected; see :func:`_references_for_condition`.
    reference
        Which reference distribution to draw from when *reference_points* is
        ``None``: ``"uniform_box"`` (default) or ``"prior_derangement"``.
        See :func:`compute_tarp_coverage`. Both are ``x``-independent, so
        both emit ``tarp_error_random`` -- the key is decided by whether a
        reference was *supplied*, not by which distribution was drawn from,
        because that is what decides whether the metric can see a posterior
        ignoring its data. The choice is recorded in the settings pin, so
        two studies drawing from different distributions read as different
        configurations rather than as comparable numbers.
    resolution
        Number of credibility levels the coverage curve is evaluated at.
    metric
        ``"euclidean"`` or ``"manhattan"``.
    standardize
        Standardize each parameter dimension before computing distances.
    seed
        Seed for the random reference draw. Unused when *reference_points*
        is given. Not defaulted to ``None``: with ``seed=None`` two
        identical calls return different numbers, which as an HPO objective
        means trials are not comparable.
    reference_id
        An optional label for *which* reference the provider produces,
        recorded in the study's joint metric settings so a resume detects a
        change of reference.

        Without it the pin records only ``reference_mode="provided"``,
        because a callable is not serializable: two studies both reporting
        ``tarp_error`` may have used entirely different providers and the
        settings check passes. Nothing can derive this automatically -- a
        provider's output depends on the data it is given -- so it is the
        caller's to supply and the caller's to change when the reference
        changes. Supplying it turns the pin from "a provider was used" into
        "this provider was used"; leaving it ``None`` keeps the weaker
        guarantee, stated rather than implied.

    Returns
    -------
    JointMetricFn
        Callable emitting one key for one condition.

    Warnings
    --------
    **A provider that samples the posterior it is auditing produces a
    reference that looks data-dependent and is not.** The provider receives
    the whole :class:`JointMetricInputs`, including ``approximator`` and
    ``draws``, because a real reference needs ``sim_batch``. Deriving the
    reference from the estimator under test reintroduces exactly the blind
    spot the two-key split exists to prevent, and *nothing can detect it* --
    the result is reported as ``tarp_error``, the key that claims teeth.
    Derive references from the data, never from the posterior.

    ``reference_mode`` is reported as ``"provided"``, never
    ``"data_dependent"``: supplying an array says nothing about how it was
    built.

    Notes
    -----
    Cheap enough to run per condition without a sub-sample: ~79 ms at 500
    simulations with 15 parameters, against ~54 s for L-C2ST on the same
    draws. Resolution is free -- 20 levels and 100 levels measured the same.
    See ``docs/plans/plan-joint-metric-path.md`` D9.

    References
    ----------
    Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023).
        Sampling-based accuracy testing of posterior estimators for general
        inference. *ICML 2023*. Algorithm 2; Section 4.3.
    """
    # Validated at construction for the same reason `make_lc2st_joint_metric`
    # validates its own: `compute_tarp_coverage` rejects these too, but only
    # once it is CALLED, so a typo becomes a per-condition exception that the
    # joint guard converts into the metric's registered worst case -- an
    # identical score on every trial, after paying for every trial's
    # training. Neither option depends on the data, so neither needs the
    # data to be checked.
    if metric not in _TARP_METRICS:
        raise ValueError(
            f"metric must be one of {sorted(_TARP_METRICS)}, got {metric!r}."
        )
    if resolution < 1:
        raise ValueError(f"resolution must be at least 1, got {resolution}.")
    if reference not in _TARP_REFERENCES:
        raise ValueError(
            f"reference must be one of {sorted(_TARP_REFERENCES)}, got "
            f"{reference!r}."
        )
    if reference_points is not None and reference != "uniform_box":
        raise ValueError(
            "reference_points and reference= both specify the reference, and "
            f"{reference!r} cannot apply to supplied points. Pass "
            "reference_points alone, or reference= alone to draw them here."
        )

    key = "tarp_error_random" if reference_points is None else "tarp_error"

    def _tarp_metric(inputs: JointMetricInputs) -> dict[str, float]:
        refs = (
            None
            if reference_points is None
            else _references_for_condition(reference_points, inputs)
        )
        result = compute_tarp_coverage(
            inputs.draws,
            inputs.true_values,
            reference_points=refs,
            reference=reference,
            resolution=resolution,
            metric=metric,
            standardize=standardize,
            # Per condition, so conditions do not share a reference draw.
            seed=seed + inputs.cond_id,
        )
        return {key: float(result["tarp_error"])}

    # Everything the score moves with, so a resumed study can tell that it
    # changed. `n_posterior_samples` is NOT listed: it is a property of the
    # validation run rather than of this metric, and the pipeline records it
    # on the result already. `reference_mode` records only that a provider
    # was supplied -- a callable is not serializable, so the pin can never
    # say WHICH one, and the two-key split is what mitigates that.
    _tarp_metric.joint_metric_settings = {  # type: ignore[attr-defined]
        "resolution": int(resolution),
        "metric": str(metric),
        "standardize": bool(standardize),
        "seed": int(seed),
        # NOT the same question as the `reference_mode` in the RESULT dict,
        # despite the shared name: this one is supplied-vs-drawn, that one is
        # which distribution was drawn from. A prior_derangement run pins
        # "random" here and reports "prior_derangement" there. The pin's
        # spelling is frozen by studies already on disk, so the collision is
        # documented in `compute_tarp_coverage` rather than renamed away;
        # `reference` below is the key that recovers the distribution.
        "reference_mode": "random" if reference_points is None else "provided",
        # None unless the caller labelled the reference. Recorded either
        # way, so a study that adds a label later reads as changed -- which
        # it is, in the only sense the pin can check.
        "reference_id": reference_id,
    }
    # Recorded only when it is not the default, and deliberately so.
    # `check_or_stamp_joint_metric_settings` compares the settings dicts for
    # EQUALITY, so an unconditional new key would make every study pinned
    # before this option existed raise on resume -- reporting a configuration
    # change to studies whose configuration did not change. Absence already
    # means "the default", which is what those pins meant when they were
    # written. A study that switches distributions still gets the signal,
    # because adding or removing the key changes the dict either way.
    if reference != "uniform_box":
        _tarp_metric.joint_metric_settings["reference"] = str(  # type: ignore[attr-defined]
            reference
        )
    return _tarp_metric


def _tarp_error_needs_a_reference(inputs: JointMetricInputs) -> dict[str, float]:
    """Placeholder for the registered ``tarp_error`` name.

    Never called: `resolve_joint_metrics` refuses it at resolve time. See
    :data:`REQUIRES_CONFIGURATION`.
    """
    raise ValueError(_TARP_ERROR_NEEDS_REFERENCE)  # pragma: no cover


_TARP_ERROR_NEEDS_REFERENCE = (
    "'tarp_error' requires reference points and cannot run at its "
    "registered default -- that is what distinguishes it from "
    "'tarp_error_random'. Build it with "
    "`make_tarp_joint_metric(reference_points=...)` and pass it to "
    "`run_validation_pipeline(joint_metrics={'tarp_error': fn})`, or use "
    "'tarp_error_random' if a random reference is what you want (a "
    "diagnostic: Lemos et al. 2023 Sec. 4.3 shows it cannot detect a "
    "posterior that ignores its data)."
)

setattr(
    _tarp_error_needs_a_reference,
    REQUIRES_CONFIGURATION,
    _TARP_ERROR_NEEDS_REFERENCE,
)


register_joint_metric(
    "tarp_error",
    _tarp_error_needs_a_reference,
    description=(
        "TARP expected-coverage error against SUPPLIED reference points "
        "(joint, data-dependent; requires configuration)"
    ),
    kind="objective",
    overwrite=True,
)

register_joint_metric(
    "tarp_error_random",
    make_tarp_joint_metric(),
    description=(
        "TARP expected-coverage error with random reference points "
        "(diagnostic: blind to a posterior that ignores its data)"
    ),
    kind="diagnostic",
    overwrite=True,
)

# Both are medians over credibility levels of |ECP - level|, and both terms
# lie in [0, 1], so the deviation is bounded by 1 and so is its median.
# Lower is better. Unlike `log_gamma` no infinite penalty is needed.
#
# 1.0 is a loose bound rather than a tight one: the attainable worst is 0.5,
# reached when every f_i collapses to one end so that ECP is 0 or 1 at every
# level. A penalty only has to dominate real values, and quoting the
# provable bound avoids a penalty that a pathological trial could beat.
register_metric_direction("tarp_error", higher_is_better=False, worst_raw=1.0)
register_metric_direction(
    "tarp_error_random", higher_is_better=False, worst_raw=1.0
)
