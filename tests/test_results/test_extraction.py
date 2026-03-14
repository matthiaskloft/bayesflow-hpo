"""Tests for bayesflow_hpo.results.extraction."""


import optuna

from bayesflow_hpo.results.extraction import (
    _fmt_param_count,
    _objective_column_names,
    get_pareto_trials,
    summarize_study,
    trials_to_dataframe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_study(n_trials=3, n_objectives=2, metric_names=None):
    """Create a real in-memory Optuna study with completed trials."""
    directions = ["minimize"] * n_objectives
    study = optuna.create_study(
        directions=directions,
        study_name="test",
    )
    if metric_names is not None:
        study.set_metric_names(metric_names)

    for i in range(n_trials):
        trial = optuna.trial.create_trial(
            params={"lr": 0.001 * (i + 1), "depth": i + 2},
            distributions={
                "lr": optuna.distributions.FloatDistribution(0.0001, 0.01),
                "depth": optuna.distributions.IntDistribution(1, 10),
            },
            values=[0.1 * (i + 1)] * n_objectives,
            state=optuna.trial.TrialState.COMPLETE,
        )
        trial.set_user_attr("param_count", 10000 * (i + 1))
        trial.set_user_attr("calibration_error", 0.05 * (i + 1))
        trial.set_user_attr("nrmse", 0.1 * (i + 1))
        study.add_trial(trial)

    return study


def _make_study_with_rejected(metric_names=None):
    """Study with one trained trial and one budget-rejected trial."""
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name="test_rejected",
    )
    if metric_names is not None:
        study.set_metric_names(metric_names)

    # Trained trial
    trained = optuna.trial.create_trial(
        params={"lr": 0.001},
        distributions={"lr": optuna.distributions.FloatDistribution(0.0001, 0.01)},
        values=[0.1, 0.5],
        state=optuna.trial.TrialState.COMPLETE,
    )
    trained.set_user_attr("param_count", 50000)
    study.add_trial(trained)

    # Rejected trial
    rejected = optuna.trial.create_trial(
        params={"lr": 0.005},
        distributions={"lr": optuna.distributions.FloatDistribution(0.0001, 0.01)},
        values=[1.0, 1.5],
        state=optuna.trial.TrialState.COMPLETE,
    )
    rejected.set_user_attr("rejected_reason", "param_count_exceeded")
    study.add_trial(rejected)

    return study


# ---------------------------------------------------------------------------
# _fmt_param_count
# ---------------------------------------------------------------------------

class TestFmtParamCount:
    def test_millions(self):
        assert _fmt_param_count(1_500_000) == "1.50M"

    def test_thousands(self):
        assert _fmt_param_count(50_000) == "50.0K"

    def test_small(self):
        assert _fmt_param_count(500) == "500"

    def test_exact_million(self):
        assert _fmt_param_count(1_000_000) == "1.00M"

    def test_exact_thousand(self):
        assert _fmt_param_count(1_000) == "1.0K"


# ---------------------------------------------------------------------------
# _objective_column_names
# ---------------------------------------------------------------------------

class TestObjectiveColumnNames:
    def test_with_metric_names(self):
        study = _make_study(metric_names=["cal_error", "cost"])
        assert _objective_column_names(study) == ["cal_error", "cost"]

    def test_without_metric_names_multi(self):
        study = _make_study(n_objectives=3)
        names = _objective_column_names(study)
        assert names == ["objective_0", "objective_1", "objective_2"]

    def test_without_metric_names_single(self):
        study = _make_study(n_objectives=1)
        assert _objective_column_names(study) == ["objective"]


# ---------------------------------------------------------------------------
# trials_to_dataframe
# ---------------------------------------------------------------------------

class TestTrialsToDataframe:
    def test_basic(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = trials_to_dataframe(study)
        assert len(df) == 3
        assert "trial_number" in df.columns
        assert "cal" in df.columns
        assert "cost" in df.columns

    def test_trained_only_excludes_rejected(self):
        study = _make_study_with_rejected(metric_names=["m1", "m2"])
        df = trials_to_dataframe(study, trained_only=True)
        assert len(df) == 1

    def test_trained_only_false_includes_all(self):
        study = _make_study_with_rejected(metric_names=["m1", "m2"])
        df = trials_to_dataframe(study, trained_only=False)
        assert len(df) == 2

    def test_includes_user_attrs(self):
        study = _make_study(n_trials=1, metric_names=["m1", "m2"])
        df = trials_to_dataframe(study)
        assert "param_count" in df.columns

    def test_extra_attrs(self):
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            study_name="test_extra",
        )
        trial = optuna.trial.create_trial(
            params={"lr": 0.001},
            distributions={"lr": optuna.distributions.FloatDistribution(0.0001, 0.01)},
            values=[0.1, 0.2],
            state=optuna.trial.TrialState.COMPLETE,
        )
        trial.set_user_attr("custom_val", 42)
        study.add_trial(trial)
        df = trials_to_dataframe(study, extra_attrs=["custom_val"])
        assert "custom_val" in df.columns


# ---------------------------------------------------------------------------
# get_pareto_trials
# ---------------------------------------------------------------------------

class TestGetParetoTrials:
    def test_returns_list(self):
        study = _make_study(n_trials=5)
        pareto = get_pareto_trials(study)
        assert isinstance(pareto, list)
        assert len(pareto) > 0


# ---------------------------------------------------------------------------
# summarize_study
# ---------------------------------------------------------------------------

class TestSummarizeStudy:
    def test_returns_string(self):
        study = _make_study(n_trials=3, metric_names=["cal_error", "cost"])
        result = summarize_study(study)
        assert isinstance(result, str)
        assert "test" in result  # study name

    def test_contains_trial_counts(self):
        study = _make_study(n_trials=3, metric_names=["cal_error", "cost"])
        result = summarize_study(study)
        assert "3 trained" in result

    def test_single_objective(self):
        study = _make_study(n_trials=2, n_objectives=1)
        result = summarize_study(study)
        assert "Best trial" in result
