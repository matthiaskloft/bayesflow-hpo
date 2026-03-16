"""Tests for bayesflow_hpo.results.extraction."""


import math

import optuna
import pytest

from bayesflow_hpo.results.extraction import (
    _display_col_name,
    _fmt_param_count,
    _objective_column_names,
    _round_value,
    best_config,
    compare_trials,
    get_pareto_trials,
    summarize_study,
    trial_table,
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
# _display_col_name
# ---------------------------------------------------------------------------

class TestDisplayColName:
    def test_time_metric_gets_suffix(self):
        assert _display_col_name("training_time_s") == "training_time_s (s)"

    def test_inference_time_gets_suffix(self):
        assert _display_col_name("inference_time_s") == "inference_time_s (s)"

    def test_non_time_metric_unchanged(self):
        assert _display_col_name("calibration_error") == "calibration_error"

    def test_already_has_suffix(self):
        assert _display_col_name("time (s)") == "time (s)"


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

    def test_contains_hint(self):
        study = _make_study(n_trials=2, metric_names=["cal", "cost"])
        result = summarize_study(study)
        assert "trial_table()" in result
        assert "best_config()" in result

    def test_no_leaderboard(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        result = summarize_study(study)
        assert "Top " not in result

    def test_no_hyperparameters(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        result = summarize_study(study)
        assert "Hyperparameters" not in result

    def test_time_metrics_show_unit(self):
        study = _make_study(
            n_trials=2, metric_names=["inference_time_s", "cost"],
        )
        result = summarize_study(study)
        assert "(s)" in result


# ---------------------------------------------------------------------------
# _round_value
# ---------------------------------------------------------------------------

class TestRoundValue:
    def test_learning_rate_scientific(self):
        assert _round_value("initial_lr", 0.000123456) == "1.23e-04"

    def test_lr_key(self):
        assert _round_value("lr", 0.005) == "5.00e-03"

    def test_learning_rate_key(self):
        assert _round_value("learning_rate", 0.01) == "1.00e-02"

    def test_dropout_two_decimals(self):
        assert _round_value("ds_dropout", 0.05454749) == 0.05

    def test_dim_to_int(self):
        assert _round_value("ds_summary_dim", 63.0) == 63
        assert isinstance(_round_value("ds_summary_dim", 63.0), int)

    def test_width_to_int(self):
        assert _round_value("fm_width", 128.0) == 128

    def test_depth_to_int(self):
        assert _round_value("fm_subnet_depth", 3.0) == 3

    def test_time_one_decimal(self):
        assert _round_value("training_time_s", 42.567) == 42.6

    def test_generic_float_four_decimals(self):
        assert _round_value("calibration_error", 0.123456789) == 0.1235

    def test_int_passthrough(self):
        assert _round_value("depth", 5) == 5
        assert isinstance(_round_value("depth", 5), int)

    def test_string_passthrough(self):
        assert _round_value("network_type", "flow_matching") == "flow_matching"

    def test_bool_passthrough(self):
        assert _round_value("use_bias", True) is True

    def test_none_passthrough(self):
        assert _round_value("something", None) is None

    def test_nan_passthrough(self):
        result = _round_value("metric", float("nan"))
        assert math.isnan(result)

    def test_inf_passthrough(self):
        result = _round_value("metric", float("inf"))
        assert math.isinf(result)


# ---------------------------------------------------------------------------
# trial_table
# ---------------------------------------------------------------------------

class TestTrialTable:
    def test_basic_ranking(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = trial_table(study)
        assert len(df) == 3
        assert "rank" in df.columns
        assert "trial" in df.columns
        assert list(df["rank"]) == [1, 2, 3]

    def test_sorted_by_objective(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = trial_table(study)
        # First trial should have lowest objective value.
        assert df.iloc[0]["cal"] <= df.iloc[1]["cal"]

    def test_top_k_filtering(self):
        study = _make_study(n_trials=5, metric_names=["cal", "cost"])
        df = trial_table(study, top_k=2)
        assert len(df) == 2

    def test_top_k_larger_than_n_trials(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = trial_table(study, top_k=10)
        assert len(df) == 3

    def test_metrics_inclusion(self):
        study = _make_study(n_trials=2, metric_names=["cal", "cost"])
        df = trial_table(study, metrics=["nrmse", "calibration_error"])
        assert "nrmse" in df.columns
        assert "calibration_error" in df.columns

    def test_single_objective(self):
        study = _make_study(n_trials=2, n_objectives=1)
        df = trial_table(study)
        assert len(df) == 2
        assert "objective" in df.columns

    def test_empty_study(self):
        study = optuna.create_study(
            directions=["minimize"],
            study_name="empty",
        )
        df = trial_table(study)
        assert len(df) == 0

    def test_rejected_only_study(self):
        study = _make_study_with_rejected(metric_names=["m1", "m2"])
        df = trial_table(study, trained_only=True)
        assert len(df) == 1  # Only the trained trial

    def test_csv_round_trip(self, tmp_path):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = trial_table(study)
        csv_path = tmp_path / "trials.csv"
        df.to_csv(csv_path, index=False)
        reloaded = __import__("pandas").read_csv(csv_path)
        assert len(reloaded) == len(df)
        assert list(reloaded.columns) == list(df.columns)

    def test_param_count_formatted(self):
        study = _make_study(n_trials=1, metric_names=["cal", "cost"])
        df = trial_table(study)
        assert "param_count" in df.columns
        # param_count is 10000, should be "10.0K"
        assert df.iloc[0]["param_count"] == "10.0K"

    def test_select_by_second_objective(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = trial_table(study, select_by=1)
        # Still sorted (ascending) by second objective.
        assert list(df["rank"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# best_config
# ---------------------------------------------------------------------------

class TestBestConfig:
    def test_by_objective(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        config = best_config(study)
        assert isinstance(config, dict)
        assert "lr" in config

    def test_by_trial_number(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        config = best_config(study, trial_number=1)
        assert isinstance(config, dict)

    def test_nonexistent_trial(self):
        study = _make_study(n_trials=2, metric_names=["cal", "cost"])
        with pytest.raises(ValueError, match="not found"):
            best_config(study, trial_number=999)

    def test_single_trial_study(self):
        study = _make_study(n_trials=1, n_objectives=1)
        config = best_config(study)
        assert "lr" in config
        assert "depth" in config

    def test_empty_study(self):
        study = optuna.create_study(
            directions=["minimize"],
            study_name="empty",
        )
        with pytest.raises(ValueError, match="no trained trials"):
            best_config(study)

    def test_values_are_rounded(self):
        study = _make_study(n_trials=1, metric_names=["cal", "cost"])
        config = best_config(study)
        # lr should be scientific notation string.
        assert isinstance(config["lr"], str)
        assert "e" in config["lr"]


# ---------------------------------------------------------------------------
# compare_trials
# ---------------------------------------------------------------------------

class TestCompareTrials:
    def test_basic_two_trials(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = compare_trials(study, trial_numbers=[0, 1])
        assert "trial_0" in df.columns
        assert "trial_1" in df.columns
        assert "lr" in df.index

    def test_three_trials(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = compare_trials(study, trial_numbers=[0, 1, 2])
        assert len(df.columns) == 3

    def test_too_few_trials(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        with pytest.raises(ValueError, match="at least 2"):
            compare_trials(study, trial_numbers=[0])

    def test_too_many_trials(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        with pytest.raises(ValueError, match="At most 5"):
            compare_trials(study, trial_numbers=[0, 0, 0, 0, 0, 0])

    def test_nonexistent_trial(self):
        study = _make_study(n_trials=2, metric_names=["cal", "cost"])
        with pytest.raises(ValueError, match="not found"):
            compare_trials(study, trial_numbers=[0, 999])

    def test_duplicate_trial_numbers(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        # Duplicates produce a single column (dict key dedup).
        df = compare_trials(study, trial_numbers=[0, 0])
        assert "trial_0" in df.columns

    def test_includes_objectives(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = compare_trials(study, trial_numbers=[0, 1])
        assert "cal" in df.index
        assert "cost" in df.index

    def test_includes_param_count(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = compare_trials(study, trial_numbers=[0, 1])
        assert "param_count" in df.index

    def test_with_metrics(self):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = compare_trials(study, trial_numbers=[0, 1], metrics=["nrmse"])
        assert "nrmse" in df.index

    def test_csv_exportable(self, tmp_path):
        study = _make_study(n_trials=3, metric_names=["cal", "cost"])
        df = compare_trials(study, trial_numbers=[0, 1])
        csv_path = tmp_path / "compare.csv"
        df.to_csv(csv_path)
        reloaded = __import__("pandas").read_csv(csv_path, index_col=0)
        assert list(reloaded.columns) == list(df.columns)
