"""Tests for bayesflow_hpo.results.extraction."""


import math

import optuna
import pytest

from bayesflow_hpo.results.extraction import (
    SelectionResult,
    _display_col_name,
    _fmt_param_count,
    _objective_column_names,
    _round_value,
    _validate_select_by,
    best_config,
    compare_trials,
    get_pareto_trials,
    select_best_trial,
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

    def test_includes_rank_columns_multi_objective(self):
        study = _make_study(n_trials=3, metric_names=["m1", "m2"])
        df = trials_to_dataframe(study)
        assert "rank" in df.columns
        assert "rank_m1" in df.columns
        assert "rank_m2" in df.columns
        assert list(df["rank_m1"]) == [1, 2, 3]

    def test_includes_rank_column_single_objective(self):
        study = _make_study(n_trials=3, n_objectives=1)
        df = trials_to_dataframe(study)
        assert "rank" in df.columns
        assert list(df["rank"]) == [1, 2, 3]

    def test_can_disable_rank_columns(self):
        study = _make_study(n_trials=3, metric_names=["m1", "m2"])
        df = trials_to_dataframe(study, include_ranks=False)
        assert "rank" not in df.columns
        assert "rank_m1" not in df.columns
        assert "rank_m2" not in df.columns


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


# ---------------------------------------------------------------------------
# _validate_select_by
# ---------------------------------------------------------------------------

class TestValidateSelectBy:
    def test_valid_index_zero(self):
        study = _make_study(n_objectives=2)
        _validate_select_by(study, 0)  # should not raise

    def test_valid_index_one(self):
        study = _make_study(n_objectives=2)
        _validate_select_by(study, 1)  # should not raise

    def test_negative_index_raises(self):
        study = _make_study(n_objectives=2)
        with pytest.raises(ValueError, match="select_by=-1"):
            _validate_select_by(study, -1)

    def test_too_large_index_raises(self):
        study = _make_study(n_objectives=2)
        with pytest.raises(ValueError, match="select_by=2"):
            _validate_select_by(study, 2)

    def test_single_objective_only_zero_valid(self):
        study = _make_study(n_objectives=1)
        _validate_select_by(study, 0)
        with pytest.raises(ValueError, match="select_by=1"):
            _validate_select_by(study, 1)

    def test_trial_table_rejects_invalid(self):
        study = _make_study(n_objectives=2)
        with pytest.raises(ValueError, match="select_by=5"):
            trial_table(study, select_by=5)

    def test_best_config_rejects_invalid(self):
        study = _make_study(n_objectives=2)
        with pytest.raises(ValueError, match="select_by=3"):
            best_config(study, select_by=3)

    def test_summarize_study_rejects_invalid(self):
        study = _make_study(n_objectives=2)
        with pytest.raises(ValueError, match="select_by=-1"):
            summarize_study(study, select_by=-1)


# ---------------------------------------------------------------------------
# select_best_trial
# ---------------------------------------------------------------------------

def _make_selection_study():
    """Create a 3-objective study with 5 diverse trials for selection tests.

    Objectives: calibration_error (min), nrmse (min), inference_time (min).
    Trial values and user_attrs are set to exercise satisficing + Pareto logic.
    """
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        study_name="selection_test",
    )
    study.set_metric_names(["calibration_error", "nrmse", "inference_time"])

    specs = [
        # trial 0: good cal, good nrmse, slow
        {"values": [0.005, 0.03, 5.0], "coverage_90": 0.92},
        # trial 1: good cal, bad nrmse, fast
        {"values": [0.008, 0.10, 1.0], "coverage_90": 0.88},
        # trial 2: bad cal, good nrmse, fast
        {"values": [0.05, 0.02, 1.5], "coverage_90": 0.95},
        # trial 3: ok cal, ok nrmse, medium
        {"values": [0.009, 0.04, 3.0], "coverage_90": 0.90},
        # trial 4: good cal, good nrmse, medium
        {"values": [0.007, 0.04, 2.5], "coverage_90": 0.91},
    ]
    for i, spec in enumerate(specs):
        trial = optuna.trial.create_trial(
            params={"lr": 0.001 * (i + 1)},
            distributions={
                "lr": optuna.distributions.FloatDistribution(0.0001, 0.01),
            },
            values=spec["values"],
            state=optuna.trial.TrialState.COMPLETE,
        )
        trial.set_user_attr("coverage_90", spec["coverage_90"])
        study.add_trial(trial)

    return study


class TestSelectBestTrial:
    def test_basic_satisficing_and_pareto(self):
        """Priorities on cal and nrmse, Pareto over inference_time."""
        study = _make_selection_study()
        trial, result = select_best_trial(
            study,
            priorities=[
                ("calibration_error", 0.01),
                ("nrmse", 0.05),
            ],
        )
        # Trials meeting cal <= 0.01: 0, 1, 3, 4
        # Of those, nrmse <= 0.05: 0, 3, 4
        # Pareto over inference_time among {0, 3, 4}: trial 4 (2.5s) dominates
        # trial 3 (3.0s) and trial 0 (5.0s) on the single remaining objective
        assert trial.number == 4
        assert result.thresholds_met == {
            "calibration_error": True,
            "nrmse": True,
        }
        assert len(result.n_candidates_per_step) == 3  # initial + 2 steps

    def test_direction_inference_minimize(self):
        """Direction inferred from study.directions (minimize → below)."""
        study = _make_selection_study()
        trial, result = select_best_trial(
            study,
            priorities=[("calibration_error", 0.01)],
        )
        # cal <= 0.01: trials 0, 1, 3, 4
        # Pareto over nrmse + inference_time: trial 0 is good on nrmse, trial 1 fast
        assert result.thresholds_met["calibration_error"] is True
        assert trial.number in {0, 1, 3, 4}  # must be from filtered set

    def test_direction_inference_maximize(self):
        """Direction inferred from study.directions (maximize → above)."""
        study = optuna.create_study(
            directions=["maximize"],
            study_name="max_test",
        )
        study.set_metric_names(["accuracy"])
        for val in [0.85, 0.90, 0.95]:
            t = optuna.trial.create_trial(
                params={"lr": 0.001},
                distributions={
                    "lr": optuna.distributions.FloatDistribution(0.0001, 0.01),
                },
                values=[val],
                state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(t)

        trial, result = select_best_trial(
            study,
            priorities=[("accuracy", 0.88)],
        )
        # accuracy >= 0.88: trials with 0.90 and 0.95
        # Phase 2 over remaining (none) → mean rank across all → best is 0.95
        assert result.thresholds_met["accuracy"] is True
        assert trial.values[0] == 0.95

    def test_explicit_direction_for_user_attr(self):
        """User_attr requires explicit direction."""
        study = _make_selection_study()
        trial, result = select_best_trial(
            study,
            priorities=[
                ("calibration_error", 0.01),
                ("coverage_90", 0.90, "above"),
            ],
        )
        # cal <= 0.01: trials 0, 1, 3, 4
        # coverage_90 >= 0.90: trials 0 (0.92), 3 (0.90), 4 (0.91)
        assert result.thresholds_met["coverage_90"] is True

    def test_user_attr_without_direction_raises(self):
        study = _make_selection_study()
        with pytest.raises(ValueError, match="not a study objective"):
            select_best_trial(
                study,
                priorities=[("coverage_90", 0.90)],
            )

    def test_metric_not_found_raises(self):
        study = _make_selection_study()
        with pytest.raises(ValueError, match="not found"):
            select_best_trial(
                study,
                priorities=[("nonexistent_metric", 0.5)],
            )

    def test_empty_priorities_raises(self):
        """Empty priorities list should raise (caller should use select_by)."""
        study = _make_selection_study()
        # Empty list means no satisficing steps and no direction to infer.
        # The function should still work: skip Phase 1, go to Phase 2.
        trial, result = select_best_trial(study, priorities=[])
        # All 5 trials survive, Pareto over all 3 objectives
        assert result.n_candidates_per_step == [5]
        assert trial is not None

    def test_all_thresholds_met_no_remaining_objectives(self):
        """All study objectives have thresholds and are met."""
        study = _make_selection_study()
        trial, result = select_best_trial(
            study,
            priorities=[
                ("calibration_error", 0.01),
                ("nrmse", 0.05),
                ("inference_time", 10.0),
            ],
        )
        # All thresholds met, no remaining objectives → mean rank across all
        assert all(result.thresholds_met.values())
        assert trial is not None

    def test_no_trial_meets_first_threshold(self):
        """Threshold too strict — all promoted to Phase 2."""
        study = _make_selection_study()
        trial, result = select_best_trial(
            study,
            priorities=[
                ("calibration_error", 0.001),  # too strict
                ("nrmse", 0.05),
            ],
        )
        assert result.thresholds_met["calibration_error"] is False
        assert result.thresholds_met["nrmse"] is False
        # All 5 trials go to Phase 2
        assert result.n_candidates_per_step == [5]

    def test_single_trial_study(self):
        study = optuna.create_study(
            directions=["minimize"],
            study_name="single",
        )
        study.set_metric_names(["cal"])
        t = optuna.trial.create_trial(
            params={"lr": 0.001},
            distributions={
                "lr": optuna.distributions.FloatDistribution(0.0001, 0.01),
            },
            values=[0.05],
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(t)

        trial, result = select_best_trial(
            study,
            priorities=[("cal", 0.01)],
        )
        assert trial.number == 0
        assert result.thresholds_met["cal"] is False

    def test_single_remaining_objective(self):
        """Only one objective remains after satisficing → simple sort."""
        study = _make_selection_study()
        trial, result = select_best_trial(
            study,
            priorities=[
                ("calibration_error", 0.01),
                ("nrmse", 0.05),
            ],
        )
        # Remaining: inference_time only → trial 4 (2.5s) is best
        assert trial.number == 4

    def test_selection_result_fields(self):
        study = _make_selection_study()
        trial, result = select_best_trial(
            study,
            priorities=[("calibration_error", 0.01)],
        )
        assert isinstance(result, SelectionResult)
        assert isinstance(result.thresholds_met, dict)
        assert isinstance(result.pareto_front, list)
        assert isinstance(result.n_candidates_per_step, list)
        assert all(isinstance(t, optuna.trial.FrozenTrial) for t in result.pareto_front)

    def test_best_config_with_priorities(self):
        study = _make_selection_study()
        config = best_config(
            study,
            priorities=[
                ("calibration_error", 0.01),
                ("nrmse", 0.05),
            ],
        )
        assert isinstance(config, dict)
        assert "lr" in config

    def test_best_config_mutual_exclusivity(self):
        study = _make_selection_study()
        with pytest.raises(ValueError, match="Cannot specify both"):
            best_config(
                study,
                select_by=1,
                priorities=[("calibration_error", 0.01)],
            )

    def test_invalid_direction_raises(self):
        study = _make_selection_study()
        with pytest.raises(ValueError, match="direction must be"):
            select_best_trial(
                study,
                priorities=[("calibration_error", 0.01, "sideways")],
            )

    def test_no_trained_trials_raises(self):
        study = optuna.create_study(directions=["minimize"], study_name="empty")
        with pytest.raises(ValueError, match="no trained trials"):
            select_best_trial(study, priorities=[("obj", 0.5)])
