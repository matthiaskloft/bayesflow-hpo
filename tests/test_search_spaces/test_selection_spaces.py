"""Tests for network/summary selection search spaces."""

import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.composite import (
    NetworkSelectionSpace,
    SummarySelectionSpace,
)
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace


def test_network_selection_space_marks_selected_type():
    space = NetworkSelectionSpace(candidates={"cf": CouplingFlowSpace()})
    params = space.sample(FakeTrial())
    assert params["_inference_network_type"] == "cf"


def test_summary_selection_space_marks_selected_type():
    space = SummarySelectionSpace(candidates={"ds": DeepSetSpace()})
    params = space.sample(FakeTrial())
    assert params["_summary_network_type"] == "ds"


def test_network_selection_space_requires_marker_key_on_build():
    space = NetworkSelectionSpace(candidates={"cf": CouplingFlowSpace()})
    with pytest.raises(ValueError, match="_inference_network_type"):
        space.build({})


def test_summary_selection_space_requires_marker_key_on_build():
    space = SummarySelectionSpace(candidates={"ds": DeepSetSpace()})
    with pytest.raises(ValueError, match="_summary_network_type"):
        space.build({})


def test_empty_candidate_spaces_raise_clear_error():
    with pytest.raises(ValueError, match="at least one candidate"):
        NetworkSelectionSpace(candidates={}).sample(FakeTrial())

    with pytest.raises(ValueError, match="at least one candidate"):
        SummarySelectionSpace(candidates={}).sample(FakeTrial())
