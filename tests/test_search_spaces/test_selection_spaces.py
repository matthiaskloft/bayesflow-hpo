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
    with pytest.raises(ValueError, match="inference_network_type"):
        space.build({})


def test_summary_selection_space_requires_marker_key_on_build():
    space = SummarySelectionSpace(candidates={"ds": DeepSetSpace()})
    with pytest.raises(ValueError, match="summary_network_type"):
        space.build({})


def test_network_selection_build_accepts_plain_key():
    """build() works with 'inference_network_type' (no underscore prefix)."""
    space = NetworkSelectionSpace(candidates={"cf": CouplingFlowSpace()})
    params = space.sample(FakeTrial())
    # Simulate what best_config() returns: plain key, no underscore prefix.
    plain_params = {k: v for k, v in params.items() if k != "_inference_network_type"}
    plain_params["inference_network_type"] = params["_inference_network_type"]
    net = space.build(plain_params)
    assert net is not None


def test_summary_selection_build_accepts_plain_key():
    """build() works with 'summary_network_type' (no underscore prefix)."""
    space = SummarySelectionSpace(candidates={"ds": DeepSetSpace()})
    params = space.sample(FakeTrial())
    plain_params = {k: v for k, v in params.items() if k != "_summary_network_type"}
    plain_params["summary_network_type"] = params["_summary_network_type"]
    net = space.build(plain_params)
    assert net is not None


def test_empty_candidate_spaces_raise_clear_error():
    with pytest.raises(ValueError, match="at least one candidate"):
        NetworkSelectionSpace(candidates={}).sample(FakeTrial())

    with pytest.raises(ValueError, match="at least one candidate"):
        SummarySelectionSpace(candidates={}).sample(FakeTrial())
