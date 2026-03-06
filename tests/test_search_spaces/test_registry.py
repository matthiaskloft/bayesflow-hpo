"""Tests for search-space registry lookup."""

import pytest

from bayesflow_hpo.search_spaces.inference import (
    ConsistencyModelSpace,
    CouplingFlowSpace,
    DiffusionModelSpace,
    FlowMatchingSpace,
    StableConsistencyModelSpace,
)
from bayesflow_hpo.search_spaces.registry import (
    get_inference_space,
    get_summary_space,
    list_inference_spaces,
    register_inference_space,
)
from bayesflow_hpo.search_spaces.summary import (
    DeepSetSpace,
    FusionTransformerSpace,
    SetTransformerSpace,
    TimeSeriesNetworkSpace,
    TimeSeriesTransformerSpace,
)


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("cf", CouplingFlowSpace),
        ("flow_matching", FlowMatchingSpace),
        ("dm", DiffusionModelSpace),
        ("consistency", ConsistencyModelSpace),
        ("scm", StableConsistencyModelSpace),
    ],
)
def test_get_inference_space_aliases(name, expected_type):
    assert isinstance(get_inference_space(name), expected_type)


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("ds", DeepSetSpace),
        ("st", SetTransformerSpace),
        ("tsn", TimeSeriesNetworkSpace),
        ("tst", TimeSeriesTransformerSpace),
        ("ft", FusionTransformerSpace),
    ],
)
def test_get_summary_space_aliases(name, expected_type):
    assert isinstance(get_summary_space(name), expected_type)


def test_registry_raises_key_error_on_unknown_space():
    with pytest.raises(KeyError):
        get_inference_space("unknown")

    with pytest.raises(KeyError):
        get_summary_space("unknown")


def test_register_custom_inference_space_with_alias():
    register_inference_space(
        name="custom_test_cf",
        factory=CouplingFlowSpace,
        aliases=["ctcf"],
        overwrite=True,
    )
    assert isinstance(get_inference_space("custom_test_cf"), CouplingFlowSpace)
    assert isinstance(get_inference_space("ctcf"), CouplingFlowSpace)
    assert "custom_test_cf" in list_inference_spaces()
