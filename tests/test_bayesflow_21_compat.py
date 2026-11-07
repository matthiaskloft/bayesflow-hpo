"""Guard against BayesFlow silently dropping keyword arguments.

BayesFlow 2.1 routes unknown ``**kwargs`` of its networks through
``layer_kwargs`` (``bayesflow/utils/dict_utils.py``), which keeps only
``dtype``, ``name`` and ``trainable`` and discards the rest without error.
A search space that still passes a 2.0.x-only argument therefore builds
fine and tunes nothing. These tests record every keyword the spaces pass
and check it against the real constructor signature.
"""

from __future__ import annotations

import inspect

import bayesflow as bf
import pytest
from conftest import FakeTrial

from bayesflow_hpo.search_spaces.inference.consistency import ConsistencyModelSpace
from bayesflow_hpo.search_spaces.inference.coupling_flow import CouplingFlowSpace
from bayesflow_hpo.search_spaces.inference.diffusion import DiffusionModelSpace
from bayesflow_hpo.search_spaces.inference.flow_matching import FlowMatchingSpace
from bayesflow_hpo.search_spaces.inference.stable_consistency import (
    StableConsistencyModelSpace,
)
from bayesflow_hpo.search_spaces.summary.deep_set import DeepSetSpace
from bayesflow_hpo.search_spaces.summary.fusion_transformer import (
    FusionTransformerSpace,
)
from bayesflow_hpo.search_spaces.summary.set_transformer import SetTransformerSpace
from bayesflow_hpo.search_spaces.summary.time_series_network import (
    TimeSeriesNetworkSpace,
)
from bayesflow_hpo.search_spaces.summary.time_series_transformer import (
    TimeSeriesTransformerSpace,
)


class _StopBuildError(Exception):
    pass


def _explicit_params(cls: type) -> set[str]:
    return {
        name
        for name, p in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    }


@pytest.mark.parametrize(
    ("space_cls", "network_cls"),
    [
        (DeepSetSpace, bf.networks.DeepSet),
        (SetTransformerSpace, bf.networks.SetTransformer),
        (TimeSeriesTransformerSpace, bf.networks.TimeSeriesTransformer),
        (FusionTransformerSpace, bf.networks.FusionTransformer),
        (CouplingFlowSpace, bf.networks.CouplingFlow),
        (FlowMatchingSpace, bf.networks.FlowMatching),
        (DiffusionModelSpace, bf.networks.DiffusionModel),
        (ConsistencyModelSpace, bf.networks.ConsistencyModel),
        (StableConsistencyModelSpace, bf.networks.StableConsistencyModel),
        (TimeSeriesNetworkSpace, bf.networks.TimeSeriesNetwork),
    ],
)
def test_space_passes_only_arguments_the_network_accepts(
    space_cls, network_cls, monkeypatch
):
    captured: dict = {}

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)
        raise _StopBuildError

    accepted = _explicit_params(network_cls)  # before patching
    monkeypatch.setattr(network_cls, "__init__", fake_init)

    space = space_cls()
    params = space.sample(FakeTrial())
    with pytest.raises(_StopBuildError):
        space.build(params)

    assert set(captured) <= accepted, sorted(set(captured) - accepted)


@pytest.mark.parametrize(
    ("space_cls", "subnet_cls_name", "key"),
    [
        (FlowMatchingSpace, "TimeMLP", "subnet_kwargs"),
        (CouplingFlowSpace, "MLP", "subnet_kwargs"),
        (DiffusionModelSpace, "TimeMLP", "subnet_kwargs"),
        (ConsistencyModelSpace, "TimeMLP", "subnet_kwargs"),
        (StableConsistencyModelSpace, "TimeMLP", "subnet_kwargs"),
    ],
)
def test_subnet_kwargs_are_accepted_by_the_subnet(
    space_cls, subnet_cls_name, key, monkeypatch
):
    captured: dict = {}
    network_cls = {
        FlowMatchingSpace: bf.networks.FlowMatching,
        CouplingFlowSpace: bf.networks.CouplingFlow,
        DiffusionModelSpace: bf.networks.DiffusionModel,
        ConsistencyModelSpace: bf.networks.ConsistencyModel,
        StableConsistencyModelSpace: bf.networks.StableConsistencyModel,
    }[space_cls]

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs[key])
        raise _StopBuildError

    accepted = _explicit_params(getattr(bf.networks, subnet_cls_name))
    monkeypatch.setattr(network_cls, "__init__", fake_init)

    space = space_cls()
    with pytest.raises(_StopBuildError):
        space.build(space.sample(FakeTrial()))

    assert set(captured) <= accepted, sorted(set(captured) - accepted)


@pytest.mark.parametrize(
    "space_cls",
    [
        DeepSetSpace,
        SetTransformerSpace,
        TimeSeriesTransformerSpace,
        FusionTransformerSpace,
        CouplingFlowSpace,
        FlowMatchingSpace,
        DiffusionModelSpace,
        ConsistencyModelSpace,
        StableConsistencyModelSpace,
        TimeSeriesNetworkSpace,
    ],
)
def test_default_space_builds_a_real_network(space_cls):
    space = space_cls()
    network = space.build(space.sample(FakeTrial()))
    assert network is not None
