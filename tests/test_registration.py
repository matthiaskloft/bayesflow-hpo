"""Tests for high-level custom registration helpers."""

from dataclasses import dataclass, field

from bayesflow_hpo import (
    list_registered_network_spaces,
    register_custom_summary_network,
)
from bayesflow_hpo.builders.registry import SUMMARY_BUILDERS
from bayesflow_hpo.registration import _register_with_aliases
from bayesflow_hpo.search_spaces.base import BaseSearchSpace, IntDimension
from bayesflow_hpo.search_spaces.registry import get_summary_space


@dataclass
class _DummySummarySpace(BaseSearchSpace):
    width: IntDimension = field(
        default_factory=lambda: IntDimension("dummy_width", 8, 16, step=8)
    )

    def build(self, params: dict):
        return {"dummy": params["dummy_width"]}


def test_register_custom_summary_network_space():
    register_custom_summary_network(
        name="dummy_summary",
        space_factory=_DummySummarySpace,
        aliases=["dummy"],
        overwrite=True,
    )

    summary_space = get_summary_space("dummy")
    assert isinstance(summary_space, _DummySummarySpace)
    assert "dummy_summary" in list_registered_network_spaces()["summary"]


def test_register_with_aliases_registers_all():
    """_register_with_aliases registers under name and all aliases."""
    registry = {}

    def fake_register(name, builder, overwrite):
        registry[name] = builder

    sentinel = lambda hp: None  # noqa: E731
    _register_with_aliases(fake_register, "main_name", sentinel, ["a1", "a2"], True)

    assert registry["main_name"] is sentinel
    assert registry["a1"] is sentinel
    assert registry["a2"] is sentinel


def test_register_with_aliases_no_aliases():
    """_register_with_aliases works when aliases is None."""
    registry = {}

    def fake_register(name, builder, overwrite):
        registry[name] = builder

    sentinel = lambda hp: None  # noqa: E731
    _register_with_aliases(fake_register, "only_name", sentinel, None, False)

    assert registry == {"only_name": sentinel}


def test_register_custom_summary_with_builder_and_aliases():
    """Builder is registered under name + aliases via _register_with_aliases."""
    builder = lambda hp: "network"  # noqa: E731
    register_custom_summary_network(
        name="alias_test_net",
        space_factory=_DummySummarySpace,
        builder=builder,
        aliases=["atn"],
        overwrite=True,
    )
    assert SUMMARY_BUILDERS["alias_test_net"] is builder
    assert SUMMARY_BUILDERS["atn"] is builder
