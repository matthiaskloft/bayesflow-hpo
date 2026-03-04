"""Tests for high-level custom registration helpers."""

from dataclasses import dataclass, field

from bayesflow_hpo import (
    list_registered_network_spaces,
    register_custom_summary_network,
)
from bayesflow_hpo.search_spaces.base import BaseSearchSpace, Dimension, IntDimension
from bayesflow_hpo.search_spaces.registry import get_summary_space


@dataclass
class _DummySummarySpace(BaseSearchSpace):
    width: IntDimension = field(default_factory=lambda: IntDimension("dummy_width", 8, 16, step=8))

    @property
    def dimensions(self) -> list[Dimension]:
        return [self.width]

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
