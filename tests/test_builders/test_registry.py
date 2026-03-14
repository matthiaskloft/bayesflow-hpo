"""Tests for bayesflow_hpo.builders.registry."""

import pytest

from bayesflow_hpo.builders.registry import (
    INFERENCE_BUILDERS,
    SUMMARY_BUILDERS,
    get_inference_builder,
    get_summary_builder,
    list_inference_builders,
    list_summary_builders,
    register_inference_builder,
    register_summary_builder,
)


@pytest.fixture(autouse=True)
def _clean_registries():
    """Save and restore registries around each test."""
    inf_backup = dict(INFERENCE_BUILDERS)
    sum_backup = dict(SUMMARY_BUILDERS)
    yield
    INFERENCE_BUILDERS.clear()
    INFERENCE_BUILDERS.update(inf_backup)
    SUMMARY_BUILDERS.clear()
    SUMMARY_BUILDERS.update(sum_backup)


def _dummy_builder(params):
    return "built"


class TestInferenceBuilderRegistry:
    def test_register_and_get(self):
        register_inference_builder("test_net", _dummy_builder)
        builder = get_inference_builder("test_net")
        assert builder is _dummy_builder

    def test_case_insensitive(self):
        register_inference_builder("MyNet", _dummy_builder)
        builder = get_inference_builder("mynet")
        assert builder is _dummy_builder

    def test_duplicate_raises(self):
        register_inference_builder("dup", _dummy_builder)
        with pytest.raises(KeyError, match="already registered"):
            register_inference_builder("dup", _dummy_builder)

    def test_overwrite(self):
        register_inference_builder("ow", _dummy_builder)
        other = lambda p: "other"  # noqa: E731
        register_inference_builder("ow", other, overwrite=True)
        assert get_inference_builder("ow") is other

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown inference builder"):
            get_inference_builder("nonexistent")

    def test_list(self):
        register_inference_builder("alpha", _dummy_builder)
        register_inference_builder("beta", _dummy_builder)
        names = list_inference_builders()
        assert "alpha" in names
        assert "beta" in names
        assert names == sorted(names)


class TestSummaryBuilderRegistry:
    def test_register_and_get(self):
        register_summary_builder("test_sum", _dummy_builder)
        builder = get_summary_builder("test_sum")
        assert builder is _dummy_builder

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown summary builder"):
            get_summary_builder("nonexistent")

    def test_list(self):
        register_summary_builder("gamma", _dummy_builder)
        names = list_summary_builders()
        assert "gamma" in names
