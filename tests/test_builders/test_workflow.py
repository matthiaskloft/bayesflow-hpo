"""Tests for approximator builder behavior."""

import keras
import numpy as np

from bayesflow_hpo.builders.workflow import (
    InverseSqrtDecay,
    _compile_for_compat,
    _make_cosine_decay_optimizer,
    _make_inverse_sqrt_optimizer,
    build_continuous_approximator,
)


class _FakeInferenceSpace:
    def build(self, params):
        return object()


class _FakeSummarySpace:
    def build(self, params):
        return object()


class _FakeSearchSpace:
    def __init__(self, summary=False):
        self.inference_space = _FakeInferenceSpace()
        self.summary_space = _FakeSummarySpace() if summary else None


class _FakeContinuousApproximator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_CA_PATH = "bayesflow_hpo.builders.workflow.bf.ContinuousApproximator"


def test_build_continuous_approximator_creates_approx(monkeypatch):
    monkeypatch.setattr(_CA_PATH, _FakeContinuousApproximator)

    approx = build_continuous_approximator(
        hparams={"initial_lr": 1e-3},
        adapter=object(),
        search_space=_FakeSearchSpace(),
    )
    assert isinstance(approx, _FakeContinuousApproximator)
    assert approx.kwargs["summary_network"] is None


def test_build_continuous_approximator_with_summary(monkeypatch):
    monkeypatch.setattr(_CA_PATH, _FakeContinuousApproximator)

    approx = build_continuous_approximator(
        hparams={"initial_lr": 1e-3},
        adapter=object(),
        search_space=_FakeSearchSpace(summary=True),
    )
    assert approx.kwargs["summary_network"] is not None


def test_make_cosine_decay_optimizer():
    opt = _make_cosine_decay_optimizer(1e-3, 1000)
    assert opt is not None


def test_cosine_decay_optimizer_supports_linear_warmup():
    opt = _make_cosine_decay_optimizer(1.0, 12, 4)
    schedule = opt._learning_rate
    values = [float(schedule(step)) for step in range(5)]
    np.testing.assert_allclose(values, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_cosine_decay_optimizer_rejects_warmup_without_decay_budget():
    with np.testing.assert_raises_regex(ValueError, "smaller than"):
        _make_cosine_decay_optimizer(1e-3, 4, 4)


def test_inverse_sqrt_decay_warms_up_then_decays():
    schedule = InverseSqrtDecay(peak_learning_rate=1.0, warmup_steps=4)
    values = [float(schedule(step)) for step in range(8)]
    np.testing.assert_allclose(values[:4], [0.25, 0.5, 0.75, 1.0])
    assert values[4] < values[3]
    assert values[7] < values[4]


def test_inverse_sqrt_decay_serialization_roundtrip():
    schedule = InverseSqrtDecay(peak_learning_rate=1e-3, warmup_steps=100)
    restored = keras.optimizers.schedules.deserialize(
        keras.optimizers.schedules.serialize(schedule)
    )
    assert isinstance(restored, InverseSqrtDecay)
    assert restored.get_config() == schedule.get_config()


def test_make_inverse_sqrt_optimizer():
    opt = _make_inverse_sqrt_optimizer(1e-3, 100)
    assert opt is not None


class _CompileNoArgsModel:
    """Model whose compile() accepts no arguments (e.g. pre-configured)."""

    def __init__(self):
        self.compile_calls = []

    def compile(self):
        self.compile_calls.append("no_args")


class _CompileKwargModel:
    """Model whose compile() requires an optimizer kwarg."""

    def __init__(self):
        self.compile_calls = []

    def compile(self, *, optimizer=None):
        self.compile_calls.append(("kwarg", optimizer))


class _CompilePositionalModel:
    """Model whose compile() accepts optimizer as a positional arg only."""

    def __init__(self):
        self.compile_calls = []

    def compile(self, optimizer):
        self.compile_calls.append(("positional", optimizer))


def test_compile_for_compat_prefers_optimizer_kwarg():
    """When compile accepts optimizer=, it should be used (not no-arg)."""
    model = _CompileKwargModel()
    optimizer = object()
    _compile_for_compat(model, optimizer)
    assert len(model.compile_calls) == 1
    assert model.compile_calls[0] == ("kwarg", optimizer)


def test_compile_for_compat_falls_back_to_positional():
    """When compile only accepts positional optimizer, use that."""
    model = _CompilePositionalModel()
    optimizer = object()
    _compile_for_compat(model, optimizer)
    assert len(model.compile_calls) == 1
    assert model.compile_calls[0] == ("positional", optimizer)


def test_compile_for_compat_falls_back_to_no_args():
    """When compile doesn't accept an optimizer at all, call without args."""
    model = _CompileNoArgsModel()
    _compile_for_compat(model, object())
    assert model.compile_calls == ["no_args"]
