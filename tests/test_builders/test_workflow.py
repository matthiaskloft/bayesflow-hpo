"""Tests for approximator builder behavior."""

from bayesflow_hpo.builders.workflow import (
    _compile_for_compat,
    _make_cosine_decay_optimizer,
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


class _CompileNoArgsModel:
    def __init__(self):
        self.compile_calls = []

    def compile(self):
        self.compile_calls.append("no_args")


class _CompileKwargModel:
    def __init__(self):
        self.compile_calls = []

    def compile(self, *args, **kwargs):
        self.compile_calls.append((args, kwargs))
        if not kwargs:
            raise TypeError("optimizer required")


def test_compile_for_compat_calls_compile_without_args():
    model = _CompileNoArgsModel()
    _compile_for_compat(model, object())
    assert model.compile_calls == ["no_args"]


def test_compile_for_compat_falls_back_to_optimizer_kwarg():
    model = _CompileKwargModel()
    optimizer = object()
    _compile_for_compat(model, optimizer)
    assert model.compile_calls[1] == ((), {"optimizer": optimizer})
