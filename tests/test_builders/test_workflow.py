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
