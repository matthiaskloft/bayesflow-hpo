"""Tests for workflow builder behavior."""

from bayesflow_hpo.builders.workflow import (
    WorkflowBuildConfig,
    _compile_candidate_for_compat,
    build_workflow,
)


class FakeBasicWorkflow:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_BW_PATH = "bayesflow_hpo.builders.workflow.bf.BasicWorkflow"


class _CompileNoArgsWorkflow:
    def __init__(self):
        self.compile_calls = []

    def compile(self):
        self.compile_calls.append(((), {}))


class _CompileOptimizerKwargWorkflow:
    def __init__(self):
        self.compile_calls = []

    def compile(self, *args, **kwargs):
        self.compile_calls.append((args, kwargs))
        if not kwargs:
            raise TypeError("optimizer required")


class _ApproxModel:
    def __init__(self):
        self.compile_calls = []

    def compile(self, *args, **kwargs):
        self.compile_calls.append((args, kwargs))
        if not kwargs:
            raise TypeError("optimizer required")


class _WorkflowWithApproxOnly:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.approximator = _ApproxModel()



def test_build_workflow_passes_optimizer_and_lr(monkeypatch):
    monkeypatch.setattr(_BW_PATH, FakeBasicWorkflow)

    wf = build_workflow(
        simulator=object(),
        adapter=object(),
        inference_network=object(),
        summary_network=None,
        params={"initial_lr": 1e-3},
        config=WorkflowBuildConfig(),
    )
    assert wf.kwargs["initial_learning_rate"] == 1e-3
    assert wf.kwargs["optimizer"] is not None


def test_build_workflow_uses_custom_optimizer(monkeypatch):
    monkeypatch.setattr(_BW_PATH, FakeBasicWorkflow)

    sentinel = object()
    wf = build_workflow(
        simulator=object(),
        adapter=object(),
        inference_network=object(),
        summary_network=None,
        params={"initial_lr": 1e-3},
        config=WorkflowBuildConfig(optimizer=sentinel),
    )
    assert wf.kwargs["optimizer"] is sentinel


def test_build_workflow_config_defaults():
    config = WorkflowBuildConfig()
    assert config.batches_per_epoch == 50
    assert config.optimizer is None
    assert config.inference_conditions is None


def test_compile_candidate_for_compat_calls_compile_without_args():
    workflow = _CompileNoArgsWorkflow()
    optimizer = object()

    _compile_candidate_for_compat(workflow, optimizer)

    assert workflow.compile_calls == [((), {})]


def test_compile_candidate_for_compat_falls_back_to_optimizer_kwarg():
    workflow = _CompileOptimizerKwargWorkflow()
    optimizer = object()

    _compile_candidate_for_compat(workflow, optimizer)

    assert workflow.compile_calls[0] == ((), {})
    assert workflow.compile_calls[1] == ((), {"optimizer": optimizer})


def test_build_workflow_compiles_approximator_for_compat(monkeypatch):
    monkeypatch.setattr(_BW_PATH, _WorkflowWithApproxOnly)

    wf = build_workflow(
        simulator=object(),
        adapter=object(),
        inference_network=object(),
        summary_network=None,
        params={"initial_lr": 1e-3},
        config=WorkflowBuildConfig(),
    )

    assert wf.approximator.compile_calls[0] == ((), {})
    assert "optimizer" in wf.approximator.compile_calls[1][1]
