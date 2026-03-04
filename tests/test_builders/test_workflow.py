"""Tests for workflow builder behavior."""

import pytest

from bayesflow_hpo.builders.workflow import WorkflowBuildConfig, build_workflow


class FailingApproximator:
    def compile(self, optimizer):
        raise RuntimeError("compile failed")


class FakeBasicWorkflow:
    def __init__(self, **kwargs):
        self.approximator = FailingApproximator()


def test_build_workflow_raises_on_compile_failure(monkeypatch):
    monkeypatch.setattr("bayesflow_hpo.builders.workflow.bf.BasicWorkflow", FakeBasicWorkflow)

    with pytest.raises(RuntimeError, match="Failed to compile workflow approximator"):
        build_workflow(
            simulator=object(),
            adapter=object(),
            inference_network=object(),
            summary_network=None,
            params={"initial_lr": 1e-3},
            config=WorkflowBuildConfig(),
        )
