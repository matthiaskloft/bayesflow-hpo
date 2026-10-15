"""Tests for the early Keras-backend check in ``bayesflow_hpo._backend``."""

import pytest

from bayesflow_hpo import _backend


def _fake_find_spec(installed):
    return lambda name: object() if name in installed else None


def test_selected_backend_missing(monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    monkeypatch.setattr(_backend, "find_spec", _fake_find_spec({"jax"}))
    with pytest.raises(ImportError, match="'torch' is not installed"):
        _backend.check_backend()


def test_selected_backend_present(monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    monkeypatch.setattr(_backend, "find_spec", _fake_find_spec({"torch"}))
    _backend.check_backend()


def test_unknown_backend_left_to_keras(monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "numpy")
    monkeypatch.setattr(_backend, "find_spec", _fake_find_spec(set()))
    _backend.check_backend()


def test_unset_and_none_installed(monkeypatch):
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    monkeypatch.setattr(_backend, "find_spec", _fake_find_spec(set()))
    with pytest.raises(ImportError, match="No Keras backend is installed"):
        _backend.check_backend()


def test_unset_and_one_installed(monkeypatch):
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    monkeypatch.setattr(_backend, "find_spec", _fake_find_spec({"jax"}))
    _backend.check_backend()
