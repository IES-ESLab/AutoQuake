from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoquake import runtime
from ParamConfig.config_model import GaMMAConfig, PhaseNetConfig

pytestmark = pytest.mark.unit


def test_native_thread_limiter(monkeypatch):
    observed = []
    marker = object()
    monkeypatch.setattr(
        runtime,
        'threadpool_limits',
        lambda *, limits: observed.append(limits) or marker,
    )

    assert runtime.limit_native_threads(1) is marker
    assert observed == [1]


def test_effective_workers_clamps_to_cpus_and_tasks(monkeypatch):
    monkeypatch.setattr(runtime, 'visible_cpu_count', lambda: 8)

    assert runtime.effective_workers(12) == 8
    assert runtime.effective_workers(6, task_count=3) == 3
    assert runtime.effective_workers(2, task_count=20) == 2


def test_gamma_cpu_number_must_be_positive():
    with pytest.raises(ValidationError):
        GaMMAConfig(enabled=False, cpu_number=0)


def test_phasenet_workers_cannot_be_negative(tmp_path: Path):
    with pytest.raises(ValidationError):
        PhaseNetConfig(
            start='20240423',
            end='20240424',
            result_path=tmp_path,
            workers=-1,
        )


def test_phasenet_prediction_threads_default_and_validation(tmp_path: Path):
    config = PhaseNetConfig(
        start='20240423',
        end='20240424',
        result_path=tmp_path,
    )
    assert config.prediction_threads == 1

    configured = config.model_copy(update={'prediction_threads': 10})
    assert configured.prediction_threads == 10

    with pytest.raises(ValidationError):
        PhaseNetConfig(
            start='20240423',
            end='20240424',
            result_path=tmp_path,
            prediction_threads=0,
        )


def test_prediction_and_dataloader_threads_are_independent(monkeypatch):
    class FakeTorch:
        def __init__(self):
            self.thread_limits = []
            self.interop_limits = []

        def set_num_threads(self, limit):
            self.thread_limits.append(limit)

        def set_num_interop_threads(self, limit):
            self.interop_limits.append(limit)

    fake_torch = FakeTorch()
    native_limits = []
    marker = object()
    monkeypatch.setattr(runtime, 'effective_workers', lambda requested: requested)
    monkeypatch.setattr(
        runtime,
        'limit_native_threads',
        lambda requested: native_limits.append(requested) or marker,
    )

    assert runtime.configure_torch_threads(fake_torch, 10, 'cpu') == (10, True)
    runtime.initialize_torch_worker(fake_torch)

    assert fake_torch.thread_limits == [10, 1]
    assert fake_torch.interop_limits == [1]
    assert native_limits == [1]
    assert runtime._worker_thread_limiter is marker


def test_max_sigma11_default():
    assert GaMMAConfig(enabled=False).max_sigma11 == 2.0


def test_pipeline_forwards_max_sigma11():
    source = Path(__file__).parents[2] / 'predict.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    gamma_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'GaMMA'
    ]

    assert len(gamma_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in gamma_calls[0].keywords}
    expected = ast.parse('config.GaMMA.max_sigma11', mode='eval').body
    assert ast.dump(keywords['max_sigma11']) == ast.dump(expected)
