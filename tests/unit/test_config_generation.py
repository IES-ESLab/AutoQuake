"""Unit tests for pure generation/helper logic in ``config_model``.

Covers date-list generation, archive time-format validation, H3DD run
bookkeeping and the legacy single-config wrapping in ``BatchConfig``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ParamConfig.config_model import (
    BatchConfig,
    H3DDConfig,
    PhaseNetConfigReceiver,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# PhaseNet date-list / time-format helpers (called on a stub to avoid the
# filesystem-touching full construction).
# ---------------------------------------------------------------------------
def test_generate_date_list_is_start_inclusive_end_exclusive():
    stub = SimpleNamespace(start='20240423', end='20240426')
    assert PhaseNetConfigReceiver._generate_date_list(stub) == [
        '20240423',
        '20240424',
        '20240425',
    ]


def test_generate_date_list_single_day():
    stub = SimpleNamespace(start='20240423', end='20240424')
    assert PhaseNetConfigReceiver._generate_date_list(stub) == ['20240423']


def test_time_code_checker_accepts_yyyymmdd():
    stub = SimpleNamespace(start='20240423', end='20240424')
    # Should not raise.
    PhaseNetConfigReceiver._time_code_checker(stub)


def test_time_code_checker_rejects_non_archive_format():
    stub = SimpleNamespace(start='2024-04-23T00:00:00', end='20240424')
    with pytest.raises(ValueError, match='YYYYMMDD'):
        PhaseNetConfigReceiver._time_code_checker(stub)


# ---------------------------------------------------------------------------
# H3DD run bookkeeping
# ---------------------------------------------------------------------------
def _run(name: str, cutoff: float) -> dict:
    return {'event_name': name, 'cutoff_distances': cutoff}


def test_h3dd_run_count_zero_without_runs():
    cfg = H3DDConfig(enabled=False)
    assert cfg.get_run_count() == 0
    assert cfg.get_run_config(0) is None


def test_h3dd_single_run():
    cfg = H3DDConfig(enabled=False, runs={'first': _run('H3DD_1', 0.0)})
    assert cfg.get_run_count() == 1
    assert cfg.get_run_config(0).event_name == 'H3DD_1'
    assert cfg.get_run_config(1) is None


def test_h3dd_double_run():
    cfg = H3DDConfig(
        enabled=False,
        runs={'first': _run('H3DD_1', 0.0), 'second': _run('H3DD_2', 2.0)},
    )
    assert cfg.get_run_count() == 2
    assert cfg.get_run_config(0).cutoff_distances == 0.0
    assert cfg.get_run_config(1).event_name == 'H3DD_2'
    assert cfg.get_run_config(2) is None


# ---------------------------------------------------------------------------
# BatchConfig legacy-format handling
# ---------------------------------------------------------------------------
def test_legacy_single_config_is_wrapped(tmp_path: Path):
    batch = BatchConfig(name='legacy', result_path=str(tmp_path / 'r'))
    assert len(batch.configs) == 1
    assert batch.configs[0].name == 'legacy'


def test_batch_format_preserves_order(tmp_path: Path):
    batch = BatchConfig(
        configs=[
            {'name': 'a', 'result_path': str(tmp_path / 'a')},
            {'name': 'b', 'result_path': str(tmp_path / 'b')},
        ]
    )
    assert [c.name for c in batch.configs] == ['a', 'b']
