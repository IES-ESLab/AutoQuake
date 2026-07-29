"""Shared pytest fixtures and environment guards for the AutoQuake test suite.

Two layers of tests live under ``tests/``:

* ``tests/unit`` -- fast, pure-python checks of the configuration layer
  (``ParamConfig/config_model.py``). Depend only on ``pydantic`` + ``pandas``,
  so they run anywhere (CI, host, container).
* ``tests/integration`` -- a real end-to-end run of ``predict.py`` on the data
  kept in ``tests/``. Needs the full conda environment (torch / obspy /
  onnxruntime), the ``EQNet`` + ``GaMMA`` submodules and the Linux ``h3dd`` /
  ``gafocal`` binaries, so it auto-skips when any of those is unavailable.

Fixtures here are resolved relative to this file, so tests no longer require
``cd tests`` to find their data.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path

import pytest

# Directory holding the test materials (this file lives in ``tests/``).
TEST_DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DATA_DIR.parent
PARAMS_TEST_JSON = TEST_DATA_DIR / 'params_test.json'


# ---------------------------------------------------------------------------
# Generic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def test_data_dir() -> Path:
    """Absolute path to the directory containing the test data + config."""
    return TEST_DATA_DIR


@pytest.fixture
def params_test_json() -> Path:
    """Absolute path to the canonical example config (``params_test.json``)."""
    return PARAMS_TEST_JSON


@pytest.fixture
def result_dir(tmp_path: Path) -> Path:
    """A throwaway result directory so runs never pollute the repo."""
    out = tmp_path / 'result'
    out.mkdir()
    return out


@pytest.fixture
def params_dict() -> dict:
    """Parsed ``params_test.json`` as a plain dict (no validation side effects)."""
    with open(PARAMS_TEST_JSON, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Integration environment guards
# ---------------------------------------------------------------------------
def _missing_modules(*names: str) -> list[str]:
    return [n for n in names if importlib.util.find_spec(n) is None]


def _missing_binaries(*paths: Path) -> list[str]:
    missing = []
    for p in paths:
        if not (p.exists() and os.access(p, os.X_OK)):
            missing.append(str(p))
    return missing


def _missing_submodules(*paths: Path) -> list[str]:
    # A submodule is "present" once it has been checked out (non-empty dir).
    return [str(p) for p in paths if not p.exists() or not any(p.iterdir())]


def full_env_skip_reason() -> str | None:
    """Return a human-readable reason the full pipeline cannot run, else None.

    Used by the integration test to ``pytest.skip`` cleanly on the macOS host
    or a bare CI runner instead of erroring on an import / missing binary.
    """
    reasons: list[str] = []

    # h3dd / gafocal are compiled Linux x86-64 ELF binaries. They carry the
    # executable bit on any OS, but only actually run on Linux, so gate on the
    # platform rather than just os.access(X_OK).
    if platform.system() != 'Linux':
        reasons.append(
            'h3dd/gafocal are Linux x86-64 binaries; current platform '
            f'is {platform.system()}'
        )

    missing_mods = _missing_modules('torch', 'obspy', 'onnxruntime', 'numba')
    if missing_mods:
        reasons.append(f'missing python modules: {", ".join(missing_mods)}')

    missing_subs = _missing_submodules(
        REPO_ROOT / 'autoquake' / 'EQNet',
        REPO_ROOT / 'autoquake' / 'GaMMA',
    )
    if missing_subs:
        reasons.append(f'uninitialised submodules: {", ".join(missing_subs)}')

    missing_bins = _missing_binaries(
        REPO_ROOT / 'H3DD' / 'h3dd',
        REPO_ROOT / 'GAfocal' / 'gafocal',
    )
    if missing_bins:
        reasons.append(f'missing/non-executable binaries: {", ".join(missing_bins)}')

    return '; '.join(reasons) if reasons else None


@pytest.fixture
def require_full_env() -> None:
    """Skip the test (cleanly) when the full pipeline env is unavailable."""
    reason = full_env_skip_reason()
    if reason:
        pytest.skip(f'full pipeline environment unavailable: {reason}')


@pytest.fixture
def e2e_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prepare an isolated copy of the test data and chdir into it.

    ``params_test.json`` uses paths relative to ``tests/`` (e.g. ``test_sac``,
    ``station.csv``, ``./vel_model/...``). We copy those materials into a temp
    directory and run there so the real pipeline writes its outputs into the
    sandbox, leaving the repo untouched.
    """
    for name in ('test_sac', 'test_pz', 'vel_model'):
        shutil.copytree(TEST_DATA_DIR / name, tmp_path / name)
    shutil.copy2(TEST_DATA_DIR / 'station.csv', tmp_path / 'station.csv')
    shutil.copy2(PARAMS_TEST_JSON, tmp_path / 'params_test.json')
    monkeypatch.chdir(tmp_path)
    return tmp_path
