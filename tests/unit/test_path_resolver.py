"""Unit tests for ``PathResolver`` explicit-path resolution.

Auto-detection is currently commented out in ``config_model.py``; these tests
pin the active behaviour: an explicit path is returned when it exists and a
missing explicit path raises ``FileNotFoundError``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ParamConfig.config_model import PathResolver

pytestmark = pytest.mark.unit

RESOLVE_METHODS = [
    'resolve_picks',
    'resolve_gamma_picks',
    'resolve_gamma_events',
    'resolve_dout',
]


@pytest.mark.parametrize('method', RESOLVE_METHODS)
def test_returns_existing_explicit_path(method: str, tmp_path: Path):
    resolver = PathResolver(tmp_path)
    target = tmp_path / 'input.dat'
    target.write_text('content')
    assert getattr(resolver, method)(target) == target


@pytest.mark.parametrize('method', RESOLVE_METHODS)
def test_missing_explicit_path_raises(method: str, tmp_path: Path):
    resolver = PathResolver(tmp_path)
    with pytest.raises(FileNotFoundError):
        getattr(resolver, method)(tmp_path / 'does_not_exist.dat')
