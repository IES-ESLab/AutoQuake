"""Unit tests for the cross-component validation rules in ``RunConfig``.

These exercise ``ParamConfig.config_model`` only (pydantic + pandas), so they
are fast and run anywhere. Every ``RunConfig`` here points ``result_path`` at a
``tmp_path`` so the implicit ``mkdir`` never touches the repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ParamConfig.config_model import (
    GaMMAConfig,
    RunConfig,
)

pytestmark = pytest.mark.unit


def _gamma_enabled(**overrides) -> dict:
    base = dict(
        enabled=True,
        station='station.csv',
        velocity_model='velocity.vel',
        center=(121.625, 24.0),
        xlim_degree=[121.0, 122.25],
        ylim_degree=[23.25, 24.75],
    )
    base.update(overrides)
    return base


def _h3dd_enabled(**overrides) -> dict:
    base = dict(
        enabled=True,
        station='station.csv',
        model_3D='tomops_H14',
        runs={'first': {'event_name': 'H3DD_1', 'cutoff_distances': 0.0}},
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Component-level required-when-enabled validation
# ---------------------------------------------------------------------------
def test_gamma_missing_required_fields_raises():
    with pytest.raises(ValidationError, match='required when GaMMA is enabled'):
        GaMMAConfig(enabled=True)  # missing station/velocity_model/center/...


def test_gamma_disabled_allows_missing_fields():
    cfg = GaMMAConfig(enabled=False)
    assert cfg.enabled is False


# ---------------------------------------------------------------------------
# RunConfig cross-component dependency rules
# ---------------------------------------------------------------------------
def test_gamma_without_phasenet_or_picks_raises(tmp_path: Path):
    with pytest.raises(ValidationError, match='no picks_csv provided'):
        RunConfig(result_path=tmp_path / 'r', GaMMA=_gamma_enabled())


def test_gamma_with_explicit_picks_ok(tmp_path: Path):
    cfg = RunConfig(
        result_path=tmp_path / 'r',
        GaMMA=_gamma_enabled(picks_csv='picks.csv'),
    )
    assert cfg.is_component_enabled('GaMMA')


def test_h3dd_without_gamma_requires_csvs(tmp_path: Path):
    with pytest.raises(
        ValidationError, match='events_csv/picks_csv not fully provided'
    ):
        RunConfig(result_path=tmp_path / 'r', H3DD=_h3dd_enabled())


def test_h3dd_with_explicit_csvs_ok(tmp_path: Path):
    cfg = RunConfig(
        result_path=tmp_path / 'r',
        H3DD=_h3dd_enabled(events_csv='e.csv', picks_csv='p.csv'),
    )
    assert cfg.is_component_enabled('H3DD')


def test_magnitude_without_h3dd_requires_dout(tmp_path: Path):
    mag = dict(
        enabled=True,
        station='station.csv',
        sac_parent_dir='test_sac',
        pz_dir='test_pz',
    )
    with pytest.raises(ValidationError, match='no dout_file provided'):
        RunConfig(result_path=tmp_path / 'r', Magnitude=mag)


def test_polarity_manual_picks_with_upstream_enabled_raises(tmp_path: Path):
    with pytest.raises(
        ValidationError, match='only be set when both GaMMA and PhaseNet are disabled'
    ):
        RunConfig(
            result_path=tmp_path / 'r',
            GaMMA=_gamma_enabled(picks_csv='picks.csv'),
            Polarity={'enabled': True, 'picks_csv': 'manual.csv'},
        )


def test_polarity_with_gamma_upstream_ok(tmp_path: Path):
    cfg = RunConfig(
        result_path=tmp_path / 'r',
        GaMMA=_gamma_enabled(picks_csv='picks.csv'),
        Polarity={'enabled': True, 'sac_parent_dir': 'test_sac'},
    )
    assert cfg.is_component_enabled('Polarity')


def test_focal_without_inputs_raises(tmp_path: Path):
    with pytest.raises(ValidationError, match='no dout_file provided'):
        RunConfig(result_path=tmp_path / 'r', Focal={'enabled': True})


# ---------------------------------------------------------------------------
# is_component_enabled defaults
# ---------------------------------------------------------------------------
def test_is_component_enabled_for_absent_component(tmp_path: Path):
    cfg = RunConfig(result_path=tmp_path / 'r')
    assert cfg.is_component_enabled('PhaseNet') is False
    assert cfg.is_component_enabled('GaMMA') is False
    assert cfg.is_component_enabled('Focal') is False
