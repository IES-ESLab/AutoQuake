"""End-to-end integration test: the real 6-stage pipeline on the test data.

One real ``run_pipeline()`` is executed and then asserted at every stage
boundary, so a regression points straight at the offending stage without any
golden fixtures to maintain. The whole thing auto-skips (via ``require_full_env``)
when the conda env, submodules or the Linux ``h3dd``/``gafocal`` binaries are
unavailable -- e.g. on the macOS host or a bare CI runner.

Run it inside the devcontainer with::

    pytest tests/integration -m integration --tb=short
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_full_pipeline_end_to_end(require_full_env, e2e_workdir: Path):
    import pandas as pd

    import predict
    from ParamConfig.config_model import BatchConfig

    result_path = e2e_workdir / 'pipeline_test'
    with open('params_test.json', encoding='utf-8') as f:
        params = json.load(f)
    params['configs'][0]['result_path'] = str(result_path)

    cfg = BatchConfig(**params).configs[0]
    predict.run_pipeline(cfg)

    # --- PhaseNet ---------------------------------------------------------
    picks = result_path / 'picks_phasenet' / 'picks.csv'
    assert picks.exists() and picks.stat().st_size > 0, 'PhaseNet picks missing'

    # --- GaMMA ------------------------------------------------------------
    gamma_events = result_path / 'gamma_events.csv'
    gamma_picks = result_path / 'gamma_picks.csv'
    assert gamma_events.exists(), 'GaMMA events missing'
    assert gamma_picks.exists(), 'GaMMA picks missing'

    df_events = pd.read_csv(gamma_events)
    assert len(df_events) > 0, 'GaMMA associated zero events'
    assert {'longitude', 'latitude', 'depth_km'} <= set(df_events.columns)
    # Events must fall inside the configured search region.
    xlim = params['configs'][0]['GaMMA']['xlim_degree']
    ylim = params['configs'][0]['GaMMA']['ylim_degree']
    assert df_events['longitude'].between(*xlim).all(), 'event longitude out of region'
    assert df_events['latitude'].between(*ylim).all(), 'event latitude out of region'

    df_picks = pd.read_csv(gamma_picks)
    assert {'station_id', 'phase_time', 'phase_type', 'event_index'} <= set(
        df_picks.columns
    )

    # --- H3DD -------------------------------------------------------------
    douts = list(result_path.glob('*.dat_ch.dout'))
    assert douts, 'H3DD produced no .dout relocation file'
    assert all(d.stat().st_size > 0 for d in douts), 'H3DD .dout file is empty'

    # --- Magnitude --------------------------------------------------------
    mag_events = result_path / 'mag_events.csv'
    assert mag_events.exists() and mag_events.stat().st_size > 0, (
        'magnitude events missing'
    )

    # --- Polarity ---------------------------------------------------------
    polarity = result_path / 'polarity_picks.csv'
    assert polarity.exists() and polarity.stat().st_size > 0, 'polarity picks missing'

    # --- Focal (final catalog) -------------------------------------------
    focal = result_path / 'gafocal_catalog.txt'
    assert focal.exists() and focal.stat().st_size > 0, 'final focal catalog missing'
