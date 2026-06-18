"""Guard the canonical example config (``tests/params_test.json``).

If the JSON drifts away from what ``config_model`` accepts, this fails fast in
CI -- before anyone discovers it by launching the real (slow) pipeline.

We build ``BatchConfig`` directly (not via ``predict.load_config``) to keep the
test pip-only: importing ``predict`` would pull in torch/obspy/onnxruntime.
``result_path`` is redirected into ``tmp_path`` so the implicit ``mkdir`` never
creates ``pipeline_test`` in the repo, and we ``chdir`` into the data dir so the
config's relative paths (``test_sac``, ``station.csv``) resolve.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ParamConfig.config_model import BatchConfig

pytestmark = pytest.mark.unit

ALL_COMPONENTS = ['PhaseNet', 'GaMMA', 'H3DD', 'Magnitude', 'Polarity', 'Focal']


@pytest.fixture
def loaded_config(
    params_dict: dict,
    tmp_path: Path,
    test_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    params_dict['configs'][0]['result_path'] = str(tmp_path / 'pipeline_test')
    monkeypatch.chdir(test_data_dir)
    return BatchConfig(**params_dict)


def test_params_test_json_parses_to_single_run(loaded_config):
    assert len(loaded_config.configs) == 1
    assert loaded_config.configs[0].name == 'pipeline'


def test_all_six_components_enabled(loaded_config):
    cfg = loaded_config.configs[0]
    for component in ALL_COMPONENTS:
        assert cfg.is_component_enabled(component), f'{component} should be enabled'


def test_phasenet_archive_expansion(loaded_config):
    phasenet = loaded_config.configs[0].PhaseNet
    # start=20240423, end=20240424 -> exactly one archive day.
    assert phasenet.date_list == ['20240423']
    assert phasenet.args_list is not None
    assert len(phasenet.args_list) == 1


def test_h3dd_configured_for_two_runs(loaded_config):
    assert loaded_config.configs[0].H3DD.get_run_count() == 2
