"""
Shared pytest fixtures for AutoQuake tests.
"""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_station_csv(tmp_path):
    """Create a minimal station CSV for testing."""
    path = tmp_path / 'station.csv'
    path.write_text(
        'station,longitude,latitude,elevation\n'
        'STA1,121.5,24.0,100\n'
        'STA2,121.6,24.1,150\n'
        'ABCD,121.7,24.2,200\n'
    )
    return path


@pytest.fixture
def sample_picks_df():
    """Create a sample picks DataFrame for testing."""
    return pd.DataFrame({
        'station_id': ['STA1', 'STA2', 'A001', 'A002', 'ABCD', 'EFGH'],
        'phase_time': [
            '2024-04-01T10:00:00.000',
            '2024-04-01T10:00:01.500',
            '2024-04-01T10:00:02.000',
            '2024-04-01T10:00:02.500',
            '2024-04-01T10:00:03.000',
            '2024-04-01T10:00:03.500',
        ],
        'phase_type': ['P', 'S', 'P', 'P', 'P', 'S'],
        'phase_score': [0.95, 0.90, 0.85, 0.88, 0.92, 0.87],
        'event_index': [0, 0, 0, 0, 0, 0],
    })


@pytest.fixture
def sample_events_df():
    """Create a sample events DataFrame for testing."""
    return pd.DataFrame({
        'time': ['2024-04-01T10:00:00.000', '2024-04-01T11:30:00.000'],
        'longitude': [121.5, 121.6],
        'latitude': [24.0, 24.1],
        'depth_km': [10.0, 15.0],
        'event_index': [0, 1],
    })


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary for testing."""
    return {
        'configs': [
            {
                'name': 'test_run',
                'result_path': '/tmp/test_results',
                'PhaseNet': {'enabled': False},
                'GaMMA': {'enabled': False},
                'H3DD': {'enabled': False},
                'Mag': {'enabled': False},
                'Diting': {'enabled': False},
                'Focal': {'enabled': False},
            }
        ]
    }


# =============================================================================
# Integration Test Fixtures
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


@pytest.fixture(scope='session')
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope='session')
def station_csv():
    """Return the path to station.csv fixture."""
    return FIXTURES_DIR / 'station.csv'


@pytest.fixture(scope='session')
def phasenet_picks_csv():
    """Return the path to phasenet_picks.csv fixture."""
    return FIXTURES_DIR / 'phasenet_picks.csv'


@pytest.fixture(scope='session')
def polarity_picks_csv():
    """Return the path to polarity_picks.csv fixture."""
    return FIXTURES_DIR / 'polarity_picks.csv'


@pytest.fixture(scope='session')
def mag_events_csv():
    """Return the path to mag_events.csv fixture."""
    return FIXTURES_DIR / 'mag_events.csv'


@pytest.fixture(scope='session')
def mag_picks_csv():
    """Return the path to mag_picks.csv fixture."""
    return FIXTURES_DIR / 'mag_picks.csv'


@pytest.fixture(scope='session')
def test_params_json():
    """Return the path to test_params.json fixture."""
    return FIXTURES_DIR / 'test_params.json'


@pytest.fixture(scope='session')
def sac_data_dir():
    """Return the path to SAC test data directory."""
    return FIXTURES_DIR / 'sac_data'


@pytest.fixture(scope='session')
def pz_data_dir():
    """Return the path to PZ (pole-zero) test data directory."""
    return FIXTURES_DIR / 'pz_data'
