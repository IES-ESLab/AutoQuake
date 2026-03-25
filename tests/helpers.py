"""
Test helper functions for integration tests.

This module contains helper functions to transform test data between
different pipeline stages. These functions are placeholders that need
to be implemented based on actual pipeline requirements.

Usage:
    from tests.helpers import for_mag_format, for_dt_format
"""

from pathlib import Path
import pandas as pd


def for_mag_format(
    phasenet_picks: Path | pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Convert PhaseNet picks to Magnitude input format.

    This function transforms PhaseNet output (picks CSV) into the format
    required by the Magnitude component (events CSV + picks CSV).

    Args:
        phasenet_picks: Path to PhaseNet picks CSV or DataFrame
        output_dir: Directory to write output files

    Returns:
        tuple[Path, Path]: Paths to (test_events.csv, test_picks.csv)

    Example:
        events_csv, picks_csv = for_mag_format(
            phasenet_picks=Path('small_phasenet_testset.csv'),
            output_dir=Path('tests/fixtures')
        )

    TODO: Implement this function based on actual format requirements.
          The function should:
          1. Read PhaseNet picks
          2. Group picks by event (using time clustering or event_index)
          3. Generate events CSV with columns: [time, longitude, latitude, depth_km, event_index]
          4. Generate picks CSV with columns required by Magnitude
    """
    raise NotImplementedError(
        'for_mag_format() not implemented. '
        'Please implement this function in tests/helpers.py'
    )


def for_dt_format(
    phasenet_picks: Path | pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Convert PhaseNet picks to DitingMotion input format.

    This function transforms PhaseNet output (picks CSV) into the format
    required by the DitingMotion component.

    Args:
        phasenet_picks: Path to PhaseNet picks CSV or DataFrame
        output_dir: Directory to write output files

    Returns:
        Path: Path to test_picks_for_dt.csv

    Example:
        dt_picks_csv = for_dt_format(
            phasenet_picks=Path('small_phasenet_testset.csv'),
            output_dir=Path('tests/fixtures')
        )

    TODO: Implement this function based on actual format requirements.
          The function should:
          1. Read PhaseNet picks
          2. Filter/transform to DitingMotion format
          3. Add event_index if needed
          4. Write output CSV with required columns
    """
    raise NotImplementedError(
        'for_dt_format() not implemented. '
        'Please implement this function in tests/helpers.py'
    )


def create_mock_gamma_output(
    station_csv: Path,
    phasenet_picks: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Create mock GaMMA output for testing H3DD without running GaMMA.

    This is useful for unit testing H3DD in isolation.

    Args:
        station_csv: Path to station CSV
        phasenet_picks: Path to PhaseNet picks CSV
        output_dir: Directory to write output files

    Returns:
        tuple[Path, Path]: Paths to (gamma_events.csv, gamma_picks.csv)

    TODO: Implement if needed for isolated H3DD testing.
    """
    raise NotImplementedError(
        'create_mock_gamma_output() not implemented. '
        'Please implement this function in tests/helpers.py'
    )
