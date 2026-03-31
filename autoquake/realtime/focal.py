"""
Realtime focal mechanism determination using GAfocal.

Uses polarity information from phasenet_plus model output.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
import shutil

import pandas as pd

logger = logging.getLogger(__name__)

# Polarity threshold for phasenet_plus output
POLARITY_THRESHOLD = 0.15


class RealtimeFocalMechanism:
    """
    Realtime focal mechanism estimator using GAfocal.

    Uses polarity information from phasenet_plus:
    - abs(phase_polarity) > 0.15: valid polarity
    - positive value: Upward (U)
    - negative value: Downward (D)

    Attributes:
        gafocal_dir: Directory containing GAfocal executable
        min_polarities: Minimum number of valid polarities required
    """

    def __init__(
        self,
        gafocal_dir: Path | None = None,
        station_info: pd.DataFrame | Path | None = None,
        min_polarities: int = 8,
    ):
        """
        Initialize the realtime focal mechanism estimator.

        Args:
            gafocal_dir: Directory containing GAfocal executable
                        (default: PROJECT_ROOT/GAfocal)
            station_info: DataFrame or path to station CSV
            min_polarities: Minimum polarities required for estimation
        """
        if gafocal_dir is None:
            self.gafocal_dir = Path(__file__).parents[2] / 'GAfocal'
        else:
            self.gafocal_dir = Path(gafocal_dir)

        if station_info is not None:
            if isinstance(station_info, Path):
                self.station_df = pd.read_csv(station_info)
            else:
                self.station_df = station_info.copy()
            self._build_station_lookup()
        else:
            self.station_df = None
            self._station_coords = {}

        self.min_polarities = min_polarities
        self._focal_count = 0

    def _build_station_lookup(self) -> None:
        """Build station coordinate lookup from DataFrame."""
        self._station_coords = {}
        for _, row in self.station_df.iterrows():
            self._station_coords[row['station']] = {
                'longitude': row['longitude'],
                'latitude': row['latitude'],
                'elevation': row.get('elevation', 0),
            }

    def extract_polarities(
        self,
        picks: list[dict],
    ) -> list[dict]:
        """
        Extract valid polarities from picks.

        Args:
            picks: List of pick dictionaries from phasenet_plus

        Returns:
            List of picks with valid polarity information
        """
        valid_picks = []

        for pick in picks:
            if pick.get('phase_type') != 'P':
                continue

            polarity_score = pick.get('phase_polarity')
            if polarity_score is None:
                continue

            try:
                polarity_score = float(polarity_score)
            except (ValueError, TypeError):
                continue

            if abs(polarity_score) < POLARITY_THRESHOLD:
                continue

            # Determine polarity direction
            if polarity_score > 0:
                polarity = 'U'  # Upward
            else:
                polarity = 'D'  # Downward

            pick_copy = pick.copy()
            pick_copy['polarity'] = polarity
            pick_copy['polarity_confidence'] = abs(polarity_score)
            valid_picks.append(pick_copy)

        return valid_picks

    def estimate(
        self,
        dout_file: Path,
    ) -> dict | None:
        """
        Estimate focal mechanism for an event by directly use their dout file.
        """
        # copy dout file to gafocal dir
        temp_dout = self.gafocal_dir / dout_file.name
        result_txt = self.gafocal_dir / 'results.txt'
        try:
            shutil.copy(dout_file, temp_dout)
        except Exception as e:
            logger.error(f'Failed to copy dout file: {e}')
            return None
        # Prepare GAfocal input
        try:
            result = self._run_gafocal(temp_dout.name)
            result_txt.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f'GAfocal failed: {e}')
            return None

        if result is not None:
            self._focal_count += 1

        return result

    def _run_gafocal(
        self,
        dout_file_name: str,
    ) -> dict | None:
        """
        Run GAfocal executable.

        Args:
            event: Event dictionary
            polarity_picks: List of picks with polarity

        Returns:
            Focal mechanism dictionary or None
        """
        # Run GAfocal
        try:
            result = subprocess.run(
                ['./gafocal'],
                input=dout_file_name.encode() + b'\n',
                cwd=self.gafocal_dir,
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.warning(f'GAfocal returned {result.returncode}')
                return None

        except subprocess.TimeoutExpired:
            logger.error('GAfocal timed out')
            return None
        except FileNotFoundError:
            logger.error('GAfocal executable not found')
            return None

        # Parse results
        results_file = self.gafocal_dir / 'results.txt'
        if not results_file.exists():
            logger.info("GAfocal did not solve the focal mechanism.")
            return None

        return self._parse_gafocal_results(results_file)

    def _parse_gafocal_results(self, results_file: Path) -> dict | None:
        """
        Parse GAfocal results file.

        Args:
            results_file: Path to results.txt

        Returns:
            Focal mechanism dictionary or None
        """
        def _get_max_columns(file_path):
            max_columns = 0

            # Open the file and analyze each line
            with open(file_path) as file:
                for line in file:
                    # Split the line using the space delimiter and count the columns
                    columns_in_line = len(line.split())
                    # Update max_columns if this line has more columns
                    if columns_in_line > max_columns:
                        max_columns = columns_in_line

            return max_columns 

        def _check_hms_gafocal(hms: str):
            """
            check whether the gafocal format's second overflow
            """
            minute = int(hms[3:5])
            second = int(hms[6:8])

            if second >= 60:
                minute += second // 60
                second = second % 60

            fixed_hms = hms[:3] + f'{minute:02d}' + hms[5:6] + f'{second:02d}'
            return fixed_hms

        try:
            max_columns = _get_max_columns(results_file)

            cols_to_read = list(range(max_columns - 1))
            df = pd.read_csv(
                results_file,
                sep=r'\s+',
                header=None,
                dtype={0: 'str', 1: 'str'},
                usecols=cols_to_read,
            )
            df[1] = df[1].apply(_check_hms_gafocal)
            df['time'] = df[0].apply(lambda x: x.replace('/', '-')) + 'T' + df[1]
            for col in [6, 8, 10]:
                df[col] = df[col].map(lambda x: int(x.split('+')[0]))
            df = df.rename(
                columns={
                    2: 'longitude',
                    3: 'latitude',
                    4: 'depth_km',
                    5: 'magnitude',
                    6: 'strike',
                    7: 'strike_err',
                    8: 'dip',
                    9: 'dip_err',
                    10: 'rake',
                    11: 'rake_err',
                    12: 'quality_index',
                    13: 'num_of_polarity',
                }
            )
            mask = [i for i in df.columns.tolist() if isinstance(i, str)]
            df = df[mask]
            return {
                'strike': int(df['strike'].iloc[0]),
                'strike_err': int(df['strike_err'].iloc[0]),
                'dip': int(df['dip'].iloc[0]),
                'dip_err': int(df['dip_err'].iloc[0]),
                'rake': int(df['rake'].iloc[0]),
                'rake_err': int(df['rake_err'].iloc[0]),
                'quality_index': int(df['quality_index'].iloc[0]),
                'num_of_polarity': int(df['num_of_polarity'].iloc[0]),
            }

        except Exception as e:
            logger.error(f'Error parsing GAfocal results: {e}')
            return None

    @property
    def stats(self) -> dict:
        """Get estimator statistics."""
        return {
            'total_focal_mechanisms': self._focal_count,
            'gafocal_dir': str(self.gafocal_dir),
            'min_polarities': self.min_polarities,
        }