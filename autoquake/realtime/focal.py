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
        event: dict,
        picks: list[dict],
    ) -> dict | None:
        """
        Estimate focal mechanism for an event.

        Args:
            event: Event dictionary with time, latitude, longitude, depth_km
            picks: List of pick dictionaries with phase_polarity

        Returns:
            Focal mechanism dictionary with strike, dip, rake or None
        """
        # Extract valid polarities
        polarity_picks = self.extract_polarities(picks)

        if len(polarity_picks) < self.min_polarities:
            logger.info(
                f'Insufficient polarities: {len(polarity_picks)} < {self.min_polarities}'
            )
            return None

        # Prepare GAfocal input
        try:
            result = self._run_gafocal(event, polarity_picks)
        except Exception as e:
            logger.error(f'GAfocal failed: {e}')
            return None

        if result is not None:
            self._focal_count += 1

        return result

    def _run_gafocal(
        self,
        event: dict,
        polarity_picks: list[dict],
    ) -> dict | None:
        """
        Run GAfocal executable.

        Args:
            event: Event dictionary
            polarity_picks: List of picks with polarity

        Returns:
            Focal mechanism dictionary or None
        """
        # Create temporary input file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.dout',
            dir=self.gafocal_dir,
            delete=False,
        ) as f:
            temp_dout = Path(f.name)

            # Write event header
            event_time = pd.to_datetime(event.get('time') or event.get('origin_time'))
            ymd = event_time.strftime('%Y%m%d')
            hour = event_time.hour
            minute = event_time.minute
            seconds = event_time.second + event_time.microsecond / 1_000_000

            lat = event['latitude']
            lon = event['longitude']
            depth = event.get('depth_km', 10.0)

            # Convert to degrees and minutes format
            lat_int = int(lat)
            lat_min = (lat - lat_int) * 60
            lon_int = int(lon)
            lon_min = (lon - lon_int) * 60

            f.write(
                f'{ymd:>9}{hour:>2}{minute:>2}{seconds:>6.2f}'
                f'{lat_int:>3}{lat_min:>5.2f}{lon_int:>4}{lon_min:>5.2f}'
                f'{depth:>6.2f}  0  0  0.00  0.00\n'
            )

            # Write polarity picks
            for pick in polarity_picks:
                station_id = pick['station_id']
                polarity = pick['polarity']

                # Get station coordinates
                if station_id in self._station_coords:
                    sta_info = self._station_coords[station_id]
                else:
                    # Try to extract short station name
                    short_name = station_id.split('.')[-1] if '.' in station_id else station_id
                    if short_name in self._station_coords:
                        sta_info = self._station_coords[short_name]
                    else:
                        continue

                # Calculate azimuth and takeoff angle
                # (simplified - GAfocal will recalculate)
                azimuth = self._calculate_azimuth(
                    event['latitude'], event['longitude'],
                    sta_info['latitude'], sta_info['longitude'],
                )

                # Simple takeoff angle estimate based on distance and depth
                dist = self._calculate_distance(
                    event['latitude'], event['longitude'],
                    sta_info['latitude'], sta_info['longitude'],
                )
                takeoff = self._estimate_takeoff_angle(dist, depth)

                f.write(
                    f'{station_id:<6}{azimuth:>6.1f}{takeoff:>6.1f} {polarity}\n'
                )

        # Run GAfocal
        try:
            result = subprocess.run(
                ['./gafocal'],
                input=temp_dout.name.encode() + b'\n',
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
        finally:
            # Clean up temp file
            if temp_dout.exists():
                temp_dout.unlink()

        # Parse results
        results_file = self.gafocal_dir / 'results.txt'
        if not results_file.exists():
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
        try:
            with open(results_file) as f:
                lines = f.readlines()

            if not lines:
                return None

            # Parse the last result (most recent)
            last_line = lines[-1].strip()
            parts = last_line.split()

            if len(parts) < 3:
                return None

            return {
                'strike': float(parts[0]),
                'dip': float(parts[1]),
                'rake': float(parts[2]),
                'misfit': float(parts[3]) if len(parts) > 3 else None,
            }

        except Exception as e:
            logger.error(f'Error parsing GAfocal results: {e}')
            return None

    def _calculate_azimuth(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Calculate azimuth from point 1 to point 2.

        Args:
            lat1, lon1: Source (event) coordinates
            lat2, lon2: Target (station) coordinates

        Returns:
            Azimuth in degrees (0-360)
        """
        import math

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        x = math.sin(dlon_rad) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)

        azimuth = math.degrees(math.atan2(x, y))

        return (azimuth + 360) % 360

    def _calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Calculate distance between two points in km.
        """
        import math

        R = 6371.0
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = math.sin(dlat/2)**2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def _estimate_takeoff_angle(self, dist: float, depth: float) -> float:
        """
        Estimate takeoff angle based on distance and depth.

        This is a simplified estimate. GAfocal will use a velocity model
        for more accurate calculation.

        Args:
            dist: Epicentral distance in km
            depth: Depth in km

        Returns:
            Estimated takeoff angle in degrees
        """
        import math

        if depth <= 0:
            depth = 0.1

        # Simple geometric estimate
        angle = math.degrees(math.atan2(dist, depth))

        # Adjust for typical velocity structure
        # (rays tend to curve due to velocity increase with depth)
        if dist < 50:
            angle *= 0.9
        elif dist < 100:
            angle *= 0.85
        else:
            angle *= 0.8

        return min(max(angle, 0), 180)

    @property
    def stats(self) -> dict:
        """Get estimator statistics."""
        return {
            'total_focal_mechanisms': self._focal_count,
            'gafocal_dir': str(self.gafocal_dir),
            'min_polarities': self.min_polarities,
        }