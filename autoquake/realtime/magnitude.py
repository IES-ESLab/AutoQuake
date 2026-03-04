"""
Realtime magnitude estimation using PGA/PGV DP Table.

Two-stage approach:
- Stage 1 (Preliminary): Quick estimate using raw amplitude
- Stage 2 (Update): Accurate calculation using WA-simulated amplitude from DP Table
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .buffers import WaveformBuffer

logger = logging.getLogger(__name__)

# Wood-Anderson simulation parameters (from magnitude.py)
WA_SIMULATE = {
    'poles': [(-6.28318 + 4.71239j), (-6.28318 - 4.71239j)],
    'zeros': [0j, 0j],
    'sensitivity': 1.0,
    'gain': 2800.0,
}
PRE_FILT = (0.1, 0.5, 30, 35)

# Pre-computed time windows in seconds
TIME_WINDOWS = [20, 30, 40, 60, 80]


def find_nearest_window(target: float) -> int:
    """Find the nearest pre-computed time window."""
    return min(TIME_WINDOWS, key=lambda x: abs(x - target))


@dataclass
class AmplitudeEntry:
    """Single amplitude measurement entry."""
    pga: float  # Peak ground acceleration (or WA-simulated amplitude)
    pgv: float | None = None  # Peak ground velocity (optional)
    timestamp: datetime | None = None


@dataclass
class AmplitudeTable:
    """
    DP Table for pre-computed amplitudes across multiple time windows.

    Structure:
        station_id -> phase_type -> phase_time_str -> window_key -> AmplitudeEntry

    This table stores amplitudes calculated for each pick at multiple time windows,
    allowing quick lookup after event location is determined.
    """

    _table: dict = field(default_factory=lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    ))
    _entry_count: int = 0
    max_entries: int = 10000  # Prevent unbounded growth

    def add_entry(
        self,
        station_id: str,
        phase_type: str,
        phase_time: datetime | str,
        window_seconds: int,
        amplitude_e: float,
        amplitude_n: float,
    ) -> None:
        """
        Add amplitude entry for a specific pick and time window.

        Args:
            station_id: Station identifier
            phase_type: 'P' or 'S'
            phase_time: Phase arrival time
            window_seconds: Time window in seconds (20, 30, 40, 60, or 80)
            amplitude_e: E-component amplitude (WA-simulated)
            amplitude_n: N-component amplitude (WA-simulated)
        """
        if isinstance(phase_time, datetime):
            phase_time_str = phase_time.isoformat(timespec='milliseconds')
        else:
            phase_time_str = phase_time

        window_key = f"window_{window_seconds}s"

        # Combined amplitude (RMS of two horizontal components)
        combined_amp = math.sqrt(amplitude_e**2 + amplitude_n**2)

        self._table[station_id][phase_type][phase_time_str][window_key] = AmplitudeEntry(
            pga=combined_amp,
            timestamp=datetime.now(),
        )
        self._entry_count += 1

        # Cleanup if too many entries
        if self._entry_count > self.max_entries:
            self._cleanup_oldest()

    def get_amplitude(
        self,
        station_id: str,
        phase_type: str,
        phase_time: datetime | str,
        window_seconds: int,
    ) -> float | None:
        """
        Get amplitude for a specific pick and time window.

        Args:
            station_id: Station identifier
            phase_type: 'P' or 'S'
            phase_time: Phase arrival time
            window_seconds: Time window in seconds

        Returns:
            Combined amplitude (RMS of E and N) or None if not found
        """
        if isinstance(phase_time, datetime):
            phase_time_str = phase_time.isoformat(timespec='milliseconds')
        else:
            phase_time_str = phase_time

        # Find nearest window
        nearest_window = find_nearest_window(window_seconds)
        window_key = f"window_{nearest_window}s"

        try:
            entry = self._table[station_id][phase_type][phase_time_str][window_key]
            return entry.pga
        except KeyError:
            return None

    def _cleanup_oldest(self) -> None:
        """Remove oldest entries to prevent unbounded growth."""
        # Simple cleanup: remove entries older than 10 minutes
        cutoff = datetime.now() - timedelta(minutes=10)

        for station_id in list(self._table.keys()):
            for phase_type in list(self._table[station_id].keys()):
                for phase_time_str in list(self._table[station_id][phase_type].keys()):
                    for window_key in list(self._table[station_id][phase_type][phase_time_str].keys()):
                        entry = self._table[station_id][phase_type][phase_time_str][window_key]
                        if entry.timestamp and entry.timestamp < cutoff:
                            del self._table[station_id][phase_type][phase_time_str][window_key]
                            self._entry_count -= 1

    def clear(self) -> None:
        """Clear all entries."""
        self._table.clear()
        self._entry_count = 0

    @property
    def stats(self) -> dict:
        """Get table statistics."""
        return {
            'total_entries': self._entry_count,
            'stations': len(self._table),
        }


class RealtimeMagnitude:
    """
    Realtime magnitude estimator with two-stage approach.

    Stage 1 (Preliminary): Quick estimate using raw phase_amplitude from picks
    Stage 2 (Update): Accurate calculation using WA-simulated amplitude from DP Table

    Uses CWA (1993) Taiwan empirical formula for local magnitude.
    """

    def __init__(
        self,
        station_info: pd.DataFrame | Path,
        pz_dir: Path | None = None,
        use_wa_simulation: bool = True,
    ):
        """
        Initialize the realtime magnitude estimator.

        Args:
            station_info: DataFrame or path to station CSV with columns:
                          [station, longitude, latitude, elevation]
            pz_dir: Directory containing PZ (pole-zero) files for WA simulation
            use_wa_simulation: Whether to use WA simulation (requires PZ files)
        """
        if isinstance(station_info, Path):
            self.station_df = pd.read_csv(station_info)
        else:
            self.station_df = station_info.copy()

        self.pz_dir = Path(pz_dir) if pz_dir else None
        self.use_wa_simulation = use_wa_simulation and pz_dir is not None

        # Build station lookup
        self._station_coords = {}
        for _, row in self.station_df.iterrows():
            self._station_coords[row['station']] = {
                'longitude': row['longitude'],
                'latitude': row['latitude'],
                'elevation': row.get('elevation', 0),
            }

        # Amplitude table for DP lookup
        self.amplitude_table = AmplitudeTable()

        self._magnitude_count = 0

    def estimate_preliminary(
        self,
        event: dict,
        picks: list[dict],
    ) -> float | None:
        """
        Stage 1: Quick preliminary magnitude estimate.

        Uses raw phase_amplitude from picks without WA simulation.
        Suitable for immediate preliminary reports.

        Args:
            event: Event dictionary with latitude, longitude, depth_km
            picks: List of pick dictionaries with station_id, phase_amplitude

        Returns:
            Preliminary magnitude estimate or None if insufficient data
        """
        magnitudes = []

        for pick in picks:
            if pick.get('phase_type') != 'P':
                continue

            amp = pick.get('phase_amplitude')
            if amp is None:
                continue

            # Convert amplitude string to float if needed
            if isinstance(amp, str):
                try:
                    amp = float(amp)
                except ValueError:
                    continue

            station_id = pick['station_id']
            if station_id not in self._station_coords:
                continue

            # Calculate distance
            dist = self._calculate_distance(
                event['latitude'], event['longitude'],
                self._station_coords[station_id]['latitude'],
                self._station_coords[station_id]['longitude'],
            )

            depth = event.get('depth_km', 10.0)
            elevation = self._station_coords[station_id].get('elevation', 0)
            actual_depth = depth + elevation / 1000.0  # Convert elevation to km

            # Simple magnitude estimate using raw amplitude
            # This is a rough approximation - Stage 2 provides better accuracy
            try:
                ml = self._amplitude_to_ml_simple(amp, dist, actual_depth)
                magnitudes.append(ml)
            except (ValueError, ZeroDivisionError):
                continue

        if not magnitudes:
            return None

        # Use median for robustness
        return float(np.median(magnitudes))

    def estimate_from_relocation(
        self,
        event: dict,
        picks: list[dict],
    ) -> float | None:
        """
        Stage 2: Accurate magnitude using relocated event and DP Table.

        Uses pre-computed WA-simulated amplitudes from the DP Table.

        Args:
            event: Relocated event with latitude, longitude, depth_km
            picks: List of associated picks

        Returns:
            Accurate magnitude estimate or None if insufficient data
        """
        magnitudes = []

        # Group picks by station
        station_picks = defaultdict(list)
        for pick in picks:
            station_picks[pick['station_id']].append(pick)

        for station_id, sta_picks in station_picks.items():
            if station_id not in self._station_coords:
                continue

            # Calculate distance to relocated event
            dist = self._calculate_distance(
                event['latitude'], event['longitude'],
                self._station_coords[station_id]['latitude'],
                self._station_coords[station_id]['longitude'],
            )

            depth = event.get('depth_km', 10.0)
            elevation = self._station_coords[station_id].get('elevation', 0)
            actual_depth = depth + elevation / 1000.0

            # Determine time window based on picks
            window_seconds = self._determine_time_window(sta_picks, dist)

            # Find P wave pick
            p_pick = next((p for p in sta_picks if p.get('phase_type') == 'P'), None)
            if p_pick is None:
                continue

            # Look up amplitude from DP Table
            amplitude = self.amplitude_table.get_amplitude(
                station_id=station_id,
                phase_type='P',
                phase_time=p_pick['phase_time'],
                window_seconds=window_seconds,
            )

            if amplitude is None:
                # Fall back to raw amplitude
                amp = p_pick.get('phase_amplitude')
                if amp is None:
                    continue
                if isinstance(amp, str):
                    try:
                        amplitude = float(amp)
                    except ValueError:
                        continue
                else:
                    amplitude = amp

            try:
                ml = self._amplitude_to_ml_cwa(amplitude, dist, actual_depth)
                magnitudes.append(ml)
            except (ValueError, ZeroDivisionError):
                continue

        if not magnitudes:
            return None

        self._magnitude_count += 1
        return float(np.median(magnitudes))

    def compute_amplitude_dp_table(
        self,
        waveform_buffer: WaveformBuffer,
        picks: list[dict],
    ) -> None:
        """
        Compute and store amplitudes in DP Table for all picks.

        This should be called while waveform data is still in the buffer,
        before it gets overwritten by new data.

        Args:
            waveform_buffer: WaveformBuffer with current waveform data
            picks: List of picks to compute amplitudes for
        """
        for pick in picks:
            if pick.get('phase_type') != 'P':
                continue

            station_id = pick['station_id']
            phase_time = pick['phase_time']

            # Get waveform data for this station
            window_data = waveform_buffer.get_window(
                station_ids=[station_id],
                duration=timedelta(seconds=max(TIME_WINDOWS) + 10),
            )

            if station_id not in window_data:
                continue

            station_data = window_data[station_id]
            data = station_data['data']
            start_time = station_data['start_time']
            sampling_rate = station_data['sampling_rate']

            # Compute amplitudes for all time windows
            self._compute_amplitudes_for_pick(
                station_id=station_id,
                phase_time=phase_time,
                data=data,
                start_time=start_time,
                sampling_rate=sampling_rate,
            )

    def _compute_amplitudes_for_pick(
        self,
        station_id: str,
        phase_time: datetime | str,
        data: np.ndarray,
        start_time: datetime,
        sampling_rate: float,
    ) -> None:
        """
        Compute amplitudes for all time windows for a single pick.

        Args:
            station_id: Station identifier
            phase_time: Phase arrival time
            data: Waveform data [3, samples]
            start_time: Start time of waveform data
            sampling_rate: Sampling rate in Hz
        """
        from obspy import Trace, UTCDateTime
        from obspy.io.sac.sacpz import attach_paz

        if isinstance(phase_time, str):
            phase_time = pd.to_datetime(phase_time)

        # Calculate sample index for phase arrival
        time_offset = (phase_time - start_time).total_seconds()
        arrival_idx = int(time_offset * sampling_rate)

        if arrival_idx < 0 or arrival_idx >= data.shape[1]:
            logger.debug(f'Phase time {phase_time} outside buffer range')
            return

        # For each time window
        for window_seconds in TIME_WINDOWS:
            window_samples = int(window_seconds * sampling_rate)

            # Calculate window indices (arrival - 3s to arrival + window)
            pre_samples = int(3 * sampling_rate)
            start_idx = max(0, arrival_idx - pre_samples)
            end_idx = min(data.shape[1], arrival_idx + window_samples)

            if end_idx - start_idx < sampling_rate:  # Less than 1 second
                continue

            # Extract window for E and N components
            e_data = data[0, start_idx:end_idx]
            n_data = data[1, start_idx:end_idx]

            if self.use_wa_simulation and self.pz_dir:
                # Method C: Rebuild obspy Trace for WA simulation
                try:
                    amp_e = self._compute_wa_amplitude(
                        station_id, 'E', e_data, start_time, sampling_rate
                    )
                    amp_n = self._compute_wa_amplitude(
                        station_id, 'N', n_data, start_time, sampling_rate
                    )
                except Exception as e:
                    logger.debug(f'WA simulation failed for {station_id}: {e}')
                    # Fall back to raw amplitude
                    amp_e = float(np.max(np.abs(e_data)))
                    amp_n = float(np.max(np.abs(n_data)))
            else:
                # Use raw amplitude
                amp_e = float(np.max(np.abs(e_data)))
                amp_n = float(np.max(np.abs(n_data)))

            # Store in DP Table
            self.amplitude_table.add_entry(
                station_id=station_id,
                phase_type='P',
                phase_time=phase_time,
                window_seconds=window_seconds,
                amplitude_e=amp_e,
                amplitude_n=amp_n,
            )

    def _compute_wa_amplitude(
        self,
        station_id: str,
        component: str,
        data: np.ndarray,
        start_time: datetime,
        sampling_rate: float,
    ) -> float:
        """
        Compute WA-simulated amplitude for a single component.

        Uses Method C: Rebuild obspy Trace from numpy array.

        Args:
            station_id: Station identifier
            component: 'E' or 'N'
            data: Waveform data for single component
            start_time: Start time
            sampling_rate: Sampling rate in Hz

        Returns:
            Maximum WA-simulated amplitude
        """
        from obspy import Trace, UTCDateTime
        from obspy.io.sac.sacpz import attach_paz

        # Create obspy Trace
        trace = Trace(data=data.copy())
        trace.stats.sampling_rate = sampling_rate
        trace.stats.starttime = UTCDateTime(start_time)

        # Find PZ file
        # Try common naming patterns
        pz_patterns = [
            f'*{station_id}*{component}*',
            f'*{station_id.split(".")[-1]}*{component}*',
        ]

        pz_file = None
        for pattern in pz_patterns:
            matches = list(self.pz_dir.glob(pattern))
            if matches:
                pz_file = matches[0]
                break

        if pz_file is None:
            raise FileNotFoundError(f'No PZ file found for {station_id} {component}')

        # Attach and simulate
        attach_paz(tr=trace, paz_file=str(pz_file))
        trace.simulate(paz_remove='self', paz_simulate=WA_SIMULATE, pre_filt=PRE_FILT)

        # Return max amplitude
        return float(max(np.max(trace.data), abs(np.min(trace.data))))

    def _determine_time_window(
        self,
        picks: list[dict],
        dist: float,
    ) -> int:
        """
        Determine appropriate time window based on picks and distance.

        Follows magnitude.py logic:
        - P+S: (S-P)*2 + 5s, max 80s
        - P only: 20s (dist≤100), 40s (100<dist≤250), 80s (dist>250)
        - S only: 30s

        Args:
            picks: List of picks for a station
            dist: Epicentral distance in km

        Returns:
            Time window in seconds
        """
        p_pick = next((p for p in picks if p.get('phase_type') == 'P'), None)
        s_pick = next((p for p in picks if p.get('phase_type') == 'S'), None)

        if p_pick and s_pick:
            # Case 1: Both P and S
            p_time = pd.to_datetime(p_pick['phase_time'])
            s_time = pd.to_datetime(s_pick['phase_time'])
            s_minus_p = (s_time - p_time).total_seconds()
            window = min(s_minus_p * 2 + 5, 80)
        elif p_pick:
            # Case 2: Only P
            if dist > 250:
                window = 80
            elif dist > 100:
                window = 40
            else:
                window = 20
        else:
            # Case 3: Only S
            window = 30

        return find_nearest_window(window)

    def _calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Calculate epicentral distance in km using Haversine formula.

        Args:
            lat1, lon1: Event coordinates
            lat2, lon2: Station coordinates

        Returns:
            Distance in km
        """
        R = 6371.0  # Earth's radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def _amplitude_to_ml_simple(
        self,
        amplitude: float,
        dist: float,
        depth: float,
    ) -> float:
        """
        Simple magnitude estimate from raw amplitude.

        This is a rough approximation for Stage 1 preliminary reports.

        Args:
            amplitude: Raw amplitude
            dist: Epicentral distance in km
            depth: Depth in km

        Returns:
            Rough magnitude estimate
        """
        if amplitude <= 0:
            raise ValueError("Amplitude must be positive")

        # Simple log-distance correction
        R = math.sqrt(dist**2 + depth**2)
        if R <= 0:
            R = 0.1

        # Empirical approximation (to be calibrated)
        log_amp = math.log10(amplitude)
        ml = log_amp + 1.0 * math.log10(R) + 0.003 * R + 2.0

        return ml

    def _amplitude_to_ml_cwa(
        self,
        amplitude: float,
        dist: float,
        depth: float,
    ) -> float:
        """
        Calculate magnitude using CWA (1993) Taiwan empirical formula.

        This is the accurate formula used in Stage 2.

        Args:
            amplitude: Combined WA-simulated amplitude (RMS of E and N)
            dist: Epicentral distance in km
            depth: Depth in km

        Returns:
            Local magnitude (Ml)
        """
        if amplitude <= 0:
            raise ValueError("Amplitude must be positive")

        nloga = math.log10(amplitude)
        R = math.sqrt(dist**2 + depth**2)

        if depth <= 35:
            if 0 <= dist <= 80:
                # Case 2: Shallow, near
                dA = -0.00716 * R - math.log10(R) - 0.39
            else:
                # Case 1: Shallow, far
                dA = -0.00261 * R - 0.83 * math.log10(R) - 1.07
        else:
            # Case 3: Deep
            dA = -0.00326 * R - 0.83 * math.log10(R) - 1.01

        return nloga - dA

    @property
    def stats(self) -> dict:
        """Get estimator statistics."""
        return {
            'total_magnitudes': self._magnitude_count,
            'stations_available': len(self._station_coords),
            'use_wa_simulation': self.use_wa_simulation,
            'amplitude_table': self.amplitude_table.stats,
        }