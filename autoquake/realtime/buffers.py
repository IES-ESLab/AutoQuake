"""
Buffer classes for realtime pick and waveform management.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class PickBuffer:
    """
    Sliding window buffer for seismic phase picks.

    This buffer manages incoming picks and triggers association when conditions are met.
    It supports:
    - Time-based windowing with configurable overlap
    - Pick recycling for unassociated picks
    - Automatic cleanup of old picks

    Attributes:
        window_size: Duration of the sliding window
        overlap: Overlap ratio between consecutive windows (0.0-0.9)
        min_picks_to_trigger: Minimum number of picks required to trigger association
        max_age: Maximum age for picks in buffer before cleanup
    """

    window_size: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    overlap: float = 0.5
    min_picks_to_trigger: int = 10
    max_age: timedelta = field(default_factory=lambda: timedelta(seconds=120))

    # Internal state
    picks: deque = field(default_factory=deque)
    recycled_picks: list = field(default_factory=list)
    last_trigger_time: datetime | None = None
    _pick_count: int = field(default=0)

    def add_pick(self, pick: dict) -> None:
        """
        Add a single pick to the buffer.

        Args:
            pick: Dictionary containing pick information with 'phase_time' key
        """
        # Ensure phase_time is datetime
        if isinstance(pick.get('phase_time'), str):
            pick = pick.copy()
            pick['phase_time'] = pd.to_datetime(pick['phase_time'])

        self.picks.append(pick)
        self._pick_count += 1
        self._cleanup_old_picks()

    def add_picks_batch(self, picks_df: pd.DataFrame) -> None:
        """
        Add multiple picks from a DataFrame.

        Args:
            picks_df: DataFrame with pick information
        """
        for _, row in picks_df.iterrows():
            self.add_pick(row.to_dict())

    def should_trigger(self) -> bool:
        """
        Check if association should be triggered.

        Returns True when:
        1. There are enough picks (>= min_picks_to_trigger)
        2. AND the time span of picks >= window_size

        Returns:
            True if association should be triggered
        """
        total_picks = len(self.picks) + len(self.recycled_picks)
        if total_picks < self.min_picks_to_trigger:
            return False

        if not self.picks:
            return False

        oldest = self.picks[0]['phase_time']
        newest = self.picks[-1]['phase_time']

        # Check time window
        if (newest - oldest) >= self.window_size:
            # Check overlap timing
            if self.last_trigger_time is not None:
                overlap_duration = self.window_size * self.overlap
                if (newest - self.last_trigger_time) < (self.window_size - overlap_duration):
                    return False
            return True

        return False

    def get_window(self) -> pd.DataFrame:
        """
        Get current window of picks for association.

        This method:
        1. Combines current picks with recycled picks
        2. Clears the recycled picks list
        3. Updates the last trigger time

        Returns:
            DataFrame with all picks in the current window
        """
        all_picks = list(self.picks) + self.recycled_picks
        self.recycled_picks = []  # Clear recycled picks

        if self.picks:
            self.last_trigger_time = self.picks[-1]['phase_time']

        df = pd.DataFrame(all_picks)
        if not df.empty and 'phase_time' in df.columns:
            df = df.sort_values('phase_time').reset_index(drop=True)
        return df

    def recycle(self, unassociated_picks: pd.DataFrame) -> None:
        """
        Recycle unassociated picks for the next window.

        Args:
            unassociated_picks: DataFrame of picks that were not associated with events
        """
        if unassociated_picks.empty:
            return

        for _, row in unassociated_picks.iterrows():
            pick_dict = row.to_dict()
            # Check if pick is still within max age
            if isinstance(pick_dict.get('phase_time'), str):
                pick_dict['phase_time'] = pd.to_datetime(pick_dict['phase_time'])

            if self.picks:
                newest = self.picks[-1]['phase_time']
                age = newest - pick_dict['phase_time']
                if age <= self.max_age:
                    self.recycled_picks.append(pick_dict)
            else:
                self.recycled_picks.append(pick_dict)

        logger.debug(f'Recycled {len(self.recycled_picks)} unassociated picks')

    def clear(self) -> None:
        """Clear all picks from the buffer."""
        self.picks.clear()
        self.recycled_picks.clear()
        self.last_trigger_time = None
        self._pick_count = 0

    def _cleanup_old_picks(self) -> None:
        """Remove picks that exceed max age."""
        if not self.picks:
            return

        newest = self.picks[-1]['phase_time']
        cutoff = newest - self.max_age

        while self.picks and self.picks[0]['phase_time'] < cutoff:
            self.picks.popleft()

    @property
    def total_picks(self) -> int:
        """Total number of picks in buffer including recycled."""
        return len(self.picks) + len(self.recycled_picks)

    @property
    def stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'current_picks': len(self.picks),
            'recycled_picks': len(self.recycled_picks),
            'total_processed': self._pick_count,
            'last_trigger': self.last_trigger_time.isoformat() if self.last_trigger_time else None,
        }


@dataclass
class WaveformBuffer:
    """
    Ring buffer for continuous waveform data.

    This buffer manages incoming waveform data from multiple stations
    and provides data for phase picking.

    Attributes:
        duration: Duration of waveform to keep in buffer
        sampling_rate: Sampling rate in Hz
    """

    duration: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    sampling_rate: float = 100.0

    # Internal state: station_id -> (timestamps, data)
    _buffers: dict = field(default_factory=dict)
    _last_update: dict = field(default_factory=dict)

    def add_data(
        self,
        station_id: str,
        timestamp: datetime,
        data: list[float] | None = None,
    ) -> None:
        """
        Add waveform data for a station.

        Args:
            station_id: Station identifier
            timestamp: Timestamp of the data
            data: Waveform samples (3-component: [E, N, Z])
        """
        if station_id not in self._buffers:
            self._buffers[station_id] = {
                'timestamps': deque(),
                'data': deque(),
            }

        buffer = self._buffers[station_id]
        buffer['timestamps'].append(timestamp)
        buffer['data'].append(data)
        self._last_update[station_id] = datetime.now()

        # Cleanup old data
        self._cleanup_station(station_id)

    def get_window(
        self,
        station_ids: list[str] | None = None,
        duration: timedelta | None = None,
    ) -> dict:
        """
        Get waveform data for specified stations.

        Args:
            station_ids: List of station IDs (None for all)
            duration: Duration to retrieve (None for full buffer)

        Returns:
            Dictionary mapping station_id to waveform data
        """
        if station_ids is None:
            station_ids = list(self._buffers.keys())

        result = {}
        for station_id in station_ids:
            if station_id in self._buffers:
                buffer = self._buffers[station_id]
                timestamps = list(buffer['timestamps'])
                data = list(buffer['data'])

                if duration and timestamps:
                    cutoff = timestamps[-1] - duration
                    valid_indices = [i for i, t in enumerate(timestamps) if t >= cutoff]
                    timestamps = [timestamps[i] for i in valid_indices]
                    data = [data[i] for i in valid_indices]

                result[station_id] = {
                    'timestamps': timestamps,
                    'data': data,
                }

        return result

    def _cleanup_station(self, station_id: str) -> None:
        """Remove old data from a station buffer."""
        if station_id not in self._buffers:
            return

        buffer = self._buffers[station_id]
        if not buffer['timestamps']:
            return

        newest = buffer['timestamps'][-1]
        cutoff = newest - self.duration

        while buffer['timestamps'] and buffer['timestamps'][0] < cutoff:
            buffer['timestamps'].popleft()
            buffer['data'].popleft()

    def clear(self) -> None:
        """Clear all waveform data."""
        self._buffers.clear()
        self._last_update.clear()

    @property
    def station_count(self) -> int:
        """Number of stations in buffer."""
        return len(self._buffers)

    @property
    def stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'stations': list(self._buffers.keys()),
            'station_count': len(self._buffers),
            'last_updates': {k: v.isoformat() for k, v in self._last_update.items()},
        }
