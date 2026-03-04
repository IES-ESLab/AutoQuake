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
    and provides data for phase picking. Supports both raw sample addition
    and obspy Stream ingestion.

    Attributes:
        duration: Duration of waveform to keep in buffer
        sampling_rate: Sampling rate in Hz
        min_samples: Minimum samples required for a valid window
    """

    duration: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    sampling_rate: float = 100.0
    min_samples: int = 1000

    # Internal state: station_id -> numpy array [3, samples]
    _buffers: dict = field(default_factory=dict)
    _start_times: dict = field(default_factory=dict)
    _last_update: dict = field(default_factory=dict)

    def add_stream(self, stream) -> None:
        """
        Add data from an obspy Stream object.

        Args:
            stream: obspy.Stream with waveform data
        """
        import numpy as np
        from collections import defaultdict

        # Group traces by station
        station_traces = defaultdict(list)
        for trace in stream:
            # Extract station ID (network.station.location)
            station_id = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"
            station_traces[station_id].append(trace)

        comp2idx = {'3': 0, '2': 1, '1': 2, 'E': 0, 'N': 1, 'Z': 2}

        for station_id, traces in station_traces.items():
            # Determine time range
            start_time = min(tr.stats.starttime for tr in traces)
            end_time = max(tr.stats.endtime for tr in traces)
            npts = int((end_time - start_time) * self.sampling_rate) + 1

            # Initialize or extend buffer
            if station_id not in self._buffers:
                self._buffers[station_id] = np.zeros((3, npts), dtype=np.float32)
                self._start_times[station_id] = start_time.datetime
            else:
                # Extend existing buffer
                old_data = self._buffers[station_id]
                old_start = self._start_times[station_id]
                new_start = min(old_start, start_time.datetime)

                # Calculate new buffer size
                total_duration = max(
                    (end_time.datetime - new_start).total_seconds(),
                    (old_start - new_start).total_seconds() + old_data.shape[1] / self.sampling_rate
                )
                new_npts = int(total_duration * self.sampling_rate) + 1

                new_buffer = np.zeros((3, new_npts), dtype=np.float32)

                # Copy old data
                old_offset = int((old_start - new_start).total_seconds() * self.sampling_rate)
                if old_offset >= 0 and old_offset < new_npts:
                    end_idx = min(old_offset + old_data.shape[1], new_npts)
                    copy_len = end_idx - old_offset
                    new_buffer[:, old_offset:end_idx] = old_data[:, :copy_len]

                self._buffers[station_id] = new_buffer
                self._start_times[station_id] = new_start

            # Add new traces
            buffer = self._buffers[station_id]
            buffer_start = self._start_times[station_id]

            for trace in traces:
                comp = trace.stats.channel[-1]
                if comp not in comp2idx:
                    continue
                idx = comp2idx[comp]

                # Calculate offset
                trace_start = trace.stats.starttime.datetime
                offset = int((trace_start - buffer_start).total_seconds() * self.sampling_rate)

                if offset < 0:
                    # Trace starts before buffer
                    trace_data = trace.data[-offset:]
                    offset = 0
                else:
                    trace_data = trace.data

                # Copy data
                end_idx = min(offset + len(trace_data), buffer.shape[1])
                copy_len = end_idx - offset
                if copy_len > 0:
                    buffer[idx, offset:end_idx] = trace_data[:copy_len].astype(np.float32)

            self._last_update[station_id] = datetime.now()

        # Cleanup old data
        self._cleanup_all()

    def add_data(
        self,
        station_id: str,
        timestamp: datetime,
        data: list[float] | None = None,
    ) -> None:
        """
        Add waveform data for a station (legacy method).

        Args:
            station_id: Station identifier
            timestamp: Timestamp of the data
            data: Waveform samples (3-component: [E, N, Z])
        """
        import numpy as np

        if data is None:
            return

        if station_id not in self._buffers:
            # Initialize with expected buffer size
            buffer_samples = int(self.duration.total_seconds() * self.sampling_rate)
            self._buffers[station_id] = np.zeros((3, buffer_samples), dtype=np.float32)
            self._start_times[station_id] = timestamp

        # This is a simplified version - for real use, use add_stream
        self._last_update[station_id] = datetime.now()

    def get_window_for_model(
        self,
        window_duration: timedelta | None = None,
    ) -> dict | None:
        """
        Get waveform data formatted for PhaseNet model input.

        Args:
            window_duration: Duration of window to extract (None for full buffer)

        Returns:
            Dictionary with 'data', 'station_id', 'begin_time', 'dt_s' keys,
            or None if insufficient data
        """
        import numpy as np
        import torch

        if not self._buffers:
            return None

        station_ids = sorted(self._buffers.keys())
        if not station_ids:
            return None

        # Determine common time range
        latest_start = max(self._start_times.values())
        window_dur = window_duration or self.duration

        # Find minimum samples across all stations
        min_samples = float('inf')
        for station_id in station_ids:
            buffer = self._buffers[station_id]
            min_samples = min(min_samples, buffer.shape[1])

        if min_samples < self.min_samples:
            logger.debug(f'Insufficient samples: {min_samples} < {self.min_samples}')
            return None

        # Extract window
        window_samples = int(window_dur.total_seconds() * self.sampling_rate)
        window_samples = min(window_samples, int(min_samples))

        nx = len(station_ids)
        nt = window_samples
        data = np.zeros((3, nt, nx), dtype=np.float32)

        for i, station_id in enumerate(station_ids):
            buffer = self._buffers[station_id]
            # Take last window_samples
            data[:, :, i] = buffer[:, -window_samples:]

        # Normalize per station
        for i in range(nx):
            for c in range(3):
                channel_data = data[c, :, i]
                std = np.std(channel_data)
                if std > 0:
                    data[c, :, i] = (channel_data - np.mean(channel_data)) / std

        return {
            'data': torch.from_numpy(data).unsqueeze(0),  # [1, 3, nt, nx]
            'station_id': station_ids,
            'begin_time': latest_start.isoformat(timespec='milliseconds'),
            'dt_s': 1.0 / self.sampling_rate,
            'nt': nt,
            'nx': nx,
        }

    def get_window(
        self,
        station_ids: list[str] | None = None,
        duration: timedelta | None = None,
    ) -> dict:
        """
        Get raw waveform data for specified stations.

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
                start_time = self._start_times.get(station_id)

                if duration and buffer.shape[1] > 0:
                    samples = int(duration.total_seconds() * self.sampling_rate)
                    samples = min(samples, buffer.shape[1])
                    data = buffer[:, -samples:]
                else:
                    data = buffer

                result[station_id] = {
                    'data': data,
                    'start_time': start_time,
                    'sampling_rate': self.sampling_rate,
                }

        return result

    def _cleanup_all(self) -> None:
        """Remove old data from all station buffers."""
        import numpy as np

        max_samples = int(self.duration.total_seconds() * self.sampling_rate)

        for station_id in list(self._buffers.keys()):
            buffer = self._buffers[station_id]
            if buffer.shape[1] > max_samples:
                # Keep only the latest samples
                self._buffers[station_id] = buffer[:, -max_samples:]
                # Update start time
                old_start = self._start_times[station_id]
                removed_samples = buffer.shape[1] - max_samples
                removed_seconds = removed_samples / self.sampling_rate
                self._start_times[station_id] = old_start + timedelta(seconds=removed_seconds)

    def _cleanup_station(self, station_id: str) -> None:
        """Remove old data from a station buffer."""
        # Implemented in _cleanup_all for efficiency
        pass

    def clear(self) -> None:
        """Clear all waveform data."""
        self._buffers.clear()
        self._start_times.clear()
        self._last_update.clear()

    @property
    def station_count(self) -> int:
        """Number of stations in buffer."""
        return len(self._buffers)

    @property
    def stats(self) -> dict:
        """Get buffer statistics."""
        sample_counts = {k: v.shape[1] for k, v in self._buffers.items()}
        return {
            'stations': list(self._buffers.keys()),
            'station_count': len(self._buffers),
            'sample_counts': sample_counts,
            'start_times': {k: v.isoformat() for k, v in self._start_times.items()},
            'last_updates': {k: v.isoformat() for k, v in self._last_update.items()},
        }
