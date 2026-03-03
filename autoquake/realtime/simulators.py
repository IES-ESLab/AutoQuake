"""
Simulators for testing realtime system with existing data.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:
    from .buffers import PickBuffer

logger = logging.getLogger(__name__)


class PickStreamSimulator:
    """
    Simulates realtime pick streaming from existing CSV data.

    This class reads picks from a CSV file and replays them as if they were
    arriving in real-time. Useful for testing the realtime system with
    historical data.

    Attributes:
        picks_file: Path to the picks CSV file
        speed: Simulation speed multiplier (1.0 = real-time, 10.0 = 10x faster)
        loop: Whether to loop the data when finished
    """

    def __init__(
        self,
        picks_file: Path,
        speed: float = 1.0,
        loop: bool = False,
        on_pick: Callable[[dict], None] | None = None,
    ):
        """
        Initialize the pick stream simulator.

        Args:
            picks_file: Path to CSV file with pick data
            speed: Simulation speed multiplier
            loop: Whether to restart from beginning when data is exhausted
            on_pick: Callback function called for each pick
        """
        self.picks_file = Path(picks_file)
        self.speed = speed
        self.loop = loop
        self.on_pick = on_pick

        self._picks_df: pd.DataFrame | None = None
        self._current_index: int = 0
        self._start_time: datetime | None = None
        self._data_start_time: datetime | None = None
        self._running: bool = False

    def load(self) -> None:
        """Load and prepare picks data."""
        logger.info(f'Loading picks from {self.picks_file}')
        self._picks_df = pd.read_csv(self.picks_file)

        # Ensure phase_time is datetime
        if 'phase_time' in self._picks_df.columns:
            self._picks_df['phase_time'] = pd.to_datetime(self._picks_df['phase_time'])
            self._picks_df = self._picks_df.sort_values('phase_time').reset_index(drop=True)
            self._data_start_time = self._picks_df['phase_time'].iloc[0]

        logger.info(f'Loaded {len(self._picks_df)} picks')

    def start(self) -> None:
        """Start the simulation."""
        if self._picks_df is None:
            self.load()

        self._current_index = 0
        self._start_time = datetime.now()
        self._running = True
        logger.info(f'Simulation started at speed {self.speed}x')

    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False
        logger.info('Simulation stopped')

    def get_elapsed_simulation_time(self) -> timedelta:
        """Get elapsed time in simulation (accounting for speed)."""
        if self._start_time is None:
            return timedelta(0)
        real_elapsed = datetime.now() - self._start_time
        return real_elapsed * self.speed

    def get_available_picks(self) -> list[dict]:
        """
        Get picks that should have arrived by now.

        Returns:
            List of pick dictionaries that are ready to be processed
        """
        if not self._running or self._picks_df is None:
            return []

        sim_elapsed = self.get_elapsed_simulation_time()
        sim_current_time = self._data_start_time + sim_elapsed

        available = []
        while self._current_index < len(self._picks_df):
            pick_time = self._picks_df.iloc[self._current_index]['phase_time']

            if pick_time <= sim_current_time:
                pick_dict = self._picks_df.iloc[self._current_index].to_dict()
                available.append(pick_dict)
                self._current_index += 1

                if self.on_pick:
                    self.on_pick(pick_dict)
            else:
                break

        # Handle looping
        if self._current_index >= len(self._picks_df):
            if self.loop:
                logger.info('Looping simulation from beginning')
                self._current_index = 0
                self._start_time = datetime.now()
            else:
                self._running = False
                logger.info('Simulation finished')

        return available

    def stream_to_buffer(
        self,
        buffer: PickBuffer,
        poll_interval: float = 0.1,
        max_duration: float | None = None,
    ) -> None:
        """
        Stream picks to a buffer continuously.

        Args:
            buffer: PickBuffer to receive picks
            poll_interval: Time between polling in seconds
            max_duration: Maximum duration to run in seconds (None for unlimited)
        """
        self.start()
        start = time.time()

        try:
            while self._running:
                picks = self.get_available_picks()
                for pick in picks:
                    buffer.add_pick(pick)

                if max_duration and (time.time() - start) > max_duration:
                    logger.info(f'Reached max duration of {max_duration}s')
                    break

                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info('Simulation interrupted by user')
        finally:
            self.stop()

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running

    @property
    def progress(self) -> float:
        """Get simulation progress as a percentage."""
        if self._picks_df is None or len(self._picks_df) == 0:
            return 0.0
        return (self._current_index / len(self._picks_df)) * 100

    @property
    def stats(self) -> dict:
        """Get simulation statistics."""
        return {
            'total_picks': len(self._picks_df) if self._picks_df is not None else 0,
            'processed_picks': self._current_index,
            'progress_percent': self.progress,
            'speed': self.speed,
            'is_running': self._running,
            'elapsed_simulation_time': str(self.get_elapsed_simulation_time()),
        }


class WaveformStreamSimulator:
    """
    Simulates realtime waveform streaming.

    This is a placeholder for future implementation when waveform
    streaming is needed.
    """

    def __init__(
        self,
        data_dir: Path,
        speed: float = 1.0,
    ):
        """
        Initialize the waveform stream simulator.

        Args:
            data_dir: Directory containing waveform files
            speed: Simulation speed multiplier
        """
        self.data_dir = Path(data_dir)
        self.speed = speed
        # TODO: Implement waveform streaming simulation
        raise NotImplementedError('WaveformStreamSimulator not yet implemented')
