"""
Main entry point for realtime earthquake detection system.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .associator import RealtimeGaMMA
from .buffers import PickBuffer
from .config import RealtimeConfig
from .event_validator import EventValidator
from .publisher import JSONPublisher
from .simulators import PickStreamSimulator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RealtimeRunner:
    """
    Main runner for the realtime earthquake detection system.

    This class orchestrates all components:
    - Pick buffer management
    - GaMMA association
    - Event publishing

    Can run with simulated data (for testing) or real data sources.
    """

    def __init__(
        self,
        config: RealtimeConfig,
    ):
        """
        Initialize the realtime runner.

        Args:
            config: RealtimeConfig instance
        """
        self.config = config
        self._running = False
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all components."""
        # Pick buffer
        self.pick_buffer = PickBuffer(
            window_size=timedelta(seconds=self.config.pick_window_seconds),
            overlap=self.config.pick_overlap,
            min_picks_to_trigger=self.config.min_picks_to_trigger,
            max_age=timedelta(seconds=self.config.max_pick_age_seconds),
        )

        # Publisher
        self.publisher = JSONPublisher(
            endpoint=self.config.publisher_endpoint,
            output_dir=self.config.publisher_output_dir,
        )

        # Validator
        self.validator = EventValidator(
            min_picks=self.config.validation_min_picks,
            max_residual=self.config.validation_max_residual,
        )

        # GaMMA associator
        self.associator = RealtimeGaMMA(
            station=self.config.station_file,
            result_path=self.config.result_path,
            pick_buffer=self.pick_buffer,
            publisher=self.publisher,
            validator=self.validator,
            center=self.config.center,
            xlim_degree=self.config.xlim_degree,
            ylim_degree=self.config.ylim_degree,
            zlim=self.config.zlim,
            method=self.config.method,
            use_amplitude=self.config.use_amplitude,
            vp=self.config.vp,
            vs=self.config.vs,
            ncpu=self.config.ncpu,
            min_picks_per_eq=self.config.min_picks_per_eq,
            min_p_picks_per_eq=self.config.min_p_picks_per_eq,
            min_s_picks_per_eq=self.config.min_s_picks_per_eq,
            max_sigma11=self.config.max_sigma11,
        )

        logger.info('Realtime system components initialized')

    def run_simulation(
        self,
        picks_file: Path,
        speed: float | None = None,
        max_duration: float | None = None,
        poll_interval: float = 0.1,
    ) -> None:
        """
        Run with simulated pick data.

        Args:
            picks_file: Path to picks CSV file
            speed: Simulation speed (None to use config value)
            max_duration: Maximum duration in seconds
            poll_interval: Polling interval in seconds
        """
        speed = speed or self.config.simulator_speed

        logger.info(f'Starting simulation with {picks_file} at {speed}x speed')

        simulator = PickStreamSimulator(
            picks_file=picks_file,
            speed=speed,
        )

        self._running = True
        self._setup_signal_handlers()

        simulator.start()
        start_time = time.time()

        try:
            while self._running and simulator.is_running:
                # Get available picks from simulator
                picks = simulator.get_available_picks()
                for pick in picks:
                    self.pick_buffer.add_pick(pick)

                # Check if we should trigger association
                events = self.associator.check_and_process()
                if events:
                    logger.info(f'Detected {len(events)} events')

                # Check duration limit
                if max_duration and (time.time() - start_time) > max_duration:
                    logger.info(f'Reached max duration of {max_duration}s')
                    break

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info('Interrupted by user')
        finally:
            self._running = False
            simulator.stop()
            self._save_results()

    def run_realtime(
        self,
        poll_interval: float = 0.1,
    ) -> None:
        """
        Run in realtime mode (for future SEEDLINK integration).

        Args:
            poll_interval: Polling interval in seconds
        """
        logger.info('Starting realtime mode')
        self._running = True
        self._setup_signal_handlers()

        try:
            while self._running:
                # TODO: Get picks from SEEDLINK or other source
                # For now, just poll the buffer
                events = self.associator.check_and_process()
                if events:
                    logger.info(f'Detected {len(events)} events')

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info('Interrupted by user')
        finally:
            self._running = False
            self._save_results()

    def add_pick(self, pick: dict) -> None:
        """
        Add a single pick to the system.

        This method can be called from external sources (e.g., SEEDLINK callback).

        Args:
            pick: Pick dictionary
        """
        self.pick_buffer.add_pick(pick)

    def process_once(self) -> list[dict]:
        """
        Process current buffer once.

        Useful for manual testing or external control.

        Returns:
            List of detected events
        """
        return self.associator.check_and_process()

    def stop(self) -> None:
        """Stop the runner."""
        self._running = False
        logger.info('Runner stopped')

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handler(signum, frame):
            logger.info(f'Received signal {signum}')
            self.stop()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _save_results(self) -> None:
        """Save results before shutdown."""
        try:
            catalog_path = self.associator.save_catalog()
            logger.info(f'Results saved to {catalog_path}')
        except Exception as e:
            logger.error(f'Failed to save results: {e}')

    @property
    def stats(self) -> dict:
        """Get runner statistics."""
        return {
            'running': self._running,
            'associator': self.associator.stats,
        }


def run_simulation_from_config(
    config_path: Path | None = None,
    picks_file: Path | None = None,
    station_file: Path | None = None,
    result_path: Path | None = None,
    speed: float = 1.0,
    max_duration: float | None = None,
) -> None:
    """
    Convenience function to run simulation.

    Args:
        config_path: Path to config JSON (optional)
        picks_file: Path to picks CSV
        station_file: Path to station CSV
        result_path: Output directory
        speed: Simulation speed
        max_duration: Maximum duration in seconds
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Create config
    if config_path:
        import json
        with open(config_path) as f:
            config_dict = json.load(f)
        config = RealtimeConfig(**config_dict)
    else:
        if not all([picks_file, station_file, result_path]):
            raise ValueError('Must provide either config_path or all of picks_file, station_file, result_path')

        config = RealtimeConfig(
            station_file=station_file,
            result_path=result_path,
            simulator_speed=speed,
        )

    # Run
    runner = RealtimeRunner(config)
    runner.run_simulation(
        picks_file=picks_file,
        speed=speed,
        max_duration=max_duration,
    )


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Run realtime earthquake detection')
    parser.add_argument('--picks', type=Path, required=True, help='Path to picks CSV')
    parser.add_argument('--station', type=Path, required=True, help='Path to station CSV')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--speed', type=float, default=1.0, help='Simulation speed')
    parser.add_argument('--duration', type=float, default=None, help='Max duration in seconds')

    args = parser.parse_args()

    run_simulation_from_config(
        picks_file=args.picks,
        station_file=args.station,
        result_path=args.output,
        speed=args.speed,
        max_duration=args.duration,
    )
