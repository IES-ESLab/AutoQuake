"""
Main entry point for realtime earthquake detection system.
"""

from __future__ import annotations

import os
import traceback
import logging
import signal
import time
from datetime import timedelta
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from .associator import RealtimeGaMMA
from .buffers import PickBuffer
from .config import RealtimeConfig
from .event_validator import EventValidator
from .focal import RealtimeFocalMechanism
from .magnitude import RealtimeMagnitude
from .publisher import JSONPublisher
from .relocator import RealtimeRelocator
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
        self._all_events: list[dict] = []
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
            realtime=self.config.realtime,
        )

        # Relocator (H3DD)
        self.relocator = None
        if self.config.enable_relocation and self.config.model_3d:
            self.relocator = RealtimeRelocator(
                station_file=self.config.station_file,
                model_3d=self.config.model_3d,
                h3dd_dir=self.config.h3dd_dir,
            )
            logger.info('H3DD relocator initialized')

        # Magnitude estimator
        self.magnitude_estimator = None
        if self.config.enable_magnitude:
            self.magnitude_estimator = RealtimeMagnitude(
                station_info=self.config.station_file,
                pz_dir=self.config.pz_dir,
                use_wa_simulation=self.config.use_wa_simulation,
            )
            logger.info('Magnitude estimator initialized')

        # Focal mechanism estimator
        self.focal_estimator = None
        if self.config.enable_focal:
            self.focal_estimator = RealtimeFocalMechanism(
                gafocal_dir=self.config.gafocal_dir,
                station_info=self.config.station_file,
                min_polarities=self.config.min_polarities,
            )
            logger.info('Focal mechanism estimator initialized')

        logger.info('Realtime system components initialized')

    def _process_event_pipeline(
        self,
        event: dict,
        picks: list[dict],
    ) -> dict:
        """
        Complete event processing pipeline for a single event.

        This method runs all refinement stages in sequence:
        1. Preliminary magnitude (quick estimate from raw amplitudes)
        2. H3DD relocation (precise hypocenter location)
        3. Accurate magnitude (using relocated position)
        4. Focal mechanism (first-motion polarity analysis)
        5. Publish final update

        Args:
            event: Event dictionary from GaMMA association
            picks: Associated picks for this event

        Returns:
            Enhanced event dictionary with all refinement results
        """
        result = event.copy()

        # Relocation (H3DD)
        relocated_event = None
        if self.relocator:
            try:
                relocated, dout_file = self.relocator.relocate_single(event, picks)
                # write into result
                if relocated:
                    # relocated_event = relocated
                    result['latitude_relocated'] = relocated['latitude']
                    result['longitude_relocated'] = relocated['longitude']
                    result['depth_km_relocated'] = relocated['depth_km']
                    result['time_relocated'] = relocated.get('time')
                    #FIXME: This is for temporary usage. Logically this should be the magnitude
                    # estimated by relocated hypocenter, but it's GaMMA's hypocenter.
                    result['magnitude'] = relocated.get('magnitude')
                    logger.debug(
                        f'Relocated: {relocated["latitude"]:.4f}, '
                        f'{relocated["longitude"]:.4f}, '
                        f'{relocated["depth_km"]:.1f}km'
                    )
            except Exception as e:
                logger.warning(f'Relocation failed: {e}')

        # Focal mechanism (GAFocal)
        if self.focal_estimator:
            try:
                focal = self.focal_estimator.estimate(dout_file)
                if focal:
                    result['focal_mechanism'] = focal
                    logger.debug(
                        f'Focal: strike={focal["strike"]:.1f}, '
                        f'dip={focal["dip"]:.1f}, '
                        f'rake={focal["rake"]:.1f}'
                    )
            except Exception as e:
                logger.warning(f'Focal mechanism failed: {e}')

        # Publish update with all refinements
        self.publisher.publish_update(result)
        return result

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

                # Full pipeline: association + refinement
                # This process single event each time!
                events_with_picks = self.associator.check_and_process()
                for event, associated_picks in events_with_picks:
                    logger.info(f'Event associated with {len(associated_picks)}')
                    refined = self._process_event_pipeline(event, associated_picks)
                    self._all_events.append(refined)
                    logger.info(
                        f"Event processed: "
                        f"M{refined.get('magnitude', '?')}, "
                        f"depth={refined.get('depth_km_relocated', refined.get('depth_km', '?'))}km\n"
                    )

                # Check duration limit
                # TODO: This is not logical, earthquake might not happen continuously.
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

                # Full pipeline: association + refinement
                events_with_picks = self.associator.check_and_process()
                for event, associated_picks in events_with_picks:
                    refined = self._process_event_pipeline(event, associated_picks)
                    self._all_events.append(refined)
                    logger.info(
                        f"Event processed: "
                        f"M{refined.get('magnitude', '?')}, "
                        f"depth={refined.get('depth_km_relocated', refined.get('depth_km', '?'))}km"
                    )

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
        Process current buffer once with full pipeline.

        Useful for manual testing or external control.

        Returns:
            List of fully processed events (with refinements applied)
        """
        events_with_picks = self.associator.check_and_process()
        processed_events = []
        for event, associated_picks in events_with_picks:
            refined = self._process_event_pipeline(event, associated_picks)
            self._all_events.append(refined)
            processed_events.append(refined)
        return processed_events

    def stop(self) -> None:
        """Stop the runner."""
        self._running = False
        logger.info('Runner stopped')

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handler(signum, _frame):
            logger.info(f'Received signal {signum} in PID {os.getpid()}')
            logger.info(f'Parent PID: {os.getppid()}')
            logger.info('Traceback:\n' + ''.join(traceback.format_stack(_frame)))
            self.stop()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _save_results(self) -> None:
        """Save results before shutdown."""
        try:
            events_df = pd.DataFrame(self._all_events)
            if not events_df.empty:
                catalog_path = Path(self.config.result_path) / 'rt_catalog.csv'
                events_df.to_csv(catalog_path, index=False)
            logger.info(f'Results saved to {catalog_path}')
        except Exception as e:
            logger.error(f'Failed to save results: {e}')

    @property
    def all_events(self) -> list[dict]:
        """Get all processed events."""
        return self._all_events.copy()

    @property
    def stats(self) -> dict:
        """Get runner statistics."""
        stats = {
            'running': self._running,
            'total_events_processed': len(self._all_events),
            'associator': self.associator.stats,
        }

        # Refinement component stats
        if self.relocator:
            stats['relocator'] = self.relocator.stats
        if self.magnitude_estimator:
            stats['magnitude'] = self.magnitude_estimator.stats
        if self.focal_estimator:
            stats['focal'] = self.focal_estimator.stats

        return stats


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
