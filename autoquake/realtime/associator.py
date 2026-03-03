"""
Realtime GaMMA association wrapper.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..associator import GaMMA
from .buffers import PickBuffer
from .event_validator import EventValidator
from .publisher import JSONPublisher

if TYPE_CHECKING:
    from .config import RealtimeConfig

logger = logging.getLogger(__name__)


class RealtimeGaMMA:
    """
    Realtime wrapper for GaMMA earthquake association.

    This class wraps the batch GaMMA associator to work with streaming
    pick data. It manages:
    - Pick buffer triggering
    - Event validation
    - JSON publishing
    - Unassociated pick recycling

    Attributes:
        station: Path to station file
        result_path: Directory for output files
        pick_buffer: Buffer for incoming picks
        publisher: JSON message publisher
        validator: Event quality validator
    """

    def __init__(
        self,
        station: Path,
        result_path: Path,
        pick_buffer: PickBuffer,
        publisher: JSONPublisher | None = None,
        validator: EventValidator | None = None,
        **gamma_kwargs,
    ):
        """
        Initialize the realtime GaMMA associator.

        Args:
            station: Path to station CSV file
            result_path: Directory for output files
            pick_buffer: PickBuffer instance for receiving picks
            publisher: JSONPublisher instance (optional)
            validator: EventValidator instance (optional)
            **gamma_kwargs: Additional arguments passed to GaMMA
        """
        self.station = Path(station)
        self.result_path = Path(result_path)
        self.result_path.mkdir(parents=True, exist_ok=True)

        self.pick_buffer = pick_buffer
        self.publisher = publisher or JSONPublisher(output_dir=self.result_path / 'events')
        self.validator = validator or EventValidator()

        self.gamma_kwargs = gamma_kwargs

        # Internal state
        self._processed_events: set = set()
        self._event_counter: int = 0
        self._window_counter: int = 0
        self._all_events: list = []
        self._all_picks: list = []

    @classmethod
    def from_config(
        cls,
        config: RealtimeConfig,
        pick_buffer: PickBuffer,
    ) -> RealtimeGaMMA:
        """
        Create RealtimeGaMMA from configuration.

        Args:
            config: RealtimeConfig instance
            pick_buffer: PickBuffer instance

        Returns:
            Configured RealtimeGaMMA instance
        """
        publisher = JSONPublisher(
            endpoint=config.publisher_endpoint,
            output_dir=config.publisher_output_dir,
        )

        validator = EventValidator(
            min_picks=config.validation_min_picks,
            max_residual=config.validation_max_residual,
        )

        return cls(
            station=config.station_file,
            result_path=config.result_path,
            pick_buffer=pick_buffer,
            publisher=publisher,
            validator=validator,
            center=config.center,
            xlim_degree=config.xlim_degree,
            ylim_degree=config.ylim_degree,
            zlim=config.zlim,
            method=config.method,
            use_amplitude=config.use_amplitude,
            vp=config.vp,
            vs=config.vs,
            ncpu=config.ncpu,
            min_picks_per_eq=config.min_picks_per_eq,
            min_p_picks_per_eq=config.min_p_picks_per_eq,
            min_s_picks_per_eq=config.min_s_picks_per_eq,
            max_sigma11=config.max_sigma11,
        )

    def check_and_process(self) -> list[dict]:
        """
        Check if processing should be triggered and process if so.

        This is the main entry point for the realtime loop.

        Returns:
            List of validated events (empty if no processing or no valid events)
        """
        if not self.pick_buffer.should_trigger():
            return []

        return self.process_window()

    def process_window(self) -> list[dict]:
        """
        Process current window of picks.

        This method:
        1. Gets picks from buffer
        2. Runs GaMMA association
        3. Validates events
        4. Publishes valid events
        5. Recycles unassociated picks

        Returns:
            List of validated event dictionaries
        """
        self._window_counter += 1
        logger.info(f'Processing window {self._window_counter}')

        # Get picks from buffer
        picks_df = self.pick_buffer.get_window()
        if picks_df.empty:
            logger.warning('No picks in window')
            return []

        logger.info(f'Window contains {len(picks_df)} picks')

        # Save picks to temporary file (GaMMA requires file path)
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            dir=self.result_path,
        ) as f:
            # Ensure correct column names
            picks_df_gamma = self._prepare_picks_for_gamma(picks_df)
            picks_df_gamma.to_csv(f.name, index=False)
            temp_picks_path = Path(f.name)

        try:
            # Run GaMMA
            gamma = GaMMA(
                station=self.station,
                pickings=temp_picks_path,
                result_path=self.result_path,
                **self.gamma_kwargs,
            )
            gamma.run_predict()

            # Read results
            events_df = pd.read_csv(gamma.get_events())
            picks_result = pd.read_csv(gamma.get_picks())

        except Exception as e:
            #TODO: 這邊我有點忘記，GaMMA沒有事件的時候會不會丟錯誤，如果不會，
            # 那需要判斷events_df是否為空，然後recycle所有的picks
            logger.error(f'GaMMA processing failed: {e}')
            # Recycle all picks on failure
            self.pick_buffer.recycle(picks_df)
            return []
        finally:
            # Cleanup temp file
            temp_picks_path.unlink(missing_ok=True)

        # Process results
        valid_events = self._process_results(events_df, picks_result)

        # Recycle unassociated picks
        unassociated = picks_result[picks_result['event_index'] < 0]
        if not unassociated.empty:
            self.pick_buffer.recycle(unassociated)
            logger.debug(f'Recycled {len(unassociated)} unassociated picks')

        return valid_events

    def _prepare_picks_for_gamma(self, picks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare picks DataFrame for GaMMA processing.

        Ensures correct column names and formats.

        Args:
            picks_df: Input picks DataFrame

        Returns:
            Formatted DataFrame ready for GaMMA
        """
        df = picks_df.copy()

        # Rename columns if needed
        column_mapping = {
            'id': 'station_id',
            'timestamp': 'phase_time',
            'type': 'phase_type',
            'prob': 'phase_score',
            'amp': 'phase_amplitude',
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Ensure phase_time is string format
        if 'phase_time' in df.columns:
            df['phase_time'] = pd.to_datetime(df['phase_time']).dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

        # Ensure phase_type is uppercase
        if 'phase_type' in df.columns:
            df['phase_type'] = df['phase_type'].str.upper()

        return df

    def _process_results(
        self,
        events_df: pd.DataFrame,
        picks_result: pd.DataFrame,
    ) -> list[dict]:
        """
        Process GaMMA results and validate events.

        Args:
            events_df: Events DataFrame from GaMMA
            picks_result: Picks DataFrame with assignments

        Returns:
            List of validated event dictionaries
        """
        if events_df.empty:
            logger.info('No events detected in window')
            return []

        valid_events = []

        for _, event in events_df.iterrows():
            event_dict = event.to_dict()

            # Add pick counts
            # event_picks = picks_result[picks_result['event_index'] == event_dict.get('event_index', -1)]
            # event_dict['num_picks'] = len(event_picks)
            # event_dict['num_p_picks'] = len(event_picks[event_picks['phase_type'].str.upper() == 'P'])
            # event_dict['num_s_picks'] = len(event_picks[event_picks['phase_type'].str.upper() == 'S'])

            # Validate and check for duplicates
            if self.validator.validate_and_register(event_dict):
                # Assign global event index
                self._event_counter += 1
                event_dict['global_event_index'] = self._event_counter

                valid_events.append(event_dict)
                self._all_events.append(event_dict)

                # Publish
                if self.publisher:
                    self.publisher.publish_preliminary(event_dict)

                logger.info(
                    f"Valid event {self._event_counter}: "
                    f"lat={event_dict.get('latitude', 0):.3f}, "
                    f"lon={event_dict.get('longitude', 0):.3f}, "
                    f"depth={event_dict.get('depth_km', 0):.1f}km, "
                    f"picks={event_dict['num_picks']}"
                )
            else:
                logger.debug(f"Event rejected: {event_dict.get('event_index')}")

        return valid_events

    def get_all_events(self) -> pd.DataFrame:
        """Get all validated events as DataFrame."""
        if not self._all_events:
            return pd.DataFrame()
        return pd.DataFrame(self._all_events)

    def save_catalog(self, output_path: Path | None = None) -> Path:
        """
        Save accumulated event catalog to CSV.

        Args:
            output_path: Output file path (default: result_path/realtime_catalog.csv)

        Returns:
            Path to saved catalog file
        """
        if output_path is None:
            output_path = self.result_path / 'realtime_catalog.csv'

        events_df = self.get_all_events()
        if not events_df.empty:
            events_df.to_csv(output_path, index=False)
            logger.info(f'Saved catalog with {len(events_df)} events to {output_path}')

        return output_path

    def reset(self) -> None:
        """Reset internal state."""
        self._processed_events.clear()
        self._event_counter = 0
        self._window_counter = 0
        self._all_events.clear()
        self._all_picks.clear()
        self.validator.reset()
        self.pick_buffer.clear()

    @property
    def stats(self) -> dict:
        """Get association statistics."""
        return {
            'windows_processed': self._window_counter,
            'total_events': self._event_counter,
            'buffer_stats': self.pick_buffer.stats,
            'validator_stats': self.validator.stats,
            'publisher_stats': self.publisher.stats if self.publisher else None,
        }
