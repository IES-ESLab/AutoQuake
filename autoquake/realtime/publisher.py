"""
JSON message publisher for realtime earthquake alerts.

Supports three message types:
1. add_event - After GaMMA association (new event with initial location and picks)
2. update_location - After H3DD relocation (updated location with pick-level info)
3. update_focal - After GAfocal (focal mechanism determination)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class JSONPublisher:
    """
    Publishes earthquake event messages via HTTP POST and/or local files.

    This publisher handles both preliminary (rapid) notifications and
    updates with additional information (magnitude, focal mechanism).

    Attributes:
        endpoint: HTTP endpoint URL for POST requests
        output_dir: Local directory for saving JSON files
        timeout: HTTP request timeout in seconds
    """

    def __init__(
        self,
        endpoint: str | None = None,
        output_dir: Path | None = None,
        timeout: float = 5.0,
    ):
        """
        Initialize the JSON publisher.

        Args:
            endpoint: HTTP endpoint URL (None to disable HTTP publishing)
            output_dir: Directory for saving JSON files (None to disable file saving)
            timeout: HTTP request timeout in seconds
        """
        self.endpoint = endpoint
        self.output_dir = Path(output_dir) if output_dir else None
        self.timeout = timeout

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._message_count = 0
        self._failed_count = 0

    def publish_add_event(
        self,
        event: dict,
        picks: list[dict],
    ) -> bool:
        """
        Publish new event after GaMMA association.

        This is the first message sent when a new event is detected.
        Contains initial location and associated picks with phase info.

        Args:
            event: Event dictionary from GaMMA association
            picks: List of associated pick dictionaries

        Returns:
            True if publishing succeeded
        """
        event_id = event.get('event_id') or self._generate_event_id(event)

        # Build associated_picks structure grouped by station
        associated_picks = self._build_picks_for_add_event(picks)

        # Count picks
        num_p = sum(1 for p in picks if p.get('phase_type', '').upper() == 'P')
        num_s = sum(1 for p in picks if p.get('phase_type', '').upper() == 'S')

        message = {
            'add_event': {
                'event_id': event_id,
                'event_time': self._format_time(
                    event.get('time') or event.get('timestamp')
                ),
                'longitude': round(event.get('longitude', 0), 3),
                'latitude': round(event.get('latitude', 0), 3),
                'depth_km': round(event.get('depth_km', 0), 3),
                'magnitude': None,  # Not available at this stage
                'num_picks': len(picks),
                'num_p_picks': num_p,
                'num_s_picks': num_s,
                'associated_picks': associated_picks,
            }
        }

        success = self._send(message, msg_type='add_event', event_id=event_id)
        if success:
            logger.info(f'Published add_event: {event_id} with {len(picks)} picks')
        return success

    def publish_update_location(
        self,
        event_id: str | int,
        relocated_event: dict,
        picks_with_info: list[dict] | pd.DataFrame,
    ) -> bool:
        """
        Publish location update after H3DD relocation.

        Contains updated hypocenter and pick-level information
        (distance, azimuth, takeoff angle, station magnitude).

        Args:
            event_id: Event identifier
            relocated_event: Relocated event dictionary with lat/lon/depth/magnitude
            picks_with_info: Picks with distance_km, azimuth, takeoff_angle info

        Returns:
            True if publishing succeeded
        """
        # Convert DataFrame to list if needed
        if isinstance(picks_with_info, pd.DataFrame):
            picks_list = picks_with_info.to_dict('records')
        else:
            picks_list = picks_with_info

        # Build associated_picks structure with location info
        associated_picks = self._build_picks_for_update_location(picks_list)

        message = {
            'update_location': {
                'event_id': event_id,
                'longitude': round(relocated_event.get('longitude', 0), 3),
                'latitude': round(relocated_event.get('latitude', 0), 3),
                'depth_km': round(relocated_event.get('depth_km', 0), 1),
                'magnitude': round(relocated_event.get('magnitude', 0), 2)
                if relocated_event.get('magnitude') is not None
                else None,
                'associated_picks': associated_picks,
            }
        }

        success = self._send(message, msg_type='update_location', event_id=event_id)
        if success:
            logger.info(
                f'Published update_location: {event_id} at '
                f'{relocated_event.get("latitude"):.3f}, '
                f'{relocated_event.get("longitude"):.3f}'
            )
        return success

    def publish_update_focal(
        self,
        event_id: str | int,
        focal: dict,
    ) -> bool:
        """
        Publish focal mechanism after GAfocal determination.

        Args:
            event_id: Event identifier
            focal: Focal mechanism dictionary with strike, dip, rake, errors, etc.

        Returns:
            True if publishing succeeded
        """
        message = {
            'update_focal': {
                'event_id': event_id,
                'strike': int(focal.get('strike', 0)),
                'strike_err': int(focal.get('strike_err', 0)),
                'dip': int(focal.get('dip', 0)),
                'dip_err': int(focal.get('dip_err', 0)),
                'rake': int(focal.get('rake', 0)),
                'rake_err': int(focal.get('rake_err', 0)),
                'quality_index': int(focal.get('quality_index', 0)),
                'num_of_polarity': int(focal.get('num_of_polarity', 0)),
            }
        }

        success = self._send(message, msg_type='update_focal', event_id=event_id)
        if success:
            logger.info(
                f'Published update_focal: {event_id} - '
                f'strike={focal.get("strike")}, dip={focal.get("dip")}, '
                f'rake={focal.get("rake")}'
            )
        return success

    def _build_picks_for_add_event(self, picks: list[dict]) -> dict:
        """
        Build associated_picks structure for add_event message.

        Groups picks by station with phase_time, phase_score, and polarity (for P).
        """
        associated = {}

        # Polarity mapping to standardize output format
        polarity_map = {
            'U': '+',
            'D': '-',
            '+': '+',
            '-': '-',
            'x': 'x',
            ' ': 'x',
        }

        for pick in picks:
            station = pick.get('station_id', 'UNKNOWN')
            phase_type = pick.get('phase_type', 'P').upper()

            if station not in associated:
                associated[station] = {}

            phase_info = {
                'phase_time': self._format_time(pick.get('phase_time')),
                'phase_score': round(pick.get('phase_score', 0), 3),
            }

            # Add polarity for P waves if available
            if phase_type == 'P' and pick.get('polarity'):
                raw_polarity = pick['polarity']
                phase_info['polarity'] = polarity_map.get(raw_polarity, 'x')

            associated[station][phase_type] = phase_info

        return associated

    def _build_picks_for_update_location(self, picks: list[dict]) -> dict:
        """
        Build associated_picks structure for update_location message.

        Groups picks by station with distance_km, azimuth, takeoff_angle, magnitude.
        """
        associated = {}

        for pick in picks:
            station = pick.get('station_id', 'UNKNOWN')
            phase_type = pick.get('phase_type', 'P').upper()

            if station not in associated:
                associated[station] = {}

            phase_info = {
                'distance_km': round(pick.get('dist', pick.get('distance_km', 0)), 1),
                'azimuth': int(pick.get('azimuth', 0)),
                'takeoff_angle': int(pick.get('takeoff_angle', 0)),
            }

            # Add station magnitude if available
            if pick.get('magnitude') is not None:
                phase_info['magnitude'] = round(pick['magnitude'], 6)

            associated[station][phase_type] = phase_info

        return associated

    def _send(
        self,
        message: dict,
        msg_type: str = 'unknown',
        event_id: str | int = 'unknown',
    ) -> bool:
        """
        Send message via HTTP and/or file.

        Args:
            message: Message dictionary to send
            msg_type: Message type (add_event, update_location, update_focal)
            event_id: Event identifier for file naming

        Returns:
            True if at least one method succeeded
        """
        success = False

        # HTTP POST
        if self.endpoint:
            try:
                response = requests.post(
                    self.endpoint,
                    json=message,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'},
                )
                response.raise_for_status()
                success = True
                logger.debug(f'HTTP POST successful: {response.status_code}')
            except requests.exceptions.RequestException as e:
                logger.error(f'HTTP POST failed: {e}')
                self._failed_count += 1

        # File output
        if self.output_dir:
            try:
                filename = f'{event_id}_{msg_type}.json'
                filepath = self.output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(message, f, indent=2, ensure_ascii=False)
                success = True
                logger.debug(f'Saved to file: {filepath}')
            except OSError as e:
                logger.error(f'File write failed: {e}')
                self._failed_count += 1

        if success:
            self._message_count += 1

        return success

    def _generate_event_id(self, event: dict) -> str:
        """Generate a unique event ID."""
        time_value = event.get('time') or event.get('timestamp')
        if time_value is None:
            return f"evt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        if isinstance(time_value, datetime):
            time_str = time_value.strftime('%Y%m%d_%H%M%S')
        else:
            time_str = str(time_value).replace(':', '').replace('-', '').replace('T', '_')[:15]

        event_idx = event.get('event_index', 0)
        return f"evt_{time_str}_{event_idx:03d}"

    def _format_time(self, time_value: Any) -> str | None:
        """Format time value to ISO string."""
        if time_value is None:
            return None
        if isinstance(time_value, datetime):
            return time_value.isoformat() + 'Z'
        if isinstance(time_value, str):
            if not time_value.endswith('Z') and '+' not in time_value:
                return time_value + 'Z'
            return time_value
        return str(time_value)

    @property
    def stats(self) -> dict:
        """Get publisher statistics."""
        return {
            'messages_sent': self._message_count,
            'failed_count': self._failed_count,
            'endpoint': self.endpoint,
            'output_dir': str(self.output_dir) if self.output_dir else None,
        }
