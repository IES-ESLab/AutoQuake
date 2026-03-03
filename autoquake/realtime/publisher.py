"""
JSON message publisher for realtime earthquake alerts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

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

    def publish_preliminary(self, event: dict) -> bool:
        """
        Publish preliminary (rapid) event notification.

        Args:
            event: Dictionary containing event information from GaMMA

        Returns:
            True if publishing succeeded
        """
        event_id = self._generate_event_id(event)

        message = {
            'event_id': event_id,
            'type': 'preliminary',
            'origin_time': self._format_time(event.get('time') or event.get('timestamp')),
            'latitude': round(event.get('latitude', 0), 4),
            'longitude': round(event.get('longitude', 0), 4),
            'depth_km': round(event.get('depth_km', 0), 2),
            'num_picks': event.get('num_picks', 0),
            'num_p_picks': event.get('num_p_picks', 0),
            'num_s_picks': event.get('num_s_picks', 0),
            'quality': {
                'sigma_time': round(event.get('sigma_time', event.get('sigma11', 0)), 3),
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }

        success = self._send(message)
        if success:
            logger.info(f"Published preliminary event: {event_id}")
        return success

    def publish_update(
        self,
        event: dict,
        magnitude: dict | None = None,
        focal_mechanism: dict | None = None,
        relocation: dict | None = None,
    ) -> bool:
        """
        Publish event update with additional information.

        Args:
            event: Dictionary containing event information
            magnitude: Magnitude information (ml, num_stations, etc.)
            focal_mechanism: Focal mechanism solution (strike, dip, rake)
            relocation: Relocated hypocenter information

        Returns:
            True if publishing succeeded
        """
        event_id = event.get('event_id') or self._generate_event_id(event)

        # Use relocated position if available
        lat = relocation.get('latitude', event.get('latitude', 0)) if relocation else event.get('latitude', 0)
        lon = relocation.get('longitude', event.get('longitude', 0)) if relocation else event.get('longitude', 0)
        depth = relocation.get('depth_km', event.get('depth_km', 0)) if relocation else event.get('depth_km', 0)
        origin_time = relocation.get('time', event.get('time')) if relocation else event.get('time')

        message = {
            'event_id': event_id,
            'type': 'update',
            'origin_time': self._format_time(origin_time),
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'depth_km': round(depth, 2),
            'magnitude': self._format_magnitude(magnitude),
            'focal_mechanism': self._format_focal_mechanism(focal_mechanism),
            'relocated': relocation is not None,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }

        success = self._send(message)
        if success:
            logger.info(f"Published update for event: {event_id}")
        return success

    def _send(self, message: dict) -> bool:
        """
        Send message via HTTP and/or file.

        Args:
            message: Message dictionary to send

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
                logger.debug(f"HTTP POST successful: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP POST failed: {e}")
                self._failed_count += 1

        # File output
        if self.output_dir:
            try:
                filename = f"{message['event_id']}_{message['type']}.json"
                filepath = self.output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(message, f, indent=2, ensure_ascii=False)
                success = True
                logger.debug(f"Saved to file: {filepath}")
            except OSError as e:
                logger.error(f"File write failed: {e}")
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

    def _format_magnitude(self, magnitude: dict | None) -> dict | None:
        """Format magnitude information."""
        if magnitude is None:
            return None
        return {
            'ml': round(magnitude.get('ml', 0), 2) if magnitude.get('ml') is not None else None,
            'num_stations': magnitude.get('num_stations', 0),
            'uncertainty': round(magnitude.get('uncertainty', 0), 2) if magnitude.get('uncertainty') is not None else None,
        }

    def _format_focal_mechanism(self, focal: dict | None) -> dict | None:
        """Format focal mechanism information."""
        if focal is None:
            return None
        return {
            'strike': round(focal.get('strike', 0), 1),
            'dip': round(focal.get('dip', 0), 1),
            'rake': round(focal.get('rake', 0), 1),
            'quality': focal.get('quality', 'unknown'),
            'num_polarities': focal.get('num_polarities', 0),
        }

    @property
    def stats(self) -> dict:
        """Get publisher statistics."""
        return {
            'messages_sent': self._message_count,
            'failed_count': self._failed_count,
            'endpoint': self.endpoint,
            'output_dir': str(self.output_dir) if self.output_dir else None,
        }
