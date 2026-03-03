"""
Event validation for realtime earthquake detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EventValidator:
    """
    Validates earthquake events based on quality criteria.

    This validator checks if detected events meet minimum quality
    requirements before they are published.

    Attributes:
        min_picks: Minimum number of picks required
        max_residual: Maximum allowed time residual (seconds)
        min_stations: Minimum number of unique stations
        duplicate_threshold: Time window for duplicate detection (seconds)
    """

    min_picks: int = 8
    max_residual: float = 2.0
    min_stations: int = 4
    duplicate_threshold: timedelta = field(default_factory=lambda: timedelta(seconds=5))

    # Internal state for duplicate detection
    _recent_events: list = field(default_factory=list)

    def is_valid(self, event: dict) -> bool:
        #TODO: This check is not that useful, but it's a good place to add more in it.
        """
        Check if an event meets quality criteria.

        Args:
            event: Dictionary containing event information

        Returns:
            True if event passes all validation checks
        """
        checks = [
            self._check_min_picks(event),
            self._check_residual(event),
            self._check_min_stations(event),
        ]

        is_valid = all(checks)
        if is_valid:
            logger.debug(f"Event validated: {event.get('event_index', 'unknown')}")
        else:
            logger.debug(f"Event rejected: {event.get('event_index', 'unknown')}")

        return is_valid

    def is_duplicate(self, event: dict) -> bool:
        """
        Check if event is a duplicate of a recent event.

        Args:
            event: Dictionary containing event information

        Returns:
            True if event is likely a duplicate
        """
        event_time = self._parse_time(event.get('time') or event.get('timestamp'))
        if event_time is None:
            return False

        event_lat = event.get('latitude', 0)
        event_lon = event.get('longitude', 0)

        for recent in self._recent_events:
            recent_time = recent['time']
            time_diff = abs((event_time - recent_time).total_seconds())

            if time_diff <= self.duplicate_threshold.total_seconds():
                # Check spatial proximity (rough check)
                lat_diff = abs(event_lat - recent['latitude'])
                lon_diff = abs(event_lon - recent['longitude'])

                if lat_diff < 0.1 and lon_diff < 0.1:
                    logger.debug(f"Duplicate event detected: {event.get('event_index')}")
                    return True

        return False

    def register_event(self, event: dict) -> None:
        """
        Register an event to track for duplicate detection.

        Args:
            event: Dictionary containing event information
        """
        event_time = self._parse_time(event.get('time') or event.get('timestamp'))
        if event_time is None:
            return

        self._recent_events.append({
            'time': event_time,
            'latitude': event.get('latitude', 0),
            'longitude': event.get('longitude', 0),
            'event_index': event.get('event_index'),
        })

        # Cleanup old events
        self._cleanup_recent_events(event_time)

    def validate_and_register(self, event: dict) -> bool:
        """
        Validate event and register if valid.

        Args:
            event: Dictionary containing event information

        Returns:
            True if event is valid and not a duplicate
        """
        if not self.is_valid(event):
            return False

        if self.is_duplicate(event):
            return False

        self.register_event(event)
        return True

    def _check_min_picks(self, event: dict) -> bool:
        """Check minimum pick count."""
        num_picks = event.get('num_picks') or event.get('num_p_picks', 0) + event.get('num_s_picks', 0)
        return num_picks >= self.min_picks

    def _check_residual(self, event: dict) -> bool:
        """Check time residual."""
        residual = event.get('sigma_time') or event.get('sigma11', 999)
        return residual <= self.max_residual

    def _check_min_stations(self, event: dict) -> bool:
        """Check minimum station count."""
        num_stations = event.get('num_stations', 0)
        # If num_stations not available, skip this check
        if num_stations == 0:
            return True
        return num_stations >= self.min_stations

    def _parse_time(self, time_value: Any) -> datetime | None:
        """Parse time value to datetime."""
        if time_value is None:
            return None
        if isinstance(time_value, datetime):
            return time_value
        if isinstance(time_value, str):
            try:
                return datetime.fromisoformat(time_value.replace('Z', '+00:00'))
            except ValueError:
                try:
                    from pandas import to_datetime
                    return to_datetime(time_value).to_pydatetime()
                except Exception:
                    return None
        return None

    def _cleanup_recent_events(self, current_time: datetime) -> None:
        """Remove events older than duplicate threshold."""
        cutoff = current_time - self.duplicate_threshold * 10  # Keep 10x threshold
        self._recent_events = [
            e for e in self._recent_events
            if e['time'] >= cutoff
        ]

    def reset(self) -> None:
        """Reset validator state."""
        self._recent_events.clear()

    @property
    def stats(self) -> dict:
        """Get validator statistics."""
        return {
            'recent_events': len(self._recent_events),
            'min_picks': self.min_picks,
            'max_residual': self.max_residual,
        }
