"""
SEEDLINK client for realtime waveform streaming.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .buffers import WaveformBuffer

logger = logging.getLogger(__name__)


@dataclass
class StationConfig:
    """Configuration for a single station."""

    network: str
    station: str
    location: str = ''
    channels: list[str] = field(default_factory=lambda: ['HH?', 'BH?', 'EH?'])

    @property
    def selector(self) -> str:
        """Get SEEDLINK selector string."""
        return f'{self.network}_{self.station}'

    def __str__(self) -> str:
        return f'{self.network}.{self.station}.{self.location}'


class SEEDLINKClient:
    """
    SEEDLINK client for receiving realtime waveform data.

    This client connects to a SEEDLINK server and streams waveform
    data to a WaveformBuffer.

    Attributes:
        server: SEEDLINK server address (host:port)
        stations: List of stations to subscribe to
        buffer: WaveformBuffer to receive data
    """

    def __init__(
        self,
        server: str,
        stations: list[StationConfig] | list[dict],
        buffer: WaveformBuffer,
        timeout: float = 30.0,
        reconnect_delay: float = 5.0,
    ):
        """
        Initialize the SEEDLINK client.

        Args:
            server: Server address in 'host:port' format
            stations: List of station configurations
            buffer: WaveformBuffer to receive data
            timeout: Connection timeout in seconds
            reconnect_delay: Delay between reconnection attempts
        """
        self.server = server
        self.buffer = buffer
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay

        # Convert dict configs to StationConfig
        self.stations = []
        for s in stations:
            if isinstance(s, dict):
                self.stations.append(StationConfig(**s))
            else:
                self.stations.append(s)

        self._client = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._packet_count = 0
        self._last_packet_time: datetime | None = None
        self._on_data: Callable | None = None

    def connect(self) -> bool:
        """
        Connect to the SEEDLINK server.

        Returns:
            True if connection successful
        """
        try:
            from obspy.clients.seedlink import Client

            logger.info(f'Connecting to SEEDLINK server: {self.server}')
            self._client = Client(self.server, timeout=self.timeout)

            # Add station subscriptions
            for station in self.stations:
                for channel in station.channels:
                    self._client.select_stream(
                        station.network,
                        station.station,
                        channel,
                    )
                logger.debug(f'Subscribed to {station}')

            logger.info(f'Connected to {self.server}, subscribed to {len(self.stations)} stations')
            return True

        except ImportError:
            logger.error('obspy.clients.seedlink not available. Install obspy.')
            return False
        except Exception as e:
            logger.error(f'Failed to connect to SEEDLINK server: {e}')
            return False

    def start(self, on_data: Callable | None = None) -> None:
        """
        Start receiving data in a background thread.

        Args:
            on_data: Optional callback for each received packet
        """
        if self._running:
            logger.warning('Client already running')
            return

        self._on_data = on_data
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info('SEEDLINK client started')

    def stop(self) -> None:
        """Stop receiving data."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        logger.info('SEEDLINK client stopped')

    def _receive_loop(self) -> None:
        """Main receive loop running in background thread."""
        while self._running:
            try:
                if self._client is None:
                    if not self.connect():
                        time.sleep(self.reconnect_delay)
                        continue

                # Get data packets
                stream = self._client.get_waveforms(timeout=5)

                if stream and len(stream) > 0:
                    self._packet_count += len(stream)
                    self._last_packet_time = datetime.now()

                    # Add to buffer
                    self.buffer.add_stream(stream)

                    # Callback
                    if self._on_data:
                        self._on_data(stream)

                    logger.debug(f'Received {len(stream)} traces')

            except Exception as e:
                logger.error(f'Error in receive loop: {e}')
                self._client = None
                time.sleep(self.reconnect_delay)

    def run_blocking(self, duration: float | None = None) -> None:
        """
        Run the client in blocking mode.

        Args:
            duration: Maximum duration in seconds (None for indefinite)
        """
        self.start()
        start_time = time.time()

        try:
            while self._running:
                if duration and (time.time() - start_time) > duration:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info('Interrupted by user')
        finally:
            self.stop()

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._client is not None

    @property
    def stats(self) -> dict:
        """Get client statistics."""
        return {
            'server': self.server,
            'stations': [str(s) for s in self.stations],
            'connected': self.is_connected,
            'running': self._running,
            'packet_count': self._packet_count,
            'last_packet': self._last_packet_time.isoformat() if self._last_packet_time else None,
        }


def load_stations_from_csv(csv_path: Path) -> list[StationConfig]:
    """
    Load station configurations from a CSV file.

    Expected CSV format:
    network,station,location,channels
    TW,YULB,00,HH?
    TW,MASB,00,BH?

    Args:
        csv_path: Path to CSV file

    Returns:
        List of StationConfig objects
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    stations = []

    for _, row in df.iterrows():
        config = StationConfig(
            network=row.get('network', 'TW'),
            station=row['station'],
            location=row.get('location', ''),
            channels=row.get('channels', 'HH?').split(',') if isinstance(row.get('channels'), str) else ['HH?'],
        )
        stations.append(config)

    return stations


def create_seedlink_client(
    server: str,
    station_file: Path | None = None,
    stations: list[dict] | None = None,
    buffer: WaveformBuffer | None = None,
) -> SEEDLINKClient:
    """
    Factory function to create a SEEDLINK client.

    Args:
        server: SEEDLINK server address
        station_file: Path to station CSV file
        stations: List of station dictionaries
        buffer: WaveformBuffer (created if None)

    Returns:
        Configured SEEDLINKClient
    """
    from .buffers import WaveformBuffer

    if buffer is None:
        buffer = WaveformBuffer()

    if station_file:
        station_configs = load_stations_from_csv(station_file)
    elif stations:
        station_configs = stations
    else:
        raise ValueError('Must provide either station_file or stations')

    return SEEDLINKClient(
        server=server,
        stations=station_configs,
        buffer=buffer,
    )