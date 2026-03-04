"""
AutoQuake Realtime Module

This module provides real-time earthquake detection and location capabilities.
"""

from .buffers import PickBuffer, WaveformBuffer
from .simulators import PickStreamSimulator
from .associator import RealtimeGaMMA
from .event_validator import EventValidator
from .publisher import JSONPublisher
from .config import RealtimeConfig
from .runner import RealtimeRunner
from .picker import RealtimePhaseNet
from .seedlink_client import SEEDLINKClient, StationConfig
from .relocator import RealtimeRelocator
from .magnitude import RealtimeMagnitude, AmplitudeTable
from .focal import RealtimeFocalMechanism

__all__ = [
    # Data ingestion & buffers
    'PickBuffer',
    'WaveformBuffer',
    'SEEDLINKClient',
    'StationConfig',
    # Picking (with polarity & amplitude)
    'RealtimePhaseNet',
    # Association & location
    'RealtimeGaMMA',
    'EventValidator',
    # Relocation
    'RealtimeRelocator',
    # Magnitude estimation
    'RealtimeMagnitude',
    'AmplitudeTable',
    # Focal mechanism
    'RealtimeFocalMechanism',
    # Publishing
    'JSONPublisher',
    # Simulation & testing
    'PickStreamSimulator',
    # Config & orchestration
    'RealtimeConfig',
    'RealtimeRunner',
]
