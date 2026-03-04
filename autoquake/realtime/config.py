"""
Realtime system configuration.
"""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class RealtimeConfig(BaseModel):
    """Configuration for the realtime earthquake detection system."""

    # Paths
    station_file: Path
    result_path: Path
    velocity_model: Path | None = None

    # Buffer settings
    pick_window_seconds: float = Field(default=60.0, description='Pick buffer window size in seconds')
    pick_overlap: float = Field(default=0.5, ge=0.0, le=0.9, description='Window overlap ratio')
    min_picks_to_trigger: int = Field(default=10, description='Minimum picks to trigger association')
    max_pick_age_seconds: float = Field(default=120.0, description='Maximum age for picks in buffer')

    # GaMMA settings
    center: tuple[float, float] | None = None
    xlim_degree: list[float] | None = None
    ylim_degree: list[float] | None = None
    zlim: tuple[float, float] = (0, 60)
    method: Literal['BGMM', 'GMM'] = 'BGMM'
    use_amplitude: bool = False
    vp: float = 6.0
    vs: float = 3.43
    ncpu: int = 4
    min_picks_per_eq: int = 8
    min_p_picks_per_eq: int = 6
    min_s_picks_per_eq: int = 2
    max_sigma11: float = 2.0

    # Event validation
    validation_min_picks: int = 8
    validation_max_residual: float = 2.0

    # Publisher settings
    publisher_endpoint: str | None = None
    publisher_output_dir: Path | None = None

    # Simulator settings (for testing)
    simulator_speed: float = Field(default=1.0, ge=0.1, le=100.0, description='Simulation speed multiplier')

    # Relocation settings (H3DD)
    enable_relocation: bool = Field(default=False, description='Enable H3DD relocation')
    h3dd_dir: Path | None = Field(default=None, description='H3DD executable directory')
    model_3d: Path | None = Field(default=None, description='3D velocity model for relocation')

    # Magnitude settings
    enable_magnitude: bool = Field(default=True, description='Enable magnitude estimation')
    pz_dir: Path | None = Field(default=None, description='PZ files directory for WA simulation')
    use_wa_simulation: bool = Field(default=True, description='Use Wood-Anderson simulation for magnitude')

    # Focal mechanism settings
    enable_focal: bool = Field(default=False, description='Enable focal mechanism estimation')
    gafocal_dir: Path | None = Field(default=None, description='GAfocal executable directory')
    min_polarities: int = Field(default=8, description='Minimum polarities for focal mechanism')

    class Config:
        arbitrary_types_allowed = True
