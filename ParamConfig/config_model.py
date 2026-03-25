#%%
from pathlib import Path
import pandas as pd
from typing import Callable, Literal, Any
from pydantic import BaseModel, model_validator, Field
import re
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def default_type_judge(x: str):
    """
    This is for DitingMotion to judge the type of station.
    For example, under the scenario here, True means the station is a seismometer,
    and False means the station is a DAS channel.
    """
    return x[1].isalpha()


class PhaseNetConfig(BaseModel):
    """
    Base parameters for PhaseNet (all except 'mode').
    """ 
    start: str
    end: str
    result_path: Path
    data_path: Path | None = None
    pz_dir: Path | None = None
    data_list: Any | None = None
    hdf5_file: str | None = None
    prefix: str = ""
    format: Literal["SAC", "h5"] = "SAC"
    dataset: Literal["das", "seismic"] = "das"
    model: Literal["phasenet", "phasenet_plus", "phasenet_das"] = "phasenet"
    resume: str = ""
    backbone: str = "unet"
    phases: list[str] = ["P", "S"]
    device: str = "cpu"
    workers: int = 0
    batch_size: int = 1
    use_deterministic_algorithms: bool = True
    amp: bool = True
    world_size: int = 1
    dist_url: str = "env://"
    plot_figure: bool = False
    min_prob: float = 0.3
    add_polarity: bool = False
    add_event: bool = True
    sampling_rate: float = 100.0
    highpass_filter: float = 0.0
    response_path: str | None = None
    response_xml: str | None = None
    subdir_level: int = 0
    # DAS parameters
    cut_patch: bool = False
    nt: int = 1024 * 20 # TODO: Modified as Midas default
    nx: int = 1024 * 5 
    resample_time: bool = False
    resample_space: bool = False
    system: str | None = None
    location: str | None = None
    skip_existing: bool = False
    # Post assigned paramters (predefine here due to BaseModel)
    dtype: str | None = None
    ptdtype: Any | None = None
    rank: int | None = None
    world_size: int | None = None
    gpu: int | None = None
    distributed: bool | None = None
    dist_bakend: str | None = None
class PhaseNetConfigReceiver(PhaseNetConfig):
    """
    Parameters for PhaseNet (all parameters including 'mode').
    """
    enabled: bool = True
    data_parent_dir: Path
    station_csv: Path
    args_list: list[PhaseNetConfig] | None = None
    date_list: list[str] | None = None  # New attribute to store the list of dates

    def _check_result_path(self):
        if self.result_path.exists() and any(self.result_path.iterdir()):
            logger.warning(f"Result path '{self.result_path}' exists and is not empty.")

    def _time_code_checker(self):
        # stream_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
        archive_pattern = r"^\d{8}$"
        # if self.mode == "stream":
        #     if not (re.match(stream_pattern, self.start) and re.match(stream_pattern, self.end)):
        #         raise ValueError(
        #             "For 'stream' mode, startdate and enddate must be in format 'YYYY-MM-DDTHH:MM:SS', e.g., '2024-04-02T00:00:00'."
        #         )
        # elif self.mode == "archive":
        if not (re.match(archive_pattern, self.start) and re.match(archive_pattern, self.end)):
            raise ValueError(
                "For 'archive' mode, startdate and enddate must be in format 'YYYYMMDD', e.g., '20240402'."
            )
        # else:
        #     raise ValueError(f"Unknown mode '{self.mode}'. Supported modes are 'stream' and 'archive'.")
    def _generate_date_list(self):
        """
        Generate a list of dates between start and end in 'YYYYMMDD' format.
        """
        start_date = datetime.strptime(self.start, '%Y%m%d')
        end_date = datetime.strptime(self.end, '%Y%m%d')
        date_list = []
        current_date = start_date
        while current_date < end_date:
            date_list.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
        return date_list

    # def _convert_stream_configs(self, interval=24) -> list[PhaseNetConfig]:
    #     """
    #     Divide the period from startdate to enddate into intervals (default 24 hrs),
    #     and return a list of PhaseNetConfig objects with updated startdate and enddate.
    #     Only startdate and enddate are replaced in each iteration.
    #     """
    #     start = datetime.strptime(self.start, "%Y-%m-%dT%H:%M:%S")
    #     end = datetime.strptime(self.end, "%Y-%m-%dT%H:%M:%S")
    #     configs = []

    #     current = start
    #     base_config = self.model_dump()
    #     while current < end:
    #         next_time = min(current + timedelta(hours=interval), end)
    #         config_kwargs = base_config.copy()
    #         config_kwargs['start'] = current.strftime("%Y-%m-%dT%H:%M:%S")
    #         config_kwargs['end'] = next_time.strftime("%Y-%m-%dT%H:%M:%S")
    #         config = PhaseNetConfig(**{k: config_kwargs[k] for k in PhaseNetConfig.model_fields.keys()})
    #         configs.append(config)
    #         current = next_time

    #     return configs

    def _convert_archive_configs(self, interval=1):
        """
        Divide the period from startdate to enddate into intervals (default 1 days),
        and return a list of PhaseNetConfig objects with updated startdate and enddate.
        Only startdate and enddate are replaced in each iteration.
        """
        def check_data_list(
            data_path: Path,
            station_csv: Path | str,
            data_list: str | Path | None,
            format: str,
            splitter='.'
            ) -> list | None:
            def _get_single_station_type(data_path, station, splitter, format):
                sta_list = list(data_path.glob(f"*.{station}.*"))
                sta_collector = []
                for sta_path in sta_list:
                    filename = sta_path.name 
                    index = filename.find(station)
                    tmp_filename = filename[index:]
                    s, loc, inst = tmp_filename.split(splitter)[:3]
                    sta_collector.append(str(data_path / f"*{s}.{loc}.{inst[:-1]}*.{format}"))
                return sta_collector
            if data_list is None:
                if format == "SAC":            
                    df_station = pd.read_csv(station_csv)
                    fname_set = set()
                    for sta_name in df_station['station']:
                        sta_collector = _get_single_station_type(data_path, sta_name, splitter, format)
                        # print(Counter(sta_collector))
                        fname_set.update(set(sta_collector))
                    return list(fname_set)
                elif format == "h5":
                    return None
                else:
                    raise ValueError(f"Unsupported format '{format}'. Supported formats are 'SAC' and 'h5'.")
            else:
                with open(data_list) as f:
                    data_list = f.read().splitlines()
                return data_list


            # if output_path is not None:
            #     with open(output_path, 'w') as f:
            #         for i in fname_set:
            #             f.write(f"{i}\n")    
           
        start = datetime.strptime(self.start, '%Y%m%d')
        end = datetime.strptime(self.end, '%Y%m%d')
        configs = []
        current = start
        base_config = self.model_dump()
        while current < end:
            next_time = min(current + timedelta(days=interval), end)
            config_kwargs = base_config.copy()
            config_kwargs["start"] = current.strftime('%Y%m%d')
            config_kwargs["end"] = next_time.strftime('%Y%m%d') # For consistency, but not use.
            data_path = self.data_parent_dir / current.strftime('%Y%m%d') 
            config_kwargs["data_path"] = data_path
            _gen_list = check_data_list(data_path, self.station_csv, self.data_list, self.format)
            config_kwargs["data_list"] = _gen_list
            print(f"Available station on {current}: {len(_gen_list)} / {pd.read_csv(self.station_csv).shape[0]}")
            config = PhaseNetConfig(**{k: config_kwargs[k] for k in PhaseNetConfig.model_fields.keys()})
            configs.append(config)
            current = next_time
        
        return configs
            
    def generate_configs(self):
        # if self.mode == "stream":
        #     return self._convert_stream_configs()
        # else:
        return self._convert_archive_configs()

    @model_validator(mode="after")
    def post_validation(self):
        """
        ### Validation and Generation
        Here we do several validations, including: \n
        - result path \n
        - time format \n
        Also we generate new parameters: \n
        - args_list \n
        - date_list
        """
        self._check_result_path()
        self._time_code_checker()
        self.date_list = self._generate_date_list()  # Generate the date list
        self.args_list = self.generate_configs()
        return self
    
class GaMMAConfig(BaseModel):
    """GaMMA configuration with optional external input."""
    enabled: bool = True
    # External input (if skipping PhaseNet)
    picks_csv: Path | None = None
    # Required parameters (optional when enabled=False)
    station: Path | None = None
    velocity_model: Path | None = None
    center: tuple[float, float] | None = None
    xlim_degree: list[float] | None = None
    ylim_degree: list[float] | None = None
    min_p_picks_per_eq: int = 6
    min_s_picks_per_eq: int = 2
    eps: float = 17.0
    cpu_number: int = 40
    use_amplitude: bool = False

    @model_validator(mode='after')
    def validate_required_when_enabled(self):
        """Validate required fields when enabled."""
        if self.enabled:
            required = ['station', 'velocity_model', 'center', 'xlim_degree', 'ylim_degree']
            for field in required:
                if getattr(self, field) is None:
                    raise ValueError(f'{field} is required when GaMMA is enabled')
        return self

class H3DDSingleRunConfig(BaseModel):
    """Configuration for a single H3DD run."""
    event_name: str
    cutoff_distances: float  # Note: singular value for each run


class H3DDRunsConfig(BaseModel):
    """Configuration for H3DD runs (first required, second optional)."""
    first: H3DDSingleRunConfig
    second: H3DDSingleRunConfig | None = None


class H3DDConfig(BaseModel):
    """H3DD configuration with nested runs structure."""
    enabled: bool = True
    # External inputs (if skipping GaMMA)
    events_csv: Path | None = None
    picks_csv: Path | None = None
    # Required parameters (optional when enabled=False)
    station: Path | None = None
    model_3D: Path | None = None
    # H3DD algorithm parameters
    weights: list[float] = [1.0, 1.0, 0.1]
    priori_weight: list[float] = [1.0, 0.75, 0.5, 0.25, 0.0]
    inv: int = 2
    damping_factor: float = 0.0
    rmscut: float = 1.0e-4
    max_iter: int = 5
    constrain_factor: float = 0.0
    joint_inv_with_single_event_method: int = 1
    consider_elevation: int = 0
    # Run configurations
    runs: H3DDRunsConfig | None = None

    @model_validator(mode='after')
    def validate_h3dd_config(self):
        """Validate H3DD configuration."""
        if self.enabled:
            # Required fields when enabled
            if self.station is None:
                raise ValueError('station is required when H3DD is enabled')
            if self.model_3D is None:
                raise ValueError('model_3D is required when H3DD is enabled')
            # runs.first is required when enabled
            if self.runs is None:
                raise ValueError('runs configuration is required when H3DD is enabled')
            # first run config cannot be empty (already enforced by H3DDRunsConfig)
        return self

    def get_run_count(self) -> int:
        """Get the number of runs to execute."""
        if self.runs is None:
            return 0
        return 2 if self.runs.second is not None else 1

    def get_run_config(self, run_index: int) -> H3DDSingleRunConfig | None:
        """Get configuration for a specific run index (0 or 1)."""
        if self.runs is None:
            return None
        if run_index == 0:
            return self.runs.first
        elif run_index == 1:
            return self.runs.second
        return None

class MagConfig(BaseModel):
    """Magnitude calculation configuration."""
    enabled: bool = True
    # External input (if skipping H3DD)
    dout_file: Path | None = None
    # Required parameters (optional when enabled=False)
    sac_parent_dir: Path | None = None
    pz_dir: Path | None = None
    station: Path | None = None
    cpu_number: int = 40

    @model_validator(mode='after')
    def validate_required_when_enabled(self):
        """Validate required fields when enabled."""
        if self.enabled:
            if self.sac_parent_dir is None:
                raise ValueError('sac_parent_dir is required when Magnitude is enabled')
            if self.pz_dir is None:
                raise ValueError('pz_dir is required when Magnitude is enabled')
            if self.station is None:
                raise ValueError('station is required when Magnitude is enabled')
        return self


class DitingConfig(BaseModel):
    """Polarity (DitingMotion) configuration."""
    enabled: bool = True
    # External input (if skipping GaMMA)
    picks_csv: Path | None = None
    # Required parameters (at least one dir required when enabled)
    sac_parent_dir: Path | None = None
    h5_parent_dir: Path | None = None
    cpu_number: int = 15
    type_judge: Callable[[str], bool] = default_type_judge

    @model_validator(mode='after')
    def validate_required_when_enabled(self):
        """Validate required fields when enabled."""
        if self.enabled:
            if self.sac_parent_dir is None and self.h5_parent_dir is None:
                raise ValueError(
                    'Either sac_parent_dir or h5_parent_dir is required when Polarity is enabled'
                )
        return self


class FocalConfig(BaseModel):
    """GAfocal configuration."""
    enabled: bool = True
    # External input
    dout_file: Path | None = None


# =============================================================================
# Path Resolver (smart path detection)
# =============================================================================
class PathResolver:
    """
    Resolves input paths for pipeline components.
    Priority: explicit path > auto-detect in result_path > raise error
    """

    def __init__(self, result_path: Path):
        self.result_path = Path(result_path)

    def resolve_picks(self, explicit_path: Path | None = None) -> Path:
        """Resolve picks CSV file path."""
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path
            raise FileNotFoundError(f'Explicit picks_csv not found: {explicit_path}')

        # Auto-detect candidates
        candidates = [
            self.result_path / 'phase_picks.csv',
            self.result_path / 'picks_phasenet.csv',
            self.result_path / 'gamma_picks.csv',
        ]
        for candidate in candidates:
            if candidate.exists():
                logger.info(f'Auto-detected picks file: {candidate}')
                return candidate

        raise FileNotFoundError(
            f'Cannot find picks CSV in {self.result_path}. '
            f'Searched: {[c.name for c in candidates]}'
        )

    def resolve_gamma_picks(self, explicit_path: Path | None = None) -> Path:
        """Resolve GaMMA picks CSV file path."""
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path
            raise FileNotFoundError(f'Explicit picks_csv not found: {explicit_path}')

        candidates = [
            self.result_path / 'gamma_picks.csv',
        ]
        for candidate in candidates:
            if candidate.exists():
                logger.info(f'Auto-detected GaMMA picks file: {candidate}')
                return candidate

        raise FileNotFoundError(
            f'Cannot find GaMMA picks CSV in {self.result_path}. '
            f'Searched: {[c.name for c in candidates]}'
        )

    def resolve_events(self, explicit_path: Path | None = None) -> Path:
        """Resolve events CSV file path."""
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path
            raise FileNotFoundError(f'Explicit events_csv not found: {explicit_path}')

        candidates = [
            self.result_path / 'gamma_event.csv',
            self.result_path / 'gamma_events.csv',
            self.result_path / 'events.csv',
        ]
        for candidate in candidates:
            if candidate.exists():
                logger.info(f'Auto-detected events file: {candidate}')
                return candidate

        raise FileNotFoundError(
            f'Cannot find events CSV in {self.result_path}. '
            f'Searched: {[c.name for c in candidates]}'
        )

    def resolve_dout(self, explicit_path: Path | None = None, event_name: str | None = None) -> Path:
        """Resolve dout file path."""
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path
            raise FileNotFoundError(f'Explicit dout_file not found: {explicit_path}')

        # Try with event_name first
        if event_name:
            candidate = self.result_path / f'{event_name}.dout'
            if candidate.exists():
                logger.info(f'Auto-detected dout file: {candidate}')
                return candidate

        # Search for any .dout file
        dout_files = list(self.result_path.glob('*.dout'))
        if dout_files:
            # Sort by modification time, use most recent
            dout_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            logger.info(f'Auto-detected dout file: {dout_files[0]}')
            return dout_files[0]

        raise FileNotFoundError(f'Cannot find dout file in {self.result_path}')


# =============================================================================
# Run Configuration (single pipeline run)
# =============================================================================
class RunConfig(BaseModel):
    """Configuration for a single pipeline run."""
    name: str = 'default'
    result_path: Path
    # Optional components (None or enabled=False means skip)
    PhaseNet: PhaseNetConfigReceiver | None = None
    GaMMA: GaMMAConfig | None = None
    H3DD: H3DDConfig | None = None
    Magnitude: MagConfig | None = Field(default=None, alias='Mag')
    Polarity: DitingConfig | None = Field(default=None, alias='Diting')
    Focal: FocalConfig | None = None

    model_config = {'populate_by_name': True}

    def is_component_enabled(self, component_name: str) -> bool:
        """Check if a component is enabled."""
        component = getattr(self, component_name, None)
        if component is None:
            return False
        return getattr(component, 'enabled', True)

    @model_validator(mode='after')
    def validate_dependencies(self):
        """Validate that enabled components have required inputs."""
        # GaMMA needs picks
        if self.is_component_enabled('GaMMA'):
            if not self.is_component_enabled('PhaseNet') and not self.GaMMA.picks_csv:
                logger.warning(
                    'GaMMA is enabled but PhaseNet is not, and no picks_csv provided. '
                    'Will attempt to auto-detect picks in result_path.'
                )

        # H3DD needs events and picks
        if self.is_component_enabled('H3DD'):
            if not self.is_component_enabled('GaMMA'):
                if not self.H3DD.events_csv or not self.H3DD.picks_csv:
                    logger.warning(
                        'H3DD is enabled but GaMMA is not, and events_csv/picks_csv not fully provided. '
                        'Will attempt to auto-detect in result_path.'
                    )

        # Magnitude needs dout
        if self.is_component_enabled('Magnitude'):
            if not self.is_component_enabled('H3DD') and not self.Magnitude.dout_file:
                logger.warning(
                    'Magnitude is enabled but H3DD is not, and no dout_file provided. '
                    'Will attempt to auto-detect in result_path.'
                )

        # Polarity needs picks
        if self.is_component_enabled('Polarity'):
            if not self.is_component_enabled('GaMMA') and not self.Polarity.picks_csv:
                logger.warning(
                    'Polarity is enabled but GaMMA is not, and no picks_csv provided. '
                    'Will attempt to auto-detect in result_path.'
                )

        return self


# =============================================================================
# Batch Configuration (multiple runs)
# =============================================================================
class BatchConfig(BaseModel):
    """Configuration for batch execution of multiple pipeline runs."""
    configs: list[RunConfig]

    @model_validator(mode='before')
    @classmethod
    def handle_legacy_format(cls, data: dict) -> dict:
        """Handle legacy single-config format for backward compatibility."""
        if 'configs' not in data:
            # Legacy format: single config at root level
            # Extract result_path and wrap as single RunConfig
            logger.info('Detected legacy config format, converting to batch format.')
            return {'configs': [data]}
        return data


# =============================================================================
# Legacy MainConfig (for backward compatibility with predict.py)
# =============================================================================
class MainConfig(BaseModel):
    """Legacy configuration format. Use BatchConfig for new implementations."""
    result_path: Path
    PhaseNet: PhaseNetConfigReceiver
    GaMMA: GaMMAConfig
    H3DD: H3DDConfig
    Mag: MagConfig
    Diting: DitingConfig

    def to_run_config(self) -> RunConfig:
        """Convert legacy MainConfig to RunConfig."""
        return RunConfig(
            name='legacy_config',
            result_path=self.result_path,
            PhaseNet=self.PhaseNet,
            GaMMA=self.GaMMA,
            H3DD=self.H3DD,
            Magnitude=self.Mag,
            Polarity=self.Diting,
            Focal=FocalConfig(enabled=True),
        )
# %%
