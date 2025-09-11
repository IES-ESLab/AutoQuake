#%%
from pathlib import Path
from typing import Callable, Literal, Any
from pydantic import BaseModel, model_validator
import re
from datetime import datetime, timedelta

def default_type_judge(x: str):
    """
    This is for DitingMotion to judge the type of station.
    For example, under the scenario here, True means the station is a seismometer,\
    and False means the station is a DAS channel.
    """
    return x[1].isalpha()


"""
We set PhaseNet as True, since it's the first step.
Other steps default as True (since it's an example)

Please note that this flow is an example and demonstrates how to\
instantiate the classes and run.

We use getter function to automatically pass the arguments, once\
you familiar with the instantiation, you can switch it to file you want to use. 
"""


class PhaseNetConfig(BaseModel):
    """
    Base parameters for PhaseNet (all except 'mode').
    """
    start: str
    end: str
    result_path: Path
    data_path: Path | None = None
    pz_dir: Path | None = None
    data_list: str | None = None
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
    # output dir
    pick_path: Path | None = None
    event_path: Path | None = None
    figure_path: Path | None = None
class PhaseNetConfigReceiver(PhaseNetConfig):
    """
    Parameters for PhaseNet (all parameters including 'mode').
    """
    data_parent_dir: Path
    mode: Literal["stream", "archive"] = "archive"
    args_list: list[PhaseNetConfig] | None = None
    

    def _check_result_path(self):
        if self.result_path.exists() and any(self.result_path.iterdir()):
            response = input(f"Warning: '{self.result_path}' exists and is not empty. Overwrite? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("Operation cancelled by user.")
    
    def _check_output_dir(self):
        """
        Make sure the dir is exist or created.
        """
        def subdir_name(mode: str, start: str) -> str:
            """
            We typicaly assume that the stream and archive is executived daily.
            """
            if mode == "stream":
                return start.split('T')[0].replace("-","")
            else:
                return start
            
        dir_name = subdir_name(self.mode, self.start)

        if self.cut_patch:
            self.pick_path = self.result_path / f'picks_{self.model}_patch' / dir_name
            self.event_path = self.result_path / f'events_{self.model}_patch' / dir_name
            self.figure_path = self.result_path / f'figures_{self.model}_patch' / dir_name
            
        else:
            self.pick_path = self.result_path / f'picks_{self.model}' / dir_name
            self.event_path = self.result_path / f'events_{self.model}' / dir_name
            self.figure_path = self.result_path / f'figures_{self.model}' / dir_name  
        
        self.pick_path.mkdir(parents=True, exist_ok=True)
        self.event_path.mkdir(parents=True, exist_ok=True)
        self.figure_path.mkdir(parents=True, exist_ok=True)

    def _time_code_checker(self):
        stream_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
        archive_pattern = r"^\d{8}$"
        if self.mode == "stream":
            if not (re.match(stream_pattern, self.start) and re.match(stream_pattern, self.end)):
                raise ValueError(
                    "For 'stream' mode, startdate and enddate must be in format 'YYYY-MM-DDTHH:MM:SS', e.g., '2024-04-02T00:00:00'."
                )
        elif self.mode == "archive":
            if not (re.match(archive_pattern, self.start) and re.match(archive_pattern, self.end)):
                raise ValueError(
                    "For 'archive' mode, startdate and enddate must be in format 'YYYYMMDD', e.g., '20240402'."
                )
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Supported modes are 'stream' and 'archive'.")
        
    def _convert_stream_configs(self, interval=24) -> list[PhaseNetConfig]:
        """
        Divide the period from startdate to enddate into intervals (default 24 hrs),
        and return a list of PhaseNetConfig objects with updated startdate and enddate.
        Only startdate and enddate are replaced in each iteration.
        """
        start = datetime.strptime(self.start, "%Y-%m-%dT%H:%M:%S")
        end = datetime.strptime(self.end, "%Y-%m-%dT%H:%M:%S")
        configs = []

        current = start
        base_config = self.model_dump()
        while current < end:
            next_time = min(current + timedelta(hours=interval), end)
            config_kwargs = base_config.copy()
            config_kwargs['start'] = current.strftime("%Y-%m-%dT%H:%M:%S")
            config_kwargs['end'] = next_time.strftime("%Y-%m-%dT%H:%M:%S")
            config = PhaseNetConfig(**{k: config_kwargs[k] for k in PhaseNetConfig.model_fields.keys()})
            configs.append(config)
            current = next_time

        return configs

    def _convert_archive_configs(self, interval=1):
        """
        Divide the period from startdate to enddate into intervals (default 1 days),
        and return a list of PhaseNetConfig objects with updated startdate and enddate.
        Only startdate and enddate are replaced in each iteration.
        """
        def check_data_list(data_path: Path, data_list, format: str):
            """
            ## This is for generating data_list from archive-sac data (if not provided).
            """
            if data_list is None:
                all_list = list(data_path.glob(f'*{format}'))
                data_list = []
                for i in all_list:
                    fname = f"{str(i).split('.D.')[0][:-1]}*"
                    if fname not in data_list:
                        data_list.append(fname)
            else:
                with open(data_list) as f:
                    data_list = f.read().splitlines()
            return data_list     
           
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
            config_kwargs["data_list"] = check_data_list(data_path, self.data_list, self.format)
            config = PhaseNetConfig(**{k: config_kwargs[k] for k in PhaseNetConfig.model_fields.keys()})
            configs.append(config)
            current = next_time
        
        return configs
            
    def generate_configs(self):
        if self.mode == "stream":
            return self._convert_stream_configs()
        else:
            return self._convert_archive_configs()

    @model_validator(mode="after")
    def post_validation(self):
        """
        ### Validation and Generation
        Here we do several validations, including: \n
        - result path \n
        - time format \n
        Also we generate a new parameters: \n
        - args_list 
        """
        self._check_result_path()
        self._time_code_checker()
        self._check_output_dir()
        self.args_list = self.generate_configs()
        return self
    
class GaMMAConfig(BaseModel):
    station: Path
    velocity_model: Path
    center: tuple[float, float]
    xlim_degree: list[float]
    ylim_degree: list[float]
    min_p_picks_per_eq: int
    min_s_picks_per_eq: int
    eps: float
    cpu_number: int
    use_amplitude: bool

class H3DDConfig(BaseModel):
    station: Path
    model_3D: Path
    cutoff_distance_for_first_h3dd: float
    event_name_for_first_h3dd: str
    cutoff_distance_for_second_h3dd: float
    event_name_for_second_h3dd: str

class MagConfig(BaseModel):
    sac_parent_dir: Path
    pz_dir: Path
    station: Path
    cpu_number: int

class DitingConfig(BaseModel):
    sac_parent_dir: Path
    cpu_number: int
    type_judge: Callable[[str], bool] = default_type_judge

class MainConfig(BaseModel):
    result_path: Path
    PhaseNet: PhaseNetConfigReceiver
    GaMMA: GaMMAConfig
    H3DD: H3DDConfig
    Mag: MagConfig
    Diting: DitingConfig
# %%
