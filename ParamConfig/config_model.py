#%%
from pathlib import Path
from collections import Counter
import pandas as pd
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
    data_parent_dir: Path
    station_csv: Path
    args_list: list[PhaseNetConfig] | None = None
    date_list: list[str] | None = None  # New attribute to store the list of dates

    def _check_result_path(self):
        if self.result_path.exists() and any(self.result_path.iterdir()):
            response = input(f"Warning: '{self.result_path}' exists and is not empty. Overwrite? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("Operation cancelled by user.")

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
