from pathlib import Path
from typing import Callable
from pydantic import BaseModel

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
    Parameters for PhaseNet.
    """
    sac_parent_dir: Path
    pz_dir: Path
    startdate: str
    enddate: str

class GaMMAConfig(BaseModel):
    station: Path
    velocity_model: Path
    center: tuple[float, float]
    xlim_degree: list[float] #TODO: a little bit weird to use list
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
    PhaseNet: PhaseNetConfig
    GaMMA: GaMMAConfig
    H3DD: H3DDConfig
    Mag: MagConfig
    Diting: DitingConfig