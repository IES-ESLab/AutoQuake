#%%
import json
import logging
from pathlib import Path

# from autoquake.associator import GaMMA
# from autoquake.focal import GAfocal
# from autoquake.magnitude import Magnitude
from autoquake.picker import PhaseNet, run_predict
# from autoquake.polarity import DitingMotion
# from autoquake.relocator import H3DD
# from autoquake.utils import gamma_preprocessing, pol_mag_to_dout, process_for_h3dd_twice
from ParamConfig.config_model import MainConfig

# Path of config file
CONFIG_JSON = Path(__file__).parents[0].resolve() / 'ParamConfig' / 'params.json'
#%%
if __name__ == '__main__':
    
    # Load config file
    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        config = MainConfig(**config_dict)

    # Initialize result path
    config.result_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=config.result_path / 'logging_file.log',
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
    )

    run_predict(config.PhaseNet.args_list)
