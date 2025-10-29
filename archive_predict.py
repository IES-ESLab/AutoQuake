#%%
import json
import logging
import logging.config
from pathlib import Path
from autoquake.picker import parallel_run_phasenet
from ParamConfig.config_model import MainConfig

# Path of config file
LOG_CONFIG = Path(__file__).parents[0].resolve() / 'log' / 'config.json'
PARAM_CONFIG = Path(__file__).parents[0].resolve() / 'ParamConfig' / 'params_stream.json'
FETCH_CONFIG = Path(__file__).parents[0].resolve() / 'ParamConfig' / 'data_client.json'

def setup_logging():
    with open(LOG_CONFIG, 'r') as f:
        data = json.load(f)
    log_config  = data["LoggingConfig"]
    logging.config.dictConfig(log_config)

if __name__ == '__main__':
    setup_logging()
    
    # Load config file
    with open(PARAM_CONFIG, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        config = MainConfig(**config_dict)
    
    logging.info('Running PhaseNet...')
    phasenet_config_list = config.PhaseNet.args_list
    parallel_run_phasenet(phasenet_config_list, workers=1) # because we can not fetch the data parallelly



# %%
