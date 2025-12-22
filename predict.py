import json
import logging
from pathlib import Path

from autoquake.associator import GaMMA
from autoquake.focal import GAfocal
from autoquake.magnitude import Magnitude
from autoquake.picker import PhaseNet, run_predict
from autoquake.polarity import DitingMotion
from autoquake.relocator import H3DD
from autoquake.utils import gamma_preprocessing, pol_mag_to_dout, process_for_h3dd_twice
from ParamConfig.config_model import MainConfig

# Path of config file
CONFIG_JSON = Path(__file__).parents[0].resolve() / 'ParamConfig' / 'params.json'

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
    
    logging.info('Running PhaseNet...')
    run_predict(config.PhaseNet.args_list)
    phase_picks = PhaseNet.concat_picks(
        date_list=config.PhaseNet.date_list,
        result_path=config.result_path
    )

    logging.info('Running GaMMA...')
    post_phasenet_pickings = gamma_preprocessing(
        pickings=phase_picks,
        output_dir=config.result_path
    )
    
    gamma = GaMMA(
        station=config.GaMMA.station,
        result_path=config.result_path,
        center=config.GaMMA.center, # optional
        xlim_degree=config.GaMMA.xlim_degree, # optional
        ylim_degree=config.GaMMA.ylim_degree, # optional
        pickings=post_phasenet_pickings,
        vel_model=config.GaMMA.velocity_model, # optional
        min_p_picks_per_eq=config.GaMMA.min_p_picks_per_eq,
        min_s_picks_per_eq=config.GaMMA.min_s_picks_per_eq,
        dbscan_eps=config.GaMMA.eps,
        ncpu=config.GaMMA.cpu_number,
        use_amplitude=config.GaMMA.use_amplitude # optional, default as False
    )
    gamma.run_predict()

    logging.info('H3DD first start')
    h3dd = H3DD(
        gamma_event=gamma.get_events(),
        gamma_picks=gamma.get_picks(),
        result_path=config.result_path,
        station=config.H3DD.station,
        model_3d=config.H3DD.model_3D,
        event_name=config.H3DD.event_name_for_first_h3dd, # This for naming dout and hout.
        cut_off_distance_for_dd=config.H3DD.cutoff_distance_for_first_h3dd
    )
    h3dd.run_h3dd()

    logging.info('processing first H3DD results')
    h3dd_events_first, h3dd_picks_first = process_for_h3dd_twice(
        station=config.H3DD.station,
        dout=h3dd.get_dout(),
        event_name_1=config.H3DD.event_name_for_first_h3dd, # This for naming csv for first H3DD.
        result_path=config.result_path
    )

    logging.info('H3DD second start')
    h3dd_2 = H3DD(
        gamma_event=h3dd_events_first,
        gamma_picks=h3dd_picks_first,
        result_path=config.result_path,
        station=config.H3DD.station,
        model_3d=config.H3DD.model_3D,
        event_name=config.H3DD.event_name_for_second_h3dd,
        cut_off_distance_for_dd=config.H3DD.cutoff_distance_for_second_h3dd
    )
    h3dd_2.run_h3dd()
    

    logging.info('Mag_started')
    mag = Magnitude(
        dout_file=h3dd_2.get_dout(),
        station=config.Mag.station,
        sac_parent_dir=config.Mag.sac_parent_dir,
        pz_dir=config.Mag.pz_dir,
        output_dir=config.result_path,
    )
    mag.run_mag(processes=config.Mag.cpu_number)
    
    logging.info('Running DitingMotion...')
    dt_polarity = DitingMotion(
        gamma_picks=gamma.get_picks(),
        output_dir=config.result_path,
        sac_parent_dir=config.Diting.sac_parent_dir,
        type_judge=config.Diting.type_judge
    )
    dt_polarity.run_parallel_predict(processes=15)

    logging.info('Format converting with pol and mag...')
    dout_file_name = pol_mag_to_dout(
        ori_dout=h3dd_2.get_dout(),
        result_path=config.result_path,
        df_reorder_event=h3dd.get_df_reorder_event(),
        polarity_picks=dt_polarity.get_picks(),
        magnitude_events=mag.get_events(),
        magnitude_picks=mag.get_picks()
    )

    logging.info('GAfocal start')
    gafocal = GAfocal(
        dout_file_name=dout_file_name, # The reason why here is a str is due to GAfocal execute in-place.
        result_path=config.result_path
        )
    gafocal.run()
