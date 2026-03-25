#!/usr/bin/env python
"""
AutoQuake CLI - Automated Earthquake Catalog Generation Pipeline

Usage:
    python main.py --config config.json
    python main.py --config config.json --dry-run
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from autoquake.associator import GaMMA
from autoquake.focal import GAfocal
from autoquake.magnitude import Magnitude
from autoquake.picker import PhaseNet, run_predict
from autoquake.polarity import DitingMotion
from autoquake.relocator import H3DD
from autoquake.utils import gamma_preprocessing, pol_mag_to_dout, process_for_h3dd_twice
from ParamConfig.config_model import (
    BatchConfig,
    PathResolver,
    RunConfig,
)

def pass_type_judge(sta: str)->bool:
    return True

def setup_logging(result_path: Path, config_name: str) -> None:
    """Setup logging for a pipeline run."""
    log_file = result_path / f'autoquake_{config_name}.log'
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def run_pipeline(config: RunConfig) -> None:
    """Execute a single pipeline run based on configuration."""
    logging.info(f'Starting pipeline run: {config.name}')

    # Initialize paths
    config.result_path.mkdir(parents=True, exist_ok=True)
    resolver = PathResolver(config.result_path)

    # Track outputs for downstream components
    phase_picks = None
    gamma_events = None
    gamma_picks = None
    h3dd_dout = None
    h3dd_reorder_event = None

    # =========================================================================
    # Phase 1: PhaseNet
    # =========================================================================
    if config.is_component_enabled('PhaseNet'):
        logging.info('Running PhaseNet...')
        run_predict(config.PhaseNet.args_list)
        phase_picks = PhaseNet.concat_picks(
            date_list=config.PhaseNet.date_list,
            result_path=config.result_path
        )
        logging.info(f'PhaseNet completed. Output: {phase_picks}')

    # =========================================================================
    # Phase 2: GaMMA
    # =========================================================================
    if config.is_component_enabled('GaMMA'):
        logging.info('Running GaMMA...')

        # Resolve picks input
        if phase_picks is None:
            picks_input = resolver.resolve_picks(config.GaMMA.picks_csv)
        else:
            picks_input = phase_picks

        # Preprocessing
        # post_phasenet_pickings = gamma_preprocessing(
        #     pickings=picks_input,
        #     output_dir=config.result_path
        # )

        gamma = GaMMA(
            station=config.GaMMA.station,
            result_path=config.result_path,
            center=config.GaMMA.center,
            xlim_degree=config.GaMMA.xlim_degree,
            ylim_degree=config.GaMMA.ylim_degree,
            pickings=picks_input,
            vel_model=config.GaMMA.velocity_model,
            min_p_picks_per_eq=config.GaMMA.min_p_picks_per_eq,
            min_s_picks_per_eq=config.GaMMA.min_s_picks_per_eq,
            dbscan_eps=config.GaMMA.eps,
            ncpu=config.GaMMA.cpu_number,
            use_amplitude=config.GaMMA.use_amplitude
        )
        gamma.run_predict()
        gamma_events = gamma.get_events()
        gamma_picks = gamma.get_picks()
        logging.info('GaMMA completed.')

    # =========================================================================
    # Phase 3: H3DD
    # =========================================================================
    if config.is_component_enabled('H3DD'):
        logging.info('Running H3DD...')

        # Resolve inputs
        if gamma_events is None:
            events_input = resolver.resolve_events(config.H3DD.events_csv)
        else:
            events_input = gamma_events

        if gamma_picks is None:
            picks_input = resolver.resolve_gamma_picks(config.H3DD.picks_csv)
        else:
            picks_input = gamma_picks

        current_events = events_input
        current_picks = picks_input

        # Get number of runs from config
        run_count = config.H3DD.get_run_count()

        # Run H3DD iterations
        for run_idx in range(run_count):
            run_config = config.H3DD.get_run_config(run_idx)
            if run_config is None:
                continue

            event_name = run_config.event_name
            cutoff = run_config.cutoff_distances

            logging.info(f'H3DD run {run_idx + 1}/{run_count}: '
                        f'event_name={event_name}, cutoff={cutoff}')

            h3dd = H3DD(
                gamma_event=current_events,
                gamma_picks=current_picks,
                result_path=config.result_path,
                station=config.H3DD.station,
                model_3d=config.H3DD.model_3D,
                event_name=event_name,
                cut_off_distance_for_dd=cutoff,
                # H3DD algorithm parameters
                weights=config.H3DD.weights,
                priori_weight=config.H3DD.priori_weight,
                inv=config.H3DD.inv,
                damping_factor=config.H3DD.damping_factor,
                rmscut=config.H3DD.rmscut,
                max_iter=config.H3DD.max_iter,
                constrain_factor=config.H3DD.constrain_factor,
                joint_inv_with_single_event_method=config.H3DD.joint_inv_with_single_event_method,
                consider_elevation=config.H3DD.consider_elevation
            )
            h3dd.run_h3dd()
            h3dd_dout = h3dd.get_dout()
            h3dd_reorder_event = h3dd.get_df_reorder_event()

            # Prepare for next iteration if needed
            if run_idx < run_count - 1:
                logging.info('Processing H3DD results for next iteration...')
                current_events, current_picks = process_for_h3dd_twice(
                    station=config.H3DD.station,
                    dout=h3dd_dout,
                    event_name_1=event_name,
                    result_path=config.result_path
                )

        logging.info('H3DD completed.')

    # =========================================================================
    # Phase 4: Magnitude
    # =========================================================================
    if config.is_component_enabled('Magnitude'):
        logging.info('Running Magnitude calculation...')

        # Resolve dout input
        if h3dd_dout is None:
            dout_input = resolver.resolve_dout(config.Magnitude.dout_file)
        else:
            dout_input = h3dd_dout

        mag = Magnitude(
            dout_file=dout_input,
            station=config.Magnitude.station,
            sac_parent_dir=config.Magnitude.sac_parent_dir,
            pz_dir=config.Magnitude.pz_dir,
            output_dir=config.result_path,
        )
        mag.run_mag(processes=config.Magnitude.cpu_number)
        mag_events = mag.get_events()
        mag_picks = mag.get_picks()
        logging.info('Magnitude calculation completed.')

    # =========================================================================
    # Phase 5: Polarity (DitingMotion)
    # =========================================================================
    polarity_picks = None
    if config.is_component_enabled('Polarity'):
        logging.info('Running DitingMotion (Polarity)...')

        # Resolve picks input
        if gamma_picks is None:
            picks_input = resolver.resolve_gamma_picks(config.Polarity.picks_csv)
        else:
            picks_input = gamma_picks

        # Determine data directory
        sac_dir = config.Polarity.sac_parent_dir
        h5_dir = config.Polarity.h5_parent_dir

        if sac_dir:
            dt_polarity = DitingMotion(
                gamma_picks=picks_input,
                output_dir=config.result_path,
                sac_parent_dir=sac_dir,
                type_judge=pass_type_judge
            )
        elif h5_dir:
            dt_polarity = DitingMotion(
                gamma_picks=picks_input,
                output_dir=config.result_path,
                h5_parent_dir=h5_dir,
                type_judge=config.Polarity.type_judge
            )
        else:
            raise ValueError('Polarity config requires either sac_parent_dir or h5_parent_dir')

        dt_polarity.run_parallel_predict(processes=config.Polarity.cpu_number)
        polarity_picks = dt_polarity.get_picks()
        logging.info('DitingMotion completed.')

    # =========================================================================
    # Phase 6: Focal Mechanism (GAfocal)
    # =========================================================================
    if config.is_component_enabled('Focal'):
        logging.info('Running GAfocal...')

        # Resolve dout input
        if h3dd_dout is None:
            dout_input = resolver.resolve_dout(config.Focal.dout_file)
        else:
            dout_input = h3dd_dout

        # Format conversion with polarity and magnitude
        if polarity_picks is not None:
            logging.info('Format converting with polarity and magnitude...')
            dout_file_name = pol_mag_to_dout(
                ori_dout=dout_input,
                result_path=config.result_path,
                df_reorder_event=h3dd_reorder_event,
                polarity_picks=polarity_picks,
                magnitude_events=mag_events if 'mag_events' in dir() else None,
                magnitude_picks=mag_picks if 'mag_picks' in dir() else None
            )
        else:
            dout_file_name = dout_input.name

        gafocal = GAfocal(
            dout_file_name=dout_file_name,
            result_path=config.result_path
        )
        gafocal.run()
        logging.info('GAfocal completed.')

    logging.info(f'Pipeline run completed: {config.name}')


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AutoQuake - Automated Earthquake Catalog Generation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --config config.json
  python main.py --config config.json --dry-run
  python main.py -c params.json
        '''
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing pipeline'
    )
    return parser.parse_args()


def load_config(config_path: Path) -> BatchConfig:
    """Load and validate configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f'Configuration file not found: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    return BatchConfig(**config_dict)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        print(f'Loading configuration from: {args.config}')
        batch_config = load_config(args.config)
        print(f'Found {len(batch_config.configs)} configuration(s) to run')

        if args.dry_run:
            print('\n=== DRY RUN MODE ===')
            for i, config in enumerate(batch_config.configs, 1):
                print(f'\nConfig {i}: {config.name}')
                print(f'  Result path: {config.result_path}')
                print(f'  Components:')
                for component in ['PhaseNet', 'GaMMA', 'H3DD', 'Magnitude', 'Polarity', 'Focal']:
                    enabled = config.is_component_enabled(component)
                    status = '[+] enabled' if enabled else '[-] disabled'
                    print(f'    - {component}: {status}')
            print('\nConfiguration is valid.')
            return 0

        # Execute each configuration
        for i, config in enumerate(batch_config.configs, 1):
            print(f'\n{"=" * 60}')
            print(f'Running configuration {i}/{len(batch_config.configs)}: {config.name}')
            print(f'{"=" * 60}')

            setup_logging(config.result_path, config.name)
            run_pipeline(config)

        print('\nAll pipeline runs completed successfully.')
        return 0

    except FileNotFoundError as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1
    except ValueError as e:
        print(f'Configuration error: {e}', file=sys.stderr)
        return 1
    except Exception as e:
        print(f'Unexpected error: {e}', file=sys.stderr)
        logging.exception('Pipeline failed')
        return 1


if __name__ == '__main__':
    sys.exit(main())
