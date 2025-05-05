from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj

from .GaMMA.gamma.utils import association, estimate_eps


def config2txt(config, filename):
    with open(filename, 'w') as f:
        for key, value in config.items():
            f.write(f'{key},{value}\n')


class GaMMA:
    def __init__(
        self,
        station: Path,
        pickings: Path,
        result_path=None,
        center=None,
        xlim_degree: list | None = None,
        x_interval=3.0,
        ylim_degree: list | None = None,
        y_interval=3.0,
        zlim=(0, 60),
        degree2km=111.32,
        method='BGMM',
        use_amplitude=False,
        vel_model: Path | None = None,
        vp=6.0,
        vs=6.0 / 1.75,
        vel_h=1.0,
        use_dbscan=True,
        dbscan_min_sample=3,
        dbscan_eps: float | None = None,
        ncpu=35,
        ins_type='seis',
        min_picks_per_eq=8,
        min_p_picks_per_eq=6,
        min_s_picks_per_eq=2,
        covariance_prior: list | None = None,
        max_sigma11=2.0,
        max_sigma22=1.0,
        max_sigma12=1.0,
    ):
        """## Configuration of GaMMA

        ### Args:
            - station (Path): Path to the station file.
            - pickings (Path): Path to the picking file.
            - result_path (Path, optional): Path to the result directory. Defaults to None.
            - center (tuple, optional): Center of the study area. Defaults to None.
            - xlim_degree (tuple, optional): Longitude range of the study area. Defaults to None.
            - x_interval (float, optional): Longitude interval of the study area. Defaults to 3.0.
            - ylim_degree (tuple, optional): Latitude range of the study area. Defaults to None.
            - y_interval (float, optional): Latitude interval of the study area. Defaults to 3.0.
            - zlim (tuple, optional): Depth range of the study area. Defaults to (0, 60).
            - degree2km (float, optional): Conversion factor from degree to km. Defaults to 111.32.
            - method (str, optional): Method to use for association. Defaults to 'BGMM'.
            - use_amplitude (bool, optional): Whether to use amplitude information. Defaults to False.
            - vel_model (Path, optional): Path to the velocity model file. Defaults to None.
            - vp (float, optional): P-wave velocity. Defaults to 6.0.
            - vs (float, optional): S-wave velocity. Defaults to 6.0 / 1.75.
            - vel_h (float, optional): Interval of grid for using velocity to compute travel time. Defaults to 1.0.
            - use_dbscan (bool, optional): Whether to use DBSCAN for outlier removal. Defaults to True.
            - dbscan_min_sample (int, optional): Minimum number of samples in a neighborhood for a point to be considered as a core point of a cluster. Defaults to 3.
            - ncpu (int, optional): Number of CPUs to use. Defaults to 35.
            - ins_type (str, optional): Type of instrument. Defaults to 'seis'.
            - covariance_prior (list, optional): Prior covariance matrix, give a larger value if events are separated. Defaults to None.
            #### These arguements are used for filtering low quality events.
            - min_picks_per_eq (int, optional): Minimum number of picks per event. Defaults to 8.
            - min_p_picks_per_eq (int, optional): Minimum number of P-wave picks per event. Defaults to 6.
            - min_s_picks_per_eq (int, optional): Minimum number of S-wave picks per event. Defaults to 2.
            - max_sigma11 (float, optional): Max phase time residual (s). Defaults to 2.0.
            - max_sigma22 (float, optional): Max phase amplitude residual (in log scale). Defaults to 1.0.
            - max_sigma12 (float, optional): Max covariance term. (Usually not used). Defaults to 1.0.

        """
        self.station = station
        self.df_station = self._check_station(station)
        self.use_amplitude = use_amplitude
        self.pickings = pickings
        self.result_path = self._result_dir(result_path)
        self.center = center
        self.xlim_degree = xlim_degree
        self.x_interval = x_interval
        self.ylim_degree = ylim_degree
        self.y_interval = y_interval
        self.zlim = zlim
        self.degree2km = degree2km
        self.method = method
        self.vel_model = vel_model
        self.vel = {'p': vp, 's': vs}
        self.vel_h = vel_h
        self.use_dbscan = use_dbscan
        self.dbscan_min_sample = dbscan_min_sample
        self.dbscan_eps = dbscan_eps
        self.ncpu = ncpu
        self.ins_type = ins_type
        self.min_picks_per_eq = min_p_picks_per_eq + min_s_picks_per_eq
        self.min_p_picks_per_eq = min_p_picks_per_eq
        self.min_s_picks_per_eq = min_s_picks_per_eq
        self.covariance_prior = covariance_prior
        self.max_sigma11 = max_sigma11
        self.max_sigma22 = max_sigma22
        self.max_sigma12 = max_sigma12

    def get_picks(self) -> Path:
        """
        Get the Path of the gamma_picks.csv
        """
        return self.picks
    
    def get_events(self) -> Path:
        """
        Get the Path of the gamma_events.csv
        """
        return self.picks

    def _check_station(self, station: Path) -> pd.DataFrame:
        df = pd.read_csv(station)
        df.rename(columns={'station': 'id'}, inplace=True)
        return df

    def _check_vel_model(self, vel_model: Path):
        if self.vel_model is not None:
            return vel_model
        else:
            return Path(__file__).parents[1] / 'vel_model' / 'midas_vel.csv'

    def _check_pickings(self) -> pd.DataFrame:
        """
        Rename the dataframe and removing the invalid amplitude (-1)
        if use_amplitude == True.
        """
        df = pd.read_csv(self.pickings)

        # if self.use_amplitude:
        #     df = df[df['phase_amplitude'] != -1]
        df.rename(
            columns={
                'station_id': 'id',
                'phase_time': 'timestamp',
                'phase_type': 'type',
                'phase_score': 'prob',
                'phase_amplitude': 'amp'
            },
            inplace=True,
        )
        return df

    def _result_dir(self, result_path):
        if result_path is not None:
            return result_path
        else:
            result_path = Path(__file__).parents[1] / 'gamma_result'
            result_path.mkdir(parents=True, exist_ok=True)
            return result_path

    def _read_vel_model(self):
        """
        Reading velocity model for config.
        """
        df = pd.read_csv(self.vel_model, names=['zz', 'vp', 'vs'])
        df = df[df['zz'] <= self.zlim[1]]
        return {'z': df['zz'].values, 'p': df['vp'].values, 's': df['vs'].values}

    def _check_dbscan(self, config: dict):
        if self.use_dbscan:
            config['use_dbscan'] = self.use_dbscan
            if self.dbscan_eps is None:
                config['dbscan_eps'] = estimate_eps(self.df_station, config['vel']['p'])
                logging.info(f"DBSCAN eps: {config['dbscan_eps']}")
            else:
                config['dbscan_eps'] = self.dbscan_eps
            config['dbscan_min_samples'] = self.dbscan_min_sample
        else:
            config['use_dbscan'] = False

    def _estimate_picks_per_eq(self, config: dict):
        """
        Estimate the picks per earthquake corresponded to ins type.
        """
        if (
            self.min_picks_per_eq is None
            and self.min_p_picks_per_eq is None
            and self.min_s_picks_per_eq is None
        ):
            if self.ins_type == 'DAS':
                sta_num = len(self.df_station)
                self.min_picks_per_eq = sta_num // 10  # suggested by author.
                self.min_s_picks_per_eq = self.min_picks_per_eq // 3
                self.min_p_picks_per_eq = (
                    self.min_picks_per_eq - self.min_s_picks_per_eq
                )

        config['min_picks_per_eq'] = self.min_picks_per_eq
        config['min_p_picks_per_eq'] = self.min_p_picks_per_eq
        config['min_s_picks_per_eq'] = self.min_s_picks_per_eq

    def config_gamma(self):
        if self.center is None:
            self.center = (
                self.df_station['longitude'].mean(),
                self.df_station['latitude'].mean(),
            )
        if self.xlim_degree is None:
            self.xlim_degree = [
                self.center[0] - self.x_interval,
                self.center[0] + self.x_interval,
            ]

        if self.ylim_degree is None:
            self.ylim_degree = [
                self.center[1] - self.y_interval,
                self.center[1] + self.y_interval,
            ]

        x_km = (
            (np.array(self.xlim_degree) - np.array(self.center[0]))
            * self.degree2km
            * np.cos(np.deg2rad(self.center[1]))
        )
        y_km = (np.array(self.ylim_degree) - np.array(self.center[1])) * self.degree2km

        config = {
            'center': self.center,
            'xlim_degree': self.xlim_degree,
            'ylim_degree': self.ylim_degree,
            'degred2km': self.degree2km,
            'use_amplitude': self.use_amplitude,
            'method': self.method,
            'vel': self.vel,
            'dims': ['x(km)', 'y(km)', 'z(km)'],
            'x(km)': x_km,
            'y(km)': y_km,
            'z(km)': self.zlim,
            'ncpu': self.ncpu,
            'max_sigma11': self.max_sigma11,
            'max_sigma22': self.max_sigma22,
            'max_sigma12': self.max_sigma12,
        }

        if self.covariance_prior is not None:
            config['covariance_prior'] = self.covariance_prior

        config['bfgs_bounds'] = (
            (config['x(km)'][0] - 1, config['x(km)'][1] + 1),  # x
            (config['y(km)'][0] - 1, config['y(km)'][1] + 1),  # y
            (0, config['z(km)'][1] + 1),  # z
            (None, None),  # t
        )

        proj = Proj(
            f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km"
        )
        self.proj = proj
        self.df_station[['x(km)', 'y(km)']] = self.df_station.apply(
            lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)),
            axis=1,
        )
        self.df_station['z(km)'] = self.df_station['elevation'].apply(
            lambda x: -x / 1e3
        )

        if config['method'] == 'BGMM':  ## BayesianGaussianMixture
            config['oversample_factor'] = 5
        elif config['method'] == 'GMM':  ## GaussianMixture
            config['oversample_factor'] = 1
        else:
            raise ValueError(f'No this kind of {self.method}, please check')

        vel = self._read_vel_model()
        config['eikonal'] = {
            'vel': vel,
            'h': self.vel_h,
            'xlim': config['x(km)'],
            'ylim': config['y(km)'],
            'zlim': config['z(km)'],
        }

        self._check_dbscan(config)

        self._estimate_picks_per_eq(config)

        config2txt(config=config, filename=self.result_path / 'gamma_config.txt')

        self.config = config

    def run_predict(self):
        self.config_gamma()
        self.df_picks = self._check_pickings()
        # logging.info(f'picks_num: {self.df_picks.head(10)}')
        event_idx0 = 0  ## current earthquake index
        assignments = []
        events, assignments = association(
            self.df_picks,
            self.df_station,
            self.config,
            event_idx0,
            self.config['method']
        )
        event_idx0 += len(events)
        logging.info(f'event_num: {event_idx0}')
        ## create catalogs
        events = pd.DataFrame(events)
        events[['longitude', 'latitude']] = events.apply(
            lambda x: pd.Series(
                self.proj(longitude=x['x(km)'], latitude=x['y(km)'], inverse=True)
            ),
            axis=1,
        )
        events['depth_km'] = events['z(km)']

        self.events = self.result_path / 'gamma_events.csv'
        events.to_csv(
            self.events,
            index=False,
            float_format='%.3f',
            date_format='%Y-%m-%dT%H:%M:%S.%f',
        )

        ## add assignment to picks
        assignments = pd.DataFrame(
            assignments, columns=['pick_index', 'event_index', 'gamma_score']
        )
        picks = (
            self.df_picks.join(assignments.set_index('pick_index'))
            .fillna(-1)
            .astype({'event_index': int})
        )
        picks.rename(
            columns={
                'id': 'station_id',
                'timestamp': 'phase_time',
                'type': 'phase_type',
                'prob': 'phase_score',
                'amp': 'phase_amplitude',
            },
            inplace=True,
        )

        self.picks = self.result_path / 'gamma_picks.csv'
        picks.to_csv(
            self.picks,
            index=False,
            date_format='%Y-%m-%dT%H:%M:%S.%f',
        )
