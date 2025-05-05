import calendar
import logging
import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple  # noqa: UP035

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read
from obspy.io.sac.sacpz import attach_paz

from .utils import dmm_trans

pre_filt = (0.1, 0.5, 30, 35)

wa_simulate = {
    'poles': [(-6.28318 + 4.71239j), (-6.28318 - 4.71239j)],
    'zeros': [0j, 0j],
    'sensitivity': 1.0,
    'gain': 2800.0,
}


def check_time(year: int, month: int, day: int, hour: int, min: int, sec: float):
    """
    Adjust time by handling overflow of minutes, hours, and days.

    Args:
        year (int): Year component of the time.
        month (int): Month component of the time (1-12).
        day (int): Day component of the time (depends on month).
        hour (int): Hour component of the time (0-23).
        min (int): Minute component of the time (0-59, can overflow).
        sec (float): Seconds component of the time.

    Returns:
        tuple: Adjusted (year, month, day, hour, min, sec) considering overflow.
    """

    # Handle second overflow (if needed)
    if sec >= 60:
        min += int(sec // 60)  # Increment minutes by seconds overflow
        sec = sec % 60  # Keep remaining seconds

    # Handle minute overflow
    if min >= 60:
        hour += min // 60  # Increment hours by minute overflow
        min = min % 60  # Keep remaining minutes

    # Handle hour overflow
    if hour >= 24:
        day += hour // 24  # Increment days by hour overflow
        hour = hour % 24  # Keep remaining hours

    # Handle day overflow (check if day exceeds days in the current month)
    while day > calendar.monthrange(year, month)[1]:  # Get number of days in month
        day -= calendar.monthrange(year, month)[1]  # Subtract days in current month
        month += 1  # Increment month

        # Handle month overflow
        if month > 12:
            month = 1
            year += 1  # Increment year if month overflows

    return f'{year}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec}'


def get_phase_utc(year, month, day, hour, min, sec, phase_min, phase_sec):
    """
    This function is to check the time of the phase.
    """
    if phase_min < min:  # example: 08:00:05 07:59:59
        hour += 1
        return check_time(year, month, day, hour, phase_min, phase_sec)
    else:
        return check_time(year, month, day, hour, phase_min, phase_sec)


def utc_get_ymd(time_string: str) -> str:
    """
    Converting utc time string into YMD format.
    """
    try:
        datetime_obj = datetime.strptime(time_string, '%Y-%m-%dT%H:%M:%S.%f')
    except Exception as e:
        datetime_obj = datetime.strptime(time_string, '%Y-%m-%dT%H:%M:%S')
        logging.info(f'{e}, so we use %Y-%m-%dT%H:%M:%S')
    ymd = datetime_obj.strftime('%Y%m%d')
    return ymd


class Magnitude:
    """
    This class is to estimate the magnitude thorugh h3dd format.

    example:
    mag = Magnitude(
        dout_file=Path('/home/patrick/Work/AutoQuake/test_format/test.dout'),
        station=Path('/home/patrick/Work/EQNet/tests/hualien_0403/station_seis.csv'),
        pz_path=Path('/data/share/for_patrick/PZ_test'),
        sac_parent_dir=Path('/data2/patrick/Hualien0403/Dataset/20240401/')
    )
    """

    def __init__(
        self,
        dout_file: Path,
        station: Path,
        sac_parent_dir: Path,
        pz_dir: Path,
        output_dir: Path | None = None,
    ):
        self.dout_file = dout_file
        self.station_info = station
        self.sac_parent_dir = sac_parent_dir
        self.pz_path = pz_dir
        self.output_dir = self._check_output(output_dir)

    def get_events(self) -> Path:
        return self.events_path

    def get_picks(self) -> Path:
        return self.picks_path
    
    def _check_output(self, output):
        if output is not None:
            return output
        else:
            output = Path(__file__).parents[1].resolve() / 'mag_output'
            output.mkdir(parents=True, exist_ok=True)
            return output

    def _merge_latest(
        self,
        sta_name: str,
        comp: str,
        t1: UTCDateTime,
        t2: UTCDateTime,
        tt1: UTCDateTime | None = None,
        tt2: UTCDateTime | None = None,
    ):
        """
        Merge the latest data from the given data path for the specified station name.
        """
        if tt1 is not None and tt2 is not None:
            ymd_list = set()
            for time in [t1, t2, tt1, tt2]:
                ymd = utc_get_ymd(time.isoformat())
                ymd_list.add(ymd)
        else:
            ymd_list = set()
            for time in [t1, t2]:
                ymd = utc_get_ymd(time.isoformat())
                ymd_list.add(ymd)

        stream = Stream()
        for ymd in ymd_list:
            sac_list = list((self.sac_parent_dir / ymd).glob(f'*{sta_name}*{comp}*'))
            if not sac_list:
                continue
            for sac_file in sac_list:
                st = read(sac_file)
                stream += st
        stream = stream.merge(fill_value='latest')
        return stream

    def process_sac(self, sac_path):
        st = read(sac_path)
        return st[0].get_id().split('.')[-1]    

    def _check_ymd_comp(self, df: pd.DataFrame, station: str) -> tuple[str, set[str]]:
        time_string = df['phase_time'].iloc[0]
        ymd = utc_get_ymd(time_string)
        sac_path_list = list((self.sac_parent_dir / ymd).glob(f'*{station}*'))

        with ThreadPoolExecutor() as executor:  # Use ThreadPoolExecutor for I/O-bound tasks
            comp_list = set(executor.map(self.process_sac, sac_path_list))

        comp_list = set(i for i in comp_list if i[-1] != 'Z')
        if len(comp_list) <= 2:
            return ymd, comp_list
        elif len(comp_list) > 2:
            priority_ends = ['N', 'E']
            secondary_ends = ['1', '2']
            prioriry_elements = {i for i in comp_list if i[-1] in priority_ends}
            secondary_elements = {i for i in comp_list if i[-1] in secondary_ends}
            if len(prioriry_elements) == 2:
                return ymd, prioriry_elements
            elif len(secondary_elements) == 2:
                return ymd, secondary_elements
            else:
                return ymd, prioriry_elements

    @staticmethod
    def process_h3dd(dout_file: Path, station_info: Path):
        """
        Processing h3dd into mag ready format
        """
        with open(dout_file) as f:
            lines = f.readlines()

        mag_event_header = [
            'year',
            'month',
            'day',
            'time',
            'total_seconds',
            'longitude',
            'latitude',
            'depth_km',
            'h3dd_event_index',
        ]
        mag_picks_header = [
            'station_id',
            'phase_time',
            'total_seconds',
            'phase_type',
            'dist',
            'azimuth',
            'takeoff_angle',
            'elevation',
            'h3dd_event_index',
        ]
        df_station = pd.read_csv(
            station_info,
            dtype={
                'station': 'str',
                'longitude': 'float',
                'latitude': 'float',
                'elevation': 'float',
            },
        )
        mag_event_list = []
        mag_picks_list = []
        event_index = -1
        for line in lines:
            if line.strip()[0].isdigit():
                event_index += 1
                year = int(line[1:5].strip())
                month = int(line[5:7].strip())
                day = int(line[7:9].strip())
                hour = int(line[9:11].strip())
                min = int(line[11:13].strip())
                sec = float(line[13:19].strip())
                total_seconds = hour * 3600 + min * 60 + sec
                utctime = f'{year:4}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:05.2f}'
                lat_part = line[19:26].strip()
                lon_part = line[26:34].strip()
                event_lon = round(dmm_trans(lon_part), 3)
                event_lat = round(dmm_trans(lat_part), 3)
                depth = line[34:40].strip()
                mag_event_list.append(
                    [
                        year,
                        month,
                        day,
                        utctime,
                        total_seconds,
                        event_lon,
                        event_lat,
                        depth,
                        event_index,
                    ]
                )
            else:
                station = line[1:5].strip()
                dist = float(line[5:11].strip())
                azi = int(line[12:15].strip())
                toa = int(line[16:19].strip())
                phase_min = int(line[20:23].strip())
                p_weight = line[35:39].strip()
                s_weight = line[51:55].strip()
                elevation = (
                    df_station[df_station['station'] == station].iloc[0].elevation
                    / 1000
                )

                if p_weight == '1.00':
                    phase_sec = float(line[23:29].strip())
                    phase_type = 'P'

                elif s_weight == '1.00':
                    phase_sec = float(line[40:45].strip())
                    phase_type = 'S'
                total_seconds = hour * 3600 + phase_min * 60 + phase_sec
                phase_time = get_phase_utc(
                    year, month, day, hour, min, sec, phase_min, phase_sec
                )
                mag_picks_list.append(
                    [
                        station,
                        phase_time,
                        total_seconds,
                        phase_type,
                        dist,
                        azi,
                        toa,
                        elevation,
                        event_index,
                    ]
                )
        df_h3dd_events = pd.DataFrame(mag_event_list, columns=mag_event_header)
        df_h3dd_picks = pd.DataFrame(mag_picks_list, columns=mag_picks_header)
        return df_h3dd_events, df_h3dd_picks

    def _kinethreshold(
        self, comp_list: set[str], station: str, t1: UTCDateTime, t2: UTCDateTime
    ):
        raw_list = []
        detrend_list = []
        for comp in comp_list:
            st = self._merge_latest(station, comp, t1, t2)
            # 1. get the maxcN and mincN

            st.trim(starttime=t1, endtime=t2)

            raw_list.extend([max(st[0].data), abs(min(st[0].data))])

            # 2. get the maxnN and minnN
            st.detrend('demean')
            st.detrend('linear')
            detrend_list.extend([max(st[0].data), abs(min(st[0].data))])
        seis_type = comp[1]
        max_amp = max(raw_list)
        max_detrend_amp = max(detrend_list)
        if seis_type == 'L':
            if max_amp <= 0.1:
                logging.info(
                    f'Code 0: Amplitude of {station} with seis type {seis_type} <= 0.1: {max_amp}'
                )
                return 0
            else:
                return 1
        elif seis_type == 'S':
            if max_amp > 1:
                logging.info(
                    f'Code 0: Amplitude of {station} with seis type {seis_type} > 1: {max_amp}'
                )
                return 0
            elif max_amp < 0.0003:
                return 2
            else:
                return 1
        elif seis_type == 'H':
            if max_detrend_amp >= 5000000:
                logging.info(
                    f'Code 0: Amplitude of {station} with seis type {seis_type} >=5000000: {max_amp}'
                )
                return 0
            elif max_amp < 0.005:
                return 2
            else:
                return 1
        else:
            logging.info(f'{station} seis_type is not correct: {comp}: {seis_type}')
            return 0
            # raise ValueError(f'{station} seis_type is not correct: {comp}: {seis_type}')

    def _calculate_time_window(
        self, df_sta_picks: pd.DataFrame
    ) -> Tuple[UTCDateTime, UTCDateTime, UTCDateTime, UTCDateTime]:  # noqa: UP006
        """
        calcuate the time window for each station.

        output
            1. t1: start time of the time window for actual analysis.
            2. t2: end time of the time window for actual analysis.
            3. tt1: start time of the time window for removing response.
            4. tt2: end time of the time window for removing response.

        """
        if len(df_sta_picks) == 2:
            # N4 mode == 1
            t1 = (
                UTCDateTime(
                    df_sta_picks[df_sta_picks['phase_type'] == 'P']['phase_time'].iloc[
                        0
                    ]
                )
                - 3
            )
            t2 = (
                t1
                + (
                    df_sta_picks[df_sta_picks['phase_type'] == 'S'][
                        'total_seconds'
                    ].iloc[0]
                    - df_sta_picks[df_sta_picks['phase_type'] == 'P'][
                        'total_seconds'
                    ].iloc[0]
                )
                * 2
                + 5
            )
            if t2 - t1 > 80:
                t2 = t1 + 80
        elif len(df_sta_picks) == 1:
            if df_sta_picks['phase_type'].iloc[0] == 'P':
                # N4 mode == 2
                if float(df_sta_picks['dist'].iloc[0]) > 250:
                    t1 = UTCDateTime(df_sta_picks['phase_time'].iloc[0]) - 3
                    t2 = t1 + 80
                elif float(df_sta_picks['dist'].iloc[0]) > 100:
                    t1 = UTCDateTime(df_sta_picks['phase_time'].iloc[0]) - 3
                    t2 = t1 + 40
                else:
                    t1 = UTCDateTime(df_sta_picks['phase_time'].iloc[0]) - 3
                    t2 = t1 + 20

                pass
            elif df_sta_picks['phase_type'].iloc[0] == 'S':
                # N4 mode == 3
                t1 = UTCDateTime(df_sta_picks['phase_time'].iloc[0]) - 3
                t2 = t1 + 30
        tt1 = t1 - 100
        tt2 = t2 + 100
        return t1, t2, tt1, tt2

    def _find_max_amp(
        self,
        code: int,
        comp_list: set[str],
        station: str,
        t1: UTCDateTime,
        t2: UTCDateTime,
        tt1: UTCDateTime,
        tt2: UTCDateTime,
        mn=1000,
        pre_filt=pre_filt,
        wa_simulate=wa_simulate,
    ):
        """
        Find maximum amplitude from different scenarios followed the
        CWA (1993).
        """

        # for code == 1 & code == 2
        response_dict = {}
        for i, comp in enumerate(comp_list):
            try:
                st = self._merge_latest(station, comp, t1, t2, tt1=tt1, tt2=tt2)
            except Exception as e:
                logging.info(f'Error exist during process {station}: {e}')
                logging.info(f'Error_time: around {tt1}')
                continue
            try:
                pz_path = list(self.pz_path.glob(f'*{station}*{comp}*'))[0]
            except IndexError as e:
                logging.info(f'{station} has no pz file: {e}')
                continue
            attach_paz(tr=st[0], paz_file=str(pz_path))
            st.trim(starttime=tt1, endtime=tt2)  # cut longer for simulate
            st.simulate(paz_remove='self', paz_simulate=wa_simulate, pre_filt=pre_filt)
            if code == 2:
                st.filter('bandpass', freqmin=1, freqmax=25)
            st.trim(starttime=t1, endtime=t2)
            response_dict[i] = max(max(st[0].data), abs(min(st[0].data)))

        if comp[1] == 'L' or comp[1] == 'H':
            response_dict = {key: value * mn for key, value in response_dict.items()}

        return response_dict

    def _calculate_mag(self, response_dict: dict, dist: float, depth: float) -> float:
        """
        Using R (dist**2 + depth**2) to correct and estimate the magnitude.
        """
        nloga = math.log10(math.sqrt(response_dict[0] ** 2 + response_dict[1] ** 2))

        R = math.sqrt(dist**2 + depth**2)
        if depth <= 35:
            if 0 <= dist <= 80:
                dA = -0.00716 * R - math.log10(R) - 0.39
            elif dist > 80:
                dA = -0.00261 * R - 0.83 * math.log10(R) - 1.07
        else:
            dA = -0.00326 * R - 0.83 * math.log10(R) - 1.01

        return nloga - dA

    def get_mag(self, event_index: int, use_das=False):
        """
        Calculate suitable time window for each station in N4 determination.
        """
        logging.info(f'Event: {event_index} started')
        df_event = self.df_h3dd_events[
            self.df_h3dd_events['h3dd_event_index'] == event_index
        ].copy()
        depth = float(df_event['depth_km'].iloc[0])
        df_picks = self.df_h3dd_picks[
            (self.df_h3dd_picks['h3dd_event_index'] == event_index)
        ].copy()

        station_num = 0
        sum_mag = 0
        code_error_set = set()
        hor_comp_error_dict = {}
        for station in set(df_picks['station_id']):
            if use_das and station[1].isdigit():
                continue
            df_sta_picks = df_picks[df_picks['station_id'] == station].copy()

            t1, t2, tt1, tt2 = self._calculate_time_window(df_sta_picks)
            # TODO: check that all time is in the same day.
            ymd, comp_list = self._check_ymd_comp(df=df_sta_picks, station=station)
            # TODO: Set a scenario is that having (E, N), (1, 2) is fine, current scenario will pass the serie like {E, N ,1, 2}
            if len(comp_list) == 2:  # ensuring we have 2 horizontal component.
                try:
                    code = self._kinethreshold(
                        comp_list=comp_list, station=station, t1=t1, t2=t2
                    )
                except Exception as e:
                    logging.info(f'We encounter {e} in event {event_index}, to be more specific, in {station}')
                    continue
                if code == 0:
                    code_error_set.add(station)
                    continue
                response_dict = self._find_max_amp(
                    code=code,
                    comp_list=comp_list,
                    station=station,
                    t1=t1,
                    t2=t2,
                    tt1=tt1,
                    tt2=tt2,
                )
                print(f'{station}: {response_dict}')
                actual_depth = depth + df_sta_picks['elevation'].iloc[0]
                if response_dict:
                    # logging.info(f'Event: {event_index}, calculating mag for {station}')
                    sta_mag = self._calculate_mag(
                        response_dict=response_dict,
                        dist=float(df_sta_picks['dist'].iloc[0]),
                        depth=actual_depth,
                    )
                    station_num += 1
                    df_picks.loc[df_picks['station_id'] == station, 'magnitude'] = (
                        sta_mag
                    )
                    sum_mag += sta_mag
                else:
                    logging.info(f'{station} has no response_dict: {response_dict}')
                    continue
            else:
                hor_comp_error_dict[station] = comp_list
                continue
        logging.info(f'code == 0 stations for event{event_index}: {code_error_set}')
        logging.info(
            f'horizontal elements not enough stations for event{event_index}: {hor_comp_error_dict}'
        )
        if station_num > 0:
            sum_mag /= station_num
            df_event['magnitude'] = sum_mag
        else:
            df_event['magnitude'] = np.nan
        return df_event, df_picks

    def run_mag(self, use_das=False, processes=10):
        """
        Spawn processes to run `get_mag` for multiple event indices in parallel.
        """
        self.df_h3dd_events, self.df_h3dd_picks = self.process_h3dd(
            dout_file=self.dout_file, station_info=self.station_info
        )

        event_indices = set(self.df_h3dd_events['h3dd_event_index'])
        partial_func = partial(self.get_mag, use_das=use_das)
        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(
                partial_func, [(event_index,) for event_index in event_indices]
            )

        # Collect all events and picks into lists
        all_events, all_picks = [], []
        for event, picks in results:
            if not event.empty:
                all_events.append(event)
            if not picks.empty:
                all_picks.append(picks)

        df_event_result = pd.concat(all_events, ignore_index=True)
        self.events_path = self.output_dir / 'mag_events.csv'
        df_event_result.to_csv(self.events_path, index=False)

        df_picks_result = pd.concat(all_picks, ignore_index=True)
        self.picks_path = self.output_dir / 'mag_picks.csv'
        df_picks_result.to_csv(self.picks_path, index=False)
