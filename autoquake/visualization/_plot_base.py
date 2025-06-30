import calendar
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
from pyproj import Geod
from cartopy.mpl.geoaxes import GeoAxes
from geopy.distance import geodesic
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from obspy import Stream, Trace, UTCDateTime, read
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball, mpl_color
from scipy.interpolate import griddata
from typing import Any, Literal 
from decimal import Decimal, ROUND_HALF_UP

# ===== Sauce =====
def add_on_utc_time(time: str, delta: float) -> str:
    """
    Adjusts a UTC time string by adding or subtracting a number of seconds.

    Args:
        utc_time_str (str): The UTC time as a string in ISO format (e.g., "2024-04-02T00:02:02.190").
        delta_seconds (int): The number of seconds to add (positive) or subtract (negative).

    Returns:
        str: The adjusted UTC time in the same format.
    """
    # Parse the string into a datetime object
    utc_time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')

    # Add or subtract the time delta
    adjusted_time = utc_time + timedelta(seconds=delta)

    # Convert it back to the original string format
    return adjusted_time.strftime('%Y-%m-%dT%H:%M:%S.%f')


def polarity_color_select(polarity: str):
    if polarity == 'x':
        return 'k'
    elif polarity == 'U':
        return 'r'
    elif polarity == 'D':
        return 'b'


def normalize(x):
    """## For plotting DAS"""
    return (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)


def formatting(num: str):  # *** why we don't use the :02 in formatting
    """Format single digit numbers with a leading zero."""
    return f'0{num}' if len(num) == 1 else num


def get_total_seconds(dt: pd.Timestamp | str) -> float:
    """## Convert a datetime string into total seconds."""
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    return (dt - dt.normalize()).total_seconds()


def utc_to_timestamp(utc_string: str):
    """Converts an ISO format string to a Unix timestamp."""
    # Ensure microseconds are correctly padded
    if '.' in utc_string:
        date_part, fractional_part = utc_string.split('.')
        fractional_part = fractional_part.ljust(6, '0')  # Pad to 6 digits
        utc_string = f'{date_part}.{fractional_part}'
    return datetime.fromisoformat(utc_string).timestamp()


def quick_line_check(path):
    with open(path) as r:
        lines = r.readlines()
    counter = 0
    for i in lines:
        if i.strip()[:4] == '2024':
            counter += 1
    return counter


def h3dd_time_trans(x: str):
    """Converting the h3dd time format into total seconds in a day."""
    return int(x[:2]) * 3600 + int(x[2:4]) * 60 + int(x[4:6]) + float(f'0.{x[6:]}')


def convert_channel_index(sta_name: str) -> int:
    """Convert first character of channel from alpha to digit for indexing."""
    if sta_name[:1] == 'A':
        channel_index = int(sta_name[1:])
    elif sta_name[:1] == 'B':
        channel_index = int(f'1{sta_name[1:]}')
    else:
        print('wrong format warning: please append the condition')
    return channel_index


def station_mask(x: str):
    """Pattern to distinguish the DAS and Seismometer station."""
    return x[1].isalpha()


def dmm_trans(coord: str):
    """
    Convert degree and minute format (e.g., '2523.47') to decimal degrees.
    :param coord: Coordinate in degree and minute format as a string.
    :return: Decimal degrees as a float.
    """
    try:
        # Split into degrees and minutes
        degrees = int(coord[:-5])  # Extract the degree part (all but last 5 chars)
        minutes = float(coord[-5:])  # Extract the minute part (last 5 chars)

        # Convert to decimal degrees
        return degrees + (minutes / 60)
    except (ValueError, IndexError):
        raise ValueError(f'Invalid coordinate format: {coord}')


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

    return year, month, day, hour, min, sec


def check_hms_h3dd(hms: str):
    """check whether the h3dd format's second overflow"""
    minute = int(hms[2:4])
    second = int(hms[4:6])

    if second >= 60:
        minute += second // 60
        second = second % 60

    fixed_hms = hms[:2] + f'{minute:02d}' + f'{second:02d}' + hms[6:]
    return fixed_hms


def _cal_dist(event_point: tuple, sta_point: tuple):
    """## round distance between station and event."""
    dist = geodesic(event_point, sta_point).km
    dist_round = np.round(dist, 1)
    return dist_round


def check_hms_gafocal(hms: str):
    """
    check whether the gafocal format's second overflow
    """
    minute = int(hms[3:5])
    second = int(hms[6:8])

    if second >= 60:
        minute += second // 60
        second = second % 60

    fixed_hms = hms[:3] + f'{minute:02d}' + hms[5:6] + f'{second:02d}'
    return fixed_hms


def _txt_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distinguish h3dd and gafocal format through YYYYMMDD & YYYY/MM/DD.
    """
    if len(df[0].iloc[0]) == 8:  # h3dd
        df[1] = df[1].apply(check_hms_h3dd)
        df['time'] = (
            df[0].apply(lambda x: f'{x[:4]}-{x[4:6]}-{x[6:8]}')
            + 'T'
            + df[1].apply(lambda x: f'{x[:2]}:{x[2:4]}:{x[4:6]}.{x[6:8]}0000')
        )
        df = df.rename(columns={2: 'latitude', 3: 'longitude', 4: 'depth_km', 5: 'magnitude'})
        mask = [i for i in df.columns.tolist() if isinstance(i, str)]
        df = df[mask]
        return df
    elif '/' in df[0].iloc[0]:  # gafocal
        df[1] = df[1].apply(check_hms_gafocal)
        df['time'] = df[0].apply(lambda x: x.replace('/', '-')) + 'T' + df[1]
        for col in [6, 8, 10]:
            df[col] = df[col].map(lambda x: int(x.split('+')[0]))
        df = df.rename(
            columns={
                2: 'longitude',
                3: 'latitude',
                4: 'depth_km',
                5: 'magnitude',
                6: 'strike',
                7: 'strike_err',
                8: 'dip',
                9: 'dip_err',
                10: 'rake',
                11: 'rake_err',
                12: 'quality_index',
                13: 'num_of_polarity',
            }
        )
        mask = [i for i in df.columns.tolist() if isinstance(i, str)]
        df = df[mask]
        return df
    else:
        raise ValueError(f'Unrecognized date format: {df[0].iloc[0]}')


def _merge_latest(data_path: Path, sta_name: str):
    """
    Merge the latest data from the given data path for the specified station name.
    """
    sac_list = list(data_path.glob(f'*{sta_name}.*Z*'))
    if not sac_list:
        # logging.info(f'{sta_name} using other component')
        sac_list = list(data_path.glob(f'*{sta_name}.*'))
        # logging.info(f'comp_list: {sac_list}')
    stream = Stream()
    for sac_file in sac_list:
        st = read(sac_file)
        stream += st
    stream = stream.merge(fill_value='latest')
    return stream


def get_max_columns(file_path):
    max_columns = 0

    # Open the file and analyze each line
    with open(file_path) as file:
        for line in file:
            # Split the line using the space delimiter and count the columns
            columns_in_line = len(line.split())
            # Update max_columns if this line has more columns
            if columns_in_line > max_columns:
                max_columns = columns_in_line

    return max_columns


def check_format(catalog, i=None):
    """
    Checking the format is GaMMA (events_catalog.csv), h3dd (.hout), gafocal (.txt), and
    converting it into union format.
    ---
    Returns
        df (pd.DataFrame): format converted catalog.
        timestamp (numpy array): total seconds of the events from catalog.
    """
    if isinstance(catalog, dict):
        file_path = catalog[i]['catalog']
    elif isinstance(catalog, Path):
        file_path = catalog
    else:
        raise ValueError('Invalid input. Provide either a dictionary or a Path object.')

    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        max_columns = get_max_columns(file_path)

        cols_to_read = list(range(max_columns - 1))
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=None,
            dtype={0: 'str', 1: 'str'},
            usecols=cols_to_read,
        )
        df = _txt_preprocessor(df)
    timestamp = df['time'].apply(utc_to_timestamp).to_numpy()
    return df, timestamp


def catalog_filter(
    catalog_df: pd.DataFrame,
    h3dd_mode=True,
    catalog_range: dict[str, float] | None = None,
    prefix='',
) -> pd.DataFrame:
    if h3dd_mode:
        prefix = 'h3dd_'
    if catalog_range is not None:
        catalog_df = catalog_df[
            (catalog_df[f'{prefix}longitude'] > catalog_range['min_lon'])
            & (catalog_df[f'{prefix}longitude'] < catalog_range['max_lon'])
            & (catalog_df[f'{prefix}latitude'] > catalog_range['min_lat'])
            & (catalog_df[f'{prefix}latitude'] < catalog_range['max_lat'])
            & (catalog_df[f'{prefix}depth_km'] > catalog_range['min_depth'])
            & (catalog_df[f'{prefix}depth_km'] < catalog_range['max_depth'])
        ].copy()

    return catalog_df


# ===== preprocess function =====


def _process_midas_scenario(time_str: str, interval=300):
    datetime = pd.to_datetime(time_str)
    ymd = f'{datetime.year}{datetime.month:>02}{datetime.day:>02}'
    total_seconds = get_total_seconds(datetime)
    hdf5_index = int(total_seconds // interval)
    sec = total_seconds % interval
    return ymd, hdf5_index, sec

def round_timestamp(timestamp: str) -> str:
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")

    # Convert microseconds to milliseconds with correct rounding
    milliseconds = Decimal(dt.microsecond / 1_000_0).quantize(Decimal('1.'), rounding=ROUND_HALF_UP)

    # Construct the final string with correctly rounded milliseconds
    return dt.strftime(f"%Y-%m-%dT%H:%M:%S.{int(milliseconds):<02}")


def preprocess_gamma_csv(
    gamma_catalog: Path, gamma_picks: Path, event_i: int, h3dd_hout=None
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    ## Preprocessing the DataFrame.

    This function takes the Path object from catalog and picks generated
    by GaMMA, creating the DataFrame object for later utilization.

    Parameters
    ----------
    gamma_catalog : Path
        The path of gamma_events.csv
    gamma_picks : Path
        The path of gamma_picks.csv

    Returns
    -------
    df_event : DataFrame
        The DataFrame object of the gamma events i's event information.
    event_dict : dict
        The dictionary contains the needed event information.
    df_event_picks : DataFrame
        The DataFrame object of the gamma events i's associated picks.
    h3dd_hout (optional, if you want to use h3dd info) : Path
        The path of h3dd.hout
    """
    df_catalog = pd.read_csv(gamma_catalog)
    df_catalog.loc[:, 'datetime'] = pd.to_datetime(df_catalog['time'])
    df_catalog = df_catalog.sort_values(by='datetime', ignore_index=True)
    df_event = df_catalog[df_catalog['event_index'] == event_i].copy()
    df_event.loc[:, 'ymd'] = df_event['datetime'].dt.strftime('%Y%m%d')
    df_event.loc[:, 'hour'] = df_event['datetime'].dt.hour
    df_event.loc[:, 'minute'] = df_event['datetime'].dt.minute
    df_event.loc[:, 'seconds'] = (
        df_event['datetime'].dt.second + df_event['datetime'].dt.microsecond / 1_000_000
    )
    event_dict = {}
    event_dict[
        'gamma'
    ] = {}  # the reason I don't want to use defaultdict is to limit it.
    event_dict['gamma'] = {
        'date': df_event['ymd'].iloc[0],
        'event_time': round_timestamp(df_event['time'].iloc[0]),
        'event_total_seconds': get_total_seconds(df_event['datetime'].iloc[0]),
        'event_lat': df_event['latitude'].iloc[0],
        'event_lon': df_event['longitude'].iloc[0],
        'event_point': (df_event['latitude'].iloc[0], df_event['longitude'].iloc[0]),
        'event_depth': df_event['depth_km'].iloc[0],
    }
    if h3dd_hout is not None:
        event_dict['h3dd'] = {}
        df_h3dd, _ = check_format(h3dd_hout)
        df_h3dd['datetime'] = pd.to_datetime(df_h3dd['time'])
        df_h3dd['datetime'].map(get_total_seconds)

        h3dd_index = df_event.index[0]
        df_h3dd = df_h3dd.iloc[h3dd_index]
        event_dict['h3dd'] = {
            'event_time': df_h3dd['time'],
            'event_total_seconds': get_total_seconds(df_h3dd['datetime']),
            'event_lat': df_h3dd['latitude'],
            'event_lon': df_h3dd['longitude'],
            'event_point': (df_h3dd['latitude'], df_h3dd['longitude']),
            'event_depth': df_h3dd['depth_km'],
        }
    #print(event_dict)
    df_picks = pd.read_csv(gamma_picks)
    df_event_picks = df_picks[df_picks['event_index'] == event_i].copy()
    # TODO: does there have any possible scenario?
    # df_event_picks['station_id'] = df_event_picks['station_id'].map(lambda x: str(x).split('.')[1])
    df_event_picks.loc[:, 'phase_time'] = pd.to_datetime(df_event_picks['phase_time'])

    return df_event, event_dict, df_event_picks


def _preprocess_phasenet_csv(
    phasenet_picks: Path, get_station, starttime: str | None = None, endtime: str | None = None
) -> pd.DataFrame:
    """## Preprocess the phasenet_csv
    Using get_station to retrieve the station name in station_id column.
    """
    df_all_picks = pd.read_csv(phasenet_picks)
    df_all_picks['phase_time'] = pd.to_datetime(df_all_picks['phase_time'])
    if starttime is not None and endtime is not None:
        df_all_picks = df_all_picks[(df_all_picks['phase_time']>=pd.Timestamp(starttime)) & (df_all_picks['phase_time']<=pd.Timestamp(endtime))]
    df_all_picks['total_seconds'] = df_all_picks['phase_time'].apply(get_total_seconds)
    # df_all_picks['system'] = df_all_picks['station_id'].apply(
    #     lambda x: str(x).split('.')[0]
    # )  # TW or MiDAS blablabla
    df_all_picks['station'] = df_all_picks['station_id'].apply(
        get_station
    )  # LONT (station name) or A001 (channel name)
    return df_all_picks


def _process_polarity(
    polarity_picks: Path, event_index: int, station_mask=station_mask
):
    """
    ### TODO:Actually station mask should not exist here, this place should already contains
    the data of DAS.
    """
    df_pol = pd.read_csv(polarity_picks)
    df_pol.rename(columns={'station_id': 'station'}, inplace=True)
    df_pol = df_pol[df_pol['event_index'] == event_index]
    df_pol = df_pol[df_pol['station'].map(station_mask)]
    return df_pol


def _midas_scenario(
    phase_time: str, hdf5_parent_dir: Path, interval=300
) -> None | Path:
    total_seconds = get_total_seconds(pd.to_datetime(phase_time))
    index = int(total_seconds // interval)
    window = f'{interval*index}_{interval*(index+1)}.h5'
    try:
        file = list(hdf5_parent_dir.glob(f'*{window}'))[0]
        return file
    except IndexError:
        logging.info(f'File not found for window {window}')
        return None


def _process_das_polarity(
    file: Path,
    channel_index: int,
    phase_time: str,
    train_window=0.64,
    visual_window=2.0,
    sampling_rate=100,
) -> tuple:
    tr = Trace()
    try:
        with h5py.File(file, 'r') as fp:
            ds = fp['data']
            data = ds[channel_index]
            tr.stats.sampling_rate = 1 / ds.attrs['dt_s']
            tr.stats.starttime = ds.attrs['begin_time']
            tr.data = data
        p_arrival = UTCDateTime(phase_time)
        train_starttime_trim = p_arrival - train_window
        train_endtime_trim = p_arrival + train_window
        window_starttime_trim = p_arrival - visual_window
        window_endtime_trim = p_arrival + visual_window

        tr.detrend('demean')
        tr.detrend('linear')
        tr.filter('bandpass', freqmin=1, freqmax=10)
        tr.taper(0.001)
        tr.resample(sampling_rate=sampling_rate)
        # this is for visualize length
        tr.trim(starttime=window_starttime_trim, endtime=window_endtime_trim)
        visual_time = np.arange(
            0, 2 * visual_window + 1 / sampling_rate, 1 / sampling_rate
        )  # using array to ensure the time length as same as time_window.
        visual_sac = tr.data
        # this is actual training length
        tr.trim(starttime=train_starttime_trim, endtime=train_endtime_trim)
        start_index = visual_window - train_window
        train_time = np.arange(
            start_index,
            start_index + 2 * train_window + 1 / sampling_rate,
            1 / sampling_rate,
        )  # using array to ensure the time length as same as time_window.
        train_sac = tr.data
        return p_arrival, visual_time, visual_sac, train_time, train_sac
    except Exception as e:
        logging.info(f'Error {e} during hdf5 and obspy process')
        return ()


# ===== Find function =====


def find_gamma_h3dd(gamma_events: Path, h3dd_events: Path, event_index: int):
    """Using gamma index to find the h3dd event."""
    df_gamma_events = pd.read_csv(gamma_events)
    h3dd_event_index = df_gamma_events[df_gamma_events['event_index'] == event_index][
        'h3dd_event_index'
    ].iloc[0]

    df_h3dd, _ = check_format(catalog=h3dd_events)
    df_h3dd.loc[h3dd_event_index]
    return df_h3dd


def find_gamma_index(gamma_events: Path, h3dd_index: int):
    """Using h3dd index to find the gamma event."""
    df_gamma_events = pd.read_csv(gamma_events)
    gamma_event_index = df_gamma_events[
        df_gamma_events['h3dd_event_index'] == h3dd_index
    ]['event_index'].iloc[0]

    return gamma_event_index


def find_das_data(
    event_index: int,
    hdf5_parent_dir: Path,
    polarity_picks: Path,
    interval=300,
    train_window=0.64,
    visual_window=2,
    sampling_rate=100,
    station_mask=station_mask,
):
    """## find the das waveform for plotting polarity waveform.
    ### TODO: We use MiDAS Naming idiom to design the searching way, it should be flexible with
    other format, but that's what we need to improve!
    """
    df_pol = _process_polarity(polarity_picks, event_index, station_mask)
    das_plot_dict = {}
    for _, row in df_pol.iterrows():
        # Polarity determination
        if row.polarity == 'U':
            polarity = '+'
        elif row.polarity == 'D':
            polarity = '-'
        else:
            polarity = ' '

        # file scenario
        file = _midas_scenario(row.phase_time, hdf5_parent_dir, interval)
        if file is None:
            continue

        # Convert A001 back to 1001 for hdf5 key.
        channel_index = convert_channel_index(row.station)
        try:
            p_arrival, visual_time, visual_sac, train_time, train_sac = (
                _process_das_polarity(
                    file=file,
                    channel_index=channel_index,
                    phase_time=row.phase_time,
                    train_window=train_window,
                    visual_window=visual_window,
                    sampling_rate=sampling_rate,
                )
            )
        except Exception as e:
            logging.info(f'Error: {e}, because it returns a tuple')
            continue

        # final writing
        if 'station_info' not in das_plot_dict:
            das_plot_dict['station_info'] = {}
        das_plot_dict['station_info'][row.station] = {
            'polarity': polarity,
            'p_arrival': p_arrival,
            'visual_time': visual_time,
            'visual_sac': visual_sac,
            'train_time': train_time,
            'train_sac': train_sac,
        }
    return das_plot_dict

def calculate_3d_distance(
    event_lon: float,
    event_lat: float,
    event_depth: float,
    sta_lon: float,
    sta_lat: float,
    sta_elevation: float
):
    """
    Calculate the 3D distance between two points given lon, lat, and depth.

    Returns:
        3D distance in km.
    """
    geod = Geod(ellps='WGS84')
    # Calculate 2D geodetic distance
    _, _, distance_2d = geod.inv(sta_lon, sta_lat, event_lon, event_lat)

    # Calculate the depth difference
    depth_diff = event_depth + sta_elevation / 1000

    # Calculate 3D distance using Pythagoras theorem
    distance_3d = math.sqrt((distance_2d / 1000) ** 2 + depth_diff**2)

    return distance_3d

def find_sac_data(
    event_time: str,
    date: str,  # redundant, but already do it in previous function, temp pass.
    event_lon: float,
    event_lat: float,
    event_depth: float,
    station: Path,
    sac_parent_dir: Path,
    amplify_index: float,
    station_mask,
    pretime,
    posttime,
) -> dict:
    """
    ## Retrieve the waveform from SAC.

    This function glob the sac file from each station, multiplying the
    amplify index to make the PGA/PGV more clear, then packs the
    waveform data into dictionary for plotting.

    This is for plotting association plot!
    """
    starttime_trim = UTCDateTime(event_time) + pretime
    endtime_trim = UTCDateTime(event_time) + posttime
    # logging.info(f'time period{starttime_trim, endtime_trim}')
    df_station = pd.read_csv(station)
    df_station = df_station[df_station['station'].map(station_mask)]

    sac_path = sac_parent_dir / date
    sac_dict = {}
    #TODO: This is not a good way, taking more time to go through all station.
    for sta in df_station['station'].to_list():
        # glob the waveform

        st = _merge_latest(data_path=sac_path, sta_name=sta)
        if len(st) == 0:
            continue
        # logging.info(f'!BEFORE TRIM!{st}')
        st.taper(type='hann', max_percentage=0.05)
        st.filter('bandpass', freqmin=1, freqmax=10)
        try:
            st[0].trim(starttime=starttime_trim, endtime=endtime_trim)
        except Exception as e:
            logging.info(f'Trimming error in {sta} find_sac_data: {e}')
            continue
        sampling_rate = 1 / st[0].stats.sampling_rate
        time_sac = np.arange(
            0, (posttime - pretime) + sampling_rate, sampling_rate
        )  # using array to ensure the time length as same as time_window.
        x_len = len(time_sac)
        try:
            data_sac_raw = st[0].data / max(st[0].data)  # normalize the amplitude.
        except Exception as e:
            logging.info(f'Check {sta} whether in gamma_picks')
            continue

        sta_lon = df_station[df_station['station'] == sta]['longitude'].iloc[0]
        sta_lat = df_station[df_station['station'] == sta]['latitude'].iloc[0]
        sta_elevation = df_station[df_station['station'] == sta]['elevation'].iloc[0]
        dist = calculate_3d_distance(
            event_lon=event_lon, event_lat=event_lat, event_depth=event_depth,
            sta_lon=sta_lon, sta_lat=sta_lat, sta_elevation=sta_elevation
        )
        # dist = _cal_dist(event_point=event_point, sta_point=sta_point)
        data_sac_raw = data_sac_raw * amplify_index + dist
        # logging.info(f'{sta} sac length: {len(data_sac_raw)}')
        # logging.info(f'{sta} time length: {x_len} with sampling rate {sampling_rate}')
        # we might have the data lack in the beginning:
        if starttime_trim < st[0].stats.starttime:
            data_sac = np.pad(
                data_sac_raw,
                (x_len - len(data_sac_raw), 0),
                mode='constant',
                constant_values=np.nan,
            )  # adding the Nan to ensure the data length as same as time window.
        else:
            # try:
            data_sac = np.pad(
                data_sac_raw,
                (0, x_len - len(data_sac_raw)),
                mode='constant',
                constant_values=np.nan,
            )  # adding the Nan to ensure the data length as same as time window.
            # except Exception as e:
            #     logging.error(
            #         f'{sta} Padding value: {x_len} (x_len) - {len(data_sac_raw)} (data_sac_raw)'
            #     )
            #     logging.error(f'Error: {e} during concat in {sta}')
                
        sac_dict[str(sta)] = {
            'time': time_sac,
            'sac_data': data_sac,
            'distance': dist,
        }
    return sac_dict


def find_gamma_pick(
    df_gamma_picks: pd.DataFrame,
    sac_dict: dict,
    event_total_seconds: float,
    first_half=30,
    dx=4.084,
    dt=0.01,
    hdf5_time=300,
    station_mask=station_mask,
):
    """
    Preparing the associated picks for scatter plot.
    """
    df_das_aso_picks = df_gamma_picks[~df_gamma_picks['station_id'].apply(station_mask)]
    if not df_das_aso_picks.empty:
        df_das_aso_picks['channel_index'] = df_das_aso_picks['station_id'].apply(
            convert_channel_index
        )
        df_das_aso_picks['x'] = (
            df_das_aso_picks['station_id'].apply(convert_channel_index) * dx
        )
        df_das_aso_picks['y'] = df_das_aso_picks['phase_index'].apply(
            lambda x: x * dt + hdf5_time if x * dt - first_half < 0 else x * dt
        )

    df_seis_aso_picks = df_gamma_picks[df_gamma_picks['station_id'].apply(station_mask)]
    if not df_seis_aso_picks.empty:
        df_seis_aso_picks['total_seconds'] = df_seis_aso_picks['phase_time'].apply(
            get_total_seconds
        )
        df_seis_aso_picks['x'] = (
            df_seis_aso_picks['total_seconds'] - event_total_seconds + first_half
        )  # Because we use -30 as start.
        df_seis_aso_picks['y'] = df_seis_aso_picks['station_id'].map(
            lambda x: sac_dict[x]['distance']
        )

    return df_seis_aso_picks, df_das_aso_picks


def _find_phasenet_das_pick(
    starttime: str,
    endtime: str,
    df_picks: pd.DataFrame,  # why this place using dataframe? because we don't want to read it several time.
    station_mask,
    dx=4.084,
    dt=0.01,
    das_time_interval=300,
):
    """
    Filtering waveform in specific time window and convert it for scatter plot.
    """
    df_picks = df_picks[
        (df_picks['phase_time'] >= pd.Timestamp(starttime))
        & (df_picks['phase_time'] <= pd.Timestamp(endtime))
    ]
    _, _, start_sec = _process_midas_scenario(starttime, interval=das_time_interval)

    df_das_picks = df_picks[~df_picks['station'].map(station_mask)]
    if df_das_picks.empty:
        return pd.DataFrame()
    # df_das_picks['channel_index'] = df_das_picks['station'].map(
    #     convert_channel_index
    # )
    df_das_picks['x'] = df_das_picks['station'].map(convert_channel_index) * dx
    df_das_picks['y'] = df_das_picks['phase_index'].map(
        lambda x: x * dt + das_time_interval if x * dt - start_sec < 0 else x * dt
    )  # 0.01 = dt

    return df_das_picks


def _find_phasenet_seis_pick(
    starttime: str,
    endtime: str,
    df_picks: pd.DataFrame,  # why this place using dataframe? because we don't want to read it several time.
    station_mask,
    sac_dict={},
    picking_check=True,
    asso_check=False
):
    """
    Filtering waveform in specific time window and convert it for scatter plot.
    """
    df_picks = df_picks[
        (df_picks['phase_time'] >= pd.Timestamp(starttime))
        & (df_picks['phase_time'] <= pd.Timestamp(endtime))
    ]
    start_datetime = pd.to_datetime(starttime)
    start_total_seconds = get_total_seconds(start_datetime)

    df_seis_picks = df_picks[df_picks['station'].map(station_mask)]
    df_seis_picks['x'] = (
        df_seis_picks['total_seconds'] - start_total_seconds
    )  # Because we use -30 as start.
    if picking_check:
        df_seis_picks['y'] = 0.0
    elif asso_check:
        df_seis_picks['y'] = df_seis_picks['station'].map(
            lambda x: sac_dict[x]['distance']
        )

    return df_seis_picks


## For focal mechanism visualization.
def _hout_generate(
    polarity_dout: Path, event_filter=lambda x: str(x)[0].isdigit()
) -> pd.DataFrame:
    """
    Because the CWA polarity dout did not have the corresponded hout file,
    so create a DataFrame with the same format as hout file.
    """
    with open(polarity_dout) as r:
        lines = r.readlines()
    data = []
    for line in lines:
        if event_filter(line.strip()):
            year = int(line[1:5].strip())
            month = int(line[5:7].strip())
            day = int(line[7:9].strip())
            hour = int(line[9:11].strip())
            min = int(line[11:13].strip())
            second = float(line[13:19].strip())
            year, month, day, hour, min, second = check_time(
                year, month, day, hour, min, second
            )
            time = f'{year:4}-{month:02}-{day:02}T{hour:02}:{min:02}:{second:09.6f}'
            lat_part = line[19:26].strip()
            lon_part = line[26:34].strip()
            event_lon = round(dmm_trans(lon_part), 3)
            event_lat = round(dmm_trans(lat_part), 3)
            depth = line[34:40].strip()
            data.append([time, event_lat, event_lon, depth])
    columns = ['time', 'latitude', 'longitude', 'depth']
    df = pd.DataFrame(data, columns=columns)
    return df


def _find_pol_waveform_seis(
    sta: str,
    sac_parent_dir: Path,
    date: str,
    p_arrival: UTCDateTime,
    train_window=0.64,
    visual_window=2,
    sampling_rate=100,
):
    sac_path = sac_parent_dir / date
    train_starttime_trim = p_arrival - train_window
    train_endtime_trim = p_arrival + train_window
    window_starttime_trim = p_arrival - visual_window
    window_endtime_trim = p_arrival + visual_window
    try:
        # TODO: filtering like using 00 but not 10.
        st = _merge_latest(sac_path, sta)
        st.detrend('demean')
        st.detrend('linear')
        st.filter('bandpass', freqmin=1, freqmax=10)
        st.taper(0.001)
        st.resample(sampling_rate=sampling_rate)
        # this is for visualize length
        st[0].trim(starttime=window_starttime_trim, endtime=window_endtime_trim)
        visual_time = np.arange(
            0, 2 * visual_window + 1 / sampling_rate, 1 / sampling_rate
        )  # using array to ensure the time length as same as time_window.
        visual_sac = st[0].data
        # this is actual training length
        st[0].trim(starttime=train_starttime_trim, endtime=train_endtime_trim)
        start_index = visual_window - train_window
        train_time = np.arange(
            start_index,
            start_index + 2 * train_window + 1 / sampling_rate,
            1 / sampling_rate,
        )  # using array to ensure the time length as same as time_window.
        train_sac = st[0].data
    except Exception:
        logging.info(f"we can't access the {sta}")
        return 0, 0, 0, 0
    return visual_time, visual_sac, train_time, train_sac


def _find_pol_waveform_das(
    sta: str,
    date: str,
    p_arrival: UTCDateTime,
    total_seconds: float,
    hdf5_parent_dir: Path,
    interval=300,
    train_window=0.64,
    visual_window=2,
    sampling_rate=100,
):
    index = int(total_seconds // interval)
    window = f'{interval*index}_{interval*(index+1)}.h5'
    h5_dir = hdf5_parent_dir / f'{date}_hdf5'
    try:
        file = list(h5_dir.glob(f'*{window}'))[0]
    except IndexError:
        logging.info(f'File not found for window {window}')
    channel_index = convert_channel_index(sta)
    tr = Trace()
    try:
        with h5py.File(file, 'r') as fp:
            ds = fp['data']
            data = ds[channel_index]
            tr.stats.sampling_rate = 1 / ds.attrs['dt_s']
            tr.stats.starttime = ds.attrs['begin_time']
            tr.data = data
        train_starttime_trim = p_arrival - train_window
        train_endtime_trim = p_arrival + train_window
        window_starttime_trim = p_arrival - visual_window
        window_endtime_trim = p_arrival + visual_window

        tr.detrend('demean')
        tr.detrend('linear')
        tr.filter('bandpass', freqmin=1, freqmax=10)
        tr.taper(0.001)
        tr.resample(sampling_rate=sampling_rate)
        # this is for visualize length
        tr.trim(starttime=window_starttime_trim, endtime=window_endtime_trim)
        visual_time = np.arange(
            0, 2 * visual_window + 1 / sampling_rate, 1 / sampling_rate
        )  # using array to ensure the time length as same as time_window.
        visual_sac = tr.data
        # this is actual training length
        tr.trim(starttime=train_starttime_trim, endtime=train_endtime_trim)
        start_index = visual_window - train_window
        train_time = np.arange(
            start_index,
            start_index + 2 * train_window + 1 / sampling_rate,
            1 / sampling_rate,
        )  # using array to ensure the time length as same as time_window.
        train_sac = tr.data
    except Exception as e:
        print(e)
        return 0, 0, 0, 0
    return visual_time, visual_sac, train_time, train_sac


def _find_station_focal(
    polarity_dout: Path,
    event_index: int,
    focal_dict: dict,
    equip_filter,
    event_filter,
    get_waveform=False,
    sac_parent_dir: Path | None = None,
    h5_parent_dir: Path | None = None,
):
    """## This is the private function to append azimuth, take-off angle, and polarity.
    Using get_waveform == True to further acquire the waveform data for validating the
    result of polarity.

    Args:
        - Event_index: the index of the event in the polarity_dout file, which is h3dd_index.
    """
    with open(polarity_dout) as r:
        lines = r.read().splitlines()
    counter = -1
    for line in lines:
        if event_filter(line.strip()):
            counter += 1
            if counter == event_index:
                year = int(line[:5].strip())
                month = int(line[5:7].strip())
                day = int(line[7:9].strip())
                hour = int(line[9:11].strip())
                date = f'{year}{month:>02}{day:>02}'
            # TODO Testify the event time again
        elif counter == event_index:
            sta = line[1:5].strip()
            azi = int(line[12:15].strip())
            toa = int(line[16:19].strip())
            polarity = line[19:20]
            p_min = int(line[20:23].strip())
            try:
                p_sec = float(line[23:29].strip())
            except Exception:
                logging.info(f'line: {line}')
                raise ValueError('dout in wrong format, see the log.')
            if 'station_info' not in focal_dict:
                focal_dict['station_info'] = {}
            year, month, day, hour, p_min, p_sec = check_time(
                year, month, day, hour, p_min, p_sec
            )
            p_arrival = UTCDateTime(year, month, day, hour, p_min, p_sec)

            if not get_waveform:
                focal_dict['station_info'][sta] = {
                    'p_arrival': p_arrival,
                    'azimuth': azi,
                    'takeoff_angle': toa,
                    'polarity': polarity,
                }
                continue

            if equip_filter(sta) and sac_parent_dir is not None:
                visual_time, visual_sac, train_time, train_sac = (
                    _find_pol_waveform_seis(sta, sac_parent_dir, date, p_arrival)
                )
            elif h5_parent_dir is not None:
                total_seconds = hour * 3600 + p_min * 60 + p_sec
                visual_time, visual_sac, train_time, train_sac = _find_pol_waveform_das(
                    sta, date, p_arrival, total_seconds, h5_parent_dir
                )
            else:
                raise ValueError(
                    'Please check the equip_filter, and please provide the corresponded data Path.'
                )

            # final writing
            focal_dict['station_info'][sta] = {
                'p_arrival': p_arrival,
                'azimuth': azi,
                'takeoff_angle': toa,
                'polarity': polarity,
                'visual_time': visual_time,
                'visual_sac': visual_sac,
                'train_time': train_time,
                'train_sac': train_sac,
            }

def _find_station_focal_new(
    polarity_picks: Path,
    polarity_dout: Path,
    event_index: int,
    focal_dict: dict,
    equip_filter,
    event_filter,
    get_waveform=False,
    sac_parent_dir: Path | None = None,
    h5_parent_dir: Path | None = None,
):
    """## This is the latest version by using csv but not lines
    Using get_waveform == True to further acquire the waveform data for validating the
    result of polarity.

    Args:
        - Event_index: the index of the event in the polarity_dout file, which is h3dd_index.
    """
    df = pd.read_csv(polarity_picks)
    df = df[df['h3dd_event_index'] == event_index].copy()
    pol_dict = {
        'D': '-',
        'U': '+',
        'x': ' '
    }
    with open(polarity_dout) as r:
        lines = r.read().splitlines()
    counter = -1
    for line in lines:
        if event_filter(line.strip()):
            counter += 1
        elif counter == event_index:
            sta = line[1:5].strip()
            azi = int(line[12:15].strip())
            toa = int(line[16:19].strip())
            polarity = line[19:20]
            if 'station_info' not in focal_dict:
                focal_dict['station_info'] = {}

            focal_dict['station_info'][sta] = {
                'azimuth': azi,
                'takeoff_angle': toa,
                'polarity': polarity,
            }
    if 'station_info' not in focal_dict:
        focal_dict['station_info'] = {}

    if not get_waveform:
        for row in df.itertuples():
            sta = row.station_id
            phase_time = row.phase_time
            phase_score = row.phase_score
            focal_dict['station_info'][sta]['p_arrival'] = UTCDateTime(phase_time)

    else:
        for row in df.itertuples():
            sta = row.station_id
            phase_time = row.phase_time
            date = phase_time[:4] + phase_time[5:7] + phase_time[8:10]
            print(date)
            p_arrival = UTCDateTime(phase_time)
            phase_score = row.phase_score
            if sac_parent_dir is not None:
                visual_time, visual_sac, train_time, train_sac = (
                    _find_pol_waveform_seis(sta, sac_parent_dir, date, p_arrival)
                )
            # elif h5_parent_dir is not None:
            #     total_seconds = hour * 3600 + p_min * 60 + p_sec
            #     visual_time, visual_sac, train_time, train_sac = _find_pol_waveform_das(
            #         sta, date, p_arrival, total_seconds, h5_parent_dir
            #     )
            else:
                raise ValueError(
                    'Please check the equip_filter, and please provide the corresponded data Path.'
                )

            # final writing
            focal_dict['station_info'][sta]['p_arrival'] = p_arrival
            focal_dict['station_info'][sta]['phase_score'] = phase_score
            focal_dict['station_info'][sta]['visual_time'] = visual_time
            focal_dict['station_info'][sta]['visual_sac'] = visual_sac
            focal_dict['station_info'][sta]['train_time'] = train_time
            focal_dict['station_info'][sta]['train_sac'] = train_sac
            

def _preprocess_focal_files(
    gafocal_txt: Path, polarity_dout: Path, hout_file: Path | None = None
):
    """## Preprocessing the gafocal and polarity to accelarate multiprocessing"""
    df_gafocal, _ = check_format(catalog=gafocal_txt)

    # TODO: currently we add hout index and polarity index here, but this would change.
    if hout_file is not None:
        df_hout, _ = check_format(hout_file)
    else:
        df_hout = _hout_generate(polarity_dout=polarity_dout)

    df_hout['timestamp'] = df_hout['time'].apply(
        lambda x: datetime.fromisoformat(x).timestamp()
    )

    df_gafocal['timestamp'] = df_gafocal['time'].apply(
        lambda x: datetime.fromisoformat(x).timestamp()
    )

    return df_gafocal, df_hout


def find_gafocal_polarity(
    df_gafocal: pd.DataFrame,
    df_hout: pd.DataFrame,
    polarity_dout: Path,
    polarity_picks: Path,
    event_index: int,
    equip_filter,
    event_filter,
    get_waveform=False,
    h5_parent_dir=None,
    sac_parent_dir=None,
    comparing_mode=False,
    get_h3dd_index=False,
):
    """## Find the corresponded azimuth, take-off angle, and polarity from polarity_dout
    , as well as strike, dip, rake from gafocal.

    Args:
        - Event_index: event_index here represents the row index of gafocal_df.
        - comparing_mode: if True, using comp_index to find the event (this works for comp_plot).
    """
    if comparing_mode:
        event_info = df_gafocal[df_gafocal['comp_index'] == event_index]
        event_info = event_info.iloc[0]
    else:
        event_info = df_gafocal.loc[event_index]

    focal_dict = {
        'utc_time': event_info.time,
        # 'timestamp': event_info.timestamp,
        'longitude': event_info.longitude,
        'latitude': event_info.latitude,
        'depth': event_info.depth_km,
        'magnitude': event_info.magnitude
    }
    focal_dict['focal_plane'] = {
        'strike': event_info.strike,#int(event_info.strike.split('+')[0]),
        'dip': event_info.dip,#int(event_info.dip.split('+')[0]),
        'rake': event_info.rake,#int(event_info.rake.split('+')[0]),
    }
    focal_dict['quality_index'] = event_info.quality_index
    focal_dict['num_of_polarity'] = event_info.num_of_polarity

    # finding corresponded dout file, and assign the correct h3dd_index.
    # df_test = df_hout[np.abs(df_hout['timestamp'] - event_info.timestamp) < 1]
    conditions = [i.seconds < 1 for i in (pd.to_datetime(df_hout['time']) - pd.Timestamp(event_info.time))]
    df_test = df_hout[conditions]
    df_test = df_test[
        (
            np.abs(df_test['longitude'] - event_info.longitude) < 0.02
        )  # Use tolerance for floats
        & (
            np.abs(df_test['latitude'] - event_info.latitude) < 0.02
        )  # Use tolerance for floats
    ]

    h3dd_index = df_test.index[0]
    _find_station_focal_new(
        polarity_picks=polarity_picks,
        polarity_dout=polarity_dout,
        focal_dict=focal_dict,
        event_index=h3dd_index,
        get_waveform=get_waveform,
        equip_filter=equip_filter,
        event_filter=event_filter,
        h5_parent_dir=h5_parent_dir,
        sac_parent_dir=sac_parent_dir,
    )
    if get_h3dd_index:
        return focal_dict, h3dd_index
    else:
        return focal_dict


def plot_beachball_info(
    focal_dict_list: list[dict],
    name_list: list[str],
    fig=None,
    gs=None,
    ax=None,
    add_info=True,
    xlim=(-1.1, 1.1),
    ylim=(-1.1, 1.1),
    das_size=5.0,
    das_edge_size=2.0,
):
    if fig is None:
        fig = plt.figure(figsize=(36, 20))
        gs = GridSpec(10, 18, left=0, right=0.9, top=0.95, bottom=0.42, wspace=0.1)
    if len(focal_dict_list) == 2:
        ax1 = fig.add_subplot(gs[:5, :])  # focal axes-upper part
        ax2 = fig.add_subplot(gs[5:, :])
        ax_list = [ax1, ax2]
    else:
        if ax is None:
            ax_list = [fig.add_subplot(gs[4:7, :])]
        else:
            ax_list = [ax]
    for focal_dict, name, ax in zip(focal_dict_list, name_list, ax_list):
        # NOTE: temp solution
        if name == 'CWA':
            get_beach(
                focal_dict=focal_dict,
                source=name,
                ax=ax,
                xlim=xlim,
                ylim=ylim,
                add_info=add_info,
                is_cwa=True,
            )
        else:
            plot_detail = False
            if name == 'detail':
                plot_detail = True
                get_beach(
                    focal_dict=focal_dict,
                    source=name,
                    ax=ax,
                    xlim=xlim,
                    ylim=ylim,
                    add_info=add_info,
                    das_size=das_size,
                    das_edge_size=das_edge_size,
                    plot_detail=plot_detail,
                    station_text=0.01,
                    station_size=20,
                )
            else:
                get_beach(
                    focal_dict=focal_dict,
                    source=name,
                    ax=ax,
                    xlim=xlim,
                    ylim=ylim,
                    add_info=add_info,
                    das_size=das_size,
                    das_edge_size=das_edge_size,
                    plot_detail=plot_detail,
                )


def target_generator(focal_dict):
    """remove no polarity
    # NOTE: Using ChatGPT to optimize it later.
    """
    tmp_list = list(focal_dict['station_info'].keys())
    final_list = []
    for sta in tmp_list:
        pol = focal_dict['station_info'][sta]['polarity']
        if pol == ' ':
            continue
        final_list.append(sta)
    return final_list


def plot_polarity_waveform(
    main_focal_dict: dict,
    comp_focal_dict: dict,
    fig=None,
    gs=None,
    n_cols=3,
    train_window=0.64,
    visual_window=2,
    lw=0.7,
    arrival_lw=0.7,
):
    focal_station_list = target_generator(main_focal_dict)
    comp_focal_station_list = list(comp_focal_dict['station_info'].keys())
    wavenum = len(focal_station_list)
    n_rows = math.ceil(wavenum / n_cols)
    if fig is None:
        fig = plt.figure(figsize=(16, 24))
    gs = GridSpec(
        n_rows, n_cols, left=0, right=0.6, top=0.8, bottom=0, hspace=0.6, wspace=0.05
    )
    diff_counter = 0
    common_counter = 0
    for index, station in enumerate(focal_station_list):
        try:
            polarity = main_focal_dict['station_info'][station]['polarity']
            phase_score = main_focal_dict['station_info'][station]['phase_score']
            ii = index // n_cols
            jj = index % n_cols
            ax = fig.add_subplot(gs[ii, jj])
            x_wide = main_focal_dict['station_info'][station]['visual_time']
            y_wide = main_focal_dict['station_info'][station]['visual_sac']
            ax.plot(x_wide, y_wide, color='k', lw=lw, zorder=2)
            ax.set_xlim(0, visual_window * 2)
            ax.grid(True, alpha=0.7)
            x = main_focal_dict['station_info'][station]['train_time']
            y = main_focal_dict['station_info'][station]['train_sac']
            # TODO: customize the line length according to the y range.

            ax.plot(x, y, color='r', lw=lw, zorder=3)
            if station in comp_focal_station_list:
                comp_polarity = comp_focal_dict['station_info'][station]['polarity']
                if polarity != comp_polarity:
                    diff_counter += 1
                    ax.set_title(
                        f'{station}(AQ: {polarity}, CWA: {comp_polarity})',
                        fontsize=20,
                        color='r',
                    )
                else:
                    common_counter += 1
                    ax.set_title(
                        f'{station}(AQ: {polarity}, CWA: {comp_polarity})',
                        fontsize=20,
                        color='g',
                    )
            else:
                ax.set_title(
                    f'{station}({polarity}, phase score: {phase_score})',
                    fontsize=20,
                    color = 'r' if polarity == '+' else 'b',
                    weight='bold'
                    )
            ax.set_xticklabels([])  # Remove the tick labels but keep the ticks
            ax.set_yticklabels([])  # Remove the tick labels but keep the ticks
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            # ax.scatter(
            #     x[int(train_window * 100)],
            #     y[int(train_window * 100)],
            #     color='c',
            #     marker='o',
            #     s=point_size,
            # )
            y_offset = 0.2 * (abs(max(y_wide)) + abs(min(y_wide)))
            ax.vlines(
                x=x[int(train_window * 100)],
                ymin=y[int(train_window * 100)] - y_offset,
                ymax=y[int(train_window * 100)] + y_offset,
                color='c',
                linewidth=arrival_lw,
                zorder=4,
            )
        except Exception as e:
            logging.info(f'{station} Error: {e}')
    # all_counter = int(common_counter) + int(diff_counter)
    # fig.text(
    #     0.9,
    #     0.05,
    #     f'common polarity: {common_counter} / {all_counter}\ndiff polarity: {diff_counter} / {all_counter}',
    #     fontsize=35,
    # )


def process_tt_table(station_csv: Path, tt_table: Path, gafocal_index_list=[]):
    df_station = pd.read_csv(station_csv)
    df_station['elevation'] = df_station['elevation'].map(lambda x: x / 1000)
    for col in ['longitude', 'latitude', 'elevation']:
        df_station[col] = df_station[col].map(lambda x: round(x, 3))
    df_station.rename(
        columns={'longitude': 'stlo', 'latitude': 'stla', 'elevation': 'stel'},
        inplace=True,
    )

    header = pd.read_csv(tt_table, delimiter=',', nrows=1).columns
    data = pd.read_csv(tt_table, delimiter='\s+', skiprows=1, header=None)
    data.columns = header
    data['gafocal_index'] = data.groupby(['evlo', 'evla', 'evdp'], sort=False).ngroup()
    merged_df = pd.merge(data, df_station, on=['stlo', 'stla', 'stel'], how='left')
    df = merged_df[['station', 'az', 'tkof', 'gafocal_index']].copy()
    df.rename(columns={'az': 'azimuth', 'tkof': 'takeoff_angle'}, inplace=True)
    df = df[~df['station'].map(station_mask)]
    if len(gafocal_index_list) != 0:
        df = df[df['gafocal_index'].isin(gafocal_index_list)]
    return df


def _append_tracer_azi_takeoff(
    station_csv: Path,
    tt_table: Path,
    pol_picks: Path,
    focal_dict: dict,
    gamma_events: Path,
    h3dd_index,
    gafocal_index,
):
    """This is using to append DAS"""
    df_table = process_tt_table(
        station_csv, tt_table, gafocal_index_list=[gafocal_index]
    )

    gamma_index = find_gamma_index(gamma_events=gamma_events, h3dd_index=h3dd_index)
    df_pol_picks = pd.read_csv(pol_picks)
    df_pol_picks.rename(columns={'station_id': 'station'}, inplace=True)
    df_pol_picks = df_pol_picks[df_pol_picks['event_index'] == gamma_index]
    df = pd.merge(df_table, df_pol_picks, on='station', how='left')
    pol_map = {
        'D': '-',
        'U': '+',
        'x': ' ',
    }

    for _, row in df.iterrows():
        sta = row['station']
        try:
            focal_dict['station_info'][sta] = {
                'p_arrival': UTCDateTime(row['phase_time']),
                'azimuth': row['azimuth'],
                'takeoff_angle': row['takeoff_angle'],
                'polarity': pol_map[row['polarity']],
            }
            logging.info(
                f"appending {sta}: azimuth: {row['azimuth']}, takeoff_angle: {row['takeoff_angle']}, polarity: {pol_map[row['polarity']]}"
            )
        except Exception:
            logging.info(f'{sta} not in the polarity list, skip')


def process_directories(parent_dir: str | Path, i):
    """
    Processes subdirectories of a given parent directory.
    Skips processing if any file matching the pattern 'Event_*.png' is found.
    """
    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)

    # Iterate through subdirectories
    for sub_dir in parent_dir.iterdir():
        if sub_dir.is_dir():  # Check if it's a directory
            # Glob for matching files
            matching_files = list(sub_dir.glob(f'Event_{i}_*.png'))

            if matching_files:
                logging.info(f'Skipping event_{i} in {sub_dir.name}')
                return 0


def get_only_beach(
    df_gafocal, polarity_picks, df_hout, event_index, output_dir,
    equip_filter=lambda x: str(x)[0].isalpha(), event_filter=lambda x: str(x)[0].isdigit(), name='GAFocal'
):
    """## plot beachball only. waveform no needed"""
    logging.info(f'event_index: {event_index}')
    # code = process_directories(output_dir, event_index)
    # if code == 0:
    #     return
    focal_dict = find_gafocal_polarity(
        df_gafocal=df_gafocal,
        df_hout=df_hout,
        polarity_picks=polarity_picks,
        event_index=event_index,
        equip_filter=equip_filter,
        event_filter=event_filter
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_beachball_info(
        focal_dict_list=[focal_dict],
        name_list=[name],
        add_info=False,
        fig=fig,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir
        / f"Q{int(round(focal_dict['quality_index'], 0))}_M{int(round(focal_dict['magnitude'],0))}_{focal_dict['utc_time'].replace('-','_').replace(':', '_')}.png",
        bbox_inches='tight',
        dpi=300,
    )
    plt.close()
    logging.info(f'event_index: {event_index} done and saved')


def get_single_beach(
    df_gafocal: pd.DataFrame,
    polarity_dout: Path,
    polarity_picks: Path, # with h3dd index
    df_hout: pd.DataFrame,
    event_index: int,
    output_dir: Path,
    get_waveform=False,
    sac_parent_dir: Path | None = None,
    h5_parent_dir: Path | None = None,
    equip_filter=lambda x: str(x)[0].isalpha(),
    event_filter=lambda x: str(x)[0].isdigit(),
    gamma_events=None,
    gamma_detailed=None,
    das_size=5.0,
    das_edge_size=2.0,
    add_info=False
):
    """## plot beachball only."""
    logging.info(f'event_index: {event_index}')
    code = process_directories(output_dir, event_index)
    if code == 0:
        return
    focal_dict, h3dd_index = find_gafocal_polarity(
        df_gafocal=df_gafocal,
        df_hout=df_hout,
        polarity_dout=polarity_dout,
        polarity_picks=polarity_picks,
        event_index=event_index,
        get_h3dd_index=True,
        get_waveform=get_waveform,
        sac_parent_dir=sac_parent_dir,
        h5_parent_dir=h5_parent_dir,
        equip_filter=equip_filter,
        event_filter=event_filter
    )
    # if gamma_events is not None and gamma_detailed is not None:
    #     gamma_index = find_gamma_index(gamma_events=gamma_events, h3dd_index=h3dd_index)
    #     df_type_table = pd.read_csv(gamma_detailed)
    #     df_type_table = df_type_table[['event_index', 'event_type']]
    #     event_type = df_type_table[df_type_table['event_index'] == gamma_index][
    #         'event_type'
    #     ].iloc[0]
        # dir_map = {
        #     1: 'Event_type_1_seis_6P2S_DAS_15P',
        #     2: 'Event_type_2_seis_6P2S',
        #     3: 'Event_type_3_DAS_15P',
        #     4: 'Event_type_4_Not_reach_the_standard',
        # }
        # output_dir = output_dir / dir_map[event_type]
        # output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(32, 32))
    gs1 = GridSpec(10, 18, left=0.65, right=0.9, top=0.95, bottom=0.05, wspace=0.1)
    plot_beachball_info(
        focal_dict_list=[focal_dict],
        name_list=['AutoQuake'],
        fig=fig,
        gs=gs1,
        das_size=das_size,
        das_edge_size=das_edge_size,
        add_info=add_info
    )
    comp_focal_dict = defaultdict(dict)
    if get_waveform:
        plot_polarity_waveform(
            main_focal_dict=focal_dict, comp_focal_dict=comp_focal_dict, fig=fig
        )
        text_name = f"{focal_dict['utc_time'].replace('-','_').replace(':', '_')}\nLon: {focal_dict['longitude']}, Lat: {focal_dict['latitude']}, Depth: {focal_dict['depth']} km"
        fig.text(0.3, 0.85, text_name, ha='center', va='center', fontsize=50)
    plt.tight_layout()
    plt.savefig(
        output_dir
        / f"Event_{event_index}_{focal_dict['utc_time'].replace('-','_').replace(':', '_')}.png",
        bbox_inches='tight',
        dpi=300,
    )
    plt.close()
    logging.info(f'event_index: {event_index} done and saved')


def get_tracer_beach(
    df_gafocal,
    polarity_dout,
    df_hout,
    event_index,
    output_dir,
    equip_filter,
    event_filter,
    gamma_events=None,
    gamma_detailed=None,
    get_waveform=False,
    sac_parent_dir: Path | None = None,
    h5_parent_dir: Path | None = None,
    input_tracer=False,
    tt_table: Path | None = None,
    extend_pol_picks: Path | None = None,
    station: Path | None = None,
    das_size=5.0,
    das_edge_size=2.0,
    xlim=(-1.1, 1.1),
    ylim=(-1.1, 1.1),
):
    """## plot beachball only."""
    logging.info(f'event_index: {event_index}')
    # code = process_directories(output_dir, event_index)
    # if code == 0:
    #     return
    focal_dict, h3dd_index = find_gafocal_polarity(
        df_gafocal=df_gafocal,
        df_hout=df_hout,
        polarity_picks=polarity_dout,
        event_index=event_index,
        get_h3dd_index=True,
        get_waveform=get_waveform,
        sac_parent_dir=sac_parent_dir,
        h5_parent_dir=h5_parent_dir,
        equip_filter=equip_filter,
        event_filter=event_filter
    )
    logging.info(f'input_tracer: {input_tracer}')
    logging.info(f'tt_table: {tt_table}')
    logging.info(f'extend_pol_picks: {extend_pol_picks}')
    if input_tracer and tt_table is not None and extend_pol_picks is not None:
        logging.info('start append tracer format')
        _append_tracer_azi_takeoff(
            station_csv=station,
            tt_table=tt_table,
            pol_picks=extend_pol_picks,
            focal_dict=focal_dict,
            gamma_events=gamma_events,
            h3dd_index=h3dd_index,
            gafocal_index=event_index,
        )
    if gamma_events is not None and gamma_detailed is not None:
        gamma_index = find_gamma_index(gamma_events=gamma_events, h3dd_index=h3dd_index)
        df_type_table = pd.read_csv(gamma_detailed)
        df_type_table = df_type_table[['event_index', 'event_type']]
        event_type = df_type_table[df_type_table['event_index'] == gamma_index][
            'event_type'
        ].iloc[0]
        dir_map = {
            1: 'Event_type_1_seis_6P2S_DAS_15P',
            2: 'Event_type_2_seis_6P2S',
            3: 'Event_type_3_DAS_15P',
            4: 'Event_type_4_Not_reach_the_standard',
        }
        output_dir = output_dir / dir_map[event_type]
        output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(32, 16))
    gs = GridSpec(10, 18, left=0, right=0.9, top=0.9, bottom=0, wspace=0.1)
    ax1 = fig.add_subplot(gs[:, :9])
    ax2 = fig.add_subplot(gs[2:8, 12:17])
    plot_beachball_info(
        focal_dict_list=[focal_dict],
        name_list=['AutoQuake'],
        fig=fig,
        ax=ax1,
        das_size=7.0,
        das_edge_size=das_edge_size,
    )
    plot_beachball_info(
        focal_dict_list=[focal_dict],
        name_list=['detail'],
        fig=fig,
        ax=ax2,
        add_info=False,
        das_size=10.0,
        das_edge_size=das_edge_size,
        xlim=xlim,
        ylim=ylim,
    )
    comp_focal_dict = defaultdict(dict)
    if get_waveform:
        plot_polarity_waveform(
            main_focal_dict=focal_dict, comp_focal_dict=comp_focal_dict, fig=fig
        )
        text_name = f"{focal_dict['utc_time'].replace('-','_').replace(':', '_')}\nLon: {focal_dict['longitude']}, Lat: {focal_dict['latitude']}, Depth: {focal_dict['depth']} km"
        fig.text(0.3, 0.9, text_name, ha='center', va='center', fontsize=25)
    plt.tight_layout()
    plt.savefig(
        output_dir
        / f"Event_{event_index}_{focal_dict['utc_time'].replace('-','_').replace(':', '_')}.png",
        bbox_inches='tight',
        dpi=300,
    )

    logging.info(f'event_index: {event_index} done and saved')


# ===== Plot private function =====
def _plot_seis(
    sac_dict: dict,
    df_phasenet_picks: pd.DataFrame,
    df_gamma_picks: pd.DataFrame,
    ax,
    bar_length=2,
):
    """
    plot the seismometer waveform for check.
    """
    for sta in list(sac_dict.keys()):
        if sta not in list(df_gamma_picks['station_id']):
            ax.plot(
                sac_dict[sta]['time'],
                sac_dict[sta]['sac_data'],
                color='k',
                linewidth=0.4,
                alpha=0.25,
                zorder=1,
            )
            ax.text(
                61,  # sac_dict[sta]['time'][-1] + 1,
                sac_dict[sta]['distance'],
                sta,
                fontsize=4,
                verticalalignment='center',
                alpha=0.25,
            )
        else:
            ax.plot(
                sac_dict[sta]['time'],
                sac_dict[sta]['sac_data'],
                color='k',
                linewidth=0.4,
                zorder=1,
            )
            ax.text(
                61,  # sac_dict[sta]['time'][-1] + 1,
                sac_dict[sta]['distance'],
                sta,
                fontsize=4,
                verticalalignment='center',
            )
    # all picks
    ax.scatter(
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P']['x'],
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P']['y'],
        color='r',
        s=1,
        zorder=2,
    )
    ax.scatter(
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S']['x'],
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S']['y'],
        color='blue',
        s=1,
        zorder=2,
    )
    # aso picks
    ax.plot(
        [
            df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['x'],
            df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['x'],
        ],
        [
            df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['y'].to_numpy()
            - bar_length,
            df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['y'].to_numpy()
            + bar_length,
        ],
        color='r',
        linewidth=0.7,
        zorder=2,
    )
    ax.plot(
        [
            df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['x'],
            df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['x'],
        ],
        [
            df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['y'].to_numpy()
            - bar_length,
            df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['y'].to_numpy()
            + bar_length,
        ],
        color='blue',
        linewidth=0.7,
        zorder=2,
    )
    # ax.set_xlim(0, 90)
    ax.set_xlim(30, 60)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='both', which='major', length=3, width=1, labelsize=5)
    # ax2.tick_params(axis='x', which='minor', length=4, width=0.5)
    ax.set_xlabel('Time (s)', fontsize=7)


# ===== add function =====
def _add_das_picks(
    starttime: str,  # event_total_seconds - 30
    endtime: str,  # event_total_seconds + 60
    main_ax: Axes,
    h5_parent_dir: Path,  # h5_parent_dir/ymd/300_600.h5
    prob_ax: Axes | None = None,
    df_das_gamma_picks: pd.DataFrame | None = None,
    df_phasenet_picks: pd.DataFrame | None = None,
    interval=300,
):
    """
    We only consider the scale across the day.
    """
    print('_add_das_picks...')
    start_ymd, start_idx, start_sec = _process_midas_scenario(
        starttime, interval=interval
    )
    end_ymd, end_idx, end_sec = _process_midas_scenario(endtime, interval=interval)
    window = [f'{start_ymd}:{interval*start_idx}_{interval*(start_idx+1)}.h5']
    same_index = True
    if start_idx != end_idx:
        # same day
        next_window = f'{end_ymd}:{interval*(end_idx)}_{interval*(end_idx+1)}.h5'
        window.append(next_window)
        same_index = False
    try:
        print('loading h5...')
        all_data = []
        for win in window:
            ymd = win.split(':')[0]
            index_file_name = win.split(':')[1]
            file = list((h5_parent_dir / f'{ymd}_hdf5').glob(f'*{index_file_name}'))[0]
            if not file:
                raise IndexError(f'File not found for window {win}')
            try:
                with h5py.File(file, 'r') as fp:
                    ds = fp['data']
                    data = ds[...]  # np.array
                    dt = ds.attrs['dt_s']  # 0.01 sampling rate
                    dx = ds.attrs['dx_m']  # interval of cable ~ 4
                    nx, nt = data.shape
                    logging.info(data.shape)
                    x = np.arange(nx) * dx
                    t = np.arange(nt) * dt
                    all_data.append(data)
            except Exception as e:
                logging.info(f'Error reading {file}: {e}')
    except IndexError:
        logging.info(f'File not found for window {window}')

    # Handle the case where there is only one array in all_data
    if len(all_data) == 1:
        concatenated_data = all_data[0]
        # logging.info('Only one data array, no need to concatenate.')
    elif len(all_data) > 1:
        # Concatenate all data arrays along the second axis (horizontally)
        concatenated_data = np.concatenate(all_data, axis=1)
        # logging.info(f'Concatenated data shape: {concatenated_data.shape}')
    nx, nt = concatenated_data.shape
    x = np.arange(nx) * dx
    t = np.arange(nt) * dt

    main_ax.imshow(
        normalize(concatenated_data).T,
        cmap='seismic',
        vmin=-1,
        vmax=1,
        aspect='auto',
        extent=[x[0], x[-1], t[-1], t[0]],
        interpolation='none',
    )

    if not same_index:
        main_ax.set_ylim(
            interval + end_sec, start_sec
        )  # concat later or not would not influence the time.
        ticks=np.linspace(interval + end_sec, start_sec, 10)
        main_ax.set_yticks(ticks=ticks, labels=ticks - start_sec)
    else:
        main_ax.set_ylim(end_sec, start_sec)
        ticks=np.linspace(end_sec, start_sec, 10)
        main_ax.set_yticks(ticks=ticks, labels=ticks - start_sec)
    print('scatter plot...')
    if df_phasenet_picks is not None:
        df_phasenet_picks = _find_phasenet_das_pick(
            starttime=starttime, endtime=endtime,
            df_picks=df_phasenet_picks, das_time_interval=interval,
            dt=dt, dx=dx, station_mask=station_mask
        )
        main_ax.scatter(
            df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P']['x'],
            df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P']['y'],
            c='r',
            s=1,
            alpha=0.05,
            zorder=2,
        )
        main_ax.scatter(
            df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S']['x'],
            df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S']['y'],
            c='c',
            s=1,
            alpha=0.05,
            zorder=2,
        )
        if prob_ax is not None:
            prob_ax.scatter(
                df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P'][
                    'phase_score'
                ],
                df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P']['y'],
                c='r',
                s=1,
            )
            prob_ax.scatter(
                df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S'][
                    'phase_score'
                ],
                df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S']['y'],
                c='c',
                s=0.6,
            )
            prob_ax.yaxis.set_ticks([])
            prob_ax.set_yticklabels([])
            prob_ax.set_xlabel('Prob')
    if df_das_gamma_picks is not None:
        main_ax.scatter(
            df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'P']['x'],
            df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'P']['y'],
            c='r',
            edgecolors='black',
            s=1,
            zorder=4,
        )
        main_ax.scatter(
            df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'S']['x'],
            df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'S']['y'],
            c='c',
            edgecolors='black',
            s=1,
            zorder=4,
        )

    main_ax.scatter([], [], c='r', label='P')
    main_ax.scatter([], [], c='c', label='S')
    main_ax.legend()
    main_ax.set_xlabel('Distance (m)')
    main_ax.set_ylabel('Time (s)')


# ===== get function =====
def get_mapview(
        region: list,
        ax: GeoAxes,
        title: str | None = None,
        use_fault=True,
        cmap='gray',
        fault_lw=2,
        coastline_lw=1,
        hillshade=True,
        resolution='15s',
        adjust_grid=False,
        inside=False,
        vert_exag=10,
        **kwargs
        ):
    # from matplotlib.colors import ListedColormap
    """## Plotting the basic map.

    Notice here, the ax should be a GeoAxes! A subclass of `matplotlib.axes.Axes`.
    """
    tw_coast = Path(__file__).parent / 'geo_map' / 'tw_coast.txt'
    topo = (
        pygmt.datasets.load_earth_relief(resolution=resolution, region=region).to_numpy()
        / 1e3
    )  # km
    x = np.linspace(region[0], region[1], topo.shape[1])
    y = np.linspace(region[2], region[3], topo.shape[0])
    dx, dy = 1, 1
    xgrid, ygrid = np.meshgrid(x, y)
    ls = LightSource(azdeg=0, altdeg=45)

    # cmap operation
    # original_cmap = plt.get_cmap(cmap)

    # # Sample only the upper half (e.g. top 50% of color range)
    # colors = original_cmap(np.linspace(0, 0.5, 256))  # change 0.5 to other values if needed
    # new_cmap = ListedColormap(colors)
    ax.pcolormesh(
        xgrid,
        ygrid,
        ls.hillshade(topo, vert_exag=vert_exag, dx=dx, dy=dy) if hillshade else topo,
        vmin=-1 if hillshade else None,
        shading='gouraud',
        cmap=cmap,
        alpha=1.0,
        antialiased=True,
        rasterized=True,
    )
    # cartopy setting
    # ax.coastlines()
    add_tw_coast(ax, tw_coast, coastline_lw, color='k')
    if use_fault:
        add_fault(ax, lw=fault_lw)
        add_fault_csv(ax, lw=fault_lw)
    ax.set_extent(region)
    if title:
        ax.set_title(title, **kwargs)
    ax.set_ylabel('Latitude', fontsize=12, weight='bold')
    if not adjust_grid:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False  # Turn off top labels
        if inside:
            gl.left_labels = False
            gl.bottom_labels = False
            gl.ylabel_style = {
                'rotation': 270,
            }
        else:
            gl.right_labels = False  # Turn off top labels
            gl.xlabel_style = {
                'weight': 'bold',
                'size': 20
            }
            gl.ylabel_style = {
                'rotation': 90,
                'weight': 'bold',
                'size': 20
            }    

def get_phasenet_das(
    picks: Path,
    hdf5_parent_dir: Path,
    event_utc_time: str | None = None,
    starttime: str | None = None,
    endtime: str | None = None,
    main_ax: Axes | None = None,
    prob_ax: Axes | None = None,
    plot_prob=True,
    get_station=lambda x: x,
    pretime=-30,
    posttime=60,
    interval=300
):
    """## Get PhaseNet DAS result.
    If you have no idea how to determine the start and end time, you can give an
    event_utc_time to let add_on_utc_time function to determine the start and end.

    ### About axes
    scenario1: using the default axes
    scenario2: providing your own axes
        - 2a. only plot main
        - 2b. plot main and prob axes
    """
    print('preprocessing...')
    df_phasenet = _preprocess_phasenet_csv(
        phasenet_picks=picks, get_station=get_station, starttime=starttime, endtime=endtime
    )
    print('preprocessing done!')
    if event_utc_time is not None:
        starttime = add_on_utc_time(event_utc_time, pretime)
        endtime = add_on_utc_time(event_utc_time, posttime)

    if starttime is not None and endtime is not None:
        # axes judge
        if plot_prob:
            if main_ax is None and prob_ax is None:
                fig = plt.figure()
                gs = GridSpec(1, 3, figure=fig)
                main_ax = fig.add_subplot(gs[0, :2])
                prob_ax = fig.add_subplot(gs[0, 2])
            else:
                raise ValueError(
                    'If you want to plot prob, please provide both Axes or both None'
                )
        else:
            if main_ax is None:
                fig, main_ax = plt.subplots()
                prob_ax = None

        _add_das_picks(
            starttime=starttime,
            endtime=endtime,
            main_ax=main_ax,
            prob_ax=prob_ax,
            h5_parent_dir=hdf5_parent_dir,
            df_phasenet_picks=df_phasenet,
            interval=interval
        )
    else:
        raise ValueError(
            'Please specify the starttime and endtime!, or input the event_utc_time'
        )


# ===== run function =====
# ===== plot function =====
def plot_waveform_check(
    sac_dict: dict,
    starttime: str,
    end_time: str,
    df_seis_phasenet_picks: pd.DataFrame | None = None,
    df_seis_gamma_picks: pd.DataFrame | None = None,
    df_das_phasenet_picks: pd.DataFrame | None = None,
    df_das_gamma_picks: pd.DataFrame | None = None,
    h5_parent_dir=None,
    das_ax=None,
    seis_ax=None,
    interval=300,
):
    if das_ax is None and seis_ax is None:
        fig = plt.figure(figsize=(8, 12))
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        seis_ax = fig.add_subplot(gs[0])
        das_ax = fig.add_subplot(gs[1])
    # TODO: Design a suitable gs
    if df_seis_gamma_picks is not None and df_seis_phasenet_picks is not None:
        _plot_seis(
            sac_dict=sac_dict,
            df_phasenet_picks=df_seis_phasenet_picks,
            df_gamma_picks=df_seis_gamma_picks,
            ax=seis_ax,
        )
    if (
        df_das_gamma_picks is not None
        and df_das_phasenet_picks is not None
        and h5_parent_dir is not None
    ):
        _add_das_picks(
            starttime=starttime,
            endtime=end_time,
            interval=interval,
            main_ax=das_ax,
            h5_parent_dir=h5_parent_dir,
            df_das_gamma_picks=df_das_gamma_picks,
            df_phasenet_picks=df_das_phasenet_picks,
        )


def plot_station(
    df_station: pd.DataFrame,
    geo_ax: Axes,
    region,
    color=None,
    plot_station_name=False,
    fontsize=10,
    text_dist=0.01,
    zorder=4,
    station_size=50,
    station_mask=station_mask,
):
    """
    plot the station distribution on map.
    """
    mask_alpha = df_station['station'].map(station_mask)
    mask_digit = ~mask_alpha
    if mask_digit.any():
        geo_ax.scatter(
            df_station[mask_digit]['longitude'],
            df_station[mask_digit]['latitude'],
            s=5,
            c='k' if color is None else color,
            marker='.',
            alpha=0.5,
            rasterized=True,
            label='DAS',
            zorder=zorder,
        )
    if mask_alpha.any():
        geo_ax.scatter(
            df_station[mask_alpha]['longitude'],
            df_station[mask_alpha]['latitude'],
            s=station_size,
            c='c' if color is None else color,
            marker='^',
            edgecolors='k',
            alpha=0.7,
            rasterized=True,
            label='Seismometer',
            zorder=zorder,
        )
        if plot_station_name:
            for x, y, s in zip(
                df_station[mask_alpha]['longitude'],
                df_station[mask_alpha]['latitude'],
                df_station[mask_alpha]['station'],
            ):
                # NOTE: quite weird, it should clip.
                if not (region[0] <= x <= region[1]) or not (
                    region[2] <= y <= region[3]
                ):
                    logging.info(f'Removing text: {s}')
                else:
                    geo_ax.text(
                        x=x + text_dist,
                        y=y + text_dist,
                        s=s,
                        fontsize=fontsize,
                        clip_on=True,
                        zorder=zorder,
                        # weight='bold'
                    )


def get_beach(
    focal_dict,
    ax,
    xlim=(-1.1, 1.1),
    ylim=(-1.1, 1.1),
    color='silver',
    only_das=False,
    add_info=False,
    source='AutoQuake',
    is_cwa=False,
    das_size=5.0,
    das_edge_size=2.0,
    plot_detail=False,
    station_text=0.025,
    station_size=10,
):
    """Plot the beachball diagram with DAS."""
    mt = pmt.MomentTensor(
        strike=focal_dict['focal_plane']['strike'],
        dip=focal_dict['focal_plane']['dip'],
        rake=focal_dict['focal_plane']['rake'],
    )

    projection = 'lambert'

    beachball.plot_beachball_mpl(
        mt,
        ax,
        position=(0.0, 0.0),
        size=2.0,
        color_t=mpl_color(color),
        projection=projection,
        size_units='data',
    )
    das_x = []
    das_y = []
    for sta in focal_dict['station_info'].keys():
        takeoff = focal_dict['station_info'][sta]['takeoff_angle']
        azi = focal_dict['station_info'][sta]['azimuth']
        polarity = focal_dict['station_info'][sta]['polarity']
        # if polarity == ' ':
        #     continue
        # to spherical coordinates, r, theta, phi in radians
        # flip direction when takeoff is upward
        rtp = np.array(
            [
                [
                    1.0 if takeoff <= 90.0 else -1.0,
                    np.deg2rad(takeoff),
                    np.deg2rad(90.0 - azi),
                ]
            ]
        )
        # to 3D coordinates (x, y, z)
        points = beachball.numpy_rtp2xyz(rtp)

        # project to 2D with same projection as used in beachball
        x, y = beachball.project(points, projection=projection).T
        # TODO: adding the DAS condition
        #TODO: PLEASE REMOVE ALL THIS CUMBERSOME SHITS.
        if sta[0].isalpha() or is_cwa:
            if polarity == '+':
                ax.plot(
                    x,
                    y,
                    '+',
                    ms=10.0,
                    mew=2.0,
                    mec='r',
                    mfc='none',
                )
            elif polarity == '-':
                ax.plot(
                    x,
                    y,
                    'o',
                    ms=10.0 / np.sqrt(2.0),
                    mew=2.0,
                    mec='b',
                    mfc='none',
                )
            else:
                continue
            ax.text(
                x + station_text,
                y + station_text,
                sta,
                clip_on=True,
                fontsize=station_size,
            )
        else:
            if polarity == '+':
                ax.plot(
                    x,
                    y,
                    '.',
                    ms=das_size,
                    mew=das_edge_size,
                    mec='none',
                    mfc='r',
                )
            elif polarity == '-':
                ax.plot(
                    x,
                    y,
                    '.',
                    ms=das_size,
                    mew=das_edge_size,
                    mec='none',
                    mfc='b',
                )
            else:
                ax.plot(
                    x,
                    y,
                    '.',
                    ms=das_size,
                    mew=das_edge_size,
                    mec='none',
                    mfc='dimgray',
                )
            if plot_detail:
                das_x.append(x)
                das_y.append(y)
    if add_info:
        ax.text(
            1.2,
            -0.5,
            f"{source}\nQuality index: {focal_dict['quality_index']}\nnum of station: {focal_dict['num_of_polarity']}\nStrike: {focal_dict['focal_plane']['strike']}\nDip: {focal_dict['focal_plane']['dip']}\nRake: {focal_dict['focal_plane']['rake']}",
            fontsize=20,
        )
    if plot_detail:
        xlim = (np.mean(das_x) - 0.1, np.mean(das_x) + 0.1)
        ylim = (np.mean(das_y) - 0.1, np.mean(das_y) + 0.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_axis_off()


def quick_indexing(
    gamma_picks, gafocal_txt, polarity_dout, gamma_events, gafocal_index_list, equip_filter, event_filter
):
    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=gafocal_txt, polarity_dout=polarity_dout
    )
    df_picks = pd.read_csv(gamma_picks)
    gamma_list = []
    for i in gafocal_index_list:
        _, h3dd_index = find_gafocal_polarity(
            df_gafocal=df_gafocal,
            df_hout=df_hout,
            polarity_dout=polarity_dout,
            event_index=i,
            get_h3dd_index=True,
            equip_filter=equip_filter,
            event_filter=event_filter
        )
        gamma_index = find_gamma_index(gamma_events=gamma_events, h3dd_index=h3dd_index)
        gamma_list.append(gamma_index)
    df_picks = df_picks[
        (df_picks['event_index'].isin(gamma_list)) & (df_picks['phase_type'] == 'P')
    ]
    return df_picks


def add_fault(
    ax: Axes, color='r', fault_txt: Path | None = None, lw=2, transform=ccrs.PlateCarree(), **kwags
):
    """
    Adding the fault line on the map.
    """
    if fault_txt is None:
        fault_txt = Path(__file__).parent / 'geo_map' / 'hualien_structure.txt'
    with open(fault_txt) as f:
        lines = f.readlines()
    fault_dir = {}
    for line in lines:
        if line[:1] == '#':
            fault_name = line.split('|')[1]
            fault_dir[fault_name] = {'longitude': [], 'latitude': []}
        else:
            lon, lat = map(float, line.split())
            fault_dir[fault_name]['longitude'].append(lon)
            fault_dir[fault_name]['latitude'].append(lat)

    for fault_name in fault_dir.keys():
        ax.plot(
            fault_dir[fault_name]['longitude'],
            fault_dir[fault_name]['latitude'],
            color=color,
            linewidth=lw,
            transform=transform,
            label=fault_name,
            zorder=5,
            **kwags
        )
        # ax.text(
        #     x = fault_dir[fault_name]['longitude'][0],
        #     y = fault_dir[fault_name]['latitude'][1],
        #     s = fault_name,
        #     fontsize=15
        # )

def add_fault_csv(
    ax: Axes, fault_csv: Path | pd.DataFrame | None = None, lw=2, transform=ccrs.PlateCarree(), **kwargs
):
    """
    Adding the fault line on the map.
    """
    if isinstance(fault_csv, Path):
        df = pd.read_csv(fault_csv)
    elif isinstance(fault_csv, pd.DataFrame):
        df = fault_csv
    else:
        fault_csv = Path(__file__).parent / 'geo_map' / 'my_georeference' / 'geomap.csv'
        df = pd.read_csv(fault_csv)
        df['color'] = 'r'

    for fault_name in df['name'].unique():
        color = df[df['name'] == fault_name]['color'].unique()[0]
        linestyle = df[df['name'] == fault_name]['line_style'].unique()[0]
        ax.plot(
            df[df['name'] == fault_name]['longitude'],
            df[df['name'] == fault_name]['latitude'],
            color=color,
            linewidth=lw,
            linestyle='--' if linestyle == 0 else None,
            transform=transform,
            label=fault_name,
            zorder=5,
            **kwargs
        )

def add_beachballs(gafocal_txt: Path, ax, on='mapview'):
    """
    1. input the gafocal_txt file
    2. plot the beachball on the ax
    3. on: mapview (lon, lat), longitude (lon, depth), latitude (depth, lat)
    """
    df_gafocal, _ = check_format(gafocal_txt)
    for i in ['strike', 'dip', 'rake']:
        df_gafocal[i] = df_gafocal[i].map(lambda x: int(x.split('+')[0]))
    for _, row in df_gafocal.iterrows():
        lon, lat, depth = row['longitude'], row['latitude'], row['depth_km']
        position_map = {
            'mapview': (lon, lat),
            'longitude': (lon, depth),
            'latitude': (depth, lat),
        }
        mt = pmt.MomentTensor(
            strike=row['strike'],
            dip=row['dip'],
            rake=row['rake'],
        )
        beachball.plot_beachball_mpl(
            mt,
            ax,
            size=20,
            beachball_type='full',
            position=position_map[on],
            linewidth=1,
            color_t=mpl_color('silver'),
            zorder=6,
        )


def add_single_beachball(ax, strike: int, dip: int, rake: int, x: float, y: float, size=20, zorder=6):
    """
    Adding single beachball on the map
    """
    mt = pmt.MomentTensor(
        strike=strike,
        dip=dip,
        rake=rake,
    )
    beachball.plot_beachball_mpl(
        mt,
        ax,
        size=size,
        beachball_type='full',
        position=(x, y),
        linewidth=1,
        color_t=mpl_color('silver'),
        zorder=zorder,
    )


def add_tw_coast(ax: Axes, tw_coast: Path, lw=1, **kwargs):
    with open(tw_coast) as f:
        lines = f.readlines()
    for line in lines:
        if line[:1] == '>':
            try:
                ax.plot(
                    lon_list,
                    lat_list,
                    transform=ccrs.PlateCarree(),
                    linewidth=lw,
                    zorder=5,
                    **kwargs
                )
                lon_list, lat_list = [], []
            except Exception as e:
                print(f'we want to pass more >, {e}')
        else:
            lon, lat = map(float, line.strip().split())
            try:
                lon_list.append(lon)
                lat_list.append(lat)
            except Exception:
                lon_list, lat_list = [], []
                lon_list.append(lon)
                lat_list.append(lat)

def add_vel_view(
        ax: Axes,
        vel_df: pd.DataFrame,
        region: list[float],
        depth=0,
        mode='vp',
        grid=False,
        alpha=0.8,
        cmap='jet_r',
        vmin=1.5,
        vmax=8.5
        ):
    """
    Plot the velocity model at a specific depth on a geographic map using Cartopy.

    Args:
        ax (Axes): Matplotlib Axes object with a Cartopy projection.
        csv_file (str): Path to the CSV file containing the velocity model.
        region (tuple): Region bounds (min_lon, max_lon, min_lat, max_lat).
        depth (float): Depth value to plot.
        mode (str): 'vp', 'vs', 'VpVs'
    """
    # Load data from CSV
    df = vel_df
    # Filter data based on depth and region
    filtered_df = df[
        (df['lon'] >= region[0]) & (df['lon'] <= region[1]) &
        (df['lat'] >= region[2]) & (df['lat'] <= region[3]) &
        np.isclose(df['depth'], depth, atol=0.5)  # Depth tolerance
    ]

    if filtered_df.empty:
        raise ValueError("No data found for the specified depth and region.")

    # Create grid for plotting
    lon = filtered_df['lon'].unique()
    lat = filtered_df['lat'].unique()
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    velocity = filtered_df.pivot(index='lat', columns='lon', values=mode).values

    if grid:
        im = ax.pcolormesh(
            lon_grid, lat_grid, velocity,
            transform=ccrs.PlateCarree(), cmap=cmap, shading='auto'
        )
    else:
        levels = np.linspace(vmin, vmax, 21)
        im = ax.contourf(
            lon_grid, lat_grid, velocity,
            transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, alpha=alpha
        )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.6)
    cbar.set_label(f"{mode} (km/s)")
    ax.set_extent(region, crs=ccrs.PlateCarree())

def _generate_profile_points(start_point, azimuth, distance, dist_interval):
    """
    Generate profile points along a geodesic based on a start point, azimuth, and distance.

    Args:
        start_point (tuple): Starting point as (longitude, latitude) in degrees.
        azimuth (float): Azimuth (direction) in degrees (clockwise from north).
        distance (float): Total distance of the profile in kilometers.
        n_points (int): Number of points to generate along the profile.

    Returns:
        profile_points (list): List of (longitude, latitude) points along the profile.
    """
    geod = Geod(ellps="WGS84")  # Define Earth model

    # Divide total distance into evenly spaced segments
    distances = np.arange(0, distance*1000, dist_interval*1000)  # Convert km to meters
    if distances[-1] != distance*1000:
        distances = np.append(distances, distance*1000)
    logging.info(f'Distances: {distances}')
    # Compute profile points
    lon, lat = start_point
    profile_points = [geod.fwd(lon, lat, azimuth, dist)[:2] for dist in distances]
    return profile_points

def add_vel_profile(
        ax: Axes,
        vel_df: pd.DataFrame,
        start_point: tuple,
        distance: Any,
        azimuth: Any,
        dist_interval=12,
        depth_range=(0, 60),
        mode='vpt',        
        cmap='RdYlBu',
        interpolate_mothod='linear',
        cax: Axes | None = None
):
    """
    Extract and plot a velocity profile along a line between two geographic points.

    Args:
        ax (Axes): Matplotlib Axes object for plotting.
    """
    # Load data from CSV
    df = vel_df

    # Generate profile points
    profile_points = _generate_profile_points(start_point, azimuth, distance, dist_interval)

    # Regular depth grid
    depth_array = df['dep'].unique()
    depth_subset = depth_array[(depth_array >= depth_range[0]) & (depth_array <= depth_range[1])]

    # Create velocity grid
    # if interpolate_mode == 'mean': # NEAREST method handcraft
    #     velocity_grid = np.full((len(depth_subset), len(profile_points)), np.nan)

    #     for i, (lon, lat) in enumerate(profile_points):
    #         point_data = df[
    #             (df['lon'].between(lon - 0.04, lon + 0.04)) &
    #             (df['lat'].between(lat - 0.04, lat + 0.04))
    #         ]
    #         if not point_data.empty:
    #             for j, depth in enumerate(depth_subset):
    #                 depth_data = point_data[point_data['dep'] == depth]
    #                 if not depth_data.empty:
    #                     velocity_grid[j, i] = depth_data[mode].mean()

    points = df[['lon', 'lat', 'dep']].values  # Known (lon, lat, depth) points
    values = df[mode].values             # Known velocity values
    lon_lat_depth_grid = [
        (lon, lat, depth) for depth in depth_subset for lon, lat in profile_points
    ]
    grid_points = np.array(lon_lat_depth_grid)
    interpolated_values = griddata(points, values, grid_points, method=interpolate_mothod)
    velocity_grid = interpolated_values.reshape(len(depth_subset), len(profile_points))
    # Plot velocity profile
    distances = np.arange(0, distance, dist_interval)  # Convert km to meters
    if distances[-1] != distance:
        distances = np.append(distances, distance)

    min_max_map = {
        'vpt': (-30, 30, cmap),
        'vst': (-25, 25, cmap),
        'vpvst': (-25, 25, f'{cmap}_r')
    }

    levels = np.linspace(min_max_map[mode][0], min_max_map[mode][1], 21)
    contourf = ax.contourf(
        distances, depth_subset, velocity_grid, levels=levels,
        cmap=min_max_map[mode][2], vmin=min_max_map[mode][0], vmax=min_max_map[mode][1]
    )
    cax_label_map = {
        'vpt': 'Vp Perturbation',
        'vst': 'Vs Perturbation',
        'vpvst': 'Vp/Vs Perturbation'
    }
    if cax is not None:
        cbar = plt.colorbar(contourf, cax=cax)
        cbar.set_label(cax_label_map[mode], fontsize=15, weight='bold')
    # else:
    #     cbar = plt.colorbar(contourf, ax=ax)
    

def extract_topo_profile_with_geod(
        csv_path: Path,
        start_point: tuple,
        azimuth: Any,
        distance_km: Any,
        lon_res=0.004579279,
        lat_res=0.004577823,
        ):
    """
    Generate a topographic profile using pyproj.Geod for great-circle calculations.
    
    Parameters:
    - csv_path: Path to the CSV file containing 'lon', 'lat', 'elevation'.
    - start_point: Tuple (lon, lat) as the starting point of the profile.
    - azimuth: Direction in degrees clockwise from North.
    - distance_km: Length of the profile in kilometers.
    - lon_res: Resolution in longitude degrees (e.g., 0.004579279).
    - lat_res: Resolution in latitude degrees (e.g., 0.004577823).

    Returns:
    - distances: Array of distances along the profile.
    - elevations: Array of interpolated elevation values.
    """
    # Step 1: Load the topographic data
    topo_data = pd.read_csv(csv_path)
    topo_data.columns = ['lon', 'lat', 'elevation']  # Ensure correct column names
    
    # Step 2: Initialize Geod for geodetic calculations
    geod = Geod(ellps="WGS84")
    
    # Calculate resolution in kilometers (average of lon_res and lat_res)
    avg_resolution_km = (lon_res + lat_res) / 2 * 111  # Approx conversion to km
    num_points = int(distance_km / avg_resolution_km)
    # Step 3: Generate points along the great-circle path
    lons, lats = [start_point[0]], [start_point[1]]  # Initialize lists with the start point
    for i in range(1, num_points + 1):
        # Calculate the next point along the path
        lon, lat, _ = geod.fwd(lons[-1], lats[-1], azimuth, avg_resolution_km * 1000)  # distance in meters
        lons.append(lon)
        lats.append(lat)
    
    # Step 4: Interpolate elevation values along the profile
    grid_elevations = griddata(
        points=(topo_data['lon'], topo_data['lat']),
        values=topo_data['elevation'],
        xi=(lons, lats),
        method='linear'
    )
    
    # Generate distances along the profile
    distances = np.linspace(0, distance_km, num_points + 1)
    
    return distances, grid_elevations

def add_topo_profile(ax: Axes, distances, elevations, profile_letter='', ax_alpha=1, show_ylabel=True):
    """
    Plot the topographic profile.
    
    Parameters:
    - distances: Array of distances along the profile.
    - elevations: Array of interpolated elevation values.
    """
    #alpha
    ax.patch.set_alpha(ax_alpha)

    ax.fill_between(distances, elevations / 1000, color="black", alpha=0.7)  # Convert to km
    ax.axhline(0, color="blue", linestyle="--", linewidth=0.8, alpha=0.8)  # Groundline
    
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)  # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    ax.spines['bottom'].set_visible(False)  # Hide the bottom spine    
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)  # No x-axis ticks/labels    
    if show_ylabel:
        ax.set_ylabel("Topo (km)", fontsize=13, weight='bold')
    ax.set_xlim(distances[0], distances[-1])

    # Only keep left y-axis
    ax.spines['left'].set_position(('outward', 5))  # Offset the left spine slightly
    ax.tick_params(axis="y", which="both", direction="out", length=5)  # Customize y-tick style
    ax.set_title(f"Profile: {profile_letter}-{profile_letter}'", fontsize=20, weight='bold')

def _add_epicenter(
    x: float,
    y: float,
    ax,
    size=10,
    color='yellow',
    markeredgecolor='black',
    label='Epicenter',
    alpha=0.5,
    zorder=10,
    markeredgewidth=1.0
):
    """## adding epicenter."""
    ax.plot(
        x,
        y,
        '*',
        markersize=size,
        color=color,
        markeredgecolor=markeredgecolor,
        markeredgewidth=markeredgewidth,
        label=label,
        alpha=alpha,
        zorder=zorder
    )


def plot_basemap(
    geo_ax,
    # axes,
    stations: list[pd.DataFrame],
    # fig_dir: Path,
    region: list,
    # fig_name='map.png',
    title=None,
    eq_list = [],
    plot_station_name=False,
    legend_on_map=False,
    hillshade=False,
    # fig_alpha=False,
    cmap='gray',
    fault_lw=2,
    station_size=50,
    station_mask=station_mask,
    colors=[None],
    resolution='15s',
    vert_exag=10,
    adjust_grid=False,
    text_dist=0.01
):
    """
    This function only plot the basemap without events.
    """
    # fig, axes = plt.subplots(
    #     1,
    #     1,
    #     figsize=(12, 14),
    # )    
    # if fig_alpha:
    #     fig.patch.set_alpha(0)
    # geo_ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    get_mapview(
        region=region,
        ax=geo_ax,
        title=title,
        cmap=cmap,
        use_fault=True,
        fault_lw=fault_lw,
        hillshade=hillshade,
        resolution=resolution,
        vert_exag=vert_exag,
        adjust_grid=adjust_grid
        )
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # Station
    for station, color in zip(stations, colors):  
        plot_station(
            df_station=station,
            geo_ax=geo_ax,
            region=region,
            plot_station_name=plot_station_name,
            station_size=station_size,
            text_dist=text_dist,
            station_mask=station_mask,
            color=color
        )    
    manual_elements = [
        Line2D(
            [0],
            [0],
            marker='^',
            color='w',
            label='Seismometer',
            markersize=7,
            markerfacecolor='c',
            markeredgecolor='k',
        ),
    ]
    for eq in eq_list:
        _add_epicenter(
            x=eq[0], y=eq[1], ax=geo_ax, size=eq[3], alpha=0.7, color=eq[4]
        )
        manual_elements.append(
            Line2D(
                [0],
                [0],
                marker='*',
                color='w',
                label=eq[3],
                markersize=10,
                markerfacecolor=eq[4],
                markeredgecolor='k',
            )
        )
    geo_ax.set_aspect('auto')
    if legend_on_map:
        geo_ax.legend(
            handles=manual_elements,  # + legend_elements,
            loc='upper left',
            markerscale=2,
            labelspacing=1.5,
            borderpad=1,
            fontsize=10,
            framealpha=0.6,
        )
    # axes.axis('off')
    # plt.tight_layout()
    # plt.savefig(fig_dir / fig_name, dpi=300, bbox_inches='tight')                         