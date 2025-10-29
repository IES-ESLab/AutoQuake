import multiprocessing as mp
import os
import calendar
from pathlib import Path
from datetime import datetime
import pandas as pd

# Absolute path for GAfocal dir
GAFOCAL_DIR = Path(__file__).parents[1].resolve() / 'GAfocal'

def utc_to_timestamp(utc_string: str):
    """Converts an ISO format string to a Unix timestamp."""
    # Ensure microseconds are correctly padded
    if '.' in utc_string:
        date_part, fractional_part = utc_string.split('.')
        fractional_part = fractional_part.ljust(6, '0')  # Pad to 6 digits
        utc_string = f'{date_part}.{fractional_part}'
    return datetime.fromisoformat(utc_string).timestamp()

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


def comparing_picks(df_gamma_picks, time_array, i, tol=3):
    print(f'event_{i}')
    df = df_gamma_picks[df_gamma_picks['event_index'] == i]
    if df.empty:
        return []
    df = df[df['station_id'].map(lambda x: x[1].isdigit())]
    std_time = df['phase_time'].apply(utc_to_timestamp).mean()
    index = [i for i, t in enumerate(time_array) if abs(std_time - t) <= tol]
    return index


def process_event_wrapper(args):
    """
    Process a single event index to extract matching rows and append the event_index.
    """
    event_index, df_gamma_picks, df_phasenet, time_array, tol = args
    index = comparing_picks(df_gamma_picks, time_array, event_index, tol)
    if not index:
        return None
    df_target = df_phasenet.loc[index].copy()
    df_target['event_index'] = event_index
    return df_target


def pseudo_picks_generator(
    phasenet_picks: Path,
    gamma_picks: Path,
    das_station: Path,
    das_station_20: Path,
    pseudo_gamma_picks: Path,
    tol=3
):
    """## Searching for phasenet_das picks that not used to associate into associate picks by time dependency.
    Example
    df_result = pseudo_picks_generator(
        phasenet_picks=phasenet_picks,
        gamma_picks=gamma_picks,
        pseudo_gamma_picks=pseudo_gamma_picks,
        das_station=Path('/home/patrick/Work/EQNet/tests/hualien_0403/station_das.csv'),
        das_station_20=Path('/home/patrick/Work/Hualien0403/stations/das_20.csv'),
    )
    """
    df_das_sta = pd.read_csv(das_station)
    df_das_20 = pd.read_csv(das_station_20)
    sta_set = set(df_das_sta['station']) - set(df_das_20['station'])
    df_phasenet = pd.read_csv(phasenet_picks)
    df_phasenet = (
        df_phasenet[df_phasenet['station_id'].isin(sta_set)]
        .copy()
        .reset_index(drop=True)
    )
    time_array = df_phasenet['phase_time'].apply(utc_to_timestamp).to_numpy()
    df_gamma_picks = pd.read_csv(gamma_picks)
    event_indices = set(x for x in df_gamma_picks['event_index'] if x != -1)
    # another function
    # Prepare arguments for multiprocessing
    args = [(i, df_gamma_picks, df_phasenet, time_array, tol) for i in event_indices]

    # Use multiprocessing Pool
    with mp.Pool(processes=40) as pool:
        results = pool.map(process_event_wrapper, args)

    # Combine results into a DataFrame
    collected_rows = pd.concat(
        [res for res in results if res is not None], ignore_index=True
    )

    collected_rows.drop(columns='channel_index', inplace=True)
    collected_rows['gamma_score'] = 999

    # Concatenate collected_rows with df_gamma_picks
    result_df = pd.concat([df_gamma_picks, collected_rows], ignore_index=True)

    # Optionally save to file
    result_df.to_csv(pseudo_gamma_picks, index=False)

    return result_df
# Processing phasenet picks
def gamma_preprocessing(pickings: Path, output_dir: Path) -> Path:
    df = pd.read_csv(pickings)
    df['station_id'] = df['station_id'].map(lambda x: str(x).split('.')[1])
    output_path = output_dir/pickings.name
    df.to_csv(output_path, index=False)
    return output_path

# Processing gamma catalog
def classify_event(row, picks):
    event_index = row['event_index']
    filtered_picks = picks[picks['event_index'] == event_index]

    # Filter DAS and seismic picks
    das_picks = filtered_picks[
        filtered_picks['station_id'].map(lambda x: x[1].isdigit())
    ]
    seis_picks = filtered_picks[
        filtered_picks['station_id'].map(lambda x: x[1].isalpha())
    ]

    # Count phase types
    das_counts = das_picks['phase_type'].value_counts()
    seis_counts = seis_picks['phase_type'].value_counts()

    # Extract counts
    das_count_p = das_counts.get('P', 0)
    das_count_s = das_counts.get('S', 0)
    seis_count_p = seis_counts.get('P', 0)
    seis_count_s = seis_counts.get('S', 0)

    # Classify event
    if seis_count_p >= 6 and seis_count_s >= 2:
        event_type = 1 if das_count_p >= 15 else 2
    elif das_count_p >= 15:
        event_type = 3
    else:
        event_type = 4

    # Add results to row
    row['seis_p_picks'] = seis_count_p
    row['seis_s_picks'] = seis_count_s
    row['das_p_picks'] = das_count_p
    row['das_s_picks'] = das_count_s
    row['event_type'] = event_type
    return row
    
def get_detailed_picks(gamma_events: Path, gamma_picks: Path):
    df_events = pd.read_csv(gamma_events)
    df_picks = pd.read_csv(gamma_picks)
    df_events = df_events.apply(
        lambda row: classify_event(row, df_picks), axis=1
    )
    return df_events

#H3DD
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
    
def process_for_h3dd_twice(
    station: Path,
    dout: Path,
    event_name_1: str,
    result_path: Path,
)-> tuple[Path, Path]:
    df_h3dd_events, df_h3dd_picks = process_h3dd(
        dout_file=dout, station_info=station
    )
    for name, df in zip(['events', 'picks'], [df_h3dd_events, df_h3dd_picks]):
        df.rename(columns={'h3dd_event_index': 'event_index'}, inplace=True)
        df.to_csv(result_path / f'{event_name_1}_{name}.csv', index=False)

    h3dd_events_first = result_path / f'{event_name_1}_events.csv'
    h3dd_picks_first = result_path / f'{event_name_1}_picks.csv'
    return h3dd_events_first, h3dd_picks_first

def get_index_table(df: pd.DataFrame) -> pd.DataFrame:
    if 'h3dd_event_index' in df.columns:
        df_table = df.loc[:, ['event_index', 'h3dd_event_index']]
    else:
        df['h3dd_event_index'] = df.index
        df_table = df.loc[:, ['event_index', 'h3dd_event_index']]
    assert isinstance(df_table, pd.DataFrame)
    return df_table


def index_h3dd2gamma(df_table: pd.DataFrame, h3dd_index: int):
    if all(col in df_table.columns for col in ['event_index', 'h3dd_event_index']):
        return df_table[df_table['h3dd_event_index'] == h3dd_index]['event_index'].iloc[
            0
        ]  # noqa: E501
    else:
        raise ValueError(
            "DataFrame must contain 'event_index' and 'h3dd_event_index' columns."
        )
    
def pol_mag_to_dout(
    ori_dout: Path,
    result_path: Path,
    polarity_picks: Path,
    magnitude_events: Path,
    magnitude_picks: Path,
    output_path=GAFOCAL_DIR,
):
    """
    Combining polarity and magnitude information into dout.
    #NOTE: Your polarity and magnitude csv must contain h3dd_event_index column.
    """
    df_pol = pd.read_csv(polarity_picks)
    df_mag_event = pd.read_csv(magnitude_events)
    df_mag_pick = pd.read_csv(magnitude_picks)
    output_dout = output_path / f'{ori_dout.name}'
    with open(ori_dout) as r:
        lines = r.readlines()
    with open(output_dout, 'w') as fo:
        h3dd_event_index = -1
        for line in lines:
            if line.strip().split()[-1] == '3DD':
                h3dd_event_index += 1
                event_mag = round(
                    df_mag_event[
                        df_mag_event['h3dd_event_index'] == h3dd_event_index
                    ]['magnitude'].iloc[0],
                    2,
                )

                fo.write(f'{line[:40]}{event_mag:4.2f}{line[44:]}')
            elif line[35:39] == '1.00':
                station = line[:5].strip()
                sta_mag = round(
                    df_mag_pick[
                        (df_mag_pick['h3dd_event_index'] == h3dd_event_index)
                        & (df_mag_pick['station_id'] == station)
                    ]['magnitude'].iloc[0],
                    2,
                )
                #TODO: There might exist the waveform skip in polarity process, find out why.
                try:
                    polarity = df_pol[
                        (
                            df_pol['h3dd_event_index']
                            == h3dd_event_index
                        )
                        & (df_pol['station_id'] == station)
                    ]['polarity'].iloc[0]
                except IndexError as e:
                    polarity = 'x'
                    # logging.info(f'H3DD event_{h3dd_event_index}: {station} no polarity picks {e}')
                if polarity == 'U':
                    polarity = '+'
                elif polarity == 'D':
                    polarity = '-'
                else:
                    polarity = ' '

                fo.write(
                    f'{line[:19]}{polarity}{line[20:55]} 0.00 0.00 0.00 {sta_mag:4.2f} 0   0.0\n'
                )
    os.system(f"cp {output_dout} {result_path}")
    return ori_dout.name

def pol_to_dout(
    ori_dout: Path,
    result_path: Path,
    df_reorder_event: pd.DataFrame,
    polarity_picks: Path,
    output_path=GAFOCAL_DIR,
):
    """
    Combining polarity and magnitude information into dout.
    """
    df_table = get_index_table(df=df_reorder_event)

    df_pol = pd.read_csv(polarity_picks)
    output_dout = output_path / f'{ori_dout.name}'
    with open(output_dout, 'w') as fo:
        with open(ori_dout) as r:
            lines = r.readlines()
        h3dd_event_index = -1
        for line in lines:
            if line.strip().split()[-1] == '3DD':
                h3dd_event_index += 1
                fo.write(line)
            elif line[35:39] == '1.00':
                station = line[:5].strip()
                polarity = df_pol[
                    (
                        df_pol['event_index']
                        == index_h3dd2gamma(df_table, h3dd_event_index)
                    )
                    & (df_pol['station_id'] == station)
                ]['polarity'].iloc[0]
                if polarity == 'U':
                    polarity = '+'
                elif polarity == 'D':
                    polarity = '-'
                else:
                    polarity = ' '

                fo.write(
                    f'{line[:19]}{polarity}{line[20:55]} 0.00 0.00 0.00 0.00 0   0.0\n'
                )
    os.system(f"cp {output_dout} {result_path}")
    return ori_dout.name