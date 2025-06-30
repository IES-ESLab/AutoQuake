# import glob
import logging

# import calendar
import multiprocessing as mp
from datetime import datetime
from multiprocessing import Pool

# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
from cartopy.mpl.geoaxes import GeoAxes

# from obspy.imaging.beachball import beach
from geopy.distance import geodesic
from matplotlib.axes import Axes
from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball, mpl_color

# from collections import defaultdict
from ._plot_base import (
    _hout_generate,
    catalog_filter,
    check_format,
    find_gafocal_polarity,
    plot_beachball_info,
    plot_polarity_waveform,
    plot_station,
    station_mask,
)

EQUIP_FILTER = lambda x: str(x)[0].isalpha()
EVENT_FILTER = lambda x: str(x)[0].isdigit()



def find_inter_station(cwa_dout, gamma_picks, station_mask=station_mask):
    """## Comparing GaMMA with CWA dout file to find the common station
    This function is to quick compare, too much detail in it!
    """
    with open(cwa_dout) as r:
        lines = r.readlines()
    cwa_sta_list = []
    for line in lines:
        if line[:2].strip().isalpha():
            cwa_sta_list.append(line[:5].strip())
    df_picks = pd.read_csv(gamma_picks)
    gamma_set = set(df_picks[df_picks['station_id'].map(station_mask)]['station_id'])
    return set(cwa_sta_list) & gamma_set


def _status_message_and_fig_name(
    catalog_list: list[Path],
    name_list: list[str],
    figure_parent_dir: Path,
    **options: dict[str, bool],
) -> Path:
    """
    print the status message and decide the fig name to save.
    """
    print(
        f'Main catalog: {name_list[0]}({catalog_list[0]})\nCompared catalog: {name_list[1]}({catalog_list[1]})\n'
    )
    print('CURRENT STATUS:')
    use_ori = options.get('use_ori', False)
    use_both = options.get('use_both', False)
    use_common = options.get('use_common', False)
    use_main = options.get('use_main', False)
    figure_dir = figure_parent_dir / f'{name_list[0]}_{name_list[1]}'
    figure_dir.mkdir(parents=True, exist_ok=True)
    if use_ori:
        if use_both:
            print('Using both catalog to plot the original distribution')
            return figure_dir / f'{name_list[0]}_{name_list[1]}_ori.png'
        else:
            if use_main:
                print(f'Using {name_list[0]} catalog to plot the original distribution')
                return figure_dir / f'{name_list[0]}_ori.png'
            else:
                print(f'Using {name_list[1]} catalog to plot the original distribution')
                return figure_dir / f'{name_list[1]}_ori.png'
    elif use_both:
        if use_common:
            print('Using both catalog to plot the common events distribution')
            return figure_dir / f'{name_list[0]}_{name_list[1]}_common.png'
        else:
            print('Using both catalog to plot the unique events distribution')
            return figure_dir / f'{name_list[0]}_{name_list[1]}_only.png'
    else:
        if use_common:
            if use_main:
                print(
                    f'Using {name_list[0]} catalog to plot the common events distribution'
                )
                return figure_dir / f'{name_list[0]}_common.png'
            else:
                print(
                    f'Using {name_list[1]} catalog to plot the common events distribution'
                )
                return figure_dir / f'{name_list[1]}_common.png'
        else:
            if use_main:
                print(
                    f'Using {name_list[0]} catalog to plot the unique events distribution'
                )
                return figure_dir / f'{name_list[0]}_only.png'
            else:
                print(
                    f'Using {name_list[1]} catalog to plot the unique events distribution'
                )
                return figure_dir / f'{name_list[1]}_only.png'


def catalog_compare(
    catalog: dict[int, dict[str, Any]], time_tol=5.0, dist_tol=10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Finding the common and unique events in 2 given catalog. (main = 0, standard = 1)
    # TODO: setting a rule to automatically decide which catalog is the main catalog.
    # TODO: a better naming idiom, comp catalog or standard catalog.
    """
    df_main, timestamp_main = check_format(catalog=catalog, i=0)
    df_standard, timestamp_std = check_format(catalog=catalog, i=1)
    df_main.loc[:, 'comp_index'] = -1
    df_standard.loc[:, 'comp_index'] = -1
    diff_time = timestamp_main - timestamp_std[:, np.newaxis]
    std_indices, main_indices = np.where(np.abs(diff_time) < time_tol)
    duplicates = {x for x in list(std_indices) if list(std_indices).count(x) > 1}
    main_loc_box = []
    for i in set(std_indices):
        std_point = (df_standard.loc[i].latitude, df_standard.loc[i].longitude)
        # print(df_standard.loc[i])
        if i in duplicates:
            locations = [index for index, value in enumerate(std_indices) if value == i]
            test_dict = {}
            for j in locations:
                main_loc = main_indices[j]
                value = diff_time[i][main_loc]
                main_point = (
                    df_main.loc[main_loc].latitude,
                    df_main.loc[main_loc].longitude,
                )
                # print(df_main.loc[main_loc])
                dist = geodesic(std_point, main_point).kilometers
                if dist > dist_tol:
                    print(f'dist: {dist}')
                    continue
                test_dict[main_loc] = value
            if test_dict:
                print(test_dict)
                smallest_key = min(test_dict, key=test_dict.get)
                df_main.at[smallest_key, 'comp_index'] = i
                df_standard.at[i, 'comp_index'] = smallest_key
        else:
            location = [index for index, value in enumerate(std_indices) if value == i][
                0
            ]
            main_loc = main_indices[location]
            main_point = (
                df_main.loc[main_loc].latitude,
                df_main.loc[main_loc].longitude,
            )
            # print(df_main.loc[main_loc])
            dist = geodesic(std_point, main_point).kilometers
            if dist > dist_tol:
                continue

            if main_loc not in main_loc_box:
                main_loc_box.append(main_loc)
            else:
                print(f'repeated main_loc: {main_loc}')
                continue
            df_main.at[main_loc, 'comp_index'] = i
            df_standard.at[i, 'comp_index'] = main_loc

    # post process of dataframe
    main_common = df_main[df_main['comp_index'] != -1]
    main_only = df_main[df_main['comp_index'] == -1]

    standard_common = df_standard[df_standard['comp_index'] != -1]
    standard_only = df_standard[df_standard['comp_index'] == -1]

    print(f'Main catalog: {len(main_common)}/{len(df_main)} founded')
    print(f'Compared catalog: {len(standard_common)}/{len(df_standard)} founded')

    return main_common, standard_common, main_only, standard_only


def pack(catalog_list: list[Path], name_list: list[str]) -> dict[int, dict[str, Any]]:
    pack_dict = {}
    for i, (catalog_path, name) in enumerate(zip(catalog_list, name_list)):
        pack_dict[i] = {'catalog': catalog_path, 'name': name}
    return pack_dict


def plot_ori(
    catalog: dict[int, dict[str, Any]],
    catalog_range: dict[str, float],
    use_both: bool,
    use_main: bool,
    axes: Axes,
    geo_ax: GeoAxes,
):
    """
    plotting original profiles if use_ori.
    """
    if use_both:
        key_list = catalog.keys()
    elif use_main:
        key_list = [0]
    else:
        key_list = [1]
    for i in key_list:
        catalog_df, _ = check_format(catalog=catalog, i=i)
        equip = catalog[i]['name']

        # cmap = "viridis"
        catalog_df = catalog_filter(catalog_df=catalog_df, catalog_range=catalog_range)
        if 'event_type' in catalog_df.columns:
            # This part we want to visualize the distribution of event_type.
            color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
            label_map = {
                1: 'seis 6P2S + DAS 15P',
                2: 'seis 6P2S',
                3: 'DAS 15P',
                4: 'Not reach the standard',
            }
            for event_type, group in catalog_df.groupby('event_type'):
                num = len(group)
                geo_ax.scatter(
                    group['longitude'],
                    group['latitude'],
                    s=5,
                    c=color_map[event_type],
                    alpha=0.5,
                    label=f'{label_map[event_type]}: {num}',
                    rasterized=True,
                )

                axes[0, 1].scatter(
                    group['depth_km'],
                    group['latitude'],
                    s=5,
                    c=color_map[event_type],
                    alpha=0.5,
                    label=label_map[event_type],
                    rasterized=True,
                )

                axes[1, 0].scatter(
                    group['longitude'],
                    group['depth_km'],
                    s=5,
                    c=color_map[event_type],
                    alpha=0.5,
                    label=label_map[event_type],
                    rasterized=True,
                )
        else:
            geo_ax.scatter(
                catalog_df['longitude'],
                catalog_df['latitude'],
                s=5,
                c='b' if i == 1 else 'r',
                alpha=0.5,
                label=f'{equip} event num: {len(catalog_df)}',
                rasterized=True,
            )

            axes[0, 1].scatter(
                catalog_df['depth_km'],
                catalog_df['latitude'],
                s=5,
                c='b' if i == 1 else 'r',
                alpha=0.5,
                label=f'{equip}',
                rasterized=True,
            )

            axes[1, 0].scatter(
                catalog_df['longitude'],
                catalog_df['depth_km'],
                s=5,
                c='b' if i == 1 else 'r',
                alpha=0.5,
                label=f'{equip}',
                rasterized=True,
            )


def plot_bypass(
    catalog: dict[int, dict[str, Any]],
    tol: int,
    use_both: bool,
    use_main: bool,
    use_common: bool,
):
    """
    Selecting the specific list format for plot_scenario.
    """
    main_common, comp_common, main_only, comp_only = catalog_compare(catalog, tol)
    if use_both:
        if use_common:
            df_list = [main_common, comp_common]
            nm_list = [catalog[0]['name'], catalog[1]['name']]
        else:
            df_list = [main_only, comp_only]
            nm_list = [catalog[0]['name'], catalog[1]['name']]
    else:
        if use_common:
            if use_main:
                df_list = [main_common]
                nm_list = [catalog[0]['name']]
            else:
                df_list = [comp_common]
                nm_list = [catalog[1]['name']]
        else:
            if use_main:
                df_list = [main_only]
                nm_list = [catalog[0]['name']]
            else:
                df_list = [comp_only]
                nm_list = [catalog[1]['name']]
    return df_list, nm_list


def plot_scenario(
    catalog: dict[int, dict[str, Any]],
    tol: int,
    catalog_range: dict[str, float],
    axes: Axes,
    geo_ax: GeoAxes,
    use_both: bool,
    use_main: bool,
    use_common: bool,
):
    """
    Plotting profiles for different scenarios.
    """
    df_list, nm_list = plot_bypass(catalog, tol, use_both, use_main, use_common)
    for i, (catalog_df, equip) in enumerate(zip(df_list, nm_list)):
        catalog_df = catalog_filter(catalog_df=catalog_df, catalog_range=catalog_range)
        if 'event_type' in catalog_df.columns:
            # This part we want to visualize the distribution of event_type.
            color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
            label_map = {
                1: 'seis 6P2S + DAS 15P',
                2: 'seis 6P2S',
                3: 'DAS 15P',
                4: 'Not reach the standard',
            }
            for event_type, group in catalog_df.groupby('event_type'):
                geo_ax.scatter(
                    group['longitude'],
                    group['latitude'],
                    s=5,
                    c=color_map[event_type],
                    alpha=0.5,
                    label=label_map[event_type],
                    rasterized=True,
                )

                axes[0, 1].scatter(
                    group['depth_km'],
                    group['latitude'],
                    s=5,
                    c=color_map[event_type],
                    alpha=0.5,
                    label=label_map[event_type],
                    rasterized=True,
                )

                axes[1, 0].scatter(
                    group['longitude'],
                    group['depth_km'],
                    s=5,
                    c=color_map[event_type],
                    alpha=0.5,
                    label=label_map[event_type],
                    rasterized=True,
                )
        else:
            geo_ax.scatter(
                catalog_df['longitude'],
                catalog_df['latitude'],
                s=5,
                c='b' if i == 1 else 'r',
                alpha=0.5,
                label=f'{equip}\ncommon event num: {len(catalog_df)}',
                rasterized=True,
            )

            axes[0, 1].scatter(
                catalog_df['depth_km'],
                catalog_df['latitude'],
                s=5,
                c='b' if i == 1 else 'r',
                alpha=0.5,
                label=equip,
                rasterized=True,
            )

            axes[1, 0].scatter(
                catalog_df['longitude'],
                catalog_df['depth_km'],
                s=5,
                c='b' if i == 1 else 'r',
                alpha=0.5,
                label=equip,
                rasterized=True,
            )


def run_profile(
    catalog: dict[int, dict[str, Any]],
    all_station_info: Path,
    map_range: dict[str, float],
    catalog_range: dict[str, float],
    tol: int,
    figure_path: Path,
    **options: dict[str, bool],
) -> None:
    """
    main ploting function, contaning the cartopy map setting.
    """
    use_ori = options.get('use_ori', False)
    use_both = options.get('use_both', False)
    use_common = options.get('use_common', False)
    use_main = options.get('use_main', False)
    params = {
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'lines.linewidth': 3,
        'lines.markersize': 10,
        'image.origin': 'lower',
        'figure.figsize': (4 * 2.5, 3 * 2.5),
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
    }
    matplotlib.rcParams.update(params)
    ls = LightSource(azdeg=0, altdeg=45)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12,
            12
            * (map_range['max_lat'] - map_range['min_lat'])
            / (
                (map_range['max_lon'] - map_range['min_lon'])
                * np.cos(np.deg2rad(map_range['min_lat']))
            ),
        ),
        gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]},
    )
    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    region = [
        map_range['min_lon'],
        map_range['max_lon'],
        map_range['min_lat'],
        map_range['max_lat'],
    ]
    topo = (
        pygmt.datasets.load_earth_relief(resolution='15s', region=region).to_numpy()
        / 1e3
    )  # km
    x = np.linspace(map_range['min_lon'], map_range['max_lon'], topo.shape[1])
    y = np.linspace(map_range['min_lat'], map_range['max_lat'], topo.shape[0])
    dx, dy = 1, 1
    xgrid, ygrid = np.meshgrid(x, y)

    geo_ax.pcolormesh(
        xgrid,
        ygrid,
        ls.hillshade(topo, vert_exag=10, dx=dx, dy=dy),
        vmin=-1,
        shading='gouraud',
        cmap='gray',
        alpha=1.0,
        antialiased=True,
        rasterized=True,
    )
    geo_ax.coastlines(resolution='10m', color='black', linewidth=1)
    geo_ax.add_feature(cfeature.BORDERS, linestyle=':')
    # geo_ax.add_feature(cfeature.LAND)
    # geo_ax.add_feature(cfeature.OCEAN)

    # You can continue to plot other subplots and customize them as needed
    # Example of setting extent and gridlines
    geo_ax.set_extent(region, crs=ccrs.PlateCarree())
    gl = geo_ax.gridlines(draw_labels=True)
    gl.top_labels = False  # Turn off top labels
    gl.right_labels = False

    if use_ori:
        plot_ori(
            catalog=catalog,
            catalog_range=catalog_range,
            use_both=use_both,
            use_main=use_main,
            axes=axes,
            geo_ax=geo_ax,
        )
    else:
        plot_scenario(
            catalog=catalog,
            tol=tol,
            catalog_range=catalog_range,
            axes=axes,
            geo_ax=geo_ax,
            use_both=use_both,
            use_common=use_common,
            use_main=use_main,
        )
    df_station = pd.read_csv(all_station_info)
    plot_station(df_station, geo_ax)
    # geo_ax.set_title
    geo_ax.autoscale(tight=True)
    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()
    # geo_ax.set_aspect(1.0/np.cos(np.deg2rad(min_lat)))
    geo_ax.set_aspect('auto')
    # geo_ax.set_xlim(xlim)
    # geo_ax.set_ylim(ylim)
    geo_ax.legend(markerscale=2)  # markerscale=5
    # geo_ax.set_ylabel("Latitude")
    axes[0, 1].autoscale(tight=True)
    axes[0, 1].set_ylim(ylim)
    axes[0, 1].set_xlim([0, map_range['max_depth'] + 1])
    axes[0, 1].set_xlabel('Depth (km)')
    axes[0, 1].set_ylabel('Latitude')

    axes[1, 0].autoscale(tight=True)
    axes[1, 0].set_xlim(xlim)
    axes[1, 0].set_ylim([0, map_range['max_depth'] + 1])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_ylabel('Depth (km)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 1].axis('off')
    axes[0, 0].axis('off')
    plt.tight_layout()
    # *** haven't finish the customized figure_path
    plt.savefig(figure_path)  # using fig.savefig when we have several figs.


# Function to plot profiles in parallel
def get_profile(
    figure_parent_dir: Path,
    catalog_list: list[Path],
    name_list: list[str],
    all_station_info: Path,
    map_range: dict[str, float],
    catalog_range: dict[str, float],
    tol: int,
    options: dict[str, bool],
):
    """
    Plotting profiles in different scenario.

    Args:
        figure_parent_dir (Path): The parent directory for saving figures.
        catalog_list (list[Path]): A list of catalog file paths.
        name_list (list[str]): A list of names corresponding to the catalog files.
        all_station_info (Path): The path to the file containing all station information.
        map_range (dict[str, float]): A dictionary specifying the map range.
            - 'min_lat' (float): Minimum latitude.
            - 'max_lat' (float): Maximum latitude.
            - 'min_lon' (float): Minimum longitude.
            - 'max_lon' (float): Maximum longitude.
            - 'min_depth' (float): Minimum depth.
            - 'max_depth' (float): Maximum depth.
        catalog_range (dict[str, float]): A dictionary specifying the catalog range.
            - 'min_lat' (float): Minimum latitude.
            - 'max_lat' (float): Maximum latitude.
            - 'min_lon' (float): Minimum longitude.
            - 'max_lon' (float): Maximum longitude.
            - 'min_depth' (float): Minimum depth.
            - 'max_depth' (float): Maximum depth.
        tol (int): The tolerance value.
        options (Dict[str, bool]): Dictionary of boolean flags such as:
            - 'use_ori' (bool): Whether to use the original catalog.
            - 'use_both' (bool): Whether to use both catalogs.
            - 'use_common' (bool): Whether to use the common catalog.
            - 'use_main' (bool): Whether to use the main catalog.

    Retruns:
        str: A message indicating the completion of the plotting process and the saved figure path.
    """
    # Unpack options
    figure_path = _status_message_and_fig_name(
        catalog_list=catalog_list,
        name_list=name_list,
        figure_parent_dir=figure_parent_dir,
        **options,
    )
    catalog_dict = pack(catalog_list, name_list)
    run_profile(
        catalog=catalog_dict,
        all_station_info=all_station_info,
        map_range=map_range,
        catalog_range=catalog_range,
        tol=tol,
        figure_path=figure_path,
        **options,
    )
    return f'complete with {figure_path}'


def plot_profile(
    figure_parent_dir: Path,
    catalog_list: list[Path],
    name_list: list[str],
    all_station_info: Path,
    map_range: dict[str, float],
    catalog_range: dict[str, float],
    tol: int,
    options_list=None,
):
    """
    Plotting profiles with multiprocessing.

    Args:
        figure_parent_dir (Path): The parent directory for saving figures.
        catalog_list (list[Path]): A list of catalog file paths.
        name_list (list[str]): A list of names corresponding to the catalog files.
        all_station_info (Path): The path to the file containing all station information.
        map_range (dict[str, float]): A dictionary specifying the map range.
        catalog_range (dict[str, float]): A dictionary specifying the catalog range.
        tol (int): The tolerance value.
        options_list (list[dict[str, bool]], optional): A list of options for plotting.

    Example:
        plot_profile(
            figure_parent_dir=Path("./figures"),
            catalog_list=[
                Path("./catalogs/catalog1.csv"), Path("./catalogs/catalog2.csv")
                ],
            name_list=["Catalog 1", "Catalog 2"],
            all_station_info=Path("./station_info.csv"),
            map_range={
                "min_lat": 30, "max_lat": 40,
                "min_lon": -120, "max_lon": -110,
                "min_depth": 0, "max_depth": 100
                },
            catalog_range={
                "min_lat": 30, "max_lat": 40,
                "min_lon": -120, "max_lon": -110,
                "min_depth": 0, "max_depth": 100
                },
            tol=10
        )
    """
    if options_list is None:
        options_list = [
            {'use_ori': True, 'use_both': True},
            {'use_ori': True, 'use_both': False, 'use_main': True},
            {'use_ori': True, 'use_both': False, 'use_main': False},
            {'use_ori': False, 'use_both': True, 'use_common': True},
            {'use_ori': False, 'use_both': False, 'use_common': True, 'use_main': True},
            {
                'use_ori': False,
                'use_both': False,
                'use_common': True,
                'use_main': False,
            },
            {'use_ori': False, 'use_both': True, 'use_common': False},
            {
                'use_ori': False,
                'use_both': False,
                'use_common': False,
                'use_main': True,
            },
            {
                'use_ori': False,
                'use_both': False,
                'use_common': False,
                'use_main': False,
            },
        ]
    with Pool(processes=9) as pool:
        results = pool.starmap(
            get_profile,
            [
                (
                    figure_parent_dir,
                    catalog_list,
                    name_list,
                    all_station_info,
                    map_range,
                    catalog_range,
                    tol,
                    options,
                )
                for options in options_list
            ],
        )

    for result in results:
        print(result)


def find_index_backward(gamma_catalog: Path, focal_dict: dict) -> int:
    """
    This is used to find the gamma event that survive in comparing CWA catalog
    and gamma catalog.
    """
    df_catalog = pd.read_csv(gamma_catalog)
    catalog_dt = pd.to_datetime(df_catalog['time'])
    std_time = pd.to_datetime(focal_dict['utc_time'])
    if min(abs((catalog_dt - std_time).dt.total_seconds())) < 1:
        close_row = abs((catalog_dt - std_time).dt.total_seconds()).idxmin()
        return df_catalog.loc[close_row].event_index
    else:
        raise ValueError('Correspnded event not founded.')


# TODO: Creating the class for comparing.
# class CompUnit:
#     """
#     This class is basic unit for comparing the catalogs.
#     here we would check the format of each format, also convert it into DataFrame.
#     """
#     def __init__(self, name: str, gafocal_catalog=None, polarity_dout=None, mag_catalog=None, gamma_catalog=None):
#         self.name = name
#         if gafocal_catalog is not None:
#             self.gafocal_catalog = gafocal_catalog
#             if polarity_dout is not None:
#                 self.polarity_dout = polarity_dout
#         if gamma_catalog is None:
#             if mag_catalog is None:
#                 if gafocal_catalog is None:
#                     raise ValueError("We need to provided at least 1 catalog.")
#             else:
#                 self.aso_events = mag_catalog
#         else:
#             self.aso_events = gamma_catalog


# class Comp:
#     def __init__(self, comp_unit_list: list[CompUnit]):
#         pass
def _plot_loc_das(df_das_station, das_region, focal_dict_list, fig=None):
    # map_proj = ccrs.PlateCarree()
    # tick_proj = ccrs.PlateCarree()

    if fig is None:
        fig = plt.figure()
    ax = fig.add_axes([0.1, 0.0, 0.4, 0.47])
    # plot das
    ax.scatter(
        x=df_das_station['longitude'],
        y=df_das_station['latitude'],
        marker='.',
        color='silver',
        s=10,
        zorder=2,
    )

    color_list = ['r', 'b']
    for i, (focal_dict, color) in enumerate(zip(focal_dict_list, color_list)):
        ax.scatter(
            x=focal_dict['longitude'],
            y=focal_dict['latitude'],
            marker='*',
            color=color,
            s=200,
            alpha=0.5,
            zorder=4,
        )
        # plot associated station
        # TODO: maybe there exists a better way, too complicated, modify this
        if i == 0:
            for sta in focal_dict['station_info'].keys():
                if sta in list(df_das_station['station']):
                    ax.scatter(
                        x=df_das_station[df_das_station['station'] == sta]['longitude'],
                        y=df_das_station[df_das_station['station'] == sta]['latitude'],
                        marker='.',
                        color='darkorange',
                        s=5,
                        zorder=3,
                    )


def _plot_loc_seis(
    df_seis_station: pd.DataFrame,
    seis_region: list[float],
    focal_dict_list: list[dict],
    name_list: list[str],
    fig=None,
    gs=None,
):
    """
    This is success! DAS and both not test yet.
    ---
    check_loc_compare(
        station_list=seis_station_info,
        focal_dict_list=[gamma_focal_dict, cwa_focal_dict],
        name_list=['GaMMA', 'CWA'],
        seis_region=[119.7, 122.5, 21.7, 25.4]
        )
    """
    map_proj = ccrs.PlateCarree()
    # tick_proj = ccrs.PlateCarree()

    if fig is None:
        fig = plt.figure(figsize=(36, 20))
        gs = GridSpec(10, 18, left=0, right=0.9, top=0.95, bottom=0.42, wspace=0.1)
    ax = fig.add_subplot(gs[1:, :4], projection=map_proj)
    ax.coastlines()
    ax.set_extent(seis_region)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)

    ax.scatter(
        x=df_seis_station['longitude'],
        y=df_seis_station['latitude'],
        marker='^',
        color='silver',
        s=80,
        zorder=2,
    )
    # plot the event
    color_list = ['r', 'b']
    for i, (focal_dict, name, color) in enumerate(
        zip(focal_dict_list, name_list, color_list)
    ):
        ax.scatter(
            x=focal_dict['longitude'],
            y=focal_dict['latitude'],
            marker='*',
            color=color,
            s=300,
            zorder=4,
        )
        ax.text(
            seis_region[0],
            seis_region[3] + 0.05 + i / 6,
            f"{name} Catalog:\nEvent time: {focal_dict['utc_time']}, Lat: {focal_dict['latitude']},Lon: {focal_dict['longitude']}, Depth: {focal_dict['depth']}",
            color=color,
            fontsize=13,
            fontweight='bold',
        )
        # plot associated station
        # TODO: maybe there exists a better way
        if i == 0:
            for sta in focal_dict['station_info'].keys():
                if sta in list(df_seis_station['station']):
                    ax.scatter(
                        x=df_seis_station[df_seis_station['station'] == sta][
                            'longitude'
                        ],
                        y=df_seis_station[df_seis_station['station'] == sta][
                            'latitude'
                        ],
                        marker='^',
                        color='darkorange',
                        s=80,
                        zorder=3,
                    )


def _plot_loc_both(
    df_seis_station, df_das_station, seis_region, das_region, focal_dict_list, fig=None
):
    """
    TODO: Find the figsize for DAS+Seis
    focal_dict_list contains 2 dict, one is CWA, another is GaMMA, they both
    got event location that needs to plot on the map.
    """
    map_proj = ccrs.PlateCarree()
    # tick_proj = ccrs.PlateCarree()
    if fig is None:
        fig = plt.figure()
    ax = fig.add_axes([0.3, 0.5, 0.4, 0.8], projection=map_proj)
    # cartopy setting
    ax.coastlines()
    ax.set_extent(seis_region)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    # DAS map
    sub_ax = fig.add_axes([0.3, -0.15, 0.4, 0.55], projection=map_proj)
    sub_ax.coastlines()
    sub_ax.set_extent(das_region)
    sub_ax.add_feature(cfeature.LAND)
    # sub_ax1.add_feature(cfeature.OCEAN)
    sub_ax.add_feature(cfeature.COASTLINE)

    # plot all station
    ax.scatter(
        x=df_seis_station['longitude'],
        y=df_seis_station['latitude'],
        marker='^',
        color='silver',
        s=50,
        zorder=2,
    )
    sub_ax.scatter(
        x=df_das_station['longitude'],
        y=df_das_station['latitude'],
        marker='.',
        color='silver',
        s=5,
        zorder=2,
    )

    # plot the event
    color_list = ['r', 'b']
    for i, (focal_dict, color) in enumerate(zip(focal_dict_list, color_list)):
        ax.scatter(
            x=focal_dict['longitude'],
            y=focal_dict['latitude'],
            marker='*',
            color=color,
            s=100,
            zorder=4,
        )
        sub_ax.scatter(
            x=focal_dict['longitude'],
            y=focal_dict['latitude'],
            marker='*',
            color=color,
            s=30,
            zorder=4,
        )

        # plot associated station
        # TODO: maybe there exists a better way
        if i == 0:
            for sta in focal_dict['station_info'].keys():
                if sta in list(df_seis_station['station']):
                    ax.scatter(
                        x=df_seis_station[df_seis_station['station'] == sta][
                            'longitude'
                        ],
                        y=df_seis_station[df_seis_station['station'] == sta][
                            'latitude'
                        ],
                        marker='^',
                        color='darkorange',
                        zorder=3,
                    )
                elif sta in list(df_das_station['station']):
                    sub_ax.scatter(
                        x=df_das_station[df_das_station['station'] == sta]['longitude'],
                        y=df_das_station[df_das_station['station'] == sta]['latitude'],
                        marker='.',
                        color='darkorange',
                        s=5,
                        zorder=3,
                    )
                else:
                    raise ValueError('No this kind of station, check it.')


# plotter
def check_loc_compare(
    station_list,
    focal_dict_list,
    name_list,
    seis_region=None,
    das_region=None,
    fig=None,
):
    """
    The aim here is to plot the epicenter of these two Catalog.
    Parameters
    ----------
    region : Tuple[float, float, float, float]
        region[0]: min longitude
        region[1]: max longitude
        region[2]: min latitude
        region[3]: max latitude
    """
    # judgement
    df_station = pd.read_csv(station_list)
    df_seis_station = df_station[df_station['station'].map(station_mask)]
    df_das_station = df_station[~df_station['station'].map(station_mask)]
    if not df_seis_station.empty:
        if not df_das_station.empty:
            _plot_loc_both(
                df_seis_station,
                df_das_station,
                seis_region,
                das_region,
                focal_dict_list,
                fig,
            )
        else:
            _plot_loc_seis(
                df_seis_station, seis_region, focal_dict_list, name_list, fig
            )
    elif not df_das_station.empty:
        _plot_loc_das(df_das_station, das_region, focal_dict_list, fig)


def add_event_index(target_file: Path, output_file: Path, analyze_year: str):
    """
    Adding the index after h3dd format (both h3dd and polarity hout).
    """
    event_index = 0
    with open(target_file) as r:
        lines = r.readlines()
    with open(output_file, 'w') as event:
        for line in lines:
            if line.strip()[:4] == analyze_year:
                event_index += 1
                event.write(f"{' ':1}{line.strip()}{' ':1}{event_index:<5}\n")
            else:
                event.write(f"{' ':1}{line.strip()}{' ':1}{event_index:<5}\n")


def find_compared_gafocal_polarity(
    df_gafocal: pd.DataFrame,
    polarity_dout: Path,
    event_index: int,
    use_gamma: bool,
    get_waveform=True,
    h5_parent_dir=None,
    sac_parent_dir=None,
    hout_file=None,
    equip_filter=EQUIP_FILTER,
    event_filter=EVENT_FILTER,
):
    """
    From gafocal to find the match event's polarity information and more.
    """
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
    focal_dict = find_gafocal_polarity(
        df_gafocal=df_gafocal,
        df_hout=df_hout,
        polarity_dout=polarity_dout,
        event_index=event_index,
        equip_filter=equip_filter,
        event_filter=event_filter,
        get_waveform=get_waveform,
        sac_parent_dir=sac_parent_dir,
        h5_parent_dir=h5_parent_dir,
        comparing_mode=use_gamma,
    )
    return focal_dict


# plotter
def _get_beach_old(focal_dict, ax, color='r', source='AutoQuake'):
    """Plot the beachball diagram. older version"""
    mt = pmt.MomentTensor(
        strike=focal_dict['focal_plane']['strike'],
        dip=focal_dict['focal_plane']['dip'],
        rake=focal_dict['focal_plane']['rake'],
    )
    ax.set_axis_off()
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
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
    for sta in focal_dict['station_info'].keys():
        takeoff = focal_dict['station_info'][sta]['takeoff_angle']
        azi = focal_dict['station_info'][sta]['azimuth']
        polarity = focal_dict['station_info'][sta]['polarity']
        if polarity == ' ':
            continue
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

        ax.plot(
            x,
            y,
            '+' if polarity == '+' else 'o',
            ms=10.0 if polarity == '+' else 10.0 / np.sqrt(2.0),
            mew=2.0,
            mec='black',
            mfc='none',
        )
        ax.text(x + 0.025, y + 0.025, sta)


#     ax.text(
#         1.2,
#         -0.5,
#         f"{source}\nQuality index: {focal_dict['quality_index']}\nnum of station: {focal_dict['num_of_polarity']}\n\
# Strike: {focal_dict['focal_plane']['strike']}\nDip: {focal_dict['focal_plane']['dip']}\n\
# Rake: {focal_dict['focal_plane']['rake']}",
#         fontsize=20,
#     )


def get_beachball_check(
    i: int,
    main_common: pd.DataFrame,
    comp_common: pd.DataFrame,
    main_polarity: Path,
    comp_polarity: Path,
    source_list: list,  # [0] == main
    fig_dir: Path,
    sac_parent_dir: Path,
    h5_parent_dir: Path,
):
    logging.info(f'Event index: {i}')
    main_focal_dict = find_compared_gafocal_polarity(
        df_gafocal=main_common,
        polarity_dout=main_polarity,
        event_index=i,
        use_gamma=True,
        sac_parent_dir=sac_parent_dir,
        h5_parent_dir=h5_parent_dir,
    )
    comp_focal_dict = find_compared_gafocal_polarity(
        df_gafocal=comp_common,
        polarity_dout=comp_polarity,
        event_index=i,
        use_gamma=False,
        get_waveform=False,
    )
    fig_name = f"Event_{i}_{comp_focal_dict['utc_time'].replace('-','_').replace(':', '_')}_map.png"
    if (fig_dir / fig_name).exists():
        logging.info(f'Event_{i} beachball already done')
        return
    fig = plt.figure(figsize=(32, 18))
    gs1 = GridSpec(10, 18, left=0.65, right=0.9, top=0.95, bottom=0.05, wspace=0.1)
    plot_beachball_info(
        focal_dict_list=[main_focal_dict, comp_focal_dict],
        name_list=source_list,
        fig=fig,
        gs=gs1,
    )
    logging.info(f'Event_{i} beachball done')
    plot_polarity_waveform(
        main_focal_dict=main_focal_dict, comp_focal_dict=comp_focal_dict, fig=fig
    )
    logging.info(f'Event_{i} polarity waveform')
    plt.tight_layout()
    plt.savefig(
        fig_dir / fig_name,
        bbox_inches='tight',
        dpi=300,
    )


def plot_beachball_check(
    main_common: pd.DataFrame,
    comp_common: pd.DataFrame,
    main_polarity: Path,
    comp_polarity: Path,
    source_list: list,
    fig_dir: Path,
    sac_parent_dir: Path,
    h5_parent_dir: Path,
    processes=5,
):
    comp_index_list = list(main_common['comp_index'])
    with mp.Pool(processes=processes) as pool:
        pool.starmap(
            get_beachball_check,
            [
                (
                    i,
                    main_common,
                    comp_common,
                    main_polarity,
                    comp_polarity,
                    source_list,
                    fig_dir,
                    sac_parent_dir,
                    h5_parent_dir,
                )
                for i in comp_index_list
            ],
        )
