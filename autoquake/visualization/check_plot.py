# import os
# import glob
# import math
import logging

# from obspy.imaging.beachball import beach
import multiprocessing as mp
import string
import time
from functools import partial

# import calendar
# from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Geod

# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# from collections import defaultdict
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball, mpl_color
from scipy.stats import gaussian_kde

from ._plot_base import (
    _find_phasenet_das_pick,
    _find_phasenet_seis_pick,
    _preprocess_focal_files,
    _preprocess_phasenet_csv,
    _add_epicenter,
    add_on_utc_time,
    add_topo_profile,
    add_vel_profile,
    add_vel_view,
    catalog_filter,
    check_format,
    extract_topo_profile_with_geod,
    find_gafocal_polarity,
    find_gamma_h3dd,
    find_gamma_index,
    find_gamma_pick,
    find_sac_data,
    get_beach,
    get_mapview,
    get_only_beach,
    get_single_beach,
    get_tracer_beach,
    plot_station,
    plot_waveform_check,
    polarity_color_select,
    preprocess_gamma_csv,
    station_mask,
)

focal_size_mapping = {
    (0, 1): 10,
    (1, 2): 10,
    (2, 3): 10,
    (3, 4): 10,
    (4, 5): 15,
    (5, 6): 20,
    (6, 7): 30,
    (7, 8): 45,
}

# Define the intervals (keys of the dictionary)
# intervals = [(0, 4), (4, 5), (5, 6), (6, 7), (7, 8)]

# # Automatically generate colors from a colormap (viridis in this case)
# cmap = plt.cm.get_cmap('viridis', len(intervals))  # Discretize colormap into N levels
# colors = [cmap(i) for i in range(len(intervals))]  # Extract N discrete colors

# # Create a color mapping for intervals
# focal_color_mapping = {interval: color for interval, color in zip(intervals, colors)}

focal_color_mapping = {
    (0, 4): 'k',
    (4, 5): 'b',
    (5, 6): 'r',
    (6, 7): 'c',
    (7, 8): 'orange',    
}

def assign_color(value, color_mapping=focal_color_mapping):
    for interval, color in color_mapping.items():
        if interval[0] <= value < interval[1]:  # Check which interval the value falls in
            return color
    return 'grey'

def _find_asso_map(df_seis_station: pd.DataFrame, df_das_station: pd.DataFrame):
    """## return a code for map setting."""
    if df_seis_station.empty:
        # typically, only DAS
        pass
    elif df_das_station.empty:
        # typically, only seismic
        pass
    else:
        # both seismic and DAS
        pass

# def quality_threshold(row):
#     quality_int = row.quality_index
#     if quality_int < 2:
#         return 'k'
#     elif 2 <= quality_int < 5:
#         return 'bisque'
#     elif 5 <= quality_int < 7:
#         return 'goldenrod'
#     elif quality_int >= 7:
#         return 'darkorange'
    
# TODO: MODIFYING THE EVENT_DICT, NOT NECESSARY
def plot_asso_map(
    station: Path,
    status: str,
    event_dict: dict | None = None,
    df_event_picks: pd.DataFrame | None = None,
    main_ax: GeoAxes | None = None,  # typically seis
    sub_ax=None,  # typically das
    main_region: list | None = None,
    sub_region: list | None = None,
    use_gamma=True,
    use_h3dd=False,
    station_mask=station_mask,
    use_legend=False
):
    """
    ## This is the map for plotting association.
    """
    map_proj = ccrs.PlateCarree()
    tick_proj = ccrs.PlateCarree()
    # if main_ax is None and sub_ax is None:
    #     fig = plt.figure(figsize=(8, 12))
    # gs = GridSpec(2, 1, height_ratios=[3, 1])
    df_station = pd.read_csv(station)
    df_seis_station = df_station[df_station['station'].map(station_mask)]
    df_das_station = df_station[~df_station['station'].map(station_mask)]
    if not df_seis_station.empty:
        # get seismic station
        if main_ax is None:
            fig = plt.figure(figsize=(8, 12))
            gs = GridSpec(2, 1, height_ratios=[3, 1])
            main_ax = fig.add_subplot(gs[0], projection=map_proj)
        if main_region is None:
            main_region = [
                df_seis_station['longitude'].min() - 0.5,
                df_seis_station['longitude'].max() + 0.5,
                df_seis_station['latitude'].min() - 0.5,
                df_seis_station['latitude'].max() + 0.5,
            ]
        get_mapview(
            main_region,
            main_ax,
            title=f"{event_dict[status]['event_time']}\nLat: {event_dict[status]['event_lat']} Lon: {event_dict[status]['event_lon']} depth: {event_dict[status]['event_depth']}",
        )
        main_ax.scatter(
            x=df_seis_station['longitude'],
            y=df_seis_station['latitude'],
            marker='^',
            color='silver',
            edgecolors='k',
            s=50,
            zorder=2,
        )
        if use_gamma and event_dict is not None:
            main_ax.scatter(
                x=event_dict['gamma']['event_lon'],
                y=event_dict['gamma']['event_lat'],
                marker='*',
                color='y',
                s=100,
                zorder=4,
                label='GaMMA',
            )
        if use_h3dd and event_dict is not None:
            main_ax.scatter(
                x=event_dict['h3dd']['event_lon'],
                y=event_dict['h3dd']['event_lat'],
                marker='*',
                color='r',
                s=100,
                zorder=4,
                label='H3DD',
            )
        if use_legend:
            main_ax.legend(loc='lower right')
        if df_event_picks is not None:
            for sta in df_event_picks[df_event_picks['station_id'].map(station_mask)][
                'station_id'
            ].unique():
                main_ax.scatter(
                    x=df_seis_station[df_seis_station['station'] == sta]['longitude'],
                    y=df_seis_station[df_seis_station['station'] == sta]['latitude'],
                    marker='^',
                    color='darkorange',
                    edgecolors='k',
                    s=50,
                    zorder=3,
                )
        seis_gl = main_ax.gridlines(draw_labels=True)
        seis_gl.top_labels = False  # Turn off top labels
        seis_gl.right_labels = False
        main_ax.autoscale(tight=True)
        main_ax.set_aspect('auto')
    if not df_das_station.empty:
        if sub_ax is None:
            sub_ax = fig.add_subplot(gs[1], projection=map_proj)
        if sub_region is None:
            sub_region = [
                df_das_station['longitude'].min() - 0.01,
                df_das_station['longitude'].max() + 0.01,
                df_das_station['latitude'].min() - 0.01,
                df_das_station['latitude'].max() + 0.01,
            ]
        get_mapview(sub_region, sub_ax, title='')
        sub_ax.scatter(
            x=df_das_station['longitude'],
            y=df_das_station['latitude'],
            marker='.',
            color='silver',
            s=5,
            zorder=2,
        )
        # plot association
        if use_gamma and event_dict is not None:
            sub_ax.scatter(
                x=event_dict['gamma']['event_lon'],
                y=event_dict['gamma']['event_lat'],
                marker='*',
                color='y',
                s=100,
                zorder=4,
                label='GaMMA',
            )
        # plot relocation
        if use_h3dd and event_dict is not None:
            sub_ax.scatter(
                x=event_dict['h3dd']['event_lon'],
                y=event_dict['h3dd']['event_lat'],
                marker='*',
                color='r',
                s=100,
                zorder=4,
                label='H3DD',
            )
        sub_ax.legend()
        # plot association
        if df_event_picks is not None:
            for sta in df_event_picks[~df_event_picks['station_id'].map(station_mask)][
                'station_id'
            ].unique():
                sub_ax.scatter(
                    x=df_das_station[df_das_station['station'] == sta]['longitude'],
                    y=df_das_station[df_das_station['station'] == sta]['latitude'],
                    marker='.',
                    color='darkorange',
                    s=5,
                    zorder=3,
                )
        das_gl = sub_ax.gridlines(draw_labels=True)
        das_gl.top_labels = False  # Turn off top labels
        das_gl.right_labels = False
        # das_ax.autoscale(tight=True)
        # das_ax.set_aspect('auto')


def return_none_if_empty(df):
    if df.empty:
        return None
    return df


# This function currently modified by Hsi-An
def plot_asso(
    df_phasenet_picks: pd.DataFrame,
    gamma_picks: Path,
    gamma_events: Path,
    station: Path,
    event_i: int,
    fig_dir: Path,
    h3dd_hout=None,
    amplify_index=5,
    sac_parent_dir=None,
    h5_parent_dir=None,
    station_mask=station_mask,
    seis_region=None,
    das_region=None,
    pretime=-30,
    posttime=60,
    use_legend=False
):
    """
    ## Plotting the gamma and h3dd info.

    Args:
        - df_phasenet_picks (pd.DataFrame): phasenet picks.
        - gamma_picks (Path): Path to the gamma picks.
        - gamma_events (Path): Path to the gamma events.
        - station (Path): Path to the station info.
        - event_i (int): Index of the event to plot.
        - fig_dir (Path): Path to save the figure.

        - h3dd_hout (Path, optional): Path to the h3dd hout. Defaults to None.
        - amplify_index (int, optional): Amplify index for sac data. Defaults to 5.
        - sac_parent_dir (Path, optional): Path to the sac data. Defaults to None.
        - sac_dir_name (Path, optional): Name of the sac data directory. Defaults to None.
        - h5_parent_dir (Path, optional): Path to the h5 data (DAS). Defaults to None.
        - station_mask (function, optional): Function to mask the station. Defaults to station_mask.
        - seis_region (list, optional): Region for seismic plot. Defaults to None.
            - 4-element list: [min_lon, max_lon, min_lat, max_lat]
        - das_region (list, optional): Region for DAS plot. Defaults to None.
            - 4-element list: [min_lon, max_lon, min_lat, max_lat]
    """
    df_event, event_dict, df_event_picks = preprocess_gamma_csv(
        gamma_catalog=gamma_events,
        gamma_picks=gamma_picks,
        event_i=event_i,
        h3dd_hout=h3dd_hout,
    )
    #TODO: That's a little be cumbersome for judging using h3dd or gamma
    if h3dd_hout is not None:
        status = 'h3dd'
        use_h3dd = True
        use_gamma = False
    else:
        status = 'gamma'
        use_h3dd = False
        use_gamma = True

    time_str = (
        event_dict[status]['event_time']
        .replace(':', '_')
        .replace('-', '_')
        .replace('.', '_')
    )
    save_name = f'event_{event_i}_{time_str}.png'

    # Check if this event has already been processed
    if (fig_dir / save_name).exists():
        logging.info(f'Skipping event {event_i}: file already exists.')
        return  # Skip processing for this event
    # figure setting
    fig = plt.figure()
    map_proj = ccrs.PlateCarree()
    # tick_proj = ccrs.PlateCarree()

    # retrieving data
    if sac_parent_dir is not None:
        sac_dict = find_sac_data(
            event_time=event_dict['gamma']['event_time'],
            date=event_dict['gamma']['date'],
            event_lon=event_dict[status]['event_lon'],
            event_lat=event_dict[status]['event_lat'],
            event_depth=event_dict[status]['event_depth'],
            station=station,
            sac_parent_dir=sac_parent_dir,
            amplify_index=amplify_index,
            station_mask=station_mask,
            pretime=pretime,
            posttime=posttime
        )
        seis_map_ax = fig.add_axes([0.3, 0.5, 0.4, 0.8], projection=map_proj)
        seis_map_ax.set_title(
            f"Event_{event_i}: {event_dict[status]['event_time']}\nlat: {event_dict[status]['event_lat']}, lon: {event_dict[status]['event_lon']}, depth: {event_dict[status]['event_depth']} km"
        )
        seis_waveform_ax = fig.add_axes([0.82, 0.5, 0.8, 0.9])
    else:
        seis_map_ax = None
        seis_waveform_ax = None
        sac_dict = {}

    if h5_parent_dir is not None:
        das_map_ax = fig.add_axes([0.3, -0.15, 0.4, 0.55], projection=map_proj)
        das_waveform_ax = fig.add_axes([0.82, -0.15, 0.8, 0.55])  # , sharey=das_map_ax)
        if seis_map_ax is None:
            das_waveform_ax.set_title(
                f"Event_{event_i}: {event_dict[status]['event_time']}, lat: {event_dict[status]['event_lat']}, lon: {event_dict[status]['event_lon']}, depth: {event_dict[status]['event_depth']} km"
            )
    else:
        das_map_ax = None
        das_waveform_ax = None

    starttime = add_on_utc_time(df_event['time'].iloc[0], pretime)
    endtime = add_on_utc_time(df_event['time'].iloc[0], posttime)

    df_das_phasenet_picks = _find_phasenet_das_pick(
        starttime=starttime,
        endtime=endtime,
        df_picks=df_phasenet_picks,
        station_mask=station_mask
    )

    df_seis_phasenet_picks = _find_phasenet_seis_pick(
        starttime=starttime,
        endtime=endtime,
        df_picks=df_phasenet_picks,
        sac_dict=sac_dict,
        picking_check=False,
        asso_check=True,
        station_mask=station_mask
    )
    df_seis_gamma_picks, df_das_gamma_picks = find_gamma_pick(
        df_gamma_picks=df_event_picks,
        sac_dict=sac_dict,
        event_total_seconds=event_dict['gamma']['event_total_seconds'],
        station_mask=station_mask,
    )
    plot_waveform_check(
        sac_dict=sac_dict,
        starttime=starttime,
        end_time=endtime,
        df_seis_phasenet_picks=return_none_if_empty(df_seis_phasenet_picks),
        df_seis_gamma_picks=return_none_if_empty(df_seis_gamma_picks),
        df_das_phasenet_picks=return_none_if_empty(df_das_phasenet_picks),
        df_das_gamma_picks=return_none_if_empty(df_das_gamma_picks),
        h5_parent_dir=h5_parent_dir,
        das_ax=das_waveform_ax,
        seis_ax=seis_waveform_ax,
    )

    plot_asso_map(
        station=station,
        event_dict=event_dict,
        df_event_picks=df_event_picks,
        main_ax=seis_map_ax,
        sub_ax=das_map_ax,
        use_gamma=use_gamma,
        use_h3dd=use_h3dd,
        status=status,
        station_mask=station_mask,
        main_region=seis_region,
        sub_region=das_region,
        use_legend=use_legend
    )

    plt.tight_layout()
    plt.savefig(fig_dir / save_name, bbox_inches='tight', dpi=300)
    plt.close()


def parallel_plot_asso(
    phasenet_picks: Path,
    gamma_events: Path,
    gamma_picks: Path,
    station: Path,
    fig_dir: Path,
    sac_parent_dir: Path,
    h3dd_hout: Path | None = None,
    h5_parent_dir: Path | None = None,
    processes=10,
    amplify_index=5,
    seis_region=None,
    station_size=10,
    get_station=lambda x: str(x).split('.')[1],
    station_mask=station_mask,
    use_legend=False
):
    fig_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=fig_dir / 'aso.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    df_events = pd.read_csv(gamma_events)
    event_list = list(df_events['event_index'])
    df_phasenet_picks = _preprocess_phasenet_csv(
        phasenet_picks, get_station=get_station
    )
    if h3dd_hout is not None:
        partial_plot_asso = partial(
            plot_asso,
            h3dd_hout=h3dd_hout,
            sac_parent_dir=sac_parent_dir,
            h5_parent_dir=h5_parent_dir,
            amplify_index=amplify_index,
            seis_region=seis_region,
            #station_size=station_size,
            station_mask=station_mask,
            use_legend=use_legend
        )
    else:
        partial_plot_asso = partial(
            plot_asso,
            sac_parent_dir=sac_parent_dir,
            h5_parent_dir=h5_parent_dir,
            amplify_index=amplify_index,
            seis_region=seis_region,
            #station_size=station_size,
            station_mask=station_mask,
            use_legend=use_legend
        )        
    with mp.Pool(processes=processes) as pool:
        pool.starmap(
            partial_plot_asso,
            [
                (
                    df_phasenet_picks,
                    gamma_picks,
                    gamma_events,
                    station,
                    event_i,
                    fig_dir,
                )
                for event_i in event_list
            ],
        )


def _add_sub_map(
    geo_ax: GeoAxes,
    epi_lon: float,
    epi_lat: float,
    epi_size=10,
    epi_color='yellow',
    alpha=0.5,
):
    """## Add sub map on the right top of the main_ax
    Only thing needs to care about is the `geo_ax`, location is important.
    #TODO: Using a smart way to automatically decide the location of the sub map.
    """
    geo_ax.coastlines(resolution='10m', color='black', linewidth=1, alpha=alpha)
    geo_ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=alpha)
    geo_ax.add_feature(cfeature.LAND, alpha=alpha)

    # epicenter
    _add_epicenter(x=epi_lon, y=epi_lat, ax=geo_ax, size=epi_size, color=epi_color)

    geo_ax.patch.set_alpha(alpha)
    geo_ax.set_aspect('auto')


def _add_polarity(
    df_station, df_polarity_picks, ax, x_coord, y_coord, marker='.', markersize=5
):
    """## Adding polarity information on profile"""
    df_intersection = pd.merge(df_station, df_polarity_picks, on='station', how='inner')
    df_intersection['color'] = df_intersection['polarity'].map(polarity_color_select)
    ax.scatter(
        df_intersection[x_coord],
        df_intersection[y_coord],
        c=df_intersection['color'],
        marker=marker,
        s=markersize,
    )


def get_polarity_map(
    geo_ax,
    region: list,
    df_station: pd.DataFrame,
    df_polarity_picks: pd.DataFrame,
    epi_lon: float,
    epi_lat: float,
    title='Polarity map',
    station_mask=station_mask,
):
    """## The map for polarity"""
    df_seis_station = df_station[df_station['station'].map(station_mask)]
    df_das_station = df_station[~df_station['station'].map(station_mask)]
    # Cartopy setting
    get_mapview(region=region, ax=geo_ax, title=title)

    # plotting station on map
    plot_station(df_station=df_station, geo_ax=geo_ax)

    # adding epicenter
    _add_epicenter(x=epi_lon, y=epi_lat, ax=geo_ax)

    # adding polarity information on it
    _add_polarity(
        df_station=df_seis_station,
        df_polarity_picks=df_polarity_picks,
        ax=geo_ax,
        x_coord='longitude',
        y_coord='latitude',
        marker='^',
        markersize=40,
    )

    _add_polarity(
        df_station=df_das_station,
        df_polarity_picks=df_polarity_picks,
        ax=geo_ax,
        x_coord='longitude',
        y_coord='latitude',
    )
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    geo_ax.legend(markerscale=2, labelspacing=1.2, borderpad=1, fontsize=12)


def get_profile(
    df_station,
    ax,
    df_polarity_picks: pd.DataFrame,
    depth_lim: tuple | None = None,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    depth_axis='x',
    markersize=5,
    grid_alpha=0.7,
):
    """
    The df_polarity_picks is already filtered by the event index.
    """
    # ax.autoscale(tight=True)
    if depth_lim is None:
        depth_min = max(df_station['elevation'])
        depth_max = min(df_station['elevation'])
        depth_lim = (depth_min + 10, depth_max - 10)
    # axes setting
    # Scenario 1: depth_axis is in x cooridnate
    if depth_axis == 'x':
        x_coord = 'elevation'
        y_coord = 'latitude'
        ax.set_ylim(ylim)
        ax.set_xlim(depth_lim)
        ax.set_xlabel('Elevation (km)')
        ax.tick_params(axis='y', labelleft=False)
    elif depth_axis == 'y':
        x_coord = 'longitude'
        y_coord = 'elevation'
        ax.set_xlim(xlim)
        ax.set_ylim(depth_lim)
        ax.invert_yaxis()
        ax.set_ylabel('Elevation (km)')
        ax.tick_params(axis='x', labelbottom=False)

    ax.grid(True, alpha=grid_alpha)
    ax.scatter(df_station[x_coord], df_station[y_coord], c='silver', s=markersize)

    # plot station with polarity
    _add_polarity(
        df_station=df_station,
        df_polarity_picks=df_polarity_picks,
        ax=ax,
        x_coord=x_coord,
        y_coord=y_coord,
        markersize=markersize,
    )


def get_polarity_profiles(
    station: Path,
    gamma_events: Path,
    h3dd_events: Path,
    polarity_picks: Path,
    polarity_dout: Path,
    focal_catalog: Path,
    figure_dir: Path,
    event_index: int,
    region: list | None = None,
    depth_lim: tuple | None = None,
    focal_xlim=(-1.1, 1.1),
    focal_ylim=(-1.1, 1.1),
    focal_add_info=False,
    focal_only_das=False,
    savefig=True,
):
    """## Plot the polarity information on the station, typically for DAS.

    ### TODO: Optimization index
    This plot do not need the gamma as input, using h3dd row index to find the event.
    Main purpose is to plot the polarity of station, and find if this event has a
    solution of focal mechanism.

    ### Usage
    no need to think too much, just follow the hint and input your Path.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    # data preprocessing
    df_h3dd = find_gamma_h3dd(gamma_events, h3dd_events, event_index)
    df_station = pd.read_csv(station)
    if region is None:
        region = [
            df_station['longitude'].min(),
            df_station['longitude'].max(),
            df_station['latitude'].min(),
            df_station['latitude'].max(),
        ]

    df_polarity_picks = pd.read_csv(polarity_picks)
    df_polarity_picks = df_polarity_picks[
        df_polarity_picks['event_index'] == event_index
    ]
    df_polarity_picks = df_polarity_picks.rename(columns={'station_id': 'station'})

    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=focal_catalog, polarity_dout=polarity_dout
    )

    focal_dict = find_gafocal_polarity(
        df_gafocal=df_gafocal,
        df_hout=df_hout,
        polarity_dout=polarity_dout,
        event_index=0,  # TODO: this need to change
    )

    # The frame of plot
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [2, 1]},
    )

    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    geo_ax1 = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())

    # Main map
    get_polarity_map(
        geo_ax=geo_ax,
        df_station=df_station,
        df_polarity_picks=df_polarity_picks,
        region=region,
        title=f"{focal_dict['utc_time']}\nLat: {focal_dict['latitude']}, Lon: {focal_dict['longitude']}, Depth: {focal_dict['depth']}",
        epi_lon=df_h3dd['longitude'].iloc[0],
        epi_lat=df_h3dd['latitude'].iloc[0],
    )

    # Adding sub map top left
    _add_sub_map(
        geo_ax1,
        epi_lon=df_h3dd['longitude'].iloc[0],
        epi_lat=df_h3dd['latitude'].iloc[0],
    )  # 121.58, 23.86)
    plot_station(df_station, geo_ax1)

    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()

    get_profile(
        df_station=df_station,
        ax=axes[0, 1],
        df_polarity_picks=df_polarity_picks,
        ylim=ylim,
        depth_axis='x',
        depth_lim=depth_lim,
    )

    get_profile(
        df_station=df_station,
        ax=axes[1, 0],
        df_polarity_picks=df_polarity_picks,
        xlim=xlim,
        depth_axis='y',
        depth_lim=depth_lim,
    )

    get_beach(
        focal_dict=focal_dict,
        ax=axes[1, 1],
        xlim=focal_xlim,
        ylim=focal_ylim,
        only_das=focal_only_das,
        add_info=focal_add_info,
    )
    axes[0, 0].axis('off')

    # plt.tight_layout()
    if savefig:
        plt.savefig(
            figure_dir / 'pol_map.png', dpi=300, bbox_inches='tight'
        )  # using fig.savefig when we have several figs.


size_mapping = {
    (0, 1): 1,
    (1, 2): 1,
    (2, 3): 1,
    (3, 4): 1,
    (4, 5): 5,
    (5, 6): 7,
    (6, 7): 10,
    (7, 8): 15,
}
legend_elements = [
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='nan',
        markersize=1,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='0-1',
        markersize=4,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='1-2',
        markersize=5,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='2-3',
        markersize=6,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='3-4',
        markersize=10,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='4-5',
        markersize=15,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='5-6',
        markersize=20,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='6-7',
        markersize=25,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='7-8',
        markersize=30,
        markerfacecolor='gray',
    ),
]


# Function to map values to sizes
def _get_size(value, size_mapping=size_mapping):
    for range_tuple, size in size_mapping.items():
        if range_tuple[0] <= value < range_tuple[1]:
            return size
    return 1


def plot_mapview(
    station: Path,
    catalog: pd.DataFrame,
    catalog_filter_: dict | None = None,
    title='Event distribution',
    main_eq=(),
    h3dd_main_eq=(),
    main_eq_size=10,
    plot_station_name=True,
    plot_event_name=True,
    savefig=True,
    fig_dir: Path | None = None,
    plot_profile=True,
    h3dd_mode=False,
    prefix='',
):
    """
    catalog should at least contains the lon and lat.

    """
    # plot setting
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]},
    )
    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    # fig, geo_ax = plt.subplots(
    #     1, 1, figsize=(12, 14), subplot_kw={'projection': ccrs.PlateCarree()}
    # )

    # sub_ax = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())

    # base_map
    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]
    get_mapview(region=region, ax=geo_ax, title=title)
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].apply(_get_size)
    # station
    df_station = pd.read_csv(station)
    # print(
    #     'Geo Axes Patch Extent:',
    #     geo_ax.get_window_extent(renderer=fig.canvas.get_renderer()),
    # )
    plot_station(
        df_station=df_station,
        geo_ax=geo_ax,
        region=region,
        plot_station_name=plot_station_name,
        text_dist=0.005,
    )
    manual_elements = [
        Line2D(
            [0],
            [0],
            marker='.',
            color='w',
            label='DAS',
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k',
        ),
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
        Line2D(
            [0],
            [0],
            marker='*',
            color='w',
            label='CWA Epicenter',
            markersize=10,
            markerfacecolor='y',
            markeredgecolor='k',
        ),
        Line2D(
            [0],
            [0],
            marker='*',
            color='w',
            label='AutoQuake Epicenter',
            markersize=10,
            markerfacecolor='darkorange',
            markeredgecolor='k',
        ),
    ]
    # add all events
    event_num = len(catalog)
    if h3dd_mode:
        prefix = 'h3dd_'
    if 'event_type' in catalog.columns:
        # This part we want to visualize the distribution of event_type.
        color_map = {
            0: 'green',
            1: 'red',
            2: 'blue',
            3: 'green',
            4: 'orange',
            5: 'blue',
            6: 'red',
        }
        label_map = {
            0: 'seis 6P2S\n(only using seismometer)',
            1: 'seis 6P2S + DAS 15P',
            2: 'seis 6P2S',
            3: 'DAS 15P',
            4: 'Not reach the standard',
            5: 'GDMS Catalog',
            6: 'Associated event',
        }

        for event_type, group in catalog.groupby('event_type'):
            num = len(group)
            group = group.sort_values(by='size', ascending=True)
            geo_ax.scatter(
                group[f'{prefix}longitude'],
                group[f'{prefix}latitude'],
                s=group['size'],
                c=color_map[event_type],
                alpha=0.5,
                label=f'{label_map[event_type]}: {num}',
                rasterized=True,
                zorder=3 + abs(int(event_type)),
            )
            if plot_event_name:
                for _, row in group.iterrows():
                    geo_ax.text(
                        x=row[f'{prefix}longitude'],
                        y=row[f'{prefix}latitude'],
                        s=row['event_index'],
                    )
            if plot_profile:
                axes[0, 1].scatter(
                    group[f'{prefix}depth_km'],
                    group[f'{prefix}latitude'],
                    s=group['size'],
                    c=color_map[event_type],
                    alpha=0.5,
                    # label=label_map[event_type],
                    rasterized=True,
                    zorder=3 + abs(int(event_type) - 4),
                )
                _add_epicenter(
                    x=main_eq[2],
                    y=main_eq[1],
                    ax=axes[0, 1],
                    size=main_eq_size,
                    alpha=0.7,
                )
                _add_epicenter(
                    x=h3dd_main_eq[2],
                    y=h3dd_main_eq[1],
                    ax=axes[0, 1],
                    size=main_eq_size,
                    alpha=0.7,
                    color='darkorange',
                )
                axes[1, 0].scatter(
                    group[f'{prefix}longitude'],
                    group[f'{prefix}depth_km'],
                    s=group['size'],
                    c=color_map[event_type],
                    alpha=0.5,
                    # label=label_map[event_type],
                    rasterized=True,
                    zorder=3 + abs(int(event_type) - 4),
                )
                _add_epicenter(
                    x=main_eq[0],
                    y=main_eq[2],
                    ax=axes[1, 0],
                    size=main_eq_size,
                    alpha=0.7,
                )
                _add_epicenter(
                    x=h3dd_main_eq[0],
                    y=h3dd_main_eq[2],
                    ax=axes[1, 0],
                    size=main_eq_size,
                    alpha=0.7,
                    color='darkorange',
                )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label=f'{label_map[event_type]}: {num}',
                    markersize=3,
                    markerfacecolor=color_map[event_type],
                    markeredgecolor='k',
                    alpha=0.5,
                )
            )
    else:
        geo_ax.scatter(
            catalog[f'{prefix}longitude'],
            catalog[f'{prefix}latitude'],
            s=group['size'],
            c='r',
            alpha=0.5,
            label=f'Event num: {event_num}',
            zorder=3,
        )
        if plot_profile:
            axes[0, 1].scatter(
                catalog[f'{prefix}depth_km'],
                catalog[f'{prefix}latitude'],
                s=5,
                c='r',
                alpha=0.5,
                rasterized=True,
            )
            axes[1, 0].scatter(
                catalog[f'{prefix}longitude'],
                catalog[f'{prefix}depth_km'],
                s=5,
                c='r',
                alpha=0.5,
                rasterized=True,
            )
    # add main eq
    _add_epicenter(x=main_eq[0], y=main_eq[1], ax=geo_ax, size=main_eq_size, alpha=0.7)
    _add_epicenter(
        x=h3dd_main_eq[0],
        y=h3dd_main_eq[1],
        ax=geo_ax,
        size=main_eq_size,
        alpha=0.7,
        color='darkorange',
    )

    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    geo_ax.legend(
        handles=manual_elements,  # + legend_elements,
        loc='upper left',
        markerscale=2,
        labelspacing=1.5,
        borderpad=1,
        fontsize=10,
        framealpha=0.6,
    )
    # geo_ax.add_artist(legend)
    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()
    axes[0, 1].autoscale(tight=True)
    axes[0, 1].set_ylim(ylim)
    axes[0, 1].set_xlim([0, catalog['depth_km'].max() + 1])
    # axes[0, 1].set_xlim([0, 62])  # gdms
    axes[0, 1].set_xlabel('Depth (km)')
    axes[0, 1].set_ylabel('Latitude')

    axes[1, 0].autoscale(tight=True)
    axes[1, 0].set_xlim(xlim)
    axes[1, 0].set_ylim([0, catalog['depth_km'].max() + 1])
    # axes[1, 0].set_ylim([0, 62])  # gdms
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_ylabel('Depth (km)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 1].axis('off')
    axes[0, 0].axis('off')
    plt.tight_layout()
    if savefig and fig_dir is not None:
        plt.savefig(fig_dir / f'{title}.png', dpi=300, bbox_inches='tight')

def calc_kde(catalog: pd.DataFrame, grid_density=500):
    coordinates = catalog[['longitude', 'latitude']].values

    # Kernel Density Estimation
    kde = gaussian_kde(coordinates.T)
    x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()

    x, y = np.linspace(x_min, x_max, grid_density), np.linspace(y_min, y_max, grid_density)
    X, Y = np.meshgrid(x, y)
    density = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, density    

def plot_events_mapview(
    station: Path,
    catalogs: list[pd.DataFrame],
    catalogs_color: list[str],
    catalogs_name: list[str],
    gafocal_file: Path | pd.DataFrame | None = None,
    focal_size_mapping=focal_size_mapping,
    catalog_filter_: dict | None = None,
    title='catalog_comparison',
    eq_list=[],
    plot_station_name=True,
    savefig=True,
    fig_dir: Path | None = None,
    plot_profile=True,
    h3dd_mode=False,
    kde_mode=False,
    kde_min=0.0,
    legend_on_map=True,
    temporal=False,
    depth_set=[0, 60],
    station_mask=station_mask,
    profile_scatter_size=5,
    size_mapping=size_mapping,
    set_alpha=1
):
    """
    catalog should at least contains the lon and lat.
    Supporting catalog in list, allowing multiple catalogs plotting on map for comparison.
    """
    # plot setting
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]},
    )
    fig.patch.set_alpha(set_alpha)
    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())

    # base_map
    if catalog_filter_ is None: # using first catalog as judges
        region = [
            catalogs[0]['longitude'].min() - 0.2,
            catalogs[0]['longitude'].max() + 0.2,
            catalogs[0]['latitude'].min() - 0.2,
            catalogs[0]['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]
    print(region)
    get_mapview(region=region, ax=geo_ax, title=title)

    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])

    # station
    df_station = pd.read_csv(station)

    plot_station(
        df_station=df_station,
        geo_ax=geo_ax,
        region=region,
        plot_station_name=plot_station_name,
        text_dist=0.005,
        zorder=6,
        station_mask=station_mask
    )

    manual_elements = [
        Line2D(
            [0],
            [0],
            marker='^',
            color='w',
            label='Seismometer',
            markersize=4,
            markerfacecolor='c',
            markeredgecolor='k',
        )
    ]

    for catalog, color, name in zip(catalogs, catalogs_color, catalogs_name):
        # trim the catalog
        catalog = catalog_filter(
            catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
        )

        # get the size corresponding to magnitude
        catalog.loc[:, 'size'] = catalog['magnitude'].map(lambda x: _get_size(x, size_mapping=size_mapping))

        # add all events
        event_num = len(catalog)

        # kde calculation
        if kde_mode:
            X, Y, density = calc_kde(catalog)
            # geo_ax.imshow(density, extent=region, origin='lower', alpha=0.7, cmap=plt.cm.gist_earth_r, zorder=5, vmin=kde_min)
            levels = np.linspace(kde_min, density.max(), 10)
            contour_lines = geo_ax.contourf(X, Y, density, levels=levels, alpha=0.7, cmap="gist_earth_r", zorder=5)
            geo_ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
        
        # adding temporal feature
        if temporal:
            colormap = plt.cm.viridis
            norm = mcolors.Normalize(
                vmin=catalog['day'].min(),
                vmax=catalog['day'].max()
                )  # Normalize days to range [0, 1]
            for day, group in catalog.groupby('day'):
                logging.info(f'day {day}: {len(group)}')
                color = colormap(norm(day))
                geo_ax.scatter(
                    group['longitude'],
                    group['latitude'],
                    s=group['size'],
                    c=color,
                    alpha=0.5,
                    label=f'Day {day} Event num: {len(group)}',
                    zorder=3,
                )
                manual_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker='.',
                        color='w',
                        label=f'Day {day} Event num: {len(group)}',
                        markersize=7,
                        markerfacecolor=color,
                        markeredgecolor='none',
                    )
                )
                if plot_profile:
                    axes[0, 1].scatter(
                        group['depth_km'],
                        group['latitude'],
                        s=5,
                        c=color,
                        edgecolors='none',
                        alpha=0.5,
                        rasterized=True,
                    )
                    if len(eq_list) > 0:
                        for eq_tuple in eq_list:
                            _add_epicenter(
                                x=eq_tuple[2],
                                y=eq_tuple[1],
                                ax=axes[0, 1],
                                size=eq_tuple[4],
                                alpha=0.7,
                                color=eq_tuple[3],
                            )                                     
                    axes[1, 0].scatter(
                        group['longitude'],
                        group['depth_km'],
                        s=5,
                        c=color,
                        edgecolors='none',
                        alpha=0.5,
                        rasterized=True,
                    )
                    if len(eq_list) > 0:
                        for eq_tuple in eq_list:
                            _add_epicenter(
                                x=eq_tuple[0],
                                y=eq_tuple[2],
                                ax=axes[1, 0],
                                size=eq_tuple[4],
                                alpha=0.7,
                                color=eq_tuple[3],
                            )     
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            # Create an inset for the colorbar within geo_ax
            cbar_ax = inset_axes(
                geo_ax,  # Parent axis (geo_ax)
                width="3%",  # Width of the colorbar relative to geo_ax
                height="50%",  # Height of the colorbar relative to geo_ax
                loc="lower left",  # Location inside the geo_ax
                bbox_to_anchor=(0.025, 0.025, 0.85, 0.9),  # Adjust position within geo_ax
                bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
                borderpad=0,  # Padding around the inset
            )
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label(f"Day Since {catalog['month'].min()}/{catalog['day'].min()}")                                

        else:            
            geo_ax.scatter(
                catalog['longitude'],
                catalog['latitude'],
                s=catalog['size'],
                c=color,
                alpha=0.5,
                label=f'{name} Event num: {event_num}',
                zorder=6,
            )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='.',
                    color='w',
                    label=f'{name} Event',
                    markersize=7,
                    markerfacecolor=color,
                    markeredgecolor='none',
                )
            )
            if plot_profile:
                axes[0, 1].scatter(
                    catalog['depth_km'],
                    catalog['latitude'],
                    s=profile_scatter_size,
                    c=color,
                    edgecolors='none',
                    alpha=0.5,
                    rasterized=True,
                )
                if len(eq_list) > 0:
                    for eq_tuple in eq_list:
                        _add_epicenter(
                            x=eq_tuple[2],
                            y=eq_tuple[1],
                            ax=axes[0, 1],
                            size=eq_tuple[4],
                            alpha=0.7,
                            color=eq_tuple[3],
                        )                                     
                axes[1, 0].scatter(
                    catalog['longitude'],
                    catalog['depth_km'],
                    s=profile_scatter_size,
                    c=color,
                    edgecolors='none',
                    alpha=0.5,
                    rasterized=True,
                )
                if len(eq_list) > 0:
                    for eq_tuple in eq_list:
                        _add_epicenter(
                            x=eq_tuple[0],
                            y=eq_tuple[2],
                            ax=axes[1, 0],
                            size=eq_tuple[4],
                            alpha=0.7,
                            color=eq_tuple[3],
                        )            
          
    # add main eq
    if len(eq_list) > 0:
        for eq_tuple in eq_list:
            _add_epicenter(
                x=eq_tuple[0],
                y=eq_tuple[1],
                ax=geo_ax,
                size=eq_tuple[4],
                alpha=0.7,
                color=eq_tuple[3],
            )    
    
    # add focal mechanism on map
    if gafocal_file is not None:
        if isinstance(gafocal_file, Path):
            df_gafocal, _ = check_format(gafocal_file)
        elif isinstance(gafocal_file, pd.DataFrame):
            df_gafocal = gafocal_file
        else:
            raise ValueError('Please provide correct format of gafocal file')
        if temporal:
            df_gafocal['utctime'] = pd.to_datetime(df_gafocal['time'])
            df_gafocal['day'] = df_gafocal['utctime'].map(lambda x: x.day)
            colormap = plt.cm.viridis
            norm = mcolors.Normalize(
                vmin=df_gafocal['day'].min(),
                vmax=df_gafocal['day'].max()
                )  # Normalize days to range [0, 1]
            df_gafocal['color'] = df_gafocal['day'].map(lambda x: colormap(norm(x)))
            add_focal(ax=geo_ax, df=df_gafocal, focal_size_mapping=focal_size_mapping, temporal=temporal)            
            
        else:
            add_focal(ax=geo_ax, df=df_gafocal, focal_size_mapping=focal_size_mapping)
    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    if legend_on_map:
        geo_ax.legend(
            handles=manual_elements,  # + legend_elements,
            loc='lower right',
            markerscale=2,
            labelspacing=1.5,
            borderpad=1,
            fontsize=10,
            framealpha=0.6,
        )
    # geo_ax.add_artist(legend)
    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()
    if plot_profile:
        axes[0, 1].autoscale(tight=True)
        axes[0, 1].set_ylim(ylim)
        # axes[0, 1].set_xlim([0, catalog['depth_km'].max() + 1])
        axes[0, 1].set_xlim(depth_set)  # gdms
        axes[0, 1].set_xlabel('Depth (km)')
        axes[0, 1].set_ylabel('Latitude')

        axes[1, 0].autoscale(tight=True)
        axes[1, 0].set_xlim(xlim)
        # axes[1, 0].set_ylim([0, catalog['depth_km'].max() + 1])
        axes[1, 0].set_ylim(depth_set)  # gdms
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_ylabel('Depth (km)')
        axes[1, 0].set_xlabel('Longitude')
    else:
        fig.delaxes(axes[1, 0])
        fig.delaxes(axes[0, 1])
    fig.delaxes(axes[1, 1])
    axes[0, 0].axis('off')
    plt.tight_layout()
    if savefig and fig_dir is not None:
        plt.savefig(fig_dir / f'{title}.png', dpi=300, bbox_inches='tight')

def plot_mapview_temp(
    catalog: pd.DataFrame,
    station: Path | pd.DataFrame | None = None,     
    other_catalog: pd.DataFrame | None = None,
    catalog_filter_: dict | None = None,
    title='Event distribution',
    eq_list=[],
    plot_eq_name=False,
    plot_station_name=True,
    plot_event_name=False,
    # station_mask=station_mask,
    savefig=True,
    fig_dir: Path | None = None,
    plot_profile=True,
    h3dd_mode=False,
    prefix='',
    savename='',
    temporal=False,
    plot_profile_line=False,
    start_point=(121.08, 23.65),
    plot_vel=False,
    vel_df: pd.DataFrame | None = None, # we should write a function to check the type
    legend_on_map=False,
    fig_alpha=False,
    fault_lw=4,
    figsize=(23, 27),
    size_mapping=size_mapping
):
    """
    catalog should at least contains the lon and lat.

    """
    # plot setting
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]},
    )
    if fig_alpha:
        fig.patch.set_alpha(0)
        pass
    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    # fig, geo_ax = plt.subplots(
    #     1, 1, figsize=(12, 14), subplot_kw={'projection': ccrs.PlateCarree()}
    # )

    # sub_ax = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())

    # base_map
    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]
    get_mapview(region=region, ax=geo_ax, title=title, fault_lw=fault_lw)
    if plot_vel and vel_df is not None:
        add_vel_view(
            ax=geo_ax,
            vel_df=vel_df,
            region=region
        )
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].apply(
        lambda x:_get_size(x, size_mapping)
        )
    # station
    # print(
    #     'Geo Axes Patch Extent:',
    #     geo_ax.get_window_extent(renderer=fig.canvas.get_renderer()),
    # )
    if station is not None:
        if isinstance(station, Path):
            df_station = pd.read_csv(station)
        elif isinstance(station, pd.DataFrame):
            df_station = station
        plot_station(
            df_station=df_station,
            geo_ax=geo_ax,
            region=region,
            plot_station_name=plot_station_name,
            text_dist=0.005,
        )
        manual_elements = [
            Line2D(
                [0],
                [0],
                marker='^',
                color='w',
                label='Seismometer',
                markersize=3,
                markerfacecolor='c',
                markeredgecolor='k',
            ),
        ]
    else:
        manual_elements=[]
    # add all events
    # event_num = len(catalog)
    # manual_elements.append(
    #     Line2D(
    #         [0],
    #         [0],
    #         marker='o',
    #         color='w',
    #         label=f'Event num: {event_num}',
    #         markersize=1,
    #         markerfacecolor='k',
    #         markeredgecolor='k',
    #     )
    # )       
    if h3dd_mode:
        prefix = 'h3dd_'
    if 'event_type' in catalog.columns:
        # This part we want to visualize the distribution of event_type.
        color_map = {
            0: 'k',
            1: 'green',
            2: 'purple',
            3: 'gray',
            # 4: 'orange',
            # 5: 'blue',
            # 6: 'red',
        }
        label_map = {
            0: 'Period 1(Main shock - 2024-04-08)',
            1: 'Period 2(2024-04-08 - 2024-04-22)',
            2: 'Period 3(2024-04-22 - 2024-04-29)',
            3: 'Huang and Wang (2022) Relocated events',
            # 4: 'GDMS Catalog',
            # 5: 'GDMS Catalog',
            # 6: 'Associated event',
        }

        for event_type, group in catalog.groupby('event_type'):
            num = len(group)
            group = group.sort_values(by='size', ascending=True)
            geo_ax.scatter(
                group[f'{prefix}longitude'],
                group[f'{prefix}latitude'],
                s=group['size'],
                color=color_map[event_type],
                alpha=0.5,
                label=f'{label_map[event_type]}: {num}',
                rasterized=True,
                zorder=3 + abs(int(event_type) - 4),
            )
            if plot_event_name:
                for _, row in group.iterrows():
                    geo_ax.text(
                        x=row[f'{prefix}longitude'],
                        y=row[f'{prefix}latitude'],
                        s=row['event_index'],
                    )
            if plot_profile:
                axes[0, 1].scatter(
                    group[f'{prefix}depth_km'],
                    group[f'{prefix}latitude'],
                    s=group['size'],
                    c=color_map[event_type],
                    alpha=0.5,
                    # label=label_map[event_type],
                    rasterized=True,
                    zorder=3 + abs(int(event_type) - 4),
                )
                for eq in eq_list:
                    _add_epicenter(
                        x=eq[2],
                        y=eq[1],
                        ax=axes[0, 1],
                        size=eq[3],
                        alpha=0.7,
                        color=eq[4],
                    )
                axes[1, 0].scatter(
                    group[f'{prefix}longitude'],
                    group[f'{prefix}depth_km'],
                    s=group['size'],
                    c=color_map[event_type],
                    alpha=0.5,
                    # label=label_map[event_type],
                    rasterized=True,
                    zorder=3 + abs(int(event_type) - 4),
                )
                for eq in eq_list:
                    _add_epicenter(
                        x=eq[0],
                        y=eq[2],
                        ax=axes[1, 0],
                        size=eq[3],
                        alpha=0.7,
                        color=eq[4],
                    )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label=f'{label_map[event_type]}: {num}',
                    markersize=3,
                    markerfacecolor=color_map[event_type],
                    markeredgecolor='k',
                    alpha=0.8,
                )
            )
        if other_catalog is not None:
            other_catalog.loc[:, 'size'] = other_catalog['magnitude'].apply(_get_size)
            geo_ax.scatter(
                other_catalog['longitude'],
                other_catalog['latitude'],
                s=other_catalog['size'],
                color='gray',
                edgecolors='none',
                alpha=0.5
            )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label='Huang and Wang (2022) Relocated events',
                    alpha=0.8,
                    markersize=3,
                    markerfacecolor='gray',
                    markeredgecolor='k',
                )
            )  

    else:
        if other_catalog is not None:
            other_catalog.loc[:, 'size'] = other_catalog['magnitude'].apply(_get_size)
            geo_ax.scatter(
                other_catalog['longitude'],
                other_catalog['latitude'],
                s=other_catalog['size'],
                color='k',
                edgecolors='none',
                alpha=0.3
            )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label='Huang and Wang (2022) Relocated events',
                    alpha=0.5,
                    markersize=1,
                    markerfacecolor='k',
                    markeredgecolor='k',
                )
            )                
        if temporal:
            colormap = plt.cm.viridis
            norm = mcolors.Normalize(
                vmin=catalog['day'].min(),
                vmax=catalog['day'].max()
                )  # Normalize days to range [0, 1]
            for day, group in catalog.groupby('day'):
                logging.info(f'day {day}: {len(group)}')
                color = colormap(norm(day))
                geo_ax.scatter(
                    group[f'{prefix}longitude'],
                    group[f'{prefix}latitude'],
                    s=group['size'],
                    color=color,
                    alpha=0.5,
                    label=f'Day {day} event num: {len(group)}',
                    zorder=3,
                )
                if plot_profile:
                    axes[0, 1].scatter(
                        group[f'{prefix}depth_km'],
                        group[f'{prefix}latitude'],
                        s=group['size'],
                        c=color,
                        alpha=0.5,
                        rasterized=True,
                    )
                    axes[1, 0].scatter(
                        group[f'{prefix}longitude'],
                        group[f'{prefix}depth_km'],
                        s=group['size'],
                        c=color,
                        alpha=0.5,
                        rasterized=True,
                    )
                # Add a colorbar to indicate progression
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            # Create an inset for the colorbar within geo_ax
            cbar_ax = inset_axes(
                geo_ax,  # Parent axis (geo_ax)
                width="3%",  # Width of the colorbar relative to geo_ax
                height="50%",  # Height of the colorbar relative to geo_ax
                loc="lower left",  # Location inside the geo_ax
                bbox_to_anchor=(0.025, 0.025, 0.85, 0.9),  # Adjust position within geo_ax
                bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
                borderpad=0,  # Padding around the inset
            )
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label(f"Day Since {catalog['month'].min()}/{catalog['day'].min()}")
        else:
            geo_ax.scatter(
                catalog[f'{prefix}longitude'],
                catalog[f'{prefix}latitude'],
                s=catalog['size'],
                color=catalog['color'],#'k',
                alpha=0.5,
                zorder=3,
            )         
            if plot_profile:
                axes[0, 1].scatter(
                    catalog[f'{prefix}depth_km'],
                    catalog[f'{prefix}latitude'],
                    s=catalog['size'],
                    c=catalog['color'],
                    alpha=0.5,
                    rasterized=True,
                )
                axes[1, 0].scatter(
                    catalog[f'{prefix}longitude'],
                    catalog[f'{prefix}depth_km'],
                    s=catalog['size'],
                    c=catalog['color'],
                    alpha=0.5,
                    rasterized=True,
                    )
    # profile
    if plot_profile_line:
        letters = string.ascii_uppercase
        #for i, (dis, letter) in enumerate(zip(np.linspace(20, 95, 16), letters[0:16])):
        for i, (dis, letter) in enumerate(zip(np.linspace(25, 85, 13), letters[1:14])):
            geod = Geod(ellps='WGS84')
            next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], 20, dis*1000)
            text_lon, text_lat, _ = geod.fwd(start_point[0], start_point[1], 20, dis*1000)
            add_profile_line(geo_ax=geo_ax, mode='start', extend_km=80, init_point=(next_lon, next_lat), azimuth=20 + 90, alpha=0.7, c='k')   
            geo_ax.text(x=text_lon -0.03, y=text_lat, s=f"{letter}-{letter}'")
        start_point = (121.485, 24.667)
        #for i, (dis, letter) in enumerate(zip(np.linspace(25, 70, 10), letters[16:26])):
        for i, (dis, letter) in enumerate(zip(np.linspace(35, 45, 3), letters[18:21])):
            geod = Geod(ellps='WGS84')
            next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], 110, dis*1000)
            text_lon, text_lat, _ = geod.fwd(start_point[0], start_point[1], 110, dis*1000)
            add_profile_line(geo_ax=geo_ax, mode='start', extend_km=120, init_point=(next_lon, next_lat), azimuth=110 + 90, alpha=0.7, c='k')   
            geo_ax.text(x=text_lon -0.02, y=text_lat, s=f"{letter}-{letter}'")                    

    # add main eq
    for eq in eq_list:
        _add_epicenter(
            x=eq[0], y=eq[1], ax=geo_ax, size=eq[3], alpha=0.7, color=eq[4]
        )
        if plot_eq_name:
            geo_ax.text(
                x=eq[0]+0.01, y=eq[1]+0.01, s=eq[5], color='k'
            )
        if plot_profile:
            _add_epicenter(
                x=eq[2], y=eq[1], ax=axes[0, 1], size=eq[3], alpha=0.7, color=eq[4]
            )
            _add_epicenter(
                x=eq[0], y=eq[2], ax=axes[1, 0], size=eq[3], alpha=0.7, color=eq[4]
            )                
        # manual_elements.append(
        #     Line2D(
        #         [0],
        #         [0],
        #         marker='*',
        #         color='w',
        #         label=eq[3],
        #         markersize=10,
        #         markerfacecolor=eq[4],
        #         markeredgecolor='k',
        #     )
        # )
    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    if legend_on_map:
        geo_ax.legend(
            handles=manual_elements,  # + legend_elements,
            loc='lower right',
            markerscale=2,
            labelspacing=1.5,
            borderpad=1,
            fontsize=10,
            framealpha=0.6,
            alignment='right'
        )
    # geo_ax.add_artist(legend)
    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()

    if plot_profile:
        axes[0, 1].autoscale(tight=True)
        axes[0, 1].set_ylim(ylim)
        axes[0, 1].set_xlim([0, catalog['depth_km'].max() + 1])
        # axes[0, 1].set_xlim([0, 62])  # gdms
        axes[0, 1].set_xlabel('Depth (km)')
        axes[0, 1].set_ylabel('Latitude')

        axes[1, 0].autoscale(tight=True)
        axes[1, 0].set_xlim(xlim)
        axes[1, 0].set_ylim([0, catalog['depth_km'].max() + 1])
        # axes[1, 0].set_ylim([0, 62])  # gdms
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_ylabel('Depth (km)')
        axes[1, 0].set_xlabel('Longitude')
    else:
        # axes[0, 1].axis('off')
        # axes[1, 0].axis('off')
        fig.delaxes(axes[0, 1])
        fig.delaxes(axes[1, 0])

    fig.delaxes(axes[1, 1])
    axes[0, 0].axis('off')
    # plt.tight_layout()
    if savefig and fig_dir is not None:
        plt.savefig(fig_dir / f'{savename}.png', dpi=300, bbox_inches='tight')


# plot single beachball check
def plot_only_beach(
    gafocal_txt: Path,
    polarity_dout: Path,
    polarity_picks: Path,
    output_dir: Path,
    hout_file: Path | None = None,
    gamma_events: Path | None = None,
    gamma_detailed: Path | None = None,
    get_waveform=False,
    sac_parent_dir: Path | None = None,
    h5_parent_dir: Path | None = None,
    processes=10,
    event_list=[],
):
    """## plot beachball only"""
    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=gafocal_txt, polarity_dout=polarity_dout, hout_file=hout_file
    )
    # df_gafocal, _ = check_format(catalog=gafocal_txt)
    if len(event_list) == 0:
        event_list = range(len(df_gafocal))
    with mp.Pool(processes=processes) as pool:
        pool.starmap(
            get_single_beach,
            [
                (
                    df_gafocal,
                    polarity_dout,
                    polarity_picks,
                    df_hout,
                    event_index,
                    output_dir,
                    # gamma_events,
                    # gamma_detailed,
                    get_waveform,
                    sac_parent_dir,
                    h5_parent_dir,
                )
                for event_index in event_list
            ],
        )


# plot single beachball check
def _equip_filter_setting(x):
    return str(x)[1].isalpha()
def _event_filter_setting(x):
    return str(x)[0].isdigit()

def plot_std_beach(
    gafocal_txt: Path,
    polarity_dout: Path,
    output_dir: Path,
    hout_file: Path | None = None,
    name='GAFocal',
    processes=10,
    event_list=[],
    # equip_filter=_equip_filter_setting,
    # event_filter=_event_filter_setting,
):
    """## plot beachball only"""
    #TODO: Haven't find a way to prevent pickle function error.
    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=gafocal_txt, polarity_dout=polarity_dout, hout_file=hout_file
    )
    if len(event_list) == 0:
        event_list = range(len(df_gafocal))
    with mp.Pool(processes=processes) as pool:
        pool.starmap(
            get_only_beach,
            [
                (df_gafocal, polarity_dout, df_hout, event_index, output_dir, name)
                for event_index in event_list
            ],
        )


# plot beachball with tracer
def plot_tracer_beach(
    gafocal_txt: Path,
    polarity_dout: Path,
    output_dir: Path,
    tt_table: Path,
    extend_pol_picks: Path,
    station: Path,
    gamma_events: Path | None = None,
    gamma_detailed: Path | None = None,
    get_waveform=False,
    sac_parent_dir: Path | None = None,
    h5_parent_dir: Path | None = None,
    processes=10,
    input_tracer=True,
    event_list=[],
    das_size=2.0,
    das_edge_size=0.0,
    xlim=(-1.1, 1.1),
    ylim=(-1.1, 1.1),
):
    """## plot beachball only
    2024/12/05
    """
    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=gafocal_txt, polarity_dout=polarity_dout
    )
    if len(event_list) == 0:
        event_list = range(len(df_gafocal))

    with mp.Pool(processes=processes) as pool:
        pool.starmap(
            get_tracer_beach,
            [
                (
                    df_gafocal,
                    polarity_dout,
                    df_hout,
                    event_index,
                    output_dir,
                    gamma_events,
                    gamma_detailed,
                    get_waveform,
                    sac_parent_dir,
                    h5_parent_dir,
                    input_tracer,
                    tt_table,
                    extend_pol_picks,
                    station,
                    das_size,
                    das_edge_size,
                    xlim,
                    ylim,
                )
                for event_index in event_list
            ],
        )


# plot pol map according to gafocal result
def plot_polarity_profiles(
    station: Path,
    gamma_events: Path,
    h3dd_events: Path,
    polarity_picks: Path,
    polarity_dout: Path,
    focal_catalog: Path,
    figure_dir: Path,
    event_index: int,
    region: list | None = None,
    depth_lim: tuple | None = None,
    focal_xlim=(-1.1, 1.1),
    focal_ylim=(-1.1, 1.1),
    focal_add_info=False,
    focal_only_das=False,
    savefig=True,
):
    """## Plot the polarity information on the station, typically for DAS.

    Args:
        Event_index: event_index here is the row num of gafocal.
        gamma_events: reorder one due to contain the column of h3dd_index.
    ### Usage
    no need to think too much, just follow the hint and input your Path.
    #NOTE: 2024/12/2 using focal row to plot polarity_map.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)

    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=focal_catalog, polarity_dout=polarity_dout
    )

    focal_dict, h3dd_index = find_gafocal_polarity(
        df_gafocal=df_gafocal,
        df_hout=df_hout,
        polarity_dout=polarity_dout,
        event_index=event_index,
        get_h3dd_index=True,
    )
    # available_sta = list(focal_dict['sta_info'].keys())
    gamma_index = find_gamma_index(gamma_events=gamma_events, h3dd_index=h3dd_index)
    # data preprocessing
    df_h3dd, _ = check_format(catalog=h3dd_events)
    df_h3dd = df_h3dd.loc[h3dd_index]
    df_station = pd.read_csv(station)
    if region is None:
        region = [
            df_station['longitude'].min(),
            df_station['longitude'].max(),
            df_station['latitude'].min(),
            df_station['latitude'].max(),
        ]

    df_polarity_picks = pd.read_csv(polarity_picks)
    df_polarity_picks = df_polarity_picks[
        df_polarity_picks['event_index'] == gamma_index
    ]
    df_polarity_picks = df_polarity_picks.rename(columns={'station_id': 'station'})

    # The frame of plot
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [2, 1]},
    )

    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    geo_ax1 = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())

    # Main map
    get_polarity_map(
        geo_ax=geo_ax,
        df_station=df_station,
        df_polarity_picks=df_polarity_picks,
        region=region,
        title=f"{focal_dict['utc_time']}\nLat: {focal_dict['latitude']}, Lon: {focal_dict['longitude']}, Depth: {focal_dict['depth']}",
        epi_lon=df_h3dd['longitude'].iloc[0],
        epi_lat=df_h3dd['latitude'].iloc[0],
    )

    # Adding sub map top left
    _add_sub_map(
        geo_ax1,
        epi_lon=df_h3dd['longitude'].iloc[0],
        epi_lat=df_h3dd['latitude'].iloc[0],
    )  # 121.58, 23.86)
    plot_station(df_station, geo_ax1)

    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()

    get_profile(
        df_station=df_station,
        ax=axes[0, 1],
        df_polarity_picks=df_polarity_picks,
        ylim=ylim,
        depth_axis='x',
        depth_lim=depth_lim,
    )

    get_profile(
        df_station=df_station,
        ax=axes[1, 0],
        df_polarity_picks=df_polarity_picks,
        xlim=xlim,
        depth_axis='y',
        depth_lim=depth_lim,
    )

    get_beach(
        focal_dict=focal_dict,
        ax=axes[1, 1],
        xlim=focal_xlim,
        ylim=focal_ylim,
        only_das=focal_only_das,
        add_info=focal_add_info,
    )
    axes[0, 0].axis('off')

    # plt.tight_layout()
    if savefig:
        plt.savefig(
            figure_dir
            / f"Event_{event_index}_{focal_dict['utc_time'].replace('-','_').replace(':', '_')}_map.png",
            dpi=300,
            bbox_inches='tight',
        )  # using fig.savefig when we have several figs.


def get_mag_distribution(df, name, start_date, end_date, fig_dir):
    color_map = {
        0: 'green',
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'orange',
        5: 'blue',
        6: 'red',
    }
    label_map = {
        0: 'seis 6P2S\n(only using seismometer)',
        1: 'seis 6P2S + DAS 15P',
        2: 'seis 6P2S',
        3: 'DAS 15P',
        4: 'Not reach the standard',
        5: 'GDMS Catalog',
        6: 'Seis + DAS',
    }
    df['datetime'] = pd.to_datetime(df['time'])
    df = df[df['magnitude'] > 0]
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(df['magnitude'], bins=16, range=(0, 8), edgecolor='k')
    for i in range(len(n)):
        if n[i] > 0:  # Only plot text if the bin's number is greater than 0
            plt.text(
                bins[i] + (bins[i + 1] - bins[i]) / 2,
                n[i],
                str(int(n[i])),
                ha='center',
                va='bottom',
            )
    plt.legend([f'Total events: {int(n.sum())}'], loc='upper right')
    plt.title('Magnitude Distribution', fontsize=15)
    plt.xlabel('Magnitude', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    # Add minor ticks with interval equal to 0.5
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='x', which='minor', length=5)
    ax.set_xlim(left=0)
    # plt.grid(True)
    plt.savefig(
        fig_dir / f'{name}_mag_histogram.png',
        dpi=300,
        bbox_inches='tight',
    )

    # Show correlation with scatter plot
    plt.figure(figsize=(15, 10))
    for event_type, group in df.groupby('event_type'):
        plt.scatter(
            group['datetime'],
            group['magnitude'],
            alpha=0.7,
            color=color_map[event_type],
            s=5,
            label=f'{label_map[event_type]}: {len(group)}',
        )
    plt.title('Magnitude vs Date', fontsize=25, pad=10)
    plt.xlabel('Date', fontsize=25, labelpad=10)
    plt.ylabel('Magnitude', fontsize=25, labelpad=10)
    plt.grid(True)
    plt.legend(loc='upper right', labelspacing=1.5, fontsize=15)
    # Customize x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', which='major', labelsize=12, labelrotation=45)
    ax.tick_params(
        axis='x', which='minor', labelsize=12, length=3
    )  # Adjust minor ticks length
    ax.tick_params(axis='y', which='major', labelsize=20)
    # Adjust offset of x-tick labels
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    # Set x-limits
    ax.set_xlim([start_date, end_date])
    ax.set_ylim(0, 8)
    plt.savefig(
        fig_dir / f'{name}_mag_daily.png',
        dpi=300,
        bbox_inches='tight',
    )


def get_mags_dist(
        df_list: list[pd.DataFrame],
        start_date: str,
        end_date: str,
        fig_dir,
        color_list: list,
        name_list: list,
        save_name='',
        figsize=(15, 10)
        ):
    """
    Date format: '2024-04-01'
    """
    color_map = {
        0: 'green',
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'orange',
        5: 'blue',
        6: 'red',
    }
    label_map = {
        0: 'seis 6P2S\n(only using seismometer)',
        1: 'seis 6P2S + DAS 15P',
        2: 'seis 6P2S',
        3: 'DAS 15P',
        4: 'Not reach the standard',
        5: 'GDMS Catalog',
        6: 'Seis + DAS',
    }
    plt.figure(figsize=figsize)
    for df, color, name in zip(df_list, color_list, name_list):
        df['datetime'] = pd.to_datetime(df['time'])
        df = df[df['magnitude'] > 0]
        # Show correlation with scatter plot
        if 'event_type' in df.columns:
            for event_type, group in df.groupby('event_type'):
                plt.scatter(
                    group['datetime'],
                    group['magnitude'],
                    alpha=0.7,
                    color=color_map[event_type],
                    s=5,
                    label=f'{label_map[event_type]}: {len(group)}',
                )
        else:
            plt.scatter(
                df['datetime'],
                df['magnitude'],
                alpha=0.5 if name == 'GDMS' else 1,
                color=color,
                s=5,
                label=f'{name} event num: {len(df)}',
            )            
    plt.title('Magnitude temporal variation', fontsize=25, pad=10)
    plt.xlabel('Date', fontsize=25, labelpad=10)
    plt.ylabel('Magnitude', fontsize=25, labelpad=10)
    plt.grid(True)
    # plt.legend(loc='upper right', markerscale=2, labelspacing=1.5, fontsize=15)
    # Customize x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', which='major', labelsize=12, labelrotation=45)
    ax.tick_params(
        axis='x', which='minor', labelsize=12, length=3
    )  # Adjust minor ticks length
    ax.tick_params(axis='y', which='major', labelsize=20)
    # Adjust offset of x-tick labels
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    # Set x-limits
    ax.set_xlim([pd.Timestamp(start_date), pd.Timestamp(end_date)])
    ax.set_ylim(0, 8)
    plt.savefig(
        fig_dir / f'mag_dist_{save_name}.png',
        dpi=300,
        bbox_inches='tight',
    )


def gamma_h3dd_combine(
        gamma_events: Path,
        h3dd_hout: Path | None = None,
        mag_events: Path | None = None,
        savefig=False,
        output: Path | None = None
        ):
    """Combine the gamma and h3dd catalog through hout or mag_events."""
    # reorder
    df_events = pd.read_csv(gamma_events)
    df_events['sort_time'] = pd.to_datetime(df_events['time'])
    df_events = df_events.sort_values(by='sort_time', ignore_index=True)
    df_events.drop(columns=['sort_time'], inplace=True)
    df_events.drop(columns=['magnitude'], inplace=True) # because it would be 999
    df_events['h3dd_event_index'] = df_events.index

    if h3dd_hout is None and mag_events is None:
        raise ValueError('Please provide h3dd_hout or mag_events to conduct combination')
    
    # adding h3dd
    if h3dd_hout is not None:
        df_h3dd, _ = check_format(h3dd_hout)
        df_h3dd = df_h3dd.add_prefix('h3dd_')
        df_h3dd['h3dd_event_index'] = df_h3dd.index

        # merge
        df = pd.merge(df_events, df_h3dd, on='h3dd_event_index', how='left')

    # adding mag
    if mag_events is not None:
        df_mag = pd.read_csv(mag_events)
        df_mag = df_mag[['time', 'longitude', 'latitude', 'depth_km', 'h3dd_event_index', 'magnitude']]
        rename_cols = ['time', 'longitude', 'latitude', 'depth_km']
        df_mag.rename(columns={x : f'h3dd_{x}' for x in rename_cols}, inplace=True)

        # merge
        df = pd.merge(df_events, df_mag, on='h3dd_event_index', how='left')

    if savefig and output is not None:
        df.to_csv(output, index=False)
    return df

def calculate_fault_azimuth(fault_start, fault_end):
    """
    計算斷層的方向角（以正北為基準）。
    """
    geod = Geod(ellps="WGS84")
    lon1, lat1 = fault_start
    lon2, lat2 = fault_end
    fault_azimuth, _, _ = geod.inv(lon1, lat1, lon2, lat2)
    return float(fault_azimuth)

def project_events_to_profile(df_catalog: pd.DataFrame, length=10, width=80, depth_range=(0, 60),
                              center: tuple | None = None, azimuth: float | None = None,
                              fault_start: tuple | None = None,
                              fault_end: tuple | None = None,
                              temporal=False,
                              flex_temporal=False,
                              focal=False
                              ):
    """
    Return the df in the profile range.
    """
    geod = Geod(ellps="WGS84")

    # 計算斷層中點
    if azimuth is None and fault_start is not None and fault_end is not None:
        lon1, lat1 = fault_start
        lon2, lat2 = fault_end
        azimuth = calculate_fault_azimuth(fault_start, fault_end)
        mid_lon, mid_lat = (lon1 + lon2) / 2, (lat1 + lat2) / 2
    elif azimuth is not None and center is not None:
        mid_lon, mid_lat = center
    # 投影地震事件到剖面方向
    results = []
    #TODO: I think optimizing it is possible.
    for event in df_catalog.itertuples():
        event_lon, event_lat, event_depth, color = event.longitude, event.latitude, event.depth_km, event.color
        if temporal:
            day = event.day
        if flex_temporal:
            group = event.time_group
        # 計算事件的方位角和距離
        event_az, _, event_dist = geod.inv(mid_lon, mid_lat, event_lon, event_lat)

        # 計算事件在剖面方向與垂直方向的距離分量
        relative_angle = np.radians(event_az - azimuth)  # 方位角差值
        y = event_dist * np.cos(relative_angle) / 1000  # 剖面方向距離（公里）
        x = event_dist * np.sin(relative_angle) / 1000  # 垂直剖面距離（公里）

        # 篩選符合剖面範圍的事件
        if (
            abs(y) <= length / 2 and  # 剖面長度範圍
            abs(x) <= width / 2 and  # 剖面寬度範圍
            depth_range[0] <= event_depth <= depth_range[1]  # 深度範圍
        ):
            if temporal:
                results.append((day, x+width/2, y, event_lon, event_lat, event_depth))
            elif flex_temporal:
                if focal:
                    results.append((group, x+width/2, y, event_lon, event_lat, event_depth, event.strike, event.dip, event.rake, event.magnitude))
                else:
                    results.append((group, x+width/2, y, event_lon, event_lat, event_depth))
            elif focal:
                results.append((x+width/2, y, event_lon, event_lat, event_depth, event.strike, event.dip, event.rake, event.magnitude, event.quality_index))
            else:
                results.append((x+width/2, y, event_lon, event_lat, event_depth, color))
    # 返回結果 DataFrame
    if temporal:
        results_df = pd.DataFrame(results, columns=['day', 'x', 'y', 'longitude', 'latitude', 'depth'])
    elif flex_temporal:
        if focal:
            results_df = pd.DataFrame(results, columns=['group', 'x', 'y', 'longitude', 'latitude', 'depth', 'strike', 'dip', 'rake', 'magnitude'])
        else:
            results_df = pd.DataFrame(results, columns=['group', 'x', 'y', 'longitude', 'latitude', 'depth'])
    elif focal:
        results_df = pd.DataFrame(results, columns=['x', 'y', 'longitude', 'latitude', 'depth', 'strike', 'dip', 'rake', 'magnitude', 'quality_index'])
    else:
        results_df = pd.DataFrame(results, columns=['x', 'y', 'longitude', 'latitude', 'depth', 'color'])
    return results_df

def _center_extend(
    geo_ax: Axes,
    lon,
    lat,
    length,
    width,
    azimuth,
    **kwargs
):
    half_length = length / 2
    half_width = width / 2

    # Step 1: Extend the midpoint to get the front and back center points
    geod = Geod(ellps="WGS84")
    front_center_lon, front_center_lat, _ = geod.fwd(lon, lat, azimuth, half_length * 1000)
    back_center_lon, back_center_lat, _ = geod.fwd(lon, lat, azimuth, -half_length * 1000)

    # Step 2: From the front and back center points, extend to the left and right to get the corners
    corners = []

    for center_lon, center_lat in [(front_center_lon, front_center_lat), (back_center_lon, back_center_lat)]:
        # Left side (-90 degrees from fault direction)
        left_lon, left_lat, _ = geod.fwd(center_lon, center_lat, azimuth - 90, half_width * 1000)
        # Right side (+90 degrees from fault direction)
        right_lon, right_lat, _ = geod.fwd(center_lon, center_lat, azimuth + 90, half_width * 1000)
        corners.append((left_lon, left_lat))
        corners.append((right_lon, right_lat))

    # Step 3: Reorder the corners to form a closed rectangle (clockwise)
    corners = [corners[0], corners[2], corners[3], corners[1], corners[0]]

    # Plot the rectangle on the GeoAxes
    lons, lats = zip(*corners)
    geo_ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)    
    

def _start_extend(
    geo_ax: Axes,
    lon,
    lat,
    length,
    width,
    azimuth,
    **kwargs        
):
    """
    Args:
        azimuth (float): azimuth here is E-W orientation, so we need to + 90 degrees to extend the width.
    """
    half_length = length / 2

    # Step 1: Extend the midpoint to get the front and back center points
    geod = Geod(ellps="WGS84")
    front_center_lon, front_center_lat, _ = geod.fwd(lon, lat, azimuth - 90, half_length * 1000)
    back_center_lon, back_center_lat, _ = geod.fwd(lon, lat, azimuth + 90, -half_length * 1000)

    # Step 2: From the front and back center points, extend to the left and right to get the corners
    corners = [(back_center_lon, back_center_lat), (front_center_lon, front_center_lat)]

    for start_lon, start_lat in [(front_center_lon, front_center_lat), (back_center_lon, back_center_lat)]:
        end_lon, end_lat, _ = geod.fwd(start_lon, start_lat, azimuth, width * 1000)
        corners.append((end_lon, end_lat))

    # Step 3: Reorder the corners to form a closed rectangle (clockwise)
    corners.append((back_center_lon, back_center_lat))

    # Plot the rectangle on the GeoAxes
    lons, lats = zip(*corners)
    geo_ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)   

def add_profile_rectangle(
    geo_ax,
    length: Any,
    width: Any,
    mode: str,
    init_point: tuple,
    azimuth: Any,
    # fault_start: tuple | None = None,
    # fault_end: tuple | None = None,
    **kwargs
    ):
    """
    Draws a dashed rectangle representing the profile range on a Cartopy GeoAxes.

    Parameters:
        extend_mode (str): 'start' / 'center'

    Returns:
        None
    """
    lon, lat = init_point
    try:
        extend_map = {
            'start': _start_extend,
            'center': _center_extend,
        }[mode](geo_ax, lon, lat, length, width, azimuth, **kwargs)
    except Exception as e:
        raise IndexError(f'Please provide the correct type for profile extension.')


    # if fault_start is not None and fault_end is not None:
    #     mid_lon, mid_lat = (fault_start[0] + fault_end[0]) / 2, (fault_start[1] + fault_end[1]) / 2
    #     azimuth = calculate_fault_azimuth(fault_start, fault_end)
    # elif center is not None and azimuth is not None:
    #     mid_lon , mid_lat = center
    # else:
    #     raise IndexError('Please provide the parameters')
    # Calculate half-length and half-width

def add_profile_line(
    geo_ax: Axes,
    extend_km: Any,
    azimuth: Any,
    mode: str,
    init_point: tuple,
    # fault_start: tuple | None = None,
    # fault_end: tuple | None = None,
    **kwargs
    ):
    """
    Draws a dashed line representing the profile range on a Cartopy GeoAxes.

    Parameters:
        geo_ax (GeoAxes): Cartopy GeoAxes object.
        azimith (float): directly orientation of the line, no need to + 90 degrees.

    Returns:
        None
    """
    geod = Geod(ellps="WGS84")

    # Calculate the midpoint of the profile
    # if fault_start is not None and fault_end is not None:
    #     mid_lon, mid_lat = (fault_start[0] + fault_end[0]) / 2, (fault_start[1] + fault_end[1]) / 2
    #     azimuth = calculate_fault_azimuth(fault_start, fault_end) # parallel to fault
    #     start_lon, start_lat, _ = geod.fwd(mid_lon, mid_lat, azimuth - 90, (extend_km / 2)*1000)
    #     azimuth = azimuth + 90 # perpendicular to fault
    # elif start_point is not None and azimuth is not None:
    #     start_lon , start_lat = start_point
    # else:
    #     raise IndexError('Please provide the parameters')

    # the extend point
    lon, lat = init_point
    if mode == 'start':
        start_lon, start_lat = init_point
        end_lon, end_lat, _ = geod.fwd(start_lon, start_lat, azimuth, extend_km * 1000)
    elif mode == 'center':
        start_lon, start_lat, _ = geod.fwd(lon, lat, azimuth - 180, (extend_km / 2)*1000)
        end_lon, end_lat, _ = geod.fwd(lon, lat, azimuth, (extend_km / 2)*1000)
    else:
        raise KeyError('Please provide correct mode.')
    # Step 2: From the front and back center points, extend to the left and right to get the corners
    corners = []
    corners.append((start_lon, start_lat))
    corners.append((end_lon, end_lat))

    # Plot the line on the GeoAxes
    lons, lats = zip(*corners)
    geo_ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)

def add_single_profile(events_df, ax: Axes, width: Any, depth_range=(0, 60), temporal=True,
                       manual_min=None, manual_max=None, show_xlabel=True, show_ylabel=True, size=5, text_mode=False, **kwargs):
    """
    Plots a seismic profile on a given axis.
    
    Parameters:
        events_df (pd.DataFrame): Contains 'x' (distance along profile) and 'depth' columns.
        ax (matplotlib.axes.Axes): The axis to plot on.
        width (float): Profile width (in km).
        depth_range (tuple): Depth range (min_depth, max_depth) in km.
        profile (str): Profile name (for labeling).
    """
    if temporal:
        colormap = plt.cm.viridis
        norm = mcolors.Normalize(
            vmin=events_df['day'].min(),
            vmax=events_df['day'].max()
            )  # Normalize days to range [0, 1]
        if manual_min is not None and manual_max is not None:
            norm = mcolors.Normalize(
                vmin=manual_min,
                vmax=manual_max
                )             
        for day, group in events_df.groupby('day'):
            # print(f'day {day}: {len(group)}')
            color = colormap(norm(day))    
            ax.scatter(
                group['x'],
                group['depth'],
                c='k',#color,
                s=5,
                alpha=0.8
                )
    else:
        ax.scatter(
            events_df['x'],
            events_df['depth'],
            s=size,
            c= events_df['color'], #TODO: profile_verify conflict with temp_profile!
            **kwargs
            )
        if text_mode:
            for _, row in events_df.iterrows():
                ax.text(
                    row.x+0.3,
                    row.depth,
                    s=str(int(row.group)),
                    c='r' if (row.longitude == 121.574) & (row.latitude == 23.88) else 'k',
                    fontsize=8
                    )            
    # ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    #TODO: if using loop here, below are repeated called.
    ax.set_xlim(0, width)#(-width / 2, width / 2)
    ax.set_ylim(depth_range[0], depth_range[1])
    ax.tick_params(axis='both', labelsize=15)
    for spine in ax.spines.values():
        spine.set_edgecolor('k')
        spine.set_linewidth(2)
    if show_xlabel:
        ax.set_xlabel("Distance (km)", fontsize=20, weight='bold')
    else:
        ax.tick_params(axis='x', labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("Depth (km)", fontsize=20, weight='bold')
    ax.invert_yaxis()
    # ax.grid()

def _daily_job(figsize, num_columns, profile_nums, start_point, other_catalog, length, width, depth_range, group, 
              num_rows, azimuth, topo_csv, plot_vel, vel_df, vel_mode, vel_cmap, manual_min, manual_max, save_dir, day):
    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(num_rows, num_columns, figure=fig)   
    letters = string.ascii_uppercase
    for i, (dis, letter) in enumerate(zip(np.linspace(0, (length / 2)*(profile_nums-1), profile_nums), letters[0:profile_nums])):    
        geod = Geod(ellps='WGS84')
        next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], azimuth, dis*1000)
        logging.info(f'Next start point: ({next_lon}, {next_lat})')
        new_lon, new_lat, _ = geod.fwd(next_lon, next_lat, azimuth + 90, (width / 2)*1000)
        new_center = (new_lon, new_lat)
        # plot background events
        if other_catalog is not None:
            other_events = project_events_to_profile(
                df_catalog=other_catalog,
                length=length, 
                width=width,
                depth_range=depth_range,
                center=new_center,
                azimuth=20,
                )                
        profile_events = project_events_to_profile(
            df_catalog=group,
            length=length, 
            width=width,
            depth_range=depth_range,
            center=new_center,
            azimuth=20
            )
        row = i % num_rows
        col = i // num_rows
        inner_gs = GridSpecFromSubplotSpec(
            2,
            3,
            subplot_spec=outer_gs[row, col],
            height_ratios=[1, 3],
            width_ratios=[1, 1, 0.05]
        )
        # plotting topo
        ax1 = fig.add_subplot(inner_gs[0, :-1])
        distances, elevations = extract_topo_profile_with_geod(
            csv_path=topo_csv,
            start_point=(next_lon, next_lat),
            azimuth=azimuth+90,
            distance_km=width
            )
        add_topo_profile(
            ax=ax1,
            distances=distances,
            elevations=elevations,
            profile_letter=letter
        )
        # plotting profile
        ax2 = fig.add_subplot(inner_gs[1, :-1])
        cax = fig.add_subplot(inner_gs[1, -1:])
        if plot_vel:

            #TODO: Adding a condition to solve possibly unbound.
            add_vel_profile(
                ax=ax2,
                vel_df=vel_df,
                start_point=(next_lon, next_lat),
                mode=vel_mode,
                cax=cax,
                cmap=vel_cmap,
                distance=width,
                azimuth=azimuth
            )
        if other_catalog is not None:
            add_single_profile(
                other_events,
                ax2,
                temporal=False,
                c='gray',
                alpha=0.5,
                width=width,
                depth_range=depth_range
                )                
        add_single_profile(
            profile_events,
            ax2,
            manual_min=manual_min,
            manual_max=manual_max,
            width=width,
            depth_range=depth_range
            )
    plt.tight_layout()
    plt.savefig(save_dir / f'gridspec_cmap_{day}.png', dpi=300)
    plt.close()

def daily_profile(
    df_catalog: pd.DataFrame,
    topo_csv,
    start_point=(121.08, 23.65),
    other_catalog: pd.DataFrame | None = None,
    length=10,
    width=80,
    azimuth=20,
    depth_range=(0, 60),
    save_dir=Path(''),
    save_name='',
    profile_nums=21,
    num_rows=3,
    figsize=(),
    manual_min=1,
    manual_max=17,
    plot_vel=False,
    vel_txt: Path | None = None,
    vel_mode='vpt',
    vel_cmap='RdYlBu',
    temporal=False
    ):
    """
    Plots a seismic profile on a given axis.
    
    Parameters:
        azimuth (int | float): the profile moving azimuth, to be more specific, the slope between the start point of A-A' and B-B'
    """
    if profile_nums % num_rows == 0:
        num_columns = profile_nums // num_rows
    else:    
        num_columns = (profile_nums // num_rows) + 1
    if len(figsize) == 0:
        figsize=(num_columns * 10, num_rows * 7)

    geod = Geod(ellps='WGS84')
    
    if vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')

    if temporal:
        args =[
            (figsize, num_columns, profile_nums, start_point,
            other_catalog, length, width, depth_range, group, 
            num_rows, azimuth, topo_csv, plot_vel, vel_df, vel_mode,
            vel_cmap, manual_min, manual_max, save_dir, day)
            for day, group in df_catalog.groupby('day')
            ]        
        with mp.Pool(processes=20) as pool:
            pool.starmap(_daily_job, args)
    else:
        fig = plt.figure(figsize=figsize)
        outer_gs = GridSpec(num_rows, num_columns, figure=fig)   
        letters = string.ascii_uppercase
        for i, (dis, letter) in enumerate(
            zip(
                np.linspace(0, (length / 2)*(profile_nums-1), profile_nums),
                letters[0:profile_nums]
                )
                ):
            geod = Geod(ellps='WGS84')
            next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], azimuth, dis*1000)
            logging.info(f'Next start point: ({next_lon}, {next_lat})')
            new_lon, new_lat, _ = geod.fwd(next_lon, next_lat, azimuth + 90, (width / 2)*1000)
            new_center = (new_lon, new_lat)
            logging.info(f'Center point: {new_center}')
            # plot background events
            if other_catalog is not None:
                other_events = project_events_to_profile(
                    df_catalog=other_catalog,
                    length=length, 
                    width=width,
                    depth_range=depth_range,
                    center=new_center,
                    azimuth=azimuth,
                    )                
            profile_events = project_events_to_profile(
                df_catalog=df_catalog,
                length=length, 
                width=width,
                depth_range=depth_range,
                center=new_center,
                azimuth=azimuth
                )
            row = i % num_rows
            col = i // num_rows
            inner_gs = GridSpecFromSubplotSpec(
                2,
                3,
                subplot_spec=outer_gs[row, col],
                height_ratios=[1, 3],
                width_ratios=[1, 1, 0.05]
            )
            # plotting topo
            ax1 = fig.add_subplot(inner_gs[0, :-1])
            distances, elevations = extract_topo_profile_with_geod(
                csv_path=topo_csv,
                start_point=(next_lon, next_lat),
                azimuth=azimuth+90,
                distance_km=width
                )
            add_topo_profile(
                ax=ax1,
                distances=distances,
                elevations=elevations,
                profile_letter=letter
            )
            # plotting profile
            ax2 = fig.add_subplot(inner_gs[1, :-1])
            cax = fig.add_subplot(inner_gs[1, -1:])
            if plot_vel:
                #TODO: Adding a condition to solve possibly unbound.
                add_vel_profile(
                    ax=ax2,
                    vel_df=vel_df,
                    start_point=(next_lon, next_lat),
                    mode=vel_mode,
                    cax=cax,
                    cmap=vel_cmap,
                    distance=width,
                    azimuth=azimuth + 90
                )
            if other_catalog is not None:
                add_single_profile(
                    other_events,
                    ax2,
                    temporal=False,
                    c='gray',
                    alpha=0.5,
                    width=width,
                    depth_range=depth_range
                    )                
            add_single_profile(
                profile_events,
                ax2,
                # manual_min=manual_min,
                # manual_max=manual_max,
                temporal=False,
                c='k',
                width=width,
                depth_range=depth_range
                )
        plt.tight_layout()
        plt.savefig(save_dir / save_name, dpi=300)
        plt.close()


def mapview_with_profile(
    station: Path,
    topo_csv,
    catalog: pd.DataFrame,
    other_catalog: pd.DataFrame | None = None,
    depth_range=(0, 60),
    profile_nums=6,
    num_rows=3,
    catalog_filter_: dict | None = None,
    title='Event distribution',
    eq_list=[],
    main_eq_size=10,
    plot_station_name=True,
    savefig=True,
    fig_dir: Path | None = None,
    h3dd_mode=False,
    prefix='',
    savename='',
    temporal=False,
    plot_profile_line=False,
    horizontal_start_point=(121.08, 23.65),
    horizontal_movement={'length': 10, 'width': 80, 'azimuth': 20, 'steps': np.linspace(0, 100, 21), 'letters': Any}, # length, width, azimuth
    vertical_start_point=(121.485, 24.667),
    vertical_movement={'length': 10, 'width': 120, 'azimuth': 110, 'steps': np.linspace(0, 70, 15), 'letters': Any},
    target_start_point=(121.485, 24.667),
    target_movement={'length': 10, 'width': 120, 'azimuth': 110, 'steps': np.linspace(0, 70, 15), 'letters': Any},
    plot_vel_on_mapview=False,
    vel_txt: Path | None = None,
    vel_mode='vpt',
    vel_cmap='RdYlBu',
    profile_init_mode='start'
):
    """
    catalog should at least contains the lon and lat.

    """
    # plot setting
    fig = plt.figure(figsize=(36, 18))
    parent_gs = GridSpec(1, 2, width_ratios=[1, 1])

    geo_ax = fig.add_subplot(parent_gs[0, 0], projection=ccrs.PlateCarree())

    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]

    get_mapview(region=region, ax=geo_ax, title=title, fontsize=25)
    if plot_vel_on_mapview and vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')
        add_vel_view(
            ax=geo_ax,
            vel_df=vel_df,
            region=region
        )
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].apply(_get_size)
    # station
    df_station = pd.read_csv(station)
    # print(
    #     'Geo Axes Patch Extent:',
    #     geo_ax.get_window_extent(renderer=fig.canvas.get_renderer()),
    # )
    plot_station(
        df_station=df_station,
        geo_ax=geo_ax,
        region=region,
        plot_station_name=plot_station_name,
        text_dist=0.005,
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
    # add all events
    event_num = len(catalog)
    manual_elements.append(
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label=f'Event num: {event_num}',
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k',
        )
    )       
    if temporal:
        colormap = plt.cm.viridis
        norm = mcolors.Normalize(
            vmin=catalog['day'].min(),
            vmax=catalog['day'].max()
            )  # Normalize days to range [0, 1]
        for day, group in catalog.groupby('day'):
            print(f'day {day}: {len(group)}')
            color = colormap(norm(day))
            geo_ax.scatter(
                group[f'{prefix}longitude'],
                group[f'{prefix}latitude'],
                s=group['size'],
                c=color,
                alpha=0.5,
                label=f'Day {day} event num: {len(group)}',
                zorder=3,
            )
            # Add a colorbar to indicate progression
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        # Create an inset for the colorbar within geo_ax
        cbar_ax = inset_axes(
            geo_ax,  # Parent axis (geo_ax)
            width="3%",  # Width of the colorbar relative to geo_ax
            height="50%",  # Height of the colorbar relative to geo_ax
            loc="lower right",  # Location inside the geo_ax
            bbox_to_anchor=(0.05, 0.05, 0.85, 0.9),  # Adjust position within geo_ax
            bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
            borderpad=0,  # Padding around the inset
        )
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label("Day Since 0401")
    else:
        geo_ax.scatter(
            catalog[f'{prefix}longitude'],
            catalog[f'{prefix}latitude'],
            s=catalog['size'],
            c='k',
            alpha=0.5,
            zorder=3,
        )         

    # profile
    geod = Geod(ellps='WGS84')
    if plot_profile_line:
        # letters = string.ascii_uppercase
        # horizontal
        for i, (dis, letter) in enumerate(zip(horizontal_movement['steps'], horizontal_movement['letters'])):
            next_lon, next_lat, _ = geod.fwd(horizontal_start_point[0], horizontal_start_point[1], horizontal_movement['azimuth'], dis*1000)
            text_lon, text_lat, _ = geod.fwd(horizontal_start_point[0], horizontal_start_point[1], horizontal_movement['azimuth'], dis*1000)
            add_profile_line(
                geo_ax=geo_ax,
                extend_km=horizontal_movement['width'],
                init_point=(next_lon, next_lat),
                mode=profile_init_mode,
                azimuth=horizontal_movement['azimuth'] + 90,
                alpha=0.7, c='k'
                )   
            geo_ax.text(x=text_lon -0.02, y=text_lat, s=f"{letter}-{letter}'")
        
        # vertical
        for i, (dis, letter) in enumerate(zip(vertical_movement['steps'], vertical_movement['letters'])):
            next_lon, next_lat, _ = geod.fwd(vertical_start_point[0], vertical_start_point[1], vertical_movement['azimuth'], dis*1000)
            text_lon, text_lat, _ = geod.fwd(vertical_start_point[0], vertical_start_point[1], vertical_movement['azimuth'], dis*1000)
            add_profile_line(
                geo_ax=geo_ax,
                extend_km=vertical_movement['width'],
                init_point=(next_lon, next_lat),
                mode=profile_init_mode,
                azimuth=vertical_movement['azimuth'] + 90,
                alpha=0.7, c='k'
                )   
            geo_ax.text(x=text_lon -0.02, y=text_lat, s=f"{letter}-{letter}'")                    

    # add main eq
    for eq in eq_list:
        _add_epicenter(
            x=eq[0], y=eq[1], ax=geo_ax, size=main_eq_size, alpha=0.7, color=eq[4]
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
    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    geo_ax.legend(
        handles=manual_elements,  # + legend_elements,
        loc='upper left',
        markerscale=2,
        labelspacing=1.5,
        borderpad=1,
        fontsize=10,
        framealpha=0.6,
    )

    # plotting profiles
    if profile_nums % num_rows == 0:
        num_columns = profile_nums // num_rows
    else:    
        num_columns = (profile_nums // num_rows) + 1

    if vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')
    outer_gs = GridSpecFromSubplotSpec(
        num_rows,
        num_columns,
        subplot_spec=parent_gs[0, 1],
    )
    for i, (dis, letter) in enumerate(zip(target_movement['steps'], target_movement['letters'])):
        geod = Geod(ellps='WGS84')
        next_lon, next_lat, _ = geod.fwd(target_start_point[0], target_start_point[1], target_movement['azimuth'], dis*1000)
        logging.info(f'Next start point: ({next_lon}, {next_lat})')
        add_profile_line(
            geo_ax=geo_ax,
            extend_km=target_movement['width'],
            azimuth=target_movement['azimuth'] + 90,
            mode=profile_init_mode,
            init_point=(next_lon, next_lat),
            c='yellow'
        )
        new_lon, new_lat, _ = geod.fwd(next_lon, next_lat, target_movement['azimuth'] + 90, (target_movement['width'] / 2)*1000)
        new_center = (new_lon, new_lat)
        logging.info(f'Center point: {new_center}')
        # plot background events
        if other_catalog is not None:
            other_events = project_events_to_profile(
                df_catalog=other_catalog,
                length=target_movement['length'], 
                width=target_movement['width'],
                depth_range=depth_range,
                center=new_center,
                azimuth=target_movement['azimuth'],
                )                
        profile_events = project_events_to_profile(
            df_catalog=catalog,
            length=target_movement['length'], 
            width=target_movement['width'],
            depth_range=depth_range,
            center=new_center,
            azimuth=target_movement['azimuth']
            )
        row = i % num_rows
        col = i // num_rows
        inner_gs = GridSpecFromSubplotSpec(
            2,
            3,
            subplot_spec=outer_gs[row, col],
            height_ratios=[1, 3],
            width_ratios=[1, 1, 0.05]
        )
        # plotting topo
        ax1 = fig.add_subplot(inner_gs[0, :-1])
        distances, elevations = extract_topo_profile_with_geod(
            csv_path=topo_csv,
            start_point=(next_lon, next_lat),
            azimuth=target_movement['azimuth']+90,
            distance_km=target_movement['width']
            )
        add_topo_profile(
            ax=ax1,
            distances=distances,
            elevations=elevations,
            profile_letter=letter
        )
        # plotting profile
        ax2 = fig.add_subplot(inner_gs[1, :-1])
        cax = fig.add_subplot(inner_gs[1, -1:])
        if vel_txt is not None:
            #TODO: Adding a condition to solve possibly unbound.
            add_vel_profile(
                ax=ax2,
                vel_df=vel_df,
                start_point=(next_lon, next_lat),
                mode=vel_mode,
                cax=cax,
                cmap=vel_cmap,
                distance=target_movement['width'],
                azimuth=target_movement['azimuth'] + 90
            )
        if other_catalog is not None:
            add_single_profile(
                other_events,
                ax2,
                temporal=False,
                c='gray',
                alpha=0.5,
                width=target_movement['width'],
                depth_range=depth_range
                )                
        add_single_profile(
            profile_events,
            ax2,
            # manual_min=manual_min,
            # manual_max=manual_max,
            temporal=False,
            c='k',
            width=target_movement['width'],
            depth_range=depth_range
            )    
    
    if savefig and fig_dir is not None:
        plt.tight_layout()
        plt.savefig(fig_dir / f'{savename}.png', dpi=300, bbox_inches='tight')

def _add_azi_fix(
        geo_ax: Axes,
        horizontal_start_point: tuple,
        horizontal_movement: dict,
        vertical_start_point: tuple,
        vertical_movement: dict,
        profile_init_mode: str,
        profile_shape: str,
):
    geod = Geod(ellps='WGS84')
    # horizontal
    for i, (dis, letter) in enumerate(zip(horizontal_movement['steps'], horizontal_movement['letters'])):
        next_lon, next_lat, _ = geod.fwd(
            horizontal_start_point[0], horizontal_start_point[1], horizontal_movement['azimuth'], dis*1000
            )
        text_lon, text_lat, _ = geod.fwd(
            horizontal_start_point[0], horizontal_start_point[1], horizontal_movement['azimuth'], dis*1000
            )
        if profile_shape == 'line':
            add_profile_line(
                geo_ax=geo_ax,
                extend_km=horizontal_movement['width'],
                mode=profile_init_mode,
                init_point=(next_lon, next_lat),
                azimuth=horizontal_movement['azimuth'] + 90,
                alpha=0.7,
                c='k',
                zorder=10
                )
        elif profile_shape == 'rectangle':
            add_profile_rectangle(
                geo_ax=geo_ax,
                length=horizontal_movement['length'],
                width=horizontal_movement['width'],
                mode=profile_init_mode,
                init_point=(next_lon, next_lat),
                azimuth=horizontal_movement['azimuth'] + 90,
                alpha=0.7,
                c='k',
                zorder=10
            )
        geo_ax.text(x=text_lon -0.02, y=text_lat, s=f"{letter}-{letter}'", zorder=12)
    
    # vertical
    for i, (dis, letter) in enumerate(zip(vertical_movement['steps'], vertical_movement['letters'])):
        next_lon, next_lat, _ = geod.fwd(
            vertical_start_point[0], vertical_start_point[1], vertical_movement['azimuth'], dis*1000
            )
        text_lon, text_lat, _ = geod.fwd(
            vertical_start_point[0], vertical_start_point[1], vertical_movement['azimuth'], dis*1000
            )
        if profile_shape == 'line':
            add_profile_line(
                geo_ax=geo_ax,
                extend_km=vertical_movement['width'],
                mode=profile_init_mode,
                init_point=(next_lon, next_lat),
                azimuth=vertical_movement['azimuth'] + 90,
                alpha=0.7,
                c='k'
                )   
        elif profile_shape == 'rectangle':
            add_profile_rectangle(
                geo_ax=geo_ax,
                length=vertical_movement['length'],
                width=vertical_movement['width'],
                mode=profile_init_mode,
                init_point=(next_lon, next_lat),
                azimuth=vertical_movement['azimuth'] + 90,
                alpha=0.7,
                c='k'
            )                
        geo_ax.text(x=text_lon -0.02, y=text_lat, s=f"{letter}-{letter}'")   

def _add_azi_free(
    geo_ax: Axes,
    profile_settings: list[dict]
):
    """
    Typically, this way should use center.
    The dict should contain keys list below:
    - init_point
    - mode (center/start)
    - azimuth (orientation you want to extend)
    - length (shorter one)
    - width (longer one)
    - letter (profile name, ex. A will display as A-A')
    - shape (line or rectangle)
    """
    geod = Geod(ellps='WGS84')
    for setting_ in profile_settings:
        start_point = setting_['start_point']
        center = setting_['center']
        mode = setting_['mode']
        azimuth = setting_['azimuth'] + 90
        length = setting_['length']
        width = setting_['width']
        letter = setting_['letter']
        shape = setting_['shape']
        if shape == 'line':
            add_profile_line(
                geo_ax=geo_ax,
                extend_km=width,
                azimuth=azimuth,
                mode=mode,
                init_point=center,
                alpha=0.7, c='k'
            )
        elif shape == 'rectangle':
            add_profile_rectangle(
                geo_ax=geo_ax,
                length=length,
                width=width,
                mode=mode,
                init_point=center,
                azimuth=azimuth,
                alpha=0.7, c='k'                
            )
        
        geo_ax.text(x=start_point[0] -0.02, y=start_point[1], s=f"{letter}-{letter}'")

def add_focal(ax: Axes, df: pd.DataFrame, focal_size_mapping: dict, color=mpl_color('k'), temporal=False, **kwargs):
    """This is for adding focal on mapview
    """
    for i in ['strike', 'dip', 'rake']:
        df[i] = df[i].map(lambda x: x.split('+')[0])
    
    df.loc[:, 'size'] = df['magnitude'].apply(lambda mag: _get_size(mag, focal_size_mapping))

    df.sort_values(by='magnitude', inplace=True)

    for row in df.itertuples(index=False):
        mt = pmt.MomentTensor(
            strike=int(row.strike),
            dip=int(row.dip),
            rake=int(row.rake),
        )
        if temporal:
            beachball.plot_beachball_mpl(
                mt, ax, size=row.size, beachball_type='full',
                position=(row.longitude, row.latitude), linewidth=0.5, color_t=row.color, zorder=6, **kwargs
                )  
        else:
            beachball.plot_beachball_mpl(
                mt, ax, size=row.size, beachball_type='full',
                position=(row.longitude, row.latitude), linewidth=0.5, color_t=color, zorder=6, **kwargs
                )  

def add_focal_profile(
    ax: Axes,
    gafocal_df: pd.DataFrame,
    length: Any,
    width: Any,
    center: tuple,
    azimuth: Any,
    view: tuple,
    cmap: str,
    focal_size_mapping=focal_size_mapping,
    flex_temporal=False,
    color=mpl_color('k'),
    quality_mode=False,
    cax: Axes | None = None
):
    gafocal_df['color'] = 'k' # for passing project function, modifying it please.
    df = project_events_to_profile(
        df_catalog=gafocal_df,
        length=length,
        width=width,
        center=center,
        azimuth=azimuth,
        focal=True,
        flex_temporal=flex_temporal
    )
    #TODO: We do it in check_format
    # for i in ['strike', 'dip', 'rake']:
    #     df[i] = df[i].map(lambda x: x.split('+')[0])
    if quality_mode:
        df.loc[:, 'size'] = df['quality_index'].apply(lambda qi: _get_size(qi, focal_size_mapping))
    else:    
        df.loc[:, 'size'] = df['magnitude'].apply(lambda mag: _get_size(mag, focal_size_mapping))

    df = df.sort_values(by='magnitude')
    # df = df.sort_values(by='quality_index')
    min_quality = df['quality_index'].min()
    max_quality = 10 # NOTE: This is because our data distribution
    norm = mcolors.Normalize(min_quality, max_quality)
    color_map = plt.get_cmap(cmap)
    df['color'] = df['quality_index'].map(lambda q: mcolors.to_hex(color_map(norm(q)))) # NOTE: read to_hex 
    for row in df.itertuples(index=False):
        mt = pmt.MomentTensor(
            strike=int(row.strike),
            dip=int(row.dip),
            rake=int(row.rake),
        )
        beachball.plot_beachball_mpl(
            mt, ax, size=row.size, beachball_type='full',
            position=(row.x, row.depth), linewidth=0.5, color_t=row.color, zorder=6,
            view=view
            ) 
    if cax is not None:
        sm = cm.ScalarMappable(norm, color_map)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Quality Index", fontsize=15, weight='bold')
        
        # Get ticks
        ticks = cbar.get_ticks()
        # Replace last tick label
        labels = [str(int(t)) for t in ticks[:-1]] + ['>10']

        # Set them back
        # cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels) 
        
def proj_epicenter(profile_center, epi_center, azimuth, width, length):
    
    geod=Geod(ellps='WGS84')

    event_az, _, event_dist = geod.inv(
        profile_center[0], profile_center[1],
        epi_center[0], epi_center[1]
        )
    # 計算事件在剖面方向與垂直方向的距離分量
    relative_angle = np.radians(event_az - azimuth)  # 方位角差值
    y = event_dist * np.cos(relative_angle) / 1000  # 剖面方向距離（公里）
    x = event_dist * np.sin(relative_angle) / 1000  # 垂直剖面距離（公里）    
    if (abs(y) <= length / 2 and abs(x) <= width / 2):    
        return x+width/2, y
    else:
        return -2, 0
    
def color_mapping(catalog: pd.DataFrame, time_reverse=False):
    PERIOD = [
        '2024-04-08T00:00:00', # period1
        '2024-04-22T00:00:00', # period2
        '2024-04-29T00:00:00' # period3        
    ]
    def judge(time):
        if time <= pd.Timestamp(PERIOD[0]):
            return 'k'
        elif time <= pd.Timestamp(PERIOD[1]):
            return 'g'
        elif time <= pd.Timestamp(PERIOD[2]):
            return 'purple'
        else:
            raise ValueError("What's this??")
    df = catalog.copy()
    df['utc_time'] = pd.to_datetime(df['time'])
    df['color'] = df['utc_time'].map(judge)
    if time_reverse:
        df = df.sort_values(by='utc_time', ascending=False, ignore_index=True)
    df = df.drop(columns="utc_time")
    return df

def profile_verify(
    station: Path,
    topo_csv,
    catalog: pd.DataFrame,
    gafocal_df: pd.DataFrame | None = None,
    other_catalog: pd.DataFrame | None = None,
    depth_range=(0, 60),
    profile_nums=2,
    num_rows=1,
    catalog_filter_: dict | None = None,
    title='Event distribution',
    eq_list=[],
    plot_station_name=True,
    savefig=True,
    fig_dir: Path | None = None,
    h3dd_mode=False,
    prefix='',
    savename='',
    temporal=False,
    profile_settings={},
    horizontal_start_point=(),
    horizontal_movement={}, # length, width, azimuth
    vertical_start_point=(),
    vertical_movement={},
    target_start_point=(),
    target_movement={},
    plot_vel_on_mapview=False,
    vel_txt: Path | None = None,
    vel_mode='all',
    vel_cmap='RdYlBu',
    kde_mode=False,
    kde_min=0.0,
    profile_fix=False,
    profile_init_mode='start',
    profile_shape='line',
    focal_size_mapping=focal_size_mapping,
    size_mapping=size_mapping,
    legend_on_map=False,
    time_reverse=True,
    set_alpha=1,
    topo_alpha=1,
    focal_cmap='Oranges'
):
    """
    catalog should at least contains the lon and lat.

    Args:
        station (Path):
        topo_csv (Path):
        horizontal_start_point=(121.08, 23.65),
        horizontal_movement={'length': 10, 'width': 80, 'azimuth': 20, 'steps': np.linspace(0, 100, 21), 'letters': Any}, # length, width, azimuth
        vertical_start_point=(121.485, 24.667),
        vertical_movement={'length': 10, 'width': 120, 'azimuth': 110, 'steps': np.linspace(0, 70, 15), 'letters': Any},
        target_start_point=(121.485, 24.667),
        target_movement={'length': 10, 'width': 120, 'azimuth': 110, 'steps': np.linspace(0, 70, 15), 'letters': Any}
    Returns:
        None
    """
    # plot setting
    fig = plt.figure(figsize=(36, 18))
    fig.patch.set_alpha(set_alpha)
    parent_gs = GridSpec(1, 2, width_ratios=[1, 2])

    geo_ax = fig.add_subplot(parent_gs[0, 0], projection=ccrs.PlateCarree())

    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]

    get_mapview(region=region, ax=geo_ax, title=title, fontsize=25)
    if plot_vel_on_mapview and vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')
        add_vel_view(
            ax=geo_ax,
            vel_df=vel_df,
            region=region
        )
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].apply(lambda x: _get_size(x, size_mapping=size_mapping))

    # define color in diff period
    catalog = color_mapping(catalog, time_reverse=time_reverse)
    # Calculate the KDE
    if kde_mode:
        X, Y, density = calc_kde(catalog)
        # geo_ax.imshow(
        #     density,
        #     extent=region,
        #     origin='lower',
        #     alpha=0.7,
        #     cmap=plt.cm.gist_earth_r,
        #     zorder=5
        #     )
        # geo_ax.contour(X, Y, density, levels=10, alpha=0.8, colors='k', zorder=5)        
        levels = np.linspace(kde_min, density.max(), 10)
        contour_lines = geo_ax.contourf(X, Y, density, levels=levels, alpha=0.7, cmap="gist_earth_r", zorder=5)
        geo_ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
    # station
    df_station = pd.read_csv(station)
    # print(
    #     'Geo Axes Patch Extent:',
    #     geo_ax.get_window_extent(renderer=fig.canvas.get_renderer()),
    # )
    plot_station(
        df_station=df_station,
        geo_ax=geo_ax,
        region=region,
        plot_station_name=plot_station_name,
        text_dist=0.005,
        zorder=6
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
    # add all events
    event_num = len(catalog)
    manual_elements.append(
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label=f'Event num: {event_num}',
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k',
        )
    )       
    if temporal:
        colormap = plt.cm.viridis
        norm = mcolors.Normalize(
            vmin=catalog['day'].min(),
            vmax=catalog['day'].max()
            )  # Normalize days to range [0, 1]
        for day, group in catalog.groupby('day'):
            print(f'day {day}: {len(group)}')
            color = colormap(norm(day))
            geo_ax.scatter(
                group[f'{prefix}longitude'],
                group[f'{prefix}latitude'],
                s=group['size'],
                c=color,
                alpha=0.5,
                label=f'Day {day} event num: {len(group)}',
                zorder=12,
            )
            # Add a colorbar to indicate progression
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        # Create an inset for the colorbar within geo_ax
        cbar_ax = inset_axes(
            geo_ax,  # Parent axis (geo_ax)
            width="3%",  # Width of the colorbar relative to geo_ax
            height="50%",  # Height of the colorbar relative to geo_ax
            loc="lower right",  # Location inside the geo_ax
            bbox_to_anchor=(0.05, 0.05, 0.85, 0.9),  # Adjust position within geo_ax
            bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
            borderpad=0,  # Padding around the inset
        )
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label("Day Since 0401")
    else:
        geo_ax.scatter(
            catalog[f'{prefix}longitude'],
            catalog[f'{prefix}latitude'],
            s=catalog['size'],
            c=catalog['color'], # for plotting diff period
            alpha=0.5,
            zorder=12,
        )         
    
    # profile
    geod = Geod(ellps='WGS84')
    
    # plot the profile with free azimuth or follow a specfic azimuth
    if profile_fix:
        _add_azi_fix(geo_ax, horizontal_start_point, horizontal_movement,
                vertical_start_point, vertical_movement, profile_init_mode, profile_shape)
    else:
        if profile_settings:
            _add_azi_free(geo_ax, profile_settings)        

    # add main eq
    for eq in eq_list:
        _add_epicenter(
            x=eq[0], y=eq[1], ax=geo_ax, size=eq[3], alpha=0.7, color=eq[4], zorder=15
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
    # geo_ax.autoscale(tight=True)
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

    ###====== geo_ax setting finish ======###
    #TODO:2025/01/20 Finishing the flexible profile!
    # plotting profiles
    if profile_nums % num_rows == 0:
        num_columns = profile_nums // num_rows
    else:    
        num_columns = (profile_nums // num_rows) + 1

    if vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')
    
    # setting the gridspec besides the mapview
    outer_gs = GridSpecFromSubplotSpec(
        num_rows,
        num_columns,
        subplot_spec=parent_gs[0, 1],
        wspace=0.2
    )

    for i, (dis, letter) in enumerate(zip(target_movement['steps'], target_movement['letters'])):
        geod = Geod(ellps='WGS84')
        next_lon, next_lat, _ = geod.fwd(target_start_point[0], target_start_point[1], target_movement['azimuth'], dis*1000)
        logging.info(f'Next start point: ({next_lon}, {next_lat})')
        add_profile_line(
            geo_ax=geo_ax,
            extend_km=target_movement['width'],
            azimuth=target_movement['azimuth'] + 90,
            mode=profile_init_mode,
            init_point=(next_lon, next_lat),
            c='darkorange',
            lw=3,
            zorder=12
        )
        geo_ax.text(x=next_lon -0.02, y=next_lat, s=f"{letter}-{letter}'", color='darkorange', zorder=12)
        new_lon, new_lat, _ = geod.fwd(next_lon, next_lat, target_movement['azimuth'] + 90, (target_movement['width'] / 2)*1000)
        new_center = (new_lon, new_lat)
        logging.info(f'Center point: {new_center}')

        # plot background events
        if other_catalog is not None:
            other_catalog['color'] = 'gray'
            other_events = project_events_to_profile(
                df_catalog=other_catalog,
                length=target_movement['length'], 
                width=target_movement['width'],
                depth_range=depth_range,
                center=new_center,
                azimuth=target_movement['azimuth'],
                )                
        profile_events = project_events_to_profile(
            df_catalog=catalog,
            length=target_movement['length'], 
            width=target_movement['width'],
            depth_range=depth_range,
            center=new_center,
            azimuth=target_movement['azimuth']
            )
        row = i % num_rows
        col = i // num_rows
        inner_gs = GridSpecFromSubplotSpec(
            4,
            3,
            subplot_spec=outer_gs[row, col],
            height_ratios=[1, 3, 3, 3],
            width_ratios=[1, 1, 0.05],
            hspace=0.1
        )

        # Topology axes
        ax1 = fig.add_subplot(inner_gs[0, :-1])
        distances, elevations = extract_topo_profile_with_geod(
            csv_path=topo_csv,
            start_point=(next_lon, next_lat),
            azimuth=target_movement['azimuth']+90,
            distance_km=target_movement['width']
            )
        add_topo_profile(
            ax=ax1,
            distances=distances,
            elevations=elevations,
            profile_letter=letter,
            ax_alpha=topo_alpha
        )

        # Velocity profile axes
        if vel_mode == 'all':
            for j, mode in zip([1, 2], ['vpt', 'vpvst']):
                ax = fig.add_subplot(inner_gs[j, :-1])
                if i == 1:
                    cax = fig.add_subplot(inner_gs[j, -1:])
                else:
                    cax=None
                if vel_txt is not None:
                    #TODO: Adding a condition to solve possibly unbound.
                    add_vel_profile(
                        ax=ax,
                        vel_df=vel_df,
                        start_point=(next_lon, next_lat),
                        mode=mode,
                        cax=cax,
                        cmap=vel_cmap,
                        distance=target_movement['width'],
                        azimuth=target_movement['azimuth'] + 90
                    )
                if other_catalog is not None:
                    logging.info(f'Other events: {other_events.columns}')
                    logging.info(f"Other evnets color list; {other_events['color'].unique()}")
                    add_single_profile(
                        other_events,
                        ax,
                        temporal=False,
                        # c='gray',
                        alpha=0.5,
                        width=target_movement['width'],
                        depth_range=depth_range,
                        show_xlabel=False,
                        show_ylabel=False if i == 1 else True
                        )                
                add_single_profile(
                    profile_events,
                    ax,
                    # manual_min=manual_min,
                    # manual_max=manual_max,
                    temporal=False,
                    # c='k',
                    width=target_movement['width'],
                    depth_range=depth_range,
                    show_xlabel=False,
                    show_ylabel=False if i == 1 else True
                    )    
                ax.tick_params(labelbottom=False)
                for eq in eq_list:
                    x, _ = proj_epicenter(
                        profile_center=new_center,
                        epi_center=(eq[0], eq[1]),
                        azimuth=target_movement['azimuth'],
                        width=target_movement['width'],
                        length=target_movement['length']
                    )
                    _add_epicenter(
                        x=x, y=eq[2], ax=ax, size=eq[3], alpha=0.7, color=eq[4]
                    )
        # Focal mechanism axes
        if gafocal_df is not None:
            ax = fig.add_subplot(inner_gs[3, :-1])
            # cax = fig.add_subplot(inner_gs[3, -1:])
            if other_catalog is not None:
                add_single_profile(
                    other_events,
                    ax,
                    temporal=False,
                    # c='gray',
                    alpha=0.5,
                    width=target_movement['width'],
                    depth_range=depth_range,
                    show_ylabel=False if i == 1 else True
                    )                
            add_single_profile(
                profile_events,
                ax,
                # manual_min=manual_min,
                # manual_max=manual_max,
                temporal=False,
                # c='k',
                width=target_movement['width'],
                depth_range=depth_range,
                show_ylabel=False if i == 1 else True
                )
            add_focal_profile(
                ax=ax,
                gafocal_df=gafocal_df,
                length=target_movement['length'],
                width=target_movement['width'],
                center=new_center,
                azimuth=target_movement['azimuth'],
                focal_size_mapping=focal_size_mapping,
                view=(),
                cmap=focal_cmap
            )
            for eq in eq_list:
                x, _ = proj_epicenter(
                    profile_center=new_center,
                    epi_center=(eq[0], eq[1]),
                    azimuth=target_movement['azimuth'],
                    width=target_movement['width'],
                    length=target_movement['length'],
                )
                _add_epicenter(
                    x=x, y=eq[2], ax=ax, size=eq[3], alpha=0.5, color=eq[4]
                )            
            # ax.set_xlim(0, target_movement['width'])#(-width / 2, width / 2)
            # ax.set_ylim(depth_range[0], depth_range[1])
            # ax.set_xlabel("Distance (km)", fontsize=20)
            # ax.set_ylabel("Depth (km)", fontsize=20)
            # ax.invert_yaxis()
            #cax.set_axis_off()                 
    
    if savefig and fig_dir is not None:
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'{savename}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_temporal_profile(
    time_period: tuple,
    event_cmap_interval: float,
    profile_settings: list[dict],
    mag_events: pd.DataFrame | Path,
    station: pd.DataFrame | Path,
    topo_csv: Path,
    num_rows: int,
    num_columns: int,
    fig_dir: Path,
    fig_name: str,
    depth_range=(0, 60),
    focal_events: pd.DataFrame | Path | None = None,
    background_events: pd.DataFrame | Path | None = None,
    background_color='r',
    catalog_filter_: dict | None = None, # same as map_region
    eq_list=[],
    vel_txt: Path | None = None,
    plot_vel_on_mapview=False,
    vel_mode='vpt',
    vel_cmap='RdYlBu',
    event_cmap=plt.cm.viridis,
    focal_size_mapping=focal_size_mapping,
    magnitude_mapping=size_mapping,
    legend_on_map=True,
    use_color_bar=True,
    fig_alpha=1,
    topo_alpha=1,
    direct_c='k',
    event_size=5,
    focal_cmap='Oranges'
):
    # plot setting
    fig = plt.figure(figsize=(45, 18))
    fig.patch.set_alpha(fig_alpha)
    parent_gs = GridSpec(1, 2, width_ratios=[1, 2.5])

    geo_ax = fig.add_subplot(parent_gs[0, 0], projection=ccrs.PlateCarree())

    # group the catalog
    def catalog_time_grouping(
            catalog: pd.DataFrame, time_period, event_cmap_interval):
        catalog['datetime'] = pd.to_datetime(catalog['time'])
        catalog = catalog[
            (catalog['datetime'] >= pd.Timestamp(time_period[0]))
            & (catalog['datetime'] <= pd.Timestamp(time_period[1]))
        ].copy()
        min_time = pd.Timestamp(time_period[0])
        if isinstance(event_cmap_interval, float):
            catalog.loc[:, 'time_group'] = (
                (catalog['datetime'] - min_time).dt.total_seconds() // event_cmap_interval
                ).astype(int)
        else:
            raise ValueError("Level must be an float, unit=seconds.") 
        return catalog
    
    ## catalog form
    if isinstance(mag_events, pd.DataFrame):
        catalog = mag_events
    elif isinstance(mag_events, Path):
        catalog = pd.read_csv(mag_events)
    else:
        raise ValueError(f'Format {type(mag_events)} is not supported, using pandas.DataFrame or pathlib.Path')
    catalog = catalog_time_grouping(catalog, time_period, event_cmap_interval)
    
    if focal_events is not None:
        if isinstance(focal_events, pd.DataFrame):
            df_focal = focal_events
        elif isinstance(focal_events, Path):
            df_focal = pd.read_csv(focal_events)
        else:
            raise ValueError(f'Format {type(focal_events)} is not supported, using pandas.DataFrame or pathlib.Path')    
        
        df_focal = catalog_time_grouping(df_focal, time_period, event_cmap_interval)
        df_focal = catalog_filter(
            catalog_df=df_focal, h3dd_mode=False, catalog_range=catalog_filter_
        )
    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]

    title = f"From {time_period[0]}~{time_period[1]}\ninterval={event_cmap_interval} s; Group num: {len(catalog['time_group'].unique())}"
    get_mapview(region=region, ax=geo_ax, title=title, fontsize=25)
    if plot_vel_on_mapview and vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')
        add_vel_view(
            ax=geo_ax,
            vel_df=vel_df,
            region=region
        )
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=False, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].map(
        lambda x: _get_size(x, magnitude_mapping)
    )

    # station
    if isinstance(station, pd.DataFrame):
        df_station = station
    elif isinstance(station, Path):
        df_station = pd.read_csv(station)
    else:
        raise ValueError(f'Format {type(station)} is not supported, using pandas.DataFrame or pathlib.Path')
    
    plot_station(
        df_station=df_station,
        geo_ax=geo_ax,
        region=region,
        plot_station_name=False,
        text_dist=0.005,
        zorder=6
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

    # add all events
    event_num = len(catalog)
    manual_elements.append(
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label=f'Event num: {event_num}',
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k',
        )
    )      

    # plot the color in diff time period
    # unique_times = sorted(catalog['time_group'].unique())
    # time_mapping = {t: i for i, t in enumerate(unique_times)}
    norm = mcolors.Normalize(vmin=catalog['time_group'].min(), vmax=catalog['time_group'].max())   
    colormap = event_cmap
    catalog['color'] = catalog['time_group'].map(lambda x: colormap(norm(x)))
    for group_id, group in catalog.groupby('time_group'):
        min_time = pd.Timestamp(time_period[0])
        color = colormap(norm(group_id))
        # print(f'color: {color}')
        geo_ax.scatter(
            group['longitude'],
            group['latitude'],
            s=group['size'],
            color=group['color'],  # Ensure proper coloring
            alpha=0.5,
            zorder=3
        )
        manual_elements.append(
            Line2D(
                [0],
                [0],
                marker='o',
                color='w',
                label=f'{min_time}~{min_time + pd.Timedelta(seconds=event_cmap_interval*int(group_id+1))}: {len(group)} events',
                markersize=5,
                markerfacecolor=color,
                markeredgecolor='k',
            )
        ) 
    if background_events is not None:
        if isinstance(background_events, Path):
            df_background_events = pd.read_csv(background_events)
        elif isinstance(background_events, pd.DataFrame):
            df_background_events = background_events
        else:
            raise ValueError(f'{type(background_events)} not supported')
        
        df_background_events.loc[:, 'size'] = df_background_events['magnitude'].map(
            lambda x: _get_size(x, magnitude_mapping)
        )        
        geo_ax.scatter(
            df_background_events['longitude'],
            df_background_events['latitude'],
            s=df_background_events['size'],
            color=background_color,  # Ensure proper coloring
            alpha=0.5,
            zorder=2
        )         
    # Add colorbar to indicate progression
    if use_color_bar:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        cbar_ax = inset_axes(
            geo_ax,  # Parent axis (geo_ax)
            width="3%",  # Width of the colorbar relative to geo_ax
            height="50%",  # Height of the colorbar relative to geo_ax
            loc="lower right",  # Location inside the geo_ax
            bbox_to_anchor=(0.05, 0.05, 0.85, 0.9),  # Adjust position within geo_ax
            bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
            borderpad=0,  # Padding around the inset
        )
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(f"Interval {event_cmap_interval} seconds")
    
    ## plot profile
    if profile_settings:
        _add_azi_free(geo_ax, profile_settings)    

    ## plot eq
    if len(eq_list) > 0:
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

    # plotting gridspec profile
    profile_nums = len(profile_settings)

    # if profile_nums % num_rows == 0:
    #     num_columns = profile_nums // num_rows
    # else:    
    #     num_columns = (profile_nums // num_rows) + 1

    if vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')

    # setting the gridspec besides the mapview
    outer_gs = GridSpecFromSubplotSpec(
        num_rows,
        num_columns,
        subplot_spec=parent_gs[0, 1],
        wspace=0.2,
        hspace=0.3
    )
    for i, profile_dict in enumerate(profile_settings):
        start_time = time.time()
        length = profile_dict['length']
        width = profile_dict['width']
        azimuth = profile_dict['azimuth']
        center = profile_dict['center']
        start_point = profile_dict['start_point']
        profile_name = profile_dict['letter']

        if background_events is not None:
            if isinstance(background_events, Path):
                df_background_events = pd.read_csv(background_events)
            elif isinstance(background_events, pd.DataFrame):
                df_background_events = background_events
            else:
                raise ValueError(f'{type(background_events)} not supported')        
            
            other_events = project_events_to_profile(
                df_catalog=df_background_events,
                length=length, 
                width=width,
                depth_range=depth_range,
                center=center,
                azimuth=azimuth
                )
        profile_events = project_events_to_profile(
            df_catalog=catalog,
            length=length, 
            width=width,
            depth_range=depth_range,
            center=center,
            azimuth=azimuth,
            flex_temporal=True
            )
        row = i % num_rows
        col = i // num_rows
        inner_gs = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer_gs[row, col],
            height_ratios=[1, 3],
            width_ratios=[1, 0.05],
            hspace=0.1
        )

        # Topology axes
        ax1 = fig.add_subplot(inner_gs[0, :-1])
        distances, elevations = extract_topo_profile_with_geod(
            csv_path=topo_csv,
            start_point=start_point,
            azimuth=azimuth+90,
            distance_km=width
            )
        add_topo_profile(
            ax=ax1,
            distances=distances,
            elevations=elevations,
            profile_letter=profile_name,
            ax_alpha=topo_alpha,
            show_ylabel=True if col == 0 else False
        )

        # event axes (x=dist, y=depth)
        ax2 = fig.add_subplot(inner_gs[1, :-1])
        
        max_col = profile_nums // num_rows
        if col == max_col:
            cax = fig.add_subplot(inner_gs[1, -1:])
        elif (profile_nums % num_rows == 0) and col == max_col -1:
            cax = fig.add_subplot(inner_gs[1, -1:])
        else:
            cax = None
        if vel_txt is not None:
            add_vel_profile(
                ax=ax2,
                vel_df=vel_df,
                start_point=start_point,
                mode=vel_mode,
                cax=cax,
                cmap=vel_cmap,
                distance=width,
                azimuth=azimuth + 90
            )
        if background_events is not None:
            add_single_profile(
                other_events,
                ax2,
                temporal=False,
                color=direct_c,
                alpha=0.5,
                width=width,
                depth_range=depth_range,
                size=event_size
                )
        profile_events.sort_values(by='group', ascending=False, inplace=True)
        for group_id, group in profile_events.groupby('group'):
            color = colormap(norm(group_id))                    
            add_single_profile(
                group,
                ax2,
                temporal=False,
                color=direct_c,
                width=width,
                depth_range=depth_range,
                # only shows x_label when the profile in the bottom or the last one in prifiles.
                show_xlabel=True if row == num_rows - 1 or i == profile_nums -1 else False,
                show_ylabel=True if col == 0 else False,
                size=event_size
                )
            if focal_events is not None:
                # df_focal['color'] = df_focal['time_group'].map(lambda x: colormap(norm(x)))
                df_part = df_focal[df_focal['time_group'] == group_id]
                if not df_part.empty:
                    add_focal_profile(
                        ax=ax2,
                        gafocal_df=df_part,
                        length=length,
                        width=width,
                        center=center,
                        azimuth=azimuth,
                        focal_size_mapping=focal_size_mapping,
                        flex_temporal=True,
                        color=color,
                        view=(),
                        cmap=focal_cmap
                    )
        if len(eq_list) > 0:
            for eq in eq_list:
                x, _ = proj_epicenter(
                    profile_center=center,
                    epi_center=(eq[0], eq[1]),
                    azimuth=azimuth,
                    width=width,
                    length=length
                )
                if x != 0:
                    _add_epicenter(
                        x=x, y=eq[2], ax=ax2, size=eq[3], alpha=0.7, color=eq[4]
                    )            
        ax2.set_xlim(0, width)#(-width / 2, width / 2)
        ax2.set_ylim(depth_range[0], depth_range[1])
        # ax2.set_xlabel("Distance (km)")
        # ax2.set_ylabel("Depth (km)")
        ax2.invert_yaxis()
        # cax.set_axis_off()                 
        end_time = time.time()
        print(f'Profile {profile_name} finished in {end_time - start_time} seconds')
    if fig_name and fig_dir is not None:
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'{fig_name}.png', dpi=300)
        plt.close()  

def plot_big_profile(
    time_period: tuple,
    event_cmap_interval: float,
    profile_settings: list[dict],
    mag_events: pd.DataFrame | Path,
    # station: pd.DataFrame | Path,
    topo_csv: Path,
    num_rows: int,
    num_columns: int,
    fig_dir: Path,
    fig_name: str,
    depth_range=(0, 60),
    # focal_events: pd.DataFrame | Path | None = None,
    background_events: pd.DataFrame | Path | None = None,
    background_color='r',
    catalog_filter_: dict | None = None, # same as map_region
    eq_list=[],
    vel_txt: Path | None = None,
    plot_vel_on_mapview=False,
    vel_mode='vpt',
    vel_cmap='RdYlBu',
    event_cmap=plt.cm.viridis,
    # focal_size_mapping=focal_size_mapping,
    magnitude_mapping=size_mapping,
    legend_on_map=True,
    use_color_bar=True,
    fig_alpha=1,
    topo_alpha=1,
    direct_c='k'
):
    # plot setting
    fig = plt.figure(figsize=(30, 18))
    fig.patch.set_alpha(fig_alpha)
    parent_gs = GridSpec(1, 2, width_ratios=[1, 2])

    geo_ax = fig.add_subplot(parent_gs[0, 0], projection=ccrs.PlateCarree())

    # group the catalog
    def catalog_time_grouping(
            catalog: pd.DataFrame, time_period, event_cmap_interval):
        catalog['datetime'] = pd.to_datetime(catalog['time'])
        catalog = catalog[
            (catalog['datetime'] >= pd.Timestamp(time_period[0]))
            & (catalog['datetime'] <= pd.Timestamp(time_period[1]))
        ].copy()
        min_time = pd.Timestamp(time_period[0])
        if isinstance(event_cmap_interval, float):
            catalog.loc[:, 'time_group'] = (
                (catalog['datetime'] - min_time).dt.total_seconds() // event_cmap_interval
                ).astype(int)
        else:
            raise ValueError("Level must be an float, unit=seconds.") 
        return catalog
    
    ## catalog form
    if isinstance(mag_events, pd.DataFrame):
        catalog = mag_events
    elif isinstance(mag_events, Path):
        catalog = pd.read_csv(mag_events)
    else:
        raise ValueError(f'Format {type(mag_events)} is not supported, using pandas.DataFrame or pathlib.Path')
    catalog = catalog_time_grouping(catalog, time_period, event_cmap_interval)
    
    # if focal_events is not None:
    #     if isinstance(focal_events, pd.DataFrame):
    #         df_focal = focal_events
    #     elif isinstance(focal_events, Path):
    #         df_focal = pd.read_csv(focal_events)
    #     else:
    #         raise ValueError(f'Format {type(focal_events)} is not supported, using pandas.DataFrame or pathlib.Path')    
        
    #     df_focal = catalog_time_grouping(df_focal, time_period, event_cmap_interval)
    #     df_focal = catalog_filter(
    #         catalog_df=df_focal, h3dd_mode=False, catalog_range=catalog_filter_
    #     )
    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]

    title = f"From {time_period[0]}~{time_period[1]}\ninterval={event_cmap_interval} s; Group num: {len(catalog['time_group'].unique())}"
    get_mapview(region=region, ax=geo_ax, title=title, fontsize=25)
    if plot_vel_on_mapview and vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')
        add_vel_view(
            ax=geo_ax,
            vel_df=vel_df,
            region=region
        )
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=False, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].map(
        lambda x: _get_size(x, magnitude_mapping)
    )

    # station
    # if isinstance(station, pd.DataFrame):
    #     df_station = station
    # elif isinstance(station, Path):
    #     df_station = pd.read_csv(station)
    # else:
    #     raise ValueError(f'Format {type(station)} is not supported, using pandas.DataFrame or pathlib.Path')
    
    # plot_station(
    #     df_station=df_station,
    #     geo_ax=geo_ax,
    #     region=region,
    #     plot_station_name=False,
    #     text_dist=0.005,
    #     zorder=6
    # )

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

    # add all events
    event_num = len(catalog)
    manual_elements.append(
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label=f'Event num: {event_num}',
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k',
        )
    )      

    # plot the color in diff time period
    # unique_times = sorted(catalog['time_group'].unique())
    # time_mapping = {t: i for i, t in enumerate(unique_times)}
    norm = mcolors.Normalize(vmin=catalog['time_group'].min(), vmax=catalog['time_group'].max())   
    colormap = event_cmap
    catalog['color'] = catalog['time_group'].map(lambda x: colormap(norm(x)))
    for group_id, group in catalog.groupby('time_group'):
        min_time = pd.Timestamp(time_period[0])
        color = colormap(norm(group_id))
        # print(f'color: {color}')
        geo_ax.scatter(
            group['longitude'],
            group['latitude'],
            s=group['size'],
            color=group['color'],  # Ensure proper coloring
            alpha=0.5,
            zorder=3
        )
        # manual_elements.append(
        #     Line2D(
        #         [0],
        #         [0],
        #         marker='o',
        #         color='w',
        #         label=f'{min_time}~{min_time + pd.Timedelta(seconds=event_cmap_interval*int(group_id+1))}: {len(group)} events',
        #         markersize=5,
        #         markerfacecolor=color,
        #         markeredgecolor='k',
        #     )
        # ) 
    if background_events is not None:
        if isinstance(background_events, Path):
            df_background_events = pd.read_csv(background_events)
        elif isinstance(background_events, pd.DataFrame):
            df_background_events = background_events
        else:
            raise ValueError(f'{type(background_events)} not supported')
        
        df_background_events.loc[:, 'size'] = df_background_events['magnitude'].map(
            lambda x: _get_size(x, magnitude_mapping)
        )        
        geo_ax.scatter(
            df_background_events['longitude'],
            df_background_events['latitude'],
            s=df_background_events['size'],
            color=background_color,  # Ensure proper coloring
            alpha=0.5,
            zorder=2
        )         
    # Add colorbar to indicate progression
    if use_color_bar:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        cbar_ax = inset_axes(
            geo_ax,  # Parent axis (geo_ax)
            width="3%",  # Width of the colorbar relative to geo_ax
            height="50%",  # Height of the colorbar relative to geo_ax
            loc="lower right",  # Location inside the geo_ax
            bbox_to_anchor=(0.05, 0.05, 0.85, 0.9),  # Adjust position within geo_ax
            bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
            borderpad=0,  # Padding around the inset
        )
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(f"Interval {event_cmap_interval} seconds")
    
    ## plot profile
    if profile_settings:
        _add_azi_free(geo_ax, profile_settings)    

    ## plot eq
    if len(eq_list) > 0:
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
                    label=eq[5],
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

    # plotting gridspec profile
    profile_nums = len(profile_settings)

    # if profile_nums % num_rows == 0:
    #     num_columns = profile_nums // num_rows
    # else:    
    #     num_columns = (profile_nums // num_rows) + 1

    if vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')

    # setting the gridspec besides the mapview
    outer_gs = GridSpecFromSubplotSpec(
        num_rows,
        num_columns,
        subplot_spec=parent_gs[0, 1],
        wspace=0.2,
        hspace=0.3
    )
    for i, profile_dict in enumerate(profile_settings):
        start_time = time.time()
        length = profile_dict['length']
        width = profile_dict['width']
        azimuth = profile_dict['azimuth']
        center = profile_dict['center']
        start_point = profile_dict['start_point']
        profile_name = profile_dict['letter']

        if background_events is not None:
            if isinstance(background_events, Path):
                df_background_events = pd.read_csv(background_events)
            elif isinstance(background_events, pd.DataFrame):
                df_background_events = background_events
            else:
                raise ValueError(f'{type(background_events)} not supported')        
            
            other_events = project_events_to_profile(
                df_catalog=df_background_events,
                length=length, 
                width=width,
                depth_range=depth_range,
                center=center,
                azimuth=azimuth
                )
        profile_events = project_events_to_profile(
            df_catalog=catalog,
            length=length, 
            width=width,
            depth_range=depth_range,
            center=center,
            azimuth=azimuth,
            flex_temporal=True
            )
        row = i % num_rows
        col = i // num_rows
        inner_gs = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer_gs[row, col],
            height_ratios=[1, 3],
            width_ratios=[1, 0.05],
            hspace=0.1
        )

        # Topology axes
        ax1 = fig.add_subplot(inner_gs[0, :-1])
        distances, elevations = extract_topo_profile_with_geod(
            csv_path=topo_csv,
            start_point=start_point,
            azimuth=azimuth+90,
            distance_km=width
            )
        add_topo_profile(
            ax=ax1,
            distances=distances,
            elevations=elevations,
            profile_letter=profile_name,
            ax_alpha=topo_alpha,
            show_ylabel=True if col == 0 else False
        )

        # event axes (x=dist, y=depth)
        ax2 = fig.add_subplot(inner_gs[1, :-1])
        
        max_col = profile_nums // num_rows
        if col == max_col:
            cax = fig.add_subplot(inner_gs[1, -1:])
        elif (profile_nums % num_rows == 0) and col == max_col -1:
            cax = fig.add_subplot(inner_gs[1, -1:])
        else:
            cax = None
        if vel_txt is not None:
            add_vel_profile(
                ax=ax2,
                vel_df=vel_df,
                start_point=start_point,
                mode=vel_mode,
                cax=cax,
                cmap=vel_cmap,
                distance=width,
                azimuth=azimuth + 90
            )
        if background_events is not None:
            add_single_profile(
                other_events,
                ax2,
                temporal=False,
                color=direct_c,
                alpha=0.5,
                width=width,
                depth_range=depth_range
                )
        profile_events.sort_values(by='group', ascending=False, inplace=True)
        for group_id, group in profile_events.groupby('group'):
            color = colormap(norm(group_id))                 
            add_single_profile(
                group,
                ax2,
                temporal=False,
                color=color,
                width=width,
                depth_range=depth_range,
                # only shows x_label when the profile in the bottom or the last one in prifiles.
                show_xlabel=True if row == num_rows - 1 or i == profile_nums -1 else False,
                show_ylabel=True if col == 0 else False,
                size=15,
                text_mode=True
                )
            # if focal_events is not None:
            #     # df_focal['color'] = df_focal['time_group'].map(lambda x: colormap(norm(x)))
            #     df_part = df_focal[df_focal['time_group'] == group_id]
            #     if not df_part.empty:
            #         add_focal_profile(
            #             ax=ax2,
            #             gafocal_df=df_part,
            #             length=length,
            #             width=width,
            #             center=center,
            #             azimuth=azimuth,
            #             focal_size_mapping=focal_size_mapping,
            #             flex_temporal=True,
            #             color=color
            #         )
        if len(eq_list) > 0:
            for eq in eq_list:
                x, _ = proj_epicenter(
                    profile_center=center,
                    epi_center=(eq[0], eq[1]),
                    azimuth=azimuth,
                    width=width,
                    length=length
                )
                if x != 0:
                    _add_epicenter(
                        x=x, y=eq[2], ax=ax2, size=eq[3], alpha=0.7, color=eq[4]
                    )            
        ax2.set_xlim(0, width)#(-width / 2, width / 2)
        ax2.set_ylim(depth_range[0], depth_range[1])
        # ax2.set_xlabel("Distance (km)")
        # ax2.set_ylabel("Depth (km)")
        ax2.invert_yaxis()
        # cax.set_axis_off()                 
        end_time = time.time()
        print(f'Profile {profile_name} finished in {end_time - start_time} seconds')
    if fig_name and fig_dir is not None:
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'{fig_name}.png', dpi=300)
        plt.close()              

def set_profile_map(
    geo_ax: Axes | GeoAxes,
    catalog: pd.DataFrame,
    station: Path | pd.DataFrame | None = None,     
    other_catalog: pd.DataFrame | None = None,
    catalog_filter_: dict | None = None,
    title='',
    eq_list=[],
    plot_eq_name=False,
    plot_station_name=True,
    plot_event_name=False,
    # station_mask=station_mask,
    h3dd_mode=False,
    prefix='',
    temporal=False,
    plot_profile_line=False,
    start_point=(121.08, 23.65),
    plot_vel=False,
    vel_df: pd.DataFrame | None = None, # we should write a function to check the type
    legend_on_map=False,
    fault_lw=4,
    size_mapping=size_mapping,
    direct_zorder=False,
    map_event_alpha=0.5,
    epicenter_outer_width=1.0
):
    """
    catalog should at least contains the lon and lat.

    """
    # base_map
    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]
    get_mapview(region=region, ax=geo_ax, title=title, fault_lw=fault_lw, adjust_grid=True)
    if plot_vel and vel_df is not None:
        add_vel_view(
            ax=geo_ax,
            vel_df=vel_df,
            region=region
        )
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].apply(
        lambda x:_get_size(x, size_mapping)
        )
    # station
    # print(
    #     'Geo Axes Patch Extent:',
    #     geo_ax.get_window_extent(renderer=fig.canvas.get_renderer()),
    # )
    if station is not None:
        if isinstance(station, Path):
            df_station = pd.read_csv(station)
        elif isinstance(station, pd.DataFrame):
            df_station = station
        plot_station(
            df_station=df_station,
            geo_ax=geo_ax,
            region=region,
            plot_station_name=plot_station_name,
            text_dist=0.005,
        )
        manual_elements = [
            Line2D(
                [0],
                [0],
                marker='^',
                color='w',
                label='Seismometer',
                markersize=3,
                markerfacecolor='c',
                markeredgecolor='k',
            ),
        ]
    else:
        manual_elements=[]
    # add all events
    # event_num = len(catalog)
    # manual_elements.append(
    #     Line2D(
    #         [0],
    #         [0],
    #         marker='o',
    #         color='w',
    #         label=f'Event num: {event_num}',
    #         markersize=1,
    #         markerfacecolor='k',
    #         markeredgecolor='k',
    #     )
    # )       
    if 'event_type' in catalog.columns:
        # This part we want to visualize the distribution of event_type.
        # TODO: Modified it as customized.
        color_map = {
            0: 'k',
            1: 'green',
            2: 'purple',
            3: 'gray',
            # 4: 'orange',
            # 5: 'blue',
            # 6: 'red',
        }
        label_map = {
            0: 'Period 1(Main shock - 2024-04-08)',
            1: 'Period 2(2024-04-08 - 2024-04-22)',
            2: 'Period 3(2024-04-22 - 2024-04-29)',
            3: 'Huang and Wang (2022) Relocated events',
            # 4: 'GDMS Catalog',
            # 5: 'GDMS Catalog',
            # 6: 'Associated event',
        }

        for event_type, group in catalog.groupby('event_type'):
            num = len(group)
            group = group.sort_values(by='size', ascending=True)
            geo_ax.scatter(
                group[f'{prefix}longitude'],
                group[f'{prefix}latitude'],
                s=group['size'],
                color=color_map[event_type],
                alpha=map_event_alpha,
                label=f'{label_map[event_type]}: {num}',
                rasterized=True,
                zorder= 7 if direct_zorder else 3 + abs(int(event_type) - 4),
                edgecolors='none'
            )
            if plot_event_name:
                for _, row in group.iterrows():
                    geo_ax.text(
                        x=row[f'{prefix}longitude'],
                        y=row[f'{prefix}latitude'],
                        s=row['event_index'],
                    )

            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label=f'{label_map[event_type]}: {num}',
                    markersize=3,
                    markerfacecolor=color_map[event_type],
                    markeredgecolor='k',
                    alpha=0.8,
                )
            )
        if other_catalog is not None:
            other_catalog.loc[:, 'size'] = other_catalog['magnitude'].apply(_get_size)
            geo_ax.scatter(
                other_catalog['longitude'],
                other_catalog['latitude'],
                s=other_catalog['size'],
                color='gray',
                edgecolors='none',
                alpha=0.5
            )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label='Huang and Wang (2022) Relocated events',
                    alpha=0.8,
                    markersize=3,
                    markerfacecolor='gray',
                    markeredgecolor='k',
                )
            )  

    else:
        if other_catalog is not None:
            other_catalog.loc[:, 'size'] = other_catalog['magnitude'].apply(_get_size)
            geo_ax.scatter(
                other_catalog['longitude'],
                other_catalog['latitude'],
                s=other_catalog['size'],
                color='k',
                edgecolors='none',
                alpha=0.3
            )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label='Huang and Wang (2022) Relocated events',
                    alpha=0.5,
                    markersize=1,
                    markerfacecolor='k',
                    markeredgecolor='k',
                )
            )                
        if temporal:
            colormap = plt.cm.viridis
            norm = mcolors.Normalize(
                vmin=catalog['day'].min(),
                vmax=catalog['day'].max()
                )  # Normalize days to range [0, 1]
            for day, group in catalog.groupby('day'):
                logging.info(f'day {day}: {len(group)}')
                color = colormap(norm(day))
                geo_ax.scatter(
                    group[f'{prefix}longitude'],
                    group[f'{prefix}latitude'],
                    s=group['size'],
                    color=color,
                    alpha=0.5,
                    label=f'Day {day} event num: {len(group)}',
                    zorder=3,
                )
                # Add a colorbar to indicate progression
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            # Create an inset for the colorbar within geo_ax
            cbar_ax = inset_axes(
                geo_ax,  # Parent axis (geo_ax)
                width="3%",  # Width of the colorbar relative to geo_ax
                height="50%",  # Height of the colorbar relative to geo_ax
                loc="lower left",  # Location inside the geo_ax
                bbox_to_anchor=(0.025, 0.025, 0.85, 0.9),  # Adjust position within geo_ax
                bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
                borderpad=0,  # Padding around the inset
            )
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label(f"Day Since {catalog['month'].min()}/{catalog['day'].min()}")
        else:
            geo_ax.scatter(
                catalog[f'{prefix}longitude'],
                catalog[f'{prefix}latitude'],
                s=catalog['size'],
                color=catalog['color'],#'k',
                alpha=0.5,
                zorder=3,
            )         
    # profile
    if plot_profile_line:
        letters = string.ascii_uppercase
        for i, (dis, letter) in enumerate(zip(np.linspace(20, 95, 16), letters[0:16])):
        # for i, (dis, letter) in enumerate(zip(np.linspace(25, 85, 13), letters[1:14])):
        # for i, (dis, letter) in enumerate(zip([30, 40, 50, 65], ['C', 'E', 'G', 'J'])):
            geod = Geod(ellps='WGS84')
            next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], 20, dis*1000)
            text_lon, text_lat, _ = geod.fwd(start_point[0], start_point[1], 20, dis*1000)
            add_profile_line(geo_ax=geo_ax, mode='start', extend_km=80, init_point=(next_lon, next_lat), azimuth=20 + 90, alpha=0.7, c='k')   
            geo_ax.text(x=text_lon -0.05, y=text_lat, s=f"{letter}-{letter}'", weight='bold', fontsize=13)
        start_point = (121.485, 24.667)
        for i, (dis, letter) in enumerate(zip(np.linspace(25, 60, 8), letters[16:24])):
        # for i, (dis, letter) in enumerate(zip(np.linspace(35, 45, 3), letters[18:21])):
        # for i, (dis, letter) in enumerate(zip([40], ['T'])):
            geod = Geod(ellps='WGS84')
            next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], 110, dis*1000)
            text_lon, text_lat, _ = geod.fwd(start_point[0], start_point[1], 110, dis*1000)
            add_profile_line(geo_ax=geo_ax, mode='start', extend_km=120, init_point=(next_lon, next_lat), azimuth=110 + 90, alpha=0.7, c='k')   
            geo_ax.text(x=text_lon, y=text_lat, s=f"{letter}-{letter}'", weight='bold', fontsize=13)                    

    # add main eq
    for eq in eq_list:
        _add_epicenter(
            x=eq[0], y=eq[1], ax=geo_ax, size=eq[3], alpha=0.7, color=eq[4], markeredgewidth=epicenter_outer_width
        )
        if plot_eq_name:
            geo_ax.text(
                x=eq[0]+0.01, y=eq[1]+0.01, s=eq[5], color='k'
            )            
        # manual_elements.append(
        #     Line2D(
        #         [0],
        #         [0],
        #         marker='*',
        #         color='w',
        #         label=eq[3],
        #         markersize=10,
        #         markerfacecolor=eq[4],
        #         markeredgecolor='k',
        #     )
        # )
    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    if legend_on_map:
        geo_ax.legend(
            handles=manual_elements,  # + legend_elements,
            loc='lower right',
            markerscale=2,
            labelspacing=1.5,
            borderpad=1,
            fontsize=10,
            framealpha=0.6,
            alignment='right'
        )
    # geo_ax.add_artist(legend)
    geo_ax.get_xlim()
    geo_ax.get_ylim()
    return geo_ax

def set_multi_catalog(
    geo_ax: GeoAxes,
    prof_ax: Axes,
    prof_direction: str,
    station: Path,
    catalogs: list[pd.DataFrame],
    catalogs_color: list[str],
    catalogs_name: list[str],
    catalog_filter_: dict | None = None,
    title='catalog_comparison',
    eq_list=[],
    plot_station_name=False,
    h3dd_mode=False,
    kde_mode=False,
    kde_min=0.0,
    legend_on_map=False,
    temporal=False,
    depth_set=[0, 60],
    station_mask=station_mask,
    profile_scatter_size=5,
    size_mapping=size_mapping,
    set_alpha=0
):
    """
    catalog should at least contains the lon and lat.
    Supporting catalog in list, allowing multiple catalogs plotting on map for comparison.
    """
    # base_map
    if catalog_filter_ is None: # using first catalog as judges
        region = [
            catalogs[0]['longitude'].min() - 0.2,
            catalogs[0]['longitude'].max() + 0.2,
            catalogs[0]['latitude'].min() - 0.2,
            catalogs[0]['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]

    get_mapview(region=region, ax=geo_ax, title=title, inside=True)

    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])

    # station
    df_station = pd.read_csv(station)

    plot_station(
        df_station=df_station,
        geo_ax=geo_ax,
        region=region,
        plot_station_name=plot_station_name,
        text_dist=0.005,
        zorder=6,
        station_mask=station_mask
    )

    manual_elements = [
        Line2D(
            [0],
            [0],
            marker='^',
            color='w',
            label='Seismometer',
            markersize=4,
            markerfacecolor='c',
            markeredgecolor='k',
        )
    ]
    if prof_direction == 'lat':
        x = 'depth_km'
        y = 'latitude'
        eq_x_idx = 2
        eq_y_idx = 1 
    elif prof_direction == 'lon':
        x = 'longitude'
        y = 'depth_km'
        eq_x_idx = 0
        eq_y_idx = 2
    else:
        raise ValueError(f'No this kind of direction {prof_direction}')
    
    for catalog, color, name in zip(catalogs, catalogs_color, catalogs_name):
        # trim the catalog
        catalog = catalog_filter(
            catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
        )

        # get the size corresponding to magnitude
        catalog.loc[:, 'size'] = catalog['magnitude'].map(lambda x: _get_size(x, size_mapping=size_mapping))

        # add all events
        event_num = len(catalog)

        # kde calculation
        if kde_mode:
            X, Y, density = calc_kde(catalog)
            # geo_ax.imshow(density, extent=region, origin='lower', alpha=0.7, cmap=plt.cm.gist_earth_r, zorder=5, vmin=kde_min)
            levels = np.linspace(kde_min, density.max(), 10)
            contour_lines = geo_ax.contourf(X, Y, density, levels=levels, alpha=0.7, cmap="gist_earth_r", zorder=5)
            geo_ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
        
        # adding temporal feature
        if temporal:
            colormap = plt.cm.viridis
            norm = mcolors.Normalize(
                vmin=catalog['day'].min(),
                vmax=catalog['day'].max()
                )  # Normalize days to range [0, 1]
            for day, group in catalog.groupby('day'):
                logging.info(f'day {day}: {len(group)}')
                color = colormap(norm(day))
                geo_ax.scatter(
                    group['longitude'],
                    group['latitude'],
                    s=group['size'],
                    c=color,
                    alpha=0.5,
                    label=f'Day {day} Event num: {len(group)}',
                    zorder=3,
                )
                manual_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker='.',
                        color='w',
                        label=f'Day {day} Event num: {len(group)}',
                        markersize=7,
                        markerfacecolor=color,
                        markeredgecolor='none',
                    )
                )
                prof_ax.scatter(
                    group[x],
                    group[y],
                    s=5,
                    c=color,
                    edgecolors='none',
                    alpha=0.5,
                    rasterized=True,
                )
                if len(eq_list) > 0:
                    for eq_tuple in eq_list:
                        _add_epicenter(
                            x=eq_tuple[eq_x_idx],
                            y=eq_tuple[eq_y_idx],
                            ax=prof_ax,
                            size=eq_tuple[4],
                            alpha=0.7,
                            color=eq_tuple[3],
                        )                                         
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            # Create an inset for the colorbar within geo_ax
            cbar_ax = inset_axes(
                geo_ax,  # Parent axis (geo_ax)
                width="3%",  # Width of the colorbar relative to geo_ax
                height="50%",  # Height of the colorbar relative to geo_ax
                loc="lower left",  # Location inside the geo_ax
                bbox_to_anchor=(0.025, 0.025, 0.85, 0.9),  # Adjust position within geo_ax
                bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
                borderpad=0,  # Padding around the inset
            )
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label(f"Day Since {catalog['month'].min()}/{catalog['day'].min()}")                                

        else:            
            geo_ax.scatter(
                catalog['longitude'],
                catalog['latitude'],
                s=catalog['size'],
                c=color,
                alpha=0.5,
                label=f'{name} Event num: {event_num}',
                zorder=6,
            )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='.',
                    color='w',
                    label=f'{name} Event',
                    markersize=7,
                    markerfacecolor=color,
                    markeredgecolor='none',
                )
            )
            prof_ax.scatter(
                catalog[x],
                catalog[y],
                s=profile_scatter_size,
                c=color,
                edgecolors='none',
                alpha=0.5,
                rasterized=True,
            )
            if len(eq_list) > 0:
                for eq_tuple in eq_list:
                    _add_epicenter(
                        x=eq_tuple[eq_x_idx],
                        y=eq_tuple[eq_y_idx],
                        ax=prof_ax,
                        size=eq_tuple[4],
                        alpha=0.7,
                        color=eq_tuple[3],
                    )                                        
          
    # add main eq
    if len(eq_list) > 0:
        for eq_tuple in eq_list:
            _add_epicenter(
                x=eq_tuple[0],
                y=eq_tuple[1],
                ax=geo_ax,
                size=eq_tuple[4],
                alpha=0.7,
                color=eq_tuple[3],
            )    
    
    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    if legend_on_map:
        geo_ax.legend(
            handles=manual_elements,  # + legend_elements,
            loc='lower right',
            markerscale=2,
            labelspacing=1.5,
            borderpad=1,
            fontsize=10,
            framealpha=0.6,
        )
    # geo_ax.add_artist(legend)
    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()
    prof_ax.autoscale(tight=True)
    prof_ax.set_xlim(xlim)
    def format_lon(x, pos):
        return f"{float(x):.1f}°E"
    # axes[1, 0].set_ylim([0, catalog['depth_km'].max() + 1])
    prof_ax.set_ylim(depth_set)  # gdms
    prof_ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    prof_ax.invert_yaxis()
    prof_ax.yaxis.tick_right()
    prof_ax.yaxis.set_label_position("right")
    prof_ax.set_ylabel('Depth (km)', fontsize=12, weight='bold')
    # prof_ax.set_xlabel('Longitude', fontsize=12, weight='bold')

def plot_profile_set(
    start_point: tuple,
    azimuth: float | int,
    dis: float | int,
    width: float | int,
    length: float | int,
    letter: str,    
    topo_csv: Path,
    catalog: pd.DataFrame,
    fig: Figure,
    fig_dir: Path | None = None,
    eq_list=[],
    size=3,
    vel_txt: Path | None = None,
    other_catalog: pd.DataFrame | None = None,    
    gafocal_df: pd.DataFrame | None = None,
    view=(), # profile focal
    focal_cmap='Oranges',
    focal_size_mapping=focal_size_mapping, 
    depth_range=(0, 50),
    vel_mode='all',
    vel_cmap='RdYlBu',
    topo_alpha=0,
    quality_mode=False,
):
    # setting the gridspec besides the mapview
    fig.patch.set_alpha(0)
    gs = GridSpec(
        nrows=4,
        ncols=3,
        figure=fig,
        height_ratios=[1, 3, 3, 3],
        width_ratios=[1, 1, 0.05],
        hspace=0.1
    )

    geod = Geod(ellps='WGS84')
    next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], azimuth, dis*1000)
    logging.info(f'Next start point: ({next_lon}, {next_lat})')

    new_lon, new_lat, _ = geod.fwd(next_lon, next_lat, azimuth + 90, (width / 2)*1000)
    new_center = (new_lon, new_lat)
    logging.info(f'Center point: {new_center}')

    # plot background events
    if other_catalog is not None:
        # other_catalog['color'] = 'gray'
        other_events = project_events_to_profile(
            df_catalog=other_catalog,
            length=length, 
            width=width,
            depth_range=depth_range,
            center=new_center,
            azimuth=azimuth,
            )                
    profile_events = project_events_to_profile(
        df_catalog=catalog,
        length=length, 
        width=width,
        depth_range=depth_range,
        center=new_center,
        azimuth=azimuth
        )

    # Topology axes
    ax1 = fig.add_subplot(gs[0, :-1])
    distances, elevations = extract_topo_profile_with_geod(
        csv_path=topo_csv,
        start_point=(next_lon, next_lat),
        azimuth=azimuth+90,
        distance_km=width
        )
    add_topo_profile(
        ax=ax1,
        distances=distances,
        elevations=elevations,
        profile_letter=letter,
        ax_alpha=topo_alpha
    )

    # Velocity profile axes
    if vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')
    if vel_mode == 'all':
        for j, mode in zip([1, 2], ['vpt', 'vpvst']):
            ax = fig.add_subplot(gs[j, :-1])
            cax = fig.add_subplot(gs[j, -1:])

            if vel_txt is not None:
                #TODO: Adding a condition to solve possibly unbound.
                add_vel_profile(
                    ax=ax,
                    vel_df=vel_df,
                    start_point=(next_lon, next_lat),
                    depth_range=depth_range,
                    mode=mode,
                    cax=cax,
                    cmap=vel_cmap,
                    distance=width,
                    azimuth=azimuth + 90
                )
            if other_catalog is not None:
                logging.info(f'Other events: {other_events.columns}')
                logging.info(f"Other evnets color list; {other_events['color'].unique()}")
                add_single_profile(
                    other_events,
                    ax,
                    temporal=False,
                    # c='gray',
                    alpha=0.5,
                    width=width,
                    depth_range=depth_range,
                    show_xlabel=True if (gafocal_df is None and j == 2) else False,
                    show_ylabel=True,
                    size=size
                    )
            for _, group in profile_events.groupby('color', sort=False):                    
                add_single_profile(
                    group,
                    ax,
                    # manual_min=manual_min,
                    # manual_max=manual_max,
                    temporal=False,
                    # c='k',
                    width=width,
                    depth_range=depth_range,
                    show_xlabel=True if (gafocal_df is None and j == 2) else False,
                    show_ylabel=True,
                    size=size
                    )    
            # ax.tick_params(labelbottom=False)
            for eq in eq_list:
                x, _ = proj_epicenter(
                    profile_center=new_center,
                    epi_center=(eq[0], eq[1]),
                    azimuth=azimuth,
                    width=width,
                    length=length
                )
                _add_epicenter(
                    x=x, y=eq[2], ax=ax, size=eq[3], alpha=0.7, color=eq[4]
                )
    # Focal mechanism axes
    if gafocal_df is not None:
        ax = fig.add_subplot(gs[3, :-1])
        cax = fig.add_subplot(gs[3, -1:])
        if other_catalog is not None:
            add_single_profile(
                other_events,
                ax,
                temporal=False,
                # c='gray',
                alpha=0.5,
                width=width,
                depth_range=depth_range,
                size=size
                )                
        for _, group in profile_events.groupby('color', sort=False): 
            add_single_profile(
                group,
                ax,
                # manual_min=manual_min,
                # manual_max=manual_max,
                temporal=False,
                # c='k',
                width=width,
                depth_range=depth_range,
                size=size
                )
        add_focal_profile(
            ax=ax,
            gafocal_df=gafocal_df,
            length=length,
            width=width,
            center=new_center,
            azimuth=azimuth,
            focal_size_mapping=focal_size_mapping,
            quality_mode=quality_mode,
            view=view,
            cmap=focal_cmap,
            cax=cax
        )
        for eq in eq_list:
            x, _ = proj_epicenter(
                profile_center=new_center,
                epi_center=(eq[0], eq[1]),
                azimuth=azimuth,
                width=width,
                length=length,
            )
            _add_epicenter(
                x=x, y=eq[2], ax=ax, size=eq[3], alpha=0.5, color=eq[4] # NOTE: we set alpha=0.3 for solving overlap with beachball
            )            
        # ax.set_xlim(0, target_movement['width'])#(-width / 2, width / 2)
        # ax.set_ylim(depth_range[0], depth_range[1])
        # ax.set_xlabel("Distance (km)", fontsize=20)
        # ax.set_ylabel("Depth (km)", fontsize=20)
        # ax.invert_yaxis()
        #cax.set_axis_off()    
    if fig_dir is not None: 
        plt.tight_layout()
        plt.savefig(fig_dir / f'{letter}_profile.png', dpi=300, bbox_inches='tight')
        plt.close()     

def plot_compare_profile_set(
    start_point: tuple,
    azimuth: float | int,
    dis: float | int,
    width: float | int,
    length: float | int,
    letter: str,    
    topo_csv: Path,
    aq_catalog: pd.DataFrame,
    gdms_catalog: pd.DataFrame,
    fig: Figure,
    fig_dir: Path | None = None,
    eq_list=[],
    size=3,
    vel_txt: Path | None = None,
    other_catalog: pd.DataFrame | None = None,    
    gafocal_df: pd.DataFrame | None = None,
    view=(), # profile focal
    focal_cmap='Oranges',
    focal_size_mapping=focal_size_mapping, 
    depth_range=(0, 50),
    vel_mode='all',
    vel_cmap='RdYlBu',
    topo_alpha=0,
    quality_mode=False,
):
    # setting the gridspec besides the mapview
    fig.patch.set_alpha(0)
    gs = GridSpec(
        nrows=4,
        ncols=3,
        figure=fig,
        height_ratios=[1, 3, 3, 3],
        width_ratios=[1, 1, 0.05],
        hspace=0.1
    )

    geod = Geod(ellps='WGS84')
    next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], azimuth, dis*1000)
    logging.info(f'Next start point: ({next_lon}, {next_lat})')

    new_lon, new_lat, _ = geod.fwd(next_lon, next_lat, azimuth + 90, (width / 2)*1000)
    new_center = (new_lon, new_lat)
    logging.info(f'Center point: {new_center}')
       
    aq_profile_events = project_events_to_profile(
        df_catalog=aq_catalog,
        length=length, 
        width=width,
        depth_range=depth_range,
        center=new_center,
        azimuth=azimuth
        )
    gdms_profile_events = project_events_to_profile(
        df_catalog=gdms_catalog,
        length=length, 
        width=width,
        depth_range=depth_range,
        center=new_center,
        azimuth=azimuth
        )    

    # Topology axes
    ax1 = fig.add_subplot(gs[0, :-1])
    distances, elevations = extract_topo_profile_with_geod(
        csv_path=topo_csv,
        start_point=(next_lon, next_lat),
        azimuth=azimuth+90,
        distance_km=width
        )
    add_topo_profile(
        ax=ax1,
        distances=distances,
        elevations=elevations,
        profile_letter=letter,
        ax_alpha=topo_alpha
    )

    # Velocity profile axes
    vel_df = pd.read_csv(vel_txt, sep='\s+')
    for j, profile_events in zip([1, 2], [aq_profile_events, gdms_profile_events]):
        ax = fig.add_subplot(gs[j, :-1])
        cax = fig.add_subplot(gs[j, -1:])
        add_vel_profile(
            ax=ax,
            vel_df=vel_df,
            start_point=(next_lon, next_lat),
            depth_range=depth_range,
            cax=cax,
            cmap=vel_cmap,
            distance=width,
            azimuth=azimuth + 90
        )

        for _, group in profile_events.groupby('color', sort=False):                    
            add_single_profile(
                group,
                ax,
                # manual_min=manual_min,
                # manual_max=manual_max,
                temporal=False,
                # c='k',
                width=width,
                depth_range=depth_range,
                show_xlabel=True if (gafocal_df is None and j == 2) else False,
                show_ylabel=True,
                size=size
                )    
        # ax.tick_params(labelbottom=False)
        for eq in eq_list:
            x, _ = proj_epicenter(
                profile_center=new_center,
                epi_center=(eq[0], eq[1]),
                azimuth=azimuth,
                width=width,
                length=length
            )
            _add_epicenter(
                x=x, y=eq[2], ax=ax, size=eq[3], alpha=0.7, color=eq[4]
            )

    if fig_dir is not None: 
        plt.tight_layout()
        plt.savefig(fig_dir / f'{letter}_profile.png', dpi=300, bbox_inches='tight')
        plt.close()              
