from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.utils.data
from tqdm import tqdm

from .EQNet import utils
from .EQNet.eqnet import models  # noqa: F401
from .EQNet.eqnet.data import DASIterableDataset, SeismicTraceIterableDataset
from .EQNet.eqnet.utils import (
    detect_peaks,
    extract_events,
    extract_picks,
    merge_events,
    merge_patch,
    merge_picks,
    plot_das,
    plot_phasenet,
    plot_phasenet_plus,
)

# mp.set_start_method("spawn", force=True)
log_dir = Path(__file__).parents[1].resolve() / 'log'
matplotlib.use('agg')
logger = logging.getLogger()


class PhaseNet:
    def __init__(
        self,
        data_parent_dir: Path,
        start_ymd: str,
        end_ymd: str,
        pz_dir: Path | None = None,
        data_list: str | None = None,
        result_path: Path | None = None,
        hdf5_file: str | None = None,
        prefix='',
        format='SAC',  # h5 for DAS.
        dataset='das', # This is the accompanning attributes if format is h5.
        model='phasenet',
        resume='',
        backbone='unet',
        phases=['P', 'S'],
        device='cpu',
        workers=0,
        batch_size=1,
        use_deterministic_algorithms=True,
        amp=True,
        world_size=1,
        dist_url='env://',
        plot_figure=False,
        min_prob=0.3,
        add_polarity=False,
        add_event=True,
        sampling_rate=100.0,
        highpass_filter=0.0,
        response_path: str | None = None,
        response_xml: str | None = None,
        subdir_level=0,
        cut_patch=False,  # DAS
        nt=1024 * 20,  # TODO: modified as MiDAS default
        nx=1024 * 5,
        resample_time=False,  # DAS
        resample_space=False,  # DAS
        system: str | None = None,  # DAS
        location: str | None = None,  # DAS
        skip_existing=False,  # DAS
    ):
        """## A class for running EQNet model using pythonic interface.

        Previous way to call the ML model is to use command line interface wrote by argparse.
        This class is a wrapper for the command line interface, which makes it easier to use the ML model in python.

        ### Args:
            - data_parent_dir (Path): Path to the parent directory of the data.
            - start_ymd (str): Start date in YYYYMMDD format.
            - end_ymd (str): End date in YYYYMMDD format.
            - pz_dir (Path | None, optional): Path to the PZ directory, integrated with newly phasenet. If None, no PZ directory will be used. Defaults to None.
            - data_list (str | None, optional): Path to the data list file. If None, the data list will be generated based on the start and end dates. Defaults to None.
            - result_path (Path | None, optional): Path to the directory where the results will be saved. If None, the results will be saved in the parent directory of current directory. Defaults to None.
            - hdf5_file (str | None, optional): Path of hdf5 file for training. If None, the hdf5 file will be generated based on the data list. Defaults to None.
            - prefix (str, optional): Prefix for the file name. Defaults to ''.
            - format (str, optional): Format of the data, select h5 or SAC. Defaults to 'SAC'.
            - dataset (str, optional): This is for if format is h5, skipping this if using seismic trace. Defaults to 'das'.
            - model (str, optional): Model to use, EQnet provides phasenet, phasenet_plus, phasenet_das. Defaults to 'phasenet'.
            - resume (str, optional): Path to the checkpoint file. Defaults to ''.
            - backbone (str, optional): Backbone to use. Defaults to 'unet'.
            - phases (list, optional): Phases to detect. Defaults to ['P', 'S'].
            - device (str, optional): Device to use, if you have GPU please type 'cuda'. Defaults to 'cpu'.
            - workers (int, optional): Number of workers to use. Defaults to 0.
            - batch_size (int, optional): Batch size to use. Defaults to 1.
            - use_deterministic_algorithms (bool, optional): Whether to use deterministic algorithms. Defaults to True.
            - amp (bool, optional): Whether to use automatic mixed precision. Defaults to True.
            - world_size (int, optional): Number of GPUs to use. Defaults to 1.
            - dist_url (str, optional): URL used to set up distributed training. Defaults to 'env://'.
            - plot_figure (bool, optional): Whether to plot the figure. Defaults to False.
            - min_prob (float, optional): Minimum probability to use. Defaults to 0.3.
            - add_polarity (bool, optional): Whether to add polarity. Defaults to False.
            - add_event (bool, optional): Whether to use event information. Defaults to True.
            - sampling_rate (float, optional): Sampling rate. Defaults to 100.0.
            - highpass_filter (float): Highpass filter. If 0.0, no highpass filter will be used. Defaults to 0.0.
            - response_path (str | None, optional): Path to the response file. If None, no response will be used. Defaults to None.
            - subdir_level (int, optional): Number of subdirectories to use. Defaults to 0.
            - cut_patch (bool, optional): Whether to cut patch. Defaults to False.
            - nt (int, optional): Number of time samples. Defaults to 1024 * 20.
            - nx (int, optional): Number of spatial samples. Defaults to 1024 * 5.
            - resample_time (bool, optional): Whether to resample time. Defaults to False.
            - resample_space (bool, optional): Whether to resample space. Defaults to False.
            - system (str | None, optional): System to use. If None, no system will be used. Defaults to None.
            - location (str | None, optional): The name of systems at location. If None, no location will be used. Defaults to None.
            - skip_existing (bool, optional): Whether to skip existing files. Defaults to False.
        """
        self.data_parent_dir = data_parent_dir
        self.start_ymd = start_ymd
        self.end_ymd = end_ymd
        self.format = format
        self.data_list = data_list
        self.result_path = self._check_result_dir(result_path)
        self.hdf5_file = hdf5_file
        self.prefix = prefix
        self.dataset = dataset
        self.model = model
        self.resume = resume
        self.backbone = backbone
        self.phases = phases
        self.device = device
        self.workers = workers
        self.batch_size = batch_size
        self.use_deterministic_algorithms = use_deterministic_algorithms
        self.amp = amp
        self.world_size = world_size
        self.dist_url = dist_url
        self.plot_figure = plot_figure
        self.min_prob = min_prob
        self.add_polarity = add_polarity
        self.add_event = add_event
        self.sampling_rate = sampling_rate
        self.highpass_filter = highpass_filter
        self.pz_dir = pz_dir
        self.response_path = response_path
        self.response_xml = response_xml
        self.subdir_level = subdir_level
        self.cut_patch = cut_patch
        self.nt = nt
        self.nx = nx
        self.resample_time = resample_time
        self.resample_space = resample_space
        self.system = system
        self.location = location
        self.skip_existing = skip_existing
        # Initialize instance variables based on parsed self.args
        self.input_to_args()


    def get_picks(self):
        return self.picks

    def _check_result_dir(self, result_path: Path | None) -> Path:
        """## Check whether result directory exists or not

        ### Args:
            - result_path (str | None, optional): Path to the result directory. If None, the default path will be used. Defaults to None.
        """
        if result_path is None:
            return Path(__file__).parents[1].resolve() / 'phasenet_result'
        else:
            return result_path

    def date_range(self):
        """## Get date range from start_ymd to end_ymd

        This step is to generate a list of dates in the format of '%Y%m%d' for parallel processing.

        Reutrn:
            - date_list (list): List of dates in the format of '%Y%m%d'
        """
        start_date = datetime.strptime(self.start_ymd, '%Y%m%d')
        end_date = datetime.strptime(self.end_ymd, '%Y%m%d')
        delta = end_date - start_date
        date_list = []
        for i in range(delta.days + 1):
            date = start_date + timedelta(days=i)
            date_list.append(date.strftime('%Y%m%d'))
        if self.model != 'phasenet_das':
            available_date = os.listdir(str(self.data_parent_dir))
            date_list = [date for date in date_list if date in available_date]

        if len(date_list) == 1:
            self.dir_name = date_list[0]
        else:
            self.dir_name = f'{date_list[0]}_{date_list[-1]}'
        logging.info(f'Using {date_list} as predicting range')
        return date_list

    def input_to_args(self):
        """## Converting input arguments to argparse.Namespace.

        Converting it in a args_list for parallel processing.
        """
        self.args_list = []
        self.date_list = self.date_range()
        for date in self.date_list:
            data_path = next(self.data_parent_dir.glob(f'*{date}*'))
            args = argparse.Namespace(
                data_path=str(data_path),
                data_list=self._check_data_list(self.data_list, data_path),
                ymd=date,
                result_path=str(self.result_path),
                hdf5_file=self.hdf5_file,
                prefix=self.prefix,
                format=self.format,
                dataset=self.dataset,
                model=self.model,
                resume=self.resume,
                backbone=self.backbone,
                phases=self.phases,
                device=self.device,
                workers=self.workers,
                batch_size=self.batch_size,
                use_deterministic_algorithms=self.use_deterministic_algorithms,
                amp=self.amp,
                world_size=self.world_size,
                dist_url=self.dist_url,
                plot_figure=self.plot_figure,
                min_prob=self.min_prob,
                add_polarity=self.add_polarity,
                add_event=self.add_event,
                sampling_rate=self.sampling_rate,
                highpass_filter=self.highpass_filter,
                pz_dir=self.pz_dir,
                response_path=self.response_path,
                response_xml=self.response_xml,
                subdir_level=self.subdir_level,
                cut_patch=self.cut_patch,
                nt=self.nt,
                nx=self.nx,
                resample_time=self.resample_time,
                resample_space=self.resample_space,
                system=self.system,
                location=self.location,
                skip_existing=self.skip_existing,
            )
            self.args_list.append(args)

    def _check_data_list(self, data_list, data_path: Path):
        """## Check whether data_list exists or not

        Default to class is to set the data_list=None that we can generate it from data_path.
        """
        if self.format != 'h5':
            if data_list is None:
                all_list = list(data_path.glob(f'*{self.format}'))
                data_list = []
                for i in all_list:
                    fname = f"{str(i).split('.D.')[0][:-1]}*"
                    if fname not in data_list:
                        data_list.append(fname)
            else:
                with open(data_list) as f:
                    data_list = f.read().splitlines()
        else:
            pass
        return data_list

    @staticmethod
    def concat_picks(date_list: list, result_path: Path, model: str, dir_name: str):
        """## Concatenate daily picks to a single csv file.

        There exists 2 scenario:
            1. single day: generating picks.csv in ymd directory.
            2. multiple days: generating picks.csv in {start_ymd}_{end_ymd} directory.

        """
        concat_list = []
        for date in date_list:
            # TODO: What about DAS data?
            picks_path = result_path / f'picks_{model}'
            if model == 'phasenet':
                df = pd.read_csv(
                    picks_path / f'{date}.csv'
                )  # because phasenet automatic combined the daily picks outside the dir.
                concat_list.append(df)
            elif model == 'phasenet_das':
                csv_list = list((picks_path / date).glob('*.csv'))
                for csv_ in csv_list:
                    try:
                        df = pd.read_csv(csv_)
                    except pd.errors.EmptyDataError:
                        logging.warning(f'{csv_} is empty, skipping...')
                        continue
                    # Converting the channel_index into a string-like station name.
                    df['station_id'] = df['channel_index'].astype(str).str.zfill(4)
                    df['station_id'] = df['station_id'].apply(
                        lambda x: f'A{x[1:]}' if x[0] == '0' else f'B{x[1:]}'
                    )
                    concat_list.append(df)

        result = pd.concat(concat_list)
        date_dir = result_path / f'picks_{model}' / dir_name
        date_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(
            date_dir / 'picks.csv',
            index=False,
        )

    @staticmethod
    def picking_filter(picks: Path, filt_station: Path, output_dir: Path | None = None):
        """## filtering the picks through station list"""
        df_sta = pd.read_csv(filt_station)
        filt_sta_list = df_sta['station'].tolist()
        df_picks = pd.read_csv(picks)
        df_picks = df_picks[df_picks['station_id'].isin(filt_sta_list)]
        if output_dir is None:
            output_dir = picks.parent
        df_picks.to_csv(output_dir / 'filt_picks.csv', index=False)

    @staticmethod
    def picks_check(
        picks: Path,
        station: Path,
        get_station=lambda x: x.split('.')[1],
        output_dir=None,
    ):
        df_picks = pd.read_csv(picks)
        df_sta = pd.read_csv(station)
        df_picks['station_id'] = df_picks['station_id'].map(get_station)
        df_picks = df_picks[df_picks['station_id'].isin(df_sta['station'])]
        if output_dir is None:
            output_dir = picks.parent
        df_picks.to_csv(output_dir / 'check_picks.csv', index=False)

    def postprocess(self, meta, output, polarity_scale=1, event_scale=16):
        nt, nx = meta['nt'], meta['nx']
        data = meta['data'][:, :, :nt, :nx]
        # data = moving_normalize(data)
        meta['data'] = data
        if 'phase' in output:
            output['phase'] = output['phase'][:, :, :nt, :nx]
        if 'polarity' in output:
            output['polarity'] = output['polarity'][
                :, :, : (nt - 1) // polarity_scale + 1, :nx
            ]
        if 'event_center' in output:
            output['event_center'] = output['event_center'][
                :, :, : (nt - 1) // event_scale + 1, :nx
            ]
        if 'event_time' in output:
            output['event_time'] = output['event_time'][
                :, :, : (nt - 1) // event_scale + 1, :nx
            ]
        return meta, output

    def pred_phasenet(self, args, model, data_loader, pick_path, figure_path):
        model.eval()
        ctx = (
            nullcontext()
            if args.device == 'cpu'
            else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
        )
        with torch.inference_mode():
            for meta in tqdm(data_loader, desc='Predicting', total=len(data_loader)):
                with ctx:
                    output = model(meta)
                    meta, output = self.postprocess(meta, output)
                if 'phase' in output:
                    phase_scores = torch.softmax(
                        output['phase'], dim=1
                    )  # [batch, nch, nt, nsta]
                    topk_phase_scores, topk_phase_inds = detect_peaks(
                        phase_scores, vmin=args.min_prob, kernel=128
                    )
                    phase_picks_ = extract_picks(
                        topk_phase_inds,
                        topk_phase_scores,
                        file_name=meta['file_name'],
                        station_id=meta['station_id'],
                        begin_time=meta['begin_time'] if 'begin_time' in meta else None,
                        begin_time_index=meta['begin_time_index']
                        if 'begin_time_index' in meta
                        else None,
                        dt=meta['dt_s'] if 'dt_s' in meta else 0.01,
                        vmin=args.min_prob,
                        phases=args.phases,
                        waveform=meta['data'],
                        window_amp=[10, 5],  # s
                    )

                for i in range(len(meta['file_name'])):
                    tmp = meta['file_name'][i].split('/')
                    parent_dir = '/'.join(tmp[-args.subdir_level - 1 : -1])
                    filename = (
                        tmp[-1].replace('*', '').replace('?', '').replace('.mseed', '')
                    )

                    if not os.path.exists(os.path.join(pick_path, parent_dir)):
                        os.makedirs(os.path.join(pick_path, parent_dir), exist_ok=True)
                    if len(phase_picks_[i]) == 0:
                        ## keep an empty file for the file with no picks to make it easier to track processed files
                        with open(
                            os.path.join(pick_path, parent_dir, filename + '.csv'), 'a'
                        ):
                            pass
                        continue
                    picks_df = pd.DataFrame(phase_picks_[i])
                    picks_df.sort_values(by=['phase_time'], inplace=True)
                    picks_df.to_csv(
                        os.path.join(pick_path, parent_dir, filename + '.csv'),
                        index=False,
                    )

                if args.plot_figure:
                    # meta["waveform_raw"] = meta["waveform"].clone()
                    # meta["data"] = moving_normalize(meta["data"])
                    plot_phasenet(
                        meta,
                        phase_scores.cpu(),
                        file_name=meta['file_name'],
                        dt=meta['dt_s'] if 'dt_s' in meta else torch.tensor(0.01),
                        figure_dir=figure_path,
                    )

        ## merge picks
        if args.distributed:
            torch.distributed.barrier()
            if utils.is_main_process():
                merge_picks(pick_path)
        else:
            merge_picks(pick_path)
        return 0

    def pred_phasenet_plus(
        self, args, model, data_loader, pick_path, event_path, figure_path
    ):
        model.eval()
        ctx = (
            nullcontext()
            if args.device in ['cpu', 'mps']
            else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
        )
        with torch.inference_mode():
            for meta in tqdm(data_loader, desc='Predicting', total=len(data_loader)):
                with ctx:
                    output = model(meta)
                    meta, output = self.postprocess(meta, output)

                dt = (
                    meta['dt_s']
                    if 'dt_s' in meta
                    else [torch.tensor(0.01)] * len(meta['data'])
                )

                if 'phase' in output:
                    phase_scores = torch.softmax(
                        output['phase'], dim=1
                    )  # [batch, nch, nt, nsta]
                    if 'polarity' in output:
                        # polarity_scores = torch.sigmoid(output["polarity"])
                        polarity_scores = torch.softmax(output['polarity'], dim=1)
                    topk_phase_scores, topk_phase_inds = detect_peaks(
                        phase_scores,
                        vmin=args.min_prob,
                        kernel=128,
                        dt=dt.min().item(),
                    )
                    phase_picks = extract_picks(
                        topk_phase_inds,
                        topk_phase_scores,
                        file_name=meta['file_name'],
                        station_id=meta['station_id'],
                        begin_time=meta['begin_time'] if 'begin_time' in meta else None,
                        begin_time_index=meta['begin_time_index']
                        if 'begin_time_index' in meta
                        else None,
                        dt=dt,
                        vmin=args.min_prob,
                        phases=args.phases,
                        polarity_score=polarity_scores,
                        waveform=meta['data'],
                        window_amp=[10, 5],  # s
                    )

                if ('event_center' in output) and (output['event_center'] is not None):
                    event_center = torch.sigmoid(output['event_center'])
                    event_time = output['event_time']
                    topk_event_scores, topk_event_inds = detect_peaks(
                        event_center,
                        vmin=args.min_prob,
                        kernel=16,
                        dt=dt.min().item() * 16.0,
                    )
                    event_detects = extract_events(
                        topk_event_inds,
                        topk_event_scores,
                        file_name=meta['file_name'],
                        station_id=meta['station_id'],
                        begin_time=meta['begin_time'] if 'begin_time' in meta else None,
                        begin_time_index=meta['begin_time_index']
                        if 'begin_time_index' in meta
                        else None,
                        dt=dt,
                        vmin=args.min_prob,
                        event_time=event_time,
                    )

                for i in range(len(meta['file_name'])):
                    tmp = meta['file_name'][i].split('/')
                    parent_dir = '/'.join(tmp[-args.subdir_level - 1 : -1])
                    filename = (
                        tmp[-1].replace('*', '').replace('?', '').replace('.mseed', '')
                    )

                    if not os.path.exists(os.path.join(pick_path, parent_dir)):
                        os.makedirs(os.path.join(pick_path, parent_dir), exist_ok=True)
                    if len(phase_picks[i]) == 0:
                        ## keep an empty file for the file with no picks to make it easier to track processed files
                        with open(
                            os.path.join(pick_path, parent_dir, filename + '.csv'), 'a'
                        ):
                            pass
                        continue
                    picks_df = pd.DataFrame(phase_picks[i])
                    picks_df.sort_values(by=['phase_time'], inplace=True)
                    picks_df.to_csv(
                        os.path.join(pick_path, parent_dir, filename + '.csv'),
                        index=False,
                    )

                    if ('event_center' in output) and ('event_time' in output):
                        if not os.path.exists(os.path.join(event_path, parent_dir)):
                            os.makedirs(
                                os.path.join(event_path, parent_dir), exist_ok=True
                            )
                        if len(event_detects[i]) == 0:
                            with open(
                                os.path.join(event_path, parent_dir, filename + '.csv'),
                                'a',
                            ):
                                pass
                            continue
                        events_df = pd.DataFrame(event_detects[i])
                        events_df.sort_values(by=['event_time'], inplace=True)
                        events_df.to_csv(
                            os.path.join(event_path, parent_dir, filename + '.csv'),
                            index=False,
                        )

                if args.plot_figure:
                    plot_phasenet_plus(
                        meta,
                        phase_scores.cpu().float(),
                        polarity_scores.cpu().float()
                        if polarity_scores is not None
                        else None,
                        event_center.cpu().float()
                        if 'event_center' in output
                        else None,
                        event_time.cpu().float() if 'event_time' in output else None,
                        phase_picks=phase_picks,
                        event_detects=event_detects,
                        file_name=meta['file_name'],
                        dt=dt,
                        figure_dir=figure_path,
                    )

        ## merge picks
        if args.distributed:
            torch.distributed.barrier()
            if utils.is_main_process():
                merge_picks(pick_path)
                merge_events(event_path)
        else:
            merge_picks(pick_path)
            merge_events(event_path)
        return 0

    def pred_phasenet_das(self, args, model, data_loader, pick_path, figure_path):
        model.eval()
        ctx = (
            nullcontext()
            if args.device == 'cpu'
            else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
        )
        with torch.inference_mode():
            # for meta in metric_logger.log_every(data_loader, 1, header):
            for meta in tqdm(data_loader, desc='Predicting', total=len(data_loader)):
                with ctx:
                    output = model(meta)

                meta, output = self.postprocess(meta, output)
                scores = torch.softmax(output['phase'], dim=1)  # [batch, nch, nt, nsta]
                topk_scores, topk_inds = detect_peaks(
                    scores, vmin=args.min_prob, kernel=21
                )

                picks_ = extract_picks(
                    topk_inds,
                    topk_scores,
                    file_name=meta['file_name'],
                    begin_time=meta['begin_time'] if 'begin_time' in meta else None,
                    begin_time_index=meta['begin_time_index']
                    if 'begin_time_index' in meta
                    else None,
                    begin_channel_index=meta['begin_channel_index']
                    if 'begin_channel_index' in meta
                    else None,
                    dt=meta['dt_s'] if 'dt_s' in meta else 0.01,
                    vmin=args.min_prob,
                    phases=args.phases,
                )

                for i in range(len(meta['file_name'])):
                    tmp = meta['file_name'][i].split('/')
                    parent_dir = '/'.join(tmp[-args.subdir_level - 1 : -1])
                    filename = tmp[-1].replace('*', '').replace(f'.{args.format}', '')
                    if not os.path.exists(os.path.join(pick_path, parent_dir)):
                        os.makedirs(os.path.join(pick_path, parent_dir), exist_ok=True)

                    if len(picks_[i]) == 0:
                        ## keep an empty file for the file with no picks to make it easier to track processed files
                        with open(
                            os.path.join(pick_path, parent_dir, filename + '.csv'), 'a'
                        ):
                            pass
                        continue
                    picks_df = pd.DataFrame(picks_[i])
                    picks_df['channel_index'] = picks_df['station_id'].apply(
                        lambda x: int(x)
                    )
                    picks_df.sort_values(
                        by=['channel_index', 'phase_index'], inplace=True
                    )
                    picks_df.to_csv(
                        os.path.join(pick_path, parent_dir, filename + '.csv'),
                        columns=[
                            'channel_index',
                            'phase_index',
                            'phase_time',
                            'phase_score',
                            'phase_type',
                        ],
                        index=False,
                    )

                if args.plot_figure:
                    plot_das(
                        meta['data'].cpu().float(),
                        scores.cpu().float(),
                        picks=picks_,
                        phases=args.phases,
                        file_name=meta['file_name'],
                        begin_time_index=meta['begin_time_index']
                        if 'begin_time_index' in meta
                        else None,
                        begin_channel_index=meta['begin_channel_index']
                        if 'begin_channel_index' in meta
                        else None,
                        dt=meta['dt_s'] if 'dt_s' in meta else torch.tensor(0.01),
                        dx=meta['dx_m'] if 'dx_m' in meta else torch.tensor(10.0),
                        figure_dir=figure_path,
                    )

        if args.distributed:
            torch.distributed.barrier()
            if args.cut_patch and utils.is_main_process():
                merge_patch(
                    pick_path, pick_path.rstrip('_patch'), return_single_file=False
                )
        else:
            if args.cut_patch:
                merge_patch(
                    pick_path, pick_path.rstrip('_patch'), return_single_file=False
                )

        return 0

    def predict(self, args):
        result_path = args.result_path
        if args.cut_patch:
            pick_path = os.path.join(result_path, f'picks_{args.model}_patch', args.ymd)
            event_path = os.path.join(
                result_path, f'events_{args.model}_patch', args.ymd
            )
            figure_path = os.path.join(
                result_path, f'figures_{args.model}_patch', args.ymd
            )
        else:
            pick_path = os.path.join(result_path, f'picks_{args.model}', args.ymd)
            event_path = os.path.join(result_path, f'events_{args.model}', args.ymd)
            figure_path = os.path.join(result_path, f'figures_{args.model}', args.ymd)
        if not os.path.exists(result_path):
            utils.mkdir(result_path)
        if not os.path.exists(pick_path):
            utils.mkdir(pick_path)
        if not os.path.exists(event_path):
            utils.mkdir(event_path)
        if not os.path.exists(figure_path):
            utils.mkdir(figure_path)

        utils.init_distributed_mode(args)

        if args.distributed:
            rank = utils.get_rank()
            world_size = utils.get_world_size()
        else:
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                rank = int(os.environ['RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
            else:
                rank = 0
                world_size = 1

        device = torch.device(args.device)
        dtype = (
            'bfloat16'
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else 'float16'
        )
        ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }[dtype]
        args.dtype, args.ptdtype = dtype, ptdtype
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        if args.use_deterministic_algorithms:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True

        if args.model in ['phasenet', 'phasenet_plus']:
            dataset = SeismicTraceIterableDataset(
                data_path=args.data_path,
                data_list=args.data_list,
                hdf5_file=args.hdf5_file,
                prefix=args.prefix,
                format=args.format,
                dataset=args.dataset,
                training=False,
                sampling_rate=args.sampling_rate,
                highpass_filter=args.highpass_filter,
                pz_dir=args.pz_dir,
                response_path=args.response_path,
                response_xml=args.response_xml,
                cut_patch=args.cut_patch,
                resample_time=args.resample_time,
                system=args.system,
                nx=args.nx,
                nt=args.nt,
                rank=rank,
                world_size=world_size,
            )
            sampler = None
        elif args.model == 'phasenet_das':
            dataset = DASIterableDataset(
                data_path=args.data_path,
                data_list=args.data_list,
                format=args.format,
                nx=args.nx,
                nt=args.nt,
                training=False,
                system=args.system,
                cut_patch=args.cut_patch,
                highpass_filter=args.highpass_filter,
                resample_time=args.resample_time,
                resample_space=args.resample_space,
                skip_existing=args.skip_existing,
                pick_path=pick_path,
                subdir_level=args.subdir_level,
                rank=rank,
                world_size=world_size,
            )
            sampler = None
        else:
            raise ('Unknown model')  # type: ignore
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=min(args.workers, mp.cpu_count()),
            collate_fn=None,
            drop_last=False,
        )
        model = models.__dict__[args.model].build_model(
            backbone=args.backbone,
            in_channels=1,
            out_channels=(len(args.phases) + 1),
        )
        # logger.info(f'Model:\n{model}')

        model.to(device)
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            # model.load_state_dict(checkpoint["model"], strict=True)
            # print("Loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint["epoch"]))
        else:
            if args.model == 'phasenet':
                if args.location is None:
                    model_url = 'https://github.com/AI4EPS/models/releases/download/PhaseNet-v1/model_99.pth'
            elif args.model == 'phasenet_plus':
                if args.location is None:
                    model_url = 'https://github.com/AI4EPS/models/releases/download/PhaseNet-Plus-v1/model_99.pth'
                elif args.location == 'LCSN':
                    model_url = 'https://github.com/AI4EPS/models/releases/download/PhaseNet-Plus-LCSN/model_99.pth'
            elif args.model == 'phasenet_das':
                if args.location is None:
                    # model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-v0/PhaseNet-DAS-v0.pth"
                    model_url = 'https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-v1/PhaseNet-DAS-v1.pth'
                elif args.location == 'forge':
                    model_url = 'https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-ConvertedPhase/model_99.pth'
                else:
                    raise ('Missing pretrained model for this location')  # type: ignore
            else:
                raise
            checkpoint = torch.hub.load_state_dict_from_url(
                model_url,
                model_dir=f'./model_{args.model}',
                progress=True,
                check_hash=True,
                map_location='cpu',
            )

            ## load model from wandb
            # if utils.is_main_process():
            #     with wandb.init() as run:
            #         artifact = run.use_artifact(model_url, type="model")
            #         artifact_dir = artifact.download()
            #     checkpoint = torch.load(glob(os.path.join(artifact_dir, "*.pth"))[0], map_location="cpu")
            #     model.load_state_dict(checkpoint["model"], strict=True)

        model_without_ddp = model
        if args.distributed:
            torch.distributed.barrier()
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
            model_without_ddp = model.module
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        if args.model == 'phasenet':
            self.pred_phasenet(args, model, data_loader, pick_path, figure_path)

        if args.model == 'phasenet_plus':
            self.pred_phasenet_plus(
                args, model, data_loader, pick_path, event_path, figure_path
            )

        if args.model == 'phasenet_das':
            self.pred_phasenet_das(args, model, data_loader, pick_path, figure_path)
        # return os.path.join(pick_path, 'picks.csv')

    def run_predict(self, processes=3):
        logging.basicConfig(
            filename=log_dir / 'phasenet.log',
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        """## Main function to run."""
        with multiprocessing.Pool(processes=processes) as pool:
            pool.map(self.predict, self.args_list)
        self.concat_picks(
            date_list=self.date_list,
            result_path=self.result_path,
            model=self.model,
            dir_name=self.dir_name,
        )
        self.picks = self.result_path / f'picks_{self.model}' / self.dir_name / 'picks.csv'
