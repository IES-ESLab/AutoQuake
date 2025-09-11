from __future__ import annotations

import argparse
import logging
logger = logging.getLogger(__name__)
import multiprocessing
import os
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

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

import sys
CONFIG_PATH = Path(__file__).parents[1].resolve() / 'ParamConfig'
sys.path.append(CONFIG_PATH)
from ParamConfig.config_model import PhaseNetConfig
# mp.set_start_method("spawn", force=True)
# log_dir = Path(__file__).parents[1].resolve() / 'log'
matplotlib.use('agg')

class PhaseNet:
    def __init__(
        self,
        args: PhaseNetConfig,
    ):
        """## A class for running EQNet model using pythonic interface.

        Previous way to call the ML model is to use command line interface wrote by argparse.
        This class is a wrapper for the command line interface, which makes it easier to use the ML model in python.

        """
        self.args = args

    def get_picks(self):
        return self.picks

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
                        logger.warning(f'{csv_} is empty, skipping...')
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
        get_station=lambda x: str(x).split('.')[1],
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
                    filename = (
                        tmp[-1].replace('*', '').replace('?', '').replace('.mseed', '')
                    )
                    output = os.path.join(pick_path, filename + '.csv')
                    if len(phase_picks_[i]) == 0:
                        ## keep an empty file for the file with no picks to make it easier to track processed files
                        with open(
                            output, 'a'
                        ):
                            pass
                        continue
                    picks_df = pd.DataFrame(phase_picks_[i])
                    picks_df.sort_values(by=['phase_time'], inplace=True)
                    picks_df.to_csv(
                        output,
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

    def predict(self):
        """
        Now we support the streaming data prediction through fetching data from GDMS and IESWS,
        If you want to start the stream mode, please assign the data_list as the configuration in json format.
        Then the conversion will automatically happen in the SeismicTraceIterableDataset creation.
        """
        args = self.args
        pick_path = str(args.pick_path)
        event_path = str(args.event_path)
        figure_path = str(args.figure_path)
        
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
                data_path=str(args.data_path),
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
                data_path=str(args.data_path),
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
                pick_path=str(pick_path),
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

def parallel_run_phasenet(configs: list, workers: int = 4):
    """
    Run multiple PhaseNet instances in parallel using multiprocessing.

    Args:
        configs (list[PhaseNetConfig]): List of configurations for PhaseNet.
        num_workers (int): Number of parallel workers to use.
    """
    def run_single_instance(config):
        phasenet = PhaseNet(config)
        phasenet.predict()

    with mp.Pool(processes=workers) as pool:
        pool.map(run_single_instance, configs)  