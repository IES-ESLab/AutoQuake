from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd
from obspy import Stream, UTCDateTime, read

diting_model = (
    Path(__file__).parents[1].resolve() / 'focal_model' / 'DiTingMotionJul.onnx'
)

warnings.filterwarnings('ignore')

def formatting(num):
    """Format number to have leading zero if needed."""
    return f'0{num}' if len(num) == 1 else num


def time_formatting(time: str):
    date_obj = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')

    # Format the datetime object to 'YYYYMMDD'
    return date_obj.strftime('%Y%m%d')


def count_files_in_dir(directory_path: Path):
    if not directory_path.is_dir():
        raise ValueError(f'The path {directory_path} is not a directory.')
    return sum(1 for item in directory_path.iterdir() if item.is_file())


def convert_channel_index(sta_name: str) -> int:
    if sta_name[:1] == 'A':
        return int(sta_name[1:])
    elif sta_name[:1] == 'B':
        return int(f'1{sta_name[1:]}')
    else:
        raise ValueError('wrong format warning: please append the condition')


def get_total_seconds(dt):
    return (dt - dt.normalize()).total_seconds()


def das_demean(data):
    return data - np.mean(data)


def das_detrend(data):
    from scipy.signal import detrend

    return detrend(data)


class DitingMotion:
    def __init__(
        self,
        gamma_picks: Path,
        type_judge: Callable,
        model_path=diting_model,
        output_dir: Path | None = None,
        sac_parent_dir=None,
        h5_parent_dir=None,
        interval=300,
        sampling_rate=100.0,
        need_resample=False, # this is pretty important, make sure whether the sampling rate of your instrument is 100 Hz
    ):
        """## Using DitingMotion to predict the polarity of the P-wave

        ### Args:
            - gamma_picks: Path to the gamma picks file
            - model_path: Path to the model file
            - output_dir: Path to the output directory
            - sac_parent_dir: Path to the sac parent directory if using .SAC files.
            - h5_parent_dir: Path to the h5 parent directory if using .h5 files.
            - interval: Interval of the h5 data to be used for searching specific 300s data.
            - sampling_rate: Sampling rate of the data.
            - type_judge: Function to judge the type of the station through name.
        """
        self.gamma_picks = gamma_picks
        self.model_path = model_path
        self.output_dir = self._check_output(output_dir)
        self.sac_parent_dir = sac_parent_dir
        self.h5_parent_dir = h5_parent_dir
        self.interval = interval
        self.sampling_rate = sampling_rate
        self.type_judge = type_judge
        self.indices = self._get_indices()
        self.need_resample = need_resample
        # self._set_thread_options()
    
    def get_picks(self):
        return self.picks

    def _check_output(self, output):
        if output is not None:
            return output
        else:
            output = Path(__file__).parents[1].resolve() / 'diting_result'
            output.mkdir(parents=True, exist_ok=True)
            return output

    # def _set_thread_options(self):
    #     os.environ['OMP_NUM_THREADS'] = '1'  # Adjust based on your system
    #     os.environ['MKL_NUM_THREADS'] = '1'
    def init_worker(self):
        """
        Initializer for each worker in the pool. Loads the ONNX model once per process.
        """
        global model_session
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        os.environ["OMP_NUM_THREADS"] = "4"  # Adjust based on CPU cores
        os.environ["MKL_NUM_THREADS"] = "4"
        model_session = ort.InferenceSession(self.model_path, sess_options=session_options)

    def _get_indices(self):
        df = pd.read_csv(self.gamma_picks)
        return [x for x in set(df['event_index']) if x != -1]

    def _merge_latest(self, data_path: Path, sta_name: str):
        """
        Merge the latest data from the given data path for the specified station name.
        """
        sac_list = list(data_path.glob(f'*{sta_name}*Z*'))
        if not sac_list:
            logging.info(f'{sta_name} using other component')
            sac_list = list(data_path.glob(f'*{sta_name}*'))
            logging.info(f'comp_list: {sac_list}')
        stream = Stream()
        for sac_file in sac_list:
            st = read(sac_file)
            stream += st
        stream = stream.merge(fill_value='latest')
        return stream

    def seis_get_data(self, sta_name: str, p_arrival_: str, time_window=0.64):
        """
        Get the 1.28 s waveform data for traning.
        """
        ymd = time_formatting(p_arrival_)
        p_arrival = UTCDateTime(p_arrival_)
        if self.sac_parent_dir is not None:
            data_path = self.sac_parent_dir / ymd  # data
        else:
            raise ValueError('Please provide sac_parent_dir')

        try:
            merge_s = time.time()
            st = self._merge_latest(data_path, sta_name)
            merge_e = time.time()
            logging.debug(f"Merging station {sta_name} cost {round(merge_e - merge_s, 2)} s")
        except Exception as e:
            logging.error(
                f'Error during merging: {sta_name}_{e} when p_arrival: {p_arrival}'
            )
            return []
        try:
            preprocess_s = time.time()
            demean_s = time.time()
            st.detrend('demean')
            st.detrend('linear')
            st.taper(0.001)
            demean_e = time.time()
            logging.debug(f'demean + linear + taper cost {round(demean_e - demean_s, 2)} s')
            if self.need_resample:
                resamp_s = time.time()
                st.resample(sampling_rate=self.sampling_rate)
                resamp_e = time.time()
                logging.debug(f'Resampling cost {round(resamp_e - resamp_s, 2)} s')
            trim_s = time.time()
            starttime_trim = p_arrival - time_window
            endtime_trim = p_arrival + time_window
            st.trim(starttime=starttime_trim, endtime=endtime_trim)
            data = st[0].data[0:128]
            trim_e = time.time()
            logging.debug(f'Data trimming cost {round(trim_e - trim_s, 2)} s')
            preprocess_e = time.time()
            logging.debug(f"Preprocessing {sta_name} total cost {round(preprocess_e - preprocess_s, 2)} s")
            return data
        except Exception as e:
            logging.error(
                f'Error exist during process {sta_name}: {e}\np_arrival: {p_arrival}'
            )
            return []

    # for DAS
    def das_for_model(self, data, total_seconds):
        """
        Calculating the index range of the data for the model input.
        """
        event_index = int((total_seconds % self.interval) * self.sampling_rate)
        return data[
            event_index - int(0.64 * self.sampling_rate) : event_index
            + int(0.64 * self.sampling_rate)
        ]

    def das_get_data(self, sta_name: str, p_arrival: str):
        """
        Get the 1.28 s DAS data for traning.
        """
        ymd = time_formatting(p_arrival)
        total_seconds = get_total_seconds(pd.to_datetime(p_arrival))
        index = int(total_seconds // self.interval)
        window = f'{self.interval*index}_{self.interval*(index+1)}.h5'
        if self.h5_parent_dir is not None:
            try:
                file = list((self.h5_parent_dir / f'{ymd}_hdf5').glob(f'*{window}'))[0]
            except IndexError:
                logging.error(f'File not found for window {window}')
                return []
        else:
            raise ValueError('Please provide h5_parent_dir')

        channel_index = convert_channel_index(sta_name)
        try:
            with h5py.File(file, 'r') as fp:
                ds = fp['data']
                # xxx = dir(ds)
                data = ds[channel_index]
        except Exception as e:
            logging.error(f'Error reading {file}: {e}')
            return []
        data = das_demean(data)
        data = das_detrend(data)
        data = self.das_for_model(data, total_seconds)
        return data

    def diting_motion(self, data, row, motion_model):
        """
        Predicting the polarity.
        """
        # create zeros array
        motion_input = np.zeros([1, 128, 2], dtype=np.float32)
        try:
            motion_input[0, :, 0] = data
        except Exception as e:
            logging.error(f'Error: {e} -> row: {row}')
            logging.error(f'data: {data}')
        if np.max(motion_input[0, :, 0]) != 0:
            motion_input[0, :, 0] -= np.mean(
                motion_input[0, :, 0]
            )  # waveform demean -> centralization
            norm_factor = np.std(motion_input[0, :, 0])  # standard deviation
            if norm_factor != 0:
                motion_input[0, :, 0] /= norm_factor  # normalization
                diff_data = np.diff(
                    motion_input[0, 64:, 0]
                )  # difference between 64: data.
                diff_sign_data = np.sign(diff_data)
                motion_input[0, 65:, 1] = diff_sign_data[:]

                # model prediction
                # logging.info(f"starting predict_chunk{i}_event_{counter}")
                ort_inputs = {motion_model.get_inputs()[0].name: motion_input}
                pred_res = motion_model.run(None, ort_inputs)
                # pred_res = motion_model.predict(motion_input)
                pred_fmp = (
                    pred_res[0]  # T0D0
                    + pred_res[1]  # T0D1
                    + pred_res[2]  # T0D2
                    + pred_res[3]  # T0D3
                ) / 4
                pred_cla = (
                    pred_res[4]  # T1D0
                    + pred_res[5]  # T1D1
                    + pred_res[6]  # T1D2
                    + pred_res[7]  # T1D3
                ) / 4
                # logging.info(f"predict done_chunk{i}_event_{counter}")
                # logging.info(pred_fmp, pred_cla)
                if np.argmax(pred_fmp[0, :]) == 1:
                    polarity = 'D'
                    symbol = '-'
                    if np.argmax(pred_cla[0, :]) == 0:
                        sharpness = 'I'
                    elif np.argmax(pred_cla[0, :]) == 1:
                        sharpness = 'E'
                    else:
                        sharpness = 'x'
                elif np.argmax(pred_fmp[0, :]) == 0:
                    polarity = 'U'
                    symbol = '+'
                    if np.argmax(pred_cla[0, :]) == 0:
                        sharpness = 'I'
                    elif np.argmax(pred_cla[0, :]) == 1:
                        sharpness = 'E'
                    else:
                        sharpness = 'x'
                else:
                    polarity = 'x'
                    symbol = ' '
                    if np.argmax(pred_cla[0, :]) == 0:
                        sharpness = 'I'
                    elif np.argmax(pred_cla[0, :]) == 1:
                        sharpness = 'E'
                    else:
                        sharpness = 'x'
                return polarity
            else:
                return 'x'
        else:
            return 'x'

    def process_row(self, row, motion_model) -> pd.DataFrame:
        """
        Processing the row from dataframe, deciding whether the station type is DAS or seismometer by giving function.
        """
        if not self.type_judge(row.station_id):
            data = self.das_get_data(
                sta_name=row.station_id,
                p_arrival=row.phase_time,
            )
        elif self.type_judge(row.station_id):
            data = self.seis_get_data(
                sta_name=row.station_id, p_arrival_=row.phase_time
            )
        diting_s = time.time()
        polarity = self.diting_motion(data, row, motion_model)
        row['polarity'] = polarity
        diting_e = time.time()
        logging.debug(f"DiTing Motion cost {round(diting_e - diting_s, 2)} s")
        return row

    def predict(self, event_index) -> list:
        # Read the gamma_picks CSV file into a DataFrame
        # print(f"predict_{event_index}")
        df_picks = pd.read_csv(self.gamma_picks)

        # Filter DataFrame for the specific event_index and phase_type 'P'
        df_selected_picks = df_picks[
            (df_picks['event_index'] == event_index) & (df_picks['phase_type'] == 'P')
        ].copy()
        # Iterate through the selected picks
        start_s = time.time()
        logging.info(f'event index: {event_index} start processing')

        processed_rows = []
        for _, row in df_selected_picks.iterrows():
            logging.debug(f'----------Station: {row.station_id}----------')
            df_row = self.process_row(row=row, motion_model=model_session)
            logging.debug('---------------------------------')
            if not df_row.empty:
                processed_rows.append(df_row)
        end_s = time.time()
        logging.info(f'event index: {event_index} processing over, cost {round(end_s - start_s, 2)} s')
        return processed_rows

    def run_parallel_predict(self, processes=3):
        self.picks = self.output_dir / 'polarity_picks.csv'
        # if output_csv.exists():
        #     print(f'remove {output_csv}')
        #     output_csv.unlink()
        #     logging.info('remove the current csv file already!')

        # Use a process pool to parallelize the work
        logging.info('Diting motion start.')
        with mp.Pool(processes=processes, initializer=self.init_worker) as pool:
            results = pool.starmap(
                self.predict, [(event_index,) for event_index in self.indices]
            )
        logging.info('Diting motion over.')
        all_picks = [item for sublist in results for item in sublist]
        if all_picks:
            df__result = pd.DataFrame(all_picks)
            df__result.to_csv(self.picks, index=False)
