from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd
from obspy import Stream, UTCDateTime, read
from scipy.signal import detrend

logger = logging.getLogger(__name__)

DITING_MODEL_PATH = (
    Path(__file__).parents[1].resolve() / 'focal_model' / 'DiTingMotionJul.onnx'
)

# Module-level model session for worker processes
_worker_model: ort.InferenceSession | None = None
_worker_config: dict | None = None


def _time_to_ymd(time_str: str) -> str:
    """Convert ISO timestamp to YYYYMMDD format."""
    date_obj = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
    return date_obj.strftime('%Y%m%d')


def _get_total_seconds(dt) -> float:
    """Get total seconds from midnight for a pandas datetime."""
    return (dt - dt.normalize()).total_seconds()


def _convert_channel_index(sta_name: str) -> int:
    """Convert DAS station name to channel index."""
    if sta_name[:1] == 'A':
        return int(sta_name[1:])
    elif sta_name[:1] == 'B':
        return int(f'1{sta_name[1:]}')
    else:
        raise ValueError(f'Unknown station format: {sta_name}')


def _init_worker(config_dict: dict) -> None:
    """Initialize worker process with ONNX model and configuration.

    Args:
        config_dict: Serializable configuration dictionary.
    """
    global _worker_model, _worker_config

    _worker_config = config_dict

    # Set thread limits BEFORE creating session for consistent behavior
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _worker_model = ort.InferenceSession(
        config_dict['model_path'], sess_options=session_options
    )


def _merge_sac_files(data_path: Path, sta_name: str) -> Stream:
    """Merge SAC files for a station into a single stream."""
    sac_list = list(data_path.glob(f'*{sta_name}*Z*'))
    if not sac_list:
        logger.debug(f'{sta_name} using other component')
        sac_list = list(data_path.glob(f'*{sta_name}*'))

    stream = Stream()
    for sac_file in sac_list:
        stream += read(sac_file)
    return stream.merge(fill_value='latest')


def _get_seis_waveform(
    sta_name: str,
    p_arrival_str: str,
    sac_parent_dir: Path,
    sampling_rate: float,
    need_resample: bool,
    time_window: float = 0.64,
) -> np.ndarray | None:
    """Extract seismometer waveform data for a single pick.

    Returns:
        128-sample waveform array or None if extraction fails.
    """
    ymd = _time_to_ymd(p_arrival_str)
    p_arrival = UTCDateTime(p_arrival_str)
    data_path = sac_parent_dir / ymd

    try:
        st = _merge_sac_files(data_path, sta_name)
    except Exception as e:
        logger.debug(f'Merge error for {sta_name}: {e}')
        return None

    try:
        st.detrend('demean')
        st.detrend('linear')
        st.taper(0.001)

        if need_resample:
            st.resample(sampling_rate=sampling_rate)

        st.trim(starttime=p_arrival - time_window, endtime=p_arrival + time_window)

        # TODO: should we pad zeros if the data is not long enough?
        if len(st) == 0 or len(st[0].data) < 128:
            return None

        return st[0].data[:128]
    except Exception as e:
        logger.debug(f'Processing error for {sta_name}: {e}')
        return None


def _get_das_waveform(
    sta_name: str,
    p_arrival_str: str,
    h5_parent_dir: Path,
    interval: int,
    sampling_rate: float,
) -> np.ndarray | None:
    """Extract DAS waveform data for a single pick.

    Returns:
        128-sample waveform array or None if extraction fails.
    """
    ymd = _time_to_ymd(p_arrival_str)
    total_seconds = _get_total_seconds(pd.to_datetime(p_arrival_str))
    index = int(total_seconds // interval)
    window = f'{interval * index}_{interval * (index + 1)}.h5'

    try:
        h5_dir = h5_parent_dir / f'{ymd}_hdf5'
        file_list = list(h5_dir.glob(f'*{window}'))
        if not file_list:
            logger.debug(f'H5 file not found for window {window}')
            return None
        file = file_list[0]
    except Exception as e:
        logger.debug(f'H5 file search error: {e}')
        return None

    try:
        channel_index = _convert_channel_index(sta_name)
        with h5py.File(file, 'r') as fp:
            data = fp['data'][channel_index]
    except Exception as e:
        logger.debug(f'H5 read error for {sta_name}: {e}')
        return None

    # Preprocess DAS data
    data = data - np.mean(data)
    data = detrend(data)

    # Extract window around arrival
    event_index = int((total_seconds % interval) * sampling_rate)
    start_idx = event_index - int(0.64 * sampling_rate)
    end_idx = event_index + int(0.64 * sampling_rate)

    if start_idx < 0 or end_idx > len(data):
        return None

    return data[start_idx:end_idx]


def _predict_polarity_batch(waveforms: np.ndarray) -> list[str]:
    """Predict polarity for a batch of waveforms.

    Args:
        waveforms: Array of shape (N, 128) containing waveforms.

    Returns:
        List of polarity characters.
    """
    global _worker_model

    if _worker_model is None or len(waveforms) == 0:
        return ['x'] * len(waveforms)

    batch_size = len(waveforms)
    motion_input = np.zeros([batch_size, 128, 2], dtype=np.float32)
    valid_mask = np.ones(batch_size, dtype=bool)

    # Prepare all inputs
    for i, waveform in enumerate(waveforms):
        if waveform is None or len(waveform) < 128:
            valid_mask[i] = False
            continue

        motion_input[i, :, 0] = waveform[:128]

        # Check for valid data (from DiTingMotion's original logic)
        if np.max(np.abs(motion_input[i, :, 0])) == 0:
            valid_mask[i] = False
            continue

        # Normalize
        motion_input[i, :, 0] -= np.mean(motion_input[i, :, 0])
        norm_factor = np.std(motion_input[i, :, 0])

        if norm_factor == 0:
            valid_mask[i] = False
            continue

        motion_input[i, :, 0] /= norm_factor

        # Compute differential feature
        diff_data = np.diff(motion_input[i, 64:, 0])
        motion_input[i, 65:, 1] = np.sign(diff_data)

    # Initialize results
    results = ['x'] * batch_size

    # Get valid samples for batch inference
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return results

    valid_input = motion_input[valid_indices]

    # Run batch inference
    ort_inputs = {_worker_model.get_inputs()[0].name: valid_input}
    pred_res = _worker_model.run(None, ort_inputs)

    # Average predictions
    pred_fmp = (pred_res[0] + pred_res[1] + pred_res[2] + pred_res[3]) / 4

    # Map results back
    for batch_idx, orig_idx in enumerate(valid_indices):
        polarity_idx = np.argmax(pred_fmp[batch_idx, :])
        if polarity_idx == 0:
            results[orig_idx] = 'U'
        elif polarity_idx == 1:
            results[orig_idx] = 'D'

    return results


def _process_pick_chunk(pick_chunk: list[dict]) -> list[dict]:
    """Process a chunk of picks in a worker process.

    Args:
        pick_chunk: List of pick dictionaries with station_id, phase_time, etc.

    Returns:
        List of result dictionaries with original data plus polarity.
    """
    global _worker_config

    if _worker_config is None:
        return [{'error': 'Worker not initialized'} for _ in pick_chunk]

    config = _worker_config
    sac_parent_dir = Path(config['sac_parent_dir']) if config['sac_parent_dir'] else None
    h5_parent_dir = Path(config['h5_parent_dir']) if config['h5_parent_dir'] else None
    das_in_data = config['das_in_data']

    waveforms = []
    for pick in pick_chunk:
        sta_name = pick['station_id']
        p_arrival = pick['phase_time']

        # Determine data type: DAS or seismometer
        if das_in_data:
            # Use type_judge_result if available, otherwise default based on data dirs
            is_seis = config['type_judge_result'].get(sta_name, False)
        else:
            # No DAS data, all stations are seismometers
            is_seis = True

        if is_seis and sac_parent_dir is not None:
            waveform = _get_seis_waveform(
                sta_name,
                p_arrival,
                sac_parent_dir,
                config['sampling_rate'],
                config['need_resample'],
            )
        elif not is_seis and h5_parent_dir is not None:
            waveform = _get_das_waveform(
                sta_name,
                p_arrival,
                h5_parent_dir,
                config['interval'],
                config['sampling_rate'],
            )
        else:
            waveform = None

        waveforms.append(waveform)

    # Batch inference
    polarities = _predict_polarity_batch(np.array(waveforms, dtype=object))

    # Build results
    results = []
    for pick, polarity in zip(pick_chunk, polarities):
        result = pick.copy()
        result['polarity'] = polarity
        results.append(result)

    return results


class DitingMotion:
    """Optimized P-wave polarity predictor using DiTing deep learning model.

    Key optimizations over the original implementation:
    1. Single CSV read at initialization (not per-event)
    2. Batch ONNX inference (multiple samples per forward pass)
    3. Fine-grained parallelization by pick chunks (balanced load)
    4. Reduced redundant I/O through better data organization
    """

    def __init__(
        self,
        gamma_picks: Path,
        model_path: Path = DITING_MODEL_PATH,
        output_dir: Path | None = None,
        sac_parent_dir: Path | None = None,
        h5_parent_dir: Path | None = None,
        interval: int = 300,
        sampling_rate: float = 100.0,
        need_resample: bool = False,
        chunk_size: int = 50,
        das_in_data: bool = False,
        type_judge: Callable[[str], bool] | None = None,
    ):
        """Initialize DitingMotion polarity predictor.

        Args:
            gamma_picks: Path to GaMMA picks CSV file.
            model_path: Path to ONNX model file.
            output_dir: Output directory for results.
            sac_parent_dir: Parent directory for SAC files (seismometer data).
            h5_parent_dir: Parent directory for HDF5 files (DAS data).
            interval: Time interval for HDF5 file segmentation (seconds).
            sampling_rate: Target sampling rate (Hz).
            need_resample: Whether to resample seismometer data.
            chunk_size: Number of picks per parallel worker task.
            das_in_data: Whether DAS data is present. If False, all stations are
                treated as seismometers and type_judge is ignored.
            type_judge: Optional function that returns True if station is seismometer,
                False if DAS. Only used when das_in_data=True. If not provided when
                das_in_data=True, all stations default to DAS.
        """
        self.gamma_picks = Path(gamma_picks)
        self.model_path = Path(model_path)
        self.output_dir = self._setup_output_dir(output_dir)
        self.sac_parent_dir = Path(sac_parent_dir) if sac_parent_dir else None
        self.h5_parent_dir = Path(h5_parent_dir) if h5_parent_dir else None
        self.interval = interval
        self.sampling_rate = sampling_rate
        self.need_resample = need_resample
        self.chunk_size = chunk_size
        self.das_in_data = das_in_data
        self.type_judge = type_judge

        # Load and preprocess picks once
        self._df_picks = self._load_picks()
        self._p_picks = self._filter_p_picks()

        # Pre-compute type_judge results for all unique stations
        self._station_types = self._compute_station_types()

        self.picks: Path | None = None

    def _setup_output_dir(self, output_dir: Path | None) -> Path:
        """Setup and return output directory."""
        if output_dir is not None:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(__file__).parents[1].resolve() / 'diting_result'
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _load_picks(self) -> pd.DataFrame:
        """Load picks CSV once at initialization."""
        return pd.read_csv(self.gamma_picks)

    def _filter_p_picks(self) -> pd.DataFrame:
        """Filter for P-phase picks with valid event indices."""
        df = self._df_picks
        mask = (df['phase_type'] == 'P') & (df['event_index'] != -1)
        return df[mask].copy()

    def _compute_station_types(self) -> dict[str, bool]:
        """Pre-compute station type (seismometer vs DAS) for all stations.

        Returns:
            Dict mapping station_id to bool (True = seismometer, False = DAS).
        """
        unique_stations = self._p_picks['station_id'].unique()

        if not self.das_in_data:
            # No DAS data, all are seismometers
            return {sta: True for sta in unique_stations}

        if self.type_judge is not None:
            # Use provided function
            return {sta: self.type_judge(sta) for sta in unique_stations}

        # das_in_data=True but no type_judge: default all to DAS
        return {sta: False for sta in unique_stations}

    def get_picks(self) -> Path | None:
        """Return path to output picks file."""
        return self.picks

    def _create_worker_config(self) -> dict:
        """Create serializable configuration for worker processes."""
        return {
            'model_path': str(self.model_path),
            'sac_parent_dir': str(self.sac_parent_dir) if self.sac_parent_dir else None,
            'h5_parent_dir': str(self.h5_parent_dir) if self.h5_parent_dir else None,
            'interval': self.interval,
            'sampling_rate': self.sampling_rate,
            'need_resample': self.need_resample,
            'das_in_data': self.das_in_data,
            'type_judge_result': self._station_types,
        }

    def _prepare_pick_chunks(self) -> list[list[dict]]:
        """Prepare balanced chunks of picks for parallel processing.

        Returns:
            List of pick chunks, each containing dictionaries with pick info.
        """
        picks_list = self._p_picks.to_dict('records')
        chunks = []

        for i in range(0, len(picks_list), self.chunk_size):
            chunks.append(picks_list[i : i + self.chunk_size])

        return chunks

    def run_parallel_predict(self, processes: int = 3) -> None:
        """Run parallel polarity prediction.

        Args:
            processes: Number of worker processes.
        """
        self.picks = self.output_dir / 'polarity_picks.csv'

        total_picks = len(self._p_picks)
        if total_picks == 0:
            logger.warning('No P-phase picks to process')
            return

        logger.info(f'DitingMotion starting: {total_picks} picks, {processes} processes')
        start_time = time.time()

        # Prepare work chunks
        chunks = self._prepare_pick_chunks()
        config = self._create_worker_config()

        logger.info(f'Processing {len(chunks)} chunks of ~{self.chunk_size} picks each')

        # Process in parallel
        all_results = []

        with mp.Pool(
            processes=processes,
            initializer=_init_worker,
            initargs=(config,),
        ) as pool:
            chunk_results = pool.map(_process_pick_chunk, chunks)

        # Flatten results
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)

        elapsed = time.time() - start_time
        logger.info(
            f'DitingMotion completed: {len(all_results)} picks in {elapsed:.2f}s '
            f'({len(all_results) / elapsed:.1f} picks/sec)'
        )

        # Save results
        if all_results:
            df_result = pd.DataFrame(all_results)
            df_result.to_csv(self.picks, index=False)
            logger.info(f'Results saved to {self.picks}')

    def run_sequential_predict(self) -> None:
        """Run sequential prediction (for debugging or small datasets)."""
        self.picks = self.output_dir / 'polarity_picks.csv'

        total_picks = len(self._p_picks)
        if total_picks == 0:
            logger.warning('No P-phase picks to process')
            return

        logger.info(f'DitingMotion (sequential): {total_picks} picks')
        start_time = time.time()

        # Initialize model in main process
        config = self._create_worker_config()
        _init_worker(config)

        # Process all picks as single chunk
        picks_list = self._p_picks.to_dict('records')
        results = _process_pick_chunk(picks_list)

        elapsed = time.time() - start_time
        logger.info(
            f'DitingMotion completed: {len(results)} picks in {elapsed:.2f}s '
            f'({len(results) / elapsed:.1f} picks/sec)'
        )

        if results:
            df_result = pd.DataFrame(results)
            df_result.to_csv(self.picks, index=False)
            logger.info(f'Results saved to {self.picks}')
