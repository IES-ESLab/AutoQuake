"""
Realtime PhaseNet phase picker.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .buffers import PickBuffer, WaveformBuffer

logger = logging.getLogger(__name__)


def padding(data, min_nt=1024, min_nx=1):
    """Pad data to minimum dimensions."""
    nch, nt, nx = data.shape[-3:]
    pad_nt = (min_nt - nt % min_nt) % min_nt
    pad_nx = (min_nx - nx % min_nx) % min_nx
    with torch.no_grad():
        if data.dim() == 3:
            data = F.pad(data, (0, pad_nx, 0, pad_nt), mode='constant')
        else:  # batch dimension
            data = F.pad(data, (0, pad_nx, 0, pad_nt), mode='constant')
    return data


class RealtimePhaseNet:
    """
    Realtime wrapper for PhaseNet phase picking.

    This class provides realtime phase picking by:
    1. Maintaining a waveform buffer
    2. Running PhaseNet inference on demand
    3. Outputting picks to a PickBuffer

    Attributes:
        model_type: PhaseNet model variant ('phasenet', 'phasenet_plus')
        device: Torch device for inference
        min_prob: Minimum probability threshold for picks
    """

    def __init__(
        self,
        waveform_buffer: WaveformBuffer,
        pick_buffer: PickBuffer,
        model_type: Literal['phasenet', 'phasenet_plus'] = 'phasenet',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        min_prob: float = 0.3,
        phases: list[str] | None = None,
        amp: bool = True,
    ):
        """
        Initialize the realtime PhaseNet picker.

        Args:
            waveform_buffer: Buffer for incoming waveforms
            pick_buffer: Buffer to output picks to
            model_type: Model variant to use
            device: Device for inference ('cpu', 'cuda', 'mps')
            min_prob: Minimum probability threshold
            phases: Phase types to detect (default: ['P', 'S'])
            amp: Whether to use automatic mixed precision
        """
        self.waveform_buffer = waveform_buffer
        self.pick_buffer = pick_buffer
        self.model_type = model_type
        self.device = torch.device(device)
        self.min_prob = min_prob
        self.phases = phases or ['P', 'S']
        self.amp = amp

        self._model = None
        self._pick_count = 0

    def _load_model(self) -> None:
        """Load the PhaseNet model."""
        if self._model is not None:
            return

        # Import here to avoid circular imports
        from ..EQNet.eqnet import models

        logger.info(f'Loading {self.model_type} model...')

        # Build model
        self._model = models.__dict__[self.model_type].build_model(
            backbone='unet',
            in_channels=1,
            out_channels=len(self.phases) + 1,
        )

        # Load pretrained weights
        if self.model_type == 'phasenet':
            model_url = 'https://github.com/AI4EPS/models/releases/download/PhaseNet-v1/model_99.pth'
        elif self.model_type == 'phasenet_plus':
            model_url = 'https://github.com/AI4EPS/models/releases/download/PhaseNet-Plus-v1/model_99.pth'
        else:
            raise ValueError(f'Unknown model type: {self.model_type}')

        checkpoint = torch.hub.load_state_dict_from_url(
            model_url,
            model_dir=f'./model_{self.model_type}',
            progress=True,
            check_hash=True,
            map_location='cpu',
        )
        self._model.load_state_dict(checkpoint['model'], strict=True)
        self._model.to(self.device)
        self._model.eval()

        logger.info(f'Model loaded on {self.device}')

    def process_buffer(self) -> list[dict]:
        """
        Process current waveform buffer and extract picks.

        Returns:
            List of pick dictionaries
        """
        # Ensure model is loaded
        self._load_model()

        # Get data from buffer
        meta = self.waveform_buffer.get_window_for_model()
        if meta is None:
            return []

        # Run inference
        picks = self._run_inference(meta)

        # Add picks to buffer
        for pick in picks:
            self.pick_buffer.add_pick(pick)
            self._pick_count += 1

        if picks:
            logger.info(f'Detected {len(picks)} picks')

        return picks

    def _run_inference(self, meta: dict) -> list[dict]:
        """
        Run model inference on waveform data.

        Args:
            meta: Dictionary with 'data', 'station_id', 'begin_time', etc.

        Returns:
            List of pick dictionaries with phase_polarity if using phasenet_plus
        """
        from ..EQNet.eqnet.utils import detect_peaks, extract_picks

        data = meta['data'].to(self.device)
        nt, nx = meta['nt'], meta['nx']

        # Pad data
        data = padding(data, min_nt=1024, min_nx=1)

        # Setup context for inference
        ctx = (
            nullcontext()
            if self.device.type == 'cpu'
            else torch.amp.autocast(device_type=self.device.type, enabled=self.amp)
        )

        picks_list = []

        with torch.inference_mode():
            with ctx:
                # Forward pass
                output = self._model({'data': data})

                # Postprocess - trim to original size
                if 'phase' in output:
                    output['phase'] = output['phase'][:, :, :nt, :nx]

                # Process polarity output if available (phasenet_plus)
                polarity_score = None
                if 'polarity' in output:
                    output['polarity'] = output['polarity'][:, :, :nt, :nx]
                    polarity_score = torch.softmax(output['polarity'], dim=1)

                # Extract phase scores
                phase_scores = torch.softmax(output['phase'], dim=1)

                # Detect peaks
                topk_scores, topk_inds = detect_peaks(
                    phase_scores,
                    vmin=self.min_prob,
                    kernel=128,
                )

                # Extract picks (with polarity if available)
                phase_picks = extract_picks(
                    topk_inds,
                    topk_scores,
                    file_name=['realtime'],
                    station_id=meta['station_id'],
                    begin_time=[meta['begin_time']],
                    dt=meta['dt_s'],
                    vmin=self.min_prob,
                    phases=self.phases,
                    polarity_score=polarity_score,
                    waveform=data,
                    window_amp=[10, 5],
                )

                # Flatten picks from all batches
                for batch_picks in phase_picks:
                    for pick in batch_picks:
                        picks_list.append(pick)

        return picks_list

    def check_and_process(self) -> list[dict]:
        """
        Check if buffer has enough data and process if so.

        This is the main entry point for the realtime loop.

        Returns:
            List of picks (empty if no processing)
        """
        # Check if we have enough data
        if self.waveform_buffer.station_count == 0:
            return []

        return self.process_buffer()

    @property
    def stats(self) -> dict:
        """Get picker statistics."""
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'model_loaded': self._model is not None,
            'total_picks': self._pick_count,
            'waveform_buffer': self.waveform_buffer.stats,
        }


class WaveformStreamSimulator:
    """
    Simulates realtime waveform streaming from existing data files.

    This class reads waveform files and replays them as if they were
    arriving in real-time. Useful for testing.
    """

    def __init__(
        self,
        data_path: Path,
        pattern: str = '*.SAC',
        speed: float = 1.0,
    ):
        """
        Initialize the waveform stream simulator.

        Args:
            data_path: Directory containing waveform files
            pattern: Glob pattern for files
            speed: Simulation speed multiplier
        """
        self.data_path = Path(data_path)
        self.pattern = pattern
        self.speed = speed

        self._files: list[Path] = []
        self._current_index = 0
        self._running = False

    def load(self) -> None:
        """Load file list."""
        self._files = sorted(self.data_path.glob(self.pattern))
        logger.info(f'Found {len(self._files)} waveform files')

    def stream_to_buffer(
        self,
        buffer: WaveformBuffer,
        max_files: int | None = None,
    ) -> None:
        """
        Stream waveforms to buffer.

        Args:
            buffer: WaveformBuffer to receive data
            max_files: Maximum number of files to process
        """
        import obspy

        if not self._files:
            self.load()

        files_to_process = self._files[:max_files] if max_files else self._files

        for fpath in files_to_process:
            try:
                stream = obspy.read(str(fpath))
                buffer.add_stream(stream)
                logger.debug(f'Added {fpath.name} to buffer')
            except Exception as e:
                logger.error(f'Error reading {fpath}: {e}')

    @property
    def stats(self) -> dict:
        """Get simulator statistics."""
        return {
            'total_files': len(self._files),
            'current_index': self._current_index,
        }