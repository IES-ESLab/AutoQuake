"""
Realtime earthquake relocator using H3DD.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class RealtimeRelocator:
    """
    Realtime wrapper for H3DD earthquake relocation.

    This class provides realtime relocation by:
    1. Accepting events and picks as DataFrames
    2. Converting to H3DD input format
    3. Running H3DD subprocess
    4. Parsing and returning relocated events

    Attributes:
        station_file: Path to station information file
        model_3d: Path to 3D velocity model
        h3dd_dir: Directory containing H3DD executable
    """

    def __init__(
        self,
        station_file: Path,
        model_3d: Path,
        h3dd_dir: Path | None = None,
        weights: list[float] | None = None,
        priori_weight: list[float] | None = None,
        cut_off_distance_for_dd: float = 3.0,
        inv: int = 2,
        damping_factor: float = 0.0,
        rmscut: float = 1.0e-4,
        max_iter: int = 5,
        constrain_factor: float = 0.0,
        joint_inv_with_single_event_method: int = 1,
        consider_elevation: int = 0,
    ):
        """
        Initialize the realtime relocator.

        Args:
            station_file: Path to station CSV (columns: station, longitude, latitude, elevation)
            model_3d: Path to 3D velocity model file
            h3dd_dir: Directory containing H3DD executable (default: PROJECT_ROOT/H3DD)
            weights: [wp, ws, wsingle] weighting (default: [1.0, 1.0, 0.1])
            priori_weight: A priori weighting for catalog data
            cut_off_distance_for_dd: Cut-off distance for DD method (km)
            inv: Inversion method (1=SVD, 2=LSQR)
            damping_factor: Damping factor for LSQR
            rmscut: RMS cut-off threshold
            max_iter: Maximum iterations
            constrain_factor: Constrain factor
            joint_inv_with_single_event_method: Joint inversion flag (1=yes, 0=no)
            consider_elevation: Consider elevation flag (1=yes, 0=no)
        """
        self.station_file = Path(station_file)
        self.model_3d = Path(model_3d)

        # Set H3DD directory
        if h3dd_dir is None:
            self.h3dd_dir = Path(__file__).parents[2] / 'H3DD'
        else:
            self.h3dd_dir = Path(h3dd_dir)

        # H3DD parameters
        self.weights = weights or [1.0, 1.0, 0.1]
        self.priori_weight = priori_weight or [1.0, 0.75, 0.5, 0.25, 0.0]
        self.cut_off_distance_for_dd = cut_off_distance_for_dd
        self.inv = inv
        self.damping_factor = damping_factor
        self.rmscut = rmscut
        self.max_iter = max_iter
        self.constrain_factor = constrain_factor
        self.joint_inv_with_single_event_method = joint_inv_with_single_event_method
        self.consider_elevation = consider_elevation

        # Prepare station file in H3DD format
        self._h3dd_station = self._prepare_station_file()

        # Copy model to H3DD directory if needed
        self._prepare_model_file()

        self._relocation_count = 0

    def _prepare_station_file(self) -> str:
        """Convert station CSV to H3DD format."""
        station_h3dd = self.h3dd_dir / 'station.all.select'

        df = pd.read_csv(self.station_file)
        with open(station_h3dd, 'w') as f:
            for _, row in df.iterrows():
                f.write(
                    f"{row['station']} {row['longitude']} {row['latitude']} "
                    f"{row.get('elevation', 0)} 19010101 21001231\n"
                )

        return 'station.all.select'

    def _prepare_model_file(self) -> None:
        """Ensure model file is in H3DD directory."""
        if self.model_3d.parent != self.h3dd_dir:
            import shutil
            dest = self.h3dd_dir / self.model_3d.name
            if not dest.exists():
                shutil.copy(self.model_3d, dest)
            self._model_name = self.model_3d.name
        else:
            self._model_name = self.model_3d.name

    def _write_h3dd_inp(self, dat_ch_name: str) -> Path:
        """Write H3DD input configuration file."""
        inp_file = self.h3dd_dir / 'h3dd.inp'

        with open(inp_file, 'w') as f:
            f.write('*1. input catalog data\n')
            f.write(f'{dat_ch_name}\n')
            f.write('*2. station information file\n')
            f.write(f'{self._h3dd_station}\n')
            f.write('*3. 3d velocity model\n')
            f.write(f'{self._model_name}\n')
            f.write('*4. weighting for p wave, s wave, and single event data\n')
            f.write('*   wp  ws  wsingle\n')
            f.write(
                f"    {self.weights[0]:<4}{self.weights[1]:<4}{self.weights[2]:>4}\n"
            )
            f.write('*5. a priori weighting for catalog data\n')
            f.write('*   0      1      2      3      4\n')
            f.write(
                f"    {self.priori_weight[0]:<5}{self.priori_weight[1]:<7}"
                f"{self.priori_weight[2]:<7}{self.priori_weight[3]:<7}"
                f"{self.priori_weight[4]:>3}\n"
            )
            f.write('*6. cut off distance for D-D method (km)\n')
            f.write(f'    {self.cut_off_distance_for_dd}\n')
            f.write('*7. inv (1=SVD 2=LSQR)\n')
            f.write(f'    {self.inv}\n')
            f.write('*8. damping factor (Only work if inv=2)\n')
            f.write(f'    {self.damping_factor}\n')
            f.write('*9. rmscut (sec)\n')
            f.write(f'    {self.rmscut}\n')
            f.write('*10. maximum interation times\n')
            f.write(f'    {self.max_iter}\n')
            f.write('*11. constrain factor\n')
            f.write(f'    {self.constrain_factor}\n')
            f.write('*12. joint inversion with single event method (1=yes 0=no)\n')
            f.write(f'    {self.joint_inv_with_single_event_method}\n')
            f.write('*13. consider elevation or not (1=yes 0=no)\n')
            f.write(f'    {self.consider_elevation}\n')

        return inp_file

    def _preprocess_data(
        self, events: pd.DataFrame, picks: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess events and picks for H3DD format."""
        df_event = events.copy()
        df_picks = picks.copy()

        # Process events
        df_event['time'] = pd.to_datetime(df_event['time'])
        df_event['ymd'] = df_event['time'].dt.strftime('%Y%m%d')
        df_event['hour'] = df_event['time'].dt.hour
        df_event['minute'] = df_event['time'].dt.minute
        df_event['seconds'] = (
            df_event['time'].dt.second
            + df_event['time'].dt.microsecond / 1_000_000
        )
        df_event['lon_int'] = df_event['longitude'].astype(int)
        df_event['lon_deg'] = (df_event['longitude'] - df_event['lon_int']) * 60
        df_event['lat_int'] = df_event['latitude'].astype(int)
        df_event['lat_deg'] = (df_event['latitude'] - df_event['lat_int']) * 60
        df_event['depth'] = df_event['depth_km'].round(2)

        # Process picks
        df_picks['phase_time'] = pd.to_datetime(df_picks['phase_time'])
        df_picks['minute'] = df_picks['phase_time'].dt.minute
        df_picks['seconds'] = (
            df_picks['phase_time'].dt.second
            + df_picks['phase_time'].dt.microsecond / 1_000_000
        )

        return df_event, df_picks

    def _write_dat_ch(
        self,
        output_file: Path,
        df_event: pd.DataFrame,
        df_picks: pd.DataFrame,
    ) -> None:
        """Write events and picks in H3DD .dat_ch format."""
        event_indices = df_event['event_index'].unique()

        with open(output_file, 'w') as f:
            for idx in event_indices:
                row = df_event[df_event['event_index'] == idx].iloc[0]

                # Write event line
                f.write(
                    f"{row['ymd']:>9}{row['hour']:>2}{row['minute']:>2}"
                    f"{row['seconds']:>6.2f}{row['lat_int']:2}"
                    f"{row['lat_deg']:0>5.2f}{row['lon_int']:3}"
                    f"{row['lon_deg']:0>5.2f}{row['depth']:>6.2f}\n"
                )

                # Write pick lines
                event_picks = df_picks[df_picks['event_index'] == idx]
                for _, pick in event_picks.iterrows():
                    # Handle minute wrap-around
                    wmm = pick['minute']
                    if row['minute'] == 59 and pick['minute'] == 0:
                        wmm = 60

                    weight = '1.00'
                    if pick['phase_type'] == 'P':
                        f.write(
                            f" {pick['station_id']:<4}{'0.0':>6}{'0':>4}{'0':>4}"
                            f"{wmm:>4}{pick['seconds']:>6.2f}{'0.01':>5}{weight:>5}"
                            f"{'0.00':>6}{'0.00':>5}{'0.00':>5}\n"
                        )
                    else:  # S wave
                        f.write(
                            f" {pick['station_id']:<4}{'0.0':>6}{'0':>4}{'0':>4}"
                            f"{wmm:>4}{'0.00':>6}{'0.00':>5}{'0.00':>5}"
                            f"{pick['seconds']:>6.2f}{'0.01':>5}{weight:>5}\n"
                        )

    def _parse_dout(self, dout_file: Path) -> pd.DataFrame | None:
        """Parse H3DD dout output file."""
        if not dout_file.exists():
            logger.warning(f'H3DD output not found: {dout_file}')
            return None

        events = []
        with open(dout_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        events.append({
                            'ymd': parts[0],
                            'hour': int(parts[1]),
                            'minute': int(parts[2]),
                            'seconds': float(parts[3]),
                            'latitude': float(parts[4]) + float(parts[5]) / 60,
                            'longitude': float(parts[6]) + float(parts[7]) / 60,
                            'depth_km': float(parts[8]),
                        })
                    except (ValueError, IndexError) as e:
                        logger.debug(f'Skipping line: {line.strip()} ({e})')
                        continue

        if not events:
            return None

        df = pd.DataFrame(events)
        # Reconstruct time
        df['time'] = pd.to_datetime(
            df['ymd'] + df['hour'].astype(str).str.zfill(2)
            + df['minute'].astype(str).str.zfill(2),
            format='%Y%m%d%H%M'
        ) + pd.to_timedelta(df['seconds'], unit='s')

        return df[['time', 'latitude', 'longitude', 'depth_km']]

    def relocate(
        self, events: pd.DataFrame, picks: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Relocate events using H3DD.

        Args:
            events: DataFrame with columns [event_index, time, latitude, longitude, depth_km]
            picks: DataFrame with columns [event_index, station_id, phase_time, phase_type]

        Returns:
            DataFrame with relocated events or None if relocation failed
        """
        if events.empty:
            logger.warning('No events to relocate')
            return None

        logger.info(f'Relocating {len(events)} events with {len(picks)} picks')

        # Preprocess data
        df_event, df_picks = self._preprocess_data(events, picks)

        # Write .dat_ch file
        dat_ch_file = self.h3dd_dir / 'realtime.dat_ch'
        self._write_dat_ch(dat_ch_file, df_event, df_picks)

        # Write h3dd.inp
        self._write_h3dd_inp('realtime.dat_ch')

        # Run H3DD
        try:
            with open(self.h3dd_dir / 'h3dd.inp') as inp_file:
                result = subprocess.run(
                    ['./h3dd'],
                    stdin=inp_file,
                    cwd=self.h3dd_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            if result.returncode != 0:
                logger.error(f'H3DD failed: {result.stderr}')
                return None

        except subprocess.TimeoutExpired:
            logger.error('H3DD timed out')
            return None
        except FileNotFoundError:
            logger.error('H3DD executable not found')
            return None

        # Parse output
        dout_file = self.h3dd_dir / 'realtime.dat_ch.dout'
        relocated = self._parse_dout(dout_file)

        if relocated is not None:
            self._relocation_count += len(relocated)
            logger.info(f'Relocated {len(relocated)} events')

        return relocated

    def relocate_single(
        self, event: dict, picks: list[dict]
    ) -> dict | None:
        """
        Relocate a single event.

        Args:
            event: Event dictionary with time, latitude, longitude, depth_km
            picks: List of pick dictionaries

        Returns:
            Relocated event dictionary or None
        """
        events_df = pd.DataFrame([{
            'event_index': 0,
            'time': event.get('time') or event.get('origin_time'),
            'latitude': event['latitude'],
            'longitude': event['longitude'],
            'depth_km': event.get('depth_km', event.get('depth', 10.0)),
        }])

        picks_df = pd.DataFrame(picks)
        if 'event_index' not in picks_df.columns:
            picks_df['event_index'] = 0

        result = self.relocate(events_df, picks_df)

        if result is not None and not result.empty:
            row = result.iloc[0]
            return {
                'time': row['time'].isoformat(),
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'depth_km': row['depth_km'],
            }

        return None

    @property
    def stats(self) -> dict:
        """Get relocator statistics."""
        return {
            'total_relocations': self._relocation_count,
            'h3dd_dir': str(self.h3dd_dir),
        }