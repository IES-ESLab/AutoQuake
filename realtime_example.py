"""
AutoQuake Realtime System - Example Usage

This script demonstrates how to use the realtime earthquake detection system
in a modular, programmatic way.

Processing flow:
    Data ingestion → Picking (polarity, amplitude) → Association + preliminary magnitude
    → Relocation → Updated magnitude → Focal mechanism
"""

from datetime import timedelta
import logging
from pathlib import Path

from autoquake.realtime import (
    RealtimeConfig,
    RealtimeRunner,
    RealtimePhaseNet,
    RealtimeGaMMA,
    RealtimeMagnitude,
    RealtimeRelocator,
    RealtimeFocalMechanism,
    WaveformBuffer,
    PickBuffer,
    SEEDLINKClient,
    StationConfig,
)


logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging for the realtime examples."""
    logging.basicConfig(
        filename='realtime_example.log',
        filemode='w',
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


def example_full_pipeline():
    """
    Example: Complete earthquake processing pipeline with manual component control.

    This example shows how to use individual components for maximum flexibility.
    For simpler usage, see example_with_runner().

    Flow:
        1. Data ingestion (SEEDLINK or file)
        2. Picking (PhaseNet with polarity and amplitude)
        3. Association (GaMMA) - returns events WITH their associated picks
        4. Preliminary magnitude (quick estimate)
        5. Relocation (H3DD)
        6. Accurate magnitude (using relocated position)
        7. Focal mechanism (GAfocal)
    """
    station_file = Path('path/to/station.csv')
    result_path = Path('./realtime_results')

    # =========================================================================
    # 1. Data ingestion - Setup buffers
    # =========================================================================
    waveform_buffer = WaveformBuffer(
        duration=timedelta(seconds=60),
        sampling_rate=100.0,
    )

    pick_buffer = PickBuffer(
        window_size=timedelta(seconds=60),
        min_picks_to_trigger=10,
    )

    # =========================================================================
    # 2. Picking - PhaseNet with polarity and amplitude
    # =========================================================================
    picker = RealtimePhaseNet(
        waveform_buffer=waveform_buffer,
        pick_buffer=pick_buffer,
        model_type='phasenet_plus',  # Outputs polarity
        device='cuda',
        min_prob=0.3,
    )

    # =========================================================================
    # 3. Association - GaMMA (returns events with associated picks)
    # =========================================================================
    associator = RealtimeGaMMA(
        station=station_file,
        result_path=result_path,
        pick_buffer=pick_buffer,
        center=(121.5, 24.0),
        xlim_degree=[120.0, 123.0],
        ylim_degree=[22.0, 26.0],
    )

    # =========================================================================
    # 4. Refinement components
    # =========================================================================
    magnitude_estimator = RealtimeMagnitude(
        station_info=station_file,
        pz_dir=Path('path/to/pz_files'),
        use_wa_simulation=True,
    )

    relocator = RealtimeRelocator(
        station_file=station_file,
        model_3d=Path('path/to/velocity_model.txt'),
        h3dd_dir=Path('path/to/H3DD'),
    )

    focal_estimator = RealtimeFocalMechanism(
        gafocal_dir=Path('path/to/GAfocal'),
        station_info=station_file,
        min_polarities=8,
    )

    # =========================================================================
    # Processing loop
    # =========================================================================
    while True:
        # ---------------------------------------------------------------------
        # Step 1: Data ingestion
        # ---------------------------------------------------------------------
        # In real usage: waveform_buffer.add_stream(stream_from_seedlink)

        # ---------------------------------------------------------------------
        # Step 2: Picking (outputs to pick_buffer automatically)
        # ---------------------------------------------------------------------
        picker.check_and_process()

        # ---------------------------------------------------------------------
        # Step 3: Association - check_and_process returns (event, picks) tuples
        # ---------------------------------------------------------------------
        # NOTE: check_and_process() internally calls should_trigger() and
        # returns list of (event_dict, associated_picks) tuples
        events_with_picks = associator.check_and_process()

        for event, event_picks in events_with_picks:
            print(f"\nEvent detected at {event.get('time')}")
            print(f"  Location: {event['latitude']:.3f}, {event['longitude']:.3f}")
            print(f"  Depth: {event.get('depth_km', 0):.1f} km")
            print(f"  Picks: {len(event_picks)}")

            # -----------------------------------------------------------------
            # Step 4: Preliminary magnitude (fast estimate)
            # -----------------------------------------------------------------
            prelim_mag = magnitude_estimator.estimate_preliminary(event, event_picks)
            if prelim_mag is not None:
                event['magnitude_preliminary'] = prelim_mag
                print(f"  Preliminary magnitude: M{prelim_mag:.1f}")

            # -----------------------------------------------------------------
            # Step 5: Relocation (H3DD)
            # -----------------------------------------------------------------
            relocated = relocator.relocate_single(event, event_picks)
            if relocated:
                print(f"  Relocated: {relocated['latitude']:.4f}, "
                      f"{relocated['longitude']:.4f}, {relocated['depth_km']:.1f} km")
            else:
                relocated = event  # Fallback to original location

            # -----------------------------------------------------------------
            # Step 6: Accurate magnitude (using relocated position)
            # -----------------------------------------------------------------
            final_mag = magnitude_estimator.estimate_from_relocation(
                relocated, event_picks
            )
            if final_mag is not None:
                event['magnitude'] = final_mag
                print(f"  Final magnitude: M{final_mag:.1f}")

            # -----------------------------------------------------------------
            # Step 7: Focal mechanism
            # -----------------------------------------------------------------
            focal_result = focal_estimator.estimate(relocated, event_picks)
            if focal_result:
                event['focal_mechanism'] = focal_result
                print(f"  Focal: strike={focal_result['strike']:.0f}, "
                      f"dip={focal_result['dip']:.0f}, rake={focal_result['rake']:.0f}")

        # Break for example
        break


def example_with_runner():
    """
    Example: Using RealtimeRunner for simplified orchestration.

    RealtimeRunner handles the complete pipeline internally:
    - Pick buffer management
    - GaMMA association
    - Preliminary magnitude
    - H3DD relocation
    - Accurate magnitude (WA-simulated)
    - Focal mechanism
    - Result publishing

    This is the RECOMMENDED approach for most use cases.
    """
    config = RealtimeConfig(
        station_file=Path('examples/testset/station.csv'),
        result_path=Path('examples/realtime_results'),
        # Association settings
        center=(121.625, 24.0),
        xlim_degree=[121.0, 122.25],
        ylim_degree=[23.25, 24.75],
        min_picks_per_eq=8,
        # Enable all refinement stages
        enable_magnitude=True,
        enable_relocation=True,
        enable_focal=True,
        # Paths for refinement components
        model_3d=Path('H3DD/tomops_H14'),
        h3dd_dir=Path('H3DD'),
        pz_dir=None,#Path('/home/patrick/Work/Hualien0403/PZ'),
        gafocal_dir=Path('GAfocal'),
    )

    runner = RealtimeRunner(config)

    # Option A: Run simulation with existing picks (for testing)
    runner.run_simulation(
        picks_file=Path('examples/testset/small_set.csv'),
        speed=10.0,  # 10x faster than real-time
        max_duration=None,  # Stop after 300 seconds
    )

    # Option B: Manual processing control
    # Returns list of fully processed events (with all refinements applied)
    # events = runner.process_once()
    # for event in events:
    #     print(f"Event: M{event.get('magnitude', '?')}")
    #     print(f"  Relocated depth: {event.get('depth_km_relocated', 'N/A')} km")
    #     if 'focal_mechanism' in event:
    #         fm = event['focal_mechanism']
    #         print(f"  Focal: {fm['strike']}/{fm['dip']}/{fm['rake']}")

    # Get statistics
    print(f"\nProcessing stats: {runner.stats}")


def example_seedlink_integration():
    """
    Example: Real-time data from SEEDLINK server.

    This shows how to connect to a SEEDLINK server and process
    continuous waveform data with the complete pipeline.
    """
    # Station configuration
    stations = [
        StationConfig(network='TW', station='HUAL', channels=['HLE', 'HLN', 'HLZ']),
        StationConfig(network='TW', station='YULI', channels=['HLE', 'HLN', 'HLZ']),
        StationConfig(network='TW', station='TPUB', channels=['HLE', 'HLN', 'HLZ']),
    ]

    # Buffers
    waveform_buffer = WaveformBuffer(
        duration=timedelta(seconds=60),
        sampling_rate=100.0,
    )

    pick_buffer = PickBuffer(
        window_size=timedelta(seconds=60),
        min_picks_to_trigger=10,
    )

    # SEEDLINK client
    client = SEEDLINKClient(
        server='rtserve.iris.washington.edu',
        port=18000,
        stations=stations,
        waveform_buffer=waveform_buffer,
    )

    # Picker
    picker = RealtimePhaseNet(
        waveform_buffer=waveform_buffer,
        pick_buffer=pick_buffer,
        model_type='phasenet_plus',
        device='cuda',
    )

    # Associator - check_and_process() returns (event, picks) tuples
    associator = RealtimeGaMMA(
        station=Path('path/to/station.csv'),
        result_path=Path('./results'),
        pick_buffer=pick_buffer,
        center=(121.5, 24.0),
    )

    # Start SEEDLINK streaming (runs in background thread)
    client.start()

    try:
        while True:
            # Picker processes waveforms and outputs picks to pick_buffer
            picker.check_and_process()

            # Association - internally checks should_trigger()
            # Returns list of (event_dict, associated_picks) tuples
            events_with_picks = associator.check_and_process()

            for event, picks in events_with_picks:
                print(f"Event: {event.get('time')} at "
                      f"{event['latitude']:.3f}, {event['longitude']:.3f}, "
                      f"depth={event.get('depth_km', 0):.1f}km, "
                      f"picks={len(picks)}")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.stop()


if __name__ == '__main__':
    setup_logging()
    logger.info('Starting AutoQuake realtime example runner')

    print("AutoQuake Realtime System Examples")
    print("=" * 50)
    print()
    print("Available examples:")
    print("  1. example_full_pipeline()      - Complete step-by-step pipeline")
    print("  2. example_with_runner()        - Simplified with RealtimeRunner")
    print("  3. example_seedlink_integration() - Real-time SEEDLINK data")
    print()
    print("Uncomment the example you want to run below.")
    print()

    # Uncomment one of the following:
    # example_full_pipeline()
    example_with_runner()
    # example_seedlink_integration()