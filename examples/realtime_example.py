"""
AutoQuake Realtime System - Example Usage

This script demonstrates how to use the realtime earthquake detection system
in a modular, programmatic way.

Processing flow:
    Data ingestion → Picking (polarity, amplitude) → Association + preliminary magnitude
    → Relocation → Updated magnitude → Focal mechanism
"""

from pathlib import Path
from datetime import timedelta

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


def example_full_pipeline():
    """
    Example: Complete earthquake processing pipeline.

    Flow:
        1. Data ingestion (SEEDLINK or file)
        2. Picking (PhaseNet with polarity and amplitude)
        3. Association + preliminary magnitude (GaMMA)
        4. Relocation (H3DD)
        5. Updated magnitude (WA-simulated)
        6. Focal mechanism (GAfocal)
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
    # 3. Association - GaMMA + preliminary magnitude
    # =========================================================================
    associator = RealtimeGaMMA(
        station=station_file,
        result_path=result_path,
        pick_buffer=pick_buffer,
        center=(121.5, 24.0),
        xlim_degree=[120.0, 123.0],
        ylim_degree=[22.0, 26.0],
    )

    magnitude_estimator = RealtimeMagnitude(
        station_info=station_file,
        pz_dir=Path('path/to/pz_files'),
        use_wa_simulation=True,
    )

    # =========================================================================
    # 4. Relocation - H3DD
    # =========================================================================
    relocator = RealtimeRelocator(
        station_file=station_file,
        model_3d=Path('path/to/velocity_model.txt'),
        h3dd_dir=Path('path/to/H3DD'),
    )

    # =========================================================================
    # 6. Focal mechanism - GAfocal
    # =========================================================================
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
        # Step 2: Picking (with polarity and amplitude for DP table)
        # ---------------------------------------------------------------------
        picks = picker.check_and_process()

        # ---------------------------------------------------------------------
        # Step 3: Association + preliminary magnitude
        # ---------------------------------------------------------------------
        if pick_buffer.should_trigger():
            events = associator.associate()

            for event in events:
                # Get associated picks for this event
                event_picks = [
                    p for p in picks
                    if p.get('event_index') == event['event_index']
                ]

                # Preliminary magnitude (fast estimate)
                prelim_mag = magnitude_estimator.estimate_preliminary(event, event_picks)
                if prelim_mag is not None:
                    event['magnitude_preliminary'] = prelim_mag
                    print(f"Event detected: M{prelim_mag:.1f} (preliminary)")

                # -----------------------------------------------------------------
                # Step 4: Relocation
                # -----------------------------------------------------------------
                relocated = relocator.relocate_single(event, event_picks)
                if relocated:
                    print(f"  Relocated: {relocated['latitude']:.4f}, "
                          f"{relocated['longitude']:.4f}, {relocated['depth_km']:.1f} km")
                else:
                    relocated = event  # Fallback to original location

                # -----------------------------------------------------------------
                # Step 5: Updated magnitude (using relocated position)
                # -----------------------------------------------------------------
                final_mag = magnitude_estimator.estimate_from_relocation(
                    relocated, event_picks
                )
                if final_mag is not None:
                    event['magnitude'] = final_mag
                    print(f"  Final magnitude: M{final_mag:.1f}")

                # -----------------------------------------------------------------
                # Step 6: Focal mechanism
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

    RealtimeRunner handles the complete pipeline internally,
    including the refine_event() method for post-association processing.
    """
    config = RealtimeConfig(
        station_file=Path('path/to/station.csv'),
        result_path=Path('./realtime_results'),
        # Association settings
        center=(121.5, 24.0),
        xlim_degree=[120.0, 123.0],
        ylim_degree=[22.0, 26.0],
        min_picks_per_eq=8,
        # Enable all refinement stages
        enable_magnitude=True,
        enable_relocation=True,
        enable_focal=True,
        # Paths for refinement components
        model_3d=Path('path/to/velocity_model.txt'),
        h3dd_dir=Path('path/to/H3DD'),
        pz_dir=Path('path/to/pz_files'),
        gafocal_dir=Path('path/to/GAfocal'),
    )

    runner = RealtimeRunner(config)

    # Option A: Run simulation with existing picks
    runner.run_simulation(
        picks_file=Path('path/to/picks.csv'),
        speed=10.0,
        max_duration=300,
    )

    # Option B: Manual control with refine_event()
    # events = runner.process_once()
    # for event in events:
    #     refined = runner.refine_event(event, associated_picks)


def example_seedlink_integration():
    """
    Example: Real-time data from SEEDLINK server.

    This shows how to connect to a SEEDLINK server and process
    continuous waveform data.
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

    # Associator
    associator = RealtimeGaMMA(
        station=Path('path/to/station.csv'),
        result_path=Path('./results'),
        pick_buffer=pick_buffer,
        center=(121.5, 24.0),
    )

    # Start SEEDLINK streaming
    client.start()

    try:
        while True:
            # Picker processes waveforms and outputs picks with polarity
            picker.check_and_process()

            # Association
            if pick_buffer.should_trigger():
                events = associator.associate()
                for event in events:
                    print(f"Event: {event['time']} at "
                          f"{event['latitude']:.3f}, {event['longitude']:.3f}")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.stop()


if __name__ == '__main__':
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
    # example_with_runner()
    # example_seedlink_integration()