# Test Fixtures

This directory contains test data for AutoQuake integration tests.

## Directory Structure

```
fixtures/
├── station.csv              # Station metadata (provided)
├── phasenet_picks.csv       # Sample PhaseNet output (provided)
├── polarity_picks.csv       # Sample polarity picks (provided)
├── mag_events.csv           # Sample magnitude events (provided)
├── mag_picks.csv            # Sample magnitude picks (provided)
├── test_params.json         # Test configuration with two flows (provided)
├── sac_data/                # SAC waveform files (YOU NEED TO ADD)
│   └── YYYYMMDD/
│       └── *.SAC
└── pz_data/                 # Pole-zero response files (YOU NEED TO ADD)
    └── *.PZ
```

## Required Test Data (User-Provided)

### SAC Data (`sac_data/`)

Add a small set of SAC files for testing the PhaseNet → Magnitude → DitingMotion flow.

Requirements:
- At least 2-3 stations
- 3-component data (BHE, BHN, BHZ or similar)
- Organized by date: `sac_data/YYYYMMDD/*.SAC`
- Matching stations in `station.csv`

Example structure:
```
sac_data/
└── 20240401/
    ├── TW.TATO.00.BHE.SAC
    ├── TW.TATO.00.BHN.SAC
    ├── TW.TATO.00.BHZ.SAC
    ├── TW.HGSD.00.BHE.SAC
    ├── TW.HGSD.00.BHN.SAC
    └── TW.HGSD.00.BHZ.SAC
```

### PZ Data (`pz_data/`)

Add corresponding pole-zero files for instrument response removal.

Example:
```
pz_data/
├── RESP.TW.TATO.00.BHZ
├── RESP.TW.HGSD.00.BHZ
└── ...
```

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration -v

# Run only CSV-based tests (no SAC data needed)
pytest tests/integration/test_flow_csv.py -v

# Run only SAC-based tests (requires SAC data)
pytest tests/integration/test_flow_sac.py -v
```

## Implementing Helper Functions

Before running the full integration tests, implement these helper functions in `tests/helpers.py`:

1. `for_mag_format()` - Convert PhaseNet picks to Magnitude input format
2. `for_dt_format()` - Convert PhaseNet picks to DitingMotion input format

See `tests/helpers.py` for function signatures and documentation.
