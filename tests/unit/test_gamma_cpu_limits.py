from __future__ import annotations

import pytest

np = pytest.importorskip('numpy')
gamma_utils = pytest.importorskip(
    'autoquake.GaMMA.gamma.utils',
    reason='GaMMA submodule is not initialized',
)

pytestmark = pytest.mark.unit


def test_gamma_worker_initializer_limits_native_threads(monkeypatch):
    observed = []
    marker = object()
    monkeypatch.setattr(
        gamma_utils,
        'threadpool_limits',
        lambda *, limits: observed.append(limits) or marker,
    )

    gamma_utils._init_association_worker()

    assert observed == [1]
    assert gamma_utils._worker_thread_limiter is marker


def test_dbscan_uses_configured_job_count(monkeypatch):
    observed_jobs = []

    class FakeDBSCAN:
        def __init__(self, *, n_jobs, **kwargs):
            observed_jobs.append(n_jobs)

        def fit(self, data, sample_weight=None):
            self.labels_ = np.zeros(len(data), dtype=int)
            return self

    monkeypatch.setattr(gamma_utils, 'DBSCAN', FakeDBSCAN)
    gamma_utils.hierarchical_dbscan_clustering(
        data=np.array([[0.0], [1.0]]),
        phase_loc=np.zeros((2, 3)),
        phase_type=np.array(['p', 's']),
        phase_weight=np.ones((2, 1)),
        vel={'p': 6.0, 's': 3.5},
        min_cluster_size=100,
        n_jobs=3,
    )

    assert observed_jobs == [3]


def test_gamma_association_is_stable_across_worker_counts(monkeypatch):
    observed_dbscan_jobs = []
    observed_pool_sizes = []
    observed_initializers = []

    def fake_convert_picks_csv(_picks, _stations, _config):
        return (
            np.arange(4, dtype=float)[:, np.newaxis],
            np.zeros((4, 3)),
            np.array(['p', 's', 'p', 's']),
            np.ones((4, 1)),
            np.arange(4),
            np.array(['A_P', 'A_S', 'B_P', 'B_S']),
            0.0,
        )

    def fake_dbscan(*_args, n_jobs, **_kwargs):
        observed_dbscan_jobs.append(n_jobs)
        return np.array([2, 0, 2, 0])

    def fake_associate(label, *_args):
        return (
            [{'event_index': 0, 'cluster_label': int(label)}],
            [(int(label), 0, 0.9)],
        )

    class FakePool:
        def __init__(self, processes, initializer):
            observed_pool_sizes.append(processes)
            observed_initializers.append(initializer)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def starmap(self, function, tasks, chunksize):
            assert chunksize == 1
            return [function(*task) for task in tasks]

    class FakeContext:
        @staticmethod
        def Pool(processes, initializer):
            return FakePool(processes, initializer)

    monkeypatch.setattr(gamma_utils, 'convert_picks_csv', fake_convert_picks_csv)
    monkeypatch.setattr(
        gamma_utils, 'hierarchical_dbscan_clustering', fake_dbscan
    )
    monkeypatch.setattr(gamma_utils, 'associate', fake_associate)
    monkeypatch.setattr(gamma_utils.mp, 'get_context', lambda _name: FakeContext())
    monkeypatch.setattr(
        gamma_utils,
        '_effective_workers',
        lambda requested, task_count=None: min(requested, task_count or requested),
    )

    base_config = {
        'min_picks_per_eq': 1,
        'vel': {'p': 6.0, 's': 3.5},
        'eikonal': None,
        'use_dbscan': True,
        'dbscan_eps': 17.0,
        'dbscan_min_samples': 3,
    }
    results = []
    for workers in (1, 10):
        config = {**base_config, 'ncpu': workers}
        results.append(
            gamma_utils.association(None, None, config, event_idx0=7)
        )

    assert results[0] == results[1]
    assert results[0] == (
        [
            {'event_index': 7, 'cluster_label': 0},
            {'event_index': 8, 'cluster_label': 2},
        ],
        [(0, 7, 0.9), (2, 8, 0.9)],
    )
    assert observed_dbscan_jobs == [1, 1]
    assert observed_pool_sizes == [1, 2]
    assert observed_initializers == [
        gamma_utils._init_association_worker,
        gamma_utils._init_association_worker,
    ]
