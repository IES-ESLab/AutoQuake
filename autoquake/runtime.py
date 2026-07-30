from __future__ import annotations

import os

from threadpoolctl import threadpool_limits


def limit_native_threads(limit: int = 1):
    """Apply a persistent native thread limit in the current worker."""
    if limit < 1:
        raise ValueError('Native thread limit must be at least 1')
    return threadpool_limits(limits=limit)


def visible_cpu_count() -> int:
    """Return CPUs available to this process, respecting scheduler affinity."""
    if hasattr(os, 'sched_getaffinity'):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def effective_workers(requested: int, task_count: int | None = None) -> int:
    """Clamp a requested worker count to visible CPUs and available tasks."""
    if requested < 1:
        raise ValueError('Worker count must be at least 1')
    workers = min(requested, visible_cpu_count())
    if task_count is not None:
        workers = min(workers, max(1, task_count))
    return workers
