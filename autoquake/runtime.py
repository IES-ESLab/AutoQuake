from __future__ import annotations

import os

from threadpoolctl import threadpool_limits

_worker_thread_limiter = None


def limit_native_threads(limit: int = 1):
    """Apply a persistent native thread limit in the current worker."""
    if limit < 1:
        raise ValueError('Native thread limit must be at least 1')
    return threadpool_limits(limits=limit)


def configure_torch_threads(
    torch_module, requested: int, device: str
) -> tuple[int, bool]:
    """Set independent inference and inter-op thread limits."""
    effective = effective_workers(requested) if device == 'cpu' else 1
    torch_module.set_num_threads(effective)
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        return effective, False
    return effective, True


def initialize_torch_worker(torch_module) -> None:
    """Keep a DataLoader worker on one native and one PyTorch thread."""
    global _worker_thread_limiter
    _worker_thread_limiter = limit_native_threads(1)
    torch_module.set_num_threads(1)


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
