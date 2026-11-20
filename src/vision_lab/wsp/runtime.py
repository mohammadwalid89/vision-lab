"""Process-level resource limits applied from inside Python.

Equivalent to ``taskset -c 1-3 nice -n 10`` with 3 BLAS/torch threads, for environments
where wrapping commands is not possible.  Call ``limit()`` before importing torch-heavy
code.  Default limit: at most 3 cores, nice 10.
"""
from __future__ import annotations

import os

DEFAULT_CORES = (1, 2, 3)


def limit(cores=DEFAULT_CORES, niceness: int = 10, threads: int | None = None) -> dict:
    # Override with WSP_CORES="1,2" and WSP_THREADS=3 on a smaller machine.
    if os.environ.get("WSP_CORES"):
        cores = tuple(int(c) for c in os.environ["WSP_CORES"].split(","))
    if os.environ.get("WSP_THREADS"):
        threads = int(os.environ["WSP_THREADS"])
    threads = threads or len(cores)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(threads)
    avail = os.sched_getaffinity(0)
    use = {c for c in cores if c in avail} or set(sorted(avail)[:len(cores)])
    os.sched_setaffinity(0, use)
    cur = os.nice(0)
    if cur < niceness:
        os.nice(niceness - cur)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass
    return dict(cores=sorted(os.sched_getaffinity(0)), nice=os.nice(0), threads=threads)
