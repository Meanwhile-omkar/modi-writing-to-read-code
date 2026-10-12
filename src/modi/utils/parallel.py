"""Parallel map over files using a process pool.

The audit is CPU-bound (JPEG decode, md5, resize, gradients) and the machine has
multiple cores, so a process pool gives a near-linear speedup. Falls back to a
serial loop when workers=1, which keeps debugging and tracebacks simple.

Worker functions must be top-level (picklable) — not lambdas or closures.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List, Sequence

from tqdm import tqdm


def default_workers() -> int:
    """Leave one core free so the machine stays usable during an overnight run."""
    return max(1, (os.cpu_count() or 2) - 1)


def pmap(fn: Callable, items: Sequence, workers: int | None = None,
         desc: str = "", chunksize: int | None = None, ordered: bool = True) -> List:
    """Map `fn` over `items`, returning results in the original order.

    `fn` must be a top-level function taking one element of `items`.
    """
    items = list(items)
    if not items:
        return []
    if workers is None:
        workers = default_workers()

    if workers <= 1:
        return [fn(x) for x in tqdm(items, desc=desc or "work")]

    if chunksize is None:
        # Bigger chunks amortise IPC overhead; cap so progress stays smooth.
        chunksize = max(1, min(64, len(items) // (workers * 8) or 1))

    results: List = [None] * len(items)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        if ordered:
            # executor.map preserves order and is cheapest for uniform work.
            for i, r in enumerate(tqdm(
                ex.map(fn, items, chunksize=chunksize),
                total=len(items), desc=desc or f"work x{workers}",
            )):
                results[i] = r
        else:
            futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
            for fut in tqdm(as_completed(futs), total=len(items),
                            desc=desc or f"work x{workers}"):
                results[futs[fut]] = fut.result()
    return results
