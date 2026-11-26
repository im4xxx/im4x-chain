"""Run a filter on a long signal in parallel pieces.

scipy's ``sosfilt``/``resample_poly`` release the GIL, so threads scale over the cores. Each
piece starts ``pre`` samples early (the IIR state settles; the extra output is dropped) and
reads ``post`` samples past its end (look-ahead of FIR filters). With a pre-roll much longer
than the filters' decay the result equals one pass over the whole signal to rounding error.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def chunked(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, pre: int, post: int = 0,
            parts: int | None = None) -> np.ndarray:
    """``fn(x)`` (same length, along axis 0) computed in ``parts`` overlapping pieces."""
    n = len(x)
    parts = parts or min(os.cpu_count() or 1, 8)
    if parts < 2 or n < parts * 4 * max(pre + post, 1):
        return fn(x)
    edges = np.linspace(0, n, parts + 1).astype(int)

    def job(i: int) -> np.ndarray:
        a, b = int(edges[i]), int(edges[i + 1])
        s, e = max(0, a - pre), min(n, b + post)
        return fn(x[s:e])[a - s: b - s]

    with ThreadPoolExecutor(parts) as ex:
        return np.concatenate(list(ex.map(job, range(parts))), axis=0)
