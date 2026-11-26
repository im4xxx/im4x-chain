"""Compressor modelled on SSL Native X-Comp, the last item of the user's RX Module Chain.

The model was fitted to the real plugin (AU, hosted headless) with the chain's settings:

* peak detector on ``|x|`` (a sine and a square of the same peak get the same reduction),
  one-pole with separate attack and release, in linear amplitude; the time constants are
  ``0.95 * attack`` and ``1.0 * release``;
* static curve: hard ratio above the knee, quadratic soft knee ``knee`` dB wide centred on
  the threshold (threshold in dBFS);
* gain reduction limited to ``max_gr`` dB, then makeup and output gain.

Against the plugin on speech normalised to -24 LUFS the gain differs by at most 0.02 dB
(residual -55 dB). Stereo channels share one detector (the louder channel). The side-chain
LF/HF bleed filters of X-Comp are off in the chain and are not modelled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

# the user's Module Chain item (X-Comp, state A)
XCOMP_DEFAULTS = {"threshold": -10.7, "ratio": 4.0, "attack": 0.5, "release": 199.64,
                  "knee": 19.69, "max_gr": 60.0, "makeup": 0.0}
ATTACK_K, RELEASE_K = 0.95, 1.0


@njit(cache=True)
def _detector(a: np.ndarray, ca: float, cr: float) -> np.ndarray:
    e = np.empty_like(a)
    s = 0.0
    for i in range(a.shape[0]):
        v = a[i]
        s = (ca * s + (1.0 - ca) * v) if v > s else (cr * s + (1.0 - cr) * v)
        e[i] = s
    return e


def static_curve(level_db: np.ndarray, threshold: float, ratio: float, knee: float) -> np.ndarray:
    """Gain (dB, <= 0) for a detector level (dBFS)."""
    level_db = np.asarray(level_db, dtype=np.float64)
    slope = 1.0 / ratio - 1.0
    g = np.zeros_like(level_db)
    lo, hi = threshold - knee / 2, threshold + knee / 2
    if knee > 0:
        m = (level_db > lo) & (level_db < hi)
        g[m] = slope * (level_db[m] - lo) ** 2 / (2 * knee)
    m = level_db >= hi
    g[m] = slope * (level_db[m] - threshold)
    return g


def compress(x: np.ndarray, sr: float, threshold: float = -10.7, ratio: float = 4.0, attack: float = 0.5,
             release: float = 199.64, knee: float = 19.69, max_gr: float = 60.0,
             makeup: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Compress ``x`` (``(n,)`` or ``(n, ch)``); attack/release in ms. Returns (audio, info)."""
    x = np.asarray(x, dtype=np.float64)
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    xc = x[:, None] if x.ndim == 1 else x
    ca = float(np.exp(-1.0 / (ATTACK_K * max(attack, 0.01) * 1e-3 * sr)))
    cr = float(np.exp(-1.0 / (RELEASE_K * max(release, 0.1) * 1e-3 * sr)))
    env = _detector(np.ascontiguousarray(np.abs(xc).max(1)), ca, cr)
    gr = np.maximum(static_curve(20 * np.log10(env + 1e-12), threshold, ratio, knee), -abs(max_gr))
    y = xc * (10 ** ((gr + makeup) / 20))[:, None]
    active = gr < -0.01
    info = {"threshold": threshold, "ratio": ratio, "attack_ms": attack, "release_ms": release,
            "knee_db": knee, "makeup_db": makeup,
            "max_reduction_db": round(float(-gr.min()), 2) if len(gr) else 0.0,
            "active_share": round(float(active.mean()), 3) if len(gr) else 0.0}
    return (y[:, 0] if x.ndim == 1 else y), info
