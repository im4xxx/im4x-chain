"""Verdict on a residual-noise curve: flat and below -120 dB (the rule an editor uses on the
yellow Residual curve of RX Spectral De-noise)."""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_DB = -120.0          # whole curve must stay below this
FLATNESS_DB = 6.0        # p95 - p5 spread over the band: the blue curve is tuned to flatten it
BUMP_DB = 6.0            # largest local bump above the smoothed curve
BAND_HZ = (60.0, 16000.0)


def verdict(freqs_a: np.ndarray, dbs_a: np.ndarray) -> dict[str, Any]:
    """Flat / below -120 dB verdict for a residual curve given as (Hz, dB) points."""
    band = (freqs_a >= BAND_HZ[0]) & (freqs_a <= BAND_HZ[1])
    curve = dbs_a[band] if band.sum() >= 20 else dbs_a
    kern = np.ones(15) / 15
    smooth = np.convolve(np.pad(curve, 7, mode="edge"), kern, mode="valid")
    max_db = float(curve.max())
    spread = float(np.percentile(curve, 95) - np.percentile(curve, 5))
    bump = float((curve - smooth).max())
    below = max_db < MAX_DB
    flat = spread <= FLATNESS_DB and bump <= BUMP_DB
    # a coarse curve for reports: one point per 1/3 octave
    edges = 10 ** np.arange(np.log10(BAND_HZ[0]), np.log10(BAND_HZ[1]) + 1e-9, 1 / 9)
    coarse = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (freqs_a >= lo) & (freqs_a < hi)
        if sel.any():
            coarse.append([round(float(np.sqrt(lo * hi))), round(float(dbs_a[sel].max()), 1)])
    return {
        "ok": below and flat,
        "below_-120": below,
        "flat": flat,
        "max_db": round(max_db, 1),
        "mean_db": round(float(curve.mean()), 1),
        "spread_db": round(spread, 1),
        "max_bump_db": round(bump, 1),
        "criteria": {"max_db": MAX_DB, "spread_db": FLATNESS_DB, "bump_db": BUMP_DB, "band_hz": BAND_HZ},
        "curve": coarse,
        "raw": {"freq": [round(float(x), 1) for x in freqs_a], "db": [round(float(x), 2) for x in dbs_a]},
    }
