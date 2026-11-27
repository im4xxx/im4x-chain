"""Find a noise-only region in an audio file for Spectral De-noise "Learn".

Strategy (what an editor does by hand):

1. Frame the file (STFT) and estimate the noise floor as a low percentile of frame levels.
2. Mark frames as "noise" when their level is close to the floor and their spectrum is close
   to the median noise spectrum (no speech/tonal events). Digital silence is excluded.
3. Join noise frames into runs and trim a guard band on both sides (speech tails).
4. Prefer a run at the very start, then at the very end, then the best pause between
   phrases. Within each group, longer and more stationary runs win.

Every candidate carries the expected flatness of RX's residual-noise curve: the spread
(in dB) of the region's spectrum around its smoothed shape. A clean noise region gives a
flat residual; speech or clicks inside the region show up as bumps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

FRAME = 2048
HOP = 512


@dataclass
class NoiseRegion:
    start: float          # seconds
    end: float            # seconds
    where: str            # "start" | "end" | "pause"
    level_db: float       # mean frame level, dBFS
    floor_db: float       # file noise floor estimate, dBFS
    stationarity_db: float  # mean per-bin std over time (lower = steadier noise)
    roughness_db: float   # spectrum deviation from its smoothed shape (residual flatness proxy)
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d


def _stft_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_level_db [n], spectrum_db [n, bins])."""
    if len(x) < FRAME:                              # shorter than one frame: pad with silence
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    win = np.hanning(FRAME).astype(np.float32)
    levels = np.empty(n, dtype=np.float32)
    specs = np.empty((n, FRAME // 2 + 1), dtype=np.float32)
    block = 4096
    for b0 in range(0, n, block):
        idx = np.arange(b0, min(n, b0 + block))
        frames = np.stack([x[i * HOP: i * HOP + FRAME] for i in idx])
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        levels[idx] = 20 * np.log10(rms + 1e-12)
        mag = np.abs(np.fft.rfft(frames * win, axis=1))
        specs[idx] = 20 * np.log10(mag + 1e-9)
    return levels, specs


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _smooth(v: np.ndarray, width: int = 31) -> np.ndarray:
    k = np.ones(width) / width
    pad = np.pad(v, (width // 2, width // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def find_noise_regions(
    x: np.ndarray,
    sr: int,
    min_duration: float = 0.25,
    max_duration: float = 4.0,
    edge_window: float = 15.0,
    guard: float = 0.06,
    floor_margin_db: float = 6.0,
    max_candidates: int = 8,
) -> dict[str, Any]:
    """Return noise-only candidate regions of ``x`` (``(n,)`` or ``(n, ch)``; channels are
    averaged), best first."""
    x = np.asarray(x, dtype=np.float32)
    x = x.mean(axis=1) if x.ndim == 2 else x
    duration = len(x) / sr
    levels, specs = _stft_db(x)
    t = (np.arange(len(levels)) * HOP + FRAME / 2) / sr

    audible = levels > -130.0                       # exclude digital silence
    if not audible.any():
        return {"ok": False, "error": "file is digital silence", "candidates": []}
    floor = float(np.percentile(levels[audible], 10))
    quiet = audible & (levels < floor + floor_margin_db)
    if quiet.sum() < 4:
        return {"ok": False, "error": "no quiet frames found", "floor_db": floor, "candidates": []}

    # reference noise spectrum = median of the quietest frames; frames whose spectrum
    # deviates a lot from it in any band (speech harmonics, clicks) are not noise
    ref = np.median(specs[quiet], axis=0)
    band = slice(4, specs.shape[1] - 4)
    excess = np.percentile(specs[:, band] - ref[band], 95, axis=1)   # loudest bands vs noise
    noise = quiet & (excess < 12.0)

    # "start"/"end" mean the beginning/end of the audible content (files may be padded or
    # gated with digital silence)
    aud_idx = np.flatnonzero(audible)
    content_start, content_end = float(t[aud_idx[0]]), float(t[aud_idx[-1]])
    guard_frames = int(round(guard * sr / HOP))
    min_frames = int(np.ceil(min_duration * sr / HOP))
    cands: list[NoiseRegion] = []
    for a, b in _runs(noise):
        a2, b2 = a + guard_frames if a > 0 else a, b - guard_frames if b < len(levels) else b
        if b2 - a2 < min_frames:
            continue
        # cap length but keep the steadiest stretch: slide a max_duration window
        max_frames = int(max_duration * sr / HOP)
        best = (a2, min(b2, a2 + max_frames))
        if b2 - a2 > max_frames:
            lv = levels[a2:b2]
            csum = np.concatenate([[0], np.cumsum(lv)])
            means = (csum[max_frames:] - csum[:-max_frames]) / max_frames
            off = int(np.argmin(means))
            best = (a2 + off, a2 + off + max_frames)
        s0, s1 = best
        seg = specs[s0:s1]
        mean_spec = seg.mean(axis=0)
        stationarity = float(seg[:, band].std(axis=0).mean())
        roughness = float(np.abs(mean_spec[band] - _smooth(mean_spec)[band]).mean())
        t0, t1 = float(t[s0] - FRAME / 2 / sr), float(t[s1 - 1] + FRAME / 2 / sr)
        t0, t1 = max(0.0, t0), min(duration, t1)
        if t0 - content_start <= edge_window:
            where = "start"
        elif content_end - t1 <= edge_window:
            where = "end"
        else:
            where = "pause"
        level = float(levels[s0:s1].mean())
        dur = t1 - t0
        group_bonus = {"start": 30.0, "end": 20.0, "pause": 0.0}[where]
        score = group_bonus + 4.0 * min(dur, max_duration) - 3.0 * stationarity - 2.0 * roughness \
            - 1.5 * max(0.0, level - floor)
        cands.append(NoiseRegion(t0, t1, where, level, floor, stationarity, roughness, score))

    cands.sort(key=lambda c: c.score, reverse=True)
    return {
        "ok": bool(cands),
        "duration": round(duration, 3),
        "sample_rate": sr,
        "floor_db": round(floor, 2),
        "candidates": [c.as_dict() for c in cands[:max_candidates]],
        "content": [round(content_start, 3), round(content_end, 3)],
        **({} if cands else {"error": f"no noise-only stretch of {min_duration}s found"}),
    }
