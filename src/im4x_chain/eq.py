"""Parametric EQ, the first stage of the chain (before Leveler and Loudness Control).

Bands are written as short strings, as on the command line (``--eq``):

* ``hp:60`` / ``hp:60:4`` — high-pass (low cut) at 60 Hz, Butterworth, order 4 = 24 dB/oct
  (order 1-8; -3 dB at the frequency);
* ``lp:16000[:order]`` — low-pass (high cut);
* ``peak:3000:+2:1.0`` — bell at 3 kHz, +2 dB, Q 1.0;
* ``lowshelf:200:-2[:0.707]`` / ``highshelf:8000:+1.5[:0.707]`` — shelves (gain dB, slope Q).

Filters are causal IIR (minimum phase, like an analog EQ); peak/shelf use the RBJ cookbook
biquads. Default: a low cut matched to the voice (``voice_bands``: 0.55 x its low pitch,
70-120 Hz, 24 dB/oct) and ``lp:18000:8``; ``DEFAULT_BANDS`` = ``["hp:70", "lp:18000:8"]`` (the EQ
item of the user's RX Module Chain) is used when the pitch cannot be found.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from im4x_chain.chunks import chunked

DEFAULT_BANDS = ["hp:70", "lp:18000:8"]     # = the EQ item of the RX Module Chain
LOW_CUT_RANGE = (70.0, 120.0)               # voice-matched low cut: low voices 70 Hz, high ones up to 120
LOW_CUT_RATIO = 0.55                        # low cut = 0.55 x the voice's low pitch (5th percentile)
KINDS = ("hp", "lp", "peak", "lowshelf", "highshelf")


def parse_band(spec: str) -> dict[str, Any]:
    parts = [p.strip() for p in spec.strip().lower().split(":")]
    kind = {"highpass": "hp", "lowcut": "hp", "lowpass": "lp", "highcut": "lp", "bell": "peak"}.get(parts[0], parts[0])
    if kind not in KINDS or len(parts) < 2:
        raise ValueError(f"bad EQ band {spec!r}: use hp:FREQ[:ORDER], lp:FREQ[:ORDER], "
                         "peak:FREQ:GAIN[:Q], lowshelf:FREQ:GAIN[:Q], highshelf:FREQ:GAIN[:Q]")
    nums = [float(p) for p in parts[1:]]
    freq = nums[0]
    if kind in ("hp", "lp"):
        order = int(nums[1]) if len(nums) > 1 else 4
        if not 1 <= order <= 8:
            raise ValueError(f"bad EQ band {spec!r}: order must be 1-8")
        return {"type": kind, "freq": freq, "order": order}
    if len(nums) < 2:
        raise ValueError(f"bad EQ band {spec!r}: {kind} needs a gain in dB")
    return {"type": kind, "freq": freq, "gain_db": nums[1],
            "q": nums[2] if len(nums) > 2 else (1.0 if kind == "peak" else 0.707)}


def _biquad(band: dict[str, Any], sr: float) -> np.ndarray:
    a_ = 10 ** (band["gain_db"] / 40)
    w = 2 * np.pi * band["freq"] / sr
    cw, alpha = np.cos(w), np.sin(w) / (2 * band["q"])
    if band["type"] == "peak":
        b = [1 + alpha * a_, -2 * cw, 1 - alpha * a_]
        a = [1 + alpha / a_, -2 * cw, 1 - alpha / a_]
    else:
        s = 2 * np.sqrt(a_) * alpha
        sign = 1 if band["type"] == "lowshelf" else -1
        b = [a_ * ((a_ + 1) - sign * (a_ - 1) * cw + s), sign * 2 * a_ * ((a_ - 1) - sign * (a_ + 1) * cw),
             a_ * ((a_ + 1) - sign * (a_ - 1) * cw - s)]
        a = [(a_ + 1) + sign * (a_ - 1) * cw + s, -sign * 2 * ((a_ - 1) + sign * (a_ + 1) * cw),
             (a_ + 1) + sign * (a_ - 1) * cw - s]
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return np.concatenate([b, a])[None, :]


def design(bands: list[str | dict[str, Any]], sr: float) -> np.ndarray:
    """Second-order sections for all bands, in order."""
    secs = []
    for spec in bands:
        band = parse_band(spec) if isinstance(spec, str) else spec
        if not 0 < band["freq"] < sr / 2:
            raise ValueError(f"EQ frequency {band['freq']} Hz is outside 0..{sr / 2:g} Hz")
        if band["type"] in ("hp", "lp"):
            secs.append(butter(band["order"], band["freq"], "highpass" if band["type"] == "hp" else "lowpass",
                               fs=sr, output="sos"))
        else:
            secs.append(_biquad(band, sr))
    return np.concatenate(secs) if secs else np.empty((0, 6))


def equalize(x: np.ndarray, sr: float, bands: list[str] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the EQ ``bands`` (default ``DEFAULT_BANDS``) to ``x`` (``(n,)`` or ``(n, ch)``)."""
    bands = DEFAULT_BANDS if bands is None else bands
    parsed = [parse_band(b) for b in bands]
    if not parsed:
        return np.asarray(x, dtype=np.float64).copy(), {"bands": []}
    base, high = _split(parsed, sr)
    sos_base = design(base, sr) if base else None
    sos_high = design(high, 2 * sr) if high else None

    def run(y: np.ndarray) -> np.ndarray:
        if sos_base is not None:
            y = sosfilt(sos_base, y, axis=0)
        if sos_high is not None:                   # near Nyquist: filter at 2x the rate
            n = len(y)
            u = sosfilt(sos_high, resample_poly(y, 2, 1, axis=0), axis=0)
            y = resample_poly(u, 1, 2, axis=0)[:n]
        return y

    # 2 s pre-roll: the IIR state of any band settles far below the 64-bit rounding level
    y = chunked(run, np.asarray(x, dtype=np.float64), pre=int(2 * sr), post=int(0.05 * sr))
    return y, {"bands": parsed}


def _split(parsed: list[dict[str, Any]], sr: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cut filters above sr/4 are run at twice the sample rate: a digital Butterworth near
    Nyquist gets much steeper than its nominal slope (an 18 kHz 48 dB/oct low-pass at 48 kHz
    is -18 dB at 19 kHz instead of about -6). At 2x the rate its shape is close to the analog
    filter, like the high cut of the RX EQ."""
    high = [b for b in parsed if b["type"] in ("hp", "lp") and b["freq"] > sr / 4]
    return [b for b in parsed if b not in high], high


def response_db(bands: list[str], sr: float, freqs: np.ndarray) -> np.ndarray:
    """Magnitude response (dB) of the chain at ``freqs`` Hz, for checks and reports."""
    from scipy.signal import sosfreqz
    base, high = _split([parse_band(b) for b in bands], sr)
    out = np.zeros(len(np.asarray(freqs)))
    for part, rate in ((base, sr), (high, 2 * sr)):
        if part:
            _, h = sosfreqz(design(part, rate), worN=np.asarray(freqs, float), fs=rate)
            out += 20 * np.log10(np.abs(h) + 1e-30)
    return out


def voice_pitch(x: np.ndarray, sr: float, max_frames: int = 20000) -> dict[str, float] | None:
    """Pitch (F0) statistics of the voiced speech in ``x``: autocorrelation on 64 ms frames at
    16 kHz over the louder 40 % of frames; a frame counts as voiced when the normalised
    autocorrelation peak in 55-800 Hz is above 0.75. The highest peak is taken, or the shortest 1/2-1/5 of
    its lag where the autocorrelation is within 85 % of it (a subharmonic would lower the pitch). None when there is no voiced speech."""
    v = np.asarray(x, dtype=np.float64)
    v = v.mean(1) if v.ndim == 2 else v
    fr = 16000
    v = resample_poly(v, fr, int(sr)) if int(sr) != fr else v
    n, hop = 1024, 512                            # 64 ms: the longest period (55 Hz) is < n/3
    m = (len(v) - n) // hop
    if m < 10:
        return None
    starts = np.arange(m) * hop
    lv = 10 * np.log10(np.array([np.mean(v[a:a + n] ** 2) for a in starts]) + 1e-20) if m < 2000 else \
        10 * np.log10(np.add.reduceat(v[: m * hop] ** 2, np.arange(0, m * hop, hop)) / hop + 1e-20)
    alive = lv > -100
    if alive.sum() < 10:
        return None
    keep = np.flatnonzero(alive & (lv > np.percentile(lv[alive], 60)))
    if len(keep) > max_frames:
        keep = np.sort(np.random.default_rng(0).choice(keep, max_frames, replace=False))
    frames = v[starts[keep][:, None] + np.arange(n)] * np.hanning(n)
    frames -= frames.mean(1, keepdims=True)
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames, 2 * n)) ** 2)[:, :n]
    ac /= ac[:, :1] + 1e-20
    w = np.hanning(n)                             # undo the window's own decay (Boersma 1993)
    aw = np.fft.irfft(np.abs(np.fft.rfft(w, 2 * n)) ** 2)[:n]
    lo, hi = fr // 800, fr // 55
    ac[:, :hi] /= (aw[:hi] / aw[0])[None, :]
    seg = ac[:, lo:hi]
    peak = seg.max(1)
    voiced = peak > 0.75
    if voiced.sum() < 50:
        return None
    ac, peak = ac[voiced], peak[voiced]
    lag = np.argmax(seg[voiced], axis=1) + lo
    rows, best = np.arange(len(lag)), lag.copy()
    for div in (2, 3, 4, 5):                      # subharmonic check: a peak just as high at
        part = np.round(lag / div).astype(int)    # 1/div of the period means F0 is div x higher
        at = np.maximum.reduce([ac[rows, np.clip(part + d, 0, n - 1)] for d in (-1, 0, 1)])
        best = np.where((part >= lo) & (at >= 0.85 * peak), part, best)
    lag = best
    f0 = fr / lag
    p5, med, p95 = np.percentile(f0, [5, 50, 95])
    return {"f0_p5_hz": round(float(p5), 1), "f0_median_hz": round(float(med), 1),
            "f0_p95_hz": round(float(p95), 1), "voiced_frames": int(len(f0))}


def voice_bands(x: np.ndarray, sr: float) -> tuple[list[str], dict[str, Any]]:
    """Default EQ with the low cut matched to the voice: 0.55 x its low pitch (5th
    percentile), 70-120 Hz in 5 Hz steps. A low male voice keeps 70 Hz (the RX chain value),
    a high female or child voice gets up to 120 Hz: rumble and plosive energy go, the voice's
    fundamental stays untouched (a 4th-order cut at 0.55 x F0 is under -0.1 dB at F0)."""
    pitch = voice_pitch(x, sr)
    if pitch is None:
        return list(DEFAULT_BANDS), {"low_cut_hz": float(DEFAULT_BANDS[0].split(":")[1]), "pitch": None}
    lo, hi = LOW_CUT_RANGE
    hz = float(np.clip(round(LOW_CUT_RATIO * pitch["f0_p5_hz"] / 5) * 5, lo, hi))
    return [f"hp:{hz:g}"] + DEFAULT_BANDS[1:], {"low_cut_hz": hz, "pitch": pitch}
