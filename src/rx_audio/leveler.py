"""Speech leveler, a replacement for the RX Leveler module.

Parameters mirror RX (Target Level, Responsiveness, Preserve Dynamic Range, Ess Reduction):

* loudness is K-weighted and measured on speech only, so pauses do not pull it;
* speech is split into phrases (pauses of ``min_pause`` or longer); each phrase gets its own
  gain, which rides the phrase's loudness only slowly inside long phrases (a Gaussian window of
  ~6 / responsiveness seconds that does not cross into neighbouring phrases);
* ``preserve`` % of a phrase's deviation from the file's speech level is kept, the rest is
  levelled, and the speech level is brought to ``target``;
* gain changes between phrases happen in the pause between them, not inside the words;
* sibilance is reduced only above 4.5 kHz (split band, exact reconstruction), by up to
  ``ess_db``; RX lowers the whole signal on "s" sounds instead.

Like RX, the result is meant to be followed by Loudness Control for the final level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, convolve1d, find_objects, label, minimum_filter1d
from scipy.signal import butter, sosfiltfilt

from rx_audio.loudness import k_weight

HOP = 0.01   # analysis frame, s


def _frames(p: np.ndarray, sr: float) -> np.ndarray:
    n = int(HOP * sr)
    k = len(p) // n
    return p[: k * n].reshape(k, n).mean(1)


def _speech_mask(loud: np.ndarray, margin_db: float) -> np.ndarray:
    alive = loud > -100
    if not alive.any():
        return np.zeros(len(loud), bool)
    floor = np.percentile(loud[alive], 10)
    sp = alive & (loud > max(floor + margin_db, np.percentile(loud[alive], 30)))
    return binary_dilation(binary_closing(sp, np.ones(15)), np.ones(11))   # bridge 150 ms, pad 50 ms


def leveler(x: np.ndarray, sr: float, target: float = -24.0, responsiveness: float = 3.3,
            preserve: float = 65.0, ess_db: float = 4.0, max_boost: float = 12.0, max_cut: float = 12.0,
            speech_margin_db: float = 12.0, min_pause: float = 0.25) -> tuple[np.ndarray, dict[str, Any]]:
    """Level speech in ``x`` (mono ``(n,)`` or ``(n, ch)``). Returns (audio, info)."""
    x = np.asarray(x, dtype=np.float64)
    xc = x[:, None] if x.ndim == 1 else x
    n = int(HOP * sr)
    pk = _frames((k_weight(xc, sr) ** 2).sum(1), sr)
    loud = 10 * np.log10(pk + 1e-20) - 0.691
    speech = _speech_mask(loud, speech_margin_db)
    w = speech.astype(float)
    if w.sum() < 10:
        return x.copy(), {"note": "no speech found", "changed": False}

    phrases, count = label(binary_closing(speech, np.ones(max(1, int(min_pause / HOP)))))
    sigma = max(0.5, 6.0 / max(responsiveness, 0.1)) / HOP
    speech_level = 10 * np.log10((pk * w).sum() / w.sum()) - 0.691
    keep = float(np.clip(preserve / 100, 0, 1))
    r = int(4 * sigma) + 1
    kern = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    gdb = np.full(len(pk), np.nan)
    for i, sl in enumerate(find_objects(phrases), start=1):
        if sl is None:
            continue
        a, b = sl[0].start, sl[0].stop                              # phrase span; weights are 0 outside,
        m = phrases[a:b] == i                                       # so only the span matters (O(n), not
        ww = w[a:b] * m                                             # O(n * phrases) on long files)
        if ww.sum() < 3:
            continue
        num = convolve1d(pk[a:b] * ww, kern, mode="constant")
        den = convolve1d(ww, kern, mode="constant")
        lp = 10 * np.log10(num[m] / np.maximum(den[m], 1e-12) + 1e-20) - 0.691
        gdb[a:b][m] = (target - speech_level) - (lp - speech_level) * (1 - keep)
    idx = np.arange(len(gdb))
    valid = np.isfinite(gdb)
    gdb = np.interp(idx, idx[valid], gdb[valid])                    # pauses: ramp phrase to phrase
    base = target - speech_level
    gdb = np.clip(gdb, base - max_cut, base + max_boost)
    k = np.hanning(7)
    gdb = np.convolve(np.pad(gdb, 3, mode="edge"), k / k.sum(), "valid")
    g = np.interp(np.arange(len(xc)) / n - 0.5, idx, 10 ** (gdb / 20))
    y = xc * g[:, None]
    dev = gdb[speech] - base
    info: dict[str, Any] = {"changed": True, "speech_level_in": round(float(speech_level), 2),
                            "base_gain_db": round(float(base), 2), "phrases": int(count),
                            "phrase_gain_db_p5_p95": [round(float(np.percentile(dev, 5)), 2),
                                                      round(float(np.percentile(dev, 95)), 2)]}
    if ess_db > 0:
        y, frames = deess(y, sr, speech, ess_db)
        info["ess_frames"] = frames
    return (y[:, 0] if x.ndim == 1 else y), info


def deess(y: np.ndarray, sr: float, speech: np.ndarray, max_db: float,
          split: float = 4500.0) -> tuple[np.ndarray, int]:
    """Lower the band above ``split`` Hz on sibilant frames by up to ``max_db``."""
    sos = butter(4, split, "highpass", fs=sr, output="sos")
    hi = sosfiltfilt(sos, y, axis=0)
    lo = y - hi                                                     # lo + hi == y exactly
    ph = _frames((hi ** 2).sum(1), sr)
    pa = _frames((y ** 2).sum(1), sr)
    m = min(len(ph), len(speech))
    ratio = 10 * np.log10(ph[:m] / (pa[:m] + 1e-20) + 1e-20)       # share of HF in the frame
    lh = 10 * np.log10(ph[:m] + 1e-20)
    sp = speech[:m]
    ref = np.percentile(lh[sp], 75) if sp.any() else -60.0
    over = np.clip((ratio + 6) / 4, 0, 1) * np.clip((lh - ref) / 6, 0, 1) * sp
    red = minimum_filter1d(-max_db * over, 3)
    k = np.hanning(7)
    red = np.convolve(red, k / k.sum(), "same")
    gh = np.interp(np.arange(len(y)) / int(HOP * sr) - 0.5, np.arange(m), 10 ** (red / 20))
    return lo + hi * gh[:, None], int((red < -0.5).sum())
