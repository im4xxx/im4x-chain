"""Loudness measurement (ITU-R BS.1770-4) and a Loudness Control replacement for RX.

Works offline on NumPy arrays: mono ``(n,)`` or multichannel ``(n, ch)``, float in [-1, 1].

``loudness_control`` hits an integrated-loudness target under a true-peak ceiling:

* plain gain when the peaks fit under the ceiling (what RX does at -24 LUFS);
* otherwise a soft-knee downward compressor on loud syllables plus a look-ahead true-peak
  limiter. The compressor threshold is searched as the highest one (least compression) that
  leaves the limiter at most ``limiter_budget_db`` of work; if no threshold reaches the target
  that way, the ratio and the limiter budget are raised step by step;
* the extra makeup that compression needs goes to speech only, so pauses (and their noise)
  keep their level under the speech;
* a target that cannot be reached without distortion is not forced: the result stops short
  and ``reached`` is False.

Planning is done on 1 ms blocks from the input's 4x-oversampled peaks and K-weighted power
(the applied gain is smooth, so both scale with it); the final render is measured exactly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, distance_transform_edt,
                           minimum_filter1d, uniform_filter1d)
from numba import njit, prange
from scipy.signal import firwin, sosfilt

from im4x_chain.chunks import chunked


# --------------------------------------------------------------------------- measurement

def _k_coeffs(sr: float) -> tuple[tuple[list[float], list[float]], tuple[list[float], list[float]]]:
    """BS.1770 K-weighting (high shelf + RLB high-pass) for any sample rate."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = ([(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
             [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    d = 1 + k / q + k * k
    highpass = ([1.0, -2.0, 1.0], [1.0, 2 * (k * k - 1) / d, (1 - k / q + k * k) / d])
    return shelf, highpass


def k_weight(x: np.ndarray, sr: float) -> np.ndarray:
    (b1, a1), (b2, a2) = _k_coeffs(sr)
    sos = np.array([b1 + a1, b2 + a2])
    return chunked(lambda v: sosfilt(sos, v, axis=0), np.asarray(x, dtype=np.float64), pre=int(2 * sr))


def _channels(x: np.ndarray) -> np.ndarray:
    return x[:, None] if x.ndim == 1 else x


def integrated_lufs(x: np.ndarray, sr: float) -> float:
    """Integrated loudness, LUFS (400 ms blocks, 75 % overlap, -70 LUFS and -10 LU gates)."""
    return _gated(gating_blocks(_channels(k_weight(_channels(x), sr)), sr))


def gating_blocks(kw: np.ndarray, sr: float) -> np.ndarray:
    """Mean power of the 400 ms gating blocks (summed over channels) of the K-weighted ``kw``
    ``(n, ch)``. A constant gain g scales them by g**2, so one pass serves every gain step."""
    n = int(0.4 * sr)
    hop = n // 4
    if len(kw) < n:
        return np.array([(kw ** 2).mean(0).sum()])
    if n == 4 * hop:                           # sum 100 ms pieces, then 4 neighbours per block
        m = len(kw) // hop
        r = kw[: m * hop].reshape(m, hop, kw.shape[1])
        return _sum4(np.einsum("ijk,ijk->i", r, r)) / n
    c = np.cumsum(np.vstack([np.zeros((1, kw.shape[1])), kw ** 2]), 0)
    s = np.arange(0, len(kw) - n + 1, hop)
    return ((c[s + n] - c[s]) / n).sum(1)


def _sum4(e: np.ndarray) -> np.ndarray:
    """Sums of 4 consecutive values (a 400 ms block from 100 ms pieces)."""
    return e[:-3] + e[1:-2] + e[2:-1] + e[3:]


def _gated(p: np.ndarray) -> float:
    loud = -0.691 + 10 * np.log10(p + 1e-30)
    p = p[loud > -70]
    if not p.size:
        return float("-inf")
    rel = -0.691 + 10 * np.log10(p.mean()) - 10
    p = p[-0.691 + 10 * np.log10(p) > rel]
    return float(-0.691 + 10 * np.log10(p.mean()))


_OS4 = firwin(81, 0.25, window=("kaiser", 5.0)) * 4     # the filter of scipy resample_poly(x, 4, 1)


@njit(cache=True, parallel=True)
def _os4_block_max(x: np.ndarray, h: np.ndarray, block: int) -> np.ndarray:
    """max |x upsampled 4x| per ``block`` input samples, over all channels.

    Same values as ``abs(resample_poly(x, 4, 1, axis=0))`` (to 1e-15) without building the
    4x array: output m = 4q + k is sum_l h[l] * u[m + 40 - l] with u[4i] = x[i].
    """
    n, ch = x.shape
    nb = (n + block - 1) // block
    out = np.zeros(nb)
    taps = h.shape[0]
    lag = (taps - 1) // 8                  # centre tap 40 = 4 * 10 input samples
    for b in prange(nb):
        best = 0.0
        for q in range(b * block, min(n, (b + 1) * block)):
            inner = q + lag < n and q + lag - (taps - 1) // 4 >= 0
            for k in range(4):             # phase k uses taps k, k+4, ... on x[q+lag], x[q+lag-1], ...
                for c in range(ch):
                    acc = 0.0
                    j = 0
                    for t in range(k, taps, 4):
                        i = q + lag - j
                        if inner or 0 <= i < n:
                            acc += h[t] * x[i, c]
                        j += 1
                    if abs(acc) > best:
                        best = abs(acc)
        out[b] = best
    return out


def true_peak_blocks(x: np.ndarray, block: int) -> np.ndarray:
    """Linear 4x-oversampled peak per ``block`` input samples."""
    return _os4_block_max(np.ascontiguousarray(_channels(np.asarray(x, dtype=np.float64))), _OS4, block)


def true_peak_db(x: np.ndarray) -> float:
    """True peak, dBTP (4x oversampling)."""
    if len(x) == 0:
        return float(20 * np.log10(1e-12))
    return float(20 * np.log10(true_peak_blocks(x, 1 << 16).max() + 1e-12))


def measure(x: np.ndarray, sr: float) -> dict[str, float]:
    return {"lufs": round(integrated_lufs(x, sr), 2), "true_peak_db": round(true_peak_db(x), 2),
            "sample_peak_db": round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2)}


# --------------------------------------------------------------------------- block planning

def _blocks(v: np.ndarray, n: int) -> np.ndarray:
    k = -(-len(v) // n)
    return np.pad(v, (0, k * n - len(v))).reshape(k, n)


def _release(gr: np.ndarray, rate: float) -> np.ndarray:
    """Gain reduction (dB, <= 0) with instant attack and a linear release of ``rate`` dB/block."""
    i = np.arange(len(gr)) * rate
    return np.minimum(np.minimum.accumulate(gr - i) + i, 0.0)


def _ramp(gr: np.ndarray, blocks: float) -> np.ndarray:
    """Onset ramp of ``blocks`` that never exceeds ``gr`` (look-ahead by construction)."""
    a = max(1, int(blocks))
    return np.minimum(uniform_filter1d(minimum_filter1d(gr, 2 * a + 1), a + 1), gr)


class _Plan:
    """Per-1 ms envelopes of the input used to plan gain curves without re-rendering audio."""

    def __init__(self, x: np.ndarray, sr: float, kw: np.ndarray | None = None,
                 peak: np.ndarray | None = None) -> None:
        self.x, self.sr = x, sr
        self.n = n = max(1, int(sr) // 1000)
        xc = _channels(x)
        self.peak = true_peak_blocks(xc, n) if peak is None else peak          # 4x peak per block
        power = (_blocks(xc.mean(1), n) ** 2).mean(1)
        self.level = 10 * np.log10(np.maximum(uniform_filter1d(power, 5), 0) + 1e-20)   # 5 ms RMS
        kw = _channels(k_weight(xc, sr)) if kw is None else kw
        self.kpow = np.sum([(_blocks(kw[:, c], n) ** 2).mean(1) for c in range(kw.shape[1])], axis=0)
        loud = 10 * np.log10(np.maximum(uniform_filter1d(self.kpow, 50), 0) + 1e-20)
        alive = loud > -100
        self.speech = np.zeros(len(loud), bool)
        if alive.any():
            sp = loud > np.percentile(loud[alive], 10) + 15
            self.speech = binary_dilation(binary_closing(sp, np.ones(120)), np.ones(41))
        self.pause_dist = (distance_transform_edt(~self.speech) if (~self.speech).any()
                           else np.zeros(len(loud)))

    def lufs(self, gdb: np.ndarray) -> float:
        p = self.kpow * 10 ** (gdb / 10)
        if len(p) < 400:
            return _gated(np.array([p.mean()]))
        m = len(p) // 100
        return _gated(_sum4(p[: m * 100].reshape(m, 100).sum(1)) / 400)

    def gains(self, g: float, thr: float | None, ceiling: float, pause_extra: float,
              ratio: float = 3.0, knee: float = 6.0) -> tuple[np.ndarray, float]:
        """Block gain (dB) and the limiter's largest reduction for gain ``g`` and threshold ``thr``."""
        total = np.full(len(self.peak), float(g))
        if thr is not None:
            over = self.level + g - thr
            gr = np.where(over <= -knee / 2, 0.0,
                          np.where(over >= knee / 2, -over * (1 - 1 / ratio),
                                   -(1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
            total += _release(_ramp(gr, 3), 0.06)                # 3 ms attack, 60 dB/s release
        if pause_extra > 0:                                      # pauses: no extra makeup
            total -= pause_extra * np.clip(self.pause_dist / 40, 0, 1)     # 40 ms ramp
        need = np.minimum(0.0, ceiling - (20 * np.log10(self.peak + 1e-12) + total))
        lim = _release(_ramp(minimum_filter1d(need, 3), 1.5), 0.12)     # 1.5 ms look-ahead, 120 dB/s
        return total + lim, float(-lim.min()) if len(lim) else 0.0

    def render(self, gdb: np.ndarray) -> np.ndarray:
        gb = 10 ** (np.minimum(gdb, minimum_filter1d(gdb, 3)) / 20)
        g = np.interp((np.arange(len(self.x)) + 0.5) / self.n - 0.5, np.arange(len(gb)), gb)
        return self.x * g if self.x.ndim == 1 else self.x * g[:, None]


# --------------------------------------------------------------------------- loudness control

def loudness_control(x: np.ndarray, sr: float, target: float = -24.0, ceiling: float = -3.0,
                     limiter_budget_db: float = 3.0, keep_pauses: bool = True,
                     max_makeup_db: float = 24.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Bring ``x`` to ``target`` LUFS with the true peak at or under ``ceiling`` dBTP.

    Returns (audio, info). ``info["mode"]`` is "gain", "limit" or "compress+limit";
    ``info["reached"]`` says whether the target was met within 0.05 LU.
    """
    # loudness and true peak of x * const are measured once and scaled for every gain step
    x = np.asarray(x, dtype=np.float64)
    kw = _channels(k_weight(_channels(x), sr))
    ms_peak = true_peak_blocks(x, max(1, int(sr) // 1000))         # also the plan's 1 ms peaks
    xp, x_tp = gating_blocks(kw, sr), float(20 * np.log10(ms_peak.max() + 1e-12))
    in_lufs = _gated(xp)
    info: dict[str, Any] = {"in_lufs": round(in_lufs, 2), "in_true_peak_db": round(x_tp, 2),
                            "target": target, "ceiling": ceiling}
    if not np.isfinite(in_lufs):
        info.update(mode="silence", reached=False, out_lufs=in_lufs)
        return x.copy(), info
    g0 = target - in_lufs
    for _ in range(4):                 # the -70 LUFS gate makes loudness not exactly linear in gain
        cur = _gated(xp * 10 ** (g0 / 10))
        if abs(cur - target) < 0.005:
            break
        g0 += target - cur
    if x_tp + g0 <= ceiling:
        info.update(mode="gain", gain_db=round(g0, 2))
        return _finish(info, x * 10 ** (g0 / 20), xp * 10 ** (g0 / 10), x_tp + g0, target)

    plan = _Plan(x, sr, kw, ms_peak)
    del kw
    lim_ceiling = ceiling - 0.15
    ratio = [3.0]

    def solve(thr: float | None) -> tuple[float, np.ndarray, float, float]:
        g = g0
        for _ in range(12):
            gdb, lim = plan.gains(g, thr, lim_ceiling, max(0.0, g - g0) if keep_pauses else 0.0, ratio[0])
            cur = plan.lufs(gdb)
            if abs(cur - target) < 0.01:
                break
            g = min(g + (target - cur), g0 + max_makeup_db)
            if g == g0 + max_makeup_db and cur < target:
                gdb, lim = plan.gains(g, thr, lim_ceiling, max(0.0, g - g0) if keep_pauses else 0.0, ratio[0])
                break
        return g, gdb, lim, plan.lufs(gdb)

    first = solve(None)
    best, thr_used, budget_used = first, None, limiter_budget_db
    if not (first[2] <= limiter_budget_db and first[3] >= target - 0.02):
        steps = [(3.0, limiter_budget_db), (3.0, max(limiter_budget_db, 5.0)),
                 (5.0, max(limiter_budget_db, 5.0)), (5.0, max(limiter_budget_db, 8.0))]
        chosen = fallback = None
        for r, budget in steps:
            ratio[0] = r
            hi, lo, cand = target + 25.0, target - 15.0, None
            for _ in range(14):        # highest threshold that keeps the limiter within budget
                mid = (hi + lo) / 2
                res = solve(mid)
                if res[2] <= budget:
                    cand, lo = (mid, res, r, budget), mid
                else:
                    hi = mid
                if hi - lo < 0.25:
                    break
            if cand is None:
                continue
            if fallback is None or cand[1][3] > fallback[1][3]:
                fallback = cand
            if cand[1][3] >= target - 0.02:
                chosen = cand
                break
        pick = chosen or fallback
        if pick is None:
            ratio[0] = 5.0
            pick = (target - 15.0, solve(target - 15.0), 5.0, None)
        thr_used, best, ratio_used, budget_used = pick
        if chosen is None:
            budget_used = None
        info["ratio"] = ratio_used

    g, gdb, lim, _ = best
    y = plan.render(gdb)
    del plan
    yp, y_tp, corr = gating_blocks(_channels(k_weight(_channels(y), sr)), sr), true_peak_db(y), 0.0
    for _ in range(3):                 # exact measurement; small corrections, never past the ceiling
        cur = _gated(yp * 10 ** (corr / 10))
        if abs(cur - target) < 0.02:
            break
        step = min(target - cur, ceiling - (y_tp + corr) - 0.01)
        if step <= 0.005:
            break
        corr += step
    if corr:
        y *= 10 ** (corr / 20)
    info.update(mode="limit" if thr_used is None else "compress+limit", gain_db=round(g, 2),
                compressor_threshold_dbfs=None if thr_used is None else round(thr_used, 2),
                limiter_max_reduction_db=round(lim, 2), limiter_budget_db=budget_used,
                speech_only_makeup_db=round(max(0.0, g - g0), 2) if keep_pauses else 0.0)
    return _finish(info, y, yp * 10 ** (corr / 10), y_tp + corr, target)


def _finish(info: dict[str, Any], y: np.ndarray, blocks: np.ndarray, tp: float,
            target: float) -> tuple[np.ndarray, dict[str, Any]]:
    """``blocks``/``tp``: gating blocks and true peak of ``y`` (scaled from one measurement)."""
    if tp > info["ceiling"]:           # rare inter-sample overshoot after the last correction
        trim = info["ceiling"] - tp - 0.01
        y *= 10 ** (trim / 20)
        blocks, tp = blocks * 10 ** (trim / 10), tp + trim
    info["out_lufs"] = round(_gated(blocks), 2)
    info["out_true_peak_db"] = round(tp, 2)
    info["reached"] = abs(info["out_lufs"] - target) <= 0.05
    return y, info
