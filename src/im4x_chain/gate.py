"""Gate: silences clicks and noises around phrases; keeps speech, the sounds between close words and phrase tails.

Offline and event-based instead of a level follower, so it can look at a whole sound before
deciding:

1. Level in 5 ms frames of the signal high-passed at 150 Hz (rumble does not open it).
   Reference = 95th percentile of non-silent frames (loud speech), floor = 10th percentile
   (the residual noise after De-noise). Digital silence does not count.
2. Events = runs of frames above ``max(floor + 15 dB, ref - 50 dB)``: every sound that stands
   out of the noise floor, speech or not.
3. An event is speech when it has at least ``min_loud_ms`` above ``ref + keep_db``
   (syllables reach the reference level; mouth clicks are too short, breaths and handling
   noise too quiet), or when it is voiced and not too quiet (a quiet word: periodicity >=
   ``voiced_keep`` over >= ``voiced_min_ms``, peak above ``ref + voiced_db``). Events that
   end within ``bridge_before_ms`` of speech or start within ``bridge_after_ms`` after it
   are kept with it: stop bursts before a vowel, final consonants after the closure.
4. Breaths and other quiet sounds stay only next to speech (``keep_breaths``):
   - between words closer than ``close_gap_ms`` everything is kept;
   - after a phrase the tail (breath, decay) is kept while the level smoothed over 60 ms
     stays ``breath_margin_db`` above the floor, up to ``tail_max_ms``, fading out over the
     last ``tail_fade_ms``.
   Breaths before a phrase (usually noise in dubbing) and breaths with no phrase next to
   them are removed. Inside what is kept, short sounds (up to ``click_max_ms``) that stick
   out of the median of the surrounding ~100 ms by ``click_prominence_db`` are clicks and
   are cut out with 2 ms fades.
5. Everything else is attenuated by ``range_db`` (None = muted). Kept events open
   ``lookahead_ms`` early with a ``attack_ms`` raised-cosine fade-in and close after
   ``hold_ms`` with a ``release_ms`` fade-out, so word onsets and tails are not cut.
6. A file without pauses (floor less than ``MIN_RANGE_DB`` under the reference, e.g. a
   tightly cut line) is left as it is: there is nothing to tell speech from noise.

Measured on a 24-minute dubbing file after De-noise: speech events peak at -15..-25 dBFS
(reference -19), breaths between phrases at -60..-90 dBFS, 200-500 ms; mouth clicks are
5-60 ms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import find_objects, label, median_filter
from scipy.signal import butter, sosfilt

FRAME_S = 0.005
# the gate needs pauses: the floor (10th percentile) must be this far under the speech. A
# tightly cut word or phrase has no pauses, its "floor" is speech, and every sound would be
# taken for noise (a 0.4-0.8 s cut came out as silence); such files are left as they are.
MIN_RANGE_DB = 30.0


def _levels(x: np.ndarray, sr: float) -> tuple[np.ndarray, np.ndarray, int]:
    mono = x if x.ndim == 1 else x.mean(axis=1)
    h = sosfilt(butter(4, 150.0, "hp", fs=sr, output="sos"), mono)
    hop = max(1, int(round(FRAME_S * sr)))
    n = len(h) // hop
    p = (h[:n * hop].reshape(n, hop) ** 2).mean(axis=1)
    silent = p < 1e-14                                 # digital (near) silence: below -140 dBFS
    return 10 * np.log10(p + 1e-30), silent, hop


def _periodicity(seg: np.ndarray, sr: float) -> float:
    """Best normalised autocorrelation at 70-400 Hz over 40 ms windows (0 = noise, 1 = tone)."""
    w = int(0.04 * sr)
    if len(seg) < w:
        return 0.0
    lo, hi = int(sr / 400), int(sr / 70)
    norm = 1.0 - np.arange(w) / w                      # window's own autocorrelation decay
    best = 0.0
    for s in range(0, len(seg) - w + 1, w // 2):
        f = seg[s:s + w] - seg[s:s + w].mean()
        spec = np.fft.rfft(f, 2 * w)
        a = np.fft.irfft(np.abs(spec) ** 2)[:w]
        if a[0] <= 0:
            continue
        best = max(best, float((a[lo:hi] / a[0] / norm[lo:hi]).max()))
    return best


def _fade(n: int, rising: bool) -> np.ndarray:
    r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
    return r if rising else r[::-1]


def gate(x: np.ndarray, sr: float, keep_db: float = -20.0, min_loud_ms: float = 30.0,
         voiced_keep: float = 0.6, voiced_min_ms: float = 60.0, voiced_db: float = -30.0,
         bridge_before_ms: float = 40.0, bridge_after_ms: float = 150.0,
         lookahead_ms: float = 15.0, attack_ms: float = 10.0, hold_ms: float = 30.0,
         release_ms: float = 80.0, keep_breaths: bool = True, breath_margin_db: float = 12.0,
         close_gap_ms: float = 350.0, tail_max_ms: float = 400.0, tail_fade_ms: float = 150.0,
         click_max_ms: float = 60.0,
         click_prominence_db: float = 12.0, range_db: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns (audio, info). See the module docstring for the parameters."""
    lv, silent, hop = _levels(x, sr)
    if (~silent).sum() < 10:
        return x.copy(), {"applied": False, "reason": "no audio"}
    ref = float(np.percentile(lv[~silent], 95))
    floor = float(np.percentile(lv[~silent], 10))
    if ref - floor < MIN_RANGE_DB:                  # no pauses: nothing tells speech from noise
        return x.copy(), {"applied": False, "reason": "no pauses to tell speech from noise",
                          "reference_db": round(ref, 1), "floor_db": round(floor, 1)}
    event_db = max(floor + 15.0, ref - 50.0)
    events, _ = label(lv > event_db)
    spans = [(sl[0].start, sl[0].stop) for sl in find_objects(events)]
    ms = FRAME_S * 1000.0
    mono = x if x.ndim == 1 else x.mean(axis=1)

    # pass 1: speech by itself
    speech = np.zeros(len(spans), dtype=bool)
    for i, (a, b) in enumerate(spans):
        e_lv = lv[a:b]
        speech[i] = (e_lv > ref + keep_db).sum() * ms >= min_loud_ms
        if not speech[i] and (b - a) * ms >= voiced_min_ms and e_lv.max() > ref + voiced_db:
            speech[i] = _periodicity(mono[a * hop:b * hop], sr) >= voiced_keep
    # pass 2: sounds right next to speech belong to it (a stop burst before the vowel, a final
    # consonant after the closure); mouth noise further away does not
    keep_ev = speech.copy()
    starts = np.array([a for (a, _), sp in zip(spans, speech) if sp])
    ends = np.array([b for (_, b), sp in zip(spans, speech) if sp])
    for i, (a, b) in enumerate(spans):
        if speech[i] or len(starts) == 0:
            continue
        nxt = starts[starts >= b]
        prv = ends[ends <= a]
        if (len(nxt) and (nxt[0] - b) * ms <= bridge_before_ms) or \
           (len(prv) and (a - prv[-1]) * ms <= bridge_after_ms):
            keep_ev[i] = True

    keep = np.zeros(len(x), dtype=np.float64)

    def open_span(start: int, end: int, fade_in: int, fade_out: int) -> None:
        keep[max(0, start):end] = 1.0
        f0, f1 = max(0, start - fade_in), max(0, start)
        if f1 > f0:
            keep[f0:f1] = np.maximum(keep[f0:f1], _fade(f1 - f0, True))
        r0, r1 = min(len(keep), end), min(len(keep), end + fade_out)
        if r1 > r0:
            keep[r0:r1] = np.maximum(keep[r0:r1], _fade(fade_out, False)[:r1 - r0])

    look, att, hold, rel = (int(v / 1000.0 * sr) for v in (lookahead_ms, attack_ms, hold_ms, release_ms))
    speech_frames = np.zeros(len(lv), dtype=bool)
    for i, (a, b) in enumerate(spans):
        if keep_ev[i]:
            open_span(a * hop - look, b * hop + hold, att, rel)
            speech_frames[a:b] = True

    # what is kept around speech besides the speech itself: everything between words that are
    # close together, and the tail after a phrase (breath, decay) with a slow fade-out.
    # Breaths before a phrase and breaths with no phrase next to them are not kept.
    regions: list[tuple[int, int]] = []
    cut: list[tuple[int, int]] = []
    gaps = tails = 0
    if keep_breaths:
        k = max(1, int(round(0.06 / FRAME_S)))
        p = np.where(silent, 0.0, 10 ** (lv / 10))
        sm = 10 * np.log10(np.convolve(p, np.ones(k) / k, "same") + 1e-30)
        above = sm > floor + breath_margin_db
        kept_spans = [sp for sp, k_ in zip(spans, keep_ev) if k_]
        tf = int(tail_fade_ms / 1000.0 * sr)
        for j, (a, b) in enumerate(kept_spans):
            nxt = kept_spans[j + 1][0] if j + 1 < len(kept_spans) else None
            if nxt is not None and (nxt - b) * ms <= close_gap_ms:
                gaps += 1
                open_span(b * hop, nxt * hop, 0, 0)
                regions.append((b, nxt))
                continue
            lim = min(len(lv), b + int(tail_max_ms / ms), nxt if nxt is not None else len(lv))
            e = b
            while e < lim and above[e]:
                e += 1
            if (e - b) * ms <= hold_ms:                     # nothing beyond the normal release
                continue
            tails += 1
            start, end = b * hop, e * hop
            fade = min(tf, end - start)
            keep[start:end - fade] = 1.0
            keep[end - fade:end] = np.maximum(keep[end - fade:end], _fade(fade, False))
            regions.append((b, e))
        # clicks sticking out of what is kept between and after words are cut again:
        # a short run of frames well above the median of the surrounding ~100 ms
        for a, b in regions:
            if b - a < 3:
                continue
            base = median_filter(lv[a:b], size=21, mode="nearest")
            spikes, _ = label((lv[a:b] - base > click_prominence_db) & ~speech_frames[a:b])
            for c in find_objects(spikes):
                ca, cb = a + c[0].start - 1, a + c[0].stop + 1        # one frame of margin
                if (cb - ca - 2) * ms <= click_max_ms:
                    cut.append((max(0, ca) * hop, min(len(lv), cb) * hop))
    cf = max(1, int(0.002 * sr))                           # 2 ms fades around a cut click
    for start, end in cut:
        g = np.zeros(end - start + 2 * cf)
        g[:cf], g[-cf:] = _fade(cf, False), _fade(cf, True)
        a0 = start - cf
        lo, hi = max(0, a0), min(len(keep), a0 + len(g))
        keep[lo:hi] = np.minimum(keep[lo:hi], g[lo - a0:hi - a0])

    removed = [(round(a * hop / sr, 2), round((b - a) * ms), round(float(lv[a:b].max()), 1))
               for i, (a, b) in enumerate(spans) if keep[a * hop:b * hop].max() < 0.5]

    g_floor = 0.0 if range_db is None else 10 ** (-abs(range_db) / 20)
    g = g_floor + (1.0 - g_floor) * keep
    y = x * (g if x.ndim == 1 else g[:, None])
    return y, {"applied": True, "reference_db": round(ref, 1), "floor_db": round(floor, 1),
               "event_threshold_db": round(event_db, 1), "keep_threshold_db": round(ref + keep_db, 1),
               "kept": int(keep_ev.sum()), "bridged": int((keep_ev & ~speech).sum()),
               "gaps_kept": gaps, "tails": tails, "clicks_cut": len(cut), "removed": len(removed), "removed_s": round(sum(r[1] for r in removed) / 1000, 2),
               "loudest_removed": [{"t_s": t, "ms": d, "peak_db": p}
                                   for t, d, p in sorted(removed, key=lambda r: -r[2])[:8]],
               "settings": {"range_db": range_db, "lookahead_ms": lookahead_ms, "attack_ms": attack_ms,
                            "hold_ms": hold_ms, "release_ms": release_ms}}
