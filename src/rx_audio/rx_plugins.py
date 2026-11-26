"""Spectral De-noise and Mouth De-click from RX 12, run as VST3 plugins without the RX app.

The plugins ship with RX 12 (``/Library/Audio/Plug-Ins/VST3/RX 12 *.vst3`` on macOS,
``C:\\Program Files\\Common Files\\VST3`` on Windows; ``IZOTOPE_RX_VST3_DIR`` overrides) and are hosted
with ``pedalboard``. Settings come from the user's Module Chain (``data/rx_chain_plugins.json``):

* **Spectral De-noise** — the module's parameters (Reduction 16, algorithm, ...), plus:
  - *Learn*: the noise profile is computed here and written into the plugin's "Extra Bytes"
    blob (5 blocks ``[2, sr, N]``, N = 1025, 2049, 1025, 513, 257: noise spectrum in dB,
    ``20*log10(|X| / sum(w))`` with a Hann window, then an array of 100s). It matches RX's
    own Learn within 0.0-0.16 dB.
  - *Reduction curve*: the DSP uses a dense table of 189 (Hz, dB) pairs that follows the
    UI nodes in the blob; a value v makes the reduction ``R - v``. It is tuned per file so
    that the residual noise is flat and below -120 dB (measured here on the noise sample
    after the plugin, the counterpart of RX's yellow Residual curve).
  The plugin's default blob with the profile, the tuned table and 21 other words copied
  from the chain renders bit-identically to the chain's own blob.
* **Mouth De-click** — Sensitivity, Frequency skew, Click widening from the chain.
* **De-plosive** — not in the chain; plugin defaults (Sensitivity 5, Strength 5, Frequency
  limit 200 Hz). It is rendered on every file, and the result is kept only where it actually
  removed plosives: places where the lows drop by more than 10 dB, at least 20 per hour.

Rendering skips digital silence (exact zeros) and splits audio into chunks of up to 60 s
with 2 s of context on both sides, processed in parallel worker processes (a VST3 plugin
can only be reset on its process's main thread). Chunked output matches a whole-file
render to about -60 dB.
"""

from __future__ import annotations

import base64
import json
import os
import re
import struct
import zlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from rx_audio import noise_finder

_VST3_DEFAULT = (r"C:\Program Files\Common Files\VST3" if os.name == "nt"
                 else "/Library/Audio/Plug-Ins/VST3")
VST3_DIR = Path(os.environ.get("IZOTOPE_RX_VST3_DIR", _VST3_DEFAULT))
DENOISE_PLUGIN = "RX 12 Spectral De-noise.vst3"
DECLICK_PLUGIN = "RX 12 Mouth De-click.vst3"
DEPLOSIVE_PLUGIN = "RX 12 De-plosive.vst3"
PLOSIVE_CUT_DB = 10.0               # a plosive = the plugin cuts the lows (< 250 Hz) by more than this
PLOSIVES_PER_HOUR = 20.0            # De-plosive is kept only from this many plosives per hour (>= 3)

PROFILE_NS = (1025, 2049, 1025, 513, 257)
NODES = 0x1dfed4 // 4               # float index of the UI curve nodes in the blob
DENSE = NODES + 51                  # float index of the dense (Hz, dB) table the DSP uses
NPAIR = 189
RESIDUAL_LIMIT = -120.0
TAIL_S = 30.0                       # the end room tone is looked for in the last 30 s
TAIL_MIN_S = 3.0
PAUSES_S = 4.0                      # pauses taken next to the tail
NOISE_MAX_S = 30.0                  # longest noise stretch used for the profile
MAX_REDUCTION_DB = 40.0             # never suppress more than this anywhere (same cap as the RX batch)

CHUNK_S, CONTEXT_S = 60.0, 2.0


# ----------------------------------------------------------------------------- plugin state

_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
_B64_IDX = {c: i for i, c in enumerate(_B64)}


def _juce_decode(s: str) -> bytes:
    """JUCE MemoryBlock::fromBase64Encoding (6 bits per char, LSB first)."""
    size, data = s.split(".", 1)
    n = int(size)
    v = np.array([_B64_IDX[c] for c in data], dtype=np.uint8)
    bits = ((v[:, None] >> np.arange(6)) & 1).ravel()[: n * 8]
    bits = np.pad(bits, (0, n * 8 - len(bits)))
    return np.packbits(bits.reshape(-1, 8), axis=1, bitorder="little").ravel().tobytes()


def _juce_encode(b: bytes) -> str:
    bits = np.unpackbits(np.frombuffer(b, np.uint8), bitorder="little")
    bits = np.pad(bits, (0, (-len(bits)) % 6)).reshape(-1, 6)
    v = (bits << np.arange(6)).sum(1)
    return f"{len(b)}." + "".join(_B64[i] for i in v)


def _unpack(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    n = int.from_bytes(raw[4:8], "little")
    xml = raw[8:8 + n].decode()
    m = re.search(r"<IComponent>([^<]*)</IComponent>", xml)
    if not m:
        raise RuntimeError("unexpected plugin state: no IComponent")
    comp = _juce_decode(m.group(1))
    magic, ver, csize, _ = struct.unpack("<4I", comp[:16])
    j = json.loads(zlib.decompress(comp[16:12 + csize]))
    return {"raw": raw, "xml": xml, "span": m.span(1), "magic": magic, "ver": ver, "tail": comp[12 + csize:]}, j


def _pack(ctx: dict[str, Any], j: dict[str, Any]) -> bytes:
    body = json.dumps(j, indent=3).encode()
    z = zlib.compress(body)
    comp = struct.pack("<4I", ctx["magic"], ctx["ver"], len(z) + 4, len(body)) + z + ctx["tail"]
    a, b = ctx["span"]
    xml = (ctx["xml"][:a] + _juce_encode(comp) + ctx["xml"][b:]).encode()
    return ctx["raw"][:4] + struct.pack("<I", len(xml)) + xml + b"\x00"


def _elements(j: dict[str, Any]) -> dict[str, Any]:
    return j["DSP State"]["Value"]["DSP Elements"]["Value"]


@lru_cache(maxsize=1)
def chain_settings() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "data" / "rx_chain_plugins.json").read_text())


def _load(name: str):
    from pedalboard import load_plugin   # heavy import, only when the stage runs
    path = VST3_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"RX plugin not found: {path} (set IZOTOPE_RX_VST3_DIR)")
    return load_plugin(str(path))


@lru_cache(maxsize=2)
def _default_state(name: str) -> bytes:
    return bytes(_load(name).raw_state)


# ----------------------------------------------------------------------------- learn + curve

def _spectrum_db(x: np.ndarray, n_bins: int) -> np.ndarray:
    n = 2 * (n_bins - 1)
    w = np.hanning(n + 1)[:-1]
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    idx = np.arange(0, max(1, len(x) - n + 1), n // 4)
    fr = np.abs(np.fft.rfft(np.stack([x[k:k + n] for k in idx]) * w, axis=1)) ** 2
    return 10 * np.log10(fr.mean(0) / np.sum(w) ** 2 + 1e-30)


Noise = np.ndarray | list[np.ndarray]


def _profile_db(noise: Noise, n_bins: int) -> np.ndarray:
    """Noise profile: the spectrum of the sample, or for several groups (end room tone, pauses)
    the louder of their spectra at each frequency, so a hum present in either is covered."""
    if isinstance(noise, list):
        return np.max([_spectrum_db(g, n_bins) for g in noise], axis=0)
    return _spectrum_db(noise, n_bins)


def _flat(noise: Noise) -> np.ndarray:
    return np.concatenate(noise) if isinstance(noise, list) else noise


def _write_profile(blob: bytearray, noise: Noise, sr: int) -> None:
    off = 4
    for n in PROFILE_NS:
        struct.pack_into("<3I", blob, off, 2, int(sr), n)
        off += 12
        blob[off:off + 4 * n] = _profile_db(noise, n).astype("<f4").tobytes()
        off += 4 * n
        blob[off:off + 4 * n] = np.full(n, 100.0, "<f4").tobytes()
        off += 4 * n
    struct.pack_into("<I", blob, 0, off - 4)


class DenoiseState:
    """Spectral De-noise plugin state: chain settings + own profile + dense curve."""

    def __init__(self, noise: np.ndarray, sr: int, reduction_db: float | None = None) -> None:
        cfg = chain_settings()["spectral_denoise"]
        self.ctx, self.j = _unpack(_default_state(DENOISE_PLUGIN))
        self.el = _elements(self.j)["DenoiserSpectral"]["Value"]
        for k, v in cfg["settings"].items():
            self.el[k]["Value"] = v
        if reduction_db is not None:
            self.el["Denoiser Canopy Denoiser Suppression Linked"]["Value"] = float(reduction_db)
        blob = bytearray(base64.b64decode(self.el["Extra Bytes"]["Value"]))
        f = np.frombuffer(blob, "<f4")
        for start, vals in cfg["blob_patches"]:
            f[start:start + len(vals)] = vals           # nodes, dense table, 21 other words
        _write_profile(blob, noise, sr)
        self.blob = blob
        self.freqs = np.frombuffer(blob, "<f4", count=2 * NPAIR, offset=4 * DENSE)[0::2].astype(float)

    def raw(self, dense_db: np.ndarray) -> bytes:
        f = np.frombuffer(self.blob, "<f4")
        f[DENSE + 1:DENSE + 2 * NPAIR:2] = np.asarray(dense_db, "<f4")
        self.el["Extra Bytes"]["Value"] = base64.b64encode(bytes(self.blob)).decode()
        return _pack(self.ctx, self.j)


def _residual(plugin, raw: bytes, noise: np.ndarray, sr: int,
              freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectrum (dB) of the noise sample after the plugin, and its maximum near each table
    frequency (+-1/4 octave; the spectrum is averaged over hundreds of frames, so narrow
    tonal peaks are real and must be pulled down too)."""
    plugin.raw_state = raw
    plugin.reset()
    y = plugin.process(np.concatenate([noise, noise])[None].astype(np.float32), sr, buffer_size=1024)[0]
    s = _spectrum_db(y[len(noise):].astype(np.float64), 1025)
    fb = np.fft.rfftfreq(2048, 1 / sr)
    return s, _window_max(s, fb, freqs), fb


def _window_max(s: np.ndarray, fb: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Maximum of ``s`` within +-1/4 octave (at least +-30 Hz) of each table frequency."""
    r = np.empty(len(freqs))
    for i, v in enumerate(freqs):
        v = max(v, 40.0)
        lo, hi = min(v / 2 ** 0.25, v - 30), max(v * 2 ** 0.25, v + 30)
        r[i] = s[(fb >= lo) & (fb <= hi)].max()
    return r


def tune_denoise(noise: Noise, sr: int, reduction_db: float | None = None, rounds: int = 30,
                 target: float = -122.0, eq_bands: list[str] | None = None) -> tuple[bytes, dict[str, Any]]:
    """Learn ``noise`` and tune the dense curve until the residual is flat and below -120 dB.

    The residual is computed the way RX draws its yellow curve: noise profile minus the
    reduction at each frequency (checked against RX's own curves: within ~1 dB). Verdict as in
    ``denoise_curve.verdict`` (max < -120 dB, p95-p5 spread <= 6 dB, no bump > 6 dB over
    60 Hz-16 kHz). Each round every table point is moved toward a goal that is at most -122 dB
    and at most ``FLATNESS_DB - 1`` above the quietest bands (p5) of the noise itself. The curve
    only adds reduction to the chain's Reduction (up to MAX_REDUCTION_DB): bands that are already
    below the goal keep the base reduction instead of being raised to it, and the verdict does
    not count them as unevenness. The output of the plugin on the noise is measured once at the end, for the report.
    """
    from rx_audio.residual import FLATNESS_DB, verdict
    st = DenoiseState(noise, sr, reduction_db)
    base = float(st.el["Denoiser Canopy Denoiser Suppression Linked"]["Value"])
    lowest = -(MAX_REDUCTION_DB - base)              # table value v gives reduction base - v
    fb = np.fft.rfftfreq(2048, 1 / sr)
    prof = _profile_db(noise, 1025)
    if eq_bands:                                    # tune on the noise as it was before the EQ
        from rx_audio.eq import response_db
        prof = prof - np.clip(response_db(eq_bands, sr, np.maximum(fb, 1.0)), -40.0, 40.0)
    band = (fb >= 60) & (fb <= 16000)
    floor = float(np.percentile(prof[band], 5))      # quietest bands of the noise itself
    goal = min(target, floor + FLATNESS_DB - 1.0)
    dense = np.zeros(NPAIR)
    quiet = goal - FLATNESS_DB + 1.0                 # bands below this are quieter than needed, not uneven
    for rnd in range(rounds):
        yellow = prof - (base - np.interp(fb, st.freqs, dense))
        v = verdict(fb, np.maximum(yellow, quiet))
        if v["ok"] and rnd > 1:
            break
        step = np.clip(goal - _window_max(yellow, fb, st.freqs), -15, 15)
        if np.abs(step).max() < 0.05:
            break
        dense = np.clip(dense + step, lowest, 0.0)  # never less than the chain's Reduction
    raw = st.raw(dense)
    s, _, _ = _residual(_load(DENOISE_PLUGIN), raw, _flat(noise), sr, st.freqs)
    rep = {"residual_max_db": v["max_db"], "residual_spread_db": v["spread_db"],
           "residual_bump_db": v["max_bump_db"], "flat": v["flat"], "passed": v["ok"], "rounds": rnd + 1,
           "reduction_db_at_50_200_1k_5k_15k": np.round(base - np.interp([50, 200, 1000, 5000, 15000],
                                                                        st.freqs, dense), 1).tolist(),
           "residual_min_db": round(float(yellow[band].min()), 1),
           "measured_output_max_db": round(float(s[band].max()), 1)}
    return raw, rep


def _tail_stretch(mono: np.ndarray, sr: int, floor_db: float, end_s: float) -> dict[str, Any] | None:
    """The room tone recorded at the end (usually between/after the closing claps): the longest
    run of quiet frames (level < floor + 6 dB) in the last TAIL_S of the content, taken whole.
    Only the level decides here, not the spectrum, so a hum or tone that is part of the room
    noise stays in the profile."""
    a = max(0, int((end_s - TAIL_S) * sr))
    lv, _ = noise_finder._stft_db(mono[a:int(end_s * sr) + 1].astype(np.float32))
    quiet = (lv > -130.0) & (lv < floor_db + 6.0)
    runs = noise_finder._runs(quiet)
    if not runs:
        return None
    r0, r1 = max(runs, key=lambda r: r[1] - r[0])
    g = int(round(0.1 * sr / noise_finder.HOP))        # keep clear of the clap edges
    r0, r1 = r0 + g, r1 - g
    t0 = a / sr + r0 * noise_finder.HOP / sr
    t1 = a / sr + (r1 * noise_finder.HOP + noise_finder.FRAME) / sr
    if t1 - t0 < TAIL_MIN_S:
        return None
    return {"start": round(t0, 3), "end": round(t1, 3), "where": "tail", "duration": t1 - t0}


def _splice(mono: np.ndarray, sr: int, cands: list[dict[str, Any]], target_s: float
            ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Candidates joined with 10 ms equal-power crossfades until ``target_s`` is reached."""
    xf = int(0.01 * sr)
    out, used, total = None, [], 0.0
    for c in cands:
        piece = mono[int(c["start"] * sr):int(c["end"] * sr)]
        if out is None:
            out = piece.copy()
        elif len(out) > xf and len(piece) > xf:
            t = np.linspace(0, np.pi / 2, xf)
            out = np.concatenate([out[:-xf], out[-xf:] * np.cos(t) + piece[:xf] * np.sin(t), piece[xf:]])
        else:
            out = np.concatenate([out, piece])
        used.append({k: c[k] for k in ("start", "end", "where")})
        total += c["duration"]
        if total >= target_s:
            break
    return out.astype(np.float64), used


def _noise_sample(x: np.ndarray, sr: int, skip: int, target_s: float = 2.0) -> tuple[Noise, list[Any]]:
    """Noise for Learn, two groups: the room-tone tail at the end taken whole (or, without one,
    the best noise stretch, whole, not cut to its quietest 4 s), and ~4 s of other pauses of the
    recording. The tail/start and the pauses can carry different hum or rumble; the profile
    takes the louder of the two at each frequency. ``skip`` drops the best N candidates (a retry)."""
    found = noise_finder.find_noise_regions("<memory>", max_candidates=16 + skip, audio=(x, sr),
                                            max_duration=NOISE_MAX_S)
    cands = found.get("candidates", [])
    mono = x.mean(1) if x.ndim == 2 else x
    tail = None
    if found.get("content"):
        tail = _tail_stretch(mono, sr, found["floor_db"], found["content"][1])
    if tail is None:                               # no tail: the best stretch (often the start)
        if len(cands) <= skip:
            raise RuntimeError(found.get("error") or "no noise-only stretch found for Learn")
        tail, cands = cands[skip], cands[:skip] + cands[skip + 1:]
    main, used = _splice(mono, sr, [tail], 0.0)
    rest = [c for c in cands if c["end"] <= tail["start"] - 1.0 or c["start"] >= tail["end"] + 1.0][skip:]
    if not rest:
        return main, used
    p, pu = _splice(mono, sr, rest, PAUSES_S)
    return [main, p], used + pu


# ----------------------------------------------------------------------------- rendering

def _chunks(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Non-silent stretches (exact zeros skipped) cut into <= CHUNK_S pieces."""
    nz = np.abs(x).max(1) > 0 if x.ndim == 2 else x != 0
    if not nz.any():
        return []
    gap = int(1.0 * sr)
    idx = np.flatnonzero(nz)
    cuts = np.flatnonzero(np.diff(idx) > gap)
    spans = zip(np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]] + 1)
    step = int(CHUNK_S * sr)
    out = []
    for s, e in spans:
        out += [(a, min(e, a + step)) for a in range(int(s), int(e), step)]
    return out


def _render_jobs(name: str, raw: bytes, sr: int,
                 jobs: list[tuple[int, int, np.ndarray]]) -> list[tuple[int, np.ndarray]]:
    """Worker: process each (keep_from, keep_len, audio with context) with a fresh reset.

    Runs in its own process, so the plugin is on that process's main thread and pedalboard
    can reset it and compensate its latency exactly, as for a whole-file render."""
    p = _load(name)
    p.raw_state = raw
    out = []
    for k, n, seg in jobs:
        y = p.process(np.ascontiguousarray(seg.T), sr, buffer_size=4096, reset=True)
        out.append((k, y.T[k:k + n]))
    return out


def _render(name: str, raw: bytes, x: np.ndarray, sr: int, workers: int | None = None) -> np.ndarray:
    xc = (x[:, None] if x.ndim == 1 else x).astype(np.float32)
    y = xc.copy()                                       # digital silence stays as it is
    chunks = _chunks(xc, sr)
    if not chunks:
        return x.copy()
    ctx = int(CONTEXT_S * sr)
    workers = workers or max(1, min(len(chunks), (os.cpu_count() or 4) - 1))
    groups: list[list[tuple[int, int, np.ndarray]]] = [[] for _ in range(workers)]
    where: list[list[int]] = [[] for _ in range(workers)]
    for i, (a, b) in enumerate(chunks):
        a0, b0 = max(0, a - ctx), min(len(xc), b + ctx)
        groups[i % workers].append((a - a0, b - a, xc[a0:b0]))
        where[i % workers].append(a)
    if workers == 1:
        results = [_render_jobs(name, raw, sr, groups[0])]
    else:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(_render_jobs, [name] * workers, [raw] * workers, [sr] * workers, groups))
    for starts, res in zip(where, results):
        for a, (_, seg) in zip(starts, res):
            y[a:a + len(seg)] = seg
    y = y.astype(np.float64)
    return y[:, 0] if x.ndim == 1 else y


def spectral_denoise(x: np.ndarray, sr: int, reduction_db: float | None = None,
                     attempts: int = 3, eq_bands: list[str] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Learn a noise profile from ``x`` itself, tune the curve, render. Returns (audio, info).

    ``eq_bands``: the EQ already applied to ``x``. The plugin gets the profile of ``x`` as it is,
    but the curve is tuned on the noise as it was before the EQ (the EQ's cut added back), so
    the reduction matches RX, where Spectral De-noise runs before the EQ: there the yellow curve
    sees the full low end, gets a deeper reduction, and the EQ cuts the lows once more."""
    tries = []
    for skip in range(attempts):
        try:
            noise, used = _noise_sample(x, sr, skip)
        except RuntimeError as e:
            if not tries:
                raise
            tries.append({"error": str(e)})
            break
        raw, rep = tune_denoise(noise, sr, reduction_db, eq_bands=eq_bands)
        tries.append({"raw": raw, "noise": used, "noise_seconds": round(len(_flat(noise)) / sr, 2), **rep})
        if rep["passed"]:
            break
    best = min((t for t in tries if "raw" in t), key=lambda t: t["residual_max_db"])
    y = _render(DENOISE_PLUGIN, best.pop("raw"), x, sr)
    info = {**best, "attempts": len(tries),
            "reduction_db": reduction_db if reduction_db is not None else
            chain_settings()["spectral_denoise"]["settings"]["Denoiser Canopy Denoiser Suppression Linked"]}
    return y, info


def mouth_declick(x: np.ndarray, sr: int, sensitivity: float | None = None,
                  frequency_skew: float | None = None, click_widening: float | None = None
                  ) -> tuple[np.ndarray, dict[str, Any]]:
    ctx, j = _unpack(_default_state(DECLICK_PLUGIN))
    el = _elements(j)["MouthDeclick"]["Value"]
    vals = dict(chain_settings()["mouth_declick"]["settings"])
    for k, v in (("Sensitivity", sensitivity), ("Frequency skew", frequency_skew), ("Click widening", click_widening)):
        if v is not None:
            vals[k] = float(v)
    for k, v in vals.items():
        el[k]["Value"] = v
    y = _render(DECLICK_PLUGIN, _pack(ctx, j), x, sr)
    return y, {k: vals[k] for k in ("Sensitivity", "Frequency skew", "Click widening")}


def deplosive(x: np.ndarray, sr: int, sensitivity: float = 5.0, strength: float = 5.0,
              frequency_limit_hz: float = 200.0, force: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """De-plosive where the recording has plosives. Returns (audio, info); ``info["applied"]``
    says whether the plugin's output was kept (``force`` keeps it always)."""
    p = _load(DEPLOSIVE_PLUGIN)
    p.sensitivity, p.strength, p.frequency_limit_hz = float(sensitivity), float(strength), float(frequency_limit_hz)
    y = _render(DEPLOSIVE_PLUGIN, bytes(p.raw_state), x, sr)
    events = _plosive_events(x, y, sr)
    hours = len(x) / sr / 3600
    need = max(3, int(np.ceil(PLOSIVES_PER_HOUR * hours)))
    applied = force or len(events) >= need
    info = {"applied": applied, "plosives": len(events), "needed": need,
            "settings": {"Sensitivity": sensitivity, "Strength": strength, "Frequency limit": frequency_limit_hz},
            "examples_s": [round(t, 2) for t, _ in sorted(events, key=lambda e: -e[1])[:5]]}
    return (y if applied else x.copy()), info


def _plosive_events(x: np.ndarray, y: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """(time s, max cut dB) of each stretch where ``y`` has the lows (< 250 Hz, 10 ms frames)
    more than 6 dB under ``x`` and somewhere more than PLOSIVE_CUT_DB under it."""
    from scipy.signal import butter, sosfilt
    lp = butter(4, 250, "lowpass", fs=sr, output="sos")
    n = max(1, int(0.01 * sr))

    def level(v: np.ndarray) -> np.ndarray:
        v = sosfilt(lp, v.mean(1) if v.ndim == 2 else v)
        m = len(v) // n
        return 10 * np.log10((v[: m * n].reshape(m, n) ** 2).mean(1) + 1e-20)

    cut = level(x) - level(y)
    d = np.diff(np.r_[0, (cut > 6).astype(np.int8), 0])
    return [(a * n / sr, float(cut[a:b].max())) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
            if cut[a:b].max() > PLOSIVE_CUT_DB]
