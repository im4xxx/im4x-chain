"""Full voice chain on audio files without the RX app.

``process_files`` runs, in this order (each stage can be switched off):

EQ (own; low cut matched to the voice, 70-120 Hz, + 18 kHz low-pass)
-> De-plosive (RX plugin, kept only when the file has plosives) -> Mouth De-click (RX plugin)
-> Spectral De-noise (RX plugin, own Learn + curve tuning) -> Gate (own) -> Leveler (own)
-> compressor (own X-Comp model) -> Loudness Control (own).

Clicks go before the noise: they would otherwise end up in the noise profile and get smeared
by the spectral processing. Loudness Control is last so the result hits the loudness target
and the true-peak ceiling exactly (a compressor after it would leave the file 1-2 LU quiet).
The gate silences clicks and noises around phrases (breaths stay) right after De-noise, where the
noise floor is lowest and before Leveler and the compressor raise what is left between phrases.

The two RX stages host the RX 12 VST3 plugins (``rx_plugins``) with the settings of the
user's Module Chain; RX itself does not run.
WAV keeps its subtype; FLAC is re-encoded to FLAC with the same bit depth (written directly by
libsndfile; tags and cover are not copied). Files are written in place unless ``output_dir`` is
given; every write goes through a temporary file that replaces the target at the end.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from im4x_chain import rx_plugins
from im4x_chain.compressor import XCOMP_DEFAULTS, compress
from im4x_chain.eq import equalize, voice_bands
from im4x_chain.gate import gate
from im4x_chain.leveler import leveler
from im4x_chain.loudness import loudness_control, measure

AUDIO_EXT = {".wav", ".flac"}
FLAC_LEVEL = 5 / 8                                # libsndfile 0..1 maps to flac -0..-8

# defaults: the user's RX Module Chain (Leveler, Loudness Control items), with denser dynamics:
# Leveler keeps 60 % of phrase differences (chain: 65 %) and the compressor works on loud
# syllables (-20 dBFS, 3:1; the chain's X-Comp at -10.7 dBFS, 4:1 only touched 3-7 % of peaks).
# Together the loudness range drops by 3-4 LU. Attack 0.1 ms (the X-Comp minimum): a slower
# attack lets word onsets through while the rest of the word is compressed, so they stick out.
LEVELER_DEFAULTS = {"target": -24.0, "responsiveness": 3.3, "preserve": 60.0, "ess_db": 4.0}
COMP_DEFAULTS = {**XCOMP_DEFAULTS, "threshold": -20.0, "ratio": 3.0, "attack": 0.1, "release": 120.0,
                 "knee": 8.0}
LOUDNESS_DEFAULTS = {"target": -24.0, "ceiling": -3.0}


def expand_paths(paths: list[str]) -> list[Path]:
    """Files and the audio files of folders, each file once (a file named twice, or also
    through its folder, would otherwise go through the chain twice)."""
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        q = Path(p).expanduser()
        found = (sorted(f for f in q.iterdir() if f.suffix.lower() in AUDIO_EXT and not f.name.startswith("."))
                 if q.is_dir() else [q])
        for f in found:
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                out.append(f)
    return out


def _check_outputs(files: list[Path], output_dir: str | None) -> None:
    """Two inputs must not end up in the same output file (same name from two folders)."""
    if not output_dir:
        return
    by_name: dict[str, Path] = {}
    for f in files:
        other = by_name.setdefault(f.name.lower(), f)
        if other is not f:
            raise ValueError(f"{other} and {f} would both be written to "
                             f"{Path(output_dir) / f.name}; process them into different folders")


def _to_pcm(y: np.ndarray, subtype: str, rng: np.random.Generator) -> np.ndarray:
    if subtype == "PCM_16":                       # TPDF dither before 16-bit quantisation,
        d = (rng.random(y.shape) - rng.random(y.shape)) / 32768
        yc = y[:, None] if y.ndim == 1 else y      # muted in 10 ms frames that round to zero anyway,
        n = 480                                    # so digital silence stays digital silence
        k = -(-len(yc) // n)
        quiet = np.pad(np.abs(yc).max(1), (0, k * n - len(yc))).reshape(k, n).max(1) < 0.5 / 32768
        y = y + d
        y[np.repeat(quiet, n)[: len(yc)]] = 0.0
    return np.clip(y, -1.0, 1.0) if subtype.startswith("PCM") else y


Progress = Callable[[dict[str, Any]], None]


def process_file(path: Path, use_leveler: bool = True, use_loudness: bool = True,
                 leveler_opts: dict[str, float] | None = None, loudness_opts: dict[str, float] | None = None,
                 output_dir: Path | None = None, eq_bands: list[str] | None = None,
                 use_comp: bool = True, comp_opts: dict[str, float] | None = None,
                 use_denoise: bool = True, denoise_reduction_db: float | None = None,
                 use_declick: bool = True, use_deplosive: bool = True,
                 progress: Progress | None = None, use_gate: bool = True,
                 gate_opts: dict[str, Any] | None = None, deplosive_opts: dict[str, float] | None = None,
                 declick_opts: dict[str, float] | None = None,
                 plugins: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """``eq_bands``: None = default EQ (low cut matched to the voice, 70-120 Hz, and an
    18 kHz low-pass), [] = no EQ. ``use_deplosive``: run De-plosive and keep it where needed.
    ``denoise_reduction_db``: None = the chain's Reduction (16 dB).

    ``progress``: optional callback for a UI (the CLI's ``--progress-file``). It receives
    ``begin`` (stage list), ``stage`` start/done events, ``done`` (loudness before/after) and
    ``error`` events. A failing callback never interrupts the audio processing.

    ``plugins``: extra RX plugins, each {"name": "De-ess", "after": "leveler", "settings": {...}};
    ``after`` is a stage of the chain or "start" (before the EQ). A plugin keeps its place even
    when that stage is switched off."""
    emit = progress or (lambda _event: None)

    def notify(event: dict[str, Any]) -> None:
        try:
            emit(event)
        except Exception:                          # noqa: BLE001 - a UI must not break a render
            pass

    t0 = time.time()
    path = Path(path)
    if path.suffix.lower() not in AUDIO_EXT:
        raise ValueError(f"unsupported file type: {path.name} (WAV or FLAC)")

    # extra RX plugins by their place in the chain; a stage id per plugin ("De-ess", "De-ess 2")
    extra: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    used: dict[str, int] = {}
    for pl in plugins or []:
        if not pl.get("enabled", True):
            continue
        used[pl["name"]] = used.get(pl["name"], 0) + 1
        sid = pl["name"] + ("" if used[pl["name"]] == 1 else f" {used[pl['name']]}")
        extra.setdefault(pl.get("after", "leveler"), []).append((sid, pl))

    stages = ["read"]
    stages += [sid for sid, _ in extra.get("start", [])]
    if eq_bands is None:
        stages.append("voice")
    for name, enabled in (("eq", eq_bands is None or bool(eq_bands)), ("deplosive", use_deplosive),
                          ("declick", use_declick), ("denoise", use_denoise), ("gate", use_gate),
                          ("leveler", use_leveler), ("compressor", use_comp), ("loudness", use_loudness)):
        if enabled:
            stages.append(name)
        stages += [sid for sid, _ in extra.get(name, [])]
    stages += ["write", "measure"]
    stage_no = {name: i + 1 for i, name in enumerate(stages)}
    notify({"event": "begin", "file": str(path), "stages": ",".join(stages), "total": len(stages),
            "pid": os.getpid()})

    def begin_stage(name: str) -> float:
        notify({"event": "stage", "stage": name, "status": "start",
                "index": stage_no[name], "total": len(stages)})
        return time.time()

    def end_stage(name: str, t: float) -> None:
        notify({"event": "stage", "stage": name, "status": "done", "seconds": round(time.time() - t, 2)})

    t = begin_stage("read")
    info = sf.info(str(path))
    x, sr = sf.read(str(path), dtype="float64", always_2d=False)
    entry: dict[str, Any] = {"file": str(path), "format": info.format, "subtype": info.subtype,
                             "sample_rate": sr, "channels": info.channels, "before": measure(x, sr)}
    y = x
    end_stage("read", t)
    timings: dict[str, float] = {"read_and_measure": round(time.time() - t0, 2)}

    def stage(name: str, fn, *a, **kw) -> None:
        nonlocal y
        t = begin_stage(name)
        y, entry[name] = fn(y, sr, *a, **kw)
        timings[name] = round(time.time() - t, 2)
        end_stage(name, t)

    def plugins_after(place: str) -> None:
        for sid, pl in extra.get(place, []):
            stage(sid, rx_plugins.rx_plugin, pl["name"], pl.get("settings"))

    plugins_after("start")
    applied_eq: list[str] = []
    if eq_bands is None or eq_bands:
        voice = None
        if eq_bands is None:                       # default: low cut matched to the voice
            t = begin_stage("voice")
            applied_eq, voice = voice_bands(x, sr)
            timings["voice_pitch"] = round(time.time() - t, 2)
            end_stage("voice", t)
        else:
            applied_eq = eq_bands
        stage("eq", equalize, applied_eq)
        if voice is not None:
            entry["eq"]["voice"] = voice
    plugins_after("eq")
    if use_deplosive:
        stage("deplosive", rx_plugins.deplosive, **(deplosive_opts or {}))
    plugins_after("deplosive")
    if use_declick:
        stage("declick", rx_plugins.mouth_declick, **(declick_opts or {}))
    plugins_after("declick")
    if use_denoise:
        stage("denoise", rx_plugins.spectral_denoise, denoise_reduction_db, eq_bands=applied_eq)
    plugins_after("denoise")
    if use_gate:
        stage("gate", gate, **(gate_opts or {}))
    plugins_after("gate")
    if use_leveler:
        stage("leveler", leveler, **{**LEVELER_DEFAULTS, **(leveler_opts or {})})
    plugins_after("leveler")
    if use_comp:
        stage("compressor", compress, **{**COMP_DEFAULTS, **(comp_opts or {})})
    plugins_after("compressor")
    if use_loudness:                               # last: exact LUFS target and true-peak ceiling
        stage("loudness", loudness_control, **{**LOUDNESS_DEFAULTS, **(loudness_opts or {})})
    plugins_after("loudness")
    entry["timings_s"] = timings
    warnings = []
    if entry.get("denoise", {}).get("applied") is False:
        warnings.append("De-noise skipped: no pause with only noise to learn from (a short or tightly cut file)")
    clipped = int((np.abs(y) > 1.0).sum())
    if clipped and info.subtype.startswith("PCM"):
        warnings.append(f"{clipped} samples above 0 dBFS were clipped; enable loudness control or lower the target")
    if warnings:
        entry["warning"] = "; ".join(warnings)
    dest = (Path(output_dir).expanduser() / path.name) if output_dir else path
    dest.parent.mkdir(parents=True, exist_ok=True)
    t = begin_stage("write")
    pcm = _to_pcm(y, info.subtype, np.random.default_rng(0))
    tmp = dest.with_name(f".{dest.stem}.im4xchain.tmp{dest.suffix}")
    try:
        if info.format == "FLAC":                  # straight to FLAC, no temp WAV; 0.625 = flac -5
            subtype = info.subtype if info.subtype in ("PCM_16", "PCM_24") else "PCM_24"
            sf.write(str(tmp), pcm, sr, subtype=subtype, format="FLAC", compression_level=FLAC_LEVEL)
            entry["flac"] = {"bits": int(subtype[4:])}
        else:
            sf.write(str(tmp), pcm, sr, subtype=info.subtype, format=info.format)
        chk = sf.info(str(tmp))
        if chk.frames != len(pcm) or chk.channels != info.channels:
            raise RuntimeError(f"written file check failed: {chk.frames} frames, {chk.channels} ch")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    end_stage("write", t)
    entry["output"] = str(dest)
    t = begin_stage("measure")
    entry["after"] = measure(sf.read(str(dest), dtype="float64", always_2d=False)[0], sr)
    end_stage("measure", t)
    entry["seconds"] = round(time.time() - t0, 2)
    notify({"event": "done", "ok": True, "output": str(dest),
            "before_lufs": entry["before"]["lufs"], "before_tp": entry["before"]["true_peak_db"],
            "after_lufs": entry["after"]["lufs"], "after_tp": entry["after"]["true_peak_db"],
            "seconds": entry["seconds"], "warning": entry.get("warning", "")})
    return entry


def _process_one(f: Path, opts: dict[str, Any], progress: Progress | None = None) -> dict[str, Any]:
    try:
        return {"ok": True, **process_file(f, progress=progress, **opts)}
    except Exception as e:  # noqa: BLE001 - report per file, keep going
        msg = f"{type(e).__name__}: {e}"
        if progress is not None:
            try:
                progress({"event": "error", "file": str(f), "message": msg})
            except Exception:              # noqa: BLE001
                pass
        return {"ok": False, "file": str(f), "error": msg}


def process_files(paths: list[str], output_dir: str | None = None, jobs: int = 1,
                  progress: Progress | None = None, **opts: Any) -> dict[str, Any]:
    """Every file of ``paths`` (files or folders) through ``process_file``; ``opts`` are its
    settings (stages, their options, ``eq_bands``, ``plugins``) and go to it as they are.

    ``jobs``: files processed at the same time, each in its own process. Two use the cores
    best (the RX plugins already spread one file over all cores; the rest of the chain is
    mostly one core): two 24-minute files take 34 s instead of 46. Results are identical
    and in the order of ``paths``. With ``progress`` files go one at a time, so the events
    of one file are not mixed with another's."""
    import inspect
    known = set(inspect.signature(process_file).parameters) - {"path", "output_dir", "progress"}
    unknown = set(opts) - known
    if unknown:                                    # a misspelt setting would fail every file
        raise TypeError(f"process_files: unknown settings {sorted(unknown)}")
    for b in opts.get("eq_bands") or []:           # fail fast on a typo, before touching files
        from im4x_chain.eq import parse_band
        parse_band(b)
    for pl in opts.get("plugins") or []:           # unknown plugin or parameter: fail before any file
        if pl.get("enabled", True):
            rx_plugins.plugin_state(pl["name"], pl.get("settings"))
    files = expand_paths(paths)
    _check_outputs(files, output_dir)
    opts["output_dir"] = Path(output_dir) if output_dir else None
    workers = min(max(1, int(jobs)), len(files))
    if workers > 1 and progress is None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(_process_one, files, [opts] * len(files)))
    else:
        results = [_process_one(f, opts, progress) for f in files]
    return {"ok": all(r["ok"] for r in results) and bool(results), "count": len(results), "results": results}
