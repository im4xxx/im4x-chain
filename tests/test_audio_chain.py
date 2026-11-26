"""Leveler / Loudness Control without RX: synthetic signals only (no RX, no real recordings)."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from rx_audio.audio_chain import process_files
from rx_audio.leveler import leveler
from rx_audio.loudness import integrated_lufs, loudness_control, true_peak_db

SR = 48000


def speechlike(levels_db: list[float], phrase_s: float = 2.0, pause_s: float = 0.6,
               noise_db: float = -80.0, seed: int = 0) -> tuple[np.ndarray, list[slice]]:
    """Phrases of syllable-modulated band noise at the given RMS levels, pauses of noise."""
    rng = np.random.default_rng(seed)
    out, spans, pos = [], [], 0
    for lv in levels_db:
        pause = rng.standard_normal(int(pause_s * SR)) * 10 ** (noise_db / 20)
        n = int(phrase_s * SR)
        t = np.arange(n) / SR
        env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2           # ~4 syllables / s
        carrier = np.convolve(rng.standard_normal(n), np.hanning(9), "same")
        ph = carrier * env
        ph *= 10 ** (lv / 20) / np.sqrt(np.mean(ph ** 2))
        out += [pause, ph]
        pos += len(pause)
        spans.append(slice(pos, pos + n))
        pos += n
    out.append(rng.standard_normal(int(pause_s * SR)) * 10 ** (noise_db / 20))
    return np.concatenate(out), spans


def _have_rx_plugins() -> bool:
    from rx_audio import rx_plugins
    return all((rx_plugins.VST3_DIR / n).exists() for n in (rx_plugins.DENOISE_PLUGIN, rx_plugins.DECLICK_PLUGIN))


needs_rx_plugins = pytest.mark.skipif(not _have_rx_plugins(), reason="RX 12 VST3 plugins not installed")



def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x ** 2) + 1e-30))


def test_meter_matches_bs1770_reference_sine() -> None:
    t = np.arange(10 * SR) / SR
    x = np.sin(2 * np.pi * 997 * t)
    assert integrated_lufs(x, SR) == pytest.approx(-3.01, abs=0.01)          # mono, 0 dBFS
    assert integrated_lufs(np.stack([x, x], 1), SR) == pytest.approx(0.0, abs=0.01)


def test_loudness_plain_gain_is_exact_and_linear() -> None:
    x, _ = speechlike([-30, -28, -32, -29])
    y, info = loudness_control(x, SR, target=-24.0, ceiling=-3.0)
    assert info["mode"] == "gain" and info["reached"]
    assert integrated_lufs(y, SR) == pytest.approx(-24.0, abs=0.02)
    ratio = y[x != 0] / x[x != 0]
    assert np.ptp(ratio) < 1e-9                                             # one constant gain


def test_loudness_high_target_respects_ceiling_and_keeps_pauses_down() -> None:
    x, spans = speechlike([-22, -20, -24, -21])
    x[spans[1].start + 1000] = 0.9                                            # a sharp click
    y, info = loudness_control(x, SR, target=-14.0, ceiling=-3.0)
    assert info["mode"] in ("limit", "compress+limit")
    assert true_peak_db(y) <= -3.0 + 0.02
    assert integrated_lufs(y, SR) <= -14.0 + 0.05
    if info["reached"]:
        assert integrated_lufs(y, SR) == pytest.approx(-14.0, abs=0.05)
    # pauses must not rise more than plain gain would raise them
    pause = slice(0, spans[0].start - 2000)
    plain = info["gain_db"] - info.get("speech_only_makeup_db", 0.0)
    assert rms_db(y[pause]) - rms_db(x[pause]) <= plain + 1.0


def test_loudness_unreachable_target_is_reported_not_forced() -> None:
    x, _ = speechlike([-20, -20])
    x[SR] = 0.99
    y, info = loudness_control(x, SR, target=-3.0, ceiling=-3.0)
    assert not info["reached"]
    assert true_peak_db(y) <= -3.0 + 0.02


def test_leveler_keeps_the_preserved_share_of_phrase_differences() -> None:
    x, spans = speechlike([-36, -24, -30, -24, -36], phrase_s=2.5, pause_s=0.8)
    y, info = leveler(x, SR, target=-24.0, preserve=65.0, ess_db=0.0)
    before = [rms_db(x[s]) for s in spans]
    after = [rms_db(y[s]) for s in spans]
    spread_in, spread_out = np.ptp(before), np.ptp(after)
    assert info["phrases"] == 5
    assert spread_out == pytest.approx(spread_in * 0.65, abs=1.5)
    # inside a phrase the gain is (nearly) constant: syllable modulation survives
    s = spans[1]
    g = np.abs(y[s]) / np.maximum(np.abs(x[s]), 1e-9)
    mid = g[len(g) // 4: 3 * len(g) // 4]
    assert 20 * np.log10(np.percentile(mid, 95) / np.percentile(mid, 5)) < 1.0


def test_leveler_without_speech_returns_input() -> None:
    x = np.random.default_rng(1).standard_normal(SR) * 1e-6
    y, info = leveler(x, SR)
    assert info["changed"] is False and np.array_equal(x, y)


@pytest.mark.skipif(not (shutil.which("flac") and shutil.which("metaflac")), reason="flac tools missing")
@needs_rx_plugins
def test_flac_round_trip_keeps_bit_depth_and_tags(tmp_path) -> None:
    x, _ = speechlike([-32, -26, -30])
    src = tmp_path / "in.flac"
    sf.write(str(src), x, SR, subtype="PCM_24")
    subprocess.run(["metaflac", "--set-tag=ARTIST=Test", str(src)], check=True)
    out = tmp_path / "out"
    res = process_files([str(src)], output_dir=str(out), use_comp=False)
    assert res["ok"], res
    assert "compressor" not in res["results"][0] and "eq" in res["results"][0]
    dst = out / "in.flac"
    assert sf.info(str(dst)).subtype == "PCM_24"
    tags = subprocess.run(["metaflac", "--export-tags-to=-", str(dst)], capture_output=True, text=True).stdout
    assert "ARTIST=Test" in tags
    y, _ = sf.read(str(dst))
    assert integrated_lufs(y, SR) == pytest.approx(-24.0, abs=0.05)
    assert sf.read(str(src))[0] == pytest.approx(x, abs=1e-6)                # source untouched


def test_eq_default_is_60hz_highpass() -> None:
    from rx_audio.eq import response_db
    r = response_db(["hp:60"], SR, np.array([15.0, 30.0, 60.0, 120.0, 1000.0]))
    assert r[2] == pytest.approx(-3.0, abs=0.1)                               # -3 dB at 60 Hz
    assert r[1] == pytest.approx(-24.0, abs=1.5)                              # 24 dB/oct
    assert r[0] < -45 and abs(r[4]) < 0.01 and abs(r[3]) < 0.5


def test_eq_peak_and_shelves_hit_their_gain() -> None:
    from rx_audio.eq import response_db
    assert response_db(["peak:3000:+4:1"], SR, np.array([3000.0]))[0] == pytest.approx(4.0, abs=0.01)
    assert response_db(["lowshelf:200:-3"], SR, np.array([20.0]))[0] == pytest.approx(-3.0, abs=0.1)
    assert response_db(["highshelf:8000:+2"], SR, np.array([20000.0]))[0] == pytest.approx(2.0, abs=0.15)
    with pytest.raises(ValueError):
        response_db(["peak:3000"], SR, np.array([1.0]))


@needs_rx_plugins
def test_cli_processes_and_measures(tmp_path, capsys) -> None:
    from rx_audio.cli import main
    x, _ = speechlike([-32, -26, -30])
    x = x + 0.05 * np.sin(2 * np.pi * 30 * np.arange(len(x)) / SR)          # 30 Hz rumble
    src = tmp_path / "a.wav"
    sf.write(str(src), x, SR, subtype="PCM_24")
    assert main(["process", str(src), "-o", str(tmp_path / "o")]) == 0
    y, _ = sf.read(str(tmp_path / "o" / "a.wav"))
    rumble = lambda s: abs(np.dot(s, np.exp(-2j * np.pi * 30 * np.arange(len(s)) / SR)))  # noqa: E731
    assert rumble(y) < rumble(x) / 10                                          # cut by > 20 dB
    assert main(["measure", str(tmp_path / "o")]) == 0
    assert "LUFS" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["process", str(src), "--eq", "bogus:1"])


# Values measured on the real SSL Native X-Comp v6 (AU) with the chain's settings.
XCOMP_STATIC = {-16: -0.39, -12: -1.39, -8: -2.99, -4: -5.21, 0: -8.02}       # square wave, peak dBFS
XCOMP_DC_STEP = {"att": {0.25: -3.474, 0.5: -5.518, 1.0: -7.191, 5.0: -8.021},  # -30 dBFS -> 0 dBFS
                 "rel": {50: -6.476, 100: -5.098, 200: -2.875, 400: -0.419}}   # 0 dBFS -> -30 dBFS


def test_compressor_static_curve_matches_xcomp() -> None:
    from rx_audio.compressor import XCOMP_DEFAULTS, static_curve
    d = XCOMP_DEFAULTS
    for lv, g in XCOMP_STATIC.items():
        assert static_curve(np.array([lv]), d["threshold"], d["ratio"], d["knee"])[0] == pytest.approx(g, abs=0.05)
    assert static_curve(np.array([-25.0]), d["threshold"], d["ratio"], d["knee"])[0] == 0.0


def test_compressor_timing_matches_xcomp() -> None:
    from rx_audio.compressor import compress
    x = np.full(3 * SR, 10 ** (-30 / 20))
    x[SR:2 * SR] = 1.0
    y, info = compress(x, SR)
    g = 20 * np.log10(y / x)
    for ms, want in XCOMP_DC_STEP["att"].items():
        assert g[SR + int(ms * SR / 1000)] == pytest.approx(want, abs=0.1)
    for ms, want in XCOMP_DC_STEP["rel"].items():
        assert g[2 * SR + int(ms * SR / 1000)] == pytest.approx(want, abs=0.1)
    assert info["max_reduction_db"] == pytest.approx(8.02, abs=0.02)


def test_compressor_links_stereo_channels() -> None:
    from rx_audio.compressor import compress
    x, _ = speechlike([-12, -10])
    st = np.stack([x, 0.05 * x], 1)
    y, _ = compress(st, SR)
    a = np.abs(x) > 0.05
    assert np.allclose(y[a, 0] / st[a, 0], y[a, 1] / st[a, 1])


@needs_rx_plugins
def test_16bit_output_keeps_digital_silence(tmp_path) -> None:
    x, spans = speechlike([-30, -26], noise_db=-200)
    x[: spans[0].start] = 0.0                                                 # digital silence at the start
    src = tmp_path / "s.wav"
    sf.write(str(src), x, SR, subtype="PCM_16")
    res = process_files([str(src)], output_dir=str(tmp_path / "o"))
    assert res["ok"], res
    y, _ = sf.read(str(tmp_path / "o" / "s.wav"), dtype="int16")
    assert not y[: spans[0].start - 4800].any()                               # no dither noise in silence
    assert np.abs(y[spans[1]]).max() > 1000


# ---------------------------------------------------------------- RX plugins without the RX app

def test_rx_plugin_state_codec_roundtrip() -> None:
    from rx_audio.rx_plugins import _juce_decode, _juce_encode
    b = bytes(np.random.default_rng(3).integers(0, 256, 1001, dtype=np.uint8))
    assert _juce_decode(_juce_encode(b)) == b


@needs_rx_plugins
def test_spectral_denoise_learns_and_lowers_noise() -> None:
    from rx_audio.rx_plugins import spectral_denoise
    x, spans = speechlike([-26, -24, -28], phrase_s=3.0, pause_s=1.5, noise_db=-60)
    y, info = spectral_denoise(x, SR)
    pause = slice(spans[1].stop + 4800, spans[2].start - 4800)
    assert rms_db(y[pause]) < rms_db(x[pause]) - 10                           # noise down by > 10 dB
    assert abs(rms_db(y[spans[1]]) - rms_db(x[spans[1]])) < 1.0               # speech level kept
    assert info["reduction_db"] == 16.0 and "residual_max_db" in info


@needs_rx_plugins
def test_plugin_render_in_chunks_matches_whole_file() -> None:
    from rx_audio import rx_plugins as R
    x, _ = speechlike([-20] * 8, phrase_s=10.0, pause_s=0.5, noise_db=-50)   # ~84 s, no digital silence
    ctx, j = R._unpack(R._default_state(R.DECLICK_PLUGIN))
    raw = R._pack(ctx, j)
    y = R._render(R.DECLICK_PLUGIN, raw, x, SR, workers=2)
    p = R._load(R.DECLICK_PLUGIN)
    p.raw_state = raw
    ref = p.process(x[None].astype(np.float32), SR, buffer_size=4096)[0]
    assert 10 * np.log10(np.sum((y - ref) ** 2) / np.sum(ref ** 2) + 1e-30) < -50
