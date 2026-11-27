"""Leveler / Loudness Control without RX: synthetic signals only (no RX, no real recordings)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from im4x_chain.audio_chain import process_files
from im4x_chain.leveler import leveler
from im4x_chain.loudness import integrated_lufs, loudness_control, true_peak_db

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
    from im4x_chain import rx_plugins
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


@needs_rx_plugins
def test_flac_round_trip_keeps_bit_depth(tmp_path) -> None:
    x, _ = speechlike([-32, -26, -30])
    src = tmp_path / "in.flac"
    sf.write(str(src), x, SR, subtype="PCM_24")
    out = tmp_path / "out"
    res = process_files([str(src)], output_dir=str(out), use_comp=False)
    assert res["ok"], res
    assert "compressor" not in res["results"][0] and "eq" in res["results"][0]
    dst = out / "in.flac"
    assert sf.info(str(dst)).subtype == "PCM_24"
    y, _ = sf.read(str(dst))
    assert integrated_lufs(y, SR) == pytest.approx(-24.0, abs=0.05)
    assert sf.read(str(src))[0] == pytest.approx(x, abs=1e-6)                # source untouched


def test_eq_default_is_60hz_highpass() -> None:
    from im4x_chain.eq import response_db
    r = response_db(["hp:60"], SR, np.array([15.0, 30.0, 60.0, 120.0, 1000.0]))
    assert r[2] == pytest.approx(-3.0, abs=0.1)                               # -3 dB at 60 Hz
    assert r[1] == pytest.approx(-24.0, abs=1.5)                              # 24 dB/oct
    assert r[0] < -45 and abs(r[4]) < 0.01 and abs(r[3]) < 0.5


def test_eq_peak_and_shelves_hit_their_gain() -> None:
    from im4x_chain.eq import response_db
    assert response_db(["peak:3000:+4:1"], SR, np.array([3000.0]))[0] == pytest.approx(4.0, abs=0.01)
    assert response_db(["lowshelf:200:-3"], SR, np.array([20.0]))[0] == pytest.approx(-3.0, abs=0.1)
    assert response_db(["highshelf:8000:+2"], SR, np.array([20000.0]))[0] == pytest.approx(2.0, abs=0.15)
    with pytest.raises(ValueError):
        response_db(["peak:3000"], SR, np.array([1.0]))


@needs_rx_plugins
def test_cli_processes_and_measures(tmp_path, capsys) -> None:
    from im4x_chain.cli import main
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


def test_progress_file_reports_stages_and_result(tmp_path) -> None:
    from im4x_chain.cli import main
    x, _ = speechlike([-30, -28])
    src = tmp_path / "p.wav"
    sf.write(str(src), x, SR, subtype="PCM_24")
    out, prog = tmp_path / "o", tmp_path / "progress.jsonl"
    args = ["process", str(src), "-o", str(out), "--no-eq", "--no-deplosive", "--no-declick",
            "--no-denoise", "--no-gate", "--no-leveler", "--no-comp", "--no-loudness",
            "--progress-file", str(prog)]
    assert main(args) == 0
    events = [json.loads(line) for line in prog.read_text().splitlines() if line.strip()]
    assert events[0]["event"] == "begin"
    assert events[0]["stages"] == "read,write,measure"
    starts = [e["stage"] for e in events if e.get("status") == "start"]
    assert starts == ["read", "write", "measure"]
    assert [e["event"] for e in events[-2:]] == ["stage", "done"]
    assert events[-1]["ok"] is True
    assert events[-1]["after_lufs"] == pytest.approx(events[-1]["before_lufs"], abs=0.1)
    assert (out / "p.wav").exists()


# Values measured on the real SSL Native X-Comp v6 (AU) with the chain's settings.
XCOMP_STATIC = {-16: -0.39, -12: -1.39, -8: -2.99, -4: -5.21, 0: -8.02}       # square wave, peak dBFS
XCOMP_DC_STEP = {"att": {0.25: -3.474, 0.5: -5.518, 1.0: -7.191, 5.0: -8.021},  # -30 dBFS -> 0 dBFS
                 "rel": {50: -6.476, 100: -5.098, 200: -2.875, 400: -0.419}}   # 0 dBFS -> -30 dBFS


def test_compressor_static_curve_matches_xcomp() -> None:
    from im4x_chain.compressor import XCOMP_DEFAULTS, static_curve
    d = XCOMP_DEFAULTS
    for lv, g in XCOMP_STATIC.items():
        assert static_curve(np.array([lv]), d["threshold"], d["ratio"], d["knee"])[0] == pytest.approx(g, abs=0.05)
    assert static_curve(np.array([-25.0]), d["threshold"], d["ratio"], d["knee"])[0] == 0.0


def test_compressor_timing_matches_xcomp() -> None:
    from im4x_chain.compressor import compress
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
    from im4x_chain.compressor import compress
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
    from im4x_chain.rx_plugins import _juce_decode, _juce_encode
    b = bytes(np.random.default_rng(3).integers(0, 256, 1001, dtype=np.uint8))
    assert _juce_decode(_juce_encode(b)) == b


@needs_rx_plugins
def test_spectral_denoise_learns_and_lowers_noise() -> None:
    from im4x_chain.rx_plugins import spectral_denoise
    x, spans = speechlike([-26, -24, -28], phrase_s=3.0, pause_s=1.5, noise_db=-60)
    y, info = spectral_denoise(x, SR)
    pause = slice(spans[1].stop + 4800, spans[2].start - 4800)
    assert rms_db(y[pause]) < rms_db(x[pause]) - 10                           # noise down by > 10 dB
    assert abs(rms_db(y[spans[1]]) - rms_db(x[spans[1]])) < 1.0               # speech level kept
    assert info["reduction_db"] == 16.0 and "residual_max_db" in info


@needs_rx_plugins
def test_plugin_render_in_chunks_matches_whole_file() -> None:
    from im4x_chain import rx_plugins as R
    x, _ = speechlike([-20] * 8, phrase_s=10.0, pause_s=0.5, noise_db=-50)   # ~84 s, no digital silence
    ctx, j = R._unpack(R._default_state(R.DECLICK_PLUGIN))
    raw = R._pack(ctx, j)
    y = R._render(R.DECLICK_PLUGIN, raw, x, SR, workers=2)
    p = R._load(R.DECLICK_PLUGIN)
    p.raw_state = raw
    ref = p.process(x[None].astype(np.float32), SR, buffer_size=4096)[0]
    assert 10 * np.log10(np.sum((y - ref) ** 2) / np.sum(ref ** 2) + 1e-30) < -50


def _gate_scene():
    """Clicks, breaths before / between / after / away from phrases, a quiet isolated word."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8 * SR) * 10 ** (-95 / 20)                   # residual noise floor

    def voiced(n, db):
        t = np.arange(n) / SR
        s = sum(np.sin(2 * np.pi * 160 * k * t) / k for k in range(1, 8))
        s *= np.hanning(n) ** 0.2
        return s * 10 ** (db / 20) / np.sqrt(np.mean(s ** 2))

    def breath(n, db):
        b = np.convolve(rng.standard_normal(n), np.ones(4) / 4, "same")
        r = int(0.02 * SR)
        env = np.ones(n)
        env[:r], env[-r:] = np.linspace(0, 1, r), np.linspace(1, 0, r)
        return b * env * 10 ** (db / 20) / np.sqrt(np.mean(b ** 2))

    def at(t0, t1):
        return int(t0 * SR), int(t1 * SR)

    spans = {"click": at(1.0, 1.01), "breath_before": at(1.5, 1.85), "phrase": at(2.0, 2.6),
             "gap": at(2.6, 2.85), "word2": at(2.85, 3.4), "tail": at(3.4, 3.7),
             "tail_click": at(3.6, 3.61), "lone_breath": at(5.0, 5.3), "quiet_word": at(6.5, 6.75)}
    for name, make, db in (("phrase", voiced, -20), ("word2", voiced, -20), ("quiet_word", voiced, -42),
                           ("breath_before", breath, -75), ("gap", breath, -75), ("tail", breath, -75),
                           ("lone_breath", breath, -75)):     # breaths after De-noise: 50-65 dB under speech
        a, b = spans[name]
        x[a:b] += make(b - a, db)
    for name in ("click", "tail_click"):
        a, b = spans[name]
        x[a:b] += rng.standard_normal(b - a) * 10 ** (-40 / 20)
    return x, spans


def test_gate_keeps_speech_gaps_and_tails_removes_the_rest():
    from im4x_chain.gate import gate
    x, spans = _gate_scene()
    y, info = gate(x, SR)
    assert info["applied"] and info["gaps_kept"] >= 1 and info["tails"] >= 1 and info["clicks_cut"] >= 1

    def rms_db(v):
        return 10 * np.log10(np.mean(v ** 2) + 1e-30)

    for name in ("click", "breath_before", "lone_breath", "tail_click"):
        a, b = spans[name]
        assert rms_db(y[a:b]) < rms_db(x[a:b]) - 40, name
    for name in ("phrase", "gap", "word2", "quiet_word"):
        a, b = spans[name]
        np.testing.assert_allclose(y[a:b], x[a:b], err_msg=name)        # untouched
    a, _ = spans["phrase"]
    assert np.all(y[a - int(0.015 * SR):a] == x[a - int(0.015 * SR):a])  # lookahead before the onset
    a, b = spans["tail"]
    np.testing.assert_allclose(y[a:a + int(0.1 * SR)], x[a:a + int(0.1 * SR)], err_msg="tail start")
    assert np.all(np.abs(y[b - int(0.01 * SR):b]) < np.abs(x[b - int(0.01 * SR):b]) + 1e-12)  # fading out
    y2, _ = gate(x, SR, keep_breaths=False)
    a, b = spans["gap"]
    assert rms_db(y2[a + int(0.12 * SR):b - int(0.04 * SR)]) < rms_db(x[a:b]) - 40  # breaths can go too


def test_gate_range_and_stereo():
    from im4x_chain.gate import gate
    x, spans = _gate_scene()
    st = np.stack([x, 0.5 * x], axis=1)
    y, _ = gate(st, SR, range_db=20)
    a, b = spans["click"]
    np.testing.assert_allclose(y[a:b], st[a:b] * 0.1, rtol=1e-9)        # -20 dB, both channels


def test_config_template_parses_and_is_all_defaults(tmp_path, monkeypatch):
    import tomllib

    from im4x_chain import config
    from im4x_chain.cli import main
    monkeypatch.setenv("IM4X_CHAIN_CONFIG", str(tmp_path / "c.toml"))
    assert main(["config", "--init"]) == 0
    text = (tmp_path / "c.toml").read_text(encoding="utf-8")
    assert tomllib.loads(text) == {s: {} for s in config.SCHEMA}         # every value commented out
    cfg, used = config.load()
    assert used == tmp_path / "c.toml" and config.effective(cfg)["loudness"]["target"] == -24.0
    assert main(["config", "--init"]) == 1                                 # no silent overwrite


@pytest.mark.parametrize("text, msg", [("[gate]\nkep_db = -20\n", "unknown key"),
                                       ("[louds]\ntarget = -23\n", "unknown section"),
                                       ("[loudness]\ntarget = \"-23\"\n", "must be float"),
                                       ("[stages]\ngate = 1\n", "must be bool"),
                                       ("[eq]\nbands = [\"hp:abc\"]\n", None)])
def test_config_rejects_typos(tmp_path, text, msg):
    from im4x_chain import config
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        config.load(p)


def test_config_file_applies_and_flags_win(tmp_path):
    from im4x_chain.cli import main
    x, _ = speechlike([-30, -28, -32])
    src = tmp_path / "s.wav"
    sf.write(str(src), x, SR, subtype="PCM_24")
    cfg = tmp_path / "c.toml"
    cfg.write_text("[stages]\ndeplosive = false\ndeclick = false\ndenoise = false\ncompressor = false\n"
                   "[loudness]\ntarget = -20.0\n[gate]\nkeep_breaths = false\n", encoding="utf-8")
    base = ["process", str(src), "--no-eq", "--config", str(cfg), "--json"]
    for extra, want in (([], -20.0), (["--target", "-22"], -22.0)):
        out = tmp_path / f"o{len(extra)}"
        assert main(base + ["-o", str(out)] + extra) == 0
        y, _ = sf.read(str(out / "s.wav"))
        assert abs(integrated_lufs(y, SR) - want) < 0.05
    with pytest.raises(SystemExit):                                       # a named file must exist
        main(base + ["-o", str(tmp_path / "o9"), "--config", str(tmp_path / "missing.toml")])


def _rx_preset(tmp_path, modules):
    """A Module Chain preset as RX writes it: base64 items with the module state as JSON text."""
    import base64
    import json

    def item(mid, enabled, params):
        state = {k: {"Type": "Float", "Value": v} for k, v in params.items()}
        d = {"ElementID": {"Type": "String", "Value": mid}, "ENABLED": {"Type": "Bool", "Value": enabled},
             "ElementState": {"Type": "String", "Value": json.dumps(state)}}
        return {"Type": "Base64", "Value": base64.b64encode(json.dumps(d).encode()).decode()}
    p = tmp_path / "My chain.xml"
    p.write_text(json.dumps({"actions": {"Type": "Array", "Value": [item(*m) for m in modules]},
                             "version": {"Type": "UInt", "Value": 1}}), encoding="utf-8")
    return p


def test_import_rx_preset_into_config(tmp_path):
    from im4x_chain import config
    from im4x_chain.cli import main
    eq = {"Band 1 Enable": True, "Band 1 Frequency": 80, "Band 1 Gain": 0, "Band 1 Q": 1, "Band 1 Shape": 4,
          "Band 2 Enable": True, "Band 2 Frequency": 3000, "Band 2 Gain": 2, "Band 2 Q": 1, "Band 2 Shape": 0,
          "Band 3 Enable": False, "Band 3 Frequency": 200, "Band 3 Gain": 0, "Band 3 Q": 1, "Band 3 Shape": 0,
          "Band 4 Enable": True, "Band 4 Frequency": 18000, "Band 4 Gain": 0, "Band 4 Q": 2, "Band 4 Shape": 3}
    preset = _rx_preset(tmp_path, [
        ("DenoiserSpectral", True, {"Denoiser Canopy Denoiser Suppression Linked": 17.899999618530273}),
        ("Equalizer", True, eq),
        ("MouthDeclick", True, {"Sensitivity": 5.5, "Frequency skew": 1.0, "Click widening": 0.5}),
        ("Leveler", True, {"Target Level": -23, "Responsiveness": 3.299999952316284,
                           "Preserve Dynamic Range": 65, "Ess Reduction": 4, "Ess Reduction Enabled": False}),
        ("PluginHost", True, {})])                               # no Loudness Control in this chain
    cfg = tmp_path / "c.toml"
    cfg.write_text("# mine\n[gate]\nclose_gap_ms = 500.0\n[declick]\n# sensitivity = 4.5\n", encoding="utf-8")
    for _ in range(2):                                           # importing again replaces, does not duplicate
        assert main(["import-rx", str(preset), "--config", str(cfg)]) == 0
    text = cfg.read_text(encoding="utf-8")
    assert "# mine" in text and text.count("sensitivity =") == 1
    got, _ = config.load(cfg)
    assert got["gate"] == {"close_gap_ms": 500.0}                # the user's own settings stay
    assert got["denoise"] == {"reduction_db": 17.9}
    assert got["declick"] == {"sensitivity": 5.5, "frequency_skew": 1.0, "click_widening": 0.5}
    assert got["leveler"] == {"target": -23.0, "responsiveness": 3.3, "preserve": 65.0, "ess_db": 0.0}
    assert got["stages"] == {"loudness": False} and not got.get("eq")   # EQ written commented out
    assert '# bands = ["hp:80:4", "peak:3000:+2:1", "lp:18000:8"]' in text
    assert main(["import-rx", str(preset), "--config", str(cfg), "--eq"]) == 0
    assert config.load(cfg)[0]["eq"] == {"bands": ["hp:80:4", "peak:3000:+2:1", "lp:18000:8"]}
    before = cfg.read_text(encoding="utf-8")
    assert main(["import-rx", str(preset), "--config", str(tmp_path / "new.toml"), "--dry-run"]) == 0
    assert not (tmp_path / "new.toml").exists() and cfg.read_text(encoding="utf-8") == before


def test_import_rx_rejects_other_files(tmp_path):
    from im4x_chain import rx_preset
    p = tmp_path / "x.xml"
    p.write_text("<xml/>", encoding="utf-8")
    with pytest.raises(ValueError, match="not an RX Module Chain preset"):
        rx_preset.read_modules(p)
    with pytest.raises(ValueError, match="RX preset not found"):
        rx_preset.find(str(tmp_path / "nope"))


def test_jobs_give_the_same_files(tmp_path):
    from im4x_chain.cli import main
    for i, levels in enumerate(([-30, -28, -32], [-26, -34, -29], [-31, -27, -33])):
        x, _ = speechlike(levels)
        sf.write(str(tmp_path / f"s{i}.wav"), x, SR, subtype="PCM_24")
    base = ["process", str(tmp_path / "s0.wav"), str(tmp_path / "s1.wav"), str(tmp_path / "s2.wav"),
            "--no-deplosive", "--no-declick", "--no-denoise", "--json"]
    assert main(base + ["-o", str(tmp_path / "one"), "--jobs", "1"]) == 0
    assert main(base + ["-o", str(tmp_path / "two"), "--jobs", "2"]) == 0
    for i in range(3):
        a, _ = sf.read(str(tmp_path / "one" / f"s{i}.wav"))
        b, _ = sf.read(str(tmp_path / "two" / f"s{i}.wav"))
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("text, msg", [
    ('[[plugins]]\nname = "De-ess"\n', "after is required"),
    ('[[plugins]]\nname = "De-ess"\nafter = "levler"\n', "after must be one of"),
    ('[[plugins]]\nname = "De-ess"\nafter = "leveler"\nwhen = 1\n', "unknown key 'when'"),
    ('[[plugins]]\nname = "De-ess"\nafter = "leveler"\n[plugins.settings]\nSpeed = "fast"\n', "must be a number"),
])
def test_config_plugins_are_checked(tmp_path, text, msg):
    from im4x_chain import config
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        config.load(p)


def test_config_template_and_dump_with_plugins_parse(tmp_path):
    import tomllib

    from im4x_chain import config
    p = tmp_path / "c.toml"
    p.write_text('[[plugins]]\nname = "De-ess"\nafter = "start"\n[plugins.settings]\n'
                 '"Gain Reduction Threshold" = -18.0\nSpeed = 1\n', encoding="utf-8")
    cfg, _ = config.load(p)
    eff = config.effective(cfg)
    assert tomllib.loads(config.dump(eff))["plugins"] == cfg["plugins"]
    assert "plugins" not in tomllib.loads(config.template())                # example is commented out


def test_extra_rx_plugin_runs_at_its_place(tmp_path):
    from im4x_chain import rx_plugins
    from im4x_chain.cli import main
    if "De-ess" not in rx_plugins.available_plugins():
        pytest.skip("RX De-ess not installed")
    rng = np.random.default_rng(3)
    x = rng.standard_normal(4 * SR) * 0.05                                 # bright noise: plenty of "ess"
    src = tmp_path / "s.wav"
    sf.write(str(src), x, SR, subtype="PCM_24")
    cfg = tmp_path / "c.toml"
    cfg.write_text('[stages]\neq = false\ndeplosive = false\ndeclick = false\ndenoise = false\ngate = false\n'
                   'leveler = false\ncompressor = false\nloudness = false\n'
                   '[[plugins]]\nname = "De-ess"\nafter = "leveler"\n[plugins.settings]\n'
                   '"Gain Reduction Threshold" = -40.0\n', encoding="utf-8")
    pf = tmp_path / "p.jsonl"
    assert main(["process", str(src), "-o", str(tmp_path / "o"), "--config", str(cfg),
                 "--progress-file", str(pf)]) == 0
    y, _ = sf.read(str(tmp_path / "o" / "s.wav"))
    assert 10 * np.log10(np.mean(y ** 2) / np.mean(x ** 2)) < -3               # the plugin cut the highs
    assert '"stages": "read,De-ess,write,measure"' in pf.read_text(encoding="utf-8")
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("-40.0", "-40.0\nTreshold = 1"), encoding="utf-8")
    with pytest.raises(SystemExit):                                         # a typo stops before any file
        main(["process", str(src), "-o", str(tmp_path / "o2"), "--config", str(cfg)])
    assert not (tmp_path / "o2").exists()
