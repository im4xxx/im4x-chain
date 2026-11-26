"""Command line for the voice chain without the RX app, and metering:
EQ -> De-plosive (where needed) -> Mouth De-click -> Spectral De-noise -> Leveler
-> compressor -> Loudness Control.

    rx-audio process FILES_OR_DIRS... [-o OUT_DIR] [--eq hp:60 --eq peak:3000:+2:1] [--no-eq]
                     [--no-deplosive] [--no-declick] [--no-denoise]
                     [--no-leveler] [--no-comp] [--no-loudness] [--target -24] [--ceiling -3] ...
    rx-audio measure FILES_OR_DIRS...

Files are changed in place unless ``-o`` is given. Exit code 1 if any file failed.
"""

from __future__ import annotations

import argparse
import json
import sys

from rx_audio import audio_chain
from rx_audio.eq import DEFAULT_BANDS


def _fmt(m: dict) -> str:
    return f"{m['lufs']:7.2f} LUFS  TP {m['true_peak_db']:6.2f} dBTP"


def _process(a: argparse.Namespace) -> int:
    bands = [] if a.no_eq else (a.eq or None)
    res = audio_chain.process_files(
        a.paths, eq_bands=bands, use_leveler=not a.no_leveler, use_loudness=not a.no_loudness,
        leveler_opts={"target": a.leveler_target, "responsiveness": a.responsiveness,
                      "preserve": a.preserve, "ess_db": a.ess},
        loudness_opts={"target": a.target, "ceiling": a.ceiling},
        use_comp=not a.no_comp, use_denoise=not a.no_denoise, denoise_reduction_db=a.reduction,
        use_declick=not a.no_declick, use_deplosive=not a.no_deplosive,
        comp_opts={"threshold": a.comp_threshold, "ratio": a.comp_ratio, "attack": a.comp_attack,
                   "release": a.comp_release, "knee": a.comp_knee, "makeup": a.comp_makeup},
        output_dir=a.output_dir)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    else:
        if not res["count"]:
            print("no WAV/FLAC files found", file=sys.stderr)
        for r in res["results"]:
            if not r["ok"]:
                print(f"FAIL {r['file']}: {r['error']}", file=sys.stderr)
                continue
            note = ""
            if "loudness" in r and not r["loudness"].get("reached", True):
                note = "  (target not reached without distortion)"
            print(f"ok   {r['output']}\n     {_fmt(r['before'])}  ->  {_fmt(r['after'])}  "
                  f"[{r['seconds']} s]{note}")
            if "denoise" in r:
                d = r["denoise"]
                print(f"     de-noise: residual max {d['residual_max_db']} dB, spread {d['residual_spread_db']} dB, "
                      f"bump {d['residual_bump_db']} dB" + ("" if d["passed"] else "  (not flat / below -120 dB)"))
            print("     " + ", ".join(f"{k} {v}s" for k, v in r.get("timings_s", {}).items()))
            if r.get("warning"):
                print(f"     warning: {r['warning']}", file=sys.stderr)
    return 0 if res["ok"] else 1


def _measure(a: argparse.Namespace) -> int:
    import soundfile as sf

    from rx_audio.loudness import measure
    rc, out = 0, []
    for f in audio_chain.expand_paths(a.paths):
        try:
            x, sr = sf.read(str(f), dtype="float64", always_2d=False)
            m = measure(x, sr)
            out.append({"file": str(f), **m})
            if not a.json:
                print(f"{_fmt(m)}  {f}")
        except Exception as e:  # noqa: BLE001
            rc = 1
            print(f"FAIL {f}: {type(e).__name__}: {e}", file=sys.stderr)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rx-audio",
                                description="Voice chain without the RX app: EQ, De-plosive, Mouth De-click, De-noise, Leveler, compressor, Loudness.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("process", help="EQ -> De-noise -> Mouth De-click -> Leveler -> Loudness -> compressor (in place unless -o)")
    pr.add_argument("paths", nargs="+", help="WAV/FLAC files or folders")
    pr.add_argument("-o", "--output-dir", help="write results here instead of replacing the files")
    pr.add_argument("--eq", action="append", metavar="BAND",
                    help=f"EQ band, repeatable; replaces the default {DEFAULT_BANDS}. "
                         "hp:F[:order] lp:F[:order] peak:F:dB[:Q] lowshelf:F:dB[:Q] highshelf:F:dB[:Q]")
    pr.add_argument("--no-eq", action="store_true", help="skip the EQ stage")
    pr.add_argument("--no-denoise", action="store_true", help="skip Spectral De-noise (RX plugin)")
    pr.add_argument("--reduction", type=float, default=None,
                    help="Spectral De-noise Reduction, dB (default: the chain's, 16)")
    pr.add_argument("--no-declick", action="store_true", help="skip Mouth De-click (RX plugin)")
    pr.add_argument("--no-deplosive", action="store_true",
                    help="skip De-plosive (RX plugin; by default kept only on files with plosives)")
    pr.add_argument("--no-leveler", action="store_true")
    pr.add_argument("--no-loudness", action="store_true")
    pr.add_argument("--leveler-target", type=float, default=-24.0, help="LUFS (default -24)")
    pr.add_argument("--responsiveness", type=float, default=3.3)
    pr.add_argument("--preserve", type=float, default=45.0, help="%% of phrase dynamics kept (default 45)")
    pr.add_argument("--ess", type=float, default=4.0, help="max de-ess reduction, dB (default 4)")
    pr.add_argument("--target", type=float, default=-24.0, help="loudness target, LUFS (default -24)")
    pr.add_argument("--ceiling", type=float, default=-3.0, help="true-peak ceiling, dBTP (default -3)")
    pr.add_argument("--no-comp", action="store_true", help="skip the compressor (X-Comp model)")
    pr.add_argument("--comp-threshold", type=float, default=-20.0, help="dBFS (default -20)")
    pr.add_argument("--comp-ratio", type=float, default=3.0)
    pr.add_argument("--comp-attack", type=float, default=0.1, help="ms (default 0.1)")
    pr.add_argument("--comp-release", type=float, default=120.0, help="ms (default 120)")
    pr.add_argument("--comp-knee", type=float, default=8.0, help="dB (default 8)")
    pr.add_argument("--comp-makeup", type=float, default=0.0, help="dB (default 0)")
    pr.add_argument("--json", action="store_true", help="print the full report as JSON")
    pr.set_defaults(func=_process)

    me = sub.add_parser("measure", help="integrated loudness and true peak")
    me.add_argument("paths", nargs="+")
    me.add_argument("--json", action="store_true")
    me.set_defaults(func=_measure)

    a = p.parse_args(argv)
    if a.cmd == "process" and a.eq and a.no_eq:
        p.error("--eq and --no-eq together")
    try:
        return a.func(a)
    except ValueError as e:
        p.error(str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
