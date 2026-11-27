"""Command line for the voice chain without the RX app, and metering:
EQ -> De-plosive (where needed) -> Mouth De-click -> Spectral De-noise -> Gate -> Leveler
-> compressor -> Loudness Control.

    im4x-chain process FILES_OR_DIRS... [-o OUT_DIR] [--eq hp:60 --eq peak:3000:+2:1] [--no-eq]
                     [--no-deplosive] [--no-declick] [--no-denoise] [--no-gate]
                     [--no-leveler] [--no-comp] [--no-loudness] [--target -24] [--ceiling -3]
                     [--progress-file PATH] ...
    im4x-chain measure FILES_OR_DIRS...
    im4x-chain config [--init [--force]] [--path] [--config PATH]
    im4x-chain import-rx [PRESET] [--eq] [--dry-run] [--config PATH]

Files are changed in place unless ``-o`` is given. Exit code 1 if any file failed.
Settings: built-in defaults < the config file (``im4x-chain config --path``) < flags.
"""

from __future__ import annotations

import argparse
import json
import sys

from im4x_chain import audio_chain, config
from im4x_chain.eq import DEFAULT_BANDS


def _fmt(m: dict) -> str:
    return f"{m['lufs']:7.2f} LUFS  TP {m['true_peak_db']:6.2f} dBTP"


def _given(**kw) -> dict:
    return {k: v for k, v in kw.items() if v is not None}


def _process(a: argparse.Namespace) -> int:
    cfg, cfg_file = config.load(a.config)
    st = {**{s: True for s in config.STAGES}, **cfg.get("stages", {})}
    for name, off in (("eq", a.no_eq), ("deplosive", a.no_deplosive), ("declick", a.no_declick),
                      ("denoise", a.no_denoise), ("gate", a.no_gate), ("leveler", a.no_leveler),
                      ("compressor", a.no_comp), ("loudness", a.no_loudness)):
        if off:
            st[name] = False
    if a.eq:
        st["eq"] = True
    bands = [] if not st["eq"] else (a.eq or cfg.get("eq", {}).get("bands") or None)
    reduction = a.reduction if a.reduction is not None else cfg.get("denoise", {}).get("reduction_db")
    if cfg_file is not None and not a.json:
        print(f"config: {cfg_file}")
    try:
        pf = open(a.progress_file, "w", encoding="utf-8", buffering=1) if a.progress_file else None
    except OSError as e:
        print(f"cannot write progress file {a.progress_file}: {e}", file=sys.stderr)
        return 1

    def progress(event: dict) -> None:
        if pf is not None:
            pf.write(json.dumps(event, ensure_ascii=False) + "\n")
            pf.flush()

    try:
        res = audio_chain.process_files(
            a.paths, eq_bands=bands, use_leveler=st["leveler"], use_loudness=st["loudness"],
            leveler_opts={**cfg.get("leveler", {}),
                          **_given(target=a.leveler_target, responsiveness=a.responsiveness,
                                   preserve=a.preserve, ess_db=a.ess)},
            loudness_opts={**cfg.get("loudness", {}), **_given(target=a.target, ceiling=a.ceiling)},
            use_comp=st["compressor"], use_denoise=st["denoise"], denoise_reduction_db=reduction,
            use_declick=st["declick"], use_deplosive=st["deplosive"], use_gate=st["gate"],
            comp_opts={**cfg.get("compressor", {}),
                       **_given(threshold=a.comp_threshold, ratio=a.comp_ratio, attack=a.comp_attack,
                                release=a.comp_release, knee=a.comp_knee, makeup=a.comp_makeup)},
            gate_opts=cfg.get("gate"), deplosive_opts=cfg.get("deplosive"), declick_opts=cfg.get("declick"),
            output_dir=a.output_dir, progress=progress if pf is not None else None,
            jobs=a.jobs if a.jobs is not None else cfg.get("process", {}).get("jobs", 2))
    finally:
        if pf is not None:
            pf.close()
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

    from im4x_chain.loudness import measure
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


def _config(a: argparse.Namespace) -> int:
    path = config.config_path(a.config)
    if a.path:
        print(path)
        return 0
    if a.init:
        if path.exists() and not a.force:
            print(f"{path} already exists (--force to overwrite)", file=sys.stderr)
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config.template(), encoding="utf-8")
        print(f"written {path}")
        return 0
    cfg, used = config.load(a.config)
    print(f"# config: {used if used else str(path) + ' (not found, built-in defaults)'}\n")
    print(config.dump(config.effective(cfg)))
    return 0


def _import_rx(a: argparse.Namespace) -> int:
    from im4x_chain import rx_preset
    if not a.preset:
        found = rx_preset.list_presets()
        print(f"RX Module Chain presets in {rx_preset.presets_dir()}:")
        for q in found:
            print(f"  {q.stem}")
        if not found:
            print("  (none)")
        print('\nimport one: im4x-chain import-rx "NAME"')
        return 0
    src = rx_preset.find(a.preset)
    values, notes = rx_preset.to_settings(rx_preset.read_modules(src))
    path = config.config_path(a.config)
    text = path.read_text(encoding="utf-8") if path.is_file() else config.template()
    commented = set() if a.eq else {("eq", "bands")}
    new = config.set_values(text, values, note=f"из RX: {src.stem}", commented=commented)
    print(f"{'would write' if a.dry_run else 'written'} {path} from {src.name}:")
    for section, keys in values.items():
        for k, v in keys.items():
            off = " (закомментировано: EQ остаётся по голосу; --eq, чтобы включить)" \
                if (section, k) in commented else ""
            print(f"  [{section}] {k} = {config._toml(v)}{off}")
    for n in notes:
        print(f"  - {n}")
    if not a.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new, encoding="utf-8")
        config.load(path)                                   # the result must read back
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="im4x-chain",
                                description="Voice chain without the RX app: EQ, De-plosive, Mouth De-click, De-noise, Gate, Leveler, compressor, Loudness.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("process", help="EQ -> De-plosive -> Mouth De-click -> De-noise -> Gate -> Leveler -> compressor -> Loudness (in place unless -o)")
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
    pr.add_argument("--no-gate", action="store_true",
                    help="skip the gate (clicks and noises around phrases; breaths stay)")
    pr.add_argument("--no-deplosive", action="store_true",
                    help="skip De-plosive (RX plugin; by default kept only on files with plosives)")
    pr.add_argument("--no-leveler", action="store_true")
    pr.add_argument("--no-loudness", action="store_true")
    pr.add_argument("--leveler-target", type=float, help="LUFS (default -24)")
    pr.add_argument("--responsiveness", type=float, help="default 3.3")
    pr.add_argument("--preserve", type=float, help="%% of phrase dynamics kept (default 60)")
    pr.add_argument("--ess", type=float, help="max de-ess reduction, dB (default 4)")
    pr.add_argument("--target", type=float, help="loudness target, LUFS (default -24)")
    pr.add_argument("--ceiling", type=float, help="true-peak ceiling, dBTP (default -3)")
    pr.add_argument("--no-comp", action="store_true", help="skip the compressor (X-Comp model)")
    pr.add_argument("--comp-threshold", type=float, help="dBFS (default -20)")
    pr.add_argument("--comp-ratio", type=float, help="default 3")
    pr.add_argument("--comp-attack", type=float, help="ms (default 0.1)")
    pr.add_argument("--comp-release", type=float, help="ms (default 120)")
    pr.add_argument("--comp-knee", type=float, help="dB (default 8)")
    pr.add_argument("--comp-makeup", type=float, help="dB (default 0)")
    pr.add_argument("--json", action="store_true", help="print the full report as JSON")
    pr.add_argument("--progress-file", metavar="PATH",
                    help="write machine-readable progress (JSON lines: begin/stage/done/error) here")
    pr.add_argument("--config", metavar="PATH", help="settings file (default: im4x-chain config --path)")
    pr.add_argument("-j", "--jobs", type=int, metavar="N",
                    help="files processed at the same time (default 2; 1 = one after another)")
    pr.set_defaults(func=_process)

    co = sub.add_parser("config", help="show the settings in effect, or write a config template")
    co.add_argument("--init", action="store_true", help="write a template with every setting, commented out")
    co.add_argument("--force", action="store_true", help="with --init: overwrite an existing file")
    co.add_argument("--path", action="store_true", help="print the config file location")
    co.add_argument("--config", metavar="PATH", help="another settings file")
    co.set_defaults(func=_config)

    ir = sub.add_parser("import-rx", help="take settings from an RX Module Chain preset into the config")
    ir.add_argument("preset", nargs="?", help="preset name from the RX folder or a path to .xml; none = list presets")
    ir.add_argument("--eq", action="store_true", help="also use the preset's EQ instead of the low cut matched to the voice")
    ir.add_argument("--dry-run", action="store_true", help="show what would be written, change nothing")
    ir.add_argument("--config", metavar="PATH", help="another settings file")
    ir.set_defaults(func=_import_rx)

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
