"""User settings for the chain: a TOML file read by the CLI on every run.

Location (the first that applies):

* ``--config PATH`` on the command line;
* the ``IM4X_CHAIN_CONFIG`` environment variable;
* Windows: ``%APPDATA%\\im4x-chain\\config.toml``;
* macOS / Linux: ``$XDG_CONFIG_HOME/im4x-chain/config.toml``, by default
  ``~/.config/im4x-chain/config.toml``.

A missing file means the built-in defaults. Every key is optional; command-line flags win over
the file, the file wins over the defaults. Unknown sections or keys and wrong types are errors,
so a typo does not silently fall back to a default. ``im4x-chain config --init`` writes a
template with every setting and its default, commented out.
"""

from __future__ import annotations

import inspect
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from im4x_chain.audio_chain import COMP_DEFAULTS, LEVELER_DEFAULTS, LOUDNESS_DEFAULTS
from im4x_chain.gate import gate
from im4x_chain.rx_plugins import chain_settings

STAGES = ["eq", "deplosive", "declick", "denoise", "gate", "leveler", "compressor", "loudness"]
GATE_DEFAULTS = {k: p.default for k, p in inspect.signature(gate).parameters.items()
                 if k not in ("x", "sr") and p.default is not None}
COMP_KEYS = ["threshold", "ratio", "attack", "release", "knee", "makeup"]
_RX = chain_settings()                       # the values of the RX Module Chain the plugins use
_DECLICK = _RX["mouth_declick"]["settings"]

# section -> key -> (type, default or None when the default is computed per file)
SCHEMA: dict[str, dict[str, tuple[type, Any]]] = {
    "stages": {s: (bool, True) for s in STAGES},
    "eq": {"bands": (list, None)},
    "deplosive": {"sensitivity": (float, 5.0), "strength": (float, 5.0), "frequency_limit_hz": (float, None)},
    "declick": {"sensitivity": (float, float(_DECLICK["Sensitivity"])),
                "frequency_skew": (float, float(_DECLICK["Frequency skew"])),
                "click_widening": (float, float(_DECLICK["Click widening"]))},
    "denoise": {"reduction_db": (float, float(_RX["spectral_denoise"]["settings"]
                                              ["Denoiser Canopy Denoiser Suppression Linked"]))},
    "gate": {**{k: (type(v), v) for k, v in GATE_DEFAULTS.items()}, "range_db": (float, None)},
    "leveler": {k: (float, v) for k, v in LEVELER_DEFAULTS.items()},
    "compressor": {k: (float, COMP_DEFAULTS[k]) for k in COMP_KEYS},
    "loudness": {k: (float, v) for k, v in LOUDNESS_DEFAULTS.items()},
}

# settings without a fixed default: what is used then, and an example value
AUTO = {("eq", "bands"): ("по голосу: hp:70..120:4 (срез под нижний тон голоса) + lp:18000:8",
                          ["hp:90:4", "lp:18000:8"]),
        ("deplosive", "frequency_limit_hz"): ("по голосу: 0.8 x нижний тон, 60-200 Hz", 120.0),
        ("gate", "range_db"): ("глушить полностью", 40.0)}

NOTES = {
    "stages": "этапы цепочки: false — выключить (как --no-<этап>)",
    "eq": "полосы EQ; без bands — срез низа по голосу (70-120 Hz) + lp:18000:8.\n"
          "# формат полосы: hp:F[:порядок] lp:F[:порядок] peak:F:dB[:Q] lowshelf:F:dB[:Q] highshelf:F:dB[:Q]",
    "deplosive": "плагин RX De-plosive; frequency_limit_hz без значения — 0.8 x нижний тон голоса (60-200 Hz)",
    "declick": "плагин RX Mouth De-click; по умолчанию — значения из Module Chain RX",
    "denoise": "плагин RX Spectral De-noise; reduction_db по умолчанию — из Module Chain RX",
    "gate": "гейт: щелчки и шумы вокруг фраз; keep_breaths — оставлять звук между близкими словами\n"
            "# и хвост после фразы; range_db без значения — глушить полностью",
    "leveler": "Leveler: target LUFS, preserve — % разницы между фразами",
    "compressor": "компрессор (модель SSL X-Comp): threshold dBFS, attack/release мс, knee dB",
    "loudness": "Loudness Control: target LUFS, ceiling — потолок true peak, dBTP",
}


def config_path(explicit: str | os.PathLike | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("IM4X_CHAIN_CONFIG")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "im4x-chain" / "config.toml"


def _check(section: str, key: str, value: Any) -> Any:
    if section not in SCHEMA:
        raise ValueError(f"config: unknown section [{section}] (known: {', '.join(SCHEMA)})")
    if key not in SCHEMA[section]:
        raise ValueError(f"config: unknown key {key!r} in [{section}] (known: {', '.join(SCHEMA[section])})")
    typ = SCHEMA[section][key][0]
    if typ is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if typ is list:
        if not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise ValueError(f"config: [{section}] {key} must be a list of strings")
        return value
    if not isinstance(value, typ) or (typ is not bool and isinstance(value, bool)):
        raise ValueError(f"config: [{section}] {key} must be {typ.__name__}, got {value!r}")
    return value


def load(path: str | os.PathLike | None = None) -> tuple[dict[str, dict[str, Any]], Path | None]:
    """Returns (settings by section, the file read or None when there is none)."""
    p = config_path(path)
    if not p.is_file():
        if path:                                    # asked for explicitly: must exist
            raise ValueError(f"config file not found: {p}")
        return {}, None
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"config {p}: {e}") from None
    out: dict[str, dict[str, Any]] = {}
    for section, values in data.items():
        if not isinstance(values, dict):
            raise ValueError(f"config: {section!r} must be a section [{section}]")
        out[section] = {k: _check(section, k, v) for k, v in values.items()}
    if "bands" in out.get("eq", {}):
        from im4x_chain.eq import parse_band
        for b in out["eq"]["bands"]:
            parse_band(b)
    return out, p


def effective(cfg: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Defaults overlaid with the file (None = computed per file)."""
    return {s: {k: cfg.get(s, {}).get(k, d) for k, (_, d) in keys.items()} for s, keys in SCHEMA.items()}


def _toml(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    return repr(float(v)) if isinstance(v, float) else str(v)


def template() -> str:
    lines = ["# im4x-chain: настройки цепочки. Раскомментируйте и измените нужное;",
             "# остальное берётся по умолчанию. Ключи командной строки важнее этого файла.", ""]
    for section, keys in SCHEMA.items():
        lines += [f"# {NOTES[section]}", f"[{section}]"]
        for k, (_, d) in keys.items():
            if d is not None:
                lines.append(f"# {k} = {_toml(d)}")
            else:
                auto, example = AUTO[(section, k)]
                lines.append(f"# {k} = {_toml(example)}   # пример; по умолчанию — {auto}")
        lines.append("")
    return "\n".join(lines)


def dump(settings: dict[str, dict[str, Any]]) -> str:
    lines = []
    for section, keys in settings.items():
        lines.append(f"[{section}]")
        for k, v in keys.items():
            lines.append(f"{k} = {_toml(v)}" if v is not None else f"# {k}: {AUTO[(section, k)][0]}")
        lines.append("")
    return "\n".join(lines)


def set_values(text: str, values: dict[str, dict[str, Any]], note: str = "",
               commented: set[tuple[str, str]] = frozenset()) -> str:
    """Sets keys in TOML text, keeping its comments and other keys: a line ``key = ...`` or
    ``# key = ...`` in the section is replaced, otherwise the key goes right after the section
    header; a missing section is appended. Keys in ``commented`` are written commented out."""
    import re
    lines = text.splitlines()
    for section, keys in values.items():
        for key, value in keys.items():
            _check(section, key, value)
            line = f"{key} = {_toml(value)}" + (f"   # {note}" if note else "")
            if (section, key) in commented:
                line = "# " + line
            head = next((i for i, s in enumerate(lines) if s.strip() == f"[{section}]"), None)
            if head is None:
                lines += ["", f"[{section}]"]
                head = len(lines) - 1
            end = next((i for i in range(head + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
            pat = re.compile(rf"^\s*#?\s*{re.escape(key)}\s*=")
            hit = next((i for i in range(head + 1, end) if pat.match(lines[i])), None)
            if hit is None:
                lines.insert(head + 1, line)
            else:
                lines[hit] = line
    return "\n".join(lines) + "\n"
