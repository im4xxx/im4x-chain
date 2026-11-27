"""Reads an RX Module Chain preset (``.xml`` saved by RX) and turns it into config settings.

RX keeps saved chains in ``Documents/iZotope/RX Audio Editor/User Presets/Module Chain``.
A preset is JSON: ``actions`` is a list of base64-encoded items, each with ``ElementID``
(the module), ``ENABLED`` and ``ElementState`` (the module's parameters as JSON text).

What is carried over (module -> config section):

* Spectral De-noise -> ``[denoise] reduction_db`` (Reduction; the noise profile is learned
  per file anyway);
* Mouth De-click -> ``[declick]`` sensitivity, frequency skew, click widening;
* Leveler -> ``[leveler]`` target, responsiveness, preserve, ess_db (0 when Ess is off);
* Loudness Control -> ``[loudness]`` target and ceiling (Margin, true peak);
* EQ -> ``[eq] bands`` (enabled bands; high/low pass, bell, shelves). Written commented out
  unless asked for, because a fixed EQ replaces the low cut matched to the voice;
* a module that is switched off or missing -> ``[stages] <stage> = false``.

Not carried over: plugins hosted in the chain (Plugin Host, e.g. the compressor: their state
is the plugin's own binary data), Leveler breath control, Loudness short-term limits.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

# RX EQ band shapes (Band N Shape); 0 = bell is the default of the middle bands
EQ_SHAPES = {0: "peak", 1: "lowshelf", 2: "highshelf", 3: "lp", 4: "hp"}
MODULES = {"DenoiserSpectral": "denoise", "MouthDeclick": "declick", "Leveler": "leveler",
           "RXLoudness": "loudness", "Equalizer": "eq"}


def presets_dir() -> Path:
    docs = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents" if sys.platform == "win32" \
        else Path.home() / "Documents"
    return docs / "iZotope" / "RX Audio Editor" / "User Presets" / "Module Chain"


def list_presets() -> list[Path]:
    d = presets_dir()
    return sorted(d.glob("*.xml")) if d.is_dir() else []


def find(name: str) -> Path:
    """A path, or a preset name from the RX folder (with or without .xml)."""
    p = Path(name).expanduser()
    if p.is_file():
        return p
    for cand in (presets_dir() / name, presets_dir() / f"{name}.xml"):
        if cand.is_file():
            return cand
    names = ", ".join(q.stem for q in list_presets()) or "none"
    raise ValueError(f"RX preset not found: {name} (presets in {presets_dir()}: {names})")


def read_modules(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data["actions"]["Value"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path} is not an RX Module Chain preset ({type(e).__name__}: {e})") from None
    out = []
    for it in items:
        d = json.loads(base64.b64decode(it["Value"]))
        state = json.loads(d.get("ElementState", {}).get("Value") or "{}")
        out.append({"id": d.get("ElementID", {}).get("Value", "?"),
                    "enabled": bool(d.get("ENABLED", {}).get("Value", True)),
                    "params": {k: v.get("Value") for k, v in state.items() if isinstance(v, dict)}})
    return out


def _band(p: dict[str, Any], n: int) -> str | None:
    kind = EQ_SHAPES.get(int(p.get(f"Band {n} Shape", -1)))
    freq = float(p[f"Band {n} Frequency"])
    q = float(p.get(f"Band {n} Q", 1.0))
    if kind is None:
        return None
    if kind in ("hp", "lp"):            # Q of a pass filter is its slope: 1 = 24 dB/oct, 2 = 48 dB/oct
        return f"{kind}:{freq:g}:{min(8, max(1, round(q * 4)))}"
    return f"{kind}:{freq:g}:{float(p[f'Band {n} Gain']):+g}:{q:.3g}"


def to_settings(modules: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Returns (config values by section, notes about what was not carried over)."""
    cfg: dict[str, dict[str, Any]] = {"stages": {}}
    notes: list[str] = []
    seen = set()
    for m in modules:
        sec, p = MODULES.get(m["id"]), m["params"]
        if sec is None:
            what = "плагин в Plugin Host (например, компрессор)" if m["id"] == "PluginHost" else m["id"]
            notes.append(f"{what}: не переносится")
            continue
        seen.add(sec)
        stage = "eq" if sec == "eq" else sec
        if not m["enabled"]:
            cfg["stages"][stage] = False
            continue
        if sec == "denoise":
            cfg["denoise"] = {"reduction_db": round(float(p["Denoiser Canopy Denoiser Suppression Linked"]), 2)}
        elif sec == "declick":
            cfg["declick"] = {"sensitivity": float(p["Sensitivity"]), "frequency_skew": float(p["Frequency skew"]),
                              "click_widening": float(p["Click widening"])}
        elif sec == "leveler":
            cfg["leveler"] = {"target": float(p["Target Level"]),
                              "responsiveness": round(float(p["Responsiveness"]), 3),
                              "preserve": float(p["Preserve Dynamic Range"]),
                              "ess_db": float(p["Ess Reduction"]) if p.get("Ess Reduction Enabled", True) else 0.0}
            if p.get("Breath Level Enabled"):
                notes.append("Leveler: Breath Level не переносится")
        elif sec == "loudness":
            cfg["loudness"] = {"target": float(p["Loudness"]), "ceiling": float(p["Margin"])}
            if p.get("Limit Short-Term Loudness"):
                notes.append("Loudness: ограничение short-term не переносится")
        elif sec == "eq":
            bands, n = [], 1
            while f"Band {n} Frequency" in p:
                if p.get(f"Band {n} Enable"):
                    b = _band(p, n)
                    if b is None:
                        notes.append(f"EQ: полоса {n} неизвестной формы, пропущена")
                    else:
                        bands.append(b)
                n += 1
            if bands:
                cfg["eq"] = {"bands": bands}
            else:
                cfg["stages"]["eq"] = False
    for sec in ("denoise", "declick", "leveler", "loudness", "eq"):
        if sec not in seen:
            cfg["stages"][sec] = False
    if not cfg["stages"]:
        del cfg["stages"]
    return cfg, notes
