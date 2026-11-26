"""Copy FLAC tags and cover art with ``metaflac`` (FLAC in, FLAC out, metadata kept)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def require_metaflac() -> None:
    if not shutil.which("metaflac"):
        raise RuntimeError("metaflac not found: install FLAC tools (macOS: brew install flac; "
                           "Windows: flac from xiph.org on PATH) so tags and cover are kept")


def copy_flac_metadata(src: Path, dst: Path) -> None:
    """Copy Vorbis tags and the picture of FLAC ``src`` (if it exists) into FLAC ``dst``."""
    if not src.exists():
        return
    tags = subprocess.run(["metaflac", "--export-tags-to=-", str(src)],
                          capture_output=True, text=True).stdout
    if tags.strip():
        subprocess.run(["metaflac", "--import-tags-from=-", str(dst)], input=tags, text=True, check=True)
    pics = subprocess.run(["metaflac", "--list", "--block-type=PICTURE", str(src)],
                          capture_output=True, text=True).stdout
    if "PICTURE" in pics:
        with tempfile.TemporaryDirectory() as td:
            pic = Path(td) / "cover"
            subprocess.run(["metaflac", f"--export-picture-to={pic}", str(src)], check=False)
            if pic.exists():
                subprocess.run(["metaflac", f"--import-picture-from={pic}", str(dst)], check=False)
