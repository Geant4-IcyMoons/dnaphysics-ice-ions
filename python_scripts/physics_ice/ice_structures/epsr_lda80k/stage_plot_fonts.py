#!/usr/bin/env python3
"""Stage the project Courier-compatible fonts for font-poor compute nodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PHYSICS_ICE = HERE.parents[1]
sys.path.insert(0, str(PHYSICS_ICE))

from ice_structures.epsr_lda80k.model import file_sha256  # noqa: E402


DEFAULT_OUTPUT = (
    PHYSICS_ICE
    / "process_evidence"
    / "ice_structures"
    / "validation"
    / "runs"
    / "epsr_lda80k"
    / "plot_fonts"
)
FILENAMES = (
    "NimbusMonoPS-Regular.otf",
    "NimbusMonoPS-Italic.otf",
    "NimbusMonoPS-Bold.otf",
    "NimbusMonoPS-BoldItalic.otf",
)
SYSTEM_DIRECTORY = Path("/usr/share/fonts/urw-base35")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-dir", type=Path, default=SYSTEM_DIRECTORY)
    return parser.parse_args()


def stage(source_dir: Path, output_dir: Path) -> dict[str, object]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FILENAMES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Courier-compatible font files missing from {source_dir}: {missing}"
        )
    hashes: dict[str, str] = {}
    for name in FILENAMES:
        output = output_dir / name
        with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(source_dir / name, temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        output.chmod(0o644)
        hashes[name] = file_sha256(output)
    manifest: dict[str, object] = {
        "purpose": "local runtime fonts for consistent paper-style plots",
        "source_directory": str(source_dir),
        "font_family": "Nimbus Mono PS (Courier-compatible URW Base 35)",
        "files": hashes,
        "distribution": "local ignored runtime dependency; not vendored",
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o644)
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(stage(args.source_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
