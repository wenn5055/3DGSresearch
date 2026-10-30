# ARCHIVE CANDIDATE (2026-10-29): Lego-only setup check; not applicable to configured Materials scenes.
# Successor: python scripts/run_experiment.py CONFIG --prepare-only (configured-dataset checks).
# Keep this only if the standalone Lego check is still wanted.

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cp4282_gs import DATA_ROOT


required = [DATA_ROOT / "init.ply"]
missing = [path for path in required if not path.exists()]


def image_candidates(frame_path: str):
    base = DATA_ROOT / frame_path
    if base.suffix:
        yield base
        for suffix in (".png", ".jpg", ".jpeg"):
            yield base.with_suffix(suffix)
    else:
        for suffix in (".png", ".jpg", ".jpeg"):
            yield base.with_suffix(suffix)


for manifest_name in ("transforms_train.json", "transforms_val.json", "transforms_test.json"):
    manifest_path = DATA_ROOT / manifest_name
    if not manifest_path.exists():
        missing.append(manifest_path)
        continue
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read {manifest_path}: {error}") from error
    for frame in manifest.get("frames", []):
        frame_path = frame.get("file_path")
        if not isinstance(frame_path, str):
            missing.append(f"{manifest_path}: frame without a file_path")
            continue
        if not any(path.exists() for path in image_candidates(frame_path)):
            missing.append(f"{manifest_path}: missing image for {frame_path}")

if missing:
    raise SystemExit("Missing setup files:\n" + "\n".join(str(path) for path in missing))

print(f"Data directory: {DATA_ROOT}")
print("Setup files are present.")
