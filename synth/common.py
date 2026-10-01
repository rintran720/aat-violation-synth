import json
import os
from pathlib import Path


def load_config(path=None):
    """Read the config named by `path`, else $SYNTH_CONFIG, else config.json.
    A config with "extends": "<base.json>" overrides only the keys it sets (per-camera configs)."""
    path = Path(path or os.environ.get("SYNTH_CONFIG", "config.json"))
    cfg = json.loads(path.read_text())
    if "extends" in cfg:
        base = load_config(cfg.pop("extends"))
        base.pop("work", None)
        cfg = {**base, **cfg}
    cfg["work"] = Path(cfg["work_dir"])
    return cfg


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def reference_images(cfg):
    """Every .png/.jpg image in the camera's reference images folder (object_frames_dir), sorted by name."""
    return sorted(p for p in Path(cfg["object_frames_dir"]).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
