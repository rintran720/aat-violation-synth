import json
from pathlib import Path


def load_config(path="config.json"):
    cfg = json.loads(Path(path).read_text())
    cfg["work"] = Path(cfg["work_dir"])
    return cfg


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def reference_images(cfg):
    """Every .png/.jpg image in the camera's reference images folder (object_frames_dir), sorted by name."""
    return sorted(p for p in Path(cfg["object_frames_dir"]).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
