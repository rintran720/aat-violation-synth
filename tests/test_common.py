import json
from pathlib import Path

from synth.common import load_config


def test_extends_overrides_only_given_keys(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"camera_id": "cam01", "work_dir": "work", "seed": 7}))
    cam = tmp_path / "ch14.json"
    cam.write_text(json.dumps({"extends": str(base), "camera_id": "ch14", "work_dir": "work/cameras/ch14"}))
    cfg = load_config(cam)
    assert cfg["camera_id"] == "ch14" and cfg["seed"] == 7
    assert cfg["work"] == Path("work/cameras/ch14")
    assert "extends" not in cfg


def test_env_var_selects_config(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"camera_id": "x", "work_dir": "w"}))
    monkeypatch.setenv("SYNTH_CONFIG", str(path))
    assert load_config()["camera_id"] == "x"
