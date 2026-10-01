"""Stage B6: one review scene per camera, to judge the size of rendered objects against real ones.

Run: SYNTH_CONFIG=config/cameras/ch14.json python -m synth.test_scene
Places a V1 row (2 LSPs in series with cargo) inside the camera's placement zone, near the camera, using a virtual
(unrendered) forklift as anchor, and writes two anchors: with this camera's scale_correction.json and without it.
Render and composite them with blender/render_anchor_poc.py and synth.composite_anchor_poc (commands printed).
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config

LOCAL = {"fork_tip_l": [-.29, 1.852, 0], "fork_tip_r": [.29, 1.852, 0], "heel_l": [-.29, .772, 0], "heel_r": [.29, .772, 0],
         "rear_l": [-.5, -1.1, 0], "rear_r": [.5, -1.1, 0]}
ROW = [(-1.0, .6), (1.0, .6), (1.0, 4.5), (-1.0, 4.5)]   # local footprint of 2 LSPs + cargo ahead of the forks
CARGO = ["cargo_3", "cargo_0"]                           # one wrapped load, one ULD


def project(cam, pts):
    Mi = np.linalg.inv(np.array(cam["matrix_world"])); K = cam["K_norm"]; out = []
    for p in pts:
        c = Mi @ np.array([*p, 1.]); out.append([(-c[0] / c[2] * K[0][0] + K[0][2]) * cam["width"], (c[1] / c[2] * K[1][1] + K[1][2]) * cam["height"]])
    return np.array(out)


def floor_xy(cam, px, py):
    K = cam["K_norm"]; M = np.array(cam["matrix_world"])
    r = M[:3, :3] @ np.array([(px / cam["width"] - K[0][2]) / K[0][0], -(py / cam["height"] - K[1][2]) / K[1][1], -1.])
    o = M[:3, 3]; return (o - o[2] / r[2] * r)[:2]


def world(pose, local):
    x, y, h = pose; c, s = math.cos(h), math.sin(h)
    return (x + c * local[0] - s * local[1], y + s * local[0] + c * local[1], local[2] if len(local) > 2 else 0.)


def main():
    cfg = load_config(); work = cfg["work"]; cam = json.loads((work / "camera.json").read_text())
    zone = np.asarray(Image.open(work / "placement_zone.png")) > 127
    ys, xs = np.nonzero(zone); best = None
    # candidate anchors: zone pixels in its nearer half, centre-ish columns; headings every 15 degrees
    for q in (.75, .65, .55, .45):
        y = int(np.quantile(ys, q)); row = xs[ys == y]
        for x in (int(np.median(row)), int(np.quantile(row, .35)), int(np.quantile(row, .65))):
            base_xy = floor_xy(cam, x, y)
            for hdeg in range(0, 360, 15):
                # put the row's centre near the picked pixel
                h = math.radians(hdeg); c = world((0, 0, h), (0, 2.55)); pose = (base_xy[0] - c[0], base_xy[1] - c[1], h)
                corners = project(cam, [world(pose, (u, v)) for u, v in ROW])
                inside = all(0 <= u < cam["width"] and 0 <= v < cam["height"] and zone[int(v), int(u)] for u, v in corners)
                if inside:
                    across = abs(math.sin(h - math.atan2(base_xy[0], base_xy[1])))  # prefer rows seen side-on
                    if best is None or across > best[0]:
                        best = (across, pose)
            if best and best[0] > .9:
                break
        if best and best[0] > .9:
            break
    if best is None:
        raise SystemExit(f"{cfg['camera_id']}: no placement of a V1 row fits the zone")
    pose = best[1]; out = []
    for tag, sc in [("scaled", None), ("unscaled", {"xy": 1., "z": 1.})]:
        a = {"background_image": cfg["background_image"], "source_calibration": f"{cfg['work_dir']}/camera.json",
             "source_floor_points": f"{cfg['work_dir']}/points_world.npy",
             "layout": {"scenario": "V1", "lsp_count": 2, "include_idle_skid": False, "cargo_assets": CARGO},
             "render": {"samples": 48, "seed": 7},
             "foreground_forklift": {"landmarks": [{"name": k, "local": l, "pixel": project(cam, [world(pose, l)])[0].round(2).tolist()}
                                                   for k, l in LOCAL.items()], "initial_pose": list(pose), "max_landmark_rmse_px": 2.0},
             "note": "B6 review scene: virtual forklift (not rendered) on the clean plate"}
        if sc:
            a["scale_correction"] = sc
        path = work / f"anchor_test_{tag}.json"; path.write_text(json.dumps(a, indent=1)); out.append(path)
    cfgp = f"config/cameras/{cfg['camera_id']}.json"
    for p in out:
        stem = f"test_{cfg['camera_id']}_{p.stem.split('_')[-1]}"
        print(f"blender -b --python-exit-code 1 --python blender/render_anchor_poc.py -- {cfgp} --anchor {p.as_posix()} --output-stem {stem}")
        print(f"python -m synth.composite_anchor_poc --config {cfgp} --anchor {p.as_posix()} --output-stem {stem}")


if __name__ == "__main__":
    main()
