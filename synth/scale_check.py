"""Stage B3: per-camera floor and height scale from forklifts of known catalogue size (silhouette fit, no MoGe).

Run: SYNTH_CONFIG=config/cameras/ch14.json python -m synth.scale_check [--frames 8] [--min-area 12000]
For large, uncut forklifts in the stage A1 library, fit the proxy pose at nominal size, then rescale the proxy
(xy_scale on length/width, z_scale on height) and keep the scale pair whose best pose matches the SAM3 mask best.
xy_scale = 0.95 means the calibrated floor reads real lengths 5 % short; z_scale likewise for heights. Medians over
frames go to <work>/scale_check.json. MoGe is not used: it flattened distant trucks into the floor (see history).
"""
import argparse
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config, write_json
from synth.forklift_proxy import build, load_spec

MODEL = "sumitomo_quapro_2t5_dual"
XY_GRID = np.round(np.arange(.80, 1.201, .05), 3)
Z_GRID = np.round(np.arange(.45, 1.101, .05), 3)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=12000, help="px; small trucks constrain scale poorly")
    parser.add_argument("--library", default="work/library/index.json")
    args = parser.parse_args()
    import torch
    from synth.pose_fit import Camera, MaskFit
    from synth.segment import load_processor
    cfg = load_config(); work = cfg["work"]; cam_id = cfg["camera_id"]
    camera = Camera(json.loads((work / "camera.json").read_text()))
    P0, L0 = build(load_spec(MODEL))
    P0 = torch.tensor(P0, device="cuda"); L = torch.tensor(L0.astype(np.int64), device="cuda")
    crops = sorted((r for r in json.loads(Path(args.library).read_text())["crops"]
                    if r["camera"] == cam_id and r["class"] == "forklift" and not r["touches_border"]
                    and r["area_px"] >= args.min_area), key=lambda r: -r["area_px"])
    picked, times = [], []
    for r in crops:
        if all(abs(r["t_s"] - t) >= 6 for t in times):
            picked.append(r); times.append(r["t_s"])
        if len(picked) == args.frames:
            break
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    sam = load_processor(); sam.confidence_threshold = .4
    rows = []
    for r in picked:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "f.jpg"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(r["t_s"]), "-i", cfg["source_video"],
                            "-frames:v", "1", "-q:v", "2", str(frame)], check=True)
            image = Image.open(frame).convert("RGB")
        out = sam.set_text_prompt(state=sam.set_image(image), prompt="forklift")
        masks = [np.asarray(m.detach().cpu().float()).squeeze() > .5 for m in out["masks"]]
        masks = [m for m in masks if m.sum() > 500]
        if not masks:
            continue
        boxes = [[*np.argwhere(m)[:, ::-1].min(0), *(np.argwhere(m)[:, ::-1].max(0) + 1)] for m in masks]
        k = int(np.argmax([box_iou(b, r["box"]) for b in boxes]))
        if box_iou(boxes[k], r["box"]) < .5:
            continue
        fit = MaskFit(camera, masks[k])
        if fit.touches_border:
            continue
        with torch.autocast("cuda", enabled=False), torch.no_grad():
            iou0, pose0 = fit.fit(P0, L)
            best = (iou0, 1., 1., pose0)
            for sxy, sz in itertools.product(XY_GRID, Z_GRID):
                P = P0 * torch.tensor([sxy, sxy, sz], device="cuda", dtype=torch.float32)
                iou, pose = fit.fit(P, L, start=pose0, coarse=False)
                if iou > best[0]:
                    best = (iou, float(sxy), float(sz), pose)
        rows.append({"t_s": r["t_s"], "area_px": r["area_px"], "iou_nominal": round(iou0, 3), "iou_best": round(best[0], 3),
                     "xy_scale": best[1], "z_scale": best[2], "pose": [round(v, 4) for v in best[3]]})
        print(f"{cam_id} t={r['t_s']}s area {r['area_px']}: iou {iou0:.3f} -> {best[0]:.3f} at xy {best[1]} z {best[2]}", flush=True)
    summary = {"camera_id": cam_id, "reference": MODEL, "frames": rows}
    for key in ("xy_scale", "z_scale"):
        vals = [x[key] for x in rows]
        summary[key] = round(float(np.median(vals)), 3) if vals else None
        summary[key + "_iqr"] = [round(float(v), 3) for v in np.percentile(vals, [25, 75])] if vals else None
    write_json(work / "scale_check.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}))


if __name__ == "__main__":
    main()
