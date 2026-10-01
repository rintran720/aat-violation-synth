"""Stage B3: per-camera height correction from forklifts of known overhead-guard height.

Run: SYNTH_CONFIG=config/cameras/ch14.json python -m synth.height_check [--frames 8]
For several library frames with a sharp, uncut forklift, MoGe-2 gives a per-frame point map; its scale is fitted
so the visible floor matches this camera's calibrated floor plane (work/cameras/<ch>/camera.json). The forklift's
top (99.5th percentile of height above the floor inside its SAM3 mask) divided by the catalogue overhead-guard
height is that frame's ratio; z_scale is the median. z_scale < 1 means calibrated heights come out too low
(cam01 earlier: ~0.85), so rendered objects and the fitting proxy should be scaled by it.
Writes <work>/height_check.json.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config, write_json

FORKLIFT_MODEL = "sumitomo_quapro_2t5_dual"


def floor_depth(cam):
    """Camera-forward distance to the calibrated floor for every pixel (NaN where the ray misses it)."""
    W, H, K, M = cam["width"], cam["height"], cam["K_norm"], np.array(cam["matrix_world"])
    yy, xx = np.mgrid[0:H, 0:W]
    x = ((xx + .5) / W - K[0][2]) / K[0][0]; y = -((yy + .5) / H - K[1][2]) / K[1][1]
    r = np.stack([x, y, -np.ones_like(x)], -1) @ M[:3, :3].T
    t = -M[2, 3] / r[..., 2]
    return np.where(t > 0, t, np.nan)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--library", default="work/library/index.json")
    args = parser.parse_args()
    import torch
    from moge.model.v2 import MoGeModel
    from synth.segment import load_processor
    cfg = load_config(); work = cfg["work"]; cam_id = cfg["camera_id"]
    roof = json.loads(Path("config/standards.json").read_text())["forklift"][FORKLIFT_MODEL]["overhead_guard_m"]
    cam = json.loads((work / "camera.json").read_text()); M = np.array(cam["matrix_world"])
    floor_mask = np.asarray(Image.open(work / "masks" / Path(cfg["background_image"]).stem / "floor.png").convert("L")) > 127
    fz = floor_depth(cam)
    crops = sorted((r for r in json.loads(Path(args.library).read_text())["crops"]
                    if r["camera"] == cam_id and r["class"] == "forklift" and not r["touches_border"] and r["area_px"] > 6000),
                   key=lambda r: -r["quality"])
    picked, times = [], []
    for r in crops:  # distinct moments, best first
        if all(abs(r["t_s"] - t) >= 6 for t in times):
            picked.append(r); times.append(r["t_s"])
        if len(picked) == args.frames:
            break
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moge = MoGeModel.from_pretrained(cfg["moge_model"]).to(device).eval()
    torch.autocast(device, dtype=torch.bfloat16).__enter__()
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
        boxes = [[*np.argwhere(m)[:, ::-1].min(0), *(np.argwhere(m)[:, ::-1].max(0) + 1)] for m in masks if m.any()]
        if not boxes:
            continue
        k = int(np.argmax([box_iou(b, r["box"]) for b in boxes]))
        if box_iou(boxes[k], r["box"]) < .5:
            continue
        mask = masks[k]
        rgb = torch.tensor(np.asarray(image, np.float32) / 255., device=device).permute(2, 0, 1)
        with torch.inference_mode(), torch.autocast(device, enabled=False):  # SAM3 needs bf16, MoGe does not
            pts = moge.infer(rgb, resolution_level=9, use_fp16=device == "cuda")["points"].float().cpu().numpy()
        objects = np.zeros_like(mask)
        for m in masks:
            objects |= m
        ok = floor_mask & ~objects & np.isfinite(fz) & (fz < 40) & np.isfinite(pts[..., 2])
        scale = float(np.median(fz[ok] / pts[..., 2][ok]))
        pc = pts * scale; pb = np.stack([pc[..., 0], -pc[..., 1], -pc[..., 2]], -1)  # OpenCV -> Blender camera
        z = (pb @ M[:3, :3].T + M[:3, 3])[..., 2]
        floor_z = float(np.median(z[ok]))
        top = float(np.percentile(z[mask], 99.5)) - floor_z
        rows.append({"t_s": r["t_s"], "box": r["box"], "moge_scale": round(scale, 3), "floor_z_m": round(floor_z, 3),
                     "top_m": round(top, 3), "ratio": round(top / roof, 3)})
        print(f"{cam_id} t={r['t_s']}s top {top:.2f} m ratio {top / roof:.3f}", flush=True)
    ratios = [x["ratio"] for x in rows]
    result = {"camera_id": cam_id, "reference": f"{FORKLIFT_MODEL} overhead guard {roof} m", "frames": rows,
              "z_scale": round(float(np.median(ratios)), 3) if ratios else None,
              "z_scale_iqr": [round(float(v), 3) for v in np.percentile(ratios, [25, 75])] if ratios else None}
    write_json(work / "height_check.json", result)
    print(json.dumps({k: result[k] for k in ("camera_id", "z_scale", "z_scale_iqr")}))


if __name__ == "__main__":
    main()
