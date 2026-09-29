"""Per-camera calibration: MoGe-2 on the clean background -> Blender camera + floor-aligned world frame,
cross-checked with the sizes and positions of the SAM3 objects of the same camera. Done once per camera and
reused by every image of that camera.

Run: python -m synth.calibrate
Writes work/camera.json, work/points_world.npy, work/scene_facts.json, work/calibration.json
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config, reference_images, write_json

CV_TO_BLENDER_CAM = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV cam (x right, y down, z fwd) -> Blender cam (-z fwd, y up)


def blender_camera(k_norm, width, height, sensor_width=36.0):
    """Normalized intrinsics (fx/W, fy/H, cx/W, cy/H) -> Blender lens/shift with sensor_fit HORIZONTAL."""
    assert width >= height, "shift_y formula assumes a landscape frame"
    fx, fy, cx, cy = k_norm[0][0], k_norm[1][1], k_norm[0][2], k_norm[1][2]
    if abs(fx * width - fy * height) > 0.02 * fx * width:
        print(f"WARNING: non-square pixels fx={fx * width:.1f}px fy={fy * height:.1f}px, using fx")
    return {
        "width": width,
        "height": height,
        "lens_mm": fx * sensor_width,
        "sensor_width_mm": sensor_width,
        "shift_x": 0.5 - cx,
        "shift_y": (cy - 0.5) * height / width,
        "hfov_deg": math.degrees(2 * math.atan(0.5 / fx)),
    }


def fit_plane_ransac(points, iters=300, thresh=0.03, seed=0):
    """Return (n, d, inlier_ratio) with n.p + d = 0, |n| = 1, oriented so the camera (origin) has d > 0."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        inliers = np.abs(points @ n - n @ a) < thresh
        if best is None or inliers.sum() > best.sum():
            best = inliers
    p = points[best]
    centroid = p.mean(axis=0)
    n = np.linalg.svd(p - centroid)[2][-1]  # least-squares refine on inliers
    d = -n @ centroid
    if d < 0:
        n, d = -n, -d
    return n, float(d), float(best.mean())


def world_from_cv(n, d):
    """4x4 transform: OpenCV camera coords -> world (z up, floor z=0, origin below camera, +y = camera forward on floor)."""
    z = n
    x = np.array([1.0, 0.0, 0.0]) - n * n[0]  # camera right projected on the floor
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])  # rows = world axes expressed in camera coords
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = r @ (d * n)  # world = R (p - o), o = -d n
    return t


def footprint_extent(xy):
    """[long, short] sides in metres of the minimum-area rectangle around a flat object's floor footprint (xy: Nx2)."""
    best = None
    for a in np.radians(np.arange(0.0, 90.0, 0.5)):
        e = np.ptp(xy @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), axis=0)
        if best is None or e[0] * e[1] < best[0] * best[1]:
            best = e
    return sorted(best.tolist(), reverse=True)


def size_ratio(extents, known, tol=0.1):
    """(known / measured size as the median over instances, number of instances used).
    Only instances whose aspect ratio matches the known one count: a sheet partly covered by cargo looks too short."""
    k_long, k_short = sorted(known, reverse=True)
    used = [e for e in extents if abs(e[1] / e[0] - k_short / k_long) < tol]
    if not used:
        return None, 0
    return float(np.median([(k_long + k_short) / (e[0] + e[1]) for e in used])), len(used)


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def run(cfg):
    import torch
    from moge.model.v2 import MoGeModel

    work = cfg["work"]
    img = Image.open(cfg["background_image"]).convert("RGB")
    w, h = img.size
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(cfg["moge_model"]).to(device).eval()

    def moge(rgb):
        with torch.no_grad():
            return model.infer(torch.tensor(rgb, device=device).permute(2, 0, 1))

    out = moge(rgb)
    pts = out["points"].cpu().numpy() * cfg["scale_correction"]
    valid = out["mask"].cpu().numpy() > 0
    k_norm = out["intrinsics"].cpu().numpy()
    assert pts.shape[:2] == (h, w), f"MoGe output {pts.shape} != image {(h, w)}"

    floor_mask = work / "masks" / Path(cfg["background_image"]).stem / "floor.png"
    floor = np.asarray(Image.open(floor_mask).convert("L")) > 127
    sel = pts[valid & floor & np.isfinite(pts).all(axis=-1)]
    n, d, inlier_ratio = fit_plane_ransac(sel[:: max(1, len(sel) // 50000)])
    t = world_from_cv(n, d)

    cam = blender_camera(k_norm.tolist(), w, h)
    cam["matrix_world"] = (t @ CV_TO_BLENDER_CAM).tolist()
    cam["K_norm"] = k_norm.tolist()
    write_json(work / "camera.json", cam)

    small = pts[::4, ::4]
    world = small @ t[:3, :3].T + t[:3, 3]
    world[~valid[::4, ::4]] = np.nan
    np.save(work / "points_world.npy", world.astype(np.float32))

    fwd = np.array(cam["matrix_world"])[:3, 2] * -1  # Blender camera looks down -Z
    write_json(work / "scene_facts.json", {
        "image_size": [w, h],
        "camera_height_m": d,
        "camera_pitch_down_deg": math.degrees(math.asin(-fwd[2])),
        "hfov_deg": cam["hfov_deg"],
        "floor_inlier_ratio": inlier_ratio,
        "ambient_rgb_linear": srgb_to_linear(rgb.reshape(-1, 3).mean(axis=0)).tolist(),
        "points_world": "work/points_world.npy shape (H//4, W//4, 3); world xyz of pixel (u, v) = P[v//4, u//4]; NaN = invalid",
    })
    print(f"camera height {d:.2f} m, hfov {cam['hfov_deg']:.1f} deg, floor inliers {inlier_ratio:.0%}")

    # cross-check with this camera's SAM3 objects: the view is fixed, so an object's floor pixels in an object
    # frame map onto the clean background's floor points
    world_full = pts @ t[:3, :3].T + t[:3, 3]
    world_full[~valid] = np.nan
    lsps = []
    for mpath in [m for f in reference_images(cfg) for m in sorted((work / "masks" / f.stem).glob("LSP_[0-9]*.png"))]:
        xy = world_full[np.asarray(Image.open(mpath).convert("L")) > 127][:, :2]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) >= 50:
            lsps.append({"mask": mpath.as_posix(), "extent_m": footprint_extent(xy), "centre_m": xy.mean(axis=0).round(2).tolist()})
    # LSP size from SAM3 key LSP_top (flat tops: single sheets and stack tops): measured in 3D from MoGe on their own
    # reference image (the sheet is not on the background), then converted to background units via the camera height
    tops = []
    for f in reference_images(cfg):
        top_masks = sorted((work / "masks" / f.stem).glob("LSP_top_*.png"))
        if not top_masks:
            continue
        o = moge(np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0)
        p_f, valid_f = o["points"].cpu().numpy(), o["mask"].cpu().numpy() > 0
        objects = np.zeros_like(floor)
        for m in (work / "masks" / f.stem).glob("*.png"):
            if "_" not in m.stem:  # class unions only: forklift, LSP, SKID, cargo
                objects |= np.asarray(Image.open(m).convert("L")) > 127
        sel_f = p_f[valid_f & floor & ~objects & np.isfinite(p_f).all(axis=-1)]
        n_f, d_f, _ = fit_plane_ransac(sel_f[:: max(1, len(sel_f) // 50000)])
        t_f = world_from_cv(n_f, d_f)
        to_bg = d / d_f  # same physical camera height in both reconstructions
        w_f = (p_f @ t_f[:3, :3].T + t_f[:3, 3]) * to_bg
        for mpath in top_masks:
            m = np.asarray(Image.open(mpath).convert("L")) > 127
            if m.sum() < 0.002 * m.size or m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
                continue  # too small (far away) or cut by the image border: not fully visible
            pts_top = w_f[m & valid_f]
            pts_top = pts_top[np.isfinite(pts_top).all(axis=1)]
            if len(pts_top) < 50:
                continue
            z = float(np.median(pts_top[:, 2]))
            flat = pts_top[np.abs(pts_top[:, 2] - z) < 0.05]  # the top plane only, not the stack's sides
            extent = footprint_extent(flat[:, :2])
            cells = np.unique(np.floor(flat[:, :2] / 0.1), axis=0)  # 10 cm grid: a sheet partly covered by cargo has holes
            tops.append({"mask": mpath.as_posix(), "extent_m": extent, "height_m": round(z, 3),
                         "centre_m": flat[:, :2].mean(axis=0).round(2).tolist(), "flat_fraction": round(len(flat) / len(pts_top), 2),
                         "fill": round(min(1.0, len(cells) * 0.01 / (extent[0] * extent[1])), 2)})
        print(f"{f.stem}: {len(top_masks)} LSP top sheets, floor fit camera height {d_f:.2f} (background {d:.2f})")
    # usable = a fully visible single sheet on the floor: flat, filling its rectangle (not covered by cargo), at floor
    # level (stack tops measure consistently larger than floor sheets, cargo and ULD tops are higher)
    usable = [o for o in tops if o["flat_fraction"] >= 0.85 and o["fill"] >= 0.85 and o["height_m"] < 0.15]
    measured = np.median([o["extent_m"] for o in usable], axis=0).round(2).tolist() if usable else None
    known = cfg["lsp_size_m"]  # real size if supplied; otherwise MoGe-2's metric scale is trusted as is
    ratio, used = size_ratio([o["extent_m"] for o in usable], known) if known else (None, len(usable))
    positions = {}
    for rec in json.loads((work / "refs" / "index.json").read_text()):
        u, v = rec["bottom_center_px"]
        p = world_full[v, u]
        if np.isfinite(p).all():
            positions.setdefault(rec["class"], []).append(p[:2].round(2).tolist())
    write_json(work / "calibration.json", {
        "camera_id": cfg["camera_id"],
        "background_image": cfg["background_image"],
        "lsp_known_size_m": known,
        "lsp_measured_size_m": measured,
        "lsp_instances": lsps,
        "lsp_top_instances": tops,
        "lsp_instances_used": used,
        "suggested_scale_correction": ratio and round(cfg["scale_correction"] * ratio, 3),
        "object_floor_positions_m": positions,
    })
    print(f"LSP size (MoGe-2): {measured} m from {len(usable)} fully visible top sheets ({len(tops)} top masks)")
    if measured is None:
        print("WARNING: no fully visible LSP top sheet; add a reference image with one (plan Task 3)")
    if known:
        print(f"known/measured = {ratio}")
        if ratio is None or abs(ratio - 1) > 0.1:
            print("WARNING: set scale_correction in config.json to suggested_scale_correction "
                  "(work/calibration.json) and re-run, or check the LSP masks")


if __name__ == "__main__":
    run(load_config())
