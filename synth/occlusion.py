"""Stage C4: which rendered pixels the real scene hides, by comparing depths along each render pixel's ray.

Real depth: MoGe-2 on the frame, scaled so its free floor matches the calibrated floor plane; inside the fitted
truck's silhouette the proxy depth replaces it (MoGe smears thin masts and guards into the background).
Load depth: ray against the oriented bounding boxes of the rendered objects (render sidecar), and for pixels
that only carry a shadow, the floor itself. A render pixel is hidden when the real scene is MARGIN_M nearer.
The compositor warps the render by a radial lens term after occlusion, so masks live in render coordinates and
photo-space quantities are sampled through that same warp.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, minimum_filter

MARGIN_M = .15


def display_coords(W, H, k1):
    """Photo pixel where each render pixel lands after composite_violation.radial_warp (inverse of its sampling)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    sx, sy = (xx - (W - 1) / 2) / ((W - 1) / 2), (yy - (H - 1) / 2) / ((H - 1) / 2)
    dx, dy = sx.copy(), sy.copy()
    for _ in range(6):  # d = s * (1 + k1 |d|^2)
        f = 1 + k1 * (dx * dx + dy * dy); dx, dy = sx * f, sy * f
    return (W - 1) / 2 + dx * (W - 1) / 2, (H - 1) / 2 + dy * (H - 1) / 2


def sample(img, px, py):
    h, w = img.shape[:2]
    return img[np.clip(np.round(py).astype(int), 0, h - 1), np.clip(np.round(px).astype(int), 0, w - 1)]


def pixel_rays(cam):
    """World ray directions with camera-forward length 1 for every render pixel (pinhole)."""
    W, H, K, M = cam["width"], cam["height"], cam["K_norm"], np.array(cam["matrix_world"])
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.stack([((xx + .5) / W - K[0][2]) / K[0][0], -((yy + .5) / H - K[1][2]) / K[1][1], -np.ones((H, W))], -1)
    return d @ M[:3, :3].T, M[:3, 3]


def load_depth(cam, sidecar, assets_dir, mask):
    rays, origin = pixel_rays(cam); depth = np.full(mask.shape, np.inf)
    yaw = math.radians(sidecar["pose"]["heading_deg"]); c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    d = rays[mask]
    for ob in sidecar["objects"]:
        a = json.loads((Path(assets_dir) / f"{ob['asset']}.json").read_text())
        k = np.array(ob.get("scale_correction_xyz", [1, 1, 1]))
        lo, hi = np.array(a["bounds_min"]) * k, np.array(a["bounds_max"]) * k
        o = (origin - np.array(ob["location_world_m"])) @ R; dd = d @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / dd, (hi - o) / dd
        tn = np.nanmax(np.minimum(t1, t2), 1); tf = np.nanmin(np.maximum(t1, t2), 1)
        hit = (tn <= tf) & (tf > 0)
        z = depth[mask]; z[hit] = np.minimum(z[hit], np.maximum(tn[hit], 0)); depth[mask] = z
    floor_t = -origin[2] / rays[..., 2]
    return depth, np.where(floor_t > 0, floor_t, np.inf)


def truck_depth(cam, P, pose, shape):
    """Camera-forward depth of the fitted proxy, z-buffered and min-filtered to close gaps between points."""
    W, H, K, Mi = cam["width"], cam["height"], cam["K_norm"], np.linalg.inv(np.array(cam["matrix_world"]))
    x, y, h = pose; c, s = math.cos(h), math.sin(h)
    wp = np.stack([x + c * P[:, 0] - s * P[:, 1], y + s * P[:, 0] + c * P[:, 1], P[:, 2], np.ones(len(P))], -1) @ Mi.T
    z = -wp[:, 2]; u = ((-wp[:, 0] / wp[:, 2]) * K[0][0] + K[0][2]) * W; v = ((wp[:, 1] / wp[:, 2]) * K[1][1] + K[1][2]) * H
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = np.full(shape, np.inf); np.minimum.at(out, (v[ok].astype(int), u[ok].astype(int)), z[ok])
    return minimum_filter(out, 5)


def scene_depth(points, cam, floor_mask, object_mask, floor_t_photo):
    """MoGe point map (OpenCV camera frame) -> camera-forward depth in metres, scaled on the free floor."""
    z = points[..., 2]; ok = floor_mask & ~object_mask & np.isfinite(floor_t_photo) & (floor_t_photo < 40) & np.isfinite(z) & (z > 0)
    scale = float(np.median(floor_t_photo[ok] / z[ok])) if ok.sum() > 500 else 1.
    return z * scale, scale


def occluders(cam, k1, alpha, sidecar, assets_dir, points, floor_mask, object_mask, truck_mask, P, pose):
    """Boolean mask (render coordinates) of rendered pixels hidden by the real scene, plus diagnostics."""
    H, W = alpha.shape
    rays, origin = pixel_rays(cam); floor_t = -origin[2] / rays[..., 2]; floor_t = np.where(floor_t > 0, floor_t, np.inf)
    real_photo, scale = scene_depth(points, cam, floor_mask, object_mask, floor_t)
    dx, dy = display_coords(W, H, k1)
    real = sample(real_photo, dx, dy)
    truck = binary_dilation(sample(truck_mask, dx, dy), iterations=3)
    tz = truck_depth(cam, P, pose, (H, W))
    real = np.where(truck & np.isfinite(tz), tz, real)
    load, floor = load_depth(cam, sidecar, assets_dir, alpha)
    target = np.where(np.isfinite(load), load, floor)  # shadow-only pixels lie on the floor
    occ = alpha & np.isfinite(target) & (real < target - MARGIN_M)
    occ = binary_opening(occ, iterations=1)
    return occ, {"moge_floor_scale": round(scale, 3), "occluded_px": int(occ.sum()), "render_px": int(alpha.sum())}
