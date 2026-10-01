"""Stage B3: refine a camera from objects of known size in the stage A1 library (no site measurements needed).

Run: SYNTH_CONFIG=config/cameras/ch14.json python -m synth.refine_camera
Unknowns: focal length, pitch, roll and mounting height (principal point fixed at the image centre).
Constraints, solved together with a robust least-squares:
  - flat LSP sheets: floor footprint of the mask outline should be lsp.size_m (config/standards.json);
  - standing people: foot-to-head height should be about PERSON_HEIGHT_M (loose prior, many samples).
The forklift is left out on purpose so it can validate the result (synth.scale_check --camera camera_refined.json).
Writes <work>/camera_refined.json and <work>/refine_camera.json (per-constraint residuals before/after).
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from synth.common import load_config, write_json

PERSON_HEIGHT_M = 1.70     # adult worker incl. shoes/helmet; per-person spread is absorbed by the robust loss
PERSON_REL_SIGMA = .07
LSP_REL_SIGMA = .04
MIN_RECTANGULARITY = .85   # mask area / fitted rectangle area on the floor: rejects stacks, fragments, floor marks


def to_params(cam):
    """camera.json -> (f_px, pitch, roll, height, yaw) with pitch/roll from the camera axes in the floor frame."""
    M = np.array(cam["matrix_world"]); R = M[:3, :3]
    fwd = -R[:, 2]; right = R[:, 0]
    pitch = math.asin(-fwd[2]); yaw = math.atan2(fwd[0], fwd[1])
    roll = math.asin(np.clip(right[2] / math.cos(pitch), -1, 1))
    return cam["K_norm"][0][0] * cam["width"], pitch, roll, M[2, 3], yaw


def rotation(pitch, roll, yaw):
    """Camera-to-world rotation (Blender camera axes) for a camera looking down by `pitch`, rolled, turned by `yaw`."""
    fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), -math.sin(pitch)])
    right0 = np.array([math.cos(yaw), -math.sin(yaw), 0.]); up0 = np.cross(right0, fwd)
    right = math.cos(roll) * right0 + math.sin(roll) * up0; up = np.cross(right, fwd)
    return np.stack([right, up, -fwd], 1)


class Cam:
    def __init__(self, f, pitch, roll, h, yaw, W, H):
        self.f, self.W, self.H = f, W, H; self.R = rotation(pitch, roll, yaw); self.o = np.array([0., 0., h])

    def rays(self, px):
        px = np.asarray(px, float)
        d = np.stack([(px[:, 0] - self.W / 2) / self.f, -(px[:, 1] - self.H / 2) / self.f, -np.ones(len(px))], 1)
        return d @ self.R.T

    def plane(self, px, z=0.):
        r = self.rays(px); t = (z - self.o[2]) / r[:, 2]
        return self.o + t[:, None] * r, t > 0

    def project(self, p):
        c = (np.asarray(p) - self.o) @ self.R
        return np.stack([self.W / 2 - c[:, 0] / c[:, 2] * self.f, self.H / 2 + c[:, 1] / c[:, 2] * self.f], 1)

    def height(self, foot, head):
        """Height of a vertical segment standing on the floor at `foot` whose top projects nearest to `head`."""
        P, ok = self.plane([foot])
        if not ok[0]:
            return np.nan
        # closest point between the vertical line P + z*e3 and the head's viewing ray o + t*d (continuous in params)
        d = self.rays([head])[0]; e = np.array([0., 0., 1.]); w = P[0] - self.o
        a, b, c = e @ e, e @ d, d @ d; dd, ee = e @ w, d @ w
        den = a * c - b * b
        return float((b * ee - c * dd) / den) if abs(den) > 1e-9 else np.nan


def lsp_outline(rec):
    a = np.asarray(Image.open(rec["crop"]))[..., 3] > 127
    cs, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float) + rec["box"][:2]
    return c[::max(1, len(c) // 120)], a.sum()


def lsp_dims(cam, outline, thickness):
    P, ok = cam.plane(outline, thickness)
    if not ok.all():
        return None
    xy = P[:, :2].astype(np.float32); (_, (a, b), _) = cv2.minAreaRect(xy)
    rect_area = a * b; poly_area = cv2.contourArea(xy)
    return sorted([a, b], reverse=True), (poly_area / rect_area if rect_area else 0)


def person_points(rec):
    a = np.asarray(Image.open(rec["crop"]))[..., 3] > 127
    ys, xs = np.nonzero(a); x0, y0 = rec["box"][:2]
    foot = [x0 + xs[ys >= ys.max() - 2].mean(), y0 + ys.max()]
    head = [x0 + xs[ys <= ys.min() + 2].mean(), y0 + ys.min()]
    return foot, head, (ys.max() - ys.min()) / max(1, xs.max() - xs.min())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", default="work/library/index.json")
    args = parser.parse_args()
    cfg = load_config(); work = cfg["work"]; cam_id = cfg["camera_id"]
    cam0 = json.loads((work / "camera.json").read_text()); W, H = cam0["width"], cam0["height"]
    lsp = json.loads(Path("config/standards.json").read_text())["lsp"]; L, S = sorted(lsp["size_m"], reverse=True)
    f0, pitch0, roll0, h0, yaw0 = to_params(cam0)
    crops = [r for r in json.loads(Path(args.library).read_text())["crops"] if r["camera"] == cam_id and not r["touches_border"]]
    base = Cam(f0, pitch0, roll0, h0, yaw0, W, H)
    sheets = []
    for r in crops:
        if r["class"] != "lsp":
            continue
        outline, _ = lsp_outline(r); d = lsp_dims(base, outline, lsp["thickness_m"])
        if d and d[1] >= MIN_RECTANGULARITY and .6 < d[0][1] / d[0][0] <= 1:
            sheets.append(outline)
    people = []
    for r in crops:
        if r["class"] == "person" and r["area_px"] >= 1500:
            foot, head, aspect = person_points(r)
            if aspect >= 2.2:  # standing and not cut by another object
                people.append((foot, head))

    def residuals(x):
        f, pitch, roll, h = f0 * x[0], pitch0 + x[1], roll0 + x[2], h0 * x[3]
        cam = Cam(f, pitch, roll, h, yaw0, W, H); res = []
        for o in sheets:
            d = lsp_dims(cam, o, lsp["thickness_m"])
            res += [math.log(d[0][0] / L) / LSP_REL_SIGMA, math.log(d[0][1] / S) / LSP_REL_SIGMA] if d else [10., 10.]
        for foot, head in people:
            z = cam.height(foot, head)
            res.append(math.log(z / PERSON_HEIGHT_M) / PERSON_REL_SIGMA if np.isfinite(z) else 10.)
        return np.array(res)

    if len(sheets) < 2 and len(people) < 5:
        raise SystemExit(f"{cam_id}: too few references ({len(sheets)} LSP sheets, {len(people)} people)")
    x0 = np.array([1., 0., 0., 1.])
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=1.5, x_scale=[.05, .02, .02, .05], diff_step=1e-3,
                        bounds=([.6, -.35, -.35, .5], [1.6, .35, .35, 2.]))
    f, pitch, roll, h = f0 * sol.x[0], pitch0 + sol.x[1], roll0 + sol.x[2], h0 * sol.x[3]
    cam = Cam(f, pitch, roll, h, yaw0, W, H)
    M = np.eye(4); M[:3, :3] = cam.R; M[:3, 3] = cam.o
    refined = {**cam0, "lens_mm": f / W * cam0["sensor_width_mm"], "hfov_deg": math.degrees(2 * math.atan(W / 2 / f)),
               "matrix_world": M.tolist(), "K_norm": [[f / W, 0, .5], [0, f / H, .5], [0, 0, 1]],
               "refined_from": "camera.json via synth.refine_camera"}
    write_json(work / "camera_refined.json", refined)

    def summary(c):
        dims = [lsp_dims(c, o, lsp["thickness_m"])[0] for o in sheets]
        hs = [c.height(ft, hd) for ft, hd in people]
        return {"lsp_median_m": np.median(dims, 0).round(2).tolist() if dims else None,
                "person_median_m": round(float(np.nanmedian(hs)), 2) if hs else None}
    report = {"camera_id": cam_id, "lsp_sheets_used": len(sheets), "people_used": len(people),
              "before": {"f_px": round(f0, 1), "pitch_deg": round(math.degrees(pitch0), 2), "roll_deg": round(math.degrees(roll0), 2),
                         "height_m": round(h0, 3), **summary(base)},
              "after": {"f_px": round(f, 1), "pitch_deg": round(math.degrees(pitch), 2), "roll_deg": round(math.degrees(roll), 2),
                        "height_m": round(h, 3), "hfov_deg": round(refined["hfov_deg"], 1), **summary(cam)}}
    write_json(work / "refine_camera.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
