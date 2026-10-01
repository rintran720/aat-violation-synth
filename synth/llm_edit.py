"""Branch C': turn a real valid frame (forklift pushing cargo on exactly 1 LSP) into a violation by asking an
image-edit model (Qwen-Image-Edit-2509 in a local ComfyUI) to add a second LSP, then verify with code.

Run: SYNTH_CONFIG=config/cameras/ch10.json python -m synth.llm_edit --frame f_00664.jpg [--rounds 3] [--seed 7]
     [--backend comfy|codex] [--codex-model NAME]
--backend codex replaces step 2 by the Codex CLI (`codex exec`, logged in beforehand; CODEX_BIN overrides the
executable, CODEX_EXTRA_ARGS adds flags): the same window and reference sheets go in as attached images and the
agent must write the edited window to a PNG of the same size. Steps 1, 3 and 4 are unchanged.
1. Edit region: a sheet rectangle is fitted to the real LSP under the load (SAM3 "black platform" next to the
   tracked truck; cargo and people may hide it, the truck may not). The new LSP is that rectangle moved one length
   further along its axis, away from the truck, and a real LSP texture is pasted there as the starting point.
2. ComfyUI API: Qwen-Image-Edit GGUF + Lightning LoRA, TextEncodeQwenImageEditPlus with the window as picture 1
   and a real LSP crop from the sample library as picture 2; the model edits the whole window by instruction.
3. Check: output size = input size; outside the mask almost nothing changed; inside it SAM3 finds a new dark
   platform covering most of the polygon. Failures are written back into the prompt (and seed/mask change) for
   at most --rounds rounds.
4. Final image: original pixels everywhere except the feathered mask, so the rest of the frame is unchanged by
   construction; the check above reports how well the model behaved on its own.
Writes work/cameras/<ch>/llm_edit/<frame stem>/seed<seed>/ (rounds, masks, diffs, report.json, final.png/.jpg).
"""
import argparse
import io
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import binary_dilation

from synth.common import load_config, write_json

COMFY_DIR = Path("E:/tools/ComfyUI_windows_portable")
COMFY_PORT = 8189  # own instance: an already running ComfyUI may predate the downloaded models
COMFY_URL = f"http://127.0.0.1:{COMFY_PORT}"
UNET = "Qwen-Image-Edit-2509-Q3_K_M.gguf"   # Q4_K_M offloaded ~5 GB to RAM on a 10 GB GPU and ran out of memory
CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LORA = "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
BASE_PROMPT = ("Picture 1 is a CCTV view of a warehouse floor: a forklift pushes stretch-wrapped cargo standing on one LSP. "
               "In front of that LSP, a second LSP carrying a second cargo load has just been pasted from other photos of "
               "this warehouse; both still look pasted (hard edges, no contact shadow). "
               "Picture 2 shows real LSPs from this warehouse: single large flat dark grey plastic slip sheets lying directly "
               "on the floor, never raised and never stacked. Picture 3 shows real cargo of this warehouse "
               "(stretch-wrapped loads, cartons) and wooden SKID pallets, which are NOT LSPs. "
               "Make the pasted LSP look exactly like a real LSP from picture 2 lying flat on this floor, with the same "
               "colour, wear, edge thickness, blur, noise and lighting as the existing LSP, and make the pasted cargo look "
               "like real cargo standing on that new LSP, lit like the existing cargo, with contact shadows. "
               "Do not move, resize or rotate the new LSP or its cargo. Keep the forklift, the existing cargo and LSP, the "
               "people, the floor markings and everything else in picture 1 unchanged.")
OUTSIDE_CHANGE_MAX = .03   # fraction of pixels outside the mask (inside the window) allowed to change by > 24 levels
NEW_SHEET_MIN = .55        # fraction of the new sheet's visible (not under cargo) area a SAM3 platform must cover
NEW_CARGO_MIN = .5         # fraction of the pasted cargo a SAM3 cargo mask must cover


# ---------- geometry ----------

def floor_xy(cam, px, py, z=0.):
    """World x, y where the pixel's ray meets the horizontal plane at height z."""
    K = cam["K_norm"]; M = np.array(cam["matrix_world"])
    d = np.stack([(np.asarray(px) / cam["width"] - K[0][2]) / K[0][0], -(np.asarray(py) / cam["height"] - K[1][2]) / K[1][1],
                  -np.ones(np.shape(px))], -1) @ M[:3, :3].T
    o = M[:3, 3]; t = (z - o[2]) / d[..., 2]
    return o[:2] + t[..., None] * d[..., :2]


def project(cam, pts):
    Mi = np.linalg.inv(np.array(cam["matrix_world"])); K = cam["K_norm"]
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Mi.T
    return np.stack([(-p[:, 0] / p[:, 2] * K[0][0] + K[0][2]) * cam["width"], (p[:, 1] / p[:, 2] * K[1][1] + K[1][2]) * cam["height"]], 1)


def to_local(pose, xy):
    x, y, h = pose; c, s = math.cos(h), math.sin(h); d = xy - [x, y]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def to_world(pose, uv):
    x, y, h = pose; c, s = math.cos(h), math.sin(h)
    return [(x + c * u - s * v, y + s * u + c * v, 0.) for u, v in uv]


def sheet_quad(cx, cy, h, length, width, z=0.):
    """Floor rectangle (world corners) centred at (cx, cy); `h` is the heading of its length axis (+y local)."""
    c, s_ = math.cos(h), math.sin(h); out = []
    for u, v in [(-width / 2, -length / 2), (width / 2, -length / 2), (width / 2, length / 2), (-width / 2, length / 2)]:
        out.append((cx + c * u - s_ * v, cy + s_ * u + c * v, z))
    return out


def fit_sheet(cam, lsp_mask, occ_mask, init, length, width, z, ds=4):
    """Pose (cx, cy, h) and size factor k of a sheet (length*k x width*k) whose projection best explains the visible
    LSP pixels. Score = |R & L| / (|L| + |R - (L | O)|): pixels hidden by cargo/people (O) are not penalised, the
    truck is not in O (a pushed sheet lies in front of it, never under it). k is free because this camera's metric
    floor scale is not reliable far from the camera (stage B); the sheet's aspect ratio is kept."""
    import cv2
    H, W = lsp_mask.shape; L = lsp_mask[::ds, ::ds]; O = occ_mask[::ds, ::ds]; nL = L.sum(); free = ~(L | O)
    def score(cx, cy, h, kl, kw):
        q = project(cam, sheet_quad(cx, cy, h, length * kl, width * kw, z)) / ds
        R = np.zeros(L.shape, np.uint8); cv2.fillPoly(R, [np.round(q).astype(np.int32)], 1); R = R.astype(bool)
        return (R & L).sum() / (nL + (R & free).sum())
    best = (-1, (*init, .8, .8))
    # coarse-to-fine; kl/kw scale length and width separately (the camera's floor error is not isotropic)
    for span, step, hspan, hstep, kspan, kstep in [(.9, .15, 40, 4, .45, .15), (.25, .05, 8, 1, .12, .04),
                                                   (.06, .015, 2, .25, .04, .01), (.015, .005, .5, .1, .01, .005)]:
        cx0, cy0, h0, kl0, kw0 = best[1]
        ks = lambda k0: np.arange(max(.3, k0 - kspan), k0 + kspan + 1e-9, kstep)
        for kl in ks(kl0):
            for kw in ks(kw0):
                for cx in np.arange(cx0 - span, cx0 + span + 1e-9, step):
                    for cy in np.arange(cy0 - span, cy0 + span + 1e-9, step):
                        for dh in np.arange(-hspan, hspan + 1e-9, hstep):
                            sc = score(cx, cy, h0 + math.radians(dh), kl, kw)
                            if sc > best[0]:
                                best = (sc, (cx, cy, h0 + math.radians(dh), float(kl), float(kw)))
    return best[1], float(best[0])


def next_sheet(real_pose, truck_xy, length):
    """The neighbouring sheet one length further along the real sheet's axis, on the side away from the truck."""
    cx, cy, h = real_pose; ax = (-math.sin(h), math.cos(h))
    ahead = (cx + ax[0] * length, cy + ax[1] * length); behind = (cx - ax[0] * length, cy - ax[1] * length)
    far = ahead if math.dist(ahead, truck_xy) > math.dist(behind, truck_xy) else behind
    return (far[0], far[1], h)


def _fit_line(pts):
    c = pts.mean(0); d = np.linalg.svd(pts - c)[2][0]; return c, d


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, q - p)[0]; return p + t * d


def image_quads(cam, real_pose, new_pose, length, width, z, real_mask, block_mask, gray, band=8):
    """Top/bottom image quads of the new sheet built from the real sheet's own image edges.

    The floor-fitted rectangle only seeds this. Each real edge line is measured on the photo with OpenCV's line
    segment detector: segments within 15 px and 20 degrees of the seed edge whose midpoint lies on the real LSP
    mask boundary, fitted with length weights (the SAM3 outline alone is biased by rounded corners and the 9 cm
    side face). The mask boundary is the fallback. The shared corners are the front edge crossed with the two side
    edges; the new sheet runs along those same side lines, so its edges continue the real sheet's in perspective."""
    import cv2
    from scipy.ndimage import binary_erosion, binary_dilation as dil, distance_transform_edt
    rq = project(cam, sheet_quad(*real_pose, length, width, z)); nq = project(cam, sheet_quad(*new_pose, length, width, z))
    boundary = real_mask & ~binary_erosion(real_mask)
    edge_px = boundary & ~dil(block_mask, iterations=3)
    ys, xs = np.nonzero(edge_px); P = np.c_[xs, ys].astype(float)
    near_boundary = distance_transform_edt(~boundary) <= 6
    centre = np.argwhere(real_mask)[:, ::-1].mean(0)
    x0, y0 = max(0, int(rq[:, 0].min()) - 40), max(0, int(rq[:, 1].min()) - 40)
    x1, y1 = int(rq[:, 0].max()) + 40, int(rq[:, 1].max()) + 40
    segs = cv2.createLineSegmentDetector().detect(gray[y0:y1, x0:x1])[0]
    segs = np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4) + [x0, y0, x0, y0]
    def refit(i, j):
        a, b = rq[i], rq[j]; ln = np.linalg.norm(b - a); d = (b - a) / ln; n = np.array([-d[1], d[0]])
        pts = []
        for s_ in segs:
            p, q = s_[:2], s_[2:]; v = q - p; L = np.linalg.norm(v)
            if L < 12 or abs(v @ d) / L < math.cos(math.radians(20)):
                continue
            if max(abs((p - a) @ n), abs((q - a) @ n)) > 15 or not (-.2 < ((p + q) / 2 - a) @ d / ln < 1.2):
                continue
            mx, my = np.round((p + q) / 2).astype(int)
            if 0 <= my < near_boundary.shape[0] and 0 <= mx < near_boundary.shape[1] and near_boundary[my, mx]:
                pts.append(p + np.linspace(0, 1, max(2, int(L)))[:, None] * v)   # one point per pixel = length weight
        if pts:
            pts = np.vstack(pts)
            if len(pts) >= 20:
                line = _fit_line(pts); split = None
                # Two parallel lines ~10 px apart = top-face edge and the 9 cm side face's floor contact.
                # Keep the inner one (towards the sheet's centre): the new sheet's top face must sit at that height.
                nrm = np.array([-line[1][1], line[1][0]]); r = (pts - line[0]) @ nrm
                if np.percentile(r, 90) - np.percentile(r, 10) > 5:
                    lo, hi = np.percentile(r, 15), np.percentile(r, 85)
                    for _ in range(10):
                        g = np.abs(r - lo) < np.abs(r - hi); lo, hi = r[g].mean(), r[~g].mean()
                    inner_sign = np.sign((centre - line[0]) @ nrm)
                    keep = g if (lo - hi) * inner_sign > 0 else ~g
                    if keep.sum() >= 20 and abs(hi - lo) > 3:
                        line, split = _fit_line(pts[keep]), round(float(abs(hi - lo)), 1)
                return line, {"source": "lsd", "points": int(len(pts)), "two_lines_px_apart": split}
        t = (P - a) @ d / ln; off = np.abs((P - a) @ n); sel = P[(off <= band) & (t > -.1) & (t < 1.1)]
        if len(sel) >= 15:
            return _fit_line(sel), {"source": "mask boundary", "points": int(len(sel))}
        return (a, d), {"source": "floor fit", "points": 0}
    ahead = np.linalg.norm(nq.mean(0) - rq[[2, 3]].mean(0)) < np.linalg.norm(nq.mean(0) - rq[[0, 1]].mean(0))
    front = (2, 3) if ahead else (0, 1)                  # real edge shared with the new sheet
    far_p, far_m = (2, 3) if ahead else (1, 0)           # new sheet's far corners on the +x / -x side lines
    (lf, nf), (lp, np_), (lm, nm) = refit(*front), refit(1, 2), refit(0, 3)
    Fp, Fm = _intersect(lf, lp), _intersect(lf, lm)
    if Fp is None or Fm is None:
        raise ValueError("real sheet edges are parallel/degenerate")
    # The real sheet's length, measured on the photo: along each side line, from the front corner to the farthest
    # real-LSP pixel near that line, both put on the top-face plane. The floor fit cannot give it (cargo and truck
    # hide the back half, so its free length scale shrinks). The longer side wins: the other may be hidden.
    yy, xx = np.nonzero(real_mask); M = np.c_[xx, yy].astype(float)
    on_top = lambda q: floor_xy(cam, q[0], q[1], z)
    def far_corner(line, F):
        d = line[1] if (nq.mean(0) - F) @ line[1] < 0 else -line[1]       # into the real sheet
        n = np.array([-d[1], d[0]]); near = np.abs((M - F) @ n) <= 12; t = (M[near] - F) @ d
        return F + np.percentile(t[t > 0], 98) * d if (t > 0).sum() >= 30 else None, -d
    (Bp, dp), (Bm, dm) = far_corner(lp, Fp), far_corner(lm, Fm)
    lengths = [float(np.linalg.norm(on_top(F) - on_top(B))) for F, B in ((Fp, Bp), (Fm, Bm)) if B is not None]
    L_img = max(lengths) if lengths else length
    # Two LSPs in series look equally long: the new sheet takes the real one's image length along the side lines
    # (the calibrated floor would shorten it ~10 % here, which reads as a smaller sheet).
    px = [float(np.linalg.norm(B - F)) for F, B in ((Fp, Bp), (Fm, Bm)) if B is not None]
    L_px = max(px) if px else float(np.linalg.norm(nq[far_p] - Fp))
    Np, Nm = Fp + L_px * dp, Fm + L_px * dm
    top = np.array([Fm, Fp, Np, Nm], np.float32)
    bottom = project(cam, np.array([[*on_top(q), 0.] for q in top])).astype(np.float32)   # the 9 cm edge, down to the floor
    real_top = np.array([Bm if Bm is not None else Fm, Bp if Bp is not None else Fp, Fp, Fm], np.float32)
    F, B = max(((Fp, Bp), (Fm, Bm)), key=lambda fb: -1 if fb[1] is None else np.linalg.norm(fb[1] - fb[0]))
    down = lambda q: project(cam, np.array([[*on_top(q), 0.]]))[0]
    real_side = np.array([F, B, down(B), down(F)], np.float32)          # top-face edge, then its floor contact
    return top, bottom, {"front": nf, "side_plus": np_, "side_minus": nm,
                         "real_length_m_per_side": [round(v, 3) for v in lengths], "length_m": round(L_img, 3),
                         "real_length_px_per_side": [round(v, 1) for v in px], "length_px": round(L_px, 1),
                         "real_top_px": real_top.round(1).tolist(), "real_side_px": real_side.round(1).tolist()}


def _inner_quad(mask, dst):
    """4-corner polygon of a sheet crop's (inner) mask, ordered like dst: same winding, and the cyclic start that
    puts its long edges on dst's long edges."""
    import cv2
    cnt = max(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    hull = cv2.convexHull(cnt); q = None
    for eps in np.linspace(.01, .2, 40) * cv2.arcLength(hull, True):
        a = cv2.approxPolyDP(hull, eps, True)
        if len(a) == 4:
            q = a.reshape(4, 2).astype(np.float32); break
    if q is None:
        q = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
    if np.sign(_signed_area(q)) != np.sign(_signed_area(dst)):
        q = q[::-1].copy()
    rel = lambda p: (lambda e: e / e.sum())(np.linalg.norm(np.roll(p, -1, 0) - p, axis=1))
    k = min(range(4), key=lambda k: ((rel(np.roll(q, -k, 0)) - rel(dst)) ** 2).sum())
    return np.roll(q, -k, 0)


def _signed_area(q):
    return .5 * (np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))


def paste_sheet(frame, top, bottom, real_mask, texture_crop, real_side=None, front_overlap=3.):
    """Prefill of the new sheet: a real flat LSP crop from the library (texture_crop, RGBA) perspective-warped onto
    the top quad, its brightness matched to the visible real sheet of this frame (lighting). Each visible vertical
    face (the 9 cm edge whose floor edge lies outside the top quad) gets the real sheet's own side face from this
    frame (real_side: its top edge then floor edge), warped onto it; the real sheet's darker tone fills the rest.
    Returns the pasted frame and the sheet's image mask."""
    import cv2
    a = np.asarray(frame).astype(np.float32); H, W = a.shape[:2]
    real = a[real_mask]; edge = np.percentile(real, 30, axis=0)
    hull = cv2.convexHull(np.vstack([top, bottom])).astype(np.int32)
    side = np.zeros((H, W), np.uint8); cv2.fillPoly(side, [hull], 1)
    crop = np.asarray(Image.open(texture_crop).convert("RGBA")).astype(np.float32)
    alpha = crop[..., 3] > 127; tex = crop[..., :3]
    top = top.astype(np.float32).copy()
    for f, n in ((0, 3), (1, 2)):               # cover the real sheet's front face (now butted): front_overlap px
        top[f] += front_overlap * (top[f] - top[n]) / np.linalg.norm(top[f] - top[n])
    dst = top.copy()
    tex[alpha] = (tex[alpha] - tex[alpha].mean(0)) + real.mean(0)    # this scene's lighting, the crop's texture
    # the crop's outline is not a perfect quad and its rim is the sample sheet's own dark edge face: keep the inner
    # top face only and fill the rest from it, so neither black nor a dark rim reaches the new sheet's edges
    rim = max(4, int(.05 * min(alpha.shape)))
    inner = cv2.erode(alpha.astype(np.uint8), np.ones((2 * rim + 1, 2 * rim + 1), np.uint8)) > 0
    tex = cv2.inpaint(np.clip(tex, 0, 255).astype(np.uint8), (~inner).astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)
    src = _inner_quad(inner, dst)
    warped = cv2.warpPerspective(tex, cv2.getPerspectiveTransform(src, dst), (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    topm = np.zeros((H, W), np.uint8); cv2.fillPoly(topm, [np.round(top).astype(np.int32)], 1)
    side |= topm
    out = a.copy(); out[side > 0] = edge
    if real_side is not None:
        top_poly = np.round(top).astype(np.float32).reshape(-1, 1, 2)
        for i in range(1, 4):                                         # edge 0 (Fm-Fp) butts against the real sheet
            j = (i + 1) % 4
            face = np.array([top[i], top[j], bottom[j], bottom[i]], np.float32)
            if cv2.pointPolygonTest(top_poly, tuple(map(float, (bottom[i] + bottom[j]) / 2)), False) >= 0:
                continue                                              # hidden behind the top face
            # the side texture keeps its pixels per unit length: a short (end) face takes the strip's first part
            k = min(1., float(np.linalg.norm(top[j] - top[i]) / max(np.linalg.norm(real_side[1] - real_side[0]), 1)))
            src = np.array([real_side[0], real_side[0] + k * (real_side[1] - real_side[0]),
                            real_side[3] + k * (real_side[2] - real_side[3]), real_side[3]], np.float32)
            fw = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, face), (W, H), flags=cv2.INTER_LINEAR)
            fm = np.zeros((H, W), np.uint8); cv2.fillPoly(fm, [np.round(face).astype(np.int32)], 1)
            out[fm > 0] = fw[fm > 0]
    out[topm > 0] = warped[topm > 0]
    out = cv2.GaussianBlur(out, (3, 3), 0) * (side > 0)[..., None] + a * (side == 0)[..., None]   # camera softness
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), side > 0


def _base(mask):
    """Centre of a load's footprint (its lowest 20 % of rows) and its overall width."""
    ys, xs = np.nonzero(mask); b = ys >= ys.max() - .2 * (ys.max() - ys.min())
    return np.array([xs[b].mean(), ys[b].mean()]), float(xs.max() - xs.min() + 1)


def library_cargo(camera, rng, target_w, target_px, library="work/library/index.json", min_solidity=.9, max_scale=4 / 3,
                  near_px=350):
    """A random whole cargo crop (stretch-wrapped load or cartons) from the sample library: from this camera and
    photographed within near_px of the target spot (same perspective) when possible, compact (mask / convex hull
    >= min_solidity, so no clipped or merged crops) and at about the target size (scale within
    [1/max_scale, max_scale], so it is not blown up or shrunk to mush)."""
    import cv2
    rows = [r for r in json.loads(Path(library).read_text())["crops"]
            if r["class"] in ("wrapped_cargo", "carton") and not r["touches_border"]]
    near = lambda r: r["camera"] == camera and math.dist(r["bottom_center_px"], target_px) <= near_px
    rng.shuffle(rows); rows.sort(key=lambda r: (not near(r), r["camera"] != camera))   # stable sort keeps the shuffle
    for r in rows:
        alpha = np.asarray(Image.open(r["crop"]).convert("RGBA"))[..., 3] > 127
        hull = cv2.convexHull(np.argwhere(alpha)[:, ::-1].astype(np.int32))
        k = target_w / _base(alpha)[1]
        if alpha.sum() / max(cv2.contourArea(hull), 1) >= min_solidity and 1 / max_scale <= k <= max_scale:
            return r
    raise SystemExit("no library cargo crop fits")


def paste_cargo(pasted, crop_path, target_w, new_quad):
    """Paste a library cargo crop on the new sheet: scaled to width target_w, its footprint centred on the new
    sheet's image centre. Returns the new frame, the pasted cargo mask and the placement."""
    import cv2
    crop = np.asarray(Image.open(crop_path).convert("RGBA")); alpha = crop[..., 3] > 127
    f, crop_w = _base(alpha)
    k = target_w / crop_w; c_new = new_quad.mean(0)
    M = np.array([[k, 0, c_new[0] - k * f[0]], [0, k, c_new[1] - k * f[1]]], np.float32)
    H, W = np.asarray(pasted).shape[:2]
    rgb = cv2.warpAffine(crop[..., :3], M, (W, H), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(alpha.astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_LINEAR) > 127
    out = np.asarray(pasted).copy(); out[m] = rgb[m]
    return Image.fromarray(out), m, {"crop": crop_path, "scale": round(float(k), 3), "width_px": round(target_w, 1),
                                     "base_centre_px": [round(float(v), 1) for v in c_new]}


# ---------- ComfyUI ----------

def comfy_up():
    try:
        urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=3); return True
    except Exception:
        return False


def start_comfy(log):
    if comfy_up():
        return
    subprocess.Popen([str(COMFY_DIR / "python_embeded" / "python.exe"), "-s", "ComfyUI/main.py", "--windows-standalone-build",
                      "--listen", "127.0.0.1", "--port", str(COMFY_PORT), "--lowvram"], cwd=COMFY_DIR, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        if comfy_up():
            return
        time.sleep(2)
    raise RuntimeError(f"ComfyUI did not start; see {log}")


def upload(path, name):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n").encode() \
        + Path(path).read_bytes() + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(COMFY_URL + "/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req).read())["name"]


def workflow(image_name, ref_names, prompt, seed, steps=8, denoise=.75):
    """Instruction edit of the whole window (Qwen-Image-Edit is trained for that); pictures 2-3 = reference sheets.
    denoise < 1 starts from the pasted window, so the sheet keeps its pasted shape while Qwen adds surface and light.
    Locality is enforced afterwards: only the mask region is composited into the original frame."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": 1.0}},
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3.0}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "qwen_image"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "6": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "7": {"class_type": "LoadImage", "inputs": {"image": ref_names[0]}},
        "15": {"class_type": "LoadImage", "inputs": {"image": ref_names[1]}},
        "8": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0], "image2": ["7", 0], "image3": ["15", 0], "prompt": prompt}},
        "9": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0], "image2": ["7", 0], "image3": ["15", 0], "prompt": ""}},
        "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["5", 0]}},
        "12": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0],
                                                    "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "llm_edit"}},
    }


def free_comfy():
    """Unload ComfyUI's models so SAM3 (checks) has room; the next round reloads them."""
    req = urllib.request.Request(COMFY_URL + "/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req).read()


# Reviewed 2026-10-01 from the 40 best library LSP crops: single sheets lying flat on the floor. Stacks of sheets
# (thick, layered sides) were excluded; they made the model draw a raised slab.
FLAT_LSP_CROPS = ["work/library/lsp/ch10_0564.0s_3.png", "work/library/lsp/ch10_0186.0s_3.png",
                  "work/library/lsp/ch10_0348.0s_6.png", "work/library/lsp/ch10_0312.0s_3.png",
                  "work/library/lsp/ch10_0567.0s_4.png", "work/library/lsp/ch10_0360.0s_1.png"]


def reference_sheet(dest, groups, library="work/library/index.json", tile=300):
    """Grid of real crops on floor grey, one labelled row per group: (label, class, n) takes the n best library
    crops of that class, (label, [paths]) uses a reviewed list."""
    crops = json.loads(Path(library).read_text())["crops"]
    rows, used = [], []
    for label, *spec in groups:
        if isinstance(spec[0], list):
            pick = [{"crop": c} for c in spec[0]]; n = len(pick)
        else:
            cls, n = spec
            pick = sorted((r for r in crops if r["class"] == cls and not r["touches_border"] and r["area_px"] > 4000),
                          key=lambda r: -r["quality"])[:n]
        row = Image.new("RGB", (tile * n, tile + 34), (128, 130, 126)); ImageDraw.Draw(row).text((8, 8), label, fill=(255, 255, 255))
        for i, r in enumerate(pick):
            im = Image.open(r["crop"]).convert("RGBA"); im.thumbnail((tile - 16, tile - 16))
            bg = Image.new("RGBA", im.size, (128, 130, 126, 255)); bg.alpha_composite(im)
            row.paste(bg.convert("RGB"), (i * tile + (tile - im.width) // 2, 34 + (tile - im.height) // 2)); used.append(r["crop"])
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (128, 130, 126)); y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    sheet.save(dest); return used


def run_comfy(wf, timeout=3600):
    pid = json.loads(urllib.request.urlopen(urllib.request.Request(COMFY_URL + "/prompt", data=json.dumps({"prompt": wf}).encode(),
                                                                   headers={"Content-Type": "application/json"})).read())["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(urllib.request.urlopen(f"{COMFY_URL}/history/{pid}").read())
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st.get("messages", []))[:2000])
            for out in hist[pid]["outputs"].values():
                for im in out.get("images", []):
                    q = f"filename={im['filename']}&subfolder={im['subfolder']}&type={im['type']}"
                    return Image.open(io.BytesIO(urllib.request.urlopen(f"{COMFY_URL}/view?{q}").read())).convert("RGB"), time.time() - t0
        time.sleep(2)
    raise TimeoutError("ComfyUI job timed out")


# ---------- checks ----------

WINDOW = (1360, 768)   # TextEncodeQwenImageEditPlus resizes references to ~1 MP; at this size the resize is the identity,
                       # so the edited window is not zoomed/shifted against the input (seen at 768x432)


def window(mask, W, H):
    ys, xs = np.nonzero(mask); cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    w, h = WINDOW
    x0 = int(min(max(0, cx - w / 2), W - w)); y0 = int(min(max(0, cy - h / 2), H - h))
    return x0, y0, x0 + w, y0 + h


def check(orig, edited, poly_mask, edit_mask, win, sam, cargo_mask=None):
    """Pixel change outside the edit mask (within the edited window), a new dark platform over the sheet's visible
    part, and a cargo detection over the pasted cargo."""
    a = np.asarray(orig).astype(int); b = np.asarray(edited).astype(int)
    changed = np.abs(a - b).max(2) > 24
    inwin = np.zeros_like(edit_mask); inwin[win[1]:win[3], win[0]:win[2]] = True
    outside = float(changed[inwin & ~edit_mask].mean())
    state = sam.set_image(edited)
    seg = lambda p: [np.asarray(m.detach().cpu().float()).squeeze() > .5 for m in sam.set_text_prompt(state=state, prompt=p)["masks"]]
    visible = poly_mask & ~cargo_mask if cargo_mask is not None else poly_mask
    cover = max([float((m & visible).sum() / max(visible.sum(), 1)) for m in seg("black platform")], default=0.)
    cargo_cover = None
    if cargo_mask is not None:
        cargo_cover = max([float((m & cargo_mask).sum() / cargo_mask.sum()) for p in ("stretch wrapped cargo", "box") for m in seg(p)],
                          default=0.)
    floor = np.median(a[binary_dilation(edit_mask, iterations=25) & ~edit_mask].mean(1))
    darker = float((b[poly_mask].mean(1) < floor - 25).mean())
    return {"size_ok": orig.size == edited.size, "outside_changed": round(outside, 4), "new_sheet_cover": round(cover, 3),
            "new_cargo_cover": None if cargo_cover is None else round(cargo_cover, 3),
            "darker_than_floor": round(darker, 3), "changed": changed}


def feedback(rep):
    msgs = []
    if rep["new_sheet_cover"] < NEW_SHEET_MIN:
        msgs.append(f"The pasted sheet was lost or blended into the floor (detected over {rep['new_sheet_cover']:.0%} of its area). "
                    "Keep it clearly visible as a complete dark grey sheet at exactly its pasted position and size.")
    if rep.get("new_cargo_cover") is not None and rep["new_cargo_cover"] < NEW_CARGO_MIN:
        msgs.append(f"The new cargo load was lost (detected over {rep['new_cargo_cover']:.0%} of it). Keep a clearly visible "
                    "stretch-wrapped load standing on the new LSP, at its pasted position and size.")
    if rep["outside_changed"] > OUTSIDE_CHANGE_MAX:
        msgs.append(f"{rep['outside_changed']:.0%} of the rest of the picture changed: keep every other pixel of picture 1 "
                    "exactly as it is, only add the new sheet and its cargo.")
    return " ".join(msgs)


def run_codex(in_path, ref_paths, prompt, out_path, model=None, timeout=1800):
    """One edit round through the Codex CLI: picture 1 = in_path, pictures 2.. = ref_paths (attached with -i).
    The agent must write the edited picture 1, same size, to out_path. Returns (image, seconds)."""
    w, h = Image.open(in_path).size
    instr = (prompt + f" Picture 1 is the attached file {in_path.name}; pictures 2 and 3 are "
             + " and ".join(p.name for p in ref_paths) + ". Edit picture 1 with your image editing model and save "
             f"the result as a PNG of exactly {w}x{h} pixels to {out_path.resolve()}. Do not crop, pad, resize or "
             "move anything, do not modify the attached files and do not write any other file.")
    exe = shlex.split(os.environ["CODEX_BIN"]) if os.environ.get("CODEX_BIN") else [shutil.which("codex") or "codex"]
    cmd = [*exe, "exec", instr, "--skip-git-repo-check", "-s", "workspace-write", "-C", str(out_path.parent.resolve())]
    for img in [in_path, *ref_paths]:
        cmd += ["-i", str(Path(img).resolve())]
    if model:
        cmd += ["-m", model]
    cmd += shlex.split(os.environ.get("CODEX_EXTRA_ARGS", ""))
    out_path.unlink(missing_ok=True)
    t0 = time.time()
    with open(out_path.with_suffix(".codex.log"), "w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=timeout)
    if not out_path.exists():
        raise RuntimeError(f"codex wrote no image to {out_path}; see {out_path.with_suffix('.codex.log')}")
    return Image.open(out_path).convert("RGB"), time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7, help="picks the LSP texture and the cargo crop")
    parser.add_argument("--backend", choices=["comfy", "codex"], default="comfy", help="image-edit model runner")
    parser.add_argument("--codex-model", default=os.environ.get("CODEX_MODEL"), help="codex exec -m (e.g. the Astra 6 model id)")
    args = parser.parse_args()
    import torch
    from synth.segment import load_processor
    cfg = load_config(); work = cfg["work"]; cam = json.loads((work / "camera.json").read_text())
    tracks = json.loads((work / "tracks.json").read_text())
    lsp = json.loads(Path("config/standards.json").read_text())["lsp"]
    out = work / "llm_edit" / Path(args.frame).stem / f"seed{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    orig = Image.open(work / "frames" / args.frame).convert("RGB"); W, H = orig.size
    det = max((d for d in tracks["detections"] if d["frame"] == args.frame), key=lambda d: d["area_px"])
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    sam0 = load_processor(); sam0.confidence_threshold = .4; state = sam0.set_image(orig)
    seg = lambda p, m=sam0, st=state: [np.asarray(x.detach().cpu().float()).squeeze() > .5 for x in m.set_text_prompt(state=st, prompt=p)["masks"]]
    tb = det["box"]
    def dist(p):
        ys, xs = np.nonzero(p); return math.hypot(xs.mean() - (tb[0] + tb[2]) / 2, ys.mean() - (tb[1] + tb[3]) / 2)
    real = min((p for p in seg("black platform") if p.any()), key=dist)
    occ = np.zeros_like(real); cargo_masks = []
    for p in ["stretch wrapped cargo", "box", "person"]:   # may hide the sheet; the truck may not (it is behind the sheet)
        for mm in seg(p):
            occ |= mm
            if p != "person":
                cargo_masks.append(mm)
    occ &= ~real
    near_real = binary_dilation(real, iterations=15)
    real_cargo = max(cargo_masks, key=lambda mm: (mm & near_real).sum() + .01 * mm.sum())   # the load on the real sheet
    sc = tracks["scale_correction"]; length, width, z = lsp["size_m"][1] * sc["xy"], lsp["size_m"][0] * sc["xy"], lsp["thickness_m"] * sc["z"]
    ys, xs = np.nonzero(real); c0 = floor_xy(cam, xs.mean(), ys.mean())
    (cx, cy, h, kl, kw), fit_score = fit_sheet(cam, real, occ, (c0[0], c0[1], det["pose"][2]), length, width, z)
    length, width = length * kl, width * kw
    pose = next_sheet((cx, cy, h), det["pose"][:2], length)
    top, bottom, edge_info = image_quads(cam, (cx, cy, h), pose, length, width, z, real, occ,
                                         np.asarray(orig.convert("L")))
    rng = random.Random(args.seed)
    sheet_crop = rng.choice(FLAT_LSP_CROPS)
    front_face = edge_info["front"].get("two_lines_px_apart") or 2.   # the real front face's measured width
    pasted, poly_mask = paste_sheet(orig, top, bottom, real, sheet_crop, np.array(edge_info["real_side_px"], np.float32),
                                    front_overlap=front_face + 1)
    target_w = _base(real_cargo)[1]   # the two sheets look the same size, so the new load is as wide as the real one
    cargo_crop = library_cargo(cfg["camera_id"], rng, target_w, top.mean(0))["crop"]
    pasted, cargo_mask, cargo_info = paste_cargo(pasted, cargo_crop, target_w, top)
    pasted.save(out / "pasted.png")
    poly = top
    geo = {"real_sheet": {"centre_m": [round(cx, 3), round(cy, 3)], "heading_deg": round(math.degrees(h) % 360, 2),
                          "size_factor_length_width": [round(kl, 3), round(kw, 3)], "fit_score": round(fit_score, 3)},
           "new_sheet_centre_m": [round(pose[0], 3), round(pose[1], 3)], "sheet_size_m": [round(length, 3), round(width, 3)],
           "image_edges": edge_info, "new_sheet_top_px": top.round(1).tolist(), "new_sheet_texture": sheet_crop,
           "new_cargo": cargo_info}
    import gc
    del sam0, state, seg; gc.collect(); torch.cuda.empty_cache()   # ComfyUI needs the GPU; SAM3 is reloaded only for the checks
    model = ({"backend": "comfy", "unet": UNET, "lora": LORA, "clip": CLIP} if args.backend == "comfy"
             else {"backend": "codex", "model": args.codex_model})
    report = {"frame": args.frame, "camera_id": cfg["camera_id"], "geometry": geo,
              "polygon_px": poly.round(1).tolist(), "model": model, "rounds": []}
    if args.backend == "comfy":
        start_comfy(out / "comfyui.log")
    report["references"] = {
        "lsp": reference_sheet(out / "reference_lsp.png", [("LSP - one flat dark grey plastic slip sheet lying directly on the floor",
                                                            FLAT_LSP_CROPS)]),
        "others": reference_sheet(out / "reference_others.png", [("cargo - stretch wrapped (NOT LSP)", "wrapped_cargo", 3),
                                                                 ("cargo - cardboard cartons (NOT LSP)", "carton", 3),
                                                                 ("SKID - wooden pallet (NOT LSP)", "skid", 3)])}
    refs = [out / "reference_lsp.png", out / "reference_others.png"]
    if args.backend == "comfy":
        ref_names = [upload(p, f"llm_edit_{out.name}_ref_{p.stem.split('_')[-1]}.png") for p in refs]
    prompt, dil, accepted = BASE_PROMPT, 10, None
    for r in range(1, args.rounds + 1):
        edit_mask = binary_dilation(poly_mask | cargo_mask, iterations=dil)
        x0, y0, x1, y1 = window(edit_mask, W, H)
        crop = orig.crop((x0, y0, x1, y1))
        pasted.crop((x0, y0, x1, y1)).save(out / f"r{r}_in.png")      # sheet and cargo already in place; Qwen makes them real
        Image.fromarray((edit_mask[y0:y1, x0:x1] * 255).astype(np.uint8)).save(out / f"r{r}_mask.png")
        if args.backend == "comfy":
            img_name = upload(out / f"r{r}_in.png", f"llm_edit_{out.name}_r{r}.png")
            res, secs = run_comfy(workflow(img_name, ref_names, prompt, seed=1000 + r))
            free_comfy()
        else:
            res, secs = run_codex(out / f"r{r}_in.png", refs, prompt, out / f"r{r}_codex_out.png", args.codex_model)
        if res.size != crop.size:
            res = res.resize(crop.size, Image.LANCZOS)
        full = orig.copy(); full.paste(res, (x0, y0)); full.save(out / f"r{r}_model_output.png")   # raw model output, same size as input
        sam = load_processor(); sam.confidence_threshold = .4
        rep = check(orig, full, poly_mask, edit_mask, (x0, y0, x1, y1), sam, cargo_mask)
        del sam; gc.collect(); torch.cuda.empty_cache()
        Image.fromarray((rep.pop("changed") * 255).astype(np.uint8)).save(out / f"r{r}_diff.png")
        ok = (rep["size_ok"] and rep["outside_changed"] <= OUTSIDE_CHANGE_MAX and rep["new_sheet_cover"] >= NEW_SHEET_MIN
              and rep["new_cargo_cover"] >= NEW_CARGO_MIN)
        report["rounds"].append({"round": r, "prompt": prompt, "mask_dilation_px": dil, "window": [x0, y0, x1, y1],
                                 "seconds": round(secs, 1), **rep, "pass": ok})
        print(f"round {r}: {rep} pass={ok} ({secs:.0f}s)", flush=True)
        if ok:
            accepted = (r, res, (x0, y0), edit_mask); break
        prompt = BASE_PROMPT + " " + feedback(rep)
    if accepted:
        r, res, (x0, y0), edit_mask = accepted
        soft = Image.fromarray((edit_mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
        full = orig.copy(); full.paste(res, (x0, y0)); final = Image.composite(full, orig, soft)  # original outside the mask
        final.save(out / "final.png"); final.save(out / "final.jpg", quality=95)
        diff = np.abs(np.asarray(final).astype(int) - np.asarray(orig).astype(int)).max(2) > 0
        report["final"] = {"round": r, "size": list(final.size), "changed_px_outside_mask": int((diff & ~binary_dilation(edit_mask, iterations=4)).sum()),
                           "label": {"class": "Forklift Pushing Multiple Lsps", "scenario": "V1", "lsp_count": 2,
                                     "cargo_on": "each", "is_violation": True}}
    else:
        report["final"] = None
    write_json(out / "report.json", report)
    print(json.dumps({"accepted_round": report["final"] and report["final"]["round"], "out": str(out)}))


if __name__ == "__main__":
    main()
