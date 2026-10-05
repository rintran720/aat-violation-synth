"""SAM3/MoGe measurements for editing one valid CCTV frame.

The calibrated polygon is the geometric target. A hidden source edge is
inferred from calibration and tagged as such; it is never described as visible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PROMPTS = {
    "forklift": ("forklift",),
    "LSP": ("black platform", "flat top of platform"),
    "cargo": ("stretch wrapped cargo", "box"),
    "floor": ("floor",),
}
GUIDE_COLOURS = ((0, 255, 70),)   # line colour of the geometry guide image


def union(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    return np.logical_or.reduce(masks) if masks else np.zeros(shape, dtype=bool)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.nonzero(mask)
    return (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1) if len(x) else None


class Sam3Masks:
    def __init__(self, cache: Path):
        self.cache = cache
        self.processor = None

    def get(self, image_path: Path, keys: tuple[str, ...]) -> dict[str, list[np.ndarray]]:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()[:20]
        folder = self.cache / digest
        folder.mkdir(parents=True, exist_ok=True)
        result: dict[str, list[np.ndarray]] = {}
        missing = [key for key in keys if not (folder / f"{key}.npz").exists()]
        if missing:
            if self.processor is None:
                from synth.segment import load_processor
                self.processor = load_processor()
                # SAM3 emits bf16 activations in part of its image backbone
                # even when loaded in float32 on CPU. Match each layer input
                # to its weight dtype without changing the vendored package.
                if self.processor.device == "cpu":
                    import torch

                    def match_weight_dtype(module, args):
                        value = args[0]
                        if value.dtype != module.weight.dtype:
                            return (value.to(module.weight.dtype),)
                        return None

                    for module in self.processor.model.modules():
                        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                            module.register_forward_pre_hook(match_weight_dtype)
            import contextlib
            import torch
            # On CUDA the SAM3 checkpoint runs in bfloat16 autocast (as synth.segment does); without it the
            # backbone mixes bf16 activations with fp32 weights and fails.
            autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if self.processor.device == "cuda"
                        else contextlib.nullcontext())
            with autocast, Image.open(image_path) as image:
                state = self.processor.set_image(image.convert("RGB"))
            for key in missing:
                masks: list[np.ndarray] = []
                for prompt in PROMPTS[key]:
                    with autocast:
                        output = self.processor.set_text_prompt(state=state, prompt=prompt)
                    for mask, score in zip(output["masks"], output["scores"]):
                        if float(score) < 0.5:
                            continue
                        array = np.asarray(mask.detach().cpu().float()).squeeze() > 0.5
                        if not any((array & old).sum() > 0.7 * min(array.sum(), old.sum()) for old in masks):
                            masks.append(array)
                np.savez_compressed(folder / f"{key}.npz", *masks)
        for key in keys:
            with np.load(folder / f"{key}.npz") as saved:
                result[key] = [saved[name].astype(bool) for name in sorted(saved.files)]
        return result


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(0)
    _, _, vt = np.linalg.svd(points - centre)
    direction = vt[0] if vt[0][0] >= 0 else -vt[0]
    return centre, direction


def top_face_edge(mask: np.ndarray, gray: np.ndarray, max_angle_deg: float = 40,
                  max_side_px: float = 15) -> tuple[list[list[int]], dict] | None:
    """Front edge of the LSP's TOP face, measured on the photo.

    The lowest row of a SAM3 sheet mask is where the 9 cm side face meets the floor, but the geometry guide
    projects at the top-face height, so the edge must be the top-face edge. OpenCV's line segment detector finds
    segments on the lower outline of the mask (within max_angle_deg of horizontal, so slanted sheets work); when
    two parallel lines a few pixels apart are found (top-face edge above, floor contact below) the upper one is
    kept. Endpoints are the mask's extent along that line. Returns None when no edge is measurable."""
    import cv2
    from scipy.ndimage import binary_erosion, distance_transform_edt

    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    band_top = y1 - max(12, int(.35 * (y1 - y0)))          # lower part of the sheet only
    near = distance_transform_edt(~(mask & ~binary_erosion(mask))) <= 6
    has = mask.any(axis=0)
    lowest = np.where(has, mask.shape[0] - 1 - np.argmax(mask[::-1], axis=0), -1)   # lowest mask row per column
    pad = 8
    cx0, cy0 = max(0, x0 - pad), max(0, band_top - pad)
    crop = gray[cy0:min(gray.shape[0], y1 + pad), cx0:min(gray.shape[1], x1 + pad)]
    found = cv2.createLineSegmentDetector().detect(np.ascontiguousarray(crop, dtype=np.uint8))[0]
    if found is None:
        return None
    points = []
    for xa, ya, xb, yb in found.reshape(-1, 4) + [cx0, cy0, cx0, cy0]:
        length = float(np.hypot(xb - xa, yb - ya))
        angle = abs(np.degrees(np.arctan2(yb - ya, xb - xa))) % 180
        if length < 12 or min(angle, 180 - angle) > max_angle_deg:
            continue
        mx, my = int(round((xa + xb) / 2)), int(round((ya + yb) / 2))
        if my < band_top or not (0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1]) or not near[my, mx]:
            continue
        if lowest[mx] < 0 or not 0 <= lowest[mx] - my <= max_side_px + 6:
            continue   # not on the sheet's lower outline (e.g. where the cargo stands on the deck)
        t = np.linspace(0, 1, max(2, int(length)))[:, None]
        points.append(np.array([xa, ya]) + t * np.array([xb - xa, yb - ya]))   # one point per pixel = length weight
    if not points or sum(len(p) for p in points) < 25:
        return None
    points = np.vstack(points)
    centre, direction = _fit_line(points)
    normal = np.array([-direction[1], direction[0]])
    normal = normal if normal[1] >= 0 else -normal          # points down the image, towards the floor contact
    offset = (points - centre) @ normal
    split = None
    if np.percentile(offset, 90) - np.percentile(offset, 10) > 4:
        low, high = np.percentile(offset, 15), np.percentile(offset, 85)
        upper = offset < (low + high) / 2
        for _ in range(10):
            upper = np.abs(offset - low) < np.abs(offset - high)
            if upper.all() or not upper.any():
                break
            low, high = offset[upper].mean(), offset[~upper].mean()
        if 20 <= upper.sum() < len(points) and 3 < high - low <= max_side_px:   # a side face, not another object
            centre, direction = _fit_line(points[upper]); split = round(float(high - low), 1)
    rel = np.c_[xs, ys] - centre
    along = rel @ direction
    across = np.abs(rel @ np.array([-direction[1], direction[0]]))
    on_line = along[across <= 6]                            # mask pixels on the fitted line
    if len(on_line) < 20:
        return None
    a, b = centre + on_line.min() * direction, centre + on_line.max() * direction
    edge = [[int(round(a[0])), int(round(a[1]))], [int(round(b[0])), int(round(b[1]))]]
    return edge, {"method": "lsd_top_face_edge", "edge_observed": True, "two_lines_px_apart": split,
                  "edge_angle_deg": round(float(np.degrees(np.arctan2(direction[1], direction[0]))), 1),
                  "lsd_points": int(len(points))}


def source_edge(masks: dict[str, list[np.ndarray]], image_size: tuple[int, int],
                gray: np.ndarray | None = None) -> tuple[list[list[int]], dict]:
    """Find the central loaded LSP by cargo overlap and the front edge of its top face (top_face_edge, needs the
    grey image); without it, or when no edge is measurable, the mask's lowest row (the floor contact)."""
    width, height = image_size
    cargo = []
    for mask in masks.get("cargo", []):
        box = bbox(mask)
        if box and box[2] - box[0] > 45 and box[3] - box[1] > 45:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            cargo.append((abs(cx - width / 2) + 0.7 * abs(cy - height * 0.36), box))
    if not cargo:
        raise RuntimeError("SAM3 found no suitable central cargo; supply --contact-edge")
    cargo_box = min(cargo)[1]
    candidates = []
    for mask in masks.get("LSP", []):
        box = bbox(mask)
        if not box:
            continue
        overlap = max(0, min(box[2], cargo_box[2]) - max(box[0], cargo_box[0]))
        if overlap < 0.35 * (cargo_box[2] - cargo_box[0]):
            continue
        if abs(box[3] - cargo_box[3]) > 90:
            continue
        score = overlap - abs(box[3] - cargo_box[3])
        candidates.append((score, mask, box))
    if not candidates:
        raise RuntimeError("SAM3 did not isolate the loaded source LSP; supply --contact-edge")
    _, mask, box = max(candidates, key=lambda item: item[0])
    if gray is not None:
        measured = top_face_edge(mask, gray)
        if measured is not None:
            edge, info = measured
            return edge, {**info, "cargo_bbox": cargo_box, "lsp_bbox": box}
    ys = np.flatnonzero(mask.any(axis=1))
    rim = mask[max(int(ys[-1]) - 10, 0):int(ys[-1]) + 1]
    xx = np.flatnonzero(rim.any(axis=0))
    if len(xx) < 35:
        raise RuntimeError("Source LSP front rim is too small or occluded; supply --contact-edge")
    edge = [[int(xx[0]), int(ys[-1])], [int(xx[-1]), int(ys[-1])]]
    return edge, {"method": "sam3_exposed_rim", "cargo_bbox": cargo_box,
                  "lsp_bbox": box, "edge_observed": True}


class MogePoints:
    """MoGe-2 metric point map (camera frame, metres) and its validity per image, cached by file hash. The model
    loads on first use and stays loaded for the next candidate."""

    def __init__(self, cache: Path):
        self.cache = cache
        self.model = None

    def get(self, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
        path = self.cache / f"{hashlib.sha256(image_path.read_bytes()).hexdigest()[:20]}.npz"
        if not path.exists():
            import torch
            from moge.model.v2 import MoGeModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.model is None:
                self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            with torch.inference_mode():
                inferred = self.model.infer(torch.tensor(rgb, device=device).permute(2, 0, 1),
                                            resolution_level=9, use_fp16=device == "cuda")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, points=inferred["points"].cpu().numpy(),
                                valid=inferred["mask"].cpu().numpy().astype(bool))
        with np.load(path) as saved, Image.open(image_path) as image:
            points, valid = saved["points"], saved["valid"].astype(bool)
            if points.shape[:2] != (image.height, image.width):
                raise RuntimeError(f"MoGe point map {points.shape[:2]} does not match the image {image_path}")
        return points, valid


def floor_frame(points: np.ndarray, valid: np.ndarray, floor: np.ndarray) -> dict:
    """Plane fitted to the free floor's MoGe points: origin, axes (two in the plane, then the normal towards the
    camera) and the fit's residuals."""
    sample = points[floor & valid]
    sample = sample[np.isfinite(sample).all(axis=1)]
    if len(sample) < 1000:
        raise RuntimeError("MoGe/SAM3 found too few free floor points to measure the sheets")
    sample = sample[::max(1, len(sample) // 25000)]
    origin = np.median(sample, axis=0)
    _, _, vh = np.linalg.svd(sample - origin, full_matrices=False)
    normal = vh[-1] if float(vh[-1] @ -origin) > 0 else -vh[-1]      # the camera is at the origin
    first = vh[0] - (vh[0] @ normal) * normal
    first /= np.linalg.norm(first)
    residual = np.abs((sample - origin) @ normal)
    return {"origin": origin, "axes": np.stack([first, np.cross(normal, first), normal]),
            "floor_points": len(sample), "median_floor_residual_m": round(float(np.median(residual)), 4),
            "p90_floor_residual_m": round(float(np.percentile(residual, 90)), 4)}


def on_floor(points: np.ndarray, valid: np.ndarray, mask: np.ndarray, frame: dict,
             max_height: float = .3) -> np.ndarray:
    """In-plane coordinates of a mask's points that lie on or just above the floor (a sheet, not cargo on it)."""
    local = (points[mask & valid] - frame["origin"]) @ frame["axes"].T
    local = local[np.isfinite(local).all(axis=1)]
    return local[(local[:, 2] > -.1) & (local[:, 2] < max_height), :2]


def point_on_floor(points: np.ndarray, valid: np.ndarray, pixel, frame: dict, radius: int = 4) -> np.ndarray:
    x, y = int(pixel[0]), int(pixel[1])
    window = np.zeros(valid.shape, bool)
    window[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = True
    local = (points[window & valid] - frame["origin"]) @ frame["axes"].T
    local = local[np.isfinite(local).all(axis=1)]
    if not len(local):
        raise RuntimeError(f"MoGe has no valid point near pixel {pixel}")
    return np.median(local[:, :2], axis=0)


def edge_on_floor(points: np.ndarray, valid: np.ndarray, edge_px: list[list[int]], toward_px,
                  frame: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """The shared edge on the floor plane: the floor position of edge_px[0], the unit direction to edge_px[1] and
    the edge's length. Located from pixels along the edge, 1 px on the original sheet's side (its top face or the
    cargo standing on it, both above the edge on the floor plane) and 5% in from the ends: the end pixels
    themselves often show the floor beside the sheet, which at a grazing view lies decimetres further away."""
    a, b = np.asarray(edge_px, dtype=float)
    normal = np.array([-(b - a)[1], (b - a)[0]]) / max(float(np.linalg.norm(b - a)), 1e-6)
    if float((np.asarray(toward_px, dtype=float) - a) @ normal) > 0:
        normal = -normal                                    # towards the original sheet
    t = np.linspace(.05, .95, 37)
    px = np.round(a + t[:, None] * (b - a) + normal).astype(int)
    px[:, 0] = px[:, 0].clip(0, valid.shape[1] - 1)
    px[:, 1] = px[:, 1].clip(0, valid.shape[0] - 1)
    ok = valid[px[:, 1], px[:, 0]]
    if ok.sum() < 10:
        raise RuntimeError("MoGe has too few valid points along the original sheet's front edge")
    xy = ((points[px[ok, 1], px[ok, 0]] - frame["origin"]) @ frame["axes"].T)[:, :2]
    centre, direction = _fit_line(xy)
    slope, start = np.polyfit(t[ok], (xy - centre) @ direction, 1)     # floor position along the edge against t
    if slope < 0:
        direction, slope, start = -direction, -slope, -start
    return centre + start * direction, direction, float(slope)


# Depth runs along the viewing direction, where neither the calibrated camera nor MoGe is good to 10% (they
# differ by about 20% on s_003), so it gets twice the width's tolerance.
SHEET_TOLERANCES = {"angle_deg": 5., "width_ratio": .10, "depth_ratio": .20, "gap_m": .10, "lateral_m": .15}


def compare_sheets(points: np.ndarray, valid: np.ndarray, floor: np.ndarray, old_mask: np.ndarray | None,
                   edge_px: list[list[int]], toward_px, new_mask: np.ndarray, depth_to_width: float,
                   tolerances: dict = SHEET_TOLERANCES) -> dict:
    """Orientation, size and position of the new sheet against the original one, on the floor plane of one MoGe
    point map (both sheets in the same map, so MoGe's scale cancels out).

    The original's front edge (edge_px, the shared edge) gives the frame: x along it from edge_px[0] (its left end
    in the image) to edge_px[1], y across it towards toward_px, where the new sheet belongs. The original's width is
    the edge; its depth is measured behind the edge when visible, else it is depth_to_width times that width (cargo
    usually hides most of a loaded sheet): the calibrated depth over the edge's width as the calibrated camera
    measures it, which is the depth the geometry guide draws, free of MoGe's scale."""
    # the floor around the sheets: MoGe's floor is not one plane over a wide-angle frame (s_003: 8 cm median
    # residual over the whole floor, 1 cm within 250 px), and the height filter of on_floor needs the local one
    yy, xx = np.mgrid[0:valid.shape[0], 0:valid.shape[1]]
    middle = np.mean(edge_px, axis=0)
    radius = max(250., 1.5 * float(np.linalg.norm(np.subtract(edge_px[1], edge_px[0]))))
    nearby = floor & (np.hypot(xx - middle[0], yy - middle[1]) < radius)
    frame = floor_frame(points, valid, nearby if (nearby & valid).sum() >= 1000 else floor)
    a, ex, old_width = edge_on_floor(points, valid, edge_px, toward_px, frame)
    ey = np.array([-ex[1], ex[0]])
    if float((point_on_floor(points, valid, toward_px, frame) - a) @ ey) < 0:
        ey = -ey

    def edge_coords(xy: np.ndarray) -> np.ndarray:
        return np.c_[(xy - a) @ ex, (xy - a) @ ey]

    expected_depth = depth_to_width * old_width
    old_depth, depth_source = expected_depth, "calibration"
    if old_mask is not None:
        behind = -edge_coords(on_floor(points, valid, old_mask, frame))[:, 1]
        behind = behind[behind > 0]
        if len(behind) >= 50 and np.percentile(behind, 99) >= .9 * expected_depth:
            old_depth, depth_source = float(np.percentile(behind, 99)), "measured"
    result = {"old_width_m": round(old_width, 3), "old_depth_m": round(old_depth, 3), "old_depth_source": depth_source,
              **{key: frame[key] for key in ("floor_points", "median_floor_residual_m", "p90_floor_residual_m")}}
    new = edge_coords(on_floor(points, valid, new_mask, frame))
    if len(new) < 50:
        return {**result, "passed": False, "failures": ["the new LSP could not be measured on the floor plane"]}

    import cv2
    corners = cv2.boxPoints(cv2.minAreaRect(new.astype(np.float32)))
    sides = [corners[1] - corners[0], corners[2] - corners[1]]
    angles = [(np.degrees(np.arctan2(s[1], s[0])) + 90) % 180 - 90 for s in sides]
    angle = float(min(angles, key=abs))                     # of the side closest to the shared edge
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    aligned = new @ np.array([[c, -s], [s, c]])             # rotated back by angle
    low, high = np.percentile(aligned, 1, axis=0), np.percentile(aligned, 99, axis=0)
    new_width, new_depth = float(high[0] - low[0]), float(high[1] - low[1])
    gap = float(np.percentile(new[:, 1], 1))
    lateral = float((np.percentile(new[:, 0], 1) + np.percentile(new[:, 0], 99)) / 2 - old_width / 2)
    width_ratio, depth_ratio = new_width / old_width, new_depth / old_depth
    failures = []
    if float(np.median(new[:, 1])) <= 0:
        failures.append("the new LSP is behind the original sheet's front edge, not in front of it")
    if abs(angle) > tolerances["angle_deg"]:
        failures.append(f"the new LSP is rotated {angle:+.1f} deg relative to the original sheet's front edge; it "
                        f"must be parallel to it (within {tolerances['angle_deg']:.0f} deg)")
    if abs(width_ratio - 1) > tolerances["width_ratio"]:
        failures.append(f"the new LSP is {width_ratio:.0%} of the original's width along the shared edge "
                        f"({new_width:.2f} m vs {old_width:.2f} m); it must be the same size")
    if abs(depth_ratio - 1) > tolerances["depth_ratio"]:
        failures.append(f"the new LSP is {depth_ratio:.0%} of the original's depth across the shared edge "
                        f"({new_depth:.2f} m vs {old_depth:.2f} m); it must be the same size")
    if gap > tolerances["gap_m"]:
        failures.append(f"there is a {gap:.2f} m gap between the sheets; the new LSP must touch the original's "
                        "front edge")
    elif gap < -tolerances["gap_m"]:
        failures.append(f"the new LSP overlaps the original sheet by {-gap:.2f} m; it must start at the original's "
                        "front edge")
    if abs(lateral) > tolerances["lateral_m"]:
        failures.append(f"the new LSP is shifted {abs(lateral):.2f} m toward the {'right' if lateral > 0 else 'left'} "
                        "end of the shared edge; centre it on the original sheet")
    return {**result, "angle_deg": round(angle, 2), "new_width_m": round(new_width, 3),
            "new_depth_m": round(new_depth, 3), "width_ratio": round(width_ratio, 3),
            "depth_ratio": round(depth_ratio, 3), "gap_m": round(gap, 3), "lateral_m": round(lateral, 3),
            "passed": not failures, "failures": failures}


def polygon_mask(size: tuple[int, int], points: list[list[int]]) -> np.ndarray:
    image = Image.new("L", size)
    ImageDraw.Draw(image).polygon([tuple(point) for point in points], fill=255)
    return np.asarray(image) > 127


def edit_region(size: tuple[int, int], polygon: list[list[int]], stage: str) -> Image.Image:
    """Feather a bounded edit into the original; protect the rest pixel for pixel."""
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    margin_x = 35 if stage == "sheet" else 55
    top = min(ys) - (30 if stage == "sheet" else 280)
    bottom = max(ys) + (45 if stage == "sheet" else 55)
    left, right = min(xs) - margin_x, max(xs) + margin_x
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).rectangle((left, top, right, bottom), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(6))


def colour_match(edited: np.ndarray, original: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Per-channel gain/offset mapping the edit onto the original over `sample` pixels (the feathered ring of the
    edit region), so a global exposure or colour shift of the image model leaves no visible frame. Gains are
    limited to 0.8-1.25."""
    if sample.sum() < 200:
        return edited
    out = edited.astype(np.float32)
    for c in range(3):
        e, o = out[..., c][sample], original[..., c][sample].astype(np.float32)
        gain = float(np.clip(o.std() / max(float(e.std()), 1e-3), .8, 1.25))
        out[..., c] = (out[..., c] - e.mean()) * gain + o.mean()
    return np.clip(out, 0, 255).astype(np.uint8)


def preserve_outside(base: Path, generated: Path, output: Path, region: Image.Image,
                     protected: list[np.ndarray] | None = None, feather_px: int = 2) -> None:
    """Composite the edit into `base` inside `region` only, colour-matched to the base. Protected SAM3 objects keep
    their exact pixels; the edit fades out over `feather_px` pixels around them instead of a hard cut."""
    with Image.open(base) as original, Image.open(generated) as edited:
        o = np.asarray(original.convert("RGB"))
        e = np.asarray(edited.convert("RGB").resize(original.size, Image.Resampling.LANCZOS))
    alpha = np.asarray(region, dtype=np.float32) / 255
    keep = np.zeros(alpha.shape, bool)
    for mask in protected or []:
        if mask.shape != alpha.shape:
            raise RuntimeError("Protected SAM3 mask dimensions differ from source")
        keep |= mask
    if keep.any():
        soft = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).filter(
            ImageFilter.GaussianBlur(feather_px)), dtype=np.float32) / 255
        alpha = alpha * (1 - np.clip(2 * soft, 0, 1))
        alpha[keep] = 0
    e = colour_match(e, o, (alpha > .05) & (alpha < .95) & ~keep)
    result = np.round(e * alpha[..., None] + o * (1 - alpha[..., None])).astype(np.uint8)
    result[alpha == 0] = o[alpha == 0]
    Image.fromarray(result).save(output)


def raw_drift(base: Path, raw: Path, region: Image.Image, protected: list[np.ndarray],
              max_changed: float = .08, max_mean: float = 9.) -> dict:
    """How much the image model changed what it had to keep (outside the edit region and on the protected
    objects), measured on its RAW output, before the composite restores those pixels."""
    with Image.open(base) as a, Image.open(raw) as b:
        original = np.asarray(a.convert("RGB"), dtype=np.int16)
        edited = np.asarray(b.convert("RGB").resize(a.size, Image.Resampling.LANCZOS), dtype=np.int16)
    keep = np.asarray(region) == 0
    for mask in protected:
        keep = keep | mask
    delta = np.abs(original - edited).mean(axis=2)[keep]
    changed = float((delta > 30).mean()) if delta.size else 0.
    mean = float(delta.mean()) if delta.size else 0.
    failures = []
    if changed > max_changed or mean > max_mean:
        failures.append(f"the image model changed the scene it had to keep ({changed:.1%} pixels, mean RGB delta "
                        f"{mean:.1f}); keep the forklift, the original LSP and cargo and the background unchanged")
    return {"passed": not failures, "failures": failures, "changed_fraction": round(changed, 4),
            "mean_rgb_delta": round(mean, 2)}


def guide_marks(base: Path, candidate: Path, region: Image.Image, tolerance: int = 45, limit: int = 40) -> dict:
    """Pixels in the geometry-guide line colours that the candidate has and the base had not: the image model
    drew the guide into the photo."""
    with Image.open(base) as a, Image.open(candidate) as b:
        original = np.asarray(a.convert("RGB"), dtype=np.int16)
        edited = np.asarray(b.convert("RGB").resize(a.size, Image.Resampling.LANCZOS), dtype=np.int16)
    inside = np.asarray(region) > 0
    count = 0
    for colour in GUIDE_COLOURS:
        target = np.array(colour, dtype=np.int16)
        hits = lambda image: int(((np.abs(image - target) <= tolerance).all(axis=2) & inside).sum())
        count += max(0, hits(edited) - hits(original))
    failures = [] if count <= limit else [f"{count} pixels in the guide-line colours: do not draw the geometry guide"]
    return {"passed": not failures, "failures": failures, "guide_coloured_pixels": count}


def zoom_pair(before: Path, after: Path, polygon: list[list[int]], output: Path, height: int = 512) -> Path:
    """Side-by-side close-up of the edit (before | after) for the visual review: at full frame the new object is
    only a few hundred pixels wide and seams are invisible."""
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    box = (int(min(xs) - .6 * w), int(min(ys) - 1.6 * h), int(max(xs) + .6 * w), int(max(ys) + .5 * h))
    tiles = []
    for path in (before, after):
        with Image.open(path) as image:
            image = image.convert("RGB")
            clip = (max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3]))
            tile = image.crop(clip)
        tiles.append(tile.resize((max(1, round(tile.width * height / tile.height)), height), Image.Resampling.LANCZOS))
    pair = Image.new("RGB", (tiles[0].width + tiles[1].width + 8, height), (255, 255, 255))
    pair.paste(tiles[0], (0, 0))
    pair.paste(tiles[1], (tiles[0].width + 8, 0))
    pair.save(output)
    return output


def verify_unchanged(base: Path, candidate: Path, region: Image.Image,
                     protected: list[np.ndarray]) -> dict:
    """Integrity check of the composite: pixels outside the region and on protected objects are the base's."""
    with Image.open(base) as a, Image.open(candidate) as b:
        original = np.asarray(a.convert("RGB"))
        edited = np.asarray(b.convert("RGB"))
    if original.shape != edited.shape:
        return {"passed": False, "failures": ["Candidate dimensions differ from base"]}
    must_match = np.asarray(region) == 0
    for mask in protected:
        must_match |= mask
    changed = int(((original != edited).any(axis=2) & must_match).sum())
    return {"passed": changed == 0, "failures": [] if changed == 0 else
            [f"{changed} protected/background pixels changed"], "changed_pixels": changed}


def sheet_at_edge(masks: list[np.ndarray], edge_px: list[list[int]], band_px: float = 20) -> np.ndarray | None:
    """The LSP instance with the most pixels within band_px of the edge segment (the original, loaded sheet)."""
    a, b = np.asarray(edge_px, dtype=float)
    d = b - a
    best, count = None, 0
    for mask in masks:
        ys, xs = np.nonzero(mask)
        t = np.clip(((xs - a[0]) * d[0] + (ys - a[1]) * d[1]) / max(float(d @ d), 1e-6), 0, 1)
        near = int((np.hypot(xs - a[0] - t * d[0], ys - a[1] - t * d[1]) <= band_px).sum())
        if near > count:
            best, count = mask, near
    return best


def verify_sheet_geometry(source: dict[str, list[np.ndarray]], candidate: dict[str, list[np.ndarray]],
                          region: Image.Image, points: np.ndarray, valid: np.ndarray, floor: np.ndarray,
                          edge_px: list[list[int]], toward_px, depth_to_width: float) -> dict:
    """One new LSP inside the edit region, no cargo yet, and the new sheet's orientation, size and position equal
    to the original's on the floor plane (compare_sheets).

    The new sheet is its SAM3 instance on the new side of the shared edge's line. It is not "instance minus the
    original's mask": that mask also covers the original's side face and shadow below its top-face edge, pixels an
    adjoining new sheet hides, and removing them measured a correct sheet as short with a gap (s_003: 68% depth,
    0.31 m gap). The line cut also splits a new sheet that SAM3 merged with the original into one instance."""
    shape = valid.shape
    inside = np.asarray(region) > 0
    source_lsp = union(source.get("LSP", []), shape)
    # SAM3 can give different masks for unrelated parked LSPs even when their pixels are unchanged; only
    # instances with new pixels inside the edit region can be the added sheet.
    new = [mask for mask in candidate.get("LSP", []) if ((mask & ~source_lsp) & inside).sum() > 1500]
    failures = []
    if len(new) != 1:
        failures.append(f"SAM3 found {len(new)} new LSP instances; expected one")
    new_cargo = union(candidate.get("cargo", []), shape) & ~union(source.get("cargo", []), shape)
    if (new_cargo & inside).sum() > 1000:
        failures.append("Cargo appeared before the cargo stage")
    result = {"new_lsp_instances": len(new)}
    if new:
        a, b = np.asarray(edge_px, dtype=float)
        normal = np.array([-(b - a)[1], (b - a)[0]])
        if float((np.asarray(toward_px, dtype=float) - a) @ normal) < 0:
            normal = -normal
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        new_side = (xx - a[0]) * normal[0] + (yy - a[1]) * normal[1] >= 0
        sheet = max(new, key=lambda mask: int(mask.sum())) & new_side
        old = sheet_at_edge(source.get("LSP", []), edge_px)
        result.update(compare_sheets(points, valid, floor, old, edge_px, toward_px, sheet, depth_to_width))
        failures = result["failures"] + failures
    result.update({"passed": not failures, "failures": failures})
    return result


def verify_cargo_masks(sheet: dict[str, list[np.ndarray]], candidate: dict[str, list[np.ndarray]],
                       polygon: list[list[int]], size: tuple[int, int],
                       min_height_px: int = 0) -> dict:
    target = polygon_mask(size, polygon)
    sheet_cargo = union(sheet.get("cargo", []), target.shape)
    edited_cargo = union(candidate.get("cargo", []), target.shape)
    added = edited_cargo & ~sheet_cargo
    box = bbox(added)
    failures = []
    if box is None or added.sum() < 2500:
        failures.append("SAM3 found no substantial new cargo")
        supported = False
    else:
        if box[3] - box[1] < min_height_px:
            failures.append(f"New cargo height {box[3] - box[1]}px < {min_height_px}px")
        bottom = added[max(0, box[3] - 12):box[3]]
        ys, xs = np.nonzero(bottom)
        if len(xs):
            support_y = max(0, box[3] - 12) + int(np.median(ys))
            support_x = int(np.median(xs))
            supported = bool(target[support_y, support_x])
        else:
            supported = False
        if not supported:
            failures.append("New cargo bottom centre is outside the accepted LSP footprint")
    new_instances = [mask for mask in candidate.get("cargo", [])
                     if (mask & ~sheet_cargo).sum() > 2000
                     and bbox(mask) is not None
                     and bbox(mask)[3] >= min(point[1] for point in polygon) - 20]
    if len(new_instances) != 1:
        failures.append(f"SAM3 found {len(new_instances)} new cargo instances; expected one")
    sheet_lsp = union(sheet.get("LSP", []), target.shape)
    edited_lsp = union(candidate.get("LSP", []), target.shape)
    # Cargo legitimately occludes some of the top surface. Compare the
    # exposed deck only, with a small tolerance around the cargo outline.
    visible_deck = target & ~edited_cargo
    lsp_change = float(((sheet_lsp ^ edited_lsp) & visible_deck).sum()
                       / max((sheet_lsp & visible_deck).sum(), 1))
    if lsp_change > 0.12:
        failures.append(f"LSP mask changed {lsp_change:.1%} after cargo; need <=12%")
    return {"passed": not failures, "failures": failures, "new_cargo_pixels": int(added.sum()),
            "cargo_supported": supported, "lsp_mask_change_fraction": lsp_change}
