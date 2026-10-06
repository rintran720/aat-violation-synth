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
# SAM3 gives a probability per pixel (bilinear from its 1008 px map); the mask is where it exceeds this. The
# probability maps are cached so the figure can change without running the model again.
MASK_PROB = 0.5


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
                probs: list[np.ndarray] = []
                for prompt in PROMPTS[key]:
                    with autocast:
                        output = self.processor.set_text_prompt(state=state, prompt=prompt)
                    for prob, score in zip(output["masks_logits"], output["scores"]):
                        if float(score) < 0.5:
                            continue
                        prob = np.asarray(prob.detach().cpu().float()).squeeze()
                        array = prob > MASK_PROB
                        if not any((array & old).sum() > 0.7 * min(array.sum(), old.sum()) for old in masks):
                            masks.append(array)
                            probs.append(prob.astype(np.float16))
                np.savez_compressed(folder / f"{key}.npz", *masks)
                np.savez_compressed(folder / f"{key}_prob.npz", *probs)
        for key in keys:
            if (folder / f"{key}_prob.npz").exists():
                with np.load(folder / f"{key}_prob.npz") as saved:
                    result[key] = [saved[name].astype(np.float32) > MASK_PROB for name in sorted(saved.files)]
            else:                                              # cached before the probability maps were kept
                with np.load(folder / f"{key}.npz") as saved:
                    result[key] = [saved[name].astype(bool) for name in sorted(saved.files)]
        return result

    def probabilities(self, image_path: Path, key: str) -> list[np.ndarray] | None:
        """The cached probability maps behind get(image_path, (key,)), None for a cache from before they were kept."""
        self.get(image_path, (key,))
        path = self.cache / hashlib.sha256(image_path.read_bytes()).hexdigest()[:20] / f"{key}_prob.npz"
        if not path.exists():
            return None
        with np.load(path) as saved:
            return [saved[name].astype(np.float32) for name in sorted(saved.files)]


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(0)
    _, _, vt = np.linalg.svd(points - centre)
    direction = vt[0] if vt[0][0] >= 0 else -vt[0]
    return centre, direction


def dominant_line(xs: np.ndarray, ys: np.ndarray, max_angle_deg: float = 40, tolerance_px: float = 1.5,
                  min_inliers: int = 30) -> tuple[float, float, np.ndarray] | None:
    """The line y = offset + slope * x that most points lie on (RANSAC over pairs, then least squares on its
    inliers), within max_angle_deg of horizontal: (slope, offset, inliers). None with fewer than min_inliers."""
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        i, j = rng.choice(len(xs), 2, replace=False)
        if abs(int(xs[i]) - int(xs[j])) < min(20, min_inliers // 2):
            continue
        slope = (ys[j] - ys[i]) / (xs[j] - xs[i])
        if abs(np.degrees(np.arctan(slope))) > max_angle_deg:
            continue
        inliers = np.abs(ys[i] + slope * (xs - xs[i]) - ys) <= tolerance_px
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < min_inliers:
        return None
    slope, offset = np.polyfit(xs[best], ys[best], 1)
    return float(slope), float(offset), np.abs(offset + slope * xs - ys) <= tolerance_px


def floor_contact_edge(mask: np.ndarray, max_angle_deg: float = 40, tolerance_px: float = 1.5,
                       max_gap_px: int = 6) -> list[list[int]] | None:
    """Front bottom edge of a sheet's mask: the straight part of its lower outline, from one front corner to the
    other (left first). On a sheet lying flat this is where the front side face meets the floor; under a sheet that
    a forklift holds up it is the lower rim of the shadow, so it gives the sheet's extent, not its height. SAM3
    rounds a mask's corners, so the ends are not where the outline leaves the line but where the mask ends within a
    few pixels above it. None when the outline has no straight part of at least 30 px within max_angle_deg of
    horizontal."""
    columns = np.flatnonzero(mask.any(axis=0))
    if len(columns) < 30:
        return None
    lowest = mask.shape[0] - 1 - np.argmax(mask[::-1, columns], axis=0)     # lowest mask row of each column
    line = dominant_line(columns, lowest, max_angle_deg, tolerance_px)
    if line is None:
        return None
    slope, offset, inliers = line
    on_line = columns[inliers]
    runs = np.split(on_line, np.flatnonzero(np.diff(on_line) > max_gap_px) + 1)
    run = max(runs, key=len)                                                 # the straight part
    if run[-1] - run[0] < 30:
        return None
    ys, xs = np.nonzero(mask)
    above = offset + slope * xs - ys                                         # pixels above the line
    band = np.unique(xs[(above >= -tolerance_px) & (above <= 4)])            # the slab's lowest 4 px
    runs = np.split(band, np.flatnonzero(np.diff(band) > max_gap_px) + 1)
    ends = next(r for r in runs if r[0] <= run[0] and r[-1] >= run[-1])      # ...out to its two corners
    return [[int(x), int(round(offset + slope * x))] for x in (ends[0], ends[-1])]


def side_face_height(mask: np.ndarray, gray: np.ndarray, contact: list[list[int]], thickness_px: float,
                     min_step: float = 30) -> list[tuple[float, float, int] | None] | None:
    """Height in pixels of a sheet's black front side face above the lower outline of its mask (contact), for the
    left and the right half of the front: [(mean column, height, columns) or None, the same]. None when the photo
    does not show the sheet's own top face above its front anywhere. The two halves are kept apart because a
    forklift may hold one end of the sheet higher than the other: the shadow is then deeper at that end.

    The rim of the side face is where the photo steps from the sheet's lighter top face down to the black side
    face. Only columns where the top face is in view count: the mask continues above the step there. Under the
    load the black band also ends in a step, but that one is the load's bottom edge, which sits further back
    whenever the load does not reach the sheet's front. In each such column the highest step is the rim (lower
    ones are fork pockets, labels, and the seam between the side face and the shadow under a sheet that a forklift
    holds up); the height most columns of a half agree on wins. A step counts from min_step grey levels: the seam
    under the side face reaches 15-30 on the cam01 samples, the rim under a visible top face 30-60. Shadow under a raised sheet is as black as the
    side face and lies below it: it adds
    to this height instead of moving the rim (s_003: 11 px, where the camera gives 7 px for the thickness).
    Heights are searched from 0.5 to 1.8 times thickness_px, the camera's figure for the sheet's thickness: on a
    dark top face the only clear step is the load's bottom edge further up (s_011: 21 px for a 9 px sheet)."""
    from scipy.ndimage import uniform_filter1d

    (x0, y0), (x1, y1) = contact
    smooth = uniform_filter1d(gray.astype(np.float32), 5, axis=1)          # along the edge
    heights = np.arange(max(1, int(.5 * thickness_px) - 1), int(max(10, 1.8 * thickness_px)) + 2)
    rims = []
    for x in range(x0 + 2, x1 - 1):
        rows = int(round(np.interp(x, [x0, x1], [y0, y1]))) - heights
        if rows.min() < 7 or rows.max() + 1 >= gray.shape[0]:
            continue
        step = smooth[rows - 1, x] - smooth[rows + 1, x]                   # brighter above, darker below
        peak = np.flatnonzero((step[1:-1] >= min_step) & (step[1:-1] >= step[2:]) & (step[1:-1] >= step[:-2])) + 1
        if not len(peak):
            continue
        top = rows[peak[-1]]                                               # the top of the black band
        face = smooth[top - 6:top - 1, x].min() - smooth[top + 1, x] >= min_step   # a face above it, not a label
        if face and mask[top - 4, x]:                                      # ...and it is the sheet's own
            rims.append((x, heights[peak[-1]]))
    halves = []
    for side in (lambda x: x < (x0 + x1) / 2, lambda x: x >= (x0 + x1) / 2):
        columns = np.array([x for x, _ in rims if side(x)])
        values = np.array([h for x, h in rims if side(x)])
        if len(values) < 12:                                               # too few columns to trust an end
            halves.append(None)
            continue
        best = max(np.unique(values), key=lambda v: (np.abs(values - v) <= 1).sum())
        near = np.abs(values - best) <= 1
        halves.append((float(columns[near].mean()), float(values[near].mean()), int(near.sum())) if near.sum() >= 12
                      else None)
    return halves if any(halves) else None


EVEN_PX = 1.5      # the black band under a sheet counts as equally high at both ends within this


def line_y(line: list[list[int]], x: float) -> float:
    (x0, y0), (x1, y1) = line
    return y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1e-6)


def parallel_inside(gray: np.ndarray, mask: np.ndarray, slope: float, offset: float, rows: np.ndarray,
                    sign: int) -> tuple[float, float, int] | None:
    """A second side edge inside the mask, parallel to the outline's line x = offset + slope * y: the line fitted to
    the detector's segments that run within 15 degrees of it, 3-25 px inward, over the same rows. Where a side
    face is in view the outline is its bottom edge and this is the top edge. Returns (slope, offset, segments)."""
    import cv2

    y0, y1 = int(rows.min()), int(rows.max())
    x_line = offset + slope * np.array([y0, y1])
    cx0 = int(max(0, min(x_line) - 30))
    cx1 = int(min(gray.shape[1], max(x_line) + 31))
    if cx1 - cx0 < 8 or y1 - y0 < 8:
        return None
    found = cv2.createLineSegmentDetector().detect(np.ascontiguousarray(gray[y0:y1 + 1, cx0:cx1]))[0]
    if found is None:
        return None
    points, length = [], 0.
    for xa, ya, xb, yb in found.reshape(-1, 4) + [cx0, y0, cx0, y0]:
        if abs(yb - ya) < 6 or np.hypot(xb - xa, yb - ya) < 8:
            continue
        angle = np.degrees(np.arctan2(xb - xa, yb - ya))                     # of x against y, like `slope`
        if abs(((angle - np.degrees(np.arctan(slope))) + 90) % 180 - 90) > 15:
            continue
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        inward = (mx - (offset + slope * my)) * sign                         # distance inward of the outline line
        if not 3 <= inward <= 25 or not mask[int(my), int(mx)]:
            continue
        points += [(xa, ya), (xb, yb)]
        length += float(np.hypot(xb - xa, yb - ya))
    if length < 20:
        return None
    pts = np.asarray(points, dtype=float)
    fit = np.polyfit(pts[:, 1], pts[:, 0], 1)                                 # x = fit[0] * y + fit[1]
    return float(fit[0]), float(fit[1]), len(points) // 2


def side_edges(mask: np.ndarray, gray: np.ndarray, contact: list[list[int]], rim: list[list[int]],
               rim_ends: tuple[int, int] | None = None) -> dict:
    """The sheet's side edges, which run back from its front corners, and the front corners they give.

    The slab has four side edges: left top, left bottom, right top, right bottom. The mask's outline shows one per
    side: the bottom edge where that side face is in view (a dark strip just inside the outline), the top edge
    where it is not. Each is a line fitted to the outline rows above the front band. Where the side face is in
    view, its top edge is a second line inside the mask, parallel to the outline (parallel_inside): the pair is
    what bounds the rim, because the rim ends at the top corner, inside the side face's width, not at the mask's
    outline (s_003: the rim reached 13 px out over the left side face, and the new sheet was painted over it).
    A line's crossing with the front line of the same height (contact line for a bottom edge, rim for a top edge)
    is a front corner, sharper than the mask's rounded extent; the other corner on that side is straight above or
    below it. Returns {"left": {...} | None, "right": {...} | None, "corners_px": {"rim": {"left": x, "right":
    x}, "contact": {...}}}; a side without a usable line keeps the contact line's ends. angle_deg is the direction
    of the edge going back from the front, from the image's +x axis with y up: 90 is straight up, more leans
    left. rim_ends: where the top face ends on the rim (top_face_ends), the rim corners to start from; under a
    sheet held off the floor the contact line is the shadow's rim, shorter than the sheet (its ends would make the
    edge too short), so the top face is what gives the sheet's extent."""
    (x0, y0), (x1, y1) = contact
    rim_left, rim_right = rim_ends if rim_ends is not None else (x0, x1)
    result = {"corners_px": {"rim": {"left": rim_left, "right": rim_right}, "contact": {"left": x0, "right": x1}}}
    band = below_line(mask, rim, toward=[(x0 + x1) / 2, max(y0, y1) + 5])
    dark = float(np.median(gray[band])) if band.any() else 20.
    face_pixels = mask & ~band
    light = float(np.percentile(gray[face_pixels], 75)) if face_pixels.any() else dark + 60

    def crossing(slope, offset, front_line):
        (fx0, fy0), (fx1, fy1) = front_line                                  # x = offset + slope * y meets the line
        m = (fy1 - fy0) / max(fx1 - fx0, 1e-6)
        return (offset + slope * (fy0 - m * fx0)) / (1 - slope * m)

    for side, pick, sign in (("left", np.argmax, 1), ("right", lambda row: len(row) - 1 - np.argmax(row[::-1]), -1)):
        rows, xs = [], []
        for y in range(int(mask.shape[0])):
            row = mask[y]
            if not row.any():
                continue
            x = int(pick(row))
            if y >= line_y(rim, x) - 2:                                      # the front band is not a side
                continue
            rows.append(y)
            xs.append(x)
        result[side] = None
        if len(rows) < 12:
            continue
        line = dominant_line(np.array(rows, dtype=float), np.array(xs, dtype=float), max_angle_deg=70, min_inliers=12)
        if line is None:
            continue
        slope, offset, inliers = line                                        # x = offset + slope * y
        rows_in = np.array(rows)[inliers]
        inside = np.array([gray[y, min(max(int(round(offset + slope * y)) + sign * 3, 0), gray.shape[1] - 1)]
                           for y in rows_in])
        angle = float(np.degrees(np.arctan2(1, -slope)))     # of the edge going back: 90 straight up, >90 leaning left
        # a side face is in view only from its own side of the slab, and from there the side edges lean the other
        # way going back (the length's vanishing point lies beyond that face): the left face shows when the left
        # edge leans left, the right face when the right edge leans right. A dark strip inside an edge that leans
        # the wrong way is the sheet's own dark rim or shadow, not a side face (run 46da745, attempt 1: 6 px of
        # sheet would have been trimmed off its right side as "shadow").
        can_show = angle > 90 if side == "left" else angle < 90
        pair = parallel_inside(gray, mask, slope, offset, rows_in, sign) if can_show else None
        if pair is not None:                                                 # a side face: dark between the two lines
            between = np.array([gray[y, min(max(int(round((offset + slope * y + pair[1] + pair[0] * y) / 2)), 0),
                                            gray.shape[1] - 1)] for y in rows_in])
            if float(np.median(between)) >= dark + .5 * (light - dark):
                pair = None
        kind = "bottom" if pair is not None or (can_show and float(np.median(inside)) < dark + .35 * (light - dark)) else "top"
        # SAM3 rounds the corner: where the outline leaves the line before the front, the corner is straight
        # below the last point on it (the slab's vertical front edge), not further along the line
        y_last = float(rows_in.max())
        x_last = offset + slope * y_last
        short = line_y(rim, x_last) - y_last > 4
        x = x_last if short else crossing(slope, offset, contact if kind == "bottom" else rim)
        result[side] = {"kind": kind, "angle_deg": round(angle, 1), "rows": int(inliers.sum()),
                        "crossing_px": round(float(x), 1), "pair": pair is not None}
        corners = result["corners_px"]
        if abs(x - corners["contact"][side]) <= 12:                           # a plausible corner refines the end
            corners["contact"][side] = int(round(x))
        if abs(x - corners["rim"][side]) <= 12:
            corners["rim"][side] = int(round(x))
        if pair is not None:                                                 # the top edge bounds the rim itself
            top_x = crossing(pair[0], pair[1], rim)
            result[side]["top_edge_crossing_px"] = round(float(top_x), 1)
            if abs(top_x - corners["rim"][side]) <= 25:
                corners["rim"][side] = int(round(top_x))
    return result


def below_line(mask: np.ndarray, line: list[list[int]], toward) -> np.ndarray:
    """The mask's pixels on the side of `line` that contains the point `toward`: below the rim this is the front
    side face and the shadow under a raised sheet. A sheet standing in front hides that band, so it needs no
    protection in the composite (the slab is 3D)."""
    (x0, y0), (x1, y1) = line
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    sign = np.sign((toward[1] - y0) * (x1 - x0) - (toward[0] - x0) * (y1 - y0)) or 1
    return mask & (((yy - y0) * (x1 - x0) - (xx - x0) * (y1 - y0)) * sign > 0)


def top_face_ends(mask: np.ndarray, rim: list[list[int]], thickness_px: float) -> tuple[int, int] | None:
    """Where the sheet's top face ends along the rim: the x extent of the mask in the rows just above the rim
    line (up to 1.5 thicknesses, at least 4 px), ignoring columns with a stray pixel. None when the top face does
    not show above the front at all (a load that reaches the front edge)."""
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    above = line_y(rim, xx) - yy
    face = mask & (above >= 1) & (above <= max(4., 1.5 * thickness_px))
    columns = np.flatnonzero(face.sum(axis=0) >= 2)
    if len(columns) < 10:
        return None
    return int(columns[0]), int(columns[-1])


def push_direction(lsp: np.ndarray, forklift: np.ndarray) -> np.ndarray:
    """Unit image vector from the forklift's centroid to the sheet's: the direction the sheet is pushed in, and
    the direction a second sheet lies in."""
    ys, xs = np.nonzero(lsp)
    fy, fx = np.nonzero(forklift)
    d = np.array([xs.mean() - fx.mean(), ys.mean() - fy.mean()], dtype=float)
    norm = float(np.linalg.norm(d))
    if norm < 1e-6:
        raise RuntimeError("the forklift and the sheet have the same centroid")
    return d / norm


def front_direction(mask: np.ndarray, push) -> np.ndarray | None:
    """Image direction of the sheet's front edge: the mask's convex outline reduced to four sides, the two sides
    least parallel to the push are the front and the back (the other two run along the push), and the front is
    the one further along it. In perspective a push across the view has its front edge leaning in the image
    (not perpendicular to the push vector), so the turned frame follows this edge, not the push. None when the
    outline does not reduce to four sides."""
    import cv2

    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    epsilon = 2.
    points = hull.reshape(-1, 2).astype(float)
    while len(points) > 4 and epsilon < 80:
        points = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2).astype(float)
        epsilon *= 1.5
    if len(points) != 4:
        return None
    d = np.asarray(push, dtype=float)
    centre = points.mean(axis=0)
    sides = []
    for i in range(4):
        a, b = points[i], points[(i + 1) % 4]
        e = b - a
        if np.linalg.norm(e) < 8:
            return None
        e /= np.linalg.norm(e)
        sides.append((abs(float(e @ d)), float(((a + b) / 2 - centre) @ d), e))
    across = sorted(sides, key=lambda side: side[0])[:2]          # the two least parallel to the push
    front = max(across, key=lambda side: side[1])[2]              # the one further along it
    return front


class Rotation:
    """The image turned so that a direction points straight down (or, with `front`, so that the front edge is
    horizontal with the push pointing down). The slab measurements (floor_contact_edge,
    side_face_height, side_edges, trim_to_slab) read the sheet's front as its lower outline, which it is when the
    forklift pushes it towards the camera (cam01). On the other cameras the push runs across the frame: the front
    edge is then a side of the mask, and the measurements run on the turned image (the calibrated lift is mapped
    through the same turn, so the thickness stays the camera's)."""

    def __init__(self, direction, shape: tuple[int, int], front=None):
        import cv2

        height, width = shape[:2]
        dx, dy = float(direction[0]), float(direction[1])
        # getRotationMatrix2D turns the picture counter-clockwise by `angle`; a vector at image angle phi (y down)
        # comes out at phi - angle, and the push must come out pointing down (90)
        self.angle = float(np.degrees(np.arctan2(dy, dx))) - 90.
        if front is not None:
            # the front edge comes out horizontal, turned the way that leaves the push pointing down
            phi_e = float(np.degrees(np.arctan2(float(front[1]), float(front[0]))))
            phi_d = float(np.degrees(np.arctan2(dy, dx)))
            self.angle = min((phi_e, phi_e - 180.), key=lambda a: -np.sin(np.radians(phi_d - a)))
        centre = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(centre, self.angle, 1.)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        self.shape = (int(round(height * cos + width * sin)), int(round(height * sin + width * cos)))
        matrix[0, 2] += self.shape[1] / 2 - centre[0]
        matrix[1, 2] += self.shape[0] / 2 - centre[1]
        self.matrix = matrix
        self.inverse = cv2.invertAffineTransform(matrix)
        self.source_shape = (height, width)

    @property
    def trivial(self) -> bool:
        return abs(self.angle) < 2.

    def forward(self, point) -> tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        return (self.matrix[0, 0] * x + self.matrix[0, 1] * y + self.matrix[0, 2],
                self.matrix[1, 0] * x + self.matrix[1, 1] * y + self.matrix[1, 2])

    def backward(self, point) -> tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        return (self.inverse[0, 0] * x + self.inverse[0, 1] * y + self.inverse[0, 2],
                self.inverse[1, 0] * x + self.inverse[1, 1] * y + self.inverse[1, 2])

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.warpAffine(mask.astype(np.uint8), self.matrix, (self.shape[1], self.shape[0]),
                              flags=cv2.INTER_NEAREST) > 0

    def warp_gray(self, gray: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.warpAffine(gray, self.matrix, (self.shape[1], self.shape[0]), flags=cv2.INTER_LINEAR)

    def unwarp_mask(self, mask: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.warpAffine(mask.astype(np.uint8), self.inverse, (self.source_shape[1], self.source_shape[0]),
                              flags=cv2.INTER_NEAREST) > 0

    def lift(self, lift):
        """The calibrated lift in the turned frame."""
        return lambda point: self.forward(lift(self.backward(point)))

    def line_back(self, line) -> list[list[int]]:
        return [[int(round(v)) for v in self.backward(point)] for point in line]


def slab_edges(mask: np.ndarray, gray: np.ndarray, lift, contact: list[list[int]] | None = None,
               direction=None, floor: np.ndarray | None = None, strict_floor: bool = True) -> tuple[list[list[int]], dict]:
    """The top front edge of one sheet's slab, from its mask, and how it was found.

    direction: the unit image vector the sheet is pushed in (push_direction); the front is the end of the slab in
    that direction, and the measurements below run on the image turned so that it points down (Rotation). Without
    it the front is the lower outline (towards the camera). floor: the floor mask; when given, the share of floor
    just ahead of the front is recorded (info["floor_ahead"]) and, with strict_floor, must reach 60% (a
    container's flat top has a side face there instead; SAM3's floor mask often stops at a sheet's shadow, so the
    sheet a forklift is known to push is not refused for it).

    The edge is the lower outline of the mask between the front corners (floor_contact_edge) moved up by the height
    of the black front side face measured on the photo (side_face_height). The lower outline alone is not enough:
    a forklift holds the sheet a little off the floor, and the shadow under it reads as more side face (s_003:
    10 px of black where the camera gives 7 px for the sheet's thickness). What gives the edge its direction
    depends on that shadow (info["reference"]):
    - "floor_line": the black band is equally high at both ends of the front (within EVEN_PX), so the sheet is
      parallel to the lower outline, the cleanest line in the photo, and the edge is that outline moved up;
    - "top_rim": the band differs between the ends (one end held higher), so the outline follows the shadow, not
      the sheet, and the edge is the line through the rim measured at the two ends;
    - "floor_line_one_end": the top face shows at one end only, so evenness cannot be checked; parallel is assumed.
    lift(pixel) is the image position of the point one sheet thickness above the floor point seen at a pixel
    (calibrated camera): it bounds the search, and it is the whole estimate only when the photo shows no rim (a
    dark load on a dark sheet), which is less accurate."""
    if direction is not None:
        turn = Rotation(direction, mask.shape, front=front_direction(mask, direction))
        if not turn.trivial:
            edge_r, info = slab_edges(turn.warp_mask(mask), turn.warp_gray(gray), turn.lift(lift),
                                      floor=None if floor is None else turn.warp_mask(floor), strict_floor=strict_floor)
            info = {**info, "floor_contact_px": turn.line_back(info["floor_contact_px"]),
                    "rotation_deg": round(turn.angle, 1), "push_direction": [round(float(v), 3) for v in direction],
                    "turned_frame": {"edge_px": edge_r, "front_corners_px": info.get("front_corners_px")}}
            return turn.line_back(edge_r), info
    if contact is None:
        # in perspective the front edge of a sheet pushed across the view can lean up to about 45 degrees
        contact = floor_contact_edge(mask, max_angle_deg=50)
        if contact is None:
            raise RuntimeError("the sheet's mask has no straight lower outline to find its front edge from")
    if floor is not None:
        # a sheet lies on the floor. SAM3's LSP prompts also return the flat tops of containers and of stacks,
        # which have a side face below their front edge, not floor (most cam01 "LSP" masks, 2026-10-05).
        xs = np.arange(contact[0][0], contact[1][0] + 1)
        ys = np.round(np.interp(xs, [contact[0][0], contact[1][0]], [contact[0][1], contact[1][1]])).astype(int)
        ahead = np.mean([floor[np.clip(ys + d, 0, floor.shape[0] - 1), np.clip(xs, 0, floor.shape[1] - 1)].mean()
                         for d in (8, 12, 16)])
        if ahead < .6 and strict_floor:
            raise RuntimeError(f"no floor ahead of the sheet's front edge ({ahead:.0%} floor): not a sheet on the "
                               "floor, or the front is not where the push direction says")
    lifted = [[int(v) for v in lift(point)] for point in contact]
    thickness = float(np.mean([c[1] - e[1] for c, e in zip(contact, lifted)]))
    info = {"floor_contact_px": contact, "thickness_px": round(thickness, 1)}
    if floor is not None:
        info["floor_ahead"] = round(float(ahead), 2)
    if thickness < 2:
        # the lift does not go up this outline: the front side face is hidden (the sheet is pushed away from the
        # camera, or across the view at a grazing angle), so the outline is the top face's own edge
        ends = top_face_ends(mask, contact, 4.)
        edge = [[x, y] for x, y in contact]
        if ends is not None:
            edge = [[ends[0], int(round(line_y(contact, ends[0])))], [ends[1], int(round(line_y(contact, ends[1])))]]
        return edge, {"method": "outline_is_top_edge", "edge_observed": True, "top_face_ends_px": ends, **info}
    halves = side_face_height(mask, gray, contact, thickness)
    if halves is None:
        return lifted, {"method": "floor_contact_lifted_by_thickness", "edge_observed": False, **info}
    left, right = halves
    if left and right and abs(left[1] - right[1]) > EVEN_PX:
        reference = "top_rim"                               # height as a line through the two measured ends
        slope = (right[1] - left[1]) / (right[0] - left[0])
        heights = [left[1] + slope * (x - left[0]) for x, _ in contact]
    else:
        reference = "floor_line" if left and right else "floor_line_one_end"
        measured = [half for half in halves if half]
        heights = [sum(h * n for _, h, n in measured) / sum(n for _, _, n in measured)] * 2
    edge = [[x, int(round(y - h))] for (x, _), (_, y), h in zip(lifted, contact, heights)]
    # the sheet's extent is where its top face ends on the rim, not where the shadow ends on the floor (a sheet
    # held off the floor casts a smaller shadow); the side edges then sharpen the corners
    ends = top_face_ends(mask, edge, thickness)
    sides = side_edges(mask, gray, contact, edge, rim_ends=ends)
    for i, side in enumerate(("left", "right")):
        x = sides["corners_px"]["rim"][side] + (lifted[i][0] - contact[i][0])
        edge[i] = [int(x), int(round(line_y(edge, x)))]
    return edge, {"method": "side_face_top", "edge_observed": True, "reference": reference,
                  "side_face_px": [round(h, 1) for h in heights],
                  "rim_columns": [half[2] if half else 0 for half in halves],
                  "side_edges": {k: v for k, v in sides.items() if k != "corners_px"},
                  "front_corners_px": sides["corners_px"], "top_face_ends_px": ends, **info}


def trim_to_slab(mask: np.ndarray, rim: list[list[int]], thickness_px: float,
                 sides: dict | None = None) -> tuple[np.ndarray, int]:
    """The mask without the shadow beside the slab. SAM3 includes the shadow in a sheet's mask: under a sheet a
    forklift holds up, in front of and beside one lying on the floor. The slab ends at its contact line, the rim
    lowered by the camera's figure for the thickness, so what the mask has below that line is shadow, not sheet
    (the mask's own lower outline is the shadow's rim). Where a side's top edge was found (side_edges, "pair"),
    the side face's bottom edge is that line lowered by the same thickness, and what lies outside it is shadow
    too. Returns (trimmed mask, pixels removed)."""
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    keep = yy <= line_y(rim, xx) + thickness_px + 1
    for side, sign in (("left", 1), ("right", -1)):
        found = (sides or {}).get(side)
        if not found or not found.get("pair"):
            continue
        angle = np.radians(found["angle_deg"])
        if abs(np.sin(angle)) < 1e-3:
            continue
        x_top = found["top_edge_crossing_px"]
        y_bottom = line_y(rim, x_top) + thickness_px
        x_at = x_top + (y_bottom - yy) * np.cos(angle) / np.sin(angle)   # the bottom edge, row by row
        keep &= (xx - x_at) * sign >= -1
    trimmed = mask & keep
    return trimmed, int(mask.sum() - trimmed.sum())


def sheet_footprint(mask: np.ndarray, gray: np.ndarray, lift, direction=None) -> tuple[np.ndarray, dict]:
    """A sheet's mask trimmed of its shadow (slab_edges, then trim_to_slab), in the frame turned to the push
    direction when one is given. A mask without a straight front outline is returned as it is."""
    turn = Rotation(direction, mask.shape) if direction is not None else None
    if turn is not None and not turn.trivial:
        trimmed, info = sheet_footprint(turn.warp_mask(mask), turn.warp_gray(gray), turn.lift(lift))
        if "rim_px" in info:
            info["rim_px"] = turn.line_back(info["rim_px"])
        return turn.unwarp_mask(trimmed) & mask, info
    try:
        rim, info = slab_edges(mask, gray, lift)
    except RuntimeError:
        return mask, {"method": "untrimmed", "shadow_px_removed": 0}
    trimmed, removed = trim_to_slab(mask, rim, info["thickness_px"], info.get("side_edges"))
    return trimmed, {"method": info["method"], "rim_px": rim, "thickness_px": info["thickness_px"],
                     "shadow_px_removed": removed}


def trim_below_rim(mask: np.ndarray, rim: list[list[int]], lift, direction=None) -> tuple[np.ndarray, int]:
    """trim_to_slab for a sheet whose rim is known (the original sheet, its rim being the shared edge), with the
    thickness at the rim from the calibrated lift, in the frame turned to the push direction."""
    turn = Rotation(direction, mask.shape) if direction is not None else None
    if turn is not None and not turn.trivial:
        rim_r = [list(turn.forward(p)) for p in rim]
        lift_r = turn.lift(lift)
        thickness = float(np.mean([y - lift_r((x, y))[1] for x, y in rim_r]))
        trimmed, removed = trim_to_slab(turn.warp_mask(mask), rim_r, thickness)
        trimmed = turn.unwarp_mask(trimmed) & mask
        return trimmed, int(mask.sum() - trimmed.sum())
    thickness = float(np.mean([y - lift((x, y))[1] for x, y in rim]))
    return trim_to_slab(mask, rim, thickness)


FORK_GAP_PX = 20       # the forks show between the truck's mask and its load
MIN_TOUCH_PX = 50


def pushed_sheet(masks: dict[str, list[np.ndarray]], shape: tuple[int, int],
                 sheet_at=None) -> tuple[np.ndarray, np.ndarray] | None:
    """The LSP a forklift pushes and that forklift's body: the largest LSP mask that touches a forklift mask (or
    a cargo touching it), mostly outside it. SAM3 often takes the load into the forklift mask; masks that sit
    mostly inside it are removed first (as find_valid_frames does). None without such a pair. sheet_at: a pixel
    on the sheet to use (find_valid_frames knows which forklift pushes which sheet from the motion between
    samples; one frame cannot tell a pushed sheet from one parked beside a forklift): then that sheet, with the
    forklift nearest to it."""
    from scipy.ndimage import binary_dilation

    def touches(near, mask):
        return int(np.count_nonzero(near & mask)) >= MIN_TOUCH_PX

    wanted = None
    if sheet_at is not None:
        x, y = int(round(sheet_at[0])), int(round(sheet_at[1]))
        on = [m for m in masks.get("LSP", []) if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x]]
        if not on:
            def distance(m):
                ys, xs = np.nonzero(m)
                return float(np.hypot(xs.mean() - x, ys.mean() - y))
            on = sorted(masks.get("LSP", []), key=distance)[:1]
            if on and distance(on[0]) > 80:
                raise RuntimeError(f"no LSP mask at or near the given sheet position {sheet_at}")
        if not on:
            raise RuntimeError("SAM3 found no LSP mask")
        wanted = max(on, key=lambda m: int(m.sum()))
    best = None
    for forklift in masks.get("forklift", []):
        inside = [m for m in masks.get("LSP", []) + masks.get("cargo", []) if np.count_nonzero(m & forklift) > .5 * np.count_nonzero(m)]
        body = forklift & ~np.logical_or.reduce(inside) if inside else forklift
        if np.count_nonzero(body) < 1200:
            body = forklift
        near = binary_dilation(body, iterations=FORK_GAP_PX)
        carried = [c for c in masks.get("cargo", []) if touches(near, c)]
        carried_near = binary_dilation(np.logical_or.reduce(carried), iterations=12) if carried else None
        for lsp in masks.get("LSP", []):
            if wanted is not None and lsp is not wanted:
                continue
            if np.count_nonzero(lsp & ~body) <= .5 * np.count_nonzero(lsp):
                continue
            if touches(near, lsp) or (carried_near is not None and touches(carried_near, lsp)):
                if best is None or np.count_nonzero(lsp) > np.count_nonzero(best[0]):
                    best = (lsp, body)
    if best is None and wanted is not None and masks.get("forklift"):
        # the wanted sheet touches no forklift mask (SAM3 cut the forks): the nearest forklift pushes it
        ys, xs = np.nonzero(wanted)
        def gap(f):
            fy, fx = np.nonzero(f)
            return float(np.hypot(fx.mean() - xs.mean(), fy.mean() - ys.mean()))
        best = (wanted, min(masks["forklift"], key=gap))
    return best


def source_edge(masks: dict[str, list[np.ndarray]], image_size: tuple[int, int], gray: np.ndarray,
                lift, sheet_at=None) -> tuple[list[list[int]], dict]:
    """The top front edge of the loaded LSP's slab (slab_edges): the edge a second sheet shares with it.
    sheet_at: a pixel on the sheet to use (see pushed_sheet).

    The sheet is the one a forklift pushes (pushed_sheet), and its front is the end of the slab in the push
    direction, from the forklift's centroid to the sheet's: the shared edge, the sheet's centre and the forklift
    lie on one line. Without a forklift mask (cam01: SAM3 finds none behind the load) the loaded sheet is found
    by cargo overlap near the frame's centre and its front is taken towards the camera."""
    width, height = image_size
    floor = union(masks.get("floor", []), (height, width))
    pair = pushed_sheet(masks, (height, width), sheet_at)
    if pair is not None:
        lsp, body = pair
        direction = push_direction(lsp, body)
        edge, info = slab_edges(lsp, gray, lift, direction=direction, floor=floor, strict_floor=False)
        fy, fx = np.nonzero(body)
        return edge, {**info, "sheet_choice": "pushed_by_forklift", "lsp_bbox": bbox(lsp),
                      "forklift_centroid_px": [round(float(fx.mean()), 1), round(float(fy.mean()), 1)],
                      "push_direction": [round(float(v), 3) for v in direction]}
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
        contact = floor_contact_edge(mask)
        if contact is None:
            continue
        # a sheet lies on the floor. SAM3's LSP prompts also return the flat tops of containers and of stacks,
        # which have a side face below their front edge, not floor (most cam01 "LSP" masks, 2026-10-05).
        xs = np.arange(contact[0][0], contact[1][0] + 1)
        ys = np.round(np.interp(xs, [contact[0][0], contact[1][0]], [contact[0][1], contact[1][1]])).astype(int)
        below = np.mean([floor[np.clip(ys + d, 0, height - 1), xs].mean() for d in (8, 12, 16)])
        if below < .6:
            continue
        score = overlap - abs(box[3] - cargo_box[3])
        candidates.append((score, mask, box, contact))
    if not candidates:
        raise RuntimeError("SAM3 did not isolate a loaded source LSP with its front bottom edge on the floor; "
                           "supply --contact-edge")
    _, mask, box, contact = max(candidates, key=lambda item: item[0])
    edge, info = slab_edges(mask, gray, lift, contact)
    return edge, {**info, "sheet_choice": "cargo_near_centre", "push_direction": [0., 1.],
                  "cargo_bbox": cargo_box, "lsp_bbox": box}


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


def added_sheet(source: dict[str, list[np.ndarray]], candidate: dict[str, list[np.ndarray]], region: Image.Image,
                edge_px: list[list[int]], toward_px) -> tuple[int, np.ndarray | None]:
    """The number of new LSP instances in the candidate and the mask of the added sheet (None without one).

    The new sheet is its SAM3 instance on the new side of the shared edge's line. It is not "instance minus the
    original's mask": that mask also covers the original's side face and shadow below its top-face edge, pixels an
    adjoining new sheet hides, and removing them measured a correct sheet as short with a gap (s_003: 68% depth,
    0.31 m gap). The line cut also splits a new sheet that SAM3 merged with the original into one instance."""
    inside = np.asarray(region) > 0
    shape = inside.shape
    source_lsp = union(source.get("LSP", []), shape)
    # SAM3 can give different masks for unrelated parked LSPs even when their pixels are unchanged; only
    # instances with new pixels inside the edit region can be the added sheet.
    new = [mask for mask in candidate.get("LSP", []) if ((mask & ~source_lsp) & inside).sum() > 1500]
    if not new:
        return 0, None
    a, b = np.asarray(edge_px, dtype=float)
    normal = np.array([-(b - a)[1], (b - a)[0]])
    if float((np.asarray(toward_px, dtype=float) - a) @ normal) < 0:
        normal = -normal
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    new_side = (xx - a[0]) * normal[0] + (yy - a[1]) * normal[1] >= 0
    return len(new), max(new, key=lambda mask: int(mask.sum())) & new_side


def outline(mask: np.ndarray) -> list[list[int]]:
    """The mask's convex outline as a few corner points (four for a sheet seen in perspective)."""
    import cv2
    ys, xs = np.nonzero(mask)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    corners = cv2.approxPolyDP(hull, .02 * cv2.arcLength(hull, True), True)
    return [[int(x), int(y)] for x, y in corners.reshape(-1, 2)]


def verify_sheet_geometry(source: dict[str, list[np.ndarray]], candidate: dict[str, list[np.ndarray]],
                          region: Image.Image, points: np.ndarray, valid: np.ndarray, floor: np.ndarray,
                          edge_px: list[list[int]], toward_px, depth_to_width: float, gray: np.ndarray | None = None,
                          lift=None, direction=None) -> dict:
    """One new LSP inside the edit region (added_sheet), no cargo yet, and the new sheet's orientation, size and
    position equal to the original's on the floor plane (compare_sheets). With the candidate's grey image and the
    camera's lift, both sheets are measured without their shadows (sheet_footprint, trim_to_slab)."""
    shape = valid.shape
    inside = np.asarray(region) > 0
    count, sheet = added_sheet(source, candidate, region, edge_px, toward_px)
    result = {"new_lsp_instances": count}
    if sheet is not None and gray is not None and lift is not None:
        sheet, footprint = sheet_footprint(sheet, gray, lift, direction)
        result["new_sheet"] = footprint
    failures = []
    if count != 1:
        failures.append(f"SAM3 found {count} new LSP instances; expected one")
    new_cargo = union(candidate.get("cargo", []), shape) & ~union(source.get("cargo", []), shape)
    if (new_cargo & inside).sum() > 1000:
        failures.append("Cargo appeared before the cargo stage")
    if sheet is not None:
        old = sheet_at_edge(source.get("LSP", []), edge_px)
        if old is not None and lift is not None:
            old, result["old_sheet_shadow_px_removed"] = trim_below_rim(old, edge_px, lift, direction)
        result.update(compare_sheets(points, valid, floor, old, edge_px, toward_px, sheet, depth_to_width))
        failures = result["failures"] + failures
    result.update({"passed": not failures, "failures": failures})
    return result


def verify_cargo_masks(sheet: dict[str, list[np.ndarray]], candidate: dict[str, list[np.ndarray]],
                       polygon: list[list[int]], size: tuple[int, int], region: Image.Image,
                       min_height_px: int = 0) -> dict:
    """One new cargo standing on the sheet (polygon: the sheet's outline), high enough, with the exposed deck
    unchanged. New cargo is looked for inside the edit region only: outside it the candidate's pixels are the
    base's, yet SAM3 segments them a little differently from one image to the next, and one such stray mask far
    away used to become the "bottom" of the new cargo (s_003, 2026-10-05)."""
    target = polygon_mask(size, polygon)
    inside = np.asarray(region) > 0
    sheet_cargo = union(sheet.get("cargo", []), target.shape)
    edited_cargo = union(candidate.get("cargo", []), target.shape)
    added = edited_cargo & ~sheet_cargo & inside
    new_instances = [mask for mask in candidate.get("cargo", [])
                     if (mask & ~sheet_cargo & inside).sum() > 2000
                     and bbox(mask) is not None
                     and bbox(mask)[3] >= min(point[1] for point in polygon) - 20]
    # the new object as SAM3 outlines it, also where it hides the original load (that part is not "added")
    box = bbox(union(new_instances, target.shape))
    failures = []
    support = None
    if box is None or added.sum() < 2500:
        failures.append("SAM3 found no substantial new cargo")
        supported = False
    else:
        if box[3] - box[1] < min_height_px:
            failures.append(f"New cargo height {box[3] - box[1]}px < {min_height_px}px")
        ys, xs = np.nonzero(union(new_instances, target.shape)[max(0, box[3] - 12):box[3]])
        support = [int(np.median(xs)), max(0, box[3] - 12) + int(np.median(ys))]
        supported = bool(target[support[1], support[0]])
        if not supported:
            failures.append("New cargo bottom centre is outside the accepted LSP footprint")
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
            "cargo_supported": supported, "cargo_box_px": list(box) if box else None, "support_px": support,
            "lsp_mask_change_fraction": lsp_change}
