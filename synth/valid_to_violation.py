"""Turn a valid forklift frame into one reviewed two-LSP violation.

Run: python -m synth.valid_to_violation <valid frame> [--reference-map work/catalogue3d/reference_maps/<stem>/reference_map.json]
     [--contact-edge x1,y1,x2,y2] [--camera work/camera.json] [--calibration work/calibration.json]
     [--aesthetic-review]

Two edits on the real frame, each gated by code. For now geometry alone decides: the scored realism review by Astra
runs only with --aesthetic-review, and then it gates too.
1. one EMPTY LSP: the top front edge of the loaded sheet (the upper rim of its black front side face, measured
   on the photo; or given) is the whole guide: the image model adds an identical sheet along it and infers its depth from the scene. (The equal
   sheet projected with the camera calibration only bounds the edit region and places the cargo: its depth was not
   reliable enough to dictate.) The result is composited back (original pixels outside a bounded region and on
   SAM3-protected objects). Code checks the raw output's shape and drift, guide marks and the composite's
   integrity, then measures the new sheet against the original with SAM3 masks and a MoGe point map of the
   candidate: orientation, width, depth, gap and lateral offset on the floor plane (one new instance, no cargo).
   A failed candidate goes back to Astra with its measured deviations, as a reference image beside the prompt:
   the retry edits the source frame again and corrects them (editing the failed candidate itself made it worse).
   The original sheet is protected in the composite except its front band below the shared edge (its front side
   face and the shadow under it): the slab in front hides that band.
2. one CARGO on that sheet, starting every retry from the accepted sheet, with the same gates (support on the sheet,
   one instance, height, unchanged exposed deck). It works on the sheet as accepted (its SAM3 outline), not on the
   projected one: the image model chose that sheet's depth. The cargo's target box is a SKID footprint centred on
   that outline, sized by the LSP's and the SKID's real dimensions (reference map, else calibration and
   DEFAULT_SKID_SIZE_M), raised by the cargo's height.
References (forklift, LSP, SKID, cargo) are appearance only; by default Step A crops (work/refs/index.json),
or a --reference-map, e.g. from synth.catalogue_reference_map (view-matched renders of the 3D catalogue). When the
map gives the cargo's size_m, the cargo's target box is the projection of that 3D box. A failed gate never publishes.
Every run, accepted or not, also writes <output stem>_summary.png next to the output: the input on the left, the
output (or the candidate with the fewest failures) on the right, and the run statistics in English below (model
calls by purpose, attempts per stage, result, last scores and failures).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synth.vision_geometry import (MogePoints, Sam3Masks, added_sheet, below_line, edit_region, guide_marks, outline,
                                   preserve_outside, raw_drift, sheet_at_edge, source_edge, union, verify_cargo_masks,
                                   verify_sheet_geometry, verify_unchanged, zoom_pair)


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "work/refs/index.json"
CAMERA = ROOT / "work/camera.json"
CALIBRATION = ROOT / "work/calibration.json"
CLASSES = ("forklift", "LSP", "SKID", "cargo")
APPEARANCE_ONLY = ("The reference images show appearance only (colour, material, labels, film, texture); they are not "
                   "to scale and their viewpoint and position must not be copied. Size and position come from the "
                   "target coordinates given here.")
SCORES = ("edges", "lighting", "texture", "contact_shadow")
AESTHETIC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["scores", "issues", "accepted", "feedback"],
    "properties": {
        "scores": {"type": "object", "additionalProperties": False, "required": list(SCORES),
                   "properties": {key: {"type": "integer", "minimum": 1, "maximum": 5} for key in SCORES}},
        "issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["what", "where"],
            "properties": {"what": {"type": "string"}, "where": {"type": "string"}}}},
        "accepted": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
}
MIN_SCORE, MIN_MEAN_SCORE = 3, 3.5      # every score >= 3 and their mean >= 3.5; decided by code, not by Astra


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def step_a_references(index_path: Path) -> dict[str, Path]:
    """Select one existing Step A crop per class; never use it as a background."""
    entries = json.loads(index_path.read_text())
    selected: dict[str, Path] = {}
    for category in CLASSES:
        candidates = [entry for entry in entries if entry.get("class") == category]
        if category == "LSP":
            # Step A also contains tall stacks of LSPs. A single thin sheet is
            # the relevant shape reference for this two-sheet scenario.
            single_sheets = [entry for entry in candidates if (
                (entry["bbox"][2] - entry["bbox"][0]) /
                max(entry["bbox"][3] - entry["bbox"][1], 1) >= 2.5
                and int(entry.get("area_px", 0)) >= 5000
            )]
            if single_sheets:
                candidates = single_sheets
        candidates.sort(key=lambda entry: (float(entry.get("score", 0)) * max(int(entry.get("area_px", 0)), 1) ** 0.5), reverse=True)
        for entry in candidates:
            crop = ROOT / str(entry["crop"])
            if crop.is_file():
                selected[category] = crop
                break
        if category not in selected:
            raise RuntimeError(f"Step A lacks a usable {category} crop in {index_path}")
    return selected


def selected_references(args: argparse.Namespace) -> dict[str, Path]:
    """Use an explicit catalogue selection when supplied, otherwise Step A."""
    manifest = getattr(args, "reference_map", None)
    if manifest is None:
        return step_a_references(args.step_a_index.resolve())
    entries = json.loads(manifest.resolve().read_text())
    selected = {}
    for category in CLASSES:
        if category not in entries:
            raise RuntimeError(f"Reference map lacks {category}: {manifest}")
        item = entries[category]
        raw_path = item["path"] if isinstance(item, dict) else item
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise RuntimeError(f"Missing {category} reference: {path}")
        selected[category] = path.resolve()
    return selected


DEFAULT_SKID_SIZE_M = [1.2, 1.0, 0.13]      # the catalogue's single SKID, when the reference map gives no size
DEFAULT_CARGO_HEIGHT_M = 1.0


def reference_size(args: argparse.Namespace, category: str) -> list[float] | None:
    """A reference object's size (x along the shared edge, y across, z height) in metres, when the reference map
    records one."""
    manifest = getattr(args, "reference_map", None)
    if manifest is None:
        return None
    item = json.loads(manifest.resolve().read_text()).get(category)
    size = item.get("size_m") if isinstance(item, dict) else None
    return [float(v) for v in size] if size and len(size) == 3 else None


def run(command: list[str], *, log: Path) -> None:
    with log.open("w") as output:
        process = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, text=True, check=False)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log}")


def astra(codex: str, images: list[Path], schema: dict, prompt: str, work: Path, name: str) -> dict:
    schema_path = work / f"{name}.schema.json"
    response_path = work / f"{name}.json"
    schema_path.write_text(json.dumps(schema))
    command = [codex, "exec", "--ephemeral", "-m", "gpt-6-astra", "-s", "read-only",
               "--output-schema", str(schema_path)]
    for path in images:
        command += ["-i", str(path)]
    run(command + ["-o", str(response_path), prompt], log=work / f"{name}.log")
    try:
        return json.loads(response_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Astra did not return valid JSON in {response_path}") from error


def aesthetic_gate(review: dict) -> tuple[bool, list[str]]:
    """Code decides from Astra's scores: every score >= MIN_SCORE and their mean >= MIN_MEAN_SCORE (Astra's own
    `accepted` must agree). Returns (passed, failure texts for the next prompt)."""
    scores = review.get("scores") or {}
    values = [scores.get(key) for key in SCORES]
    failures = []
    if any(not isinstance(v, int) for v in values):
        failures.append("visual review returned no scores")
    else:
        failures += [f"{key} scored {scores[key]}/5" for key in SCORES if scores[key] < MIN_SCORE]
        if sum(values) / len(values) < MIN_MEAN_SCORE:
            failures.append(f"mean realism score {sum(values) / len(values):.1f}/5 < {MIN_MEAN_SCORE}")
    if review.get("accepted") is not True:
        failures.append("visual review did not accept")
    passed = not failures
    if not passed:   # listed defects only guide the retry; they do not fail an image whose scores pass
        failures += [f"{i.get('what', '')} ({i.get('where', '')})" for i in review.get("issues") or []]
    return passed, failures


class Camera:
    """Pinhole camera from camera.json (Blender convention: looks down -Z, +Y up), principal point from K_norm."""

    def __init__(self, path: Path):
        camera = json.loads(path.read_text())
        self.matrix = np.asarray(camera["matrix_world"], dtype=float)
        self.width, self.height = int(camera["width"]), int(camera["height"])
        K = camera["K_norm"]
        self.fx, self.fy = float(K[0][0]) * self.width, float(K[1][1]) * self.height
        self.cx, self.cy = float(K[0][2]) * self.width, float(K[1][2]) * self.height

    def unproject(self, pixel, plane_z: float) -> np.ndarray:
        direction = self.matrix[:3, :3] @ np.array([(pixel[0] - self.cx) / self.fx, -(pixel[1] - self.cy) / self.fy, -1.0])
        amount = (plane_z - self.matrix[2, 3]) / direction[2]
        if amount <= 0:
            raise RuntimeError("Camera geometry places LSP contact behind the camera")
        return self.matrix[:3, 3] + amount * direction

    def project(self, point: np.ndarray) -> tuple[int, int]:
        local = self.matrix[:3, :3].T @ (np.asarray(point, dtype=float) - self.matrix[:3, 3])
        return (round(self.cx + self.fx * local[0] / -local[2]), round(self.cy - self.fy * local[1] / -local[2]))

    def lift(self, pixel, height: float) -> tuple[int, int]:
        """Image position of the point `height` above the floor point seen at pixel."""
        return self.project(self.unproject(pixel, 0.) + np.array([0, 0, height]))


def make_geometry_guide(source: Path, contact_edge: list[list[int]], work: Path,
                        camera_path: Path | None = None, calibration_path: Path | None = None) -> Path:
    """Mark the original sheet's shared edge on the frame (the guide image) and project one calibrated LSP depth
    from it.

    The projected sheet is not shown to the image model: it bounds the edit region and, through geometry.json (which
    also keeps the new sheet's world corners on its top face), places the cargo's target box.
    """
    camera = Camera(camera_path or CAMERA)
    calibration = json.loads((calibration_path or CALIBRATION).read_text())
    plane_z = float(json.loads((ROOT / "config.json").read_text())["lsp_thickness_m"])
    a, b = [camera.unproject(point, plane_z) for point in contact_edge]
    across = b[:2] - a[:2]
    edge_width_m = float(np.linalg.norm(across))
    if edge_width_m < 0.5:
        raise RuntimeError("Implausibly short original LSP contact edge")
    normal = np.array([-across[1], across[0]]) / edge_width_m
    depth_m = float(calibration["lsp_measured_size_m"][1])
    candidates = []
    for sign in (-1, 1):
        delta = np.array([*(normal * depth_m * sign), 0.0])
        candidates.append((camera.project(a + delta), camera.project(b + delta), a + delta, b + delta))
    near_left, near_right, far_a, far_b = max(candidates, key=lambda c: (c[0][1] + c[1][1]) / 2)   # toward the camera
    shared_left, shared_right = tuple(contact_edge[0]), tuple(contact_edge[1])
    with Image.open(source) as original:
        guide = original.convert("RGB")
        if guide.size != (camera.width, camera.height):
            raise RuntimeError("Input dimensions do not match the camera calibration")
    draw = ImageDraw.Draw(guide)
    draw.line([shared_left, shared_right], fill=(0, 255, 70), width=5)
    draw.text((near_left[0], near_left[1] + 12), "GEOMETRY GUIDE ONLY - DO NOT RENDER LINES", fill=(255, 255, 255))
    path = work / "geometry_guide.png"
    guide.save(path)
    (work / "geometry.json").write_text(json.dumps({
        "shared_edge_px": [shared_left, shared_right],
        "new_lsp_near_edge_px": [near_left, near_right],
        "new_lsp_top_world_m": [[round(float(v), 4) for v in p] for p in (a, b, far_b, far_a)],
        "original_contact_edge_width_m": round(edge_width_m, 3),
        "new_lsp_length_m": depth_m,
        "source": str(source),
        "note": "Projection for the edit region and the cargo box only; the image model is given the shared edge",
    }, indent=2))
    return path


def sheet_quad(outline: list[list[int]], edge: list[list[int]]) -> np.ndarray:
    """The accepted sheet's four corners in the image, ordered rear-left, rear-right, front-right, front-left; the
    rear corners are the two nearest the shared edge."""
    import cv2
    points = np.asarray(outline, dtype=np.float32)
    epsilon = 2.
    while len(points) > 4 and epsilon < 60:
        points = cv2.approxPolyDP(np.asarray(outline, dtype=np.float32), epsilon, True).reshape(-1, 2)
        epsilon *= 1.5
    if len(points) != 4:
        xs, ys = np.asarray(outline)[:, 0], np.asarray(outline)[:, 1]
        points = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float32)
    (ax, ay), (bx, by) = edge
    distance = np.abs((points[:, 0] - ax) * (by - ay) - (points[:, 1] - ay) * (bx - ax)) / max(np.hypot(bx - ax, by - ay), 1e-6)
    rear = sorted(np.argsort(distance)[:2], key=lambda i: points[i][0])
    front = sorted([i for i in range(4) if i not in rear], key=lambda i: points[i][0])
    return points[[rear[0], rear[1], front[1], front[0]]]


def cargo_box_on_sheet(outline: list[list[int]], edge: list[list[int]], lsp_size_m: list[float],
                       skid_size_m: list[float], height_m: float, raise_) -> list[int]:
    """Image box of a cargo standing centred on the accepted sheet: the SKID's footprint on the sheet, placed by the
    homography from the sheet's real size (lsp_size_m: along the shared edge, across) to its image corners, then
    raised by height_m with raise_(pixel, metres) (the calibrated camera). No camera distance scale is involved:
    the sheet itself is the ruler."""
    import cv2
    quad = sheet_quad(outline, edge)
    w, d = float(lsp_size_m[0]), float(lsp_size_m[1])
    homography = cv2.getPerspectiveTransform(np.array([[0, 0], [w, 0], [w, d], [0, d]], np.float32), quad)
    sw, sd = float(skid_size_m[0]), float(skid_size_m[1])
    footprint = np.array([[(w - sw) / 2, (d - sd) / 2], [(w + sw) / 2, (d - sd) / 2],
                          [(w + sw) / 2, (d + sd) / 2], [(w - sw) / 2, (d + sd) / 2]], np.float32)
    bottom = cv2.perspectiveTransform(footprint.reshape(-1, 1, 2), homography).reshape(-1, 2)
    top = np.array([raise_(p, height_m) for p in bottom], dtype=float)
    points = np.vstack([bottom, top])
    return [int(np.floor(points[:, 0].min())), int(np.floor(points[:, 1].min())),
            int(np.ceil(points[:, 0].max())), int(np.ceil(points[:, 1].max()))]


def image_shape_ok(source: Path, candidate: Path) -> bool:
    """The image model's RAW output keeps the source aspect ratio and a usable size (a composite always would)."""
    with Image.open(source) as original, Image.open(candidate) as edited:
        source_ratio = original.width / original.height
        edited_ratio = edited.width / edited.height
        return (abs(source_ratio - edited_ratio) / source_ratio < 0.01
                and edited.width >= 1024 and edited.height >= 576)


def publish(candidate: Path, source: Path, output: Path) -> None:
    with Image.open(source) as original, Image.open(candidate) as edited:
        result = edited.convert("RGB")
        if result.size != original.size:
            result = result.resize(original.size, Image.Resampling.LANCZOS)
        result.save(output, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Valid CCTV image with one forklift-loaded LSP")
    parser.add_argument("--output", type=Path, help="Accepted PNG path; default work/out/<input>_violation.png")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--contact-edge",
                        help="Override the loaded LSP's top front edge: x1,y1,x2,y2 in source pixels")
    parser.add_argument("--reuse-sheet", type=Path, help="Reverify a previously generated empty LSP and continue with cargo")
    parser.add_argument("--reuse-cargo", type=Path, help="Reverify a previously generated cargo candidate after --reuse-sheet")
    parser.add_argument("--editor", choices=("codex", "api"), default="codex",
                        help="codex uses signed-in Astra + imagegen; api uses OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-image-2.5-sunburst")
    parser.add_argument("--camera", type=Path, default=CAMERA, help="camera.json of the input's camera")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION, help="calibration.json with lsp_measured_size_m")
    parser.add_argument("--aesthetic-review", action="store_true",
                        help="also gate on Astra's realism scores (skipped for now: geometry first)")
    parser.add_argument("--step-a-index", type=Path, default=INDEX)
    parser.add_argument("--reference-map", type=Path, help="Four selected reference images; overrides Step A index")
    parser.add_argument("--image-cli", type=Path, default=Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "skills/.system/imagegen/scripts/image_gen.py")
    parser.add_argument("--codex-bin", default="codex")
    return parser.parse_args()


def guided_edit(args: argparse.Namespace, images: list[Path], roles: list[str], prompt: str,
                output: Path, work: Path) -> None:
    """Generate a full candidate from at most five images; locality follows later."""
    if len(images) > 5:
        raise RuntimeError("Image editor accepts at most five references")
    instruction = "Use the supplied images in this order: " + "; ".join(roles) + ". " + prompt
    (work / f"{output.stem}.prompt.txt").write_text(instruction)
    if args.editor == "codex":
        if not output.is_relative_to(ROOT):
            raise RuntimeError("Codex image editor needs an output inside the workspace")
        command = [args.codex_bin, "exec", "--ephemeral", "-m", "gpt-6-astra",
                   "-s", "workspace-write", "-C", str(ROOT)]
        for image in images:
            command.extend(["-i", str(image)])
        command += ["-o", str(work / f"{output.stem}.astra.txt"),
                    "Use image_gen.imagegen to EDIT Image 1. Pass these image paths as referenced_image_paths "
                    "in the supplied order, transparent_background=false. " + instruction
                    + " Save the generated result as a nonempty PNG at " + str(output)
                    + ". Do not edit any other repository file or use Blender/3D compositing."]
    else:
        command = [sys.executable, str(args.image_cli), "edit", "--model", args.model,
                   "--quality", "high", "--size", "auto", "--prompt-file",
                   str(work / f"{output.stem}.prompt.txt"), "--output-format", "png"]
        for image in images:
            command.extend(["--image", str(image)])
        command.extend(["--out", str(output)])
    run(command, log=work / f"{output.stem}.imagegen.log")
    if not output.is_file() or not output.stat().st_size:
        raise RuntimeError(f"No image generated at {output}")


def gate(stage: str, base: Path, raw: Path | None, candidate: Path, region, protected, measured: dict,
         source: Path, polygon, ref: Path, args: argparse.Namespace, work: Path, attempt: int, describe: str) -> dict:
    """All code checks of one candidate and, with --aesthetic-review, the scored visual review on a close-up. raw
    is None for a reused candidate (only the composite exists)."""
    checks = {"masks": measured}
    if raw is not None:
        checks["raw_shape_ok"] = image_shape_ok(source, raw)
        checks["raw_drift"] = raw_drift(base, raw, region, protected)
    checks["guide_marks"] = guide_marks(base, candidate, region)
    checks["composite"] = verify_unchanged(base, candidate, region, protected)
    failures = list(measured["failures"])
    if raw is not None and not checks["raw_shape_ok"]:
        failures.append("the edited image changed the frame's aspect ratio or is too small; keep the full frame")
    for key in ("raw_drift", "guide_marks", "composite"):
        if key in checks:
            failures += checks[key]["failures"]
    zoom = zoom_pair(base, candidate, polygon, work / f"{stage}_zoom_{attempt:02d}.png")
    if not args.aesthetic_review:
        return {"checks": checks, "aesthetic": None, "failures": failures, "accepted": not failures}
    review = astra(args.codex_bin, [zoom, candidate, ref], AESTHETIC_SCHEMA,
                   f"Image 1 is a close-up: left BEFORE, right AFTER {describe}. Image 2 is the full frame after the "
                   "edit. Image 3 is an appearance reference. Score ONLY the realism of the added object in the "
                   "AFTER close-up, 1 (fake) to 5 (indistinguishable from the CCTV footage): edges (blending, no "
                   "halo or seam), lighting (matches the scene), texture (CCTV blur, noise, material like the "
                   "reference), contact_shadow (grounded, plausible shadow). List every visible defect in issues "
                   "with where it is. Do not judge geometry or object counts; code checks those. JSON only.",
                   work, f"{stage}_aesthetic_{attempt:02d}")
    visual_ok, visual_failures = aesthetic_gate(review)
    passed = not failures and visual_ok
    return {"checks": checks, "aesthetic": review, "failures": failures + visual_failures, "accepted": passed}


def summary_sheet(report: dict, path: Path, panel_height: int = 540) -> Path:
    """One image per run: the source on the left, the output on the right (the published image, else the candidate
    of the last stage with the fewest failures, marked NOT ACCEPTED), and below them the run statistics in English:
    model calls by purpose, attempts per stage, the result and the last failures."""
    from PIL import ImageFont

    def font(size: int, bold: bool = False):
        for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def panel(image_path: str | None, title: str) -> Image.Image:
        if image_path and Path(image_path).is_file():
            with Image.open(image_path) as image:
                tile = image.convert("RGB")
            tile = tile.resize((round(tile.width * panel_height / tile.height), panel_height), Image.Resampling.LANCZOS)
        else:
            tile = Image.new("RGB", (round(panel_height * 16 / 9), panel_height), (40, 40, 40))
            ImageDraw.Draw(tile).text((20, panel_height // 2), "no output image", font=font(28), fill=(220, 220, 220))
        out = Image.new("RGB", (tile.width, tile.height + 40), (255, 255, 255))
        out.paste(tile, (0, 40))
        ImageDraw.Draw(out).text((10, 8), title, font=font(24, True), fill=(20, 20, 20))
        return out

    stages = report.get("stages", {})
    best = None
    for stage in ("cargo", "sheet"):
        if stages.get(stage):
            best = min(stages[stage], key=lambda attempt: len(attempt.get("failures") or []))
            break
    accepted = bool(report.get("accepted"))
    right_path = report.get("output") if accepted else (best or {}).get("candidate")
    left = panel(report.get("input"), "Input (valid frame)")
    right = panel(right_path, "Output (violation)" if accepted else "Best candidate - NOT ACCEPTED")
    calls = report.get("calls", {})
    lines = [("Result: " + ("ACCEPTED" if accepted else f"NOT ACCEPTED (stopped at {report.get('stopped_at', 'error')})"),
              (20, 130, 60) if accepted else (190, 30, 30), True)]
    if report.get("error"):
        lines.append((f"Error: {report['error']}"[:160], (190, 30, 30), False))
    lines.append((f"Model calls: {sum(calls.values())} in total", (20, 20, 20), True))
    lines += [(f"  {purpose}: {n}", (20, 20, 20), False) for purpose, n in calls.items()]
    for stage, name in (("sheet", "Stage 1 - empty LSP"), ("cargo", "Stage 2 - cargo")):
        runs = stages.get(stage) or []
        if not runs:
            continue
        passed = sum(bool(r.get("accepted")) for r in runs)
        lines.append((f"{name}: {len(runs)} attempt(s), {passed} accepted", (20, 20, 20), True))
        shape = (runs[-1].get("checks") or {}).get("masks") or {}
        if "angle_deg" in shape:
            lines.append((f"  last LSP vs original: angle {shape['angle_deg']:+.1f} deg, width {shape['width_ratio']:.0%}, "
                          f"depth {shape['depth_ratio']:.0%}, gap {shape['gap_m']:.2f} m, lateral {shape['lateral_m']:.2f} m",
                          (20, 20, 20), False))
        scores = (runs[-1].get("aesthetic") or {}).get("scores") or {}
        if scores:
            lines.append(("  last realism scores: " + ", ".join(f"{k} {v}/5" for k, v in scores.items()), (20, 20, 20), False))
        for failure in (runs[-1].get("failures") or [])[:3]:
            lines.append((f"  last failure: {failure}"[:150], (150, 70, 0), False))
    refs = report.get("reference_map") or "Step A crops"
    lines.append((f"References: {refs}"[:150], (90, 90, 90), False))
    if report.get("started_at") and report.get("finished_at"):
        lines.append((f"Run time: {report['finished_at'] - report['started_at']:.0f} s"
                      f"   Editor: {report.get('editor')}   Input: {Path(report['input']).name}", (90, 90, 90), False))
    width = left.width + right.width + 20
    text_h = 30 + 30 * len(lines)
    sheet = Image.new("RGB", (width, left.height + text_h), (255, 255, 255))
    sheet.paste(left, (0, 0))
    sheet.paste(right, (left.width + 20, 0))
    draw = ImageDraw.Draw(sheet)
    y = left.height + 15
    for text, colour, bold in lines:
        draw.text((14, y), text, font=font(22, bold), fill=colour)
        y += 30
    sheet.save(path)
    return path


def count_call(report: dict, purpose: str) -> None:
    report.setdefault("calls", {})
    report["calls"][purpose] = report["calls"].get(purpose, 0) + 1


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "work/huggingface"))
    source = args.input.resolve()
    if not source.is_file():
        raise RuntimeError(f"Input image does not exist: {source}")
    if args.max_attempts < 1:
        raise RuntimeError("--max-attempts must be positive")
    if args.editor == "api" and (not args.image_cli.is_file() or not os.environ.get("OPENAI_API_KEY")):
        raise RuntimeError("--editor api requires the imagegen CLI and OPENAI_API_KEY")
    if (args.editor == "codex" or args.aesthetic_review) and not shutil.which(args.codex_bin):
        raise RuntimeError(f"Codex CLI is unavailable: {args.codex_bin}")
    camera_path = getattr(args, "camera", None) or CAMERA
    calibration_path = getattr(args, "calibration", None) or CALIBRATION
    refs = selected_references(args)
    ref_name = "catalogue" if getattr(args, "reference_map", None) else "Step A"
    output = (args.output or ROOT / "work/out" / f"{source.stem}_violation.png").resolve()
    if output.exists():
        raise RuntimeError(f"Output already exists; choose --output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = ROOT / "work/out" / f"{output.stem}_vision_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    work.mkdir(parents=True)
    report_path = work / "report.json"
    report = {"input": str(source), "input_sha256": sha256(source), "camera": str(camera_path),
              "step_a_index": str(args.step_a_index.resolve()),
              "reference_map": str(args.reference_map.resolve()) if getattr(args, "reference_map", None) else None,
              "references": {key: str(value) for key, value in refs.items()},
              "editor": args.editor, "aesthetic_review": args.aesthetic_review,
              "stages": {"sheet": [], "cargo": []}, "accepted": False,
              "calls": {}, "started_at": time.time()}
    editor_name = "Astra + imagegen" if args.editor == "codex" else f"API {args.model}"

    def save_report() -> None:
        report_path.write_text(json.dumps(report, indent=2))

    try:
        segmenter = Sam3Masks(ROOT / "work/vision_cache/sam3")
        source_masks = segmenter.get(source, ("forklift", "LSP", "cargo", "floor"))
        with Image.open(source) as image:
            size = image.size
            gray = np.asarray(image.convert("L"))
        thickness = float(json.loads((ROOT / "config.json").read_text())["lsp_thickness_m"])
        if args.contact_edge:
            values = [int(value.strip()) for value in args.contact_edge.split(",")]
            if (len(values) != 4 or values[0] >= values[2]
                    or any(not 0 <= value < size[index % 2] for index, value in enumerate(values))):
                raise RuntimeError("--contact-edge must be x1,y1,x2,y2 within input dimensions")
            edge = [[values[0], values[1]], [values[2], values[3]]]
            anchor_info = {"method": "manual_override", "edge_observed": False}
        else:
            camera = Camera(camera_path)
            edge, anchor_info = source_edge(source_masks, size, gray, lambda pixel: camera.lift(pixel, thickness))
        report["anchor"] = {"contact_edge_px": edge, **anchor_info}
        save_report()
        guide = make_geometry_guide(source, edge, work, camera_path=camera_path, calibration_path=calibration_path)
        geometry = json.loads((work / "geometry.json").read_text())
        expected_width, calibrated_depth = (float(v) for v in
                                            json.loads(calibration_path.read_text())["lsp_measured_size_m"][:2])
        measured_width = float(geometry["original_contact_edge_width_m"])
        if not 0.75 * expected_width <= measured_width <= 1.25 * expected_width:
            raise RuntimeError(f"Source LSP edge width {measured_width:.2f}m differs from the calibrated "
                               f"{expected_width:.2f}m by over 25%; inspect the anchor or use --contact-edge")
        polygon = [*geometry["shared_edge_px"], *reversed(geometry["new_lsp_near_edge_px"])]
        report["geometry_guide"] = str(guide)
        report["target_polygon_px"] = polygon
        save_report()
        protected = [union(source_masks.get(key, []), (size[1], size[0]))
                     for key in ("forklift", "cargo", "LSP")]
        # the loaded sheet's front side face and the shadow under it lie below the shared edge, between its front
        # corners; the new sheet in front hides them, so they are free to paint (everything else of the original
        # stays pixel for pixel: its side faces beyond the corners, and other sheets further down the frame)
        loaded = sheet_at_edge(source_masks.get("LSP", []), edge)
        if loaded is not None:
            columns = np.arange(size[0])
            span = (columns >= min(edge[0][0], edge[1][0])) & (columns <= max(edge[0][0], edge[1][0]))
            protected[2] &= ~(below_line(loaded, edge, toward=np.mean(geometry["new_lsp_near_edge_px"], axis=0)) & span)

        # ---- stage 1: one empty LSP ----
        sheet_region = edit_region(size, polygon, "sheet")
        # the floor every candidate keeps from the source: free floor outside the edit region
        floor = (union(source_masks.get("floor", []), (size[1], size[0])) & ~np.logical_or.reduce(protected)
                 & (np.asarray(sheet_region) == 0))
        toward = np.mean(geometry["new_lsp_near_edge_px"], axis=0).round().astype(int).tolist()
        moge = MogePoints(ROOT / "work/vision_cache/moge")
        accepted_sheet = sheet_masks = rejected = None
        feedback = ""
        for attempt in ([0] if getattr(args, "reuse_sheet", None) else range(1, args.max_attempts + 1)):
            raw = None
            if attempt == 0:
                candidate = args.reuse_sheet.resolve()
                if not candidate.is_file():
                    raise RuntimeError(f"--reuse-sheet image is missing: {candidate}")
            else:
                raw = work / f"sheet_raw_{attempt:02d}.png"
                candidate = work / f"sheet_{attempt:02d}.png"
                prompt = ("Add exactly one EMPTY LSP on the floor directly in front of the loaded LSP, on the side of "
                          "its front edge that is nearer the camera. The green line in Image 5 marks that front edge, "
                          f"from {edge[0]} to {edge[1]} in source pixels. The new sheet is identical to the loaded "
                          "one: its rear edge lies along the green line over its full width, and it has the same "
                          "width, depth, thickness and orientation (its sides parallel to the loaded sheet's), in the "
                          "scene's perspective. Do not add cargo. Match the "
                          f"{ref_name} LSP appearance and the CCTV texture. Keep the original forklift, cargo and LSP "
                          "intact. The green line is a mark only and must not appear in the photograph. "
                          + APPEARANCE_ONLY + " " + feedback)
                count_call(report, f"image edit - add empty LSP ({editor_name})")
                # a retry shows Astra the rejected attempt in place of the SKID reference (five images at most)
                fourth = ((refs["SKID"], f"Image 4 {ref_name} SKID preservation reference") if rejected is None else
                          (rejected, "Image 4 the rejected previous attempt at this edit (a reference, not the base)"))
                guided_edit(args, [source, refs["forklift"], refs["LSP"], fourth[0], guide],
                            ["Image 1 original valid source", f"Image 2 {ref_name} forklift appearance",
                             f"Image 3 {ref_name} LSP appearance", fourth[1], "Image 5 geometry guide"],
                            prompt, raw, work)
                preserve_outside(source, raw, candidate, sheet_region, protected)
            candidate_masks = segmenter.get(candidate, ("LSP", "cargo"))
            points, valid = moge.get(candidate)
            measured = verify_sheet_geometry(source_masks, candidate_masks, sheet_region, points, valid, floor, edge,
                                             toward, calibrated_depth / measured_width)
            if args.aesthetic_review:
                count_call(report, "visual review - LSP realism (Astra)")
            result = gate("sheet", source, raw, candidate, sheet_region, protected, measured, source, polygon,
                          refs["LSP"], args, work, attempt, "adding one empty LSP")
            report["stages"]["sheet"].append({"number": attempt, "candidate": str(candidate), **result})
            save_report()
            print(f"LSP {'reuse' if attempt == 0 else f'{attempt}/{args.max_attempts}'}: "
                  f"{'PASS' if result['accepted'] else 'FAIL'} — " + "; ".join(result["failures"]), flush=True)
            if result["accepted"]:
                accepted_sheet, sheet_masks = candidate, candidate_masks
                break
            rejected = candidate
            feedback = ("Image 4 is a previous attempt at this edit. Code measured its new sheet against the loaded "
                        "one on the floor plane (SAM3 masks, MoGe depth) and rejected it: "
                        + "; ".join(result["failures"]) + ". Edit Image 1 again and correct exactly these "
                        "deviations; keep what Image 4 already has right.")
        if accepted_sheet is None:
            report["stopped_at"] = "sheet_geometry_or_realism"
            return 2
        report["accepted_sheet"] = str(accepted_sheet)
        # the sheet as the image model drew it; the projected polygon only suggested where
        sheet = added_sheet(source_masks, sheet_masks, sheet_region, edge, toward)[1]
        sheet_outline = outline(sheet)
        report["accepted_sheet_outline_px"] = sheet_outline
        save_report()

        # ---- stage 2: one cargo on the new sheet ----
        # sized by the real LSP and SKID, not by the original load: the accepted sheet is the ruler
        lsp_size = reference_size(args, "LSP") or [expected_width, calibrated_depth]
        skid_size = reference_size(args, "SKID") or DEFAULT_SKID_SIZE_M
        cargo_size = reference_size(args, "cargo")
        cargo_height = cargo_size[2] if cargo_size else DEFAULT_CARGO_HEIGHT_M
        camera = Camera(camera_path)
        cargo_box = cargo_box_on_sheet(sheet_outline, edge, lsp_size, skid_size, cargo_height,
                                       lambda pixel, metres: camera.project(camera.unproject(pixel, thickness) + np.array([0, 0, metres])))
        report["cargo_target"] = {"method": "skid_footprint_on_sheet", "lsp_size_m": lsp_size[:2],
                                  "skid_size_m": skid_size[:2], "height_m": cargo_height, "box_px": cargo_box}
        min_height = round((cargo_box[3] - cargo_box[1]) * 0.7)
        # The closer new cargo may correctly occlude part of the original load.
        # Preserve original cargo everywhere outside that planned occlusion box.
        occlusion = (slice(max(0, cargo_box[1]), cargo_box[3]), slice(max(0, cargo_box[0]), cargo_box[2]))
        original_lsp_protected = protected[2].copy(); original_lsp_protected[occlusion] = False
        original_cargo_protected = protected[1].copy(); original_cargo_protected[occlusion] = False
        cargo_protected = [protected[0], original_lsp_protected, original_cargo_protected]
        cargo_region = edit_region(size, sheet_outline, "cargo")
        for attempt in ([0] if getattr(args, "reuse_cargo", None) else range(1, args.max_attempts + 1)):
            raw = None
            if attempt == 0:
                candidate = args.reuse_cargo.resolve()
                if not candidate.is_file():
                    raise RuntimeError(f"--reuse-cargo image is missing: {candidate}")
            else:
                raw = work / f"cargo_raw_{attempt:02d}.png"
                candidate = work / f"cargo_{attempt:02d}.png"
                prompt = (f"Add exactly ONE UPRIGHT, FULL-SIZE cargo like Image 4 on the new empty LSP, standing on a "
                          "SKID like Image 3. Do not make a small parcel or a flat stack. Its footprint is the SKID's "
                          f"({skid_size[0]:.1f} x {skid_size[1]:.1f} m on a {lsp_size[0]:.1f} x {lsp_size[1]:.1f} m LSP), "
                          f"centred on the new LSP, and its approximate bounding box is {cargo_box} in the "
                          f"{size[0]}x{size[1]} source image, with its bottom on the deck. Keep the cargo fully within "
                          "the LSP footprint at the bottom, leave the two LSP junction side ends visible, and match "
                          "lighting and contact shadow. The nearer cargo may naturally occlude the load behind it. "
                          "Render a single coherent cargo surface without a triangular patch or dark strip at its "
                          "bottom. Do not move either LSP, the forklift or the background. The new LSP's outline in "
                          f"source pixels: {sheet_outline}. " + APPEARANCE_ONLY + " " + feedback)
                count_call(report, f"image edit - add cargo ({editor_name})")
                guided_edit(args, [accepted_sheet, source, refs["SKID"], refs["cargo"], guide],
                            ["Image 1 geometry-accepted empty LSP", "Image 2 original valid source",
                             f"Image 3 {ref_name} SKID appearance", f"Image 4 {ref_name} cargo appearance",
                             "Image 5 geometry guide"], prompt, raw, work)
                preserve_outside(accepted_sheet, raw, candidate, cargo_region, cargo_protected)
            candidate_masks = segmenter.get(candidate, ("LSP", "cargo"))
            measured = verify_cargo_masks(sheet_masks, candidate_masks, sheet_outline, size, cargo_region,
                                          min_height_px=min_height)
            if args.aesthetic_review:
                count_call(report, "visual review - cargo realism (Astra)")
            result = gate("cargo", accepted_sheet, raw, candidate, cargo_region, cargo_protected, measured, source,
                          sheet_outline, refs["cargo"], args, work, attempt, "adding one cargo on the new LSP")
            report["stages"]["cargo"].append({"number": attempt, "candidate": str(candidate), **result})
            save_report()
            print(f"Cargo {'reuse' if attempt == 0 else f'{attempt}/{args.max_attempts}'}: "
                  f"{'PASS' if result['accepted'] else 'FAIL'} — " + "; ".join(result["failures"]), flush=True)
            if result["accepted"]:
                publish(candidate, source, output)
                report.update({"accepted": True, "output": str(output), "output_sha256": sha256(output)})
                print(f"Accepted image: {output}\nReview report: {report_path}")
                return 0
            feedback = "Fix these failures: " + "; ".join(result["failures"])
        report["stopped_at"] = "cargo_support_or_realism"
        return 2
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["finished_at"] = time.time()
        # one picture per run: input | output, with the run statistics underneath
        summary = summary_sheet(report, work / "summary.png")
        report["summary_image"] = str(output.with_name(f"{output.stem}_summary.png"))
        shutil.copyfile(summary, report["summary_image"])
        save_report()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
