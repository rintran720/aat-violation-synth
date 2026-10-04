"""Turn a cam01 valid forklift frame into one reviewed two-LSP violation.

GPT-6 Astra plans, calls imagegen, and reviews each attempt through Codex CLI.
Step A's SAM3 crop index supplies object references. A failed review never
publishes a final image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synth.vision_geometry import (Sam3Masks, edit_region, moge_floor_diagnostic,
                                   bbox, preserve_outside, source_edge, union,
                                   verify_cargo_masks, verify_sheet_masks, verify_unchanged)


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "work/refs/index.json"
CAMERA = ROOT / "work/camera.json"
CALIBRATION = ROOT / "work/calibration.json"
CLASSES = ("forklift", "LSP", "SKID", "cargo")
PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["valid_source", "reason", "source_loaded_lsp_count", "contact_edge_px", "edit_prompt"],
    "properties": {
        "valid_source": {"type": "boolean"},
        "reason": {"type": "string"},
        "source_loaded_lsp_count": {"type": "integer"},
        "contact_edge_px": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "edit_prompt": {"type": "string"},
    },
}
REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["accepted", "checks", "landmarks", "feedback", "revised_prompt"],
    "properties": {
        "accepted": {"type": "boolean"},
        "checks": {
            "type": "object",
            "additionalProperties": False,
            "required": ["two_loaded_lsps", "no_extra_lsp", "same_dimensions", "edge_contact", "cargo_supported", "natural_composite", "background_preserved"],
            "properties": {key: {"type": "boolean"} for key in (
                "two_loaded_lsps", "no_extra_lsp", "same_dimensions", "edge_contact",
                "cargo_supported", "natural_composite", "background_preserved"
            )},
        },
        "landmarks": {
            "type": "object", "additionalProperties": False,
            "required": ["original_front_edge", "new_rear_edge", "new_front_edge", "both_junction_ends_visible", "measurement_note"],
            "properties": {
                "original_front_edge": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "new_rear_edge": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "new_front_edge": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "both_junction_ends_visible": {"type": "boolean"},
                "measurement_note": {"type": "string"},
            },
        },
        "feedback": {"type": "string"},
        "revised_prompt": {"type": "string"},
    },
}
SHEET_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["accepted", "checks", "landmarks", "feedback", "revised_prompt"],
    "properties": {
        "accepted": {"type": "boolean"},
        "checks": {"type": "object", "additionalProperties": False,
                   "required": ["one_new_empty_lsp", "no_extra_lsp", "same_dimensions", "edge_contact",
                                "grounded", "natural_composite", "background_preserved"],
                   "properties": {key: {"type": "boolean"} for key in
                                  ("one_new_empty_lsp", "no_extra_lsp", "same_dimensions", "edge_contact",
                                   "grounded", "natural_composite", "background_preserved")}},
        "landmarks": REVIEW_SCHEMA["properties"]["landmarks"],
        "feedback": {"type": "string"},
        "revised_prompt": {"type": "string"},
    },
}
AESTHETIC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["accepted", "natural_edges", "realistic_lighting", "feedback"],
    "properties": {"accepted": {"type": "boolean"}, "natural_edges": {"type": "boolean"},
                   "realistic_lighting": {"type": "boolean"}, "feedback": {"type": "string"}},
}


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


def base_prompt(plan: dict, refs: dict[str, Path]) -> str:
    edge = plan["contact_edge_px"]
    if len(edge) != 2 or any(len(point) != 2 for point in edge):
        raise RuntimeError("Astra did not provide a two-point LSP contact edge")
    return (
        "The real valid CCTV source MUST remain the background. "
        "Add exactly ONE dark thin LSP and exactly ONE cargo container directly "
        "ahead of the existing loaded LSP in the forklift travel direction. "
        "Keep the original forklift, original loaded LSP, stationary cargo, "
        "SKIDs, people and warehouse unchanged. Two loaded LSPs total in the "
        "forklift train; no third or empty intermediate sheet. The two LSPs "
        "are identical in real length, width and thickness. Their touching "
        f"cross-edge is one shared line segment near {edge[0]} to {edge[1]} "
        "in source-image pixels, so they have equal projected width exactly at "
        "contact. Only the nearer edge may widen gradually from perspective. "
        "Both sheets lie on the floor, cargo sits fully on the new sheet, "
        "edges are complete and naturally blended with CCTV grain and shadow. "
        "Use Step A crops only as appearance references. Do not add a SKID. "
        + str(plan["edit_prompt"])
    )


def repair_prompt(plan: dict, review: dict) -> str:
    return (
        "This is a LOCAL CORRECTION of an existing candidate, not a new scene. "
        "Produce exactly two loaded LSPs in the forklift train, including the original. "
        "Remove any extra empty LSP if present; do not duplicate original objects. "
        "Keep the candidate's good appearance and layout, with Image 2 as the true original "
        "for restoring unchanged source objects. Fix these specific review failures: "
        + str(review.get("feedback", "")) + " Guidance from Astra: "
        + str(review.get("revised_prompt", ""))
        + " Measured failures: " + "; ".join(review.get("measured_failures", [])) + ". "
        + " The added LSP is IN FRONT of the existing LSP, toward the camera. "
        "The added LSP rear edge must exactly share both endpoints of the original LSP front edge "
        f"{plan['contact_edge_px']}. Use the geometry guide (Image 5) for that edge and the "
        "near edge; the colored lines and labels must not appear in the photograph. "
        "At the left and right side of the loads, leave a tiny visible portion of the sheet "
        "junction so contact and equal width can be checked. No broad empty strip or third LSP. "
        "Ignore any wording that reverses the train direction."
    )


def sheet_prompt(plan: dict, feedback: str = "") -> str:
    return (
        "Edit the valid source to add EXACTLY ONE EMPTY LSP directly in front of the existing "
        "loaded LSP, toward the camera. DO NOT ADD CARGO YET. This is a temporary geometry stage. "
        "Preserve the original forklift, original loaded LSP and cargo, all other objects and background. "
        "The new LSP must match the existing LSP's real length, width and thickness. Its rear edge "
        f"must touch the original LSP front edge at both endpoints {plan['contact_edge_px']}, "
        "with zero gap, overlap or vertical step. Keep both junction endpoints plainly visible. "
        "Use the geometry guide if supplied, but do not render its colored lines or labels. "
        "Match CCTV blur, lighting, texture and floor-contact shadow. Step A crops are appearance "
        "references only. " + feedback
    )


def cargo_prompt(review: dict | None = None) -> str:
    feedback = "" if review is None else (
        str(review.get("feedback", "")) + " " + "; ".join(review.get("measured_failures", [])) + " "
        + str(review.get("revised_prompt", "")))
    return (
        "Image 1 contains a GEOMETRY-ACCEPTED empty LSP in front of the original loaded LSP. "
        "Add exactly ONE cargo container onto that empty LSP. Do not add, remove, resize or move "
        "either LSP, the original forklift or any other source object. Keep the new cargo inside "
        "its LSP footprint with plausible contact shadow; keep BOTH endpoints of the shared LSP "
        "edge visible beside the cargo. Preserve the accepted LSP geometry and source background "
        "pixel positions. Match the Step A cargo appearance and CCTV lighting. " + feedback
    )


def make_geometry_guide(source: Path, contact_edge: list[list[int]], work: Path) -> Path:
    """Project one measured LSP depth from the original sheet's shared edge.

    This is an editing guide, not a claim that imagegen follows its pixels.
    The two sheets use the same shared endpoints by construction.
    """
    camera = json.loads(CAMERA.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    matrix = np.asarray(camera["matrix_world"], dtype=float)
    width, height = int(camera["width"]), int(camera["height"])
    fx = float(camera["K_norm"][0][0]) * width
    fy = float(camera["K_norm"][1][1]) * height
    cx, cy = width / 2, height / 2
    plane_z = float(json.loads((ROOT / "config.json").read_text())["lsp_thickness_m"])

    def unproject(pixel: list[int]) -> np.ndarray:
        direction = matrix[:3, :3] @ np.array([(pixel[0] - cx) / fx,
                                                 -(pixel[1] - cy) / fy, -1.0])
        amount = (plane_z - matrix[2, 3]) / direction[2]
        if amount <= 0:
            raise RuntimeError("Camera geometry places LSP contact behind the camera")
        return matrix[:3, 3] + amount * direction

    def project(point: np.ndarray) -> tuple[int, int]:
        local = matrix[:3, :3].T @ (point - matrix[:3, 3])
        return (round(cx + fx * local[0] / -local[2]),
                round(cy - fy * local[1] / -local[2]))

    a, b = [unproject(point) for point in contact_edge]
    across = b[:2] - a[:2]
    edge_width_m = float(np.linalg.norm(across))
    if edge_width_m < 0.5:
        raise RuntimeError("Implausibly short original LSP contact edge")
    normal = np.array([-across[1], across[0]]) / edge_width_m
    depth_m = float(calibration["lsp_measured_size_m"][1])
    candidates = []
    for sign in (-1, 1):
        delta = np.array([*(normal * depth_m * sign), 0.0])
        near_a, near_b = project(a + delta), project(b + delta)
        candidates.append((near_a, near_b))
    near_left, near_right = max(candidates, key=lambda pair: (pair[0][1] + pair[1][1]) / 2)
    shared_left, shared_right = tuple(contact_edge[0]), tuple(contact_edge[1])
    with Image.open(source) as original:
        guide = original.convert("RGB")
        if guide.size != (width, height):
            raise RuntimeError("Input dimensions do not match the Step A camera calibration")
    draw = ImageDraw.Draw(guide)
    draw.line([shared_left, shared_right], fill=(0, 255, 70), width=5)
    draw.line([shared_left, near_left, near_right, shared_right], fill=(255, 170, 0), width=5)
    draw.text((near_left[0], near_left[1] + 12), "GEOMETRY GUIDE ONLY - DO NOT RENDER LINES", fill=(255, 255, 255))
    path = work / "geometry_guide.png"
    guide.save(path)
    (work / "geometry.json").write_text(json.dumps({
        "shared_edge_px": [shared_left, shared_right],
        "new_lsp_near_edge_px": [near_left, near_right],
        "original_contact_edge_width_m": round(edge_width_m, 3),
        "new_lsp_length_m": depth_m,
        "source": str(source),
        "note": "Projection guide only; final image pixels require visual verification",
    }, indent=2))
    return path


def image_edit_api(image_cli: Path, source: Path, previous: Path | None, refs: dict[str, Path],
                   prompt: str, output: Path, work: Path, model: str, guide: Path) -> None:
    prompt_path = work / f"{output.stem}.prompt.txt"
    roles = ["Image 1: original valid source and background"] if previous is None else [
        "Image 1: previous candidate to correct", "Image 2: original valid source for background restoration"]
    categories = CLASSES if previous is None else ("LSP", "cargo")
    for category in categories:
        roles.append(f"Image {len(roles) + 1}: Step A {category} appearance reference ({refs[category].name})")
    if previous:
        roles.append(f"Image {len(roles) + 1}: mathematical LSP geometry guide; do not render colored lines")
    prompt = ". ".join(roles) + ". " + prompt
    prompt_path.write_text(prompt)
    command = [sys.executable, str(image_cli), "edit", "--model", model,
               "--quality", "high", "--size", "auto", "--prompt-file", str(prompt_path),
               "--output-format", "png", "--image", str(previous or source)]
    if previous:
        command += ["--image", str(source)]
    # image_gen accepts at most five referenced images. A retry uses the
    # previous candidate, source, LSP crop, cargo crop and geometry guide.
    for category in categories:
        command += ["--image", str(refs[category])]
    if previous:
        command += ["--image", str(guide)]
    command += ["--out", str(output)]
    run(command, log=work / f"{output.stem}.imagegen.log")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Image editor returned no image: {output}")


def image_edit_codex(codex: str, source: Path, previous: Path | None,
                     refs: dict[str, Path], prompt: str, output: Path, work: Path,
                     guide: Path) -> None:
    if not output.is_relative_to(ROOT):
        raise RuntimeError("Codex image editor requires attempts inside the repository workspace")
    images = [source] if previous is None else [previous, source]
    roles = ["Image 1: original valid CCTV source and ONLY background"] if previous is None else [
        "Image 1: previous candidate to correct", "Image 2: original valid source for background restoration"]
    categories = CLASSES if previous is None else ("LSP", "cargo")
    for category in categories:
        images.append(refs[category])
        roles.append(f"Image {len(images)}: Step A {category} appearance reference")
    if previous:
        images.append(guide)
        roles.append(f"Image {len(images)}: mathematical LSP guide; use its shared edge and sheet outline but DO NOT render its colored lines or text")
    instruction = (
        "Use the available image_gen.imagegen tool to EDIT Image 1. "
        "Set referenced_image_paths to the attached images' local paths, "
        "in exactly the attached order. Set transparent_background=false. "
        "Use these image roles: " + "; ".join(roles) + ". " + prompt + " "
        "Save the imagegen result to the exact absolute path " + str(output) + ". "
        "The tool normally reports a generated file path; copy that result "
        "to the destination, and verify that a nonempty PNG exists. "
        "Do not use procedural drawing, Blender, 3D compositing, or edit any "
        "other repository file. If the image cannot be saved, explain why."
    )
    prompt_path = work / f"{output.stem}.prompt.txt"
    prompt_path.write_text(instruction)
    command = [codex, "exec", "--ephemeral", "-m", "gpt-6-astra",
               "-s", "workspace-write", "-C", str(ROOT)]
    for path in images:
        command += ["-i", str(path)]
    run(command + ["-o", str(work / f"{output.stem}.astra.txt"), instruction],
        log=work / f"{output.stem}.imagegen.log")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Astra/imagegen returned no saved image: {output}")


def image_shape_ok(source: Path, candidate: Path) -> bool:
    with Image.open(source) as original, Image.open(candidate) as edited:
        source_ratio = original.width / original.height
        edited_ratio = edited.width / edited.height
        return (abs(source_ratio - edited_ratio) / source_ratio < 0.01
                and edited.width >= 1024 and edited.height >= 576)


def verify_pixels_and_geometry(source: Path, candidate: Path, plan: dict, guide_path: Path,
                               review: dict) -> dict:
    """Independently gate edit locality and visible LSP edge geometry.

    Landmarks are visual estimates supplied by Astra, not ground truth. Missing
    or hidden endpoints fail closed; numeric tolerances cover CCTV blur only.
    """
    with Image.open(source) as image, Image.open(candidate) as edited:
        original = np.asarray(image.convert("RGB"), dtype=np.int16)
        result = np.asarray(edited.convert("RGB").resize(image.size, Image.Resampling.LANCZOS), dtype=np.int16)
        sx, sy = image.width / edited.width, image.height / edited.height
    guide = json.loads(guide_path.with_name("geometry.json").read_text())
    points = [*guide["shared_edge_px"], *guide["new_lsp_near_edge_px"]]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    # Leave the existing cargo and the new cargo room above the floor footprint.
    # Original pixels outside this zone should survive an image edit.
    x0, x1 = max(0, min(xs) - 90), min(original.shape[1], max(xs) + 90)
    y0, y1 = max(0, min(ys) - 230), min(original.shape[0], max(ys) + 90)
    outside = np.ones(original.shape[:2], dtype=bool)
    outside[y0:y1, x0:x1] = False
    delta = np.abs(original - result).mean(axis=2)
    changed_fraction = float((delta[outside] > 30).mean())
    mean_delta = float(delta[outside].mean())
    locality_ok = changed_fraction <= 0.08 and mean_delta <= 9.0

    landmarks = review.get("landmarks", {})
    failures = []

    def edge(name: str) -> np.ndarray | None:
        raw = landmarks.get(name)
        if not isinstance(raw, list) or len(raw) != 2 or any(not isinstance(p, list) or len(p) != 2 for p in raw):
            failures.append(f"{name}: missing two visible endpoints")
            return None
        return np.asarray(raw, dtype=float) * np.array([sx, sy])

    original_front, new_rear, new_front = (edge(name) for name in
        ("original_front_edge", "new_rear_edge", "new_front_edge"))
    if not landmarks.get("both_junction_ends_visible"):
        failures.append("one or both shared-edge endpoints are hidden")
    endpoint_error = None
    width_error = None
    guide_error = None
    original_anchor_error = None
    if original_front is not None and new_rear is not None:
        endpoint_error = float(max(np.linalg.norm(original_front - new_rear, axis=1)))
        original_width = float(np.linalg.norm(original_front[1] - original_front[0]))
        new_width = float(np.linalg.norm(new_rear[1] - new_rear[0]))
        width_error = abs(new_width - original_width) / max(original_width, 1)
        if endpoint_error > 10:
            failures.append(f"shared-edge endpoint error {endpoint_error:.1f}px > 10px")
        if width_error > 0.05:
            failures.append(f"shared-edge width mismatch {width_error:.1%} > 5%")
    if original_front is not None:
        planned_edge = np.asarray(plan["contact_edge_px"], dtype=float)
        original_anchor_error = float(max(np.linalg.norm(original_front - planned_edge, axis=1)))
        if original_anchor_error > 15:
            failures.append(f"original LSP edge moved {original_anchor_error:.1f}px > 15px")
    if new_front is not None:
        guide_front = np.asarray(guide["new_lsp_near_edge_px"], dtype=float)
        guide_error = float(max(np.linalg.norm(new_front - guide_front, axis=1)))
        if guide_error > 35:
            failures.append(f"new LSP front edge differs from calibrated guide by {guide_error:.1f}px > 35px")
    if not locality_ok:
        failures.append(f"background outside edit zone changed: {changed_fraction:.1%} pixels, mean RGB delta {mean_delta:.1f}")
    return {"passed": not failures, "failures": failures, "edit_zone_xyxy": [x0, y0, x1, y1],
            "outside_changed_fraction": round(changed_fraction, 4), "outside_mean_rgb_delta": round(mean_delta, 2),
            "endpoint_error_px": endpoint_error, "original_anchor_error_px": original_anchor_error,
            "shared_width_error_fraction": width_error,
            "front_edge_guide_error_px": guide_error}


def verify_sheet_preserved(sheet: Path, cargo: Path, sheet_review: dict, cargo_review: dict) -> dict:
    """Check that the cargo edit did not move the accepted sheet landmarks."""
    failures = []
    with Image.open(sheet) as before, Image.open(cargo) as after:
        before_scale = np.array([1920 / before.width, 1080 / before.height])
        after_scale = np.array([1920 / after.width, 1080 / after.height])
    for key in ("original_front_edge", "new_rear_edge", "new_front_edge"):
        a = sheet_review.get("landmarks", {}).get(key, [])
        b = cargo_review.get("landmarks", {}).get(key, [])
        if len(a) != 2 or len(b) != 2:
            failures.append(f"{key}: cannot verify preservation")
            continue
        drift = float(max(np.linalg.norm(np.asarray(a, dtype=float) * before_scale
                                         - np.asarray(b, dtype=float) * after_scale, axis=1)))
        if drift > 8:
            failures.append(f"{key} moved {drift:.1f}px after cargo edit > 8px")
    return {"passed": not failures, "failures": failures}


def publish(candidate: Path, source: Path, output: Path) -> None:
    with Image.open(source) as original, Image.open(candidate) as edited:
        result = edited.convert("RGB")
        if result.size != original.size:
            result = result.resize(original.size, Image.Resampling.LANCZOS)
        result.save(output, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Valid cam01 CCTV image with one forklift-loaded LSP")
    parser.add_argument("--output", type=Path, help="Accepted PNG path; default work/out/<input>_violation.png")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", type=Path, help="Continue an earlier attempts directory; max-attempts is the total limit")
    parser.add_argument("--joint", action="store_true", help="Use the older one-step LSP+cargo workflow")
    parser.add_argument("--legacy-staged", action="store_true", help="Use the older Astra-landmark sequential workflow")
    parser.add_argument("--contact-edge", help="Override inferred LSP front edge: x1,y1,x2,y2 in source pixels")
    parser.add_argument("--reuse-sheet", type=Path, help="Reverify a previously generated empty LSP and continue with cargo")
    parser.add_argument("--reuse-cargo", type=Path, help="Reverify a previously generated cargo candidate after --reuse-sheet")
    parser.add_argument("--editor", choices=("codex", "api"), default="codex",
                        help="codex uses signed-in Astra + imagegen; api uses OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-image-2.5-sunburst")
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


def vision_main(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / "work/huggingface"))
    source = args.input.resolve()
    if not source.is_file():
        raise RuntimeError(f"Input image does not exist: {source}")
    if args.max_attempts < 1:
        raise RuntimeError("--max-attempts must be positive")
    if args.resume:
        raise RuntimeError("--resume currently applies only to --joint runs")
    if args.editor == "api" and (not args.image_cli.is_file() or not os.environ.get("OPENAI_API_KEY")):
        raise RuntimeError("--editor api requires the imagegen CLI and OPENAI_API_KEY")
    if not shutil.which(args.codex_bin):
        raise RuntimeError(f"Codex CLI is unavailable: {args.codex_bin}")
    refs = selected_references(args)
    output = (args.output or ROOT / "work/out" / f"{source.stem}_violation.png").resolve()
    if output.exists():
        raise RuntimeError(f"Output already exists; choose --output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = ROOT / "work/out" / f"{output.stem}_vision_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    work.mkdir(parents=True)
    report_path = work / "report.json"
    report = {"input": str(source), "input_sha256": sha256(source),
              "step_a_index": str(args.step_a_index.resolve()),
              "reference_map": str(args.reference_map.resolve()) if getattr(args, "reference_map", None) else None,
              "references": {key: str(value) for key, value in refs.items()},
              "editor": args.editor, "stages": {"sheet": [], "cargo": []}, "accepted": False}

    def save_report() -> None:
        report_path.write_text(json.dumps(report, indent=2))

    try:
        segmenter = Sam3Masks(ROOT / "work/vision_cache/sam3")
        source_masks = segmenter.get(source, ("forklift", "LSP", "cargo", "floor"))
        with Image.open(source) as image:
            size = image.size
        if args.contact_edge:
            values = [int(value.strip()) for value in args.contact_edge.split(",")]
            if (len(values) != 4 or values[0] >= values[2]
                    or any(not 0 <= value < size[index % 2] for index, value in enumerate(values))):
                raise RuntimeError("--contact-edge must be x1,y1,x2,y2 within input dimensions")
            edge = [[values[0], values[1]], [values[2], values[3]]]
            anchor_info = {"method": "manual_override", "edge_observed": False}
        else:
            edge, anchor_info = source_edge(source_masks, size)
        report["anchor"] = {"contact_edge_px": edge, **anchor_info}
        save_report()
        point_cache = ROOT / "work/vision_cache/moge" / f"{sha256(source)[:20]}.npz"
        point_cache.parent.mkdir(parents=True, exist_ok=True)
        report["moge"] = moge_floor_diagnostic(source, source_masks, point_cache)
        plan = {"contact_edge_px": edge}
        guide = make_geometry_guide(source, edge, work)
        geometry = json.loads((work / "geometry.json").read_text())
        expected_width = float(json.loads(CALIBRATION.read_text())["lsp_measured_size_m"][0])
        measured_width = float(geometry["original_contact_edge_width_m"])
        if not 0.75 * expected_width <= measured_width <= 1.25 * expected_width:
            raise RuntimeError(f"Source LSP edge width {measured_width:.2f}m differs from Step A "
                               f"{expected_width:.2f}m by over 25%; inspect the anchor or use --contact-edge")
        polygon = [*geometry["shared_edge_px"], *reversed(geometry["new_lsp_near_edge_px"])]
        report["geometry_guide"] = str(guide)
        report["target_polygon_px"] = polygon
        save_report()
        protected = [union(source_masks.get(key, []), (size[1], size[0]))
                     for key in ("forklift", "cargo", "LSP")]
        accepted_sheet = None
        sheet_masks = None
        feedback = ""
        sheet_attempts = [0] if getattr(args, "reuse_sheet", None) else range(1, args.max_attempts + 1)
        for attempt in sheet_attempts:
            if attempt == 0:
                candidate = args.reuse_sheet.resolve()
                if not candidate.is_file():
                    raise RuntimeError(f"--reuse-sheet image is missing: {candidate}")
            else:
                raw = work / f"sheet_raw_{attempt:02d}.png"
                candidate = work / f"sheet_{attempt:02d}.png"
                guide_note = (f"Target LSP top corners in source pixels: {polygon}. The shared rear edge "
                              f"is {edge}. These are calibrated positions, including any obscured source edge. ")
                prompt = ("Add exactly one EMPTY LSP on the floor at the guide polygon. "
                          "Do not add cargo. Match the Step A LSP appearance, perspective, thickness and CCTV "
                          "texture. Keep the original forklift, cargo and LSP intact. Colored guide marks "
                          "must not appear in the photograph. " + guide_note + feedback)
                guided_edit(args, [source, refs["forklift"], refs["LSP"], refs["SKID"], guide],
                            ["Image 1 original valid source", "Image 2 Step A forklift appearance",
                             "Image 3 Step A LSP appearance", "Image 4 Step A SKID preservation reference",
                             "Image 5 geometry guide"], prompt, raw, work)
                preserve_outside(source, raw, candidate, edit_region(size, polygon, "sheet"), protected)
            candidate_masks = segmenter.get(candidate, ("LSP", "cargo"))
            measured = verify_sheet_masks(source_masks, candidate_masks, polygon, size)
            preservation = verify_unchanged(source, candidate, edit_region(size, polygon, "sheet"), protected)
            measured["preservation"] = preservation
            measured["failures"].extend(preservation["failures"])
            measured["passed"] = measured["passed"] and preservation["passed"]
            aesthetic = astra(args.codex_bin, [source, candidate, refs["LSP"]], AESTHETIC_SCHEMA,
                              "Image 1 is source, Image 2 adds one empty LSP, Image 3 is Step A LSP. "
                              "Judge ONLY whether the new sheet has natural CCTV edges, texture, floor shadow "
                              "and lighting. Do not judge hidden geometry or count objects. JSON only.",
                              work, f"sheet_aesthetic_{attempt:02d}")
            passed = (measured["passed"] and image_shape_ok(source, candidate)
                      and aesthetic.get("natural_edges") is True
                      and aesthetic.get("realistic_lighting") is True and aesthetic.get("accepted") is True)
            report["stages"]["sheet"].append({"number": attempt, "candidate": str(candidate),
                                               "measured": measured, "aesthetic": aesthetic, "accepted": passed})
            save_report()
            print(f"LSP {'reuse' if attempt == 0 else f'{attempt}/{args.max_attempts}'}: {'PASS' if passed else 'FAIL'} — "
                  + "; ".join(measured["failures"] + [aesthetic.get("feedback", "")]), flush=True)
            if passed:
                accepted_sheet, sheet_masks = candidate, candidate_masks
                break
            feedback = "Fix these failures: " + "; ".join(measured["failures"] + [aesthetic.get("feedback", "")])
        if accepted_sheet is None:
            report["stopped_at"] = "sheet_geometry_or_realism"
            return 2
        report["accepted_sheet"] = str(accepted_sheet)
        save_report()
        original_cargo = report["anchor"].get("cargo_bbox")
        if original_cargo is None:
            boxes = [bbox(mask) for mask in source_masks.get("cargo", [])]
            boxes = [box for box in boxes if box is not None]
            if boxes:
                edge_x = (edge[0][0] + edge[1][0]) / 2
                edge_y = (edge[0][1] + edge[1][1]) / 2
                original_cargo = min(boxes, key=lambda box:
                                     abs((box[0] + box[2]) / 2 - edge_x) + abs(box[3] - edge_y))
            else:
                width_guess = edge[1][0] - edge[0][0]
                original_cargo = [edge[0][0], edge[0][1] - width_guess,
                                  edge[1][0], edge[0][1]]
        cargo_height = original_cargo[3] - original_cargo[1]
        floor_y = round(sum(point[1] for point in geometry["new_lsp_near_edge_px"]) / 2) - 18
        cargo_x = round(sum(point[0] for point in polygon) / 4) + 18
        cargo_width = round((original_cargo[2] - original_cargo[0]) * 0.9)
        cargo_box = [cargo_x - cargo_width // 2, floor_y - round(cargo_height * 1.05),
                     cargo_x + cargo_width // 2, floor_y]
        report["cargo_target_box_px"] = cargo_box
        # The closer new cargo may correctly occlude part of the original load.
        # Preserve original cargo everywhere outside that planned occlusion box.
        original_lsp_protected = protected[2].copy()
        original_lsp_protected[max(0, cargo_box[1]):cargo_box[3],
                               max(0, cargo_box[0]):cargo_box[2]] = False
        cargo_protected = [protected[0], original_lsp_protected]
        original_cargo_protected = protected[1].copy()
        original_cargo_protected[max(0, cargo_box[1]):cargo_box[3],
                                 max(0, cargo_box[0]):cargo_box[2]] = False
        cargo_protected.append(original_cargo_protected)
        cargo_attempts = [0] if getattr(args, "reuse_cargo", None) else range(1, args.max_attempts + 1)
        for attempt in cargo_attempts:
            if attempt == 0:
                candidate = args.reuse_cargo.resolve()
                if not candidate.is_file():
                    raise RuntimeError(f"--reuse-cargo image is missing: {candidate}")
            else:
                raw = work / f"cargo_raw_{attempt:02d}.png"
                candidate = work / f"cargo_{attempt:02d}.png"
                prompt = ("Add exactly ONE UPRIGHT, FULL-SIZE wrapped cargo like Image 4 on the new empty LSP. "
                          "Do not make a small parcel or a flat stack. Match the original cargo's apparent "
                          "height and nearly its width at this camera distance. Place its approximate bounding "
                          f"box at {cargo_box} in the 1920x1080 source image, with its bottom on the deck. "
                          "Keep the cargo fully within the LSP footprint at the bottom, leave the two LSP "
                          "junction side ends visible, and match lighting and contact shadow. The nearer "
                          "cargo may naturally occlude the load behind it. Render a single coherent cargo "
                          "surface without a triangular patch or dark strip at its bottom. Do not move either LSP, forklift "
                          "or background. Geometry guide polygon: " + str(polygon) + ". " + feedback)
                guided_edit(args, [accepted_sheet, source, refs["LSP"], refs["cargo"], guide],
                            ["Image 1 geometry-accepted empty LSP", "Image 2 original valid source",
                             "Image 3 Step A LSP", "Image 4 Step A cargo", "Image 5 geometry guide"],
                            prompt, raw, work)
                preserve_outside(accepted_sheet, raw, candidate,
                                 edit_region(size, polygon, "cargo"), cargo_protected)
            candidate_masks = segmenter.get(candidate, ("LSP", "cargo"))
            measured = verify_cargo_masks(sheet_masks, candidate_masks, polygon, size,
                                          min_height_px=round(cargo_height * 0.7))
            preservation = verify_unchanged(accepted_sheet, candidate,
                                             edit_region(size, polygon, "cargo"), cargo_protected)
            measured["preservation"] = preservation
            measured["failures"].extend(preservation["failures"])
            measured["passed"] = measured["passed"] and preservation["passed"]
            aesthetic = astra(args.codex_bin, [accepted_sheet, candidate, refs["cargo"]], AESTHETIC_SCHEMA,
                              "Image 1 is the accepted empty-LSP stage. Image 2 adds cargo. Image 3 is the "
                              "Step A cargo reference. Judge ONLY natural cargo edges, lighting and contact "
                              "shadow; geometry and object count are checked by code. JSON only.",
                              work, f"cargo_aesthetic_{attempt:02d}")
            passed = (measured["passed"] and image_shape_ok(source, candidate)
                      and aesthetic.get("natural_edges") is True
                      and aesthetic.get("realistic_lighting") is True and aesthetic.get("accepted") is True)
            report["stages"]["cargo"].append({"number": attempt, "candidate": str(candidate),
                                               "measured": measured, "aesthetic": aesthetic, "accepted": passed})
            save_report()
            print(f"Cargo {'reuse' if attempt == 0 else f'{attempt}/{args.max_attempts}'}: {'PASS' if passed else 'FAIL'} — "
                  + "; ".join(measured["failures"] + [aesthetic.get("feedback", "")]), flush=True)
            if passed:
                publish(candidate, source, output)
                report.update({"accepted": True, "output": str(output), "output_sha256": sha256(output)})
                print(f"Accepted image: {output}\nReview report: {report_path}")
                return 0
            feedback = "Fix these failures: " + "; ".join(measured["failures"] + [aesthetic.get("feedback", "")])
        report["stopped_at"] = "cargo_support_or_realism"
        return 2
    finally:
        save_report()


def sequential_main(args: argparse.Namespace) -> int:
    """Gate a visible empty LSP before adding cargo in a separate edit."""
    source = args.input.resolve()
    if not source.is_file():
        raise RuntimeError(f"Input image does not exist: {source}")
    if args.max_attempts < 1:
        raise RuntimeError("--max-attempts must be positive")
    if args.resume:
        raise RuntimeError("--resume currently applies to --joint runs; start a new sequential run")
    if args.editor == "api" and (not args.image_cli.is_file() or not os.environ.get("OPENAI_API_KEY")):
        raise RuntimeError("--editor api requires the imagegen CLI and OPENAI_API_KEY")
    if not shutil.which(args.codex_bin):
        raise RuntimeError(f"Codex CLI is unavailable: {args.codex_bin}")
    refs = selected_references(args)
    output = (args.output or ROOT / "work/out" / f"{source.stem}_violation.png").resolve()
    if output.exists():
        raise RuntimeError(f"Output already exists; choose --output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = ROOT / "work/out" / f"{output.stem}_staged_{timestamp}"
    work.mkdir(parents=True)
    report_path = work / "report.json"
    report = {"input": str(source), "input_sha256": sha256(source), "step_a_index": str(args.step_a_index.resolve()),
              "references": {key: str(value) for key, value in refs.items()}, "editor": args.editor,
              "reference_map": str(args.reference_map.resolve()) if getattr(args, "reference_map", None) else None,
              "stages": {"sheet": [], "cargo": []}, "accepted": False}

    def edit(previous: Path | None, prompt: str, candidate: Path, guide: Path) -> None:
        if args.editor == "codex":
            image_edit_codex(args.codex_bin, source, previous, refs, prompt, candidate, work, guide)
        else:
            image_edit_api(args.image_cli, source, previous, refs, prompt, candidate, work, args.model, guide)

    try:
        plan = astra(args.codex_bin, [source, *refs.values()], PLAN_SCHEMA,
                     "Image 1 is the valid source; other images are Step A forklift, LSP, SKID and cargo crops. "
                     "Confirm exactly one loaded LSP in the central forklift train, then identify its FRONT "
                     "cross-edge as left-to-right pixel endpoints. We will add an empty LSP first, cargo later. JSON only.",
                     work, "plan")
        report["plan"] = plan
        report_path.write_text(json.dumps(report, indent=2))
        if not plan.get("valid_source") or plan.get("source_loaded_lsp_count") != 1:
            raise RuntimeError(f"Input is not a confirmed one-loaded-LSP valid scene: {plan.get('reason')}")
        guide = make_geometry_guide(source, plan["contact_edge_px"], work)
        report["geometry_guide"] = str(guide)
        previous_sheet = None
        prompt = sheet_prompt(plan)
        accepted_sheet = None
        sheet_review = None
        for number in range(1, args.max_attempts + 1):
            candidate = work / f"sheet_{number:02d}.png"
            edit(previous_sheet, prompt, candidate, guide)
            review = astra(args.codex_bin, [source, candidate, *refs.values()], SHEET_SCHEMA,
                           "Image 1 is original valid source; Image 2 should have exactly ONE NEW EMPTY LSP "
                           "touching the original loaded LSP. Other images are Step A references. "
                           "Check that no cargo has been added yet, no extra LSP exists, both sheets are grounded "
                           "and their real dimensions match. In landmarks report left-to-right pixel endpoints "
                           "on Image 2 at native resolution: original_front_edge, new_rear_edge, new_front_edge. "
                           "Use empty arrays when hidden or uncertain; never guess. Both junction endpoints "
                           "must be visible. Set accepted true only if every check passes. JSON only.",
                           work, f"sheet_review_{number:02d}")
            measured = verify_pixels_and_geometry(source, candidate, plan, guide, review)
            review["measured_failures"] = measured["failures"]
            shape_ok = image_shape_ok(source, candidate)
            accepted = (shape_ok and measured["passed"] and bool(review.get("accepted"))
                        and all(review.get("checks", {}).get(key) is True
                                for key in SHEET_SCHEMA["properties"]["checks"]["required"]))
            report["stages"]["sheet"].append({"number": number, "candidate": str(candidate),
                                                "review": review, "measured": measured, "accepted": accepted})
            report_path.write_text(json.dumps(report, indent=2))
            print(f"LSP {number}/{args.max_attempts}: {'PASS' if accepted else 'FAIL'} — "
                  f"{review.get('feedback', '')}; {'; '.join(measured['failures'])}", flush=True)
            if accepted:
                accepted_sheet, sheet_review = candidate, review
                break
            previous_sheet = candidate
            prompt = sheet_prompt(plan, str(review.get("feedback", "")) + " "
                                  + "; ".join(measured["failures"]) + " "
                                  + str(review.get("revised_prompt", "")))
        if accepted_sheet is None:
            report["stopped_at"] = "sheet_geometry"
            return 2
        report["accepted_sheet"] = str(accepted_sheet)
        report_path.write_text(json.dumps(report, indent=2))
        cargo_feedback = None
        for number in range(1, args.max_attempts + 1):
            candidate = work / f"cargo_{number:02d}.png"
            # Always edit the accepted empty sheet; a bad cargo attempt cannot
            # accumulate geometry drift in the next attempt.
            edit(accepted_sheet, cargo_prompt(cargo_feedback), candidate, guide)
            review = astra(args.codex_bin, [source, accepted_sheet, candidate, *refs.values()], REVIEW_SCHEMA,
                           "Image 1 is original source. Image 2 is accepted empty-LSP stage. Image 3 is "
                           "cargo-added candidate. Other images are Step A refs. Require exactly two LOADED "
                           "LSPs, no extra empty LSP, only one new cargo, preserved original scene and preserved "
                           "accepted LSP geometry. In landmarks give left-to-right endpoints on Image 3 at its "
                           "native resolution for original_front_edge, new_rear_edge, new_front_edge. Use empty "
                           "arrays for hidden or uncertain edges, never guess. Both junction ends must remain "
                           "visible. Set accepted true only if every check passes. JSON only.",
                           work, f"cargo_review_{number:02d}")
            measured = verify_pixels_and_geometry(source, candidate, plan, guide, review)
            preservation = verify_sheet_preserved(accepted_sheet, candidate, sheet_review, review)
            measured["sheet_preservation"] = preservation
            measured["failures"].extend(preservation["failures"])
            measured["passed"] = measured["passed"] and preservation["passed"]
            review["measured_failures"] = measured["failures"]
            shape_ok = image_shape_ok(source, candidate)
            accepted = (shape_ok and measured["passed"] and bool(review.get("accepted"))
                        and all(review.get("checks", {}).get(key) is True
                                for key in REVIEW_SCHEMA["properties"]["checks"]["required"]))
            report["stages"]["cargo"].append({"number": number, "candidate": str(candidate),
                                                "review": review, "measured": measured, "accepted": accepted})
            report_path.write_text(json.dumps(report, indent=2))
            print(f"Cargo {number}/{args.max_attempts}: {'PASS' if accepted else 'FAIL'} — "
                  f"{review.get('feedback', '')}; {'; '.join(measured['failures'])}", flush=True)
            if accepted:
                publish(candidate, source, output)
                report.update({"accepted": True, "output": str(output), "output_sha256": sha256(output)})
                print(f"Accepted image: {output}\nReview report: {report_path}")
                return 0
            cargo_feedback = review
        report["stopped_at"] = "cargo"
        return 2
    finally:
        report_path.write_text(json.dumps(report, indent=2))


def main() -> int:
    args = parse_args()
    if not getattr(args, "joint", False) and not getattr(args, "legacy_staged", False):
        return vision_main(args)
    if not getattr(args, "joint", True):
        return sequential_main(args)
    source = args.input.resolve()
    if not source.is_file():
        raise RuntimeError(f"Input image does not exist: {source}")
    if args.max_attempts < 1:
        raise RuntimeError("--max-attempts must be positive")
    if args.editor == "api":
        if not args.image_cli.is_file():
            raise RuntimeError(f"Imagegen CLI is missing: {args.image_cli}")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --editor api")
    if not shutil.which(args.codex_bin):
        raise RuntimeError(f"Codex CLI is unavailable: {args.codex_bin}")

    refs = selected_references(args)
    output = (args.output or ROOT / "work/out" / f"{source.stem}_violation.png").resolve()
    if output.exists():
        raise RuntimeError(f"Output already exists; choose --output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        work = args.resume.resolve()
        report_path = work / "report.json"
        report = json.loads(report_path.read_text())
        if report.get("input_sha256") != sha256(source) or report.get("input") != str(source):
            raise RuntimeError("Resume input differs from the saved run")
        if report.get("editor") != args.editor or report.get("accepted"):
            raise RuntimeError("Resume editor differs or the run is already accepted")
        if not report.get("attempts"):
            raise RuntimeError("Resume report has no completed attempt")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        work = ROOT / "work/out" / f"{output.stem}_attempts_{timestamp}"
        work.parent.mkdir(parents=True, exist_ok=True)
        work.mkdir()
        report = {
            "input": str(source), "input_sha256": sha256(source), "step_a_index": str(args.step_a_index.resolve()),
            "references": {key: str(value) for key, value in refs.items()},
            "editor": args.editor, "model": args.model if args.editor == "api" else "Codex imagegen tool",
            "attempts": [], "accepted": False,
        }
    report_path = work / "report.json"
    try:
        if args.resume:
            plan = report["plan"]
            guide = Path(report["geometry_guide"])
            previous = Path(report["attempts"][-1]["candidate"])
            if not guide.is_file() or not previous.is_file():
                raise RuntimeError("Resume guide or previous candidate is missing")
            prompt = repair_prompt(plan, report["attempts"][-1]["review"])
            start = len(report["attempts"]) + 1
        else:
            plan = astra(args.codex_bin, [source, *refs.values()], PLAN_SCHEMA,
                "Image 1 is the real input; later images are Step A SAM3 crops for forklift, LSP, SKID and cargo. "
                "Check that Image 1 shows a forklift pushing exactly one loaded LSP (valid source). "
                "Return valid_source=false if not. Locate the front cross-edge of that LSP in original-image pixels "
                "as two endpoints. Write a concise image-edit prompt adding exactly one same-sized loaded LSP "
                "immediately ahead, sharing that entire cross-edge. Preserve all original objects; do not add "
                "a forklift or SKID. JSON only.", work, "plan")
            report["plan"] = plan
            report_path.write_text(json.dumps(report, indent=2))
            if not plan.get("valid_source") or plan.get("source_loaded_lsp_count") != 1:
                raise RuntimeError(f"Input is not a confirmed one-loaded-LSP valid scene: {plan.get('reason')}")
            guide = make_geometry_guide(source, plan["contact_edge_px"], work)
            report["geometry_guide"] = str(guide)
            prompt = base_prompt(plan, refs)
            previous = None
            start = 1
        if start > args.max_attempts:
            raise RuntimeError("Resume run already reached --max-attempts; raise the total limit")
        for number in range(start, args.max_attempts + 1):
            candidate = work / f"attempt_{number:02d}.png"
            if args.editor == "codex":
                image_edit_codex(args.codex_bin, source, previous, refs, prompt, candidate, work, guide)
            else:
                image_edit_api(args.image_cli, source, previous, refs, prompt, candidate, work, args.model, guide)
            shape_ok = image_shape_ok(source, candidate)
            review = astra(args.codex_bin, [source, candidate, *refs.values()], REVIEW_SCHEMA,
                "Image 1 is the original valid source; Image 2 is this candidate; remaining images are Step A "
                "object crops. Review the candidate skeptically. Require exactly two loaded LSPs in the "
                "forklift train, no intermediate empty sheet, one added cargo, ground contact, natural edges, "
                "unchanged source objects and background. The two LSPs must have equal REAL length, width "
                "and thickness: at their shared end edge the projected width and endpoints must coincide; "
                "a nearer front edge may grow slightly by perspective. The cargo can hide the middle of the "
                "shared edge; judge contact and equal width from BOTH visible side endpoints and the side-edge "
                "continuations. Do not demand that the hidden middle be visible, but fail if either endpoint "
                "is hidden or visibly misaligned. In landmarks, give two LEFT-TO-RIGHT pixel endpoints "
                "for the original LSP front edge, new LSP rear edge and new LSP front edge, measured on "
                "Image 2 at its native resolution. Use empty arrays for hidden or uncertain edges; never guess. "
                "Mark both_junction_ends_visible=false if either end is hidden. "
                "Set accepted=true only if EVERY check passes. Give concrete "
                "visual evidence in feedback and a complete revised edit prompt fixing the failures. JSON only.",
                work, f"review_{number:02d}")
            checks = review.get("checks", {})
            measured = verify_pixels_and_geometry(source, candidate, plan, guide, review)
            review["measured_failures"] = measured["failures"]
            accepted = (bool(review.get("accepted")) and shape_ok and measured["passed"]
                        and all(checks.get(key) is True for key in REVIEW_SCHEMA["properties"]["checks"]["required"]))
            report["attempts"].append({"number": number, "candidate": str(candidate), "shape_ok": shape_ok,
                                       "review": review, "measured": measured, "accepted": accepted})
            report_path.write_text(json.dumps(report, indent=2))
            print(f"Attempt {number}/{args.max_attempts}: {'PASS' if accepted else 'FAIL'} — {review.get('feedback', '')}", flush=True)
            if accepted:
                publish(candidate, source, output)
                report["accepted"] = True
                report["output"] = str(output)
                report["output_sha256"] = sha256(output)
                report_path.write_text(json.dumps(report, indent=2))
                print(f"Accepted image: {output}\nReview report: {report_path}")
                return 0
            previous = candidate
            prompt = repair_prompt(plan, review)
        best = max(report["attempts"], key=lambda item: (
            sum(value is True for value in item["review"].get("checks", {}).values()),
            item["number"],
        ))
        review_candidate = output.with_name(f"{output.stem}_needs_review.png")
        if review_candidate.exists():
            raise RuntimeError(f"Review candidate already exists: {review_candidate}")
        publish(Path(best["candidate"]), source, review_candidate)
        report["review_candidate"] = str(review_candidate)
        report["review_candidate_sha256"] = sha256(review_candidate)
        print(f"No accepted image after {args.max_attempts} attempts. "
              f"Best candidate: {review_candidate}. Inspect {report_path}", file=sys.stderr)
        return 2
    finally:
        report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
