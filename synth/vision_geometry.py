"""SAM3/MoGe measurements for editing one valid cam01 frame.

The calibrated polygon is the geometric target. A hidden source edge is
inferred from calibration and tagged as such; it is never described as visible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PROMPTS = {
    "forklift": ("forklift",),
    "LSP": ("black platform", "flat top of platform"),
    "cargo": ("stretch wrapped cargo", "box"),
    "floor": ("floor",),
}


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
            with Image.open(image_path) as image:
                state = self.processor.set_image(image.convert("RGB"))
            for key in missing:
                masks: list[np.ndarray] = []
                for prompt in PROMPTS[key]:
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


def source_edge(masks: dict[str, list[np.ndarray]], image_size: tuple[int, int]) -> tuple[list[list[int]], dict]:
    """Find the central loaded LSP by cargo overlap and its exposed front rim."""
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
    ys = np.flatnonzero(mask.any(axis=1))
    rim = mask[max(int(ys[-1]) - 10, 0):int(ys[-1]) + 1]
    xx = np.flatnonzero(rim.any(axis=0))
    if len(xx) < 35:
        raise RuntimeError("Source LSP front rim is too small or occluded; supply --contact-edge")
    edge = [[int(xx[0]), int(ys[-1])], [int(xx[-1]), int(ys[-1])]]
    return edge, {"method": "sam3_exposed_rim", "cargo_bbox": cargo_box,
                  "lsp_bbox": box, "edge_observed": True}


def moge_floor_diagnostic(source: Path, masks: dict[str, list[np.ndarray]], output: Path) -> dict:
    """Infer the source point map once and fit a floor plane to free pixels."""
    import torch
    from moge.model.v2 import MoGeModel

    if output.exists():
        with np.load(output) as saved:
            points, valid = saved["points"], saved["valid"].astype(bool)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        with torch.inference_mode():
            inferred = model.infer(torch.tensor(rgb, device=device).permute(2, 0, 1),
                                   resolution_level=9, use_fp16=device == "cuda")
        points = inferred["points"].cpu().numpy()
        valid = inferred["mask"].cpu().numpy().astype(bool)
        np.savez_compressed(output, points=points, valid=valid)
        del model
    floor = union(masks.get("floor", []), next(iter(masks["floor"])).shape if masks.get("floor") else valid.shape)
    for key in ("forklift", "LSP", "cargo"):
        floor &= ~union(masks.get(key, []), floor.shape)
    if floor.shape != valid.shape:
        floor = np.asarray(Image.fromarray(floor.astype(np.uint8) * 255).resize(
            (valid.shape[1], valid.shape[0]), Image.Resampling.NEAREST)) > 127
    floor &= valid
    sample = points[floor][::max(1, int(floor.sum() / 25000))]
    sample = sample[np.isfinite(sample).all(axis=1)]
    if len(sample) < 1000:
        raise RuntimeError("MoGe/SAM3 found too few valid floor points")
    centre = np.median(sample, axis=0)
    _, _, vh = np.linalg.svd(sample - centre, full_matrices=False)
    residual = np.abs((sample - centre) @ vh[-1])
    return {"floor_points": len(sample), "median_floor_residual_m": float(np.median(residual)),
            "p90_floor_residual_m": float(np.percentile(residual, 90)), "point_map": str(output)}


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


def preserve_outside(base: Path, generated: Path, output: Path, region: Image.Image,
                     protected: list[np.ndarray] | None = None) -> None:
    with Image.open(base) as original, Image.open(generated) as edited:
        original = original.convert("RGB")
        edited = edited.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
        alpha = np.asarray(region, dtype=np.uint8).copy()
        for mask in protected or []:
            if mask.shape != alpha.shape:
                raise RuntimeError("Protected SAM3 mask dimensions differ from source")
            alpha[mask] = 0
        result = Image.composite(edited, original, Image.fromarray(alpha))
        result.save(output)


def verify_unchanged(base: Path, candidate: Path, region: Image.Image,
                     protected: list[np.ndarray]) -> dict:
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


def verify_sheet_masks(source: dict[str, list[np.ndarray]], candidate: dict[str, list[np.ndarray]],
                       polygon: list[list[int]], size: tuple[int, int]) -> dict:
    target = polygon_mask(size, polygon)
    source_lsp = union(source.get("LSP", []), target.shape)
    # SAM3 can give different masks for unrelated parked LSPs even when their
    # pixels are unchanged. Only instances intersecting the planned footprint
    # can represent the newly added sheet.
    new_instances = [mask for mask in candidate.get("LSP", [])
                     if ((mask & ~source_lsp) & target).sum() > 1500]
    added = union(new_instances, target.shape) & ~source_lsp
    coverage = float((added & target).sum() / max(target.sum(), 1))
    outside = float((added & ~target).sum() / max(added.sum(), 1))
    # A thin dark side wall may extend beyond the projected top polygon.
    failures = []
    if coverage < 0.45:
        failures.append(f"SAM3 new LSP covers {coverage:.1%} of target; need >=45%")
    if outside > 0.25:
        failures.append(f"SAM3 new LSP outside target {outside:.1%}; need <=25%")
    if added.sum() < 2500:
        failures.append("SAM3 found too little new LSP area")
    if len(new_instances) != 1:
        failures.append(f"SAM3 found {len(new_instances)} new LSP instances; expected one")
    # Source cargo remains on the original LSP; sheet stage adds none.
    source_cargo = union(source.get("cargo", []), target.shape)
    edited_cargo = union(candidate.get("cargo", []), target.shape)
    new_cargo = edited_cargo & ~source_cargo
    if (new_cargo & target).sum() > 1000:
        failures.append("Cargo appeared before the cargo stage")
    return {"passed": not failures, "failures": failures, "coverage": coverage,
            "outside_fraction": outside, "added_lsp_pixels": int(added.sum())}


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
