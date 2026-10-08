"""Add exactly one loaded LSP to the real s_003 valid frame for visual review."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/cam01/rtsp_samples/s_003.jpg"
DONOR = ROOT / "work/out/s003_adjacent_scaled.png"
TEXTURE = ROOT / "data/cam01/references/bg1.jpg"
OUTPUT = ROOT / "work/out/s003_one_added_loaded_lsp.png"


def main() -> None:
    source = np.asarray(Image.open(SOURCE).convert("RGB"))
    donor = np.asarray(Image.open(DONOR).convert("RGB"))
    texture = np.asarray(Image.open(TEXTURE).convert("RGB"))
    height, width = source.shape[:2]
    if donor.shape != source.shape or texture.shape != source.shape:
        raise ValueError("All images must share the 1920x1080 camera frame")

    # Astra 6 reviewed the source and set one quadrilateral, joined to the
    # original sheet at its front edge. No intermediate sheet is constructed.
    rear_left, rear_right = (863, 470), (1029, 478)
    front_left, front_right = (885, 539), (1049, 545)
    quad = np.float32([rear_left, rear_right, front_right, front_left])
    canvas = source.astype(np.float32).copy()

    sheet_mask = Image.new("L", (width, height))
    sheet_draw = ImageDraw.Draw(sheet_mask)
    sheet_draw.polygon([tuple(map(int, point)) for point in quad], fill=255)
    # The exposed rear-left strip was repeatedly read as a third empty LSP.
    # It is occluded by the load in this view, so keep only the short seam at
    # the source sheet and the visible side/front of the added sheet.
    sheet_draw.polygon([(858, 476), (895, 476), (895, 524),
                        (879, 536), (858, 536)], fill=0)
    mask = np.asarray(sheet_mask.filter(ImageFilter.GaussianBlur(1.1)), dtype=np.float32) / 255
    shadow = np.asarray(sheet_mask.filter(ImageFilter.GaussianBlur(10)), dtype=np.float32) / 255
    shadow = cv2.warpAffine(shadow, np.float32([[1, 0, 3], [0, 1, 6]]), (width, height)) * 0.28
    canvas *= 1 - shadow[..., None]

    # This top surface is a real LSP crop from the same camera's references.
    left, top, right, bottom = 324, 582, 576, 690
    crop = texture[top:bottom, left:right].astype(np.float32)
    src_quad = np.float32([[0, 0], [crop.shape[1]-1, 0],
                           [crop.shape[1]-1, crop.shape[0]-1], [0, crop.shape[0]-1]])
    transform = cv2.getPerspectiveTransform(src_quad, quad)
    warped = cv2.warpPerspective(crop, transform, (width, height))
    # Match the dark top face of the original loaded LSP in s_003.
    warped = cv2.GaussianBlur(warped, (11, 11), 0)
    warped = np.clip(warped * 0.20 + np.array([39, 44, 48]) * 0.80, 0, 255)
    canvas = canvas * (1 - mask[..., None]) + warped * mask[..., None]

    # A single front lip establishes the 1-2 cm sheet thickness. The rear
    # boundary shares the original sheet edge and receives no duplicate lip.
    lip = Image.new("L", (width, height))
    draw = ImageDraw.Draw(lip)
    # The alpha edge plus shadow forms one thin front edge; a second dark lip
    # made the sheet look like a raised, empty platform.
    lip_alpha = np.asarray(lip, dtype=np.float32) / 255
    canvas = canvas * (1 - lip_alpha[..., None]) + np.array([24, 27, 30]) * lip_alpha[..., None]

    # Crop only the newly generated white container. The previously generated
    # dark strip at x=858-898, y=473-519 is deliberately outside this mask.
    cargo_outline = [(899, 436), (1016, 438), (1047, 479), (1047, 487),
                     (1027, 569), (901, 571), (894, 550), (892, 469)]
    cargo_mask = Image.new("L", (width, height))
    ImageDraw.Draw(cargo_mask).polygon(cargo_outline, fill=255)
    cargo_alpha = np.asarray(cargo_mask.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) / 255
    cargo_alpha *= np.clip((donor.astype(np.float32).mean(axis=2) - 50) / 30, 0, 1)
    # The earlier donor made a nearer container smaller than the original.
    # Scale it about its floor contact so the bottom remains seated on the LSP.
    scale = 1.11
    cx, floor_y = 965, 576
    shift_y = -40
    matrix = np.float32([[scale, 0, cx * (1-scale)],
                         [0, scale, floor_y * (1-scale) + shift_y]])
    cargo_alpha = cv2.warpAffine(cargo_alpha, matrix, (width, height))
    cargo_colour = cv2.warpAffine(donor, matrix, (width, height))
    canvas = canvas * (1 - cargo_alpha[..., None]) + cargo_colour * cargo_alpha[..., None]

    result = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
    support = (mask > 0) | (shadow > 0) | (lip_alpha > 0) | (cargo_alpha > 0)
    # The source LSP's exposed left tongue becomes a false third sheet after
    # the new cargo occludes its center. Remove only that exposed tongue.
    tongue = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(tongue, [np.int32([(857, 463), (889, 464),
                                   (892, 479), (857, 481)])], 255)
    result = cv2.inpaint(result, tongue, 5, cv2.INPAINT_TELEA)
    support |= tongue > 0
    result[~support] = source[~support]
    Image.fromarray(result).save(OUTPUT)
    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "event": "Forklift Pushing Multiple Lsps",
        "original_loaded_lsp_count": 1,
        "added_lsp_count": 1,
        "added_cargo_count": 1,
        "added_lsp_corners": quad.astype(int).tolist(),
        "rear_edge_width_px": round(float(np.linalg.norm(quad[1] - quad[0])), 1),
        "front_edge_width_px": round(float(np.linalg.norm(quad[2] - quad[3])), 1),
        "outside_support_unchanged": bool(np.array_equal(result[~support], source[~support])),
        "status": "visual_review_candidate",
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
