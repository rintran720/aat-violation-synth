"""Render the original clean-background workflow and make a four-panel JPG."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synth.composite_violation import composite


def render_one(config: dict, blender: str, scenario: str, count: int, stem: str,
               samples: int, reuse_render: bool = False) -> Path:
    work = Path(config["work_dir"])
    command = [
        blender, "--background", "--threads", "4", "--python-exit-code", "1",
        "--python", "blender/render_violation.py", "--", "config.json",
        "--scenario", scenario, "--lsp-count", str(count),
        "--output-stem", stem, "--samples", str(samples),
    ]
    rgba_path = work / "renders" / f"{stem}.png"
    if not reuse_render or not rgba_path.exists():
        subprocess.run(command, check=True)
    rgba = np.asarray(Image.open(rgba_path).convert("RGBA"))
    background = np.asarray(Image.open(config["background_image"]).convert("RGB"))
    if rgba.shape[:2] != background.shape[:2]:
        raise ValueError(f"Render size {rgba.shape[1]}x{rgba.shape[0]} does not match background")
    settings = config.get("composite", {})
    result, support = composite(
        background, rgba,
        sigma=settings.get("blur_sigma", .8),
        saturation=settings.get("object_saturation", .78),
        object_mask=rgba[..., 3] > 0,
        noise_sigma=settings.get("noise_sigma", .004),
        alpha_erosion_px=settings.get("alpha_erosion_px", 1),
        lens_k1=settings.get("lens_k1", -.035),
        ambient_strength=settings.get("ambient_strength", .45),
    )
    output = work / "out" / f"{stem}_clean_bg.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output, quality=92, subsampling=1)
    meta_path = work / "out" / f"{stem}.json"
    meta = json.loads(meta_path.read_text())
    meta["clean_background_poc"] = {
        "background": config["background_image"],
        "output_jpg": str(output),
        "rendered_forklift": True,
        "rendered_support_pixels": int(support.sum()),
        "workflow": "3D forklift and load rendered over a clean camera background",
    }
    meta["limitations"] = list(meta.get("limitations", []))
    meta["limitations"] = [
        item for item in meta["limitations"]
        if "Source plate contains distant scene objects" not in item
    ]
    meta["limitations"].append(
        "Source plate contains distant scene objects; only the main forklift placement area is clear."
    )
    meta["limitations"].append(
        "Forklift position and heading are manually calibrated for this camera; the real-forklift-anchor approach was unreliable when fork direction was unclear."
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    return output


def make_contact_sheet(outputs: list[tuple[str, Path]], destination: Path) -> None:
    images = [(label, Image.open(path).convert("RGB")) for label, path in outputs]
    tile_w, tile_h = 960, 540
    gutter, header = 18, 42
    sheet = Image.new("RGB", (gutter * 3 + tile_w * 2, gutter * 3 + (tile_h + header) * 2), "#111820")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, image) in enumerate(images):
        col, row = index % 2, index // 2
        x, y = gutter + col * (tile_w + gutter), gutter + row * (tile_h + header + gutter)
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        if image.size != (tile_w, tile_h):
            image = image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        draw.text((x, y + 5), label, font=font, fill="white")
        sheet.paste(image, (x, y + header))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "JPEG", quality=92, subsampling=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--reuse-renders", action="store_true",
                        help="Composite existing Blender PNGs instead of rendering the scenes again")
    parser.add_argument("--output", default="work/out/clean_background_three_scenes_preview.jpg")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    cases = [
        ("VALID · 1 LSP", "N1", 1, "clean_bg_valid_1lsp"),
        ("VIOLATION · 2 LSPs", "V1", 2, "clean_bg_violation_2lsp"),
        ("VIOLATION · 3 LSPs", "V2", 3, "clean_bg_violation_3lsp"),
    ]
    background = Path(config["background_image"])
    outputs: list[tuple[str, Path]] = [("CLEAN BACKGROUND", background)]
    for label, scenario, count, stem in cases:
        output = render_one(config, args.blender, scenario, count, stem, args.samples,
                            reuse_render=args.reuse_renders)
        outputs.append((label, output))
    destination = Path(args.output)
    make_contact_sheet(outputs, destination)
    print(json.dumps({"preview": str(destination), "panels": [label for label, _ in outputs]}, indent=2))


if __name__ == "__main__":
    main()
