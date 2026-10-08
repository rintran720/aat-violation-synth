"""One Astra image edit that turns a valid forklift frame into a two-LSP violation: exactly one output, no scoring,
no measuring of the result.

Run: python -m synth.astra_violation <valid frame> [--output work/out/<stem>_astra_violation.png]
     [--contact-edge x1,y1,x2,y2 | --sheet-at x,y] [--reference-map <map.json>]
     [--seed 0] [--cargo-kind cargo_wrap] [--cargo-skid yes|no|any] [--scene "..."] [--change "..."]
1. References: the LSP, SKID and cargo of a view-matched 3D catalogue map (synth.catalogue_reference_map), built for
   this frame from the loaded sheet's shared edge, or a given --reference-map. The edge is --contact-edge, else
   measured on the frame like valid_to_violation (SAM3 source_edge); when it cannot be measured, the renders are
   matched at the image centre and the prompt names no position.
2. Prompt, after the "4 reference images" template: what images 1-4 are, keep the CCTV viewpoint and look, make ONE
   change (--change; by default a second LSP with a SKID and cargo in front of the loaded one, along that edge).
3. Astra (gpt-6-astra, through the Codex CLI) edits image 1 with imagegen. The result is resized to the frame's size
   and written to --output; nothing else is checked or composited.
Writes the output and, beside it, <output stem>_run/: refs/ (the reference map), prompt.txt, astra.log, astra.txt,
raw.png (imagegen's own size) and run.json (inputs, edge, references, command time).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT / "work/camera.json"
STEP_A_INDEX = ROOT / "work/refs/index.json"
REF_CLASSES = ("LSP", "SKID", "cargo")

SCENE = "a counterbalance forklift carrying ONE LSP loaded with cargo on its forks"
CHANGE = ("the forklift now carries TWO LSPs at the same time: a second LSP (like image 2), identical in size, "
          "thickness and orientation to the loaded one, lies flat directly in front of it on the side nearer the "
          "camera, its rear edge touching the loaded sheet's front edge along its full width.{where} On the new LSP "
          "stands one skid (like image 3) carrying cargo (like image 4), centred on the sheet, about as tall as the "
          "existing load. Both LSPs must be clearly visible as two distinct flat plates, with a natural contact "
          "shadow. The existing forklift, LSP and load stay as they are.")
WHERE = " That front edge runs from {a} to {b} in the {w}x{h} frame (pixels)."
PROMPT = """You are given 4 reference images from a real CCTV camera in an air-cargo terminal warehouse:
1. Full scene: the warehouse floor with {scene}.
2. An LSP (load spreader plate): a flat grey metal plate.
3. A skid: a pallet.
4. Cargo: the load that stands on a skid.
Images 2-4 are 3D renders showing appearance only (colour, material, texture); they are not to scale and their
viewpoint must not be copied.

Edit image 1 only. Keep the SAME fixed high-angle CCTV viewpoint, lighting, colors and image quality (slightly blurry
wide-angle security-camera look). Do not change the camera angle, the floor, the red laser lines, the background,
the other objects or the people. Keep the full frame and its aspect ratio.

Make this ONE change: {change}

Everything else stays exactly as in image 1. No labels, bounding boxes, text or watermarks."""


def parse_edge(text: str, size: tuple[int, int]) -> list[list[int]]:
    values = [int(v.strip()) for v in text.split(",")]
    if len(values) != 4 or any(not 0 <= v < size[i % 2] for i, v in enumerate(values)):
        raise RuntimeError("--contact-edge must be x1,y1,x2,y2 within the frame")
    return [values[:2], values[2:]]


def measured_edge(source: Path, args: argparse.Namespace) -> tuple[list[list[int]] | None, str]:
    """The loaded sheet's shared edge as valid_to_violation measures it, or None with the reason."""
    import numpy as np
    from synth.valid_to_violation import Camera, sheet_position
    from synth.vision_geometry import Sam3Masks, source_edge
    os.environ.setdefault("HF_HOME", str(ROOT / "work/huggingface"))
    masks = Sam3Masks(ROOT / "work/vision_cache/sam3").get(source, ("forklift", "LSP", "cargo", "floor"))
    with Image.open(source) as image:
        size, gray = image.size, np.asarray(image.convert("L"))
    camera = Camera(args.camera)
    thickness = float(json.loads((ROOT / "config.json").read_text())["lsp_thickness_m"])
    try:
        edge, info = source_edge(masks, size, gray, lambda pixel: camera.lift(pixel, thickness),
                                 sheet_position(args, source))
    except RuntimeError as error:
        return None, str(error)
    return [[int(v) for v in p] for p in edge], info.get("method", "measured")


def references(source: Path, edge, args: argparse.Namespace, run: Path) -> tuple[dict[str, Path], Path]:
    if args.reference_map:
        manifest = args.reference_map.resolve()
    else:
        from synth.catalogue_reference_map import build_reference_map
        with Image.open(source) as image:
            w, h = image.size
        manifest = build_reference_map(source, edge or [[w * .4, h * .5], [w * .6, h * .5]], run / "refs",
                                       camera=args.camera, step_a_index=args.step_a_index, seed=args.seed,
                                       cargo_kind=args.cargo_kind, cargo_skid=args.cargo_skid,
                                       edge_given=edge is not None)
    entries = json.loads(manifest.read_text())
    refs = {}
    for category in REF_CLASSES:
        item = entries.get(category)
        if item is None:
            raise RuntimeError(f"reference map lacks {category}: {manifest}")
        path = Path(item["path"] if isinstance(item, dict) else item)
        refs[category] = path if path.is_absolute() else ROOT / path
        if not refs[category].is_file():
            raise RuntimeError(f"missing {category} reference: {refs[category]}")
    return refs, manifest


def build_prompt(args: argparse.Namespace, edge, size: tuple[int, int]) -> str:
    if args.change:
        change = args.change
    else:
        where = WHERE.format(a=tuple(edge[0]), b=tuple(edge[1]), w=size[0], h=size[1]) if edge else ""
        change = CHANGE.format(where=where)
    return PROMPT.format(scene=args.scene, change=change)


def astra_command(args: argparse.Namespace, images: list[Path], prompt: str, raw: Path, run: Path) -> list[str]:
    command = [args.codex_bin, "exec", "--ephemeral", "-m", args.model, "-s", "workspace-write", "-C", str(ROOT)]
    for path in images:
        command += ["-i", str(path)]
    return command + ["-o", str(run / "astra.txt"),
                      "Use image_gen.imagegen to EDIT Image 1. Pass these image paths as referenced_image_paths in "
                      "the supplied order, transparent_background=false: " + ", ".join(map(str, images)) + ".\n\n"
                      + prompt + f"\n\nSave the generated result as a nonempty PNG at {raw}. Do not edit any other "
                      "repository file or use Blender/3D compositing."]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="valid CCTV frame: a forklift carrying one loaded LSP")
    parser.add_argument("--output", type=Path, help="output PNG; default work/out/<input stem>_astra_violation.png")
    parser.add_argument("--contact-edge", help="the loaded LSP's shared (front) edge x1,y1,x2,y2; default measured")
    parser.add_argument("--sheet-at", help="x,y of a pixel on the loaded LSP, to pick it when measuring the edge")
    parser.add_argument("--reference-map", type=Path, help="ready reference map; default built from work/catalogue3d")
    parser.add_argument("--seed", type=int, default=0, help="seed of the catalogue's model choice")
    parser.add_argument("--cargo-kind", help="catalogue cargo kind, e.g. cargo_wrap, cargo_wooden; default any")
    parser.add_argument("--cargo-skid", choices=("yes", "no", "any"), default="any")
    parser.add_argument("--scene", default=SCENE, help="what image 1 shows, after 'the warehouse floor with'")
    parser.add_argument("--change", help="the ONE edit, replacing the default two-LSP instruction")
    parser.add_argument("--camera", type=Path, default=CAMERA)
    parser.add_argument("--step-a-index", type=Path, default=STEP_A_INDEX)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--codex-bin", default="codex")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    source = args.input.resolve()
    if not source.is_file():
        raise RuntimeError(f"input does not exist: {source}")
    if not shutil.which(args.codex_bin):
        raise RuntimeError(f"Codex CLI is unavailable: {args.codex_bin}")
    output = (args.output or ROOT / "work/out" / f"{source.stem}_astra_violation.png").resolve()
    if output.exists():
        raise RuntimeError(f"output already exists; choose --output: {output}")
    run = output.with_name(f"{output.stem}_run")
    if not run.is_relative_to(ROOT):
        raise RuntimeError("the Codex image editor needs the output inside the workspace")
    run.mkdir(parents=True)
    with Image.open(source) as image:
        size = image.size

    if args.contact_edge:
        edge, edge_method = parse_edge(args.contact_edge, size), "given"
    else:
        edge, edge_method = measured_edge(source, args)
        if edge is None:
            print(f"warning: shared edge not measured ({edge_method}); references matched at the image centre, "
                  "no position in the prompt", file=sys.stderr)
    refs, manifest = references(source, edge, args, run)
    prompt = build_prompt(args, edge, size)
    (run / "prompt.txt").write_text(prompt)
    images = [source] + [refs[c] for c in REF_CLASSES]
    raw = run / "raw.png"
    report = {"input": str(source), "output": str(output), "edge_px": edge, "edge_method": edge_method,
              "reference_map": str(manifest), "images": [str(p) for p in images], "model": args.model}
    (run / "run.json").write_text(json.dumps(report, indent=2))

    print(f"Astra edit of {source.name} -> {output}", flush=True)
    started = time.time()
    with (run / "astra.log").open("w") as log:
        code = subprocess.run(astra_command(args, images, prompt, raw, run), cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    report["seconds"] = round(time.time() - started)
    report["codex_exit"] = code
    (run / "run.json").write_text(json.dumps(report, indent=2))
    if code or not raw.is_file() or not raw.stat().st_size:
        raise RuntimeError(f"Astra produced no image (codex exit {code}); see {run / 'astra.log'}")
    with Image.open(raw) as edited:
        result = edited.convert("RGB")
    report["raw_size"] = list(result.size)
    if result.size != size:
        result = result.resize(size, Image.Resampling.LANCZOS)
    result.save(output, format="PNG")
    (run / "run.json").write_text(json.dumps(report, indent=2))
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
