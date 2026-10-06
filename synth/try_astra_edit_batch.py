"""Experiment, batch form of synth.try_astra_edit: N valid frames -> N Astra edits, all outputs in one folder.

Run: python -m synth.try_astra_edit_batch <frame or folder> [...] [--out work/out/astra_batch_<UTC time>]
     [--jobs 3] [--seed 0] [--scene "..."] [--change "..."]
For every frame, as try_astra_edit does for one: images 2-3 are the LSP and SKID of synth.catalogue_reference_map
(no contact edge is measured, so the renders are matched at the image centre; --seed + the frame's index picks the
models, so the frames get different ones), the prompt follows try_astra_edit's reference-images template with a
scene and a change that fit any frame, and Astra (gpt-6-astra) edits image 1 with imagegen. No measuring, no checks:
each result is only resized to its frame's size.
No cargo render goes in: the new cargo copies a load already in the frame (its kind, look and height), so it has
that load's CCTV blur and noise; a catalogue cargo came out sharper than the frame. The new LSP takes its size from
the loaded LSP in the frame, not from the render. The prompt names the LSP and SKID by their catalogue kind (in the
words of work/catalogue3d/README.md).
Writes <out>/outputs/<frame stem>.png (the N outputs together), <out>/runs/<frame stem>/ (refs/, prompt.txt,
astra.log, astra.txt, raw.png) and <out>/results.json (per frame: output or error, seconds).
Run again with the same --out to continue: frames that already have an output are skipped, an unfinished run folder
is redone, and results.json keeps the earlier rows.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from synth.try_astra_edit import ROOT

SCENE = "a counterbalance forklift carrying ONE LSP loaded with cargo on its forks"
# the catalogue kinds (work/catalogue3d/variants.json "object"), described as in work/catalogue3d/README.md
KINDS = {
    "lsp": "a flat dark grey / blue-black plastic slip sheet, about 9 cm thin, with rounded corners and a worn, "
           "scuffed top",
    "skid": "a wooden base of top deck boards on 3 runners, with no bottom deck",
}
PROMPT = """You are given 3 reference images from a real CCTV camera in an air-cargo terminal warehouse:
1. Full scene: the warehouse floor with {scene}.
2. An LSP (load spreader plate): {lsp}.
3. A skid: {skid}.
Images 2-3 are 3D renders of real objects from this warehouse: the large tile is the main view, the small tiles are
other views of the SAME object. Copy only their appearance (colour, material, texture); they are not to scale, and
neither their size nor their viewpoint may be copied.

Edit image 1 only. Keep the SAME fixed high-angle CCTV viewpoint, lighting, colors and image quality (slightly blurry
wide-angle security-camera look). Do not change the camera angle, the floor, the red laser lines, the background,
the other objects or the people. Keep the full frame size and aspect ratio.

Make this ONE change: {change}

Everything else stays exactly as in image 1. No labels, bounding boxes, text or watermarks."""
CHANGE = ("the forklift that carries the loaded LSP now carries TWO LSPs at the same time: a second LSP (like image 2) "
          "lies flat directly in front of the loaded one in the direction the forks point, its rear edge touching the "
          "loaded sheet's front edge along its full width. The new LSP has EXACTLY the size of the loaded LSP in "
          "image 1: the same length, width and thickness, the same orientation (its sides parallel to the loaded "
          "sheet's) and the same perspective at that spot, so the two plates look like two identical sheets laid end "
          "to end; it is not smaller, not larger, not wider and not deeper. Take its size from the loaded LSP in image "
          "1, never from image 2. On the new LSP stands one skid (like image 3) carrying one cargo load, centred on the "
          "sheet. The cargo copies the real loads already in image 1: the same kind of load (for example "
          "film-wrapped, a carton stack, a crate), the same look, colour and wrapping, and about the same height as "
          "the load on the forklift, with the same CCTV blur, noise and lighting as those loads, so it looks like "
          "one more of the loads that are already in the warehouse. Both LSPs must be clearly visible as two "
          "distinct flat plates, with a natural contact shadow. Every other forklift, LSP and load stays as it is.")
IMAGE_TYPES = (".jpg", ".jpeg", ".png")


def frames(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if path.is_dir():
            found += sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_TYPES)
        elif path.is_file():
            found.append(path)
        else:
            raise SystemExit(f"no such frame or folder: {path}")
    return [p.resolve() for p in found]


def reference_prompt(args: argparse.Namespace, reference_map: Path) -> str:
    """The prompt for one frame: the LSP and SKID references named by their catalogue kind."""
    entries = json.loads(reference_map.read_text())
    variants = json.loads((ROOT / "work/catalogue3d/variants.json").read_text())
    def kind(category: str) -> str:
        return variants[entries[category]["variant"]]["object"]
    return PROMPT.format(scene=args.scene, lsp=KINDS[kind("LSP")], skid=KINDS[kind("SKID")],
                         change=args.change or CHANGE)


def edit(image: Path, seed: int, args: argparse.Namespace, out: Path) -> dict:
    """try_astra_edit for one frame; returns its result row."""
    started = time.time()
    run = out / "runs" / image.stem
    if run.exists():          # an earlier run stopped before this frame's output
        shutil.rmtree(run)
    run.mkdir(parents=True)
    row = {"input": str(image), "seed": seed}
    try:
        command = [sys.executable, "-m", "synth.catalogue_reference_map", "--image", str(image), "--seed", str(seed),
                   "--out", str(run / "refs")]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        images = [image] + [run / "refs" / f"{category}.png" for category in ("LSP", "SKID")]
        prompt = reference_prompt(args, run / "refs" / "reference_map.json")
        (run / "prompt.txt").write_text(prompt)
        raw = run / "raw.png"
        command = [args.codex_bin, "exec", "--ephemeral", "-m", "gpt-6-astra", "-s", "workspace-write", "-C", str(ROOT)]
        for path in images:
            command += ["-i", str(path)]
        command += ["-o", str(run / "astra.txt"),
                    "Use image_gen.imagegen to EDIT Image 1. Pass these image paths as referenced_image_paths in the "
                    "supplied order, transparent_background=false: " + ", ".join(map(str, images)) + ".\n\n" + prompt
                    + f"\n\nSave the generated result as a nonempty PNG at {raw}. Do not edit any other repository "
                    "file or use Blender/3D compositing."]
        with (run / "astra.log").open("w") as log:
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code or not raw.is_file() or not raw.stat().st_size:
            raise RuntimeError(f"no image from Astra (codex exit {code}); see {run / 'astra.log'}")
        with Image.open(image) as source, Image.open(raw) as edited:
            result = edited.convert("RGB")
            row["raw_size"] = list(result.size)
            if result.size != source.size:
                result = result.resize(source.size, Image.Resampling.LANCZOS)
        output = out / "outputs" / f"{image.stem}.png"
        result.save(output, format="PNG")
        row["output"] = str(output)
    except subprocess.CalledProcessError as error:
        row["error"] = f"reference map failed: {(error.stderr or '').strip()[-300:]}"
    except (RuntimeError, OSError, KeyError) as error:
        row["error"] = str(error)
    row["seconds"] = round(time.time() - started)
    print(f"{image.name}: " + (f"-> {row['output']}" if "output" in row else f"FAILED {row['error']}")
          + f" ({row['seconds']} s)", flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+", help="frames, or folders searched for .jpg/.png frames")
    parser.add_argument("--out", type=Path, help="default work/out/astra_batch_<UTC time>")
    parser.add_argument("--jobs", type=int, default=3, help="frames edited at the same time")
    parser.add_argument("--seed", type=int, default=0, help="catalogue seed of the first frame; +1 per frame")
    parser.add_argument("--scene", default=SCENE, help="what image 1 shows, after 'the warehouse floor with'")
    parser.add_argument("--change", help="the ONE edit to make; default: the two-LSP edit, cargo copied from the frame")
    parser.add_argument("--codex-bin", default="codex")
    args = parser.parse_args()

    images = frames(args.inputs)
    if not images:
        raise SystemExit("no frames found")
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise SystemExit("two frames share a file name; outputs are named by it")
    out = (args.out or ROOT / "work/out" / f"astra_batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit("the Codex image editor needs --out inside the workspace")
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    todo = [(i, p) for i, p in enumerate(images) if not (out / "outputs" / f"{p.stem}.png").is_file()]
    print(f"{len(todo)} of {len(images)} frames to edit ({len(images) - len(todo)} already done), "
          f"{args.jobs} at a time -> {out / 'outputs'}", flush=True)
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        new_rows = list(pool.map(lambda item: edit(item[1], args.seed + item[0], args, out), todo))
    results = out / "results.json"
    rows = {row["input"]: row for row in (json.loads(results.read_text()) if results.is_file() else [])}
    rows.update({row["input"]: row for row in new_rows})
    results.write_text(json.dumps(list(rows.values()), indent=2))
    done = sum("output" in row for row in new_rows)
    print(f"{done}/{len(new_rows)} new outputs in {out / 'outputs'}; per frame: {results}")
    return 0 if done == len(new_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
