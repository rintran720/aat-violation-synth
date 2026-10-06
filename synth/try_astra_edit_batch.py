"""Experiment, batch form of synth.try_astra_edit: N valid frames -> N Astra edits, all outputs in one folder.

Run: python -m synth.try_astra_edit_batch <frame or folder> [...] [--out work/out/astra_batch_<UTC time>]
     [--jobs 3] [--seed 0] [--cargo-kind cargo_wrap] [--scene "..."] [--change "..."]
For every frame, as try_astra_edit does for one: images 2-4 are the LSP, SKID and cargo of
synth.catalogue_reference_map (no contact edge is measured, so the renders are matched at the image centre;
--seed + the frame's index picks the models, so the frames get different ones), the prompt follows try_astra_edit's
"4 reference images" template with a scene and a change that fit any frame, and Astra (gpt-6-astra) edits image 1
with imagegen. No measuring, no checks: each result is only resized to its frame's size.
The prompt names what each reference is (the catalogue kind, in the words of work/catalogue3d/README.md) and gives
the cargo its catalogue height. The new cargo is told to copy image 4 and not the load already in the frame: with
only "as tall as the existing load" Astra copied that load and ignored the reference.
Writes <out>/outputs/<frame stem>.png (the N outputs together), <out>/runs/<frame stem>/ (refs/, prompt.txt,
astra.log, astra.txt, raw.png) and <out>/results.json (per frame: output or error, seconds).
"""
import argparse
import json
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
    "cargo_wrap": "a load wrapped in clear / grey stretch film that covers it whole, blurred labels behind the film, "
                  "soft rounded edges",
    "cargo_carton_stack": "a stack of cartons without film, each carton and its label visible, a grid of small boxes "
                          "on every side",
    "cargo_black_net": "a load in dark glossy black film and / or under a net with visible cords",
    "cargo_strapped": "a large tall crate with vertical straps or a frame and label panels / stickers",
    "cargo_wooden": "a brown / yellow wooden or heavy-carton crate with dark straps",
    "cargo_small_carton": "a few small loose cartons, often with tape bands",
}
PROMPT = """You are given 4 reference images from a real CCTV camera in an air-cargo terminal warehouse:
1. Full scene: the warehouse floor with {scene}.
2. An LSP (load spreader plate): {lsp}.
3. A skid: {skid}.
4. The cargo to add: {cargo}.
Images 2-4 are 3D renders of real objects from this warehouse: the large tile is the main view, the small tiles are
other views of the SAME object. Copy their appearance (shape, colour, material, film, straps, labels); they are not
to scale and their viewpoint must not be copied.

Edit image 1 only. Keep the SAME fixed high-angle CCTV viewpoint, lighting, colors and image quality (slightly blurry
wide-angle security-camera look). Do not change the camera angle, the floor, the red laser lines, the background,
the other objects or the people. Keep the full frame size and aspect ratio.

Make this ONE change: {change}

Everything else stays exactly as in image 1. No labels, bounding boxes, text or watermarks."""
CHANGE = ("the forklift that carries the loaded LSP now carries TWO LSPs at the same time: a second LSP (like image 2), "
          "identical in size, thickness and orientation to the loaded one, lies flat directly in front of it in the "
          "direction the forks point, its rear edge touching the loaded sheet's front edge along its full width. On "
          "the new LSP stands {on_sheet}, centred on the sheet, about {height:.1f} m tall (the LSP is 1.9 m wide, so "
          "about {ratio:.0%} of the LSP's width). The new cargo must look like image 4 ({cargo_kind}): its shape, "
          "colour, material and wrapping come from image 4, NOT from the load already on the forklift or any other "
          "load in image 1, even when they differ. Render it in the CCTV look of image 1 (same blur, noise and "
          "lighting). Both LSPs must be clearly visible as two distinct flat plates, with a natural contact shadow. "
          "Every other forklift, LSP and load stays as it is.")
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
    """The prompt for one frame: each reference named by its catalogue kind, the cargo given its catalogue height."""
    entries = json.loads(reference_map.read_text())
    variants = json.loads((ROOT / "work/catalogue3d/variants.json").read_text())
    def kind(category: str) -> str:
        return variants[entries[category]["variant"]]["object"]
    cargo_kind = kind("cargo")
    cargo = KINDS.get(cargo_kind, "a cargo load")
    with_skid = entries["cargo"]["variant"].endswith("_skid")
    if with_skid:
        cargo += ", shown standing on its skid"
    on_sheet = ("the cargo of image 4 on its skid" if with_skid
                else "one skid (like image 3) carrying the cargo of image 4")
    height = float(entries["cargo"]["size_m"][2])
    change = args.change or CHANGE.format(on_sheet=on_sheet, height=height, ratio=height / 1.9,
                                          cargo_kind=KINDS.get(cargo_kind, cargo_kind).split(",")[0])
    return PROMPT.format(scene=args.scene, lsp=KINDS[kind("LSP")], skid=KINDS[kind("SKID")], cargo=cargo,
                         change=change)


def edit(image: Path, seed: int, args: argparse.Namespace, out: Path) -> dict:
    """try_astra_edit for one frame; returns its result row."""
    started = time.time()
    run = out / "runs" / image.stem
    run.mkdir(parents=True)
    row = {"input": str(image), "seed": seed}
    try:
        command = [sys.executable, "-m", "synth.catalogue_reference_map", "--image", str(image), "--seed", str(seed),
                   "--out", str(run / "refs")]
        if args.cargo_kind:
            command += ["--cargo-kind", args.cargo_kind]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        images = [image] + [run / "refs" / f"{category}.png" for category in ("LSP", "SKID", "cargo")]
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
    parser.add_argument("--cargo-kind", help="catalogue cargo kind, e.g. cargo_wrap; default any")
    parser.add_argument("--scene", default=SCENE, help="what image 1 shows, after 'the warehouse floor with'")
    parser.add_argument("--change", help="the ONE edit to make; default: the two-LSP edit with this frame's cargo")
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
    (out / "outputs").mkdir(parents=True)
    print(f"{len(images)} frames, {args.jobs} at a time -> {out / 'outputs'}", flush=True)
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        rows = list(pool.map(lambda item: edit(item[1], args.seed + item[0], args, out), enumerate(images)))
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    done = sum("output" in row for row in rows)
    print(f"{done}/{len(rows)} outputs in {out / 'outputs'}; per frame: {out / 'results.json'}")
    return 0 if done == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
