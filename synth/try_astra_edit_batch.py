"""Experiment, batch form of synth.try_astra_edit: N valid frames -> N Astra edits, all outputs in one folder.

Run: python -m synth.try_astra_edit_batch <frame or folder> [...] [--out work/out/astra_batch_<UTC time>]
     [--jobs 3] [--seed 0] [--engine forklift-pushing-multiple-lsps] [--case push_2_lsp_cargo]
     [--input-kind forklift-with-lsp-cargo] [--scene "..."] [--change "..."]
The ONE change is a case of synth.violation_cases for frames of one input kind (--engine, --case, --input-kind; the
kind also gives the scene), or --change. edit_frame() edits one frame and is what the generation service
(service/app.py) runs too.
For every frame, as try_astra_edit does for one: images 2-3 are the LSP and SKID of synth.catalogue_reference_map
(only the SKID for a case whose refs are just the SKID)
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
from synth.violation_cases import DEFAULT_CASE, DEFAULT_ENGINE, DEFAULT_INPUT, ENGINES, INPUT_KINDS, case, change

SCENE = INPUT_KINDS[DEFAULT_INPUT]["scene"]
# the catalogue kinds (work/catalogue3d/variants.json "object"), described as in work/catalogue3d/README.md
KINDS = {
    "lsp": "a flat dark grey / blue-black plastic slip sheet, about 9 cm thin, with rounded corners and a worn, "
           "scuffed top",
    "skid": "a wooden base of top deck boards on 3 runners, with no bottom deck",
}
REF_NAMES = {"LSP": "An LSP (load spreader plate)", "SKID": "A skid"}
RENDERS = {  # what the references are, for one and for several of them
    1: ("Image 2 is a 3D render of a real object from this warehouse: the large tile is the main view, the small "
        "tiles are\nother views of the SAME object. Copy only its appearance (colour, material, texture); it is not to "
        "scale, and\nneither its size nor its viewpoint may be copied."),
    2: ("Images 2-{last} are 3D renders of real objects from this warehouse: the large tile is the main view, the "
        "small tiles are\nother views of the SAME object. Copy only their appearance (colour, material, texture); they "
        "are not to scale, and\nneither their size nor their viewpoint may be copied."),
}
PROMPT = """You are given {count} reference images from a real CCTV camera in an air-cargo terminal warehouse:
1. Full scene: the warehouse floor with {scene}.
{references}
{renders}

Edit image 1 only. Keep the SAME fixed high-angle CCTV viewpoint, lighting, colors and image quality (slightly blurry
wide-angle security-camera look). Do not change the camera angle, the floor, the red laser lines, the background,
the other objects or the people. Keep the full frame size and aspect ratio.

Make this ONE change: {change}

Everything else stays exactly as in image 1. No labels, bounding boxes, text or watermarks."""
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


def reference_prompt(reference_map: Path, change: str, scene: str = SCENE, refs=("LSP", "SKID")) -> str:
    """The prompt for one frame: the references (images 2, 3, ...) named by their catalogue kind, then the ONE
    change."""
    entries = json.loads(reference_map.read_text())
    variants = json.loads((ROOT / "work/catalogue3d/variants.json").read_text())
    def kind(category: str) -> str:
        return variants[entries[category]["variant"]]["object"]
    references = "\n".join(f"{i}. {REF_NAMES[ref]}: {KINDS[kind(ref)]}." for i, ref in enumerate(refs, 2))
    renders = RENDERS[min(len(refs), 2)].format(last=len(refs) + 1)
    return PROMPT.format(count=len(refs) + 1, scene=scene, references=references, renders=renders, change=change)


def camera_sized(image: Path, run: Path) -> Path:
    """The frame at the catalogue camera's size: catalogue_reference_map matches its views with work/camera.json
    and refuses another size, so a frame of another size gets a resized copy for the references only."""
    camera = json.loads((ROOT / "work/camera.json").read_text())
    size = (int(camera["width"]), int(camera["height"]))
    with Image.open(image) as frame:
        if frame.size == size:
            return image
        copy = run / "frame_for_refs.png"
        frame.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(copy)
    return copy


def edit_frame(image: Path, run: Path, output: Path, seed: int, change: str, scene: str = SCENE,
               codex_bin: str = "codex", refs=("LSP", "SKID")) -> dict:
    """One Astra edit of one frame: the catalogue references in refs (seed picks the models) into run/, the edited
    frame at its own size to output. Returns its result row (output or error, seconds); never raises."""
    started = time.time()
    if run.exists():          # an earlier run stopped before this frame's output
        shutil.rmtree(run)
    run.mkdir(parents=True)
    row = {"input": str(image), "seed": seed}
    try:
        command = [sys.executable, "-m", "synth.catalogue_reference_map", "--image", str(camera_sized(image, run)),
                   "--seed", str(seed), "--out", str(run / "refs")]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        images = [image] + [run / "refs" / f"{category}.png" for category in refs]
        prompt = reference_prompt(run / "refs" / "reference_map.json", change, scene, refs)
        (run / "prompt.txt").write_text(prompt)
        raw = run / "raw.png"
        command = [codex_bin, "exec", "--ephemeral", "-m", "gpt-6-astra", "-s", "workspace-write", "-C", str(ROOT)]
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
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output, format="PNG")
        row["output"] = str(output)
    except subprocess.CalledProcessError as error:
        row["error"] = f"reference map failed: {(error.stderr or '').strip()[-300:]}"
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        row["error"] = str(error)
    row["seconds"] = round(time.time() - started)
    return row


def edit(image: Path, seed: int, args: argparse.Namespace, out: Path) -> dict:
    """edit_frame for one frame of a batch, printing its result."""
    row = edit_frame(image, out / "runs" / image.stem, out / "outputs" / f"{image.stem}.png", seed, args.change_text,
                     args.scene, args.codex_bin, args.refs)
    print(f"{image.name}: " + (f"-> {row['output']}" if "output" in row else f"FAILED {row['error']}")
          + f" ({row['seconds']} s)", flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+", help="frames, or folders searched for .jpg/.png frames")
    parser.add_argument("--out", type=Path, help="default work/out/astra_batch_<UTC time>")
    parser.add_argument("--jobs", type=int, default=3, help="frames edited at the same time")
    parser.add_argument("--seed", type=int, default=0, help="catalogue seed of the first frame; +1 per frame")
    parser.add_argument("--scene", help="what image 1 shows, after 'the warehouse floor with'; default: the "
                                         "input kind's")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, choices=sorted(ENGINES))
    parser.add_argument("--case", default=DEFAULT_CASE, help="a case of the engine (synth/violation_cases.py)")
    parser.add_argument("--input-kind", default=DEFAULT_INPUT, choices=sorted(INPUT_KINDS),
                        help="what the frames show; picks the case's change and the scene")
    parser.add_argument("--change", help="the ONE edit to make, replacing --case")
    parser.add_argument("--codex-bin", default="codex")
    args = parser.parse_args()
    try:
        args.change_text = args.change or change(args.engine, args.case, args.input_kind)
        args.refs = case(args.engine, args.case)["refs"]
    except KeyError as error:
        raise SystemExit(str(error).strip('"'))
    args.scene = args.scene or INPUT_KINDS[args.input_kind]["scene"]

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
