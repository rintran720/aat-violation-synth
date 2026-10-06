"""Experiment: one Astra image edit of a valid frame with 3D catalogue references, no measuring and no checks.

Run: python -m synth.try_astra_edit [--image data/cam01/rtsp_samples/s_003.jpg] [--seed 0]
     [--contact-edge 877,461,1041,469] [--change "..."] [--codex-bin codex]
Image 1 is the frame; images 2-4 are the LSP, SKID and cargo of synth.catalogue_reference_map (renders whose view
matches the camera at --contact-edge, the loaded sheet's front edge). The prompt follows the "4 reference images"
template: describe the inputs, keep the CCTV look, make ONE change (--change). Astra (gpt-6-astra) runs imagegen
and saves the result; this script only records what went in and what came out.
Writes work/out/try_astra_<image stem>_<UTC time>/: refs/, prompt.txt, astra.txt, astra.log, output.png.
"""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCENE = ("the warehouse floor with a counterbalance forklift in the upper middle of the frame, facing the camera, "
         "carrying ONE LSP loaded with a white AKE container on its forks")
CHANGE = ("the forklift in the upper middle of the frame now carries TWO LSPs at the same time: a second LSP (like "
          "image 2), identical in size and orientation to the loaded one, lies flat directly in front of it on the "
          "side nearer the camera, its rear edge touching the loaded sheet's front edge along its full width. On the "
          "new LSP stands one skid (like image 3) carrying cargo (like image 4), centred on the sheet, about as tall "
          "as the existing container. Both LSPs must be clearly visible as two distinct flat plates, with a natural "
          "contact shadow.")
PROMPT = """You are given 4 reference images from a real CCTV camera in an air-cargo terminal warehouse:
1. Full scene: {scene}.
2. An LSP (load spreader plate): a flat grey metal plate.
3. A skid: a wooden/plastic pallet.
4. Cargo: the load that stands on a skid.
Images 2-4 are 3D renders showing appearance only (colour, material, texture); they are not to scale and their
viewpoint must not be copied.

Edit image 1 only. Keep the SAME fixed high-angle CCTV viewpoint, lighting, colors and image quality (slightly blurry
wide-angle security-camera look). Do not change the camera angle, the floor, the red laser lines, the background,
the other objects or the people. Keep the full frame size and aspect ratio.

Make this ONE change: {change}

Everything else stays exactly as in image 1. No labels, bounding boxes, text or watermarks."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=ROOT / "data/cam01/rtsp_samples/s_003.jpg")
    parser.add_argument("--contact-edge", default="877,461,1041,469",
                        help="the loaded LSP's front edge x1,y1,x2,y2 (s_003's, as source_edge measures it)")
    parser.add_argument("--seed", type=int, default=0, help="which catalogue LSP/SKID/cargo models")
    parser.add_argument("--scene", default=SCENE, help="what image 1 shows")
    parser.add_argument("--change", default=CHANGE, help="the ONE edit to make")
    parser.add_argument("--codex-bin", default="codex")
    args = parser.parse_args()

    image = args.image.resolve()
    work = ROOT / "work/out" / f"try_astra_{image.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    work.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "synth.catalogue_reference_map", "--image", str(image),
                    "--contact-edge", args.contact_edge, "--seed", str(args.seed), "--out", str(work / "refs")],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    images = [image] + [work / "refs" / f"{category}.png" for category in ("LSP", "SKID", "cargo")]
    prompt = PROMPT.format(scene=args.scene, change=args.change)
    (work / "prompt.txt").write_text(prompt)
    output = work / "output.png"

    command = [args.codex_bin, "exec", "--ephemeral", "-m", "gpt-6-astra", "-s", "workspace-write", "-C", str(ROOT)]
    for path in images:
        command += ["-i", str(path)]
    command += ["-o", str(work / "astra.txt"),
                "Use image_gen.imagegen to EDIT Image 1. Pass these image paths as referenced_image_paths in the "
                "supplied order, transparent_background=false: " + ", ".join(map(str, images)) + ".\n\n" + prompt
                + f"\n\nSave the generated result as a nonempty PNG at {output}. Do not edit any other repository "
                "file or use Blender/3D compositing."]
    print(f"Astra edit of {image.name} with catalogue refs (seed {args.seed}) -> {work}", flush=True)
    with (work / "astra.log").open("w") as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if code or not output.is_file() or not output.stat().st_size:
        print(f"error: no output (codex exit {code}); see {work / 'astra.log'}", file=sys.stderr)
        return 1
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
