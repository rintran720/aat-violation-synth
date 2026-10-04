"""Reference map for synth.valid_to_violation (--reference-map) built from the 3D catalogue, view-matched.

Run: python -m synth.catalogue_reference_map --image <valid frame> [--contact-edge x1,y1,x2,y2]
     [--camera work/camera.json] [--seed 0] [--cargo-kind cargo_wrap] [--cargo-skid yes|no|any]
Picks one labelled LSP, one SKID and one cargo model from work/catalogue3d (blender/build_catalogue.py) and,
for each, the render whose view is closest to how the camera sees the spot where the new LSP goes:
  - elevation: the camera's angle above the floor at the contact-edge midpoint (renders exist at 10, 30, 55 deg);
  - azimuth: the camera direction around the new sheet, whose X axis runs along the contact edge and whose Y axis
    along the edge normal (renders every 45 deg; boxes repeat every 180 deg).
Without --contact-edge the spot is the image centre with an edge parallel to the image rows.
Each LSP / SKID / cargo reference is ONE object: the matched view large, plus --views - 1 neighbouring views of the
same model as small tiles (one image per class keeps the editor's input short; never several different objects,
which an editor tends to blend). No text or measurements are drawn on the references: they are appearance-only;
geometry comes from the contact edge and calibration and is checked by SAM3 in valid_to_violation.
Renders are flattened on floor grey (image editors may drop alpha). The forklift is not in the catalogue, so it
keeps the Step A crop (work/refs/index.json), as valid_to_violation does by default.
Writes work/catalogue3d/reference_maps/<image stem>/: forklift/LSP/SKID/cargo PNGs, reference_map.json and
compare.jpg (Step A references above, catalogue references below).
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "work/catalogue3d"
ELEVATIONS = (10, 30, 55)
FLOOR_GREY = (128, 130, 126)


def floor_point(cam, px, py):
    """World point where pixel (px, py) meets the floor (z = 0), and the camera position."""
    K = cam["K_norm"]; M = np.array(cam["matrix_world"])
    ray = M[:3, :3] @ np.array([(px / cam["width"] - K[0][2]) / K[0][0], -(py / cam["height"] - K[1][2]) / K[1][1], -1.])
    origin = M[:3, 3]
    return origin - origin[2] / ray[2] * ray, origin


def view_angles(cam, edge):
    """(elevation, azimuth) in degrees of the camera seen from the new sheet's spot, in the catalogue's frame."""
    (x1, y1), (x2, y2) = edge
    a, origin = floor_point(cam, x1, y1); b, _ = floor_point(cam, x2, y2)
    mid = (a + b) / 2; to_cam = origin - mid
    elevation = math.degrees(math.atan2(to_cam[2], math.hypot(to_cam[0], to_cam[1])))
    ex = (b - a)[:2] / np.linalg.norm((b - a)[:2]); ey = np.array([-ex[1], ex[0]])
    dx, dy = to_cam[:2] @ ex, to_cam[:2] @ ey
    azimuth = math.degrees(math.atan2(dx, -dy)) % 360   # build_catalogue: camera at (sin az, -cos az)
    return elevation, azimuth


def nearest_view(elevation, azimuth):
    el = min(ELEVATIONS, key=lambda e: abs(e - elevation))
    az = int(round(azimuth / 45.)) * 45 % 360
    return el, az


def neighbour_views(el, az, n):
    """The matched view first, then the closest other views of the same model: azimuth +-45 deg, then the next
    elevation (shows the top and the sides of the same object)."""
    others = [(el, (az + 45) % 360), (el, (az - 45) % 360)]
    i = ELEVATIONS.index(el)
    others += [(ELEVATIONS[j], az) for j in (i + 1, i - 1) if 0 <= j < len(ELEVATIONS)]
    return [(el, az)] + others[:max(0, n - 1)]


def view_sheet(renders, dest, size=512):
    """One reference image per object: the matched view large on the left, the other views of the SAME object
    stacked on the right. No text on the image (an image editor may copy it into the result)."""
    tiles = [flatten(r, None) for r in renders]
    if len(tiles) == 1:
        tiles[0].save(dest); return dest
    small = size // (len(tiles) - 1)
    out = Image.new("RGB", (size + small, size), FLOOR_GREY); out.paste(tiles[0], (0, 0))
    for k, t in enumerate(tiles[1:]):
        out.paste(t.resize((small, small), Image.LANCZOS), (size, k * small))
    out.save(dest)
    return dest


def flatten(path, dest):
    im = Image.open(path).convert("RGBA"); bg = Image.new("RGBA", im.size, FLOOR_GREY + (255,)); bg.alpha_composite(im)
    box = im.getchannel("A").point(lambda a: 255 if a > 200 else 0).getbbox()   # the object, not its soft shadow
    out = bg.convert("RGB")
    if box:   # trim to the object with a margin, square, padded with floor grey where it leaves the render
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2; half = int(.6 * max(box[2] - box[0], box[3] - box[1]))
        sq = Image.new("RGB", (2 * half, 2 * half), FLOOR_GREY)
        sq.paste(out, (half - int(cx), half - int(cy)))
        out = sq.resize((512, 512), Image.LANCZOS)
    if dest is None:
        return out
    out.save(dest)
    return dest


def forklift_crop(index_path):
    """The Step A forklift crop that valid_to_violation would pick (largest confident one)."""
    entries = [e for e in json.loads(index_path.read_text()) if e.get("class") == "forklift"]
    entries.sort(key=lambda e: float(e.get("score", 0)) * max(int(e.get("area_px", 0)), 1) ** .5, reverse=True)
    for e in entries:
        if (ROOT / e["crop"]).is_file():
            return ROOT / e["crop"]
    raise SystemExit(f"no forklift crop in {index_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--contact-edge", help="x1,y1,x2,y2 of the original LSP's shared edge in image pixels")
    parser.add_argument("--camera", type=Path, default=ROOT / "work/camera.json")
    parser.add_argument("--step-a-index", type=Path, default=ROOT / "work/refs/index.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cargo-kind", help="e.g. cargo_wrap, cargo_wooden; default any")
    parser.add_argument("--cargo-skid", choices=["yes", "no", "any"], default="any")
    parser.add_argument("--views", type=int, default=3, help="views of the same object per reference image (1-5)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cam = json.loads(args.camera.read_text())
    with Image.open(args.image) as im:
        if im.size != (cam["width"], cam["height"]):
            raise SystemExit(f"{args.image} is {im.size}, the camera is {cam['width']}x{cam['height']}")
        source = im.convert("RGB")
    if args.contact_edge:
        x1, y1, x2, y2 = map(float, args.contact_edge.split(","))
    else:
        x1, y1, x2, y2 = cam["width"] * .4, cam["height"] * .5, cam["width"] * .6, cam["height"] * .5
    edge = [(x1, y1), (x2, y2)]
    elevation, azimuth = view_angles(cam, edge)
    el, az = nearest_view(elevation, azimuth)

    variants = json.loads((CAT / "variants.json").read_text()); rng = random.Random(args.seed)
    def pick(pred):
        names = sorted(n for n, v in variants.items() if pred(n, v))
        if not names:
            raise SystemExit("no catalogue variant matches")
        return rng.choice(names)
    picks = {
        "LSP": pick(lambda n, v: v["object"] == "lsp" and not n.startswith("lsp_stack")),
        "SKID": pick(lambda n, v: v["object"] == "skid" and not n.startswith("skid_stack")),
        "cargo": pick(lambda n, v: v["object"].startswith("cargo")
                      and (not args.cargo_kind or v["object"] == args.cargo_kind)
                      and (args.cargo_skid == "any" or n.endswith("_skid" if args.cargo_skid == "yes" else "_noskid"))),
    }
    out = args.out or CAT / "reference_maps" / args.image.stem; out.mkdir(parents=True, exist_ok=True)
    ref = {"forklift": {"path": str(forklift_crop(args.step_a_index).relative_to(ROOT).as_posix()), "source": "Step A crop"}}
    for cat, name in picks.items():
        v = variants[name]
        views = neighbour_views(el, az, args.views)
        renders = [CAT / "renders" / v["object"] / name / f"e{e:02d}_a{a:03d}.png" for e, a in views]
        dest = view_sheet(renders, out / f"{cat}.png")
        ref[cat] = {"path": str(dest.relative_to(ROOT).as_posix()), "source": "3D catalogue", "variant": name,
                    "label": v["label"], "size_m": v["size_m"],
                    "views": [{"elevation_deg": e, "azimuth_deg": a, "render": str(r.relative_to(ROOT).as_posix())}
                              for (e, a), r in zip(views, renders)]}
    ref["note"] = ("Appearance references only (colour, material, labels, film, deck boards). One object per image; "
                   "for LSP / SKID / cargo the large tile is the view matched to the camera and the small tiles are other "
                   "views of the SAME object. Not to scale and not for measuring: position, size and heading come from "
                   "the contact edge and the camera calibration, and are checked with SAM3.")
    ref["view"] = {"contact_edge_px": [list(map(round, p)) for p in edge], "edge_given": bool(args.contact_edge),
                   "camera": str(args.camera), "elevation_deg": round(elevation, 1), "azimuth_deg": round(azimuth, 1),
                   "render_view": {"elevation_deg": el, "azimuth_deg": az}, "seed": args.seed}
    (out / "reference_map.json").write_text(json.dumps(ref, indent=1))

    # comparison: Step A references (valid_to_violation default) above, catalogue references below
    from synth.valid_to_violation import step_a_references
    try:
        old = step_a_references(args.step_a_index)
    except RuntimeError:
        old = {}
    T = 260; sheet = Image.new("RGB", (5 * T, 2 * T + 60), (60, 60, 60)); d = ImageDraw.Draw(sheet)
    view = source.copy(); dd = ImageDraw.Draw(view); dd.line(edge, fill=(0, 255, 70), width=8)
    view.thumbnail((T, T)); sheet.paste(view, (0, 30))
    d.text((4, 4), f"camera view at the edge: elevation {elevation:.0f} deg, azimuth {azimuth:.0f} deg "
                   f"-> render e{el} a{az}", fill=(255, 255, 255))
    for c, cat in enumerate(["forklift", "LSP", "SKID", "cargo"], 1):
        for r, (label, path) in enumerate([("Step A", old.get(cat)), ("catalogue", ROOT / ref[cat]["path"])]):
            if path:
                im = Image.open(path).convert("RGBA"); im.thumbnail((T - 10, T - 10))
                bg = Image.new("RGBA", im.size, FLOOR_GREY + (255,)); bg.alpha_composite(im)
                sheet.paste(bg.convert("RGB"), (c * T + 5, 30 + r * (T + 15) + 5))
            d.text((c * T + 6, 30 + r * (T + 15) + T - 12), f"{label}: {cat}", fill=(255, 255, 160))
    sheet.save(out / "compare.jpg", quality=90)
    print(json.dumps({k: (v.get("variant") or v["path"]) if isinstance(v, dict) and "path" in v else v for k, v in ref.items()}, indent=1))
    print(f"-> {out / 'reference_map.json'}  (python -m synth.valid_to_violation <image> --reference-map {out / 'reference_map.json'})")


if __name__ == "__main__":
    main()
