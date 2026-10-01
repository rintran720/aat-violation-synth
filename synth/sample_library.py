"""Stage A1: multi-camera object sample library from recorded videos.

Run: python -m synth.sample_library --videos data/multicam/video/*.mp4 --every 3
Writes work/library/<class>/<camera>_<t>s_<k>.png (RGBA crop, alpha = SAM3 mask), work/library/index.json and
work/library/<class>_top.jpg contact sheets. Instances of one class that keep the same box in one camera
(a parked object seen in many frames) are merged and only the best-quality crop is kept.
"""
import argparse
import glob
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from synth.common import write_json

# class -> SAM3 text prompts (instances of all prompts are merged, duplicates dropped)
PROMPTS = {
    "forklift": ["forklift"],
    "lsp": ["black platform"],
    # No text prompt separates ULDs here: "container"/"large white box" also match wrapped pallet loads
    # (checked on ch10, 2026-10-01). ULD assets come from reviewed cam01 crops instead.
    "uld_container": ["air cargo container"],
    "carton": ["cardboard box"],
    "wrapped_cargo": ["stretch wrapped cargo"],
    "skid": ["wooden pallet"],
    "person": ["person"],
}
THRESHOLD = {"person": .5, "forklift": .5}
CARGO = ["uld_container", "wrapped_cargo", "carton"]   # one physical object may match several cargo prompts
LSP_MAX_CARGO_OVERLAP = .3   # an LSP mask mostly covered by cargo is a loaded stack, not a usable sheet texture
LSP_MAX_MEAN_VALUE = .5      # LSP sheets are dark grey/blue; brighter masks are wrapped cargo or floor
MIN_AREA = 1200          # px; smaller crops carry no usable texture
SAME_OBJECT_IOU = .8     # same camera + class + box overlap above this = the same parked object


def extract_frames(video, every, dest):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", f"fps=1/{every}",
                    "-q:v", "2", str(dest / "f_%05d.jpg")], check=True)
    return sorted(dest.glob("f_*.jpg"))


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.


def sharpness(gray, mask):
    """Variance of a 4-neighbour Laplacian inside the (eroded) mask."""
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    inner = mask.copy(); inner[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return float(lap[inner].var()) if inner.sum() > 20 else 0.


def quality(rec):
    """Bigger, sharper, confident, uncut crops rank first."""
    q = np.log(rec["area_px"]) * min(rec["sharpness"], 400) ** .5 * rec["score"]
    return float(q * (.4 if rec["touches_border"] else 1.))


def segment(processor, image, cls):
    state = processor.set_image(image); masks, scores = [], []
    processor.confidence_threshold = THRESHOLD.get(cls, .4)
    for p in PROMPTS[cls]:
        out = processor.set_text_prompt(state=state, prompt=p)
        for m, s in zip(out["masks"], out["scores"]):
            m = np.asarray(m.detach().cpu().float()).squeeze() > .5
            if m.sum() >= MIN_AREA and all((m & k).sum() < .5 * (m | k).sum() for k in masks):
                masks.append(m); scores.append(float(s))
    return masks, scores


def frame_instances(processor, image, skid_filter, classes=tuple(PROMPTS)):
    """All classes for one frame, with cross-class rules: each cargo object keeps its best-scoring class,
    LSP masks under cargo are dropped, SKID masks must be wood-coloured (config sam3_filters.SKID)."""
    found = {cls: list(zip(*segment(processor, image, cls))) for cls in classes}
    cargo = sorted(((s, cls, m) for cls in CARGO if cls in found for m, s in found[cls]), key=lambda x: -x[0]); taken = []
    for s, cls, m in cargo:
        if all((m & k).sum() < .5 * min(m.sum(), k.sum()) for _, _, k in taken):
            taken.append((s, cls, m))
    for cls in CARGO:
        if cls in found:
            found[cls] = [(m, s) for s, c, m in taken if c == cls]
    cargo_union = np.zeros(np.asarray(image).shape[:2], bool)
    for _, _, m in taken:
        cargo_union |= m
    value = np.asarray(image.convert("HSV"), dtype=np.float32)[..., 2] / 255.
    found["lsp"] = [(m, s) for m, s in found.get("lsp", []) if (m & cargo_union).sum() < LSP_MAX_CARGO_OVERLAP * m.sum()
                    and value[m].mean() < LSP_MAX_MEAN_VALUE]
    if skid_filter and "skid" in found:
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.
        h, sat, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
        wood = ((h >= skid_filter["hue_deg"][0]) & (h <= skid_filter["hue_deg"][1]) & (sat >= skid_filter["sat"][0])
                & (sat <= skid_filter["sat"][1]) & (v >= skid_filter["val_min"]))
        found["skid"] = [(m, s) for m, s in found["skid"] if wood[m].mean() >= skid_filter["min_colour_fraction"]]
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--every", type=float, default=3.0, help="seconds between sampled frames")
    parser.add_argument("--out", default="work/library")
    parser.add_argument("--classes", nargs="*", choices=list(PROMPTS), help="subset of classes; default all")
    parser.add_argument("--append", action="store_true",
                        help="replace only --classes in an existing index (keeps its other crops and camera colour stats)")
    args = parser.parse_args()
    classes = args.classes or list(PROMPTS)
    import torch
    from synth.segment import load_processor
    torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16).__enter__()
    processor = load_processor()
    skid_filter = json.loads(Path("config.json").read_text()).get("sam3_filters", {}).get("SKID")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    kept = {}        # (camera, class) -> list of records
    cameras = {}
    videos = [v for pattern in args.videos for v in sorted(glob.glob(pattern))]
    for video in videos:
        cam = Path(video).stem.split("_")[0]
        with tempfile.TemporaryDirectory() as tmp:
            frames = extract_frames(video, args.every, Path(tmp))
            floor_rgb = []
            for idx, frame in enumerate(frames):
                t = round((idx + .5) * args.every, 1)  # ffmpeg fps=1/N picks the frame in the middle of each N-s slot
                image = Image.open(frame).convert("RGB"); pixels = np.asarray(image)
                gray = np.asarray(image.convert("L"), dtype=np.float32)
                occupied = np.zeros(gray.shape, bool)
                for cls, inst in frame_instances(processor, image, skid_filter, classes).items():
                    for k, (m, s) in enumerate(inst):
                        occupied |= m
                        ys, xs = np.nonzero(m); box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                        rec = {"class": cls, "camera": cam, "video": Path(video).name, "t_s": t, "box": box,
                               "area_px": int(m.sum()), "score": round(s, 3), "sharpness": round(sharpness(gray, m), 1),
                               "touches_border": bool(box[0] <= 2 or box[1] <= 2 or box[2] >= m.shape[1] - 2 or box[3] >= m.shape[0] - 2),
                               "bottom_center_px": [int(round(xs.mean())), int(ys.max())]}
                        rec["quality"] = round(quality(rec), 2)
                        group = kept.setdefault((cam, cls), [])
                        same = next((r for r in group if box_iou(r["box"], box) > SAME_OBJECT_IOU), None)
                        if same is not None:
                            same["seen"] += 1
                            if rec["quality"] <= same["quality"]:
                                continue
                            rec["seen"] = same["seen"]; group.remove(same)
                            Path(same["crop"]).unlink(missing_ok=True)
                        else:
                            rec["seen"] = 1
                        x0, y0, x1, y1 = box
                        crop = np.dstack([pixels[y0:y1, x0:x1], m[y0:y1, x0:x1].astype(np.uint8) * 255])
                        name = out / cls / f"{cam}_{t:06.1f}s_{k}.png"; name.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(crop, "RGBA").save(name); rec["crop"] = name.as_posix(); group.append(rec)
                # floor colour of this frame: lower half of the image outside every object mask
                lower = np.zeros_like(occupied); lower[gray.shape[0] // 2:] = True
                floor_rgb.append(np.median(pixels[lower & ~occupied], axis=0))
                print(f"{cam} {t:6.1f}s: {sum(len(v) for (c, _), v in kept.items() if c == cam)} kept", flush=True)
            cameras[cam] = {"video": Path(video).name, "frames": len(frames),
                            "floor_rgb_median": np.median(floor_rgb, axis=0).round(1).tolist()}
    records = [r for group in kept.values() for r in group]
    if args.append:
        old = json.loads((out / "index.json").read_text())
        for r in old["crops"]:
            if r["class"] in classes and r["crop"] not in {x["crop"] for x in records}:
                Path(r["crop"]).unlink(missing_ok=True)
        records += [r for r in old["crops"] if r["class"] not in classes]
        cameras = old["cameras"]
    records.sort(key=lambda r: (r["class"], -r["quality"]))
    # colour normalisation: per-camera gains that map each camera's floor colour to the all-camera median floor
    ref = np.median([c["floor_rgb_median"] for c in cameras.values()], axis=0)
    for c in cameras.values():
        c["rgb_gain_to_reference"] = (ref / np.maximum(c["floor_rgb_median"], 1)).round(3).tolist()
    write_json(out / "index.json", {"prompts": PROMPTS, "every_s": args.every, "cameras": cameras,
                                    "reference_floor_rgb": ref.round(1).tolist(), "crops": records})
    for cls in classes:
        top = [r for r in records if r["class"] == cls][:40]
        if not top: continue
        sheet = Image.new("RGB", (8 * 200, ((len(top) + 7) // 8) * 220), "white"); d = ImageDraw.Draw(sheet)
        for i, r in enumerate(top):
            im = Image.open(r["crop"]).convert("RGBA"); im.thumbnail((190, 190))
            x, y = (i % 8) * 200, (i // 8) * 220; sheet.paste(im, (x + 5, y + 5), im)
            d.text((x + 5, y + 200), f"{r['camera']} {r['t_s']}s q{r['quality']:.0f}", fill="black")
        sheet.save(out / f"{cls}_top.jpg", quality=88)
    print(json.dumps({cls: sum(r["class"] == cls for r in records) for cls in PROMPTS}))


if __name__ == "__main__":
    main()
