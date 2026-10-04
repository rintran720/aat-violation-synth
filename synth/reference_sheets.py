"""Review sheets: one image per object class from the sample library, chosen to cover as many viewpoints as possible.

Run: python -m synth.reference_sheets [--per-class 30]
Viewpoint = camera x where the object stands in the frame (3x3 grid of its bottom centre: near/far, left/right) x
crop shape (tall / square / wide, roughly its heading). Each (camera, cell, shape) bucket gives its best crop by
library quality, buckets are taken round-robin over cameras, and crops of one object seen again within 15 s at the
same spot are skipped. Every tile is labelled camera, time, size and bucket. Writes work/library/reference/<class>.jpg
and a summary of crops per class and camera (summary.jpg).
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

LIB = Path("work/library")
CLASSES = {"forklift": "forklift", "lsp": "LSP (black slip sheet)", "wrapped_cargo": "cargo - stretch wrapped",
           "carton": "cargo - cartons", "skid": "SKID (wooden pallet)", "uld_container": "ULD (air cargo container)",
           "person": "person"}
TILE, COLS, BG = 220, 6, (128, 130, 126)
FRAME_W, FRAME_H = 1920, 1080


def bucket(r):
    x, y = r["bottom_center_px"]; w, h = r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]
    cell = ("far", "mid", "near")[min(2, int(3 * y / FRAME_H))] + "-" + ("left", "centre", "right")[min(2, int(3 * x / FRAME_W))]
    shape = "tall" if h > 1.3 * w else "wide" if w > 1.3 * h else "square"
    return r["camera"], cell, shape


def pick(rows, n):
    groups = defaultdict(list)
    for r in sorted(rows, key=lambda r: -r["quality"]):
        groups[bucket(r)].append(r)
    by_cam = defaultdict(list)
    for key, rs in sorted(groups.items(), key=lambda kv: -kv[1][0]["quality"]):
        by_cam[key[0]].append(rs[0])
    out, seen = [], []
    while len(out) < n and any(by_cam.values()):
        for cam in sorted(by_cam):
            if by_cam[cam] and len(out) < n:
                r = by_cam[cam].pop(0)
                if any(s["camera"] == r["camera"] and abs(s["t_s"] - r["t_s"]) < 15
                       and math.dist(s["bottom_center_px"], r["bottom_center_px"]) < 80 for s in seen):
                    continue
                out.append(r); seen.append(r)
    return out


def sheet(cls, rows, n):
    chosen = pick(rows, n)
    cams = Counter(r["camera"] for r in rows)
    head = (f"{CLASSES.get(cls, cls)}: {len(rows)} crops in library  |  per camera: "
            + ", ".join(f"{c} {cams[c]}" for c in sorted(cams)) + f"  |  shown: {len(chosen)} (diverse viewpoints)")
    nrows = max(1, math.ceil(len(chosen) / COLS))
    img = Image.new("RGB", (COLS * TILE, 40 + nrows * (TILE + 30)), BG); d = ImageDraw.Draw(img)
    d.text((8, 12), head if rows else f"{CLASSES.get(cls, cls)}: NO crops in the library", fill=(255, 255, 255))
    for i, r in enumerate(chosen):
        x, y = (i % COLS) * TILE, 40 + (i // COLS) * (TILE + 30)
        im = Image.open(r["crop"]).convert("RGBA"); im.thumbnail((TILE - 12, TILE - 12))
        bg = Image.new("RGBA", im.size, BG + (255,)); bg.alpha_composite(im)
        img.paste(bg.convert("RGB"), (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2))
        cam, cell, shape = bucket(r); w, h = r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]
        d.text((x + 6, y + TILE + 2), f"{cam} {r['t_s']:.0f}s {w}x{h}px", fill=(255, 255, 255))
        d.text((x + 6, y + TILE + 15), f"{cell} {shape}", fill=(230, 230, 180))
    return img, len(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=30)
    args = parser.parse_args()
    lib = json.loads((LIB / "index.json").read_text())
    crops = [r for r in lib["crops"] if not r["touches_border"]]
    dest = LIB / "reference"; dest.mkdir(exist_ok=True)
    cams = sorted(lib["cameras"])
    rows_txt = []
    for cls in CLASSES:
        rows = [r for r in crops if r["class"] == cls]
        img, k = sheet(cls, rows, args.per_class)
        img.save(dest / f"{cls}.jpg", quality=90)
        cnt = Counter(r["camera"] for r in rows)
        rows_txt.append((CLASSES[cls], [cnt[c] for c in cams], len({bucket(r) for r in rows})))
        print(f"{cls}: {len(rows)} crops, {len({bucket(r) for r in rows})} viewpoint buckets, {k} shown")
    s = Image.new("RGB", (900, 60 + 26 * len(rows_txt)), BG); d = ImageDraw.Draw(s)
    d.text((10, 10), "crops per camera (not touching the frame border)        viewpoint buckets", fill=(255, 255, 255))
    d.text((10, 32), "object".ljust(28) + "".join(c.rjust(7) for c in cams), fill=(255, 255, 255))
    for i, (name, n, b) in enumerate(rows_txt):
        d.text((10, 58 + 26 * i), name[:27].ljust(28) + "".join(str(v).rjust(7) for v in n) + str(b).rjust(10),
               fill=(255, 120, 120) if sum(n) == 0 else (255, 255, 255))
    s.save(dest / "summary.jpg", quality=92)


if __name__ == "__main__":
    main()
