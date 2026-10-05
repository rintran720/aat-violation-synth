"""Find valid source frames for synth.valid_to_violation in recorded videos.

Run: python -m synth.find_valid_frames --videos "data/multicam/video/*_20261005_*.mp4" [--every 2] [--out work/valid_frames]

A frame is kept when one forklift pushes exactly ONE LSP (with or without cargo on it) and the floor ahead of
that LSP is free for a second, equal sheet:
- attached: an LSP mask touches the forklift mask, or touches a cargo that touches the forklift;
- exactly one attached LSP of at least MIN_LSP_AREA px, and no other LSP touches it (a second sheet would
  already be a violation);
- pushed: in the previous or next sample the same pair is found and the LSP moved together with the forklift;
- free ahead: the LSP mask shifted by its own size away from the forklift stays in the frame and does not hit
  another forklift, LSP, cargo or person mask.
Passing samples of one push are grouped and the best one (freest, largest sheet) is kept.
Writes <out>/<camera>/<video stem>_<t>s.jpg (the frame), ..._overlay.jpg (masks and the free region),
<out>/index.json and <out>/<camera>_sheet.jpg. The screen is 2D and approximate; review the sheets.
"""
import argparse
import glob
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from synth.common import write_json
from synth.vision_geometry import PROMPTS as GEOMETRY_PROMPTS

# On ch10-ch16 most LSPs are grey or blue metal sheets that "black platform" misses (checked on ch11, 2026-10-05).
PROMPTS = {"forklift": GEOMETRY_PROMPTS["forklift"], "cargo": GEOMETRY_PROMPTS["cargo"], "person": ("person",),
           "LSP": (*GEOMETRY_PROMPTS["LSP"], "flat metal sheet", "metal plate on floor")}
MIN_AREA = {"forklift": 2500, "LSP": 1500, "cargo": 1500, "person": 300}
TOUCH_PX = 12          # masks closer than this touch
MIN_TOUCH = 50         # px of contact
GAP_PX = 6             # between the LSP and the free region ahead
MIN_INSIDE = .9        # of the free region inside the frame
MAX_OCCUPIED = .08     # of the free region covered by other objects
MIN_LSP_AREA = 12000   # px; smaller (far) sheets are too small to edit
MIN_MOVE_PX = 20       # LSP and forklift motion between samples that counts as moving
MAX_JUMP_PX = 400      # LSP centroid motion allowed when matching samples


def extract_frames(video, every, dest):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", f"fps=1/{every}",
                    "-q:v", "2", str(dest / "f_%05d.jpg")], check=True)
    return sorted(dest.glob("f_*.jpg"))


def segment(processor, state, key):
    masks = []
    for prompt in PROMPTS[key]:
        out = processor.set_text_prompt(state=state, prompt=prompt)
        for mask, score in zip(out["masks"], out["scores"]):
            if float(score) < .5:
                continue
            m = np.asarray(mask.detach().cpu().float()).squeeze() > .5
            if m.sum() >= MIN_AREA[key] and not any((m & k).sum() > .7 * min(m.sum(), k.sum()) for k in masks):
                masks.append(m)
    return masks


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def touches(near, mask):
    return int((near & mask).sum()) >= MIN_TOUCH


def ahead_region(lsp, forklift, shape):
    """The LSP's pixels shifted by its own size, away from the forklift: (xs, ys, inside)."""
    ys, xs = np.nonzero(lsp)
    d = centroid(lsp) - centroid(forklift)
    d /= max(float(np.linalg.norm(d)), 1e-6)
    along = xs * d[0] + ys * d[1]
    across = xs * -d[1] + ys * d[0]
    shift = max(float(np.ptp(along)), .6 * float(np.ptp(across))) + GAP_PX
    nx, ny = np.round(xs + d[0] * shift).astype(int), np.round(ys + d[1] * shift).astype(int)
    inside = (nx >= 0) & (nx < shape[1]) & (ny >= 0) & (ny < shape[0])
    return nx, ny, inside, d


def analyse(found, shape):
    """Every forklift with exactly one attached LSP, and whether that LSP has free floor ahead."""
    pairs = []
    for fi, forklift in enumerate(found["forklift"]):
        near = binary_dilation(forklift, iterations=TOUCH_PX)
        carried = [c for c in found["cargo"] if touches(near, c)]
        carried_near = binary_dilation(np.logical_or.reduce(carried), iterations=TOUCH_PX) if carried else None
        attached = [i for i, lsp in enumerate(found["LSP"])
                    if (lsp & ~forklift).sum() > .5 * lsp.sum()
                    and (touches(near, lsp) or (carried_near is not None and touches(carried_near, lsp)))]
        if len(attached) != 1:
            if attached:
                pairs.append({"forklift": fi, "lsp_count": len(attached), "passed": False,
                              "reason": f"{len(attached)} LSPs attached"})
            continue
        li = attached[0]
        lsp = found["LSP"][li]
        lsp_near = binary_dilation(lsp, iterations=TOUCH_PX)
        chained = [j for j, other in enumerate(found["LSP"]) if j != li and touches(lsp_near, other)]
        own_cargo = [k for k, c in enumerate(found["cargo"]) if touches(lsp_near, c)]
        obstacles = np.zeros(shape, bool)
        for key in ("forklift", "LSP", "cargo", "person"):
            for k, m in enumerate(found[key]):
                if not ((key == "LSP" and k == li) or (key == "cargo" and k in own_cargo)):
                    obstacles |= m
        nx, ny, inside, d = ahead_region(lsp, forklift, shape)
        inside_frac = float(inside.mean())
        occupied = float(obstacles[ny[inside], nx[inside]].mean()) if inside.any() else 1.
        reasons = []
        if lsp.sum() < MIN_LSP_AREA:
            reasons.append(f"LSP {int(lsp.sum())} px too small")
        if chained:
            reasons.append(f"{len(chained)} other LSP touching")
        if inside_frac < MIN_INSIDE:
            reasons.append(f"free region {inside_frac:.0%} inside frame")
        if occupied > MAX_OCCUPIED:
            reasons.append(f"free region {occupied:.0%} occupied")
        pairs.append({"forklift": fi, "lsp": li, "lsp_count": 1, "loaded": bool(own_cargo), "cargo": own_cargo,
                      "forklift_c": centroid(forklift).round(1).tolist(), "lsp_c": centroid(lsp).round(1).tolist(),
                      "lsp_area_px": int(lsp.sum()), "push_dir": d.round(3).tolist(),
                      "ahead_inside": round(inside_frac, 3), "ahead_occupied": round(occupied, 3),
                      "passed": not reasons, "reason": "; ".join(reasons),
                      "_ahead": (nx[inside], ny[inside])})
    return pairs


def moved_together(pair, other):
    dl = np.subtract(pair["lsp_c"], other["lsp_c"])
    df = np.subtract(pair["forklift_c"], other["forklift_c"])
    step = float(np.linalg.norm(dl))
    # both move, by about the same vector: a parked truck beside a sheet whose mask changes under people or
    # cargo being handled does not count
    return (MIN_MOVE_PX <= step <= MAX_JUMP_PX and float(np.linalg.norm(df)) >= MIN_MOVE_PX
            and float(np.linalg.norm(dl - df)) <= max(25., .5 * step))


def overlay(image, found, pair, path):
    pixels = np.asarray(image).astype(np.float32)
    tint = [(found["forklift"][pair["forklift"]], (255, 210, 0)), (found["LSP"][pair["lsp"]], (0, 230, 80))]
    tint += [(found["cargo"][k], (40, 120, 255)) for k in pair["cargo"]]
    for mask, colour in tint:
        pixels[mask] = .55 * pixels[mask] + .45 * np.array(colour)
    nx, ny = pair["_ahead"]
    pixels[ny, nx] = .5 * pixels[ny, nx] + .5 * np.array((0, 230, 255))
    out = Image.fromarray(pixels.astype(np.uint8))
    ImageDraw.Draw(out).text((12, 12), f"{'loaded' if pair['loaded'] else 'empty'} LSP | ahead occupied "
                             f"{pair['ahead_occupied']:.0%}", fill=(255, 255, 255))
    out.save(path, quality=90)


def contact_sheet(records, path, columns=4, width=480):
    if not records:
        return
    height = round(width * 9 / 16) + 22
    sheet = Image.new("RGB", (columns * width, ((len(records) + columns - 1) // columns) * height), "white")
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        with Image.open(record["overlay"]) as image:
            tile = image.convert("RGB").resize((width - 6, round(width * 9 / 16)))
        x, y = (i % columns) * width, (i // columns) * height
        sheet.paste(tile, (x + 3, y))
        draw.text((x + 5, y + height - 19), f"#{i} {Path(record['frame']).name}", fill="black")
    sheet.save(path, quality=88)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    parser.add_argument("--out", default="work/valid_frames")
    args = parser.parse_args()
    import torch
    from synth.segment import load_processor
    torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16).__enter__()
    processor = load_processor()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"every_s": args.every, "videos": {}, "frames": []}
    videos = [v for pattern in args.videos for v in sorted(glob.glob(pattern))]
    for video in videos:
        name = Path(video).name
        if name in index["videos"]:
            print(f"{name}: already done", flush=True)
            continue
        cam = Path(video).stem.split("_")[0]
        (out / cam).mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            frames = extract_frames(video, args.every, Path(tmp))
            samples = []          # per frame: list of pairs (masks dropped)
            for idx, frame in enumerate(frames):
                image = Image.open(frame).convert("RGB")
                state = processor.set_image(image)
                found = {"forklift": segment(processor, state, "forklift")}
                pairs = []
                if found["forklift"]:
                    for key in ("LSP", "cargo", "person"):
                        found[key] = segment(processor, state, key)
                    pairs = analyse(found, (image.height, image.width))
                    for p, pair in enumerate(pairs):
                        if pair["passed"]:
                            pair["overlay"] = str(Path(tmp) / f"{frame.stem}_{p}_overlay.jpg")
                            overlay(image, found, pair, pair["overlay"])
                        pair.pop("_ahead", None)
                samples.append(pairs)
                t = round((idx + .5) * args.every, 1)
                print(f"{name} {t:7.1f}s: forklifts {len(found['forklift'])}, "
                      f"one-LSP pairs {sum(p['lsp_count'] == 1 for p in pairs)}, "
                      f"free ahead {sum(p['passed'] for p in pairs)}", flush=True)
            # pushed: the pair moved together with its forklift in an adjacent sample
            passing = []
            for idx, pairs in enumerate(samples):
                for pair in pairs:
                    if not pair["passed"]:
                        continue
                    neighbours = [o for j in (idx - 1, idx + 1) if 0 <= j < len(samples)
                                  for o in samples[j] if "lsp_c" in o]
                    if any(moved_together(pair, o) for o in neighbours):
                        passing.append((idx, pair))
            # one frame per push: group samples close in time and space, keep the freest / largest sheet
            events = []
            for idx, pair in passing:
                last = events[-1][-1] if events else None
                if last and idx - last[0] <= 2 and np.linalg.norm(np.subtract(pair["lsp_c"], last[1]["lsp_c"])) <= MAX_JUMP_PX:
                    events[-1].append((idx, pair))
                else:
                    events.append([(idx, pair)])
            kept = 0
            for event in events:
                idx, pair = min(event, key=lambda e: (e[1]["ahead_occupied"], -e[1]["lsp_area_px"]))
                t = round((idx + .5) * args.every, 1)
                stem = f"{Path(video).stem}_{t:07.1f}s"
                dest = out / cam / f"{stem}.jpg"
                Image.open(frames[idx]).save(dest, quality=95)
                Image.open(pair["overlay"]).save(out / cam / f"{stem}_overlay.jpg", quality=90)
                index["frames"].append({"camera": cam, "video": name, "t_s": t, "frame": dest.as_posix(),
                                        "overlay": (out / cam / f"{stem}_overlay.jpg").as_posix(),
                                        "event_samples": len(event),
                                        **{k: v for k, v in pair.items() if k not in ("overlay", "passed", "reason")}})
                kept += 1
        index["videos"][name] = {"samples": len(samples), "free_ahead_samples": sum(p["passed"] for s in samples for p in s),
                                 "pushed_samples": len(passing), "events_kept": kept}
        write_json(index_path, index)
        contact_sheet([r for r in index["frames"] if r["camera"] == cam], out / f"{cam}_sheet.jpg")
        print(f"{name}: {kept} frames kept", flush=True)


if __name__ == "__main__":
    main()
