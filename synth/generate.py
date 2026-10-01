"""Stage C: violation / look-alike images from a tracked recording (run synth.track_video first).

Run: SYNTH_CONFIG=config/cameras/ch10.json python -m synth.generate --count 20 [--seed 7]
For each candidate frame (truck moving steadily, uncut, large enough, spaced in time):
  C2  heading from motion: the front axle (fixed axle of a rear-steer truck) is smoothed along the track and its
      velocity gives the heading; the silhouette only decides forwards vs reversing;
  C3  a push scenario (config/scenarios.json, active, weighted) and random cargo assets; the LSP row footprint must
      lie on free floor: inside the placement zone and clear of people, cargo, LSPs and other trucks in the frame;
  C4  occlusion from per-pixel depth (synth.occlusion);
  C5  composite (synth.composite_anchor_poc) and a sidecar with source frame, pose, scenario and checks.
Outputs <work>/out/gen_<camera>_<n>.{jpg,png,json} and <work>/out/gen_contact.jpg.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter

from synth.common import load_config, write_json
from synth.forklift_proxy import build, layout, load_spec

MODEL = "sumitomo_quapro_2t5_dual"
LANDMARKS = {"fork_tip_l": [-.29, 1.852, 0], "fork_tip_r": [.29, 1.852, 0], "heel_l": [-.29, .772, 0], "heel_r": [.29, .772, 0],
             "rear_l": [-.5, -1.1, 0], "rear_r": [.5, -1.1, 0]}
MIN_SPEED = .35           # m/s
MAX_TURN_DEG_S = 20       # steady driving only
MIN_AREA = 6000           # px
MIN_IOU = .55
MIN_GAP_S = 3.0
MAX_SIL_MOTION_DEG = 30    # silhouette and motion headings must agree; reversing / flipped fits are skipped
OBSTACLE_MAX = .02        # fraction of the footprint allowed to overlap other objects
ZONE_MIN = .85            # fraction of the footprint inside the placement zone
RED_MARGIN = .01           # red-light cue difference needed to flip the fork end
FORKS_LOADED_MAX = .25    # cargo/LSP cover of the fork area above which the truck already carries a load
# "container" and "large white box" catch ULDs, e.g. one a truck already carries
OBSTACLE_PROMPTS = {"person": ["person"], "forklift": ["forklift"],
                    "cargo": ["stretch wrapped cargo", "box", "container", "large white box"], "lsp": ["black platform"]}


def runs(idx):
    out, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b[0] == cur[-1][0] + 1:
            cur.append(b)
        else:
            out.append(cur); cur = [b]
    return out + [cur]


def motion_poses(tracks, fps, axle_y):
    """{(track, frame index): (x, y, heading, speed, turn_rate)} from smoothed front-axle tracks."""
    by_track = {}
    for d in tracks["detections"]:
        by_track.setdefault(d["track"], []).append((d["i"], d))
    out = {}
    for tid, items in by_track.items():
        for run in runs(sorted(items, key=lambda x: x[0])):
            if len(run) < 7:
                continue
            pose = np.array([d["pose"] for _, d in run])
            ax = pose[:, 0] - np.sin(pose[:, 2]) * axle_y; ay = pose[:, 1] + np.cos(pose[:, 2]) * axle_y
            w = min(9, len(run) // 2 * 2 - 1)
            sx, sy = savgol_filter(ax, w, 2), savgol_filter(ay, w, 2)
            vx, vy = savgol_filter(ax, w, 2, deriv=1) * fps, savgol_filter(ay, w, 2, deriv=1) * fps
            sil = savgol_filter(np.unwrap(pose[:, 2]), w, 2)
            heading = np.arctan2(-vx, vy)
            back = np.abs((heading - sil + np.pi) % (2 * np.pi) - np.pi) > np.pi / 2
            heading = np.where(back, heading + np.pi, heading)  # reversing: the silhouette keeps the fork end
            turn = np.degrees(np.gradient(np.unwrap(heading))) * fps
            for k, (i, _) in enumerate(run):
                h = float(heading[k])
                out[(tid, i)] = (float(sx[k] + math.sin(h) * axle_y), float(sy[k] - math.cos(h) * axle_y), h,
                                 float(math.hypot(vx[k], vy[k])), float(turn[k]), bool(back[k]))
    return out


def project(cam, pts):
    Mi = np.linalg.inv(np.array(cam["matrix_world"])); K = cam["K_norm"]; out = []
    for p in pts:
        c = Mi @ np.array([*p, 1.])
        out.append([(-c[0] / c[2] * K[0][0] + K[0][2]) * cam["width"], (c[1] / c[2] * K[1][1] + K[1][2]) * cam["height"]])
    return np.array(out)


def world(pose, local):
    x, y, h = pose[:3]; c, s = math.cos(h), math.sin(h)
    return (x + c * local[0] - s * local[1], y + s * local[0] + c * local[1], local[2] if len(local) > 2 else 0.)


def footprint_mask(cam, pose, lsp_count, lsp, sxy, shape):
    half = lsp["size_m"][0] / 2 * sxy; y0 = (.732 - .08) * sxy; y1 = y0 + lsp_count * (lsp["size_m"][1] + .008) * sxy
    poly = project(cam, [world(pose, (u, v)) for u, v in [(-half, y0), (half, y0), (half, y1), (-half, y1)]])
    im = Image.new("L", (shape[1], shape[0])); ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=1)
    inside = all(0 <= u < shape[1] and 0 <= v < shape[0] for u, v in poly)
    return np.asarray(im, bool), inside


def fork_area(cam, pose, sxy, shape):
    """Image polygon of the space on and just above the forks (local x +-0.7 m, y 0.75..1.9 m)."""
    poly = project(cam, [world(pose, (u * sxy, v * sxy, z)) for u, v in [(-.7, .75), (.7, .75), (.7, 1.9), (-.7, 1.9)] for z in (0., 1.)])
    im = Image.new("L", (shape[1], shape[0]))
    from scipy.spatial import ConvexHull
    hull = poly[ConvexHull(poly).vertices]; ImageDraw.Draw(im).polygon([tuple(p) for p in hull], fill=1)
    return np.asarray(im, bool)


def red_light_cue(cam, pose, sxy, image):
    """Red floor U light: closed end ~2.4 m behind the truck, open toward the forks. Fraction of red floor pixels in a
    band behind the counterweight and in front of the forks; a front-heavy result means the fork end is flipped."""
    a = np.asarray(image).astype(int); red = (a[..., 0] > 120) & (a[..., 0] - a[..., 1] > 45) & (a[..., 0] - a[..., 2] > 30)
    def band(y0, y1):
        poly = project(cam, [world(pose, (u * sxy, v * sxy)) for u, v in [(-1.6, y0), (1.6, y0), (1.6, y1), (-1.6, y1)]])
        im = Image.new("L", (red.shape[1], red.shape[0])); ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=1)
        m = np.asarray(im, bool); return float(red[m].mean()) if m.any() else 0.
    return band(-2.8, -1.9), band(1.0, 2.2)


def flip(pose, body_centre_y):
    """Same truck footprint, fork end swapped: rotate 180 degrees about the body centre."""
    x, y, h = pose; c = world(pose, (0, body_centre_y))
    h2 = h + math.pi; o = world((0, 0, h2), (0, body_centre_y))
    return (c[0] - o[0], c[1] - o[1], h2)


def segment(sam, image):
    state = sam.set_image(image); found = {}
    for cls, prompts in OBSTACLE_PROMPTS.items():
        found[cls] = []
        for p in prompts:
            out = sam.set_text_prompt(state=state, prompt=p)
            found[cls] += [m for m in (np.asarray(m.detach().cpu().float()).squeeze() > .5 for m in out["masks"]) if m.sum() > 200]
    return found


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config")
    args = parser.parse_args()
    import torch
    from moge.model.v2 import MoGeModel
    from synth.composite_anchor_poc import main as composite
    from synth.occlusion import occluders
    from synth.segment import load_processor
    cfg_path = args.config or os.environ.get("SYNTH_CONFIG", "config.json")
    cfg = load_config(cfg_path); work = cfg["work"]; cam_id = cfg["camera_id"]
    shared = Path(cfg.get("shared_work_dir", cfg["work_dir"]))
    cam = json.loads((work / "camera.json").read_text()); k1 = cfg["composite"].get("lens_k1", 0.)
    tracks = json.loads((work / "tracks.json").read_text()); fps = tracks["fps"]
    sc = tracks["scale_correction"]; sxy, sz = sc["xy"], sc["z"]
    spec = load_spec(MODEL); axle_y = layout(spec)["front_axle"] * sxy
    g = layout(spec); body_centre_y = (.807 + g["body_rear"]) / 2 * sxy
    P, _ = build(spec); P = P * np.array([sxy, sxy, sz], np.float32)
    lsp = json.loads(Path("config/standards.json").read_text())["lsp"]
    scen = [s for s in json.loads(Path("config/scenarios.json").read_text())["scenarios"]
            if s["family"] == "push" and s["status"] == "active" and s.get("requires", {}).get("forklift", True)]
    cargo_assets = sorted(p.stem for p in (shared / "assets").glob("cargo_*.json"))
    zone = np.asarray(Image.open(work / "placement_zone.png")) > 127
    floor_mask = np.asarray(Image.open(work / "masks" / Path(cfg["background_image"]).stem / "floor.png").convert("L")) > 127
    rng = random.Random(args.seed)
    motion = motion_poses(tracks, fps, axle_y)
    cands = []
    for d in tracks["detections"]:
        m = motion.get((d["track"], d["i"]))
        agree = m and abs((m[2] - d["pose"][2] + math.pi) % (2 * math.pi) - math.pi) <= math.radians(MAX_SIL_MOTION_DEG)
        if (m and agree and not m[5] and m[3] >= MIN_SPEED and abs(m[4]) <= MAX_TURN_DEG_S and not d["touches_border"]
                and d["area_px"] >= MIN_AREA and d["iou"] >= MIN_IOU):
            cands.append((d, m))
    cands.sort(key=lambda x: -x[0]["area_px"])
    picked = []
    for d, m in cands:  # spread in time, largest trucks first
        if all(abs(d["t_s"] - p[0]["t_s"]) >= MIN_GAP_S for p in picked):
            picked.append((d, m))
    picked.sort(key=lambda x: x[0]["t_s"])
    print(f"{cam_id}: {len(cands)} candidate detections, {len(picked)} spaced frames", flush=True)
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    sam = load_processor(); sam.confidence_threshold = .45
    moge = MoGeModel.from_pretrained(cfg["moge_model"]).to("cuda").eval()
    blender = shutil.which("blender") or os.environ.get("BLENDER", "blender")
    out_dir = work / "out"; out_dir.mkdir(parents=True, exist_ok=True)
    made, rejected = [], []
    for d, m in picked:
        if len(made) >= args.count:
            break
        frame = work / "frames" / d["frame"]; image = Image.open(frame).convert("RGB")
        s = rng.choices(scen, weights=[x["weight"] for x in scen])[0]; n_lsp = s["layout"]["lsp_count"]
        pose = (m[0], m[1], m[2])
        rear_red, front_red = red_light_cue(cam, pose, sxy, image); flipped = front_red > rear_red + RED_MARGIN
        if flipped:
            pose = flip(pose, body_centre_y)
        foot, inside = footprint_mask(cam, pose, n_lsp, lsp, sxy, zone.shape)
        if not inside:
            rejected.append((d["frame"], "row leaves the image")); continue
        in_zone = float((foot & zone).sum() / foot.sum())
        found = segment(sam, image)
        trucks = found["forklift"]
        boxes = [[*np.argwhere(t)[:, ::-1].min(0), *(np.argwhere(t)[:, ::-1].max(0) + 1)] for t in trucks]
        k = int(np.argmax([box_iou(b, d["box"]) for b in boxes])) if boxes else -1
        truck = trucks[k] if k >= 0 and box_iou(boxes[k], d["box"]) > .5 else np.zeros(zone.shape, bool)
        others = np.zeros(zone.shape, bool)
        for cls, masks in found.items():
            for j, mm in enumerate(masks):
                if not (cls == "forklift" and j == k):
                    others |= mm
        blocked = float((foot & others & ~truck).sum() / foot.sum())
        loads = np.zeros(zone.shape, bool)
        for mm in found["cargo"] + found["lsp"]:
            loads |= mm
        forks = fork_area(cam, pose, sxy, zone.shape)
        loaded = float((forks & loads).sum() / max(forks.sum(), 1))  # truck mask ignored: SAM3 merges a carried load into it
        if loaded > FORKS_LOADED_MAX:
            rejected.append((d["frame"], f"truck already loaded: {loaded:.2f} of the fork area")); continue
        if in_zone < ZONE_MIN or blocked > OBSTACLE_MAX:
            rejected.append((d["frame"], f"floor not free: zone {in_zone:.2f}, blocked {blocked:.3f}")); continue
        stem = f"gen_{cam_id}_{len(made) + 1:03d}"
        anchor = {"background_image": frame.as_posix(), "source_calibration": (work / "camera.json").as_posix(),
                  "source_floor_points": (work / "points_world.npy").as_posix(),
                  "layout": {"scenario": s["id"], "lsp_count": n_lsp, "include_idle_skid": False,
                             "cargo_assets": [rng.choice(cargo_assets) for _ in range(n_lsp)]},
                  "render": {"samples": 48, "seed": rng.randrange(1 << 16)},
                  "foreground_forklift": {"landmarks": [{"name": n, "local": l, "pixel": project(cam, [world(pose, l)])[0].round(2).tolist()}
                                                        for n, l in LANDMARKS.items()], "initial_pose": list(pose), "max_landmark_rmse_px": 2.0}}
        anchor_path = work / "anchors" / f"{stem}.json"; anchor_path.parent.mkdir(exist_ok=True)
        anchor_path.write_text(json.dumps(anchor, indent=1))
        r = subprocess.run([blender, "-b", "--python-exit-code", "1", "--python", "blender/render_anchor_poc.py", "--",
                            cfg_path, "--anchor", str(anchor_path), "--output-stem", stem], capture_output=True, text=True)
        if r.returncode:
            rejected.append((d["frame"], "render failed: " + (r.stdout + r.stderr).strip().splitlines()[-1][:120])); continue
        sidecar = json.loads((out_dir / f"{stem}.json").read_text())
        alpha = np.asarray(Image.open(work / "renders" / f"{stem}.png"))[..., 3] > 0
        rgb = torch.tensor(np.asarray(image, np.float32) / 255., device="cuda").permute(2, 0, 1)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            points = moge.infer(rgb, resolution_level=9, use_fp16=True)["points"].float().cpu().numpy()
        occ, occ_info = occluders(cam, k1, alpha, sidecar, shared / "assets", points, floor_mask, others | truck, truck, P, pose)
        occ_path = work / "anchors" / f"{stem}_occluder.png"; Image.fromarray((occ * 255).astype(np.uint8)).save(occ_path)
        anchor["occluder_masks"] = [occ_path.as_posix()]; anchor_path.write_text(json.dumps(anchor, indent=1))
        composite(cfg_path, str(anchor_path), stem)
        meta = json.loads((out_dir / f"{stem}.json").read_text())
        meta["pipeline"] = {"stage": "C", "camera_id": cam_id, "source_video": tracks["video"], "t_s": d["t_s"], "track": d["track"],
                            "heading": {"source": "motion (front axle)", "speed_mps": round(m[3], 2), "turn_deg_s": round(m[4], 1),
                                        "reversing": m[5] != flipped, "silhouette_yaw_deg": round(math.degrees(d["pose"][2]) % 360, 1),
                                        "red_light_rear_front": [round(rear_red, 3), round(front_red, 3)], "flipped_by_red_light": flipped},
                            "silhouette_iou": d["iou"], "free_floor": {"in_zone": round(in_zone, 3), "blocked": round(blocked, 4), "forks_loaded": round(loaded, 3)},
                            "occlusion": occ_info, "scenario_title": s["title"]}
        write_json(out_dir / f"{stem}.json", meta)
        made.append(stem); print(f"{stem}: {s['id']} t={d['t_s']}s {anchor['layout']['cargo_assets']} occluded {occ_info['occluded_px']}px", flush=True)
    write_json(out_dir / "gen_summary.json", {"made": made, "rejected": rejected})
    tiles = []
    for stem in made:
        im = Image.open(out_dir / f"{stem}.jpg"); im.thumbnail((640, 360)); tiles.append((stem, im))
    if tiles:
        sheet = Image.new("RGB", (640 * 4, 380 * ((len(tiles) + 3) // 4)), "white"); dr = ImageDraw.Draw(sheet)
        for i, (stem, im) in enumerate(tiles):
            x, y = (i % 4) * 640, (i // 4) * 380; sheet.paste(im, (x, y + 20)); dr.text((x + 4, y + 4), stem, fill="black")
        sheet.save(out_dir / "gen_contact.jpg", quality=85)
    print(f"{len(made)} images, {len(rejected)} rejected -> {out_dir}")


if __name__ == "__main__":
    main()
