"""Stage C1+C2: track forklifts through a camera's recording and fit their floor pose in every frame.

Run: SYNTH_CONFIG=config/cameras/ch10.json python -m synth.track_video [--fps 2] [--video PATH]
Frames go to <work>/frames/f_%05d.jpg. Each SAM3 forklift mask is linked to the nearest track seen in the last
MAX_GAP frames; the pose is a full silhouette search for a new track and a local search from the track's last
pose otherwise. The proxy is the catalogue QuaPro scaled by this camera's scale_correction.json, in the same
frame as forklift.blend, so poses drive the renderer directly. Writes <work>/tracks.json.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from synth.common import load_config, write_json
from synth.forklift_proxy import build, load_spec

MODEL = "sumitomo_quapro_2t5_dual"
MIN_AREA = 2500     # px; smaller trucks give unusable poses
MAX_GAP = 3         # frames a track may be unseen before it ends
MAX_JUMP_PX = 220   # bottom-centre motion allowed between linked detections


def proxy(work, device):
    import torch
    sc = json.loads((work / "scale_correction.json").read_text()) if (work / "scale_correction.json").exists() else {"xy": 1., "z": 1.}
    P, L = build(load_spec(MODEL))
    P = P * np.array([sc["xy"], sc["xy"], sc["z"]], np.float32)
    return torch.tensor(P, device=device), torch.tensor(L.astype(np.int64), device=device), sc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--video")
    args = parser.parse_args()
    import torch
    from synth.pose_fit import Camera, MaskFit
    from synth.segment import load_processor
    cfg = load_config(); work = cfg["work"]; video = args.video or cfg["source_video"]
    frames_dir = work / "frames"; frames_dir.mkdir(parents=True, exist_ok=True)
    if not any(frames_dir.glob("f_*.jpg")):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video, "-vf", f"fps={args.fps}",
                        "-q:v", "2", str(frames_dir / "f_%05d.jpg")], check=True)
    frames = sorted(frames_dir.glob("f_*.jpg"))
    camera = Camera(json.loads((work / "camera.json").read_text()))
    P, L, sc = proxy(work, "cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    sam = load_processor(); sam.confidence_threshold = .5
    tracks = []      # {"id", "last_i", "last_c", "last_pose"}
    dets = []
    for i, f in enumerate(frames):
        t0 = time.time()
        out = sam.set_text_prompt(state=sam.set_image(Image.open(f).convert("RGB")), prompt="forklift")
        masks = [m for m in (np.asarray(m.detach().cpu().float()).squeeze() > .5 for m in out["masks"]) if m.sum() >= MIN_AREA]
        used = set()
        for m in sorted(masks, key=lambda m: -m.sum()):
            ys, xs = np.nonzero(m); c = np.array([xs.mean(), ys.max()])
            live = [t for t in tracks if i - t["last_i"] <= MAX_GAP and t["id"] not in used]
            near = min(live, key=lambda t: np.linalg.norm(t["last_c"] - c), default=None)
            if near is not None and np.linalg.norm(near["last_c"] - c) <= MAX_JUMP_PX * (i - near["last_i"]):
                track = near
            else:
                track = {"id": len(tracks), "last_pose": None}; tracks.append(track)
            used.add(track["id"])
            fit = MaskFit(camera, m)
            with torch.autocast("cuda", enabled=False), torch.no_grad():
                iou, pose = fit.fit(P, L, start=track["last_pose"])
            track.update(last_i=i, last_c=c, last_pose=pose)
            dets.append({"frame": f.name, "i": i, "t_s": round(i / args.fps, 2), "track": track["id"],
                         "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                         "area_px": int(m.sum()), "touches_border": fit.touches_border,
                         "iou": round(iou, 3), "pose": [round(v, 4) for v in pose]})
        if i % 20 == 0:
            print(f"{cfg['camera_id']} frame {i}/{len(frames)}: {len(masks)} forklifts, {len(tracks)} tracks, {time.time() - t0:.1f}s", flush=True)
    write_json(work / "tracks.json", {"camera_id": cfg["camera_id"], "video": Path(video).as_posix(), "fps": args.fps,
                                      "proxy": MODEL, "scale_correction": {"xy": sc["xy"], "z": sc["z"]}, "detections": dets})
    print(f"{len(dets)} detections in {len(tracks)} tracks -> {work / 'tracks.json'}")


if __name__ == "__main__":
    main()
