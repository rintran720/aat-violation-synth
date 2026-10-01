"""Stage B1: clean background plate per camera = per-pixel temporal median of frames sampled across a recording.

Run: python -m synth.clean_plate data/multicam/video/ch14_*.mp4 [--every 4] [--out data/cameras]
Writes data/cameras/<camera>/background.jpg (+ background_spread.png: per-pixel spread, bright = often occupied).
People and moving forklifts vanish when each pixel is free in more than half of the samples; objects parked for
most of the recording (idle LSP stacks, cargo) stay, as they are part of the scene.
"""
import argparse
import glob
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--every", type=float, default=4.0, help="seconds between sampled frames")
    parser.add_argument("--out", default="data/cameras")
    args = parser.parse_args()
    for video in [v for p in args.videos for v in sorted(glob.glob(p))]:
        cam = Path(video).stem.split("_")[0]
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video, "-vf", f"fps=1/{args.every}",
                            "-q:v", "2", str(Path(tmp) / "f_%05d.jpg")], check=True)
            frames = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in sorted(Path(tmp).glob("f_*.jpg"))])
        median = np.empty(frames.shape[1:], np.uint8); spread = np.empty(frames.shape[1:3], np.float32)
        for y in range(0, frames.shape[1], 60):  # row strips keep the float temporaries small
            strip = frames[:, y:y + 60]
            median[y:y + 60] = np.median(strip, axis=0).astype(np.uint8)
            q25, q75 = np.percentile(strip.mean(axis=3), [25, 75], axis=0); spread[y:y + 60] = q75 - q25
        dest = Path(args.out) / cam; dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray(median).save(dest / "background.jpg", quality=95)
        Image.fromarray(np.clip(spread * 4, 0, 255).astype(np.uint8)).save(dest / "background_spread.png")
        print(f"{cam}: {len(frames)} frames -> {dest / 'background.jpg'}", flush=True)


if __name__ == "__main__":
    main()
