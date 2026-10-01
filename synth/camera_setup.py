"""Stage B2 inputs: per-camera reference frames and config.

Run: python -m synth.camera_setup [--channels ch10 ch14 ...]
For each camera with a clean plate (data/cameras/<ch>/background.jpg, from synth.clean_plate) and a recording
in data/multicam/video, picks up to 10 reference frames where the stage A1 library saw sharp LSPs or forklifts,
extracts them to data/cameras/<ch>/references/, and writes config/cameras/<ch>.json (extends config.json).
Then: SYNTH_CONFIG=config/cameras/<ch>.json python -m synth.segment && ... python -m synth.calibrate
"""
import argparse
import json
import subprocess
from pathlib import Path

MAX_REFS = 10
MIN_GAP_S = 15


def pick_times(crops, cam):
    """LSP crops first (they drive the floor-scale check), then forklifts; best quality first, spread in time."""
    ranked = sorted((r for r in crops if r["camera"] == cam and r["class"] in ("lsp", "forklift") and not r["touches_border"]),
                    key=lambda r: (r["class"] != "lsp", -r["quality"]))
    times = []
    for r in ranked:
        if all(abs(r["t_s"] - t) >= MIN_GAP_S for t in times):
            times.append(r["t_s"])
        if len(times) == MAX_REFS:
            break
    return sorted(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channels", nargs="*")
    parser.add_argument("--library", default="work/library/index.json")
    args = parser.parse_args()
    library = json.loads(Path(args.library).read_text())
    lsp = json.loads(Path("config/standards.json").read_text())["lsp"]
    channels = args.channels or sorted(p.name for p in Path("data/cameras").iterdir() if (p / "background.jpg").exists())
    for cam in channels:
        video = Path("data/multicam/video") / library["cameras"][cam]["video"]
        refs = Path("data/cameras") / cam / "references"; refs.mkdir(parents=True, exist_ok=True)
        for old in refs.glob("*.jpg"):
            old.unlink()
        times = pick_times(library["crops"], cam)
        for t in times:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", str(video),
                            "-frames:v", "1", "-q:v", "2", str(refs / f"{cam}_t{int(t):04d}.jpg")], check=True)
        cfg = {"extends": "config.json", "camera_id": cam,
               "background_image": f"data/cameras/{cam}/background.jpg",
               "object_frames_dir": f"data/cameras/{cam}/references",
               "work_dir": f"work/cameras/{cam}", "lsp_size_m": lsp["size_m"],
               "source_video": video.as_posix()}
        dest = Path("config/cameras") / f"{cam}.json"; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"{cam}: {len(times)} reference frames at {times} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
