"""Record several RTSP cameras at the same time (stream copy, no re-encode).

Run: python -m synth.capture_rtsp --seconds 600
     python -m synth.capture_rtsp --seconds 300 --channels ch14 ch16
Reads data/rtsp_sources.json ({"ch16": "rtsp://user:pass@host/...", ...}); data/ is gitignored, so
credentials never enter the repo. Writes data/multicam/video/<channel>_<timestamp>.mp4.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default="data/rtsp_sources.json")
    parser.add_argument("--channels", nargs="*", help="subset of channel names; default all")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--out", default="data/multicam/video")
    args = parser.parse_args()

    sources = json.loads(Path(args.sources).read_text())
    channels = args.channels or sorted(sources)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    procs = {}
    for ch in channels:
        dest = out / f"{ch}_{stamp}.mp4"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", sources[ch],
               "-t", str(args.seconds), "-c", "copy", "-an", str(dest)]
        procs[ch] = (subprocess.Popen(cmd), dest)
        print(f"{ch}: recording {args.seconds}s -> {dest}", flush=True)
    failed = []
    for ch, (proc, dest) in procs.items():
        if proc.wait() != 0 or not dest.exists():
            failed.append(ch)
        else:
            print(f"{ch}: done ({dest.stat().st_size / 1e6:.0f} MB)", flush=True)
    if failed:
        raise SystemExit(f"recording failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
