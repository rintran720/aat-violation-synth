"""Record several RTSP cameras at the same time (stream copy, no re-encode).

Run: python -m synth.capture_rtsp --seconds 600
     python -m synth.capture_rtsp --seconds 300 --channels ch14 ch16
Reads data/rtsp_sources.json ({"ch16": "rtsp://user:pass@host/...", ...}); data/ is gitignored, so
credentials never enter the repo. Writes data/multicam/video/<channel>_<timestamp>.mp4; a stream that drops is
reconnected and the rest goes to <channel>_<timestamp>_part<N>.mp4.
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
    deadline = time.time() + args.seconds

    def start(ch, part):
        # -timeout: a stream that stops sending ends ffmpeg after 15 s (and the MP4 is finalised) instead of
        # hanging with an unreadable file (seen on 2026-10-05: all channels stalled after 16 min)
        seconds = round(deadline - time.time())
        dest = out / (f"{ch}_{stamp}.mp4" if part == 1 else f"{ch}_{stamp}_part{part}.mp4")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-timeout", "15000000",
               "-i", sources[ch], "-t", str(seconds), "-c", "copy", "-an", str(dest)]
        print(f"{ch}: recording {seconds}s -> {dest}", flush=True)
        return subprocess.Popen(cmd), dest, part

    procs = {ch: start(ch, 1) for ch in channels}
    recorded = {ch: [] for ch in channels}
    while procs:
        time.sleep(2)
        for ch, (proc, dest, part) in list(procs.items()):
            if proc.poll() is None:
                continue
            del procs[ch]
            if dest.exists() and dest.stat().st_size:
                recorded[ch].append(dest)
                print(f"{ch}: part {part} ended ({dest.stat().st_size / 1e6:.0f} MB)", flush=True)
            if deadline - time.time() > 10:   # the stream dropped early: record the rest into a new part
                procs[ch] = start(ch, part + 1)
    failed = [ch for ch in channels if not recorded[ch]]
    if failed:
        raise SystemExit(f"recording failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
