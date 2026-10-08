"""Grab N still frames from one RTSP stream, spaced in time, as inputs of one input kind.

Run: python -m synth.grab_rtsp_frames <source> --kind floor-closed --count 10 --every 20
     [--out work/inputs/<source name>]
<source> is a key of data/rtsp_sources.json ({"cmw01-ch54": "rtsp://user:pass@host/...", ...}; data/ is gitignored,
so credentials never enter the repo) or an rtsp:// URL. Writes <out>/<kind>/<source name>_<UTC time>_NN.jpg, one
frame every --every seconds, so the folder is a job input of the generation service. A frame that cannot be read is
retried up to --retries times; the run fails when the stream gives no frame at all.
"""
import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from synth.violation_cases import INPUT_KINDS

ROOT = Path(__file__).resolve().parents[1]
STREAM_TIMEOUT_US = 30_000_000      # ffmpeg ends after 30 s without data instead of hanging


def stream_url(source: str, sources: Path) -> tuple[str, str]:
    """The URL and a file-safe name for a sources key or a URL."""
    if source.startswith("rtsp://"):
        return source, re.sub(r"\W+", "-", re.sub(r"^rtsp://[^@]*@", "", source)).strip("-")[:60]
    table = json.loads(sources.read_text()) if sources.is_file() else {}
    if source not in table:
        raise SystemExit(f"{source!r} is neither an rtsp:// URL nor a key of {sources}")
    return table[source], source


def grab(url: str, target: Path) -> str | None:
    """One frame of the stream to target; the error text, or None when it worked."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-rtsp_transport", "tcp", "-timeout",
               str(STREAM_TIMEOUT_US), "-i", url, "-frames:v", "1", "-q:v", "2", str(target)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=STREAM_TIMEOUT_US // 1_000_000 + 30)
    if result.returncode == 0 and target.is_file() and target.stat().st_size:
        return None
    return (result.stderr.strip().splitlines() or [f"ffmpeg exit {result.returncode}"])[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="a key of --sources or an rtsp:// URL")
    parser.add_argument("--kind", required=True, choices=sorted(INPUT_KINDS), help="what the frames show")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--every", type=float, default=20, help="seconds between frames")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sources", type=Path, default=ROOT / "data/rtsp_sources.json")
    parser.add_argument("--out", type=Path, help="default work/inputs/<source name>")
    args = parser.parse_args()
    url, name = stream_url(args.source, args.sources)
    folder = (args.out or ROOT / "work/inputs" / name) / args.kind
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved = 0
    for index in range(1, args.count + 1):
        target = folder / f"{name}_{stamp}_{index:02d}.jpg"
        for attempt in range(1, args.retries + 1):
            error = grab(url, target)
            if error is None:
                saved += 1
                print(f"{index}/{args.count}: {target}", flush=True)
                break
            print(f"{index}/{args.count}: attempt {attempt} failed: {error}", flush=True)
        if index < args.count:
            time.sleep(args.every)
    print(f"{saved} of {args.count} frames in {folder}")
    if not saved:
        raise SystemExit("the stream gave no frame")
    return 0 if saved == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
