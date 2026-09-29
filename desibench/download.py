"""Download the two episodes from YouTube into videos/<id>.mp4 (about 830 MB together, 1080p, same as the original run)."""
from __future__ import annotations

import shutil
import subprocess
import sys

from .config import EPISODES, VIDEOS


def download(only: list[str] | None = None) -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is missing. macOS: brew install ffmpeg · Ubuntu: sudo apt install ffmpeg")
    VIDEOS.mkdir(exist_ok=True)
    for vid, label in EPISODES.items():
        if only and vid not in only:
            continue
        out = VIDEOS / f"{vid}.mp4"
        if out.exists():
            print(f"{label} ({vid}): already downloaded")
            continue
        print(f"{label} ({vid}): downloading…", flush=True)
        subprocess.run([sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
                        "--no-playlist", "-o", str(out), f"https://www.youtube.com/watch?v={vid}"], check=True)
    print("videos ready in", VIDEOS)
