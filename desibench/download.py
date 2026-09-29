"""Download the two episodes from YouTube into videos/<id>.mp4 (about 830 MB together, 1080p, same as the original run)."""
from __future__ import annotations

import shutil
import subprocess
import sys

from .config import EPISODES, VIDEOS

# yt-dlp in its own process, but trusting the system's certificates like the API calls do (truststore), so it works
# behind corporate proxies and antivirus TLS inspection instead of failing with CERTIFICATE_VERIFY_FAILED.
YT_DLP = "import sys, truststore; truststore.inject_into_ssl(); import yt_dlp; sys.exit(yt_dlp.main(sys.argv[1:]))"


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
        r = subprocess.run([sys.executable, "-c", YT_DLP, "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
                            "--no-playlist", "-o", str(out), f"https://www.youtube.com/watch?v={vid}"])
        if r.returncode:
            raise SystemExit(f"\n{label} ({vid}) didn't download (yt-dlp's error is above). YouTube changes often: "
                             "run `pip install -U yt-dlp` and the same command again; finished videos are kept.")
    print("videos ready in", VIDEOS)
