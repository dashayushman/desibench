"""Step 1, ingest (shared by both AIs): audio (wav/mp3/chunks), frames every 5 s (single JPGs for
Sarvam Vision + 4x4 grids for GPT), YouTube markers → ingest.json."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .config import ANSWERS, CHUNK_SECONDS, FRAME_EVERY_S, GRID_SECONDS, GRID_TILES, VIDEOS, video_dir
from .ledger import Ledger


def _run(cmd: list[str]):
    subprocess.run(cmd, check=True, capture_output=True)


def _duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def find_video_file(video_id: str) -> Path:
    p = VIDEOS / f"{video_id}.mp4"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing. Run: python -m desibench download")
    return p


def ingest(video_id: str, force: bool = False) -> dict:
    d = video_dir(video_id)
    out_path = d / "ingest.json"
    if out_path.exists() and not force:
        info = json.loads(out_path.read_text())
        if "frames" in info:
            return info
    led = Ledger(d / "ledger.jsonl")
    video = find_video_file(video_id)
    duration = _duration(video)
    (d / "chunks").mkdir(exist_ok=True)
    (d / "grids").mkdir(exist_ok=True)
    (d / "frames").mkdir(exist_ok=True)

    steps = [
        ("audio_wav", d / "audio.wav", ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(d / "audio.wav")]),
        ("audio_mp3", d / "audio.mp3", ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(d / "audio.wav"),
                                        "-c:a", "libmp3lame", "-b:a", "48k", str(d / "audio.mp3")]),
        ("chunks", d / "chunks" / "chunk_000.mp3", ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
                                                     str(d / "audio.wav"), "-f", "segment", "-segment_time",
                                                     str(CHUNK_SECONDS), "-c:a", "libmp3lame", "-b:a", "48k",
                                                     str(d / "chunks" / "chunk_%03d.mp3")]),
        ("grids", d / "grids" / "grid_0001.jpg", ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                                                   "-vf", f"fps=1/{FRAME_EVERY_S},scale=480:-2,tile=4x4", "-q:v", "3",
                                                   str(d / "grids" / "grid_%04d.jpg")]),
        ("frames", d / "frames" / "frame_00001.jpg", ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                                                       "-vf", f"fps=1/{FRAME_EVERY_S},scale=1280:-2", "-q:v", "3",
                                                       str(d / "frames" / "frame_%05d.jpg")]),
    ]
    for name, marker, cmd in steps:
        if marker.exists() and not force:
            continue
        t0 = time.time()
        _run(cmd)
        led.log("ingest", name, seconds=time.time() - t0)

    chunks = []
    for i, p in enumerate(sorted((d / "chunks").glob("chunk_*.mp3"))):
        start = i * CHUNK_SECONDS
        chunks.append({"idx": i, "file": p.name, "start": start, "end": min(start + CHUNK_SECONDS, duration)})
    grids = []
    for i, p in enumerate(sorted((d / "grids").glob("grid_*.jpg"))):
        start = i * GRID_SECONDS
        grids.append({"idx": i, "file": p.name, "start": start,
                      "tile_times": [start + j * FRAME_EVERY_S for j in range(GRID_TILES)]})

    frames = [{"t": i * FRAME_EVERY_S, "file": p.name} for i, p in enumerate(sorted((d / "frames").glob("frame_*.jpg")))]

    markers_path = ANSWERS / video_id / "youtube_markers.json"  # saved from the YouTube page
    markers = json.loads(markers_path.read_text()) if markers_path.exists() else {}
    km = markers.get("key_moments")
    if isinstance(km, dict):  # {"markers": [{t, title}]} as saved from the watch page
        km = km.get("markers")

    info = {
        "video_id": video_id, "duration_s": duration, "title": markers.get("title"),
        "view_count": markers.get("view_count"), "upload_date": markers.get("upload_date"),
        "description_head": markers.get("description_head"),
        "chunks": chunks, "grids": grids, "frames": frames,
        "heatmap": markers.get("heatmap"), "key_moments": km,
    }
    out_path.write_text(json.dumps(info, indent=1))
    return info
