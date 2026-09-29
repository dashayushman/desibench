"""Paths, API keys, the two AIs' model choices, and the price table."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANSWERS = ROOT / "answers"   # the ground truth + per-episode inputs (committed)
VIDEOS = ROOT / "videos"     # downloaded episodes (git-ignored)
WORK = ROOT / "work"         # everything a run produces (git-ignored)
RESULTS = ROOT / "results"   # scored results: reference.json (my run) and yours.json (your run)

# The two episodes, in the order the site calls them.
EPISODES = {
    "_3UvCy7FMTg": "Video 1",
    "2pxmtBus3f8": "Video 2",
}


def _dotenv() -> dict[str, str]:
    out: dict[str, str] = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def cred(name: str) -> str:
    """An API key from the environment or .env. Never printed."""
    v = os.environ.get(name) or _dotenv().get(name)
    if not v:
        raise RuntimeError(f"Missing {name}. Put it in .env (see .env.example) or export it.")
    return v


def has_cred(name: str) -> bool:
    return bool(os.environ.get(name) or _dotenv().get(name))


OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or _dotenv().get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

# Two vendor-pure stacks. Every model call in a stack goes to that vendor only.
# Speech: each vendor's newest speech-to-text. LLMs: the same price tier (sarvam-105b ≈ $0.35/$0.87 per M tokens,
# gpt-6-luna $0.10/$0.50; OpenAI's next tier, gpt-6-sol, is 6-11x pricier, so it would be an unfair fight).
ARMS = {
    # Sarvam's documented Hinglish setup: codemix + hi-IN (Hindi stays in Devanagari, English words in English),
    # no keyterms (keyterms inserted phantom lines in testing).
    "sarvam-tuned": {"vendor": "sarvam", "stt": "saaras:v4", "mode": "codemix", "language": "hi-IN", "keyterms": False,
                     "refine": "sarvam-105b", "see": "sarvam-vision", "llm": "sarvam-105b"},
    # No name hints for OpenAI's transcription either, so both are scored the same way (the site compares these two).
    "openai": {"vendor": "openai", "stt": "gpt-transcribe", "diarize": "gpt-4o-transcribe-diarize", "keyterms": False,
               "refine": "gpt-audio-mini", "see": "gpt-6-luna", "llm": "gpt-6-luna"},
}
AI_NAME = {"sarvam-tuned": "Sarvam", "openai": "OpenAI"}
LLM_OF = {"sarvam-tuned": "sarvam-105b", "openai": "gpt-6-luna"}
SARVAM_MODE = "transcribe"

INR_PER_USD = 84.0
# USD per 1M tokens (vendor pricing pages, 2026-09-26). audio_in = audio input tokens.
PRICES = {
    "gpt-audio-mini": {"in": 0.60, "cached": 0.60, "out": 2.40, "audio_in": 10.0},
    "gpt-6-luna": {"in": 0.10, "cached": 0.01, "out": 0.50},
    "gpt-4o-transcribe-diarize": {"in": 2.50, "cached": 2.50, "out": 10.0, "audio_in": 2.50},
    "sarvam-105b": {"in": 29.28 / INR_PER_USD, "cached": 10.98 / INR_PER_USD, "out": 73.2 / INR_PER_USD},
}
GPT_TRANSCRIBE_USD_PER_MIN = 0.0045
SARVAM_INR_PER_MIN_DIARIZED = 45 / 60  # ₹45/hour (batch speech-to-text with diarization)
SARVAM_VISION_INR_PER_PAGE = 0.5

CHUNK_SECONDS = 600          # audio-LLM chunk length
FRAME_EVERY_S = 5            # one frame every 5 s
GRID_TILES = 16              # 4x4 → one grid covers 80 s
GRID_SECONDS = FRAME_EVERY_S * GRID_TILES
WINDOW_SECONDS = 600         # "making sense of the show" works in 10-minute windows

SHOW_CONTEXT = (
    "The Nation Wants to Guess is an Indian comedy quiz show hosted by Gursimran Khamba. "
    "Three comedian guests stand at podiums labelled GUESSER 1/2/3 and buzz in to answer questions; "
    "rounds include a buzzer round 'True Story' (questions based on real news), an audience round and "
    "others; there are running scores and a prize. Speech is Hinglish (English with Hindi mixed in)."
)


def video_dir(video_id: str) -> Path:
    """Shared per-episode inputs: audio, chunks, frames, grids, ingest.json."""
    d = WORK / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(video_id: str, arm: str) -> Path:
    """One AI's outputs for one episode: segments, refined, frames, fused, episode.json, raw/, ledger."""
    d = WORK / video_id / "runs" / arm
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d


def context(video_id: str) -> dict:
    """The people on the show, given identically to both AIs (answers/<episode>/context.json)."""
    p = ANSWERS / video_id / "context.json"
    return json.loads(p.read_text()) if p.exists() else {"people": [], "keyterms": []}
