"""Append-only cost/time/token ledger per video (data/<id>/ledger.jsonl)."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from .config import INR_PER_USD, PRICES, SARVAM_INR_PER_MIN_DIARIZED


def openai_cost(model: str, usage: dict) -> float:
    p = PRICES.get(model) or PRICES.get(model.rsplit("-", 1)[0]) or {}
    if not p:
        return 0.0
    pt = usage.get("prompt_tokens") or 0
    ct = usage.get("completion_tokens") or 0
    det = usage.get("prompt_tokens_details") or {}
    audio = det.get("audio_tokens") or 0
    cached = det.get("cached_tokens") or 0
    text_in = max(pt - audio - cached, 0)
    return (
        text_in * p["in"] + cached * p.get("cached", p["in"]) + audio * p.get("audio_in", 0) + ct * p["out"]
    ) / 1e6


def sarvam_cost(seconds: float) -> float:
    return seconds / 60 * SARVAM_INR_PER_MIN_DIARIZED / INR_PER_USD


class Ledger:
    def __init__(self, path: Path):
        self.path = path

    def log(self, stage: str, call: str, *, model: str = "", seconds: float = 0.0,
            usage: dict | None = None, cost_usd: float | None = None, **extra):
        usage = usage or {}
        if cost_usd is None:
            cost_usd = openai_cost(model, usage) if model else 0.0
        row = {"ts": time.time(), "stage": stage, "call": call, "model": model,
               "seconds": round(seconds, 2), "usage": usage, "cost_usd": round(cost_usd, 6), **extra}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def summary(self) -> dict:
        agg: dict[str, dict] = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "in": 0, "audio_in": 0,
                                                    "cached": 0, "out": 0, "cost_usd": 0.0, "models": set()})
        for r in self.rows():
            a = agg[r["stage"]]
            u = r.get("usage") or {}
            det = u.get("prompt_tokens_details") or {}
            a["calls"] += 1
            a["seconds"] += r.get("seconds", 0)
            a["in"] += u.get("prompt_tokens") or 0
            a["audio_in"] += det.get("audio_tokens") or 0
            a["cached"] += det.get("cached_tokens") or 0
            a["out"] += u.get("completion_tokens") or 0
            a["cost_usd"] += r.get("cost_usd", 0)
            if r.get("model"):
                a["models"].add(r["model"])
        out = {}
        for k, v in agg.items():
            v["models"] = sorted(v["models"])
            out[k] = v
        out["_total"] = {
            "cost_usd": sum(v["cost_usd"] for v in agg.values()),
            "seconds": sum(v["seconds"] for v in agg.values()),
            "in": sum(v["in"] for v in agg.values()),
            "out": sum(v["out"] for v in agg.values()),
        }
        return out
