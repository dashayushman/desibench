"""Minimal chat-completions client for both vendors (Sarvam and OpenAI), with retries."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
import truststore

from .config import OPENAI_BASE_URL, cred

truststore.inject_into_ssl()

_client: httpx.Client | None = None


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=httpx.Timeout(900.0, connect=30.0))
    return _client


def _headers() -> dict:
    return {"Authorization": f"Bearer {cred('OPENAI_API_KEY')}", "Content-Type": "application/json"}


SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"


def _sarvam_stream(url: str, headers: dict, body: dict):
    """Stream a Sarvam chat completion; returns (content, usage) like the non-streaming path."""
    parts, usage, finish = [], {}, None
    with _http().stream("POST", url, headers=headers, json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.read().decode()[:500]}")
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            j = json.loads(line[5:])
            for ch in j.get("choices") or []:
                parts.append((ch.get("delta") or {}).get("content") or "")
                finish = ch.get("finish_reason") or finish
            usage = j.get("usage") or usage
    usage = dict(usage or {})
    if usage.get("reasoning_tokens") is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": usage["reasoning_tokens"]}
    usage["_finish_reason"] = finish
    content = "".join(parts)
    return (content or None), usage


def chat(model: str, messages: list, *, json_mode: bool = False, max_retries: int = 4,
         max_tokens: int | None = None, **kw):
    """Returns (content:str, usage:dict, seconds:float). sarvam-* models go to api.sarvam.ai,
    everything else to OpenAI (OPENAI_BASE_URL, api.openai.com by default)."""
    sarvam = model.startswith("sarvam-")
    if sarvam:
        url, headers = SARVAM_CHAT_URL, {"api-subscription-key": cred("SARVAM_API_KEY"), "Content-Type": "application/json"}
    else:
        url, headers = OPENAI_BASE_URL + "/chat/completions", _headers()
    body = {"model": model, "messages": messages, **kw}
    if max_tokens:
        body["max_tokens" if sarvam else "max_completion_tokens"] = max_tokens
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if sarvam:  # long silent reasoning over a plain request stalls (friction log #24) → stream it
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    last = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            if sarvam:
                return (*_sarvam_stream(url, headers, body), time.time() - t0)
            r = _http().post(url, headers=headers, json=body)
            if r.status_code == 200:
                j = r.json()
                usage = j.get("usage", {}) or {}
                usage["_finish_reason"] = j["choices"][0].get("finish_reason")
                return j["choices"][0]["message"]["content"], usage, time.time() - t0
            last = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code in (400, 401, 403, 404):
                break
        except Exception as e:  # network / timeout
            last = repr(e)[:300]
            if "HTTP 400" in last or "HTTP 401" in last:
                break
        time.sleep((20 if "429" in str(last) else 5) * (attempt + 1))
    raise RuntimeError(f"{model} call failed: {last}")


def transcribe_file(model: str, path, fields: dict, *, max_retries: int = 3):
    """POST /audio/transcriptions (multipart). List values repeat the key (e.g. "keywords[]").
    Returns (json, seconds)."""
    url = OPENAI_BASE_URL + "/audio/transcriptions"
    hdrs = {k: v for k, v in _headers().items() if k != "Content-Type"}
    data = {"model": model, **fields}
    last = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            with open(path, "rb") as fh:
                r = _http().post(url, headers=hdrs, data=data, files={"file": (Path(path).name, fh, "audio/mpeg")})
            if r.status_code == 200:
                return r.json(), time.time() - t0
            last = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code in (400, 401, 403, 404, 413):
                break
        except Exception as e:
            last = repr(e)[:300]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{model} transcription failed: {last}")


def chat_json_cached(model: str, messages: list, raw_path, led, stage: str, call: str, *, max_tokens: int = 64000):
    """JSON chat with a raw-response cache and one retry. Same policy for every vendor:
    attempt 1 = model defaults; attempt 2 = reasoning_effort "low" (sarvam-105b can spend the whole
    budget reasoning and return nothing — friction log #10/#20). Every paid attempt is logged."""
    if raw_path.exists():
        try:
            return parse_json(json.loads(raw_path.read_text())["content"])
        except Exception:
            pass  # cached empty/unparseable answer → call again
    last = None
    for attempt in (1, 2):
        kw = {"reasoning_effort": "low"} if attempt == 2 and model.startswith("sarvam-") else {}
        content, usage, secs = chat(model, messages, json_mode=True, max_tokens=max_tokens, **kw)
        raw_path.write_text(json.dumps({"usage": usage, "seconds": secs, "content": content}, indent=1, ensure_ascii=False))
        led.log(stage, call, model=model, seconds=secs, usage=usage, attempt=attempt, finish_reason=usage.get("_finish_reason"))
        try:
            return parse_json(content or "")
        except Exception as e:
            last = e
    raise RuntimeError(f"{model} {stage}/{call}: no parseable JSON after 2 attempts ({last})")


def parse_json(text: str, repair: bool = True):
    """Lenient JSON parse: strips ``` fences, finds first {...}, then falls back to json_repair
    (handles unescaped quotes and truncated tails). Raises ValueError if nothing usable."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    cand = t[s:e + 1] if (s != -1 and e > s) else t
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        if not repair:
            raise
        from json_repair import repair_json
        obj = repair_json(t[s:] if s != -1 else t, return_objects=True)
        if not isinstance(obj, dict) or not obj:
            raise ValueError("unrepairable JSON")
        obj["_repaired"] = True
        return obj
