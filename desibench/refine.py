"""S3 Refine → runs/<arm>/refined.json. Per 10-min chunk, an LLM corrects the STT draft.
openai: gpt-audio-mini LISTENS to the chunk (audio + draft) and also marks audience reactions.
sarvam: Sarvam has no audio-input LLM, so sarvam-105b corrects the draft from text + context only
        (it cannot hear laughter; reactions stay empty unless the text itself shows them)."""
from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor

from .config import ARMS, SHOW_CONTEXT, context, run_dir, video_dir
from .ledger import Ledger
from .llm import chat, parse_json

AUDIO_INTRO = """You are given AUDIO for the window {w0}–{w1} of this episode (mm:ss), and a DRAFT diarized transcript from an ASR system for the same window. Timestamps in the draft and in your output are SECONDS FROM THE START OF THIS AUDIO CLIP.

Using the audio as ground truth:"""
TEXT_INTRO = """You are given a DRAFT diarized transcript from an ASR system for the window {w0}–{w1} of this episode (mm:ss). There is no audio: correct only what the text and context make clearly wrong (misheard names, game terms, obvious homophones); when unsure keep the draft. Timestamps in the draft and in your output are SECONDS FROM THE START OF THIS WINDOW.

Using the draft and context:"""

PROMPT = """{show}

Episode: {title}
People on the show: {people}
YouTube key moments (episode time): {key_moments}

{intro}
1. Correct the draft: fix misheard words, names and proper nouns; keep Hindi/Hinglish as spoken, written in Latin script as commonly typed. Return ONLY the lines you change, by their [#i] number, with the full corrected text of that line. Lines you don't return are kept as-is. If a line contains no real speech (e.g. text that was never said), return it with "text": "".
2. For each speaker id, say who you think it is (host or a guest name) with confidence 0-1 and one line of evidence (names used, role, voice continuity). Use "unknown" if unsure.
3. Mark audience reactions (laughter, applause, groan, cheer) with start seconds and intensity 1-3 — only ones you can actually hear/see evidence of; otherwise return an empty list.
4. Note anything notable (music sting, round change, score, prize mention).

Return ONLY JSON:
{{"edits":[{{"i":0,"text":"..."}}],
 "speaker_hints":{{"<speaker id>":{{"name":"...","confidence":0.0,"evidence":"..."}}}},
 "reactions":[{{"t":0.0,"type":"laughter|applause|groan|cheer","intensity":1}}],
 "events":[{{"t":0.0,"type":"music|round_change|score|other","note":"..."}}],
 "notes":"..."}}

DRAFT:
{draft}"""


def people_str(video_id: str, info: dict) -> str:
    ppl = context(video_id).get("people") or []
    if ppl:
        return "; ".join(f"{p['name']} ({p['role']})" for p in ppl)
    return ", ".join(w for w in (info.get("title") or "").split() if w.startswith("@")) or "n/a"


REACTION_TYPES = ("laughter", "applause", "cheer", "groan")


def _norm_reaction(r: dict) -> dict | None:
    t = str(r.get("type", "")).lower()
    kind = next((k for k in REACTION_TYPES if k[:5] in t), None)
    return {**r, "type": kind} if kind else None


def _mmss(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def _refine_chunk(d, rd, info, segs, chunk, led, people, cfg):
    audio = cfg["vendor"] == "openai"
    c0, c1 = chunk["start"], chunk["end"]
    window = [s for s in segs if c0 <= s["start"] < c1]  # each segment belongs to exactly one chunk
    draft = "\n".join(f"[#{i}] [{s['start'] - c0:.1f}-{s['end'] - c0:.1f}] {s['speaker']}: {s['text']}" for i, s in enumerate(window))
    km = [f"{_mmss(k['t'])} {k['title']}" for k in (info.get("key_moments") or [])]
    intro = (AUDIO_INTRO if audio else TEXT_INTRO).format(w0=_mmss(c0), w1=_mmss(c1))
    prompt = PROMPT.format(show=SHOW_CONTEXT, title=info.get("title"), people=people, key_moments="; ".join(km) or "n/a",
                           intro=intro, draft=draft)
    content_parts = [{"type": "text", "text": prompt}]
    if audio:
        audio_b64 = base64.b64encode((d / "chunks" / chunk["file"]).read_bytes()).decode()
        content_parts.append({"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}})
    messages = [{"role": "user", "content": content_parts if audio else prompt}]  # Sarvam chat wants a plain string
    kw = {"modalities": ["text"]} if audio else {"json_mode": True}
    raw_dir = rd / "raw" / "refine"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"refine_chunk_{chunk['idx']:03d}.json"
    out = None
    # reuse a cached raw response if it parses (resume without re-billing)
    if raw_path.exists():
        try:
            out = parse_json(json.loads(raw_path.read_text())["content"])
        except Exception:
            out = None
    attempts = 0
    while out is None and attempts < 2:
        attempts += 1
        if not audio:  # sarvam-105b reasons at length before answering (friction log #10): big budget, then low effort
            kw = {"json_mode": True, **({"reasoning_effort": "low"} if attempts > 1 else {})}
        content, usage, secs = chat(cfg["refine"], messages, max_tokens=16000 if audio else 64000, **kw)
        raw_path.write_text(json.dumps({"usage": usage, "seconds": secs, "content": content}, indent=1, ensure_ascii=False))
        led.log("refine", f"chunk_{chunk['idx']:03d}", model=cfg["refine"], seconds=secs, usage=usage,
                attempt=attempts, finish_reason=usage.get("_finish_reason"))
        try:
            out = parse_json(content)
        except Exception:
            out = None
    if out is None:
        raise RuntimeError(f"refine chunk {chunk['idx']} returned unparseable JSON twice")
    # apply edits-only corrections; empty text = the model says no speech there (dropped, counted)
    fixed = {int(e["i"]): str(e.get("text") or "") for e in out.get("edits", []) if str(e.get("i", "")).lstrip("-").isdigit()}
    out["segments"] = [{**s, "text": fixed.get(i, s["text"]).strip()} for i, s in enumerate(window)
                       if fixed.get(i, s["text"]).strip()]
    out["n_edits"] = len(fixed)
    out["n_removed"] = sum(1 for v in fixed.values() if not v.strip())
    for r in out.get("reactions", []):
        r["t"] = round(float(r["t"]) + c0, 2)
    for e in out.get("events", []):
        e["t"] = round(float(e.get("t", 0)) + c0, 2)
    out["chunk"] = chunk["idx"]
    return out


def refine(video_id: str, arm: str, force: bool = False, workers: int = 4) -> dict:
    d, rd, cfg = video_dir(video_id), run_dir(video_id, arm), ARMS[arm]
    out_path = rd / "refined.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    info = json.loads((d / "ingest.json").read_text())
    segs = json.loads((rd / "segments.json").read_text())["segments"]
    led = Ledger(rd / "ledger.jsonl")
    people = people_str(video_id, info)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda c: _refine_chunk(d, rd, info, segs, c, led, people, cfg), info["chunks"]))

    merged = {"segments": [], "reactions": [], "events": [], "speaker_hints_by_chunk": {}, "notes_by_chunk": {},
              "edits_by_chunk": {}}
    for r in results:
        merged["segments"] += r.get("segments", [])
        merged["reactions"] += [x for x in map(_norm_reaction, r.get("reactions", [])) if x]
        merged["events"] += r.get("events", [])
        merged["speaker_hints_by_chunk"][r["chunk"]] = r.get("speaker_hints", {})
        merged["notes_by_chunk"][r["chunk"]] = r.get("notes", "")
        merged["edits_by_chunk"][r["chunk"]] = {"edits": r.get("n_edits", 0), "removed": r.get("n_removed", 0)}
    merged["segments"].sort(key=lambda s: s["start"])
    merged["reactions"].sort(key=lambda s: s["t"])
    out_path.write_text(json.dumps(merged, indent=1))
    return merged
