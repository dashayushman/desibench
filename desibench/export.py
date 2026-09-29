"""S6 Export: deterministic minute table (+ LLM one-liners for every minute), episode.json for the
web app, and a formatted minute-by-minute transcript (transcript.md)."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from .config import ARMS, SHOW_CONTEXT, WINDOW_SECONDS, run_dir, video_dir
from .ledger import Ledger
from .llm import chat_json_cached

MIN_PROMPT = """{show}
Episode: {title}. Below is the transcript (speaker-named) of minutes {m0}-{m1} of the episode with audience reactions, plus the round each minute belongs to.
For EVERY minute number in this list: {minutes}
write "summary" (one factual line, ≤ 25 words, what happens in that minute) and "topics" (1-3 short tags).
Return ONLY JSON: {{"minutes": {{"<minute>": {{"summary": "...", "topics": ["..."]}}}}}}

TRANSCRIPT:
{transcript}"""


def _mmss(s: float) -> str:
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600 else _mmss(s)


def _name(spk: str, smap: dict) -> str:
    v = smap.get(spk)
    if isinstance(v, dict):
        v = v.get("person")
    return v or spk


def resolve_people(fused: dict, segs: list[dict]) -> None:
    """Name each segment's speaker: the fuse window's own map first (ids can be per-chunk, e.g. OpenAI
    diarization), then the episode-level map. Sets s["person"] in place."""
    ep_map = (fused.get("episode") or {}).get("speaker_map") or {}
    win = {int(w["window"][0]): w.get("speaker_map") or {} for w in fused.get("windows", []) if w.get("window")}
    for s in segs:
        wm = win.get(int(s["start"] // WINDOW_SECONDS * WINDOW_SECONDS), {})
        n = _name(s["speaker"], wm) if s["speaker"] in wm else s["speaker"]
        if n == s["speaker"] or str(n).lower().startswith("unknown"):
            n = _name(s["speaker"], ep_map)
        s["person"] = n


def _round_at(t: float, rounds: list) -> str | None:
    for r in rounds:
        s, e = r.get("start") or 0, r.get("end")
        if t >= s and (e is None or t < e):
            return r.get("name")
    return None


def build_minutes(info, refined, frames, fused, led, model, rd) -> list[dict]:
    dur = info["duration_s"]
    n = math.ceil(dur / 60)
    ep = fused.get("episode", {})
    smap = ep.get("speaker_map") or {}
    rounds = ep.get("rounds") or []
    segs = refined["segments"]
    reacts = refined["reactions"]
    tiles = sorted(frames.get("tiles", []), key=lambda t: t.get("t", 0))
    hm = info.get("heatmap") or {}
    scores, bucket = hm.get("scores") or [], hm.get("bucket_s") or 0
    llm_minutes = {int(m["minute"]): m for m in fused.get("minutes", []) if isinstance(m.get("minute"), (int, float))}
    scenes = fused.get("scenes", [])

    rows = []
    for m in range(n):
        a, b = m * 60, min((m + 1) * 60, dur)
        share: Counter = Counter()
        words = 0
        for s in segs:
            ov = min(s["end"], b) - max(s["start"], a)
            if ov > 0:
                share[s["person"]] += ov
                words += len(s["text"].split()) * (ov / max(s["end"] - s["start"], 0.1))
        tot = sum(share.values()) or 1
        rx = [r for r in reacts if a <= r["t"] < b]
        cnt = Counter(r["type"] for r in rx)
        laugh_int = sum(r.get("intensity", 1) for r in rx if r["type"] == "laughter")
        heats = [s for i, s in enumerate(scores) if a <= i * bucket < b] if bucket else []
        if not heats and bucket:
            i = min(int(a // bucket), len(scores) - 1)
            heats = [scores[i]] if scores else []
        mt = [t for t in tiles if a <= t.get("t", -1) < b]
        cuts = sum(1 for x, y in zip(mt, mt[1:]) if x.get("shot") != y.get("shot"))
        shot = Counter(t.get("shot") for t in mt).most_common(1)[0][0] if mt else None
        texts = []
        for t in mt:
            tx = (t.get("text") or "").strip()
            if tx and tx not in texts:
                texts.append(tx[:120])
        scene = next((s for s in scenes if s.get("start", 0) <= a < (s.get("end") or 0)), None)
        lm = llm_minutes.get(m, {})
        rows.append({
            "minute": m, "start": a, "end": round(b, 1), "label": _mmss(a),
            "round": _round_at(a, rounds), "scene": scene.get("title") if scene else None,
            "speakers": {k: round(v / tot, 2) for k, v in share.most_common()},
            "words": int(words), "laughs": cnt.get("laughter", 0), "laugh_intensity": laugh_int,
            "applause": cnt.get("applause", 0), "cheers": cnt.get("cheer", 0), "groans": cnt.get("groan", 0),
            "heat": round(sum(heats) / len(heats), 3) if heats else None,
            "cuts": cuts, "shot": shot, "on_screen": texts[:4],
            "summary": lm.get("summary"), "topics": lm.get("topics") or [],
        })
    # fill missing summaries with a cheap model, per window
    missing = [r for r in rows if not r["summary"]]
    if missing:
        title = info.get("title")

        def fill(w0):
            w1 = min(w0 + WINDOW_SECONDS, dur)
            mins = [r["minute"] for r in missing if w0 <= r["start"] < w1]
            if not mins:
                return {}
            lines = []
            for s in segs:
                if s["end"] > w0 and s["start"] < w1:
                    lines.append(f"[{_mmss(s['start'])}] {s['person']}: {s['text']}")
            for r in reacts:
                if w0 <= r["t"] < w1:
                    lines.append(f"[{_mmss(r['t'])}] *{r['type']}*")
            lines.sort()
            prompt = MIN_PROMPT.format(show=SHOW_CONTEXT, title=title, m0=int(w0 // 60), m1=int((w1 - 1) // 60),
                                       minutes=mins, transcript="\n".join(lines))
            (rd / "raw" / "export").mkdir(parents=True, exist_ok=True)
            out = chat_json_cached(model, [{"role": "user", "content": prompt}],
                                   rd / "raw" / "export" / f"minutes_{int(w0 // 60):03d}.json", led, "export",
                                   f"minute_summaries_{int(w0 // 60):03d}")
            return out.get("minutes") or {}

        starts = list(range(0, int(dur), WINDOW_SECONDS))
        with ThreadPoolExecutor(max_workers=4) as ex:
            for res in ex.map(fill, starts):
                for k, v in res.items():
                    try:
                        mi = int(k)
                    except ValueError:
                        continue
                    if mi < len(rows):
                        rows[mi]["summary"] = rows[mi]["summary"] or v.get("summary")
                        rows[mi]["topics"] = rows[mi]["topics"] or v.get("topics") or []
    return rows


def build_transcript_md(info, refined, fused, minutes) -> str:
    ep = fused.get("episode", {})
    smap = ep.get("speaker_map") or {}
    vid = info["video_id"]
    out = [f"# {info.get('title')}", "", f"Minute-by-minute transcript · {_hms(info['duration_s'])} · speakers: "
           + ", ".join(sorted({s["person"] for s in refined["segments"]})), ""]
    segs = refined["segments"]
    reacts = refined["reactions"]
    for r in minutes:
        a, b = r["start"], r["end"]
        head = f"## {r['label']} — {r['round'] or ''}"
        meta = f"laughs {r['laughs']} · applause {r['applause']} · heat {r['heat'] if r['heat'] is not None else '–'}"
        out += [head, f"*{r['summary'] or ''}*  ", f"`{meta}` · [▶ YouTube](https://www.youtube.com/watch?v={vid}&t={int(a)}s)", ""]
        events = [(s["start"], "seg", s) for s in segs if s["start"] >= a and s["start"] < b]
        events += [(x["t"], "rx", x) for x in reacts if a <= x["t"] < b]
        events.sort(key=lambda e: e[0])
        for t, kind, obj in events:
            if kind == "seg":
                out.append(f"[{_mmss(t)}] **{obj['person']}:** {obj['text']}  ")
            else:
                out.append(f"[{_mmss(t)}] *({obj['type']}{' ×' + str(obj['intensity']) if obj.get('intensity', 1) > 1 else ''})*  ")
        out.append("")
    return "\n".join(out)


def export(video_id: str, arm: str, force: bool = False) -> dict:
    d, rd, cfg = video_dir(video_id), run_dir(video_id, arm), ARMS[arm]
    out_path = rd / "episode.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    info = json.loads((d / "ingest.json").read_text())
    refined = json.loads((rd / "refined.json").read_text())
    frames = json.loads((rd / "frames.json").read_text())
    fused = json.loads((rd / "fused.json").read_text())
    led = Ledger(rd / "ledger.jsonl")
    ep = fused.get("episode", {})
    smap = ep.get("speaker_map") or {}

    resolve_people(fused, refined["segments"])
    minutes = build_minutes(info, refined, frames, fused, led, cfg["llm"], rd)

    # speaker stats
    persons: dict = defaultdict(lambda: {"talk_s": 0.0, "segments": 0, "words": 0, "laughs_after": 0})
    segs = refined["segments"]
    reacts = refined["reactions"]
    laugh_ts = sorted(r["t"] for r in reacts if r["type"] == "laughter")
    for s in segs:
        p = persons[s["person"]]
        p["talk_s"] += max(s["end"] - s["start"], 0)
        p["segments"] += 1
        p["words"] += len(s["text"].split())
        # laugh within 3 s after the segment ends, credited to the speaker
        if any(s["end"] - 0.5 <= t <= s["end"] + 3.0 for t in laugh_ts):
            p["laughs_after"] += 1
    tot_talk = sum(p["talk_s"] for p in persons.values()) or 1
    speaker_stats = [{"person": k, **{kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()},
                      "share": round(v["talk_s"] / tot_talk, 3)} for k, v in persons.items()]
    speaker_stats.sort(key=lambda x: -x["talk_s"])

    tiles = [{"t": t.get("t"), "shot": t.get("shot"), "speaking": t.get("speaking"), "text": (t.get("text") or "")[:160],
              "desc": (t.get("desc") or "")[:160]} for t in frames.get("tiles", [])]
    ledger_summary = led.summary()

    episode = {
        "video_id": video_id, "arm": arm, "models": cfg, "title": info.get("title"), "duration_s": info["duration_s"],
        "view_count": info.get("view_count"), "upload_date": info.get("upload_date"),
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
        "speaker_map": smap, "persons": ep.get("persons"), "rounds": ep.get("rounds"), "scores": ep.get("scores"),
        "winner": ep.get("winner"), "summary": ep.get("summary"), "top_moments": ep.get("top_moments"),
        "stats": ep.get("stats"),
        "minutes": minutes,
        "scenes": fused.get("scenes"), "questions": fused.get("questions"), "bits": fused.get("bits"),
        "transcript": [{"start": s["start"], "end": s["end"], "speaker": s["person"], "speaker_id": s["speaker"], "text": s["text"]} for s in segs],
        "reactions": reacts, "events": refined.get("events"),
        "heatmap": info.get("heatmap"), "key_moments": info.get("key_moments"),
        "tiles": tiles, "people_visual": frames.get("people"),
        "speaker_stats": speaker_stats,
        "cost": {k: {"cost_usd": round(v["cost_usd"], 3), "calls": v["calls"], "in": v["in"], "out": v["out"]}
                 for k, v in ledger_summary.items() if not k.startswith("_") and not k.endswith("_superseded")},
        "cost_total_usd": round(sum(v["cost_usd"] for k, v in ledger_summary.items()
                                    if not k.startswith("_") and not k.endswith("_superseded")), 3),
        "grid_seconds": 80, "frame_every_s": 5,
    }
    out_path.write_text(json.dumps(episode, ensure_ascii=False))
    (rd / "transcript.md").write_text(build_transcript_md(info, refined, fused, minutes))
    return episode
