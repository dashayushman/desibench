"""S5 Fuse → runs/<arm>/fused.json: the arm's LLM does a pass per 10-min window + an episode pass."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from .config import ARMS, SHOW_CONTEXT, WINDOW_SECONDS, run_dir, video_dir
from .ledger import Ledger
from .llm import chat_json_cached
from .refine import people_str

WINDOW_PROMPT = """{show}
Episode: {title}. People on the show: {handles}.
People seen on screen (from frame analysis): {people}
Speaker-id hints from the audio model (per chunk): {hints}

Below is everything we know about the window {w0}-{w1} s of the episode. All times are episode seconds.

TRANSCRIPT (corrected; speaker ids in this window: {ids}):
{transcript}

AUDIENCE REACTIONS (t, type, intensity):
{reactions}

FRAMES, one every 5 s (t | shot | visible | speaking | on-screen text | description); empty fields = not provided by the vision model:
{frames}

YOUTUBE: key moments in window: {key_moments}; "most replayed" heat per ~{bucket}s bucket (t:score 0-1): {heat}

Produce, for THIS window only, ONLY JSON:
{{"speaker_map":{{"<speaker id exactly as in the transcript>":{{"person":"...","confidence":0.0}}}},
 "rounds":[{{"name":"...","start":0,"end":null,"rules":"..."}}],
 "questions":[{{"t":0,"round":"...","question":"...","topic":"...","answered_by":"...","answer":"...","correct":true,"points":0}}],
 "bits":[{{"t":0,"type":"joke|roast|callback|bit|banter","performer":"...","target":"...","summary":"...","laugh":0}}],
 "moments":[{{"t":0,"title":"...","why":"..."}}],
 "minutes":[{{"minute":0,"round":"...","speakers":{{"Name":0.0}},"laughs":0,"applause":0,"heat":0.0,"score_on_screen":"...","topics":["..."],"summary":"..."}}],
 "scenes":[{{"start":0,"end":0,"title":"...","participants":["..."],"summary":"...","laugh":0,"heat":0.0}}]}}
Rules: speaker_map must have one entry per speaker id listed above, keyed by that exact id; name people using it (host + guests); minutes must cover every whole minute that starts in the window; laugh = count of laughter reactions in the minute/scene; heat = mean heat score; keep summaries to one line; be factual, cite nothing you cannot see in the inputs."""

EPISODE_PROMPT = """{show}
Episode: {title}. People on the show: {handles}. Duration {dur} s.
Per-window results (speaker maps, rounds, questions, bits, moments, scenes) follow. Reconcile them into one episode-level view. ONLY JSON:
{{"speaker_map":{{"<speaker id exactly as in the windows>":"Person"}},"persons":["..."],
 "rounds":[{{"name":"...","start":0,"end":0,"rules":"..."}}],
 "scores":{{"Person":0}},"winner":"...",
 "top_moments":[{{"t":0,"title":"...","why":"..."}}],
 "summary":"3-5 sentences","stats":{{"n_questions":0,"n_bits":0,"n_laughs":0}}}}

WINDOWS:
{windows}"""


def _fmt_hints(hints: dict) -> str:
    return "; ".join(f"chunk{c}: " + ", ".join(f"{k}={v.get('name')}({v.get('confidence')})" for k, v in h.items())
                     for c, h in sorted(hints.items(), key=lambda x: int(x[0])))


def _window(rd, info, refined, frames, w_idx, w0, w1, led, handles, people_seen, hints_str, model):
    segs = [s for s in refined["segments"] if s["end"] > w0 and s["start"] < w1]
    transcript = "\n".join(f"[{s['start']:.0f}-{s['end']:.0f}] {s['speaker']}: {s['text']}" for s in segs)
    reacts = [r for r in refined["reactions"] if w0 <= r["t"] < w1]
    reactions = "; ".join(f"{r['t']:.0f} {r['type']} {r.get('intensity', 1)}" for r in reacts) or "none"
    tiles = [t for t in frames["tiles"] if w0 <= t.get("t", -1) < w1]
    frame_lines = "\n".join(f"{t['t']:.0f} | {t.get('shot') or ''} | {t.get('visible') or ''} | {t.get('speaking') or ''} | {str(t.get('text') or '')[:160]} | {str(t.get('desc') or '')[:160]}" for t in tiles)
    km = "; ".join(f"{k['t']:.0f} {k['title']}" for k in (info.get("key_moments") or []) if w0 <= k["t"] < w1) or "none"
    hm = info.get("heatmap") or {}
    bucket = hm.get("bucket_s") or 0
    heat = " ".join(f"{int(i * bucket)}:{s}" for i, s in enumerate(hm.get("scores") or []) if w0 <= i * bucket < w1) or "n/a"
    ids = ", ".join(sorted({s["speaker"] for s in segs}))
    prompt = WINDOW_PROMPT.format(show=SHOW_CONTEXT, title=info.get("title"), handles=handles, people=people_seen, ids=ids,
                                  hints=hints_str, w0=w0, w1=w1, transcript=transcript, reactions=reactions,
                                  frames=frame_lines, key_moments=km, bucket=int(bucket), heat=heat)
    out = chat_json_cached(model, [{"role": "user", "content": prompt}], rd / "raw" / "fuse" / f"window_{w_idx:03d}.json",
                           led, "fuse", f"window_{w_idx:03d}")
    out["window"] = [w0, w1]
    return out


def fuse(video_id: str, arm: str, force: bool = False, workers: int = 4) -> dict:
    d, rd, model = video_dir(video_id), run_dir(video_id, arm), ARMS[arm]["llm"]
    out_path = rd / "fused.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    info = json.loads((d / "ingest.json").read_text())
    refined = json.loads((rd / "refined.json").read_text())
    frames = json.loads((rd / "frames.json").read_text())
    led = Ledger(rd / "ledger.jsonl")
    (rd / "raw" / "fuse").mkdir(parents=True, exist_ok=True)
    title = info.get("title") or ""
    handles = people_str(video_id, info)
    people_seen = json.dumps(frames.get("people") or {})[:2500]
    hints_str = _fmt_hints(refined.get("speaker_hints_by_chunk") or {})

    dur = info["duration_s"]
    windows = [(i, w0, min(w0 + WINDOW_SECONDS, dur)) for i, w0 in enumerate(range(0, int(dur), WINDOW_SECONDS))]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda w: _window(rd, info, refined, frames, w[0], w[1], w[2], led, handles, people_seen, hints_str, model), windows))

    compact = [{k: r.get(k) for k in ("window", "speaker_map", "rounds", "questions", "bits", "moments", "scenes")} for r in results]
    ep_prompt = EPISODE_PROMPT.format(show=SHOW_CONTEXT, title=title, handles=handles, dur=int(dur),
                                      windows=json.dumps(compact)[:180000])
    episode = chat_json_cached(model, [{"role": "user", "content": ep_prompt}], rd / "raw" / "fuse" / "episode.json",
                               led, "fuse", "episode")

    def items(key):  # keep only well-formed dict items; count the rest (schema discipline, reported in the benchmark)
        good = [x for r in results for x in (r.get(key) or []) if isinstance(x, dict)]
        bad = sum(1 for r in results for x in (r.get(key) or []) if not isinstance(x, dict))
        return good, bad

    fused = {"episode": episode, "windows": results, "schema_errors": {}}
    for key, sort_key in (("minutes", "minute"), ("scenes", "start"), ("questions", "t"), ("bits", "t")):
        good, bad = items(key)
        fused[key] = sorted(good, key=lambda m: m.get(sort_key) or 0 if isinstance(m.get(sort_key) or 0, (int, float)) else 0)
        fused["schema_errors"][key] = bad
    out_path.write_text(json.dumps(fused, indent=1))
    return fused
