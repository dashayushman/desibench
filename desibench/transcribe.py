"""S2 Transcribe → runs/<arm>/segments.json (diarized, timestamped segments).

sarvam: saaras:v4 batch job on the whole episode (diarization + timestamps + keyterms).
openai: per 10-min chunk, gpt-transcribe (newest, best text, no timestamps/speakers) and
        gpt-4o-transcribe-diarize (the only OpenAI model with speakers + timestamps); the
        gpt-transcribe words are aligned onto the diarized segments. Speaker ids are per chunk
        ("c03B") because OpenAI diarization labels restart on every request.
Raw vendor outputs are kept under runs/<arm>/raw/stt/ for the benchmark."""
from __future__ import annotations

import difflib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import truststore

from .config import ARMS, GPT_TRANSCRIBE_USD_PER_MIN, PRICES, SARVAM_MODE, context, cred, run_dir, video_dir
from .ledger import Ledger, sarvam_cost
from .llm import transcribe_file

truststore.inject_into_ssl()


def transcribe(video_id: str, arm: str, force: bool = False) -> list[dict]:
    rd = run_dir(video_id, arm)
    out_path = rd / "segments.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())["segments"]
    d = video_dir(video_id)
    info = json.loads((d / "ingest.json").read_text())
    led = Ledger(rd / "ledger.jsonl")
    raw_dir = rd / "raw" / "stt"
    raw_dir.mkdir(parents=True, exist_ok=True)
    keyterms = (context(video_id).get("keyterms") or []) if ARMS[arm].get("keyterms", True) else []
    fn = _sarvam if ARMS[arm]["vendor"] == "sarvam" else _openai
    segs, meta = fn(d, info, raw_dir, led, keyterms, ARMS[arm])
    segs.sort(key=lambda s: s["start"])
    out_path.write_text(json.dumps({"engine": meta, "keyterms": keyterms, "segments": segs}, indent=1, ensure_ascii=False))
    return segs


def stt_ablation(video_id: str, arm: str) -> None:
    """Same STT with NO keyterms/keywords → raw/stt_nokeyterms/ (benchmark only, not used downstream)."""
    d, rd, cfg = video_dir(video_id), run_dir(video_id, arm), ARMS[arm]
    info = json.loads((d / "ingest.json").read_text())
    raw_dir = rd / "raw" / "stt_nokeyterms"
    raw_dir.mkdir(parents=True, exist_ok=True)
    led = Ledger(rd / "ledger_ablation.jsonl")
    if cfg["vendor"] == "sarvam":
        _sarvam(d, info, raw_dir, led, [], cfg)
        return
    fields = {"response_format": "json"}
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda c: _cached(raw_dir / f"gpt-transcribe_chunk_{c['idx']:03d}.json",
                                      lambda: transcribe_file(cfg["stt"], d / "chunks" / c["file"], fields)), info["chunks"]))


# ---------------------------------------------------------------- Sarvam
def _sarvam(d, info, raw_dir, led, keyterms, cfg):
    from sarvamai import SarvamAI

    files = list(raw_dir.glob("*.json"))
    if not files:
        client = SarvamAI(api_subscription_key=cred("SARVAM_API_KEY"))
        t0 = time.time()
        job = client.speech_to_text_job.create_job(
            model=cfg["stt"], mode=cfg.get("mode", SARVAM_MODE), language_code=cfg.get("language", "unknown"),
            with_diarization=True, with_timestamps=True, keyterms=keyterms[:50] or None)
        job.upload_files(file_paths=[str(d / "audio.mp3")])
        t_up = time.time()
        job.start()
        job.wait_until_complete(timeout=3600)
        status = job.get_status()
        t_done = time.time()
        job.download_outputs(output_dir=str(raw_dir))
        led.log("transcribe", "sarvam_batch", model=cfg["stt"], seconds=t_done - t0,
                cost_usd=sarvam_cost(info["duration_s"]), job_id=job.job_id, state=str(status.job_state),
                upload_s=round(t_up - t0, 1), audio_s=info["duration_s"])
        files = list(raw_dir.glob("*.json"))
        if not files:
            raise RuntimeError(f"Sarvam job {job.job_id} produced no output ({status.job_state})")
    raw = json.loads(files[0].read_text())
    entries = (raw.get("diarized_transcript") or {}).get("entries") or []
    segs = [{"start": round(e["start_time_seconds"], 2), "end": round(e["end_time_seconds"], 2),
             "speaker": f"spk{e['speaker_id']}", "text": e["transcript"]} for e in entries]
    return segs, {"vendor": "sarvam", "model": cfg["stt"], "mode": cfg.get("mode", SARVAM_MODE),
                  "language": cfg.get("language", "unknown"), "keyterms": bool(keyterms), "language_code": raw.get("language_code")}


# ---------------------------------------------------------------- OpenAI
def _norm(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


def align_words(diar_segs: list[dict], text: str) -> tuple[list[dict], float]:
    """Put the words of `text` (no timing) onto the diarized segments by sequence alignment.
    Unmatched words follow the previous matched word. Returns (segments, matched_fraction)."""
    words = text.split()
    dw = [(_norm(w), i) for i, s in enumerate(diar_segs) for w in s["text"].split()]
    sm = difflib.SequenceMatcher(a=[x for x, _ in dw], b=[_norm(w) for w in words], autojunk=False)
    owner: list = [None] * len(words)
    matched = 0
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            owner[b + k] = dw[a + k][1]
        matched += n
    last = next((o for o in owner if o is not None), 0)
    for i, o in enumerate(owner):
        if o is None:
            owner[i] = last
        else:
            last = o
    buckets: dict[int, list[str]] = {}
    for w, o in zip(words, owner):
        buckets.setdefault(o, []).append(w)
    out = []
    for i, s in enumerate(diar_segs):
        if i in buckets:
            out.append({**s, "text": " ".join(buckets[i])})
    return out, matched / max(len(words), 1)


def _cached(path, call):
    if path.exists():
        return json.loads(path.read_text()), None
    j, secs = call()
    path.write_text(json.dumps(j, ensure_ascii=False))
    return j, secs


def _openai_chunk(d, chunk, raw_dir, led, keyterms, cfg):
    f = d / "chunks" / chunk["file"]
    idx, c0 = chunk["idx"], chunk["start"]
    fields = {"response_format": "json"}
    if keyterms:
        fields["keywords[]"] = keyterms
    gt, s1 = _cached(raw_dir / f"gpt-transcribe_chunk_{idx:03d}.json",
                     lambda: transcribe_file(cfg["stt"], f, fields))
    if s1 is not None:
        secs_audio = (gt.get("usage") or {}).get("seconds") or (chunk["end"] - c0)
        led.log("transcribe", f"gpt-transcribe_{idx:03d}", model=cfg["stt"], seconds=s1,
                cost_usd=secs_audio / 60 * GPT_TRANSCRIBE_USD_PER_MIN, audio_s=secs_audio)
    di, s2 = _cached(raw_dir / f"diarize_chunk_{idx:03d}.json",
                     lambda: transcribe_file(cfg["diarize"], f, {"response_format": "diarized_json", "chunking_strategy": "auto"}))
    if s2 is not None:
        u = di.get("usage") or {}
        usage = {"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0),
                 "prompt_tokens_details": {"audio_tokens": (u.get("input_token_details") or {}).get("audio_tokens", 0)}}
        p = PRICES[cfg["diarize"]]
        led.log("transcribe", f"diarize_{idx:03d}", model=cfg["diarize"], seconds=s2, usage=usage,
                cost_usd=(usage["prompt_tokens"] * p["in"] + usage["completion_tokens"] * p["out"]) / 1e6)
    dsegs = [{"start": round(s["start"] + c0, 2), "end": round(s["end"] + c0, 2),
              "speaker": f"c{idx:02d}{s.get('speaker', '?')}", "text": s.get("text", "").strip()}
             for s in di.get("segments") or [] if s.get("text", "").strip()]
    segs, matched = align_words(dsegs, gt.get("text", ""))
    return segs, {"chunk": idx, "aligned_fraction": round(matched, 3), "diarize_segments": len(dsegs),
                  "languages": gt.get("languages")}


def _openai(d, info, raw_dir, led, keyterms, cfg):
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda c: _openai_chunk(d, c, raw_dir, led, keyterms, cfg), info["chunks"]))
    segs = [s for r in res for s in r[0]]
    return segs, {"vendor": "openai", "model": f"{cfg['stt']} + {cfg['diarize']}", "chunks": [r[1] for r in res]}
