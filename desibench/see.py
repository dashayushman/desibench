"""Step 4, watching the screen → runs/<arm>/frames.json. Both AIs look at the same frames (1 every 5 s).
openai: gpt-6-luna on each 4x4 grid (16 frames per image, prompted for shot/people/speaker/text).
sarvam: Sarvam Vision (Document AI "digitise") on the single frames, 10 per ZIP job. It is not
        promptable: it returns OCR text blocks, or a caption ("image" block) when there is no text."""
from __future__ import annotations

import base64
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

from .config import ARMS, FRAME_EVERY_S, INR_PER_USD, SARVAM_VISION_INR_PER_PAGE, SHOW_CONTEXT, cred, run_dir, video_dir
from .ledger import Ledger
from .llm import chat, parse_json
from .refine import people_str

PROMPT = """{show}
Episode: {title}. People on the show: {handles}.

This image is a 4x4 grid of frames, read left-to-right then top-to-bottom. Tile 1 is at episode time {t0} and each following tile is +{step} s (tile i = {t0} + (i-1)*{step} s). Tiles may be black/empty at the very end.

Known people so far (from earlier frames; use to keep names consistent, update if you learn more):
{people}

For EACH of the 16 tiles return: "i" (1-16), "t" (episode seconds), "shot" (wide|two-shot|close-up|graphic|other), "visible" (names if identifiable else short descriptions), "speaking" (who appears to be talking, or null), "text" (ALL on-screen text read verbatim: scores, round cards, question cards, lower-thirds; "" if none), "desc" (one short line).
Then "people": for every person identified, {{"position": "...", "appearance": "..."}} (podium label if visible).
Then "notes": anything structural (round card shown, score change, set change, audience shot).

Return ONLY JSON: {{"tiles":[...16 items...],"people":{{...}},"notes":"..."}}"""


def _see_grid(d, rd, info, grid, people_ctx, led, handles, model):
    img = base64.b64encode((d / "grids" / grid["file"]).read_bytes()).decode()
    prompt = PROMPT.format(show=SHOW_CONTEXT, title=info.get("title"), handles=handles,
                           t0=grid["start"], step=FRAME_EVERY_S, people=json.dumps(people_ctx) if people_ctx else "none yet")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"}},
    ]}]
    raw = rd / "raw" / "see" / f"grid_{grid['idx']:04d}.json"
    if raw.exists():
        content = json.loads(raw.read_text())["content"]
    else:
        content, usage, secs = chat(model, messages, json_mode=True, max_tokens=16000)
        raw.write_text(json.dumps({"usage": usage, "seconds": secs, "content": content}, ensure_ascii=False))
        led.log("see", f"grid_{grid['idx']:04d}", model=model, seconds=secs, usage=usage)
    out = parse_json(content)
    out["grid"] = grid["idx"]
    # normalise tile times from the known mapping (don't trust model arithmetic)
    for t in out.get("tiles", []):
        try:
            i = int(t.get("i", 0))
            if 1 <= i <= len(grid["tile_times"]):
                t["t"] = grid["tile_times"][i - 1]
        except (TypeError, ValueError):
            pass
    return out


def see(video_id: str, arm: str, force: bool = False, workers: int = 6) -> dict:
    d, rd, cfg = video_dir(video_id), run_dir(video_id, arm), ARMS[arm]
    out_path = rd / "frames.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    info = json.loads((d / "ingest.json").read_text())
    led = Ledger(rd / "ledger.jsonl")
    (rd / "raw" / "see").mkdir(parents=True, exist_ok=True)
    if cfg.get("share_frames_with"):  # identical vision input/model → reuse that arm's frames (and its cost rows)
        src = run_dir(video_id, cfg["share_frames_with"])
        out = json.loads((src / "frames.json").read_text())
        for r in Ledger(src / "ledger.jsonl").rows():
            if r["stage"] == "see":
                led.log("see", r["call"], model=r["model"], seconds=r["seconds"], usage=r.get("usage"), cost_usd=r["cost_usd"], shared_from=cfg["share_frames_with"])
        out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        return out
    if cfg["vendor"] == "sarvam":
        out = _see_sarvam(d, rd, info, led)
        out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        return out
    handles = people_str(video_id, info)
    model = cfg["see"]
    grids = info["grids"]

    # Phase 1: first few grids sequentially to build the people context (intro has lower-thirds).
    people: dict = {}
    results = []
    warm = grids[:3]
    for g in warm:
        r = _see_grid(d, rd, info, g, people, led, handles, model)
        people.update(r.get("people") or {})
        results.append(r)
    # Phase 2: the rest concurrently with the frozen context.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results += list(ex.map(lambda g: _see_grid(d, rd, info, g, people, led, handles, model), grids[len(warm):]))

    for r in results:
        people.update(r.get("people") or {})
    tiles = [t for r in results for t in r.get("tiles", [])]
    tiles.sort(key=lambda t: t.get("t", 0))
    out = {"people": people, "tiles": tiles, "grid_notes": {r["grid"]: r.get("notes", "") for r in results}}
    out_path.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- Sarvam Vision
def _sarvam_job(client, rd, led, batch_idx, frames):
    """One Document AI digitise job over ≤10 frames. Cached per batch in raw/see/."""
    raw = rd / "raw" / "see" / f"docai_{batch_idx:04d}.json"
    if raw.exists():
        return json.loads(raw.read_text())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, f in enumerate(frames):
            z.write(f["path"], f"p{i:02d}.jpg")  # pages come back in this order
    for attempt in range(6):
        try:
            t0 = time.time()
            job = client.doc_ai.digitise(file=[("frames.zip", buf.getvalue(), "application/zip")],
                                         output_format="json", language="en-IN")
            while True:
                time.sleep(20)
                try:
                    st = client.doc_ai.get_status(job.job_id)
                except Exception:  # rate-limited poll: keep waiting, never resubmit a started job
                    continue
                if st.status in ("completed", "failed", "partially_completed"):
                    break
            for _ in range(5):
                try:
                    res = json.loads(client.doc_ai.get_results(job.job_id, format="json").model_dump_json())
                    break
                except Exception:
                    time.sleep(20)
            else:
                raise RuntimeError(f"doc-ai results unavailable for {job.job_id}")
            secs = time.time() - t0
            break
        except Exception as e:  # rate limit (10 req/min) or transient
            if attempt == 5:
                raise
            time.sleep(30 * (attempt + 1))
    pages = st.usage.pages_succeeded if st.usage else len(frames)
    led.log("see", f"docai_{batch_idx:04d}", model="sarvam-vision", seconds=secs,
            cost_usd=pages * SARVAM_VISION_INR_PER_PAGE / INR_PER_USD, pages=pages, status=st.status, job_id=job.job_id)
    out = {"frames": frames, "status": st.status, "seconds": secs, "result": res}
    raw.write_text(json.dumps(out, ensure_ascii=False))
    return out


def _see_sarvam(d, rd, info, led, workers: int = 3):
    import truststore
    from sarvamai import SarvamAI

    truststore.inject_into_ssl()
    client = SarvamAI(api_subscription_key=cred("SARVAM_API_KEY"))
    frames = [{"t": f["t"], "path": str(d / "frames" / f["file"])} for f in info["frames"]]
    batches = [frames[i:i + 10] for i in range(0, len(frames), 10)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        jobs = list(ex.map(lambda b: _sarvam_job(client, rd, led, b[0], b[1]), enumerate(batches)))
    tiles = []
    for j in jobs:
        pages = [p for doc in j["result"].get("documents") or [] for p in doc.get("pages") or []]
        pages.sort(key=lambda p: p.get("page_num") or 0)
        for f, p in zip(j["frames"], pages):
            blocks = sorted(p.get("blocks") or [], key=lambda b: b.get("reading_order") or 0)
            text = " | ".join(b["text"].strip() for b in blocks if b.get("layout_tag") != "image" and b.get("text"))
            desc = " ".join(b["text"].strip() for b in blocks if b.get("layout_tag") == "image" and b.get("text"))
            tiles.append({"t": f["t"], "shot": None, "visible": None, "speaking": None, "text": text, "desc": desc,
                          "layout_tags": [b.get("layout_tag") for b in blocks]})
    tiles.sort(key=lambda t: t["t"])
    return {"people": {}, "tiles": tiles, "engine": "sarvam-vision (doc-ai digitise)"}
