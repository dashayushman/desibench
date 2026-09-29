"""desibench: run the Sarvam vs OpenAI benchmark on your laptop.

    python -m desibench doctor      # check Python, ffmpeg, yt-dlp and your two keys (never printed)
    python -m desibench download    # the two episodes from YouTube → videos/
    python -m desibench estimate    # what a full run costs and how long it takes
    python -m desibench run         # everything, both AIs, both episodes (asks before spending; --yes to skip)
    python -m desibench score       # your run vs the ground truth → results/yours.json
    python -m desibench serve       # the results page: http://localhost:8000/viewer/
    python -m desibench all         # doctor → download → run → score → serve

Options for estimate/run/all: --episode <id> (one episode), --ai sarvam|openai (one AI); run/all: --yes, --force.
Every step is resumable: run it again and it skips what's done.
"""
from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socketserver
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor

from .config import AI_NAME, ARMS, EPISODES, RESULTS, ROOT, VIDEOS, WORK, has_cred, run_dir, video_dir
from .ledger import Ledger

STAGES = ["transcribe", "refine", "see", "fuse", "export"]
STAGE_WORDS = {"ingest": "cutting audio and frames", "transcribe": "writing down every word", "refine": "cleaning up, hearing the room",
               "see": "watching the screen", "fuse": "making sense of the show", "export": "writing the results"}
AI_ARG = {"sarvam": "sarvam-tuned", "openai": "openai"}
EP_SECONDS = {"_3UvCy7FMTg": 4485, "2pxmtBus3f8": 5415}
MINUTES_PER_EPISODE = {"sarvam-tuned": 120, "openai": 25}  # from my run; Sarvam's vision reads frames one at a time


def _ref() -> dict:
    p = RESULTS / "reference.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------ doctor
def doctor() -> bool:
    ok = True

    def line(good, text, fix=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'✓' if good else '✗'} {text}" + (f"\n      → {fix}" if not good and fix else ""))

    print("desibench doctor")
    line(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}", "Python 3.10 or newer")
    line(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe", "macOS: brew install ffmpeg · Ubuntu: sudo apt install ffmpeg")
    try:
        import httpx, jiwer, json_repair, num2words, sarvamai, truststore, yt_dlp  # noqa: F401
        from indic_transliteration import sanscript  # noqa: F401
        line(True, "Python packages")
    except ImportError as e:
        line(False, f"Python packages ({e.name} missing)", "pip install -r requirements.txt")
    for k in ("SARVAM_API_KEY", "OPENAI_API_KEY"):
        line(has_cred(k), f"{k} set", f"put your key after {k}= in .env" if (ROOT / ".env").exists() else "copy .env.example to .env and fill it in")
    free = shutil.disk_usage(ROOT).free / 1e9
    line(free > 6, f"{free:.0f} GB free (a full run needs about 6 GB)", "free up some disk space")
    have = [v for v in EPISODES if (VIDEOS / f"{v}.mp4").exists()]
    print(f"  · videos downloaded: {len(have)} of {len(EPISODES)}" + ("" if len(have) == len(EPISODES) else "  (python -m desibench download)"))
    print("ready." if ok else "fix the ✗ lines above, then run doctor again.")
    return ok


# ------------------------------------------------------------------ estimate
def estimate(eps: list[str], arms: list[str]) -> dict:
    ref = _ref().get("bill", {})
    hours = sum(EP_SECONDS[v] for v in eps) / 3600
    usd = {a: (ref.get(a, {}).get("usd_per_hour") or {"sarvam-tuned": 5.5, "openai": 1.84}[a]) * hours for a in arms}
    print(f"Estimate for {len(eps)} episode(s), {hours:.2f} hours of video (from my run's logged costs):")
    for a in arms:
        print(f"  {AI_NAME[a]:7} ≈ ${usd[a]:.2f} (≈ ₹{usd[a] * 84:.0f}), about {MINUTES_PER_EPISODE[a] * len(eps) / 60:.1f} h")
    print("  quiz + Who Am I ≈ $0.10")
    total = sum(usd.values()) + 0.10
    print(f"  total ≈ ${total:.2f}. The AIs run side by side, so it takes about as long as the slower one.")
    return {"usd": round(total, 2), "hours_of_video": round(hours, 2)}


# ------------------------------------------------------------------ run
def _stage(vid: str, arm: str, st: str, force: bool):
    from . import export, fuse, refine, see, transcribe
    fns = {"transcribe": transcribe.transcribe, "refine": refine.refine, "see": see.see, "fuse": fuse.fuse, "export": export.export}
    t0 = time.time()
    print(f"[{EPISODES[vid]} · {AI_NAME[arm]}] {STAGE_WORDS[st]}…", flush=True)
    fns[st](vid, arm, force=force)
    Ledger(run_dir(vid, arm) / "ledger.jsonl").log("_wall", st, seconds=time.time() - t0)
    print(f"[{EPISODES[vid]} · {AI_NAME[arm]}] {STAGE_WORDS[st]}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


def run(eps: list[str], arms: list[str], yes: bool, force: bool):
    from . import controlled, ingest
    for v in eps:
        if not (VIDEOS / f"{v}.mp4").exists():
            raise SystemExit(f"{EPISODES[v]} isn't downloaded yet. Run: python -m desibench download")
    if not yes:
        estimate(eps, arms)
        if input("Start? This calls the paid APIs. [y/N] ").strip().lower() not in ("y", "yes"):
            print("Stopped. Nothing was spent.")
            return
    for v in eps:
        print(f"[{EPISODES[v]}] {STAGE_WORDS['ingest']}…", flush=True)
        ingest.ingest(v, force=force)

    def one_ai(arm):  # each AI works through the episodes in order; the two AIs run side by side
        for v in eps:
            for st in STAGES:
                _stage(v, arm, st, force)

    with ThreadPoolExecutor(max_workers=len(arms)) as ex:
        for f in [ex.submit(one_ai, a) for a in arms]:
            f.result()
    print("quiz and Who Am I…", flush=True)
    controlled.quiz()
    controlled.who_am_i()
    score()


# ------------------------------------------------------------------ score, serve
def score() -> dict:
    from .score import score_run
    r = score_run(WORK)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "yours.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print("results/yours.json written.")
    for k, w in r["winners"].items():
        print(f"  {k:9} {AI_NAME[w] if w else '— (see the results page)'}")
    print(f"  points   Sarvam {r['points']['sarvam-tuned']} · OpenAI {r['points']['openai']}")
    return r


def serve(port: int = 8000, open_browser: bool = True):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(ROOT), **k)

        def log_message(self, *a):
            pass

    url = f"http://localhost:{port}/viewer/"
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Quiet) as httpd:
        print(f"results page: {url}   (Ctrl+C to stop)", flush=True)
        if open_browser:
            webbrowser.open(url)
        httpd.serve_forever()


def main(argv=None):
    p = argparse.ArgumentParser(prog="desibench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    d = sub.add_parser("download")
    d.add_argument("--episode", choices=list(EPISODES))
    for name in ("estimate", "run", "all"):
        s = sub.add_parser(name)
        s.add_argument("--episode", choices=list(EPISODES))
        s.add_argument("--ai", choices=list(AI_ARG))
        if name != "estimate":
            s.add_argument("--yes", action="store_true", help="don't ask before spending")
            s.add_argument("--force", action="store_true", help="redo steps that are already done")
        if name == "all":
            s.add_argument("--port", type=int, default=8000)
    sub.add_parser("score")
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--no-open", action="store_true")
    a = p.parse_args(argv)
    eps = [a.episode] if getattr(a, "episode", None) else list(EPISODES)
    arms = [AI_ARG[a.ai]] if getattr(a, "ai", None) else list(ARMS)
    if a.cmd == "doctor":
        return 0 if doctor() else 1
    if a.cmd == "download":
        from .download import download
        download(eps)
    elif a.cmd == "estimate":
        estimate(eps, arms)
    elif a.cmd == "run":
        run(eps, arms, a.yes, a.force)
    elif a.cmd == "score":
        score()
    elif a.cmd == "serve":
        serve(a.port, not a.no_open)
    elif a.cmd == "all":
        if not doctor():
            return 1
        from .download import download
        download(eps)
        run(eps, arms, a.yes, a.force)
        serve(a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
