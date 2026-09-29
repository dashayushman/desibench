"""Score a run against the ground truth in answers/. Deterministic: no model calls, same result every time.

    python -m desibench score                 # your run (work/)            → results/yours.json
    python -m desibench score --reference DIR # a run laid out like work/  → results/reference.json

Six rounds, as on the site. Each one uses only what can be checked against an answer key:
  1 ears      words wrong vs the reference transcripts (24 one-minute clips), script-agnostic
  2 speakers  voices heard; who said each reference line (host and the three comedians)
  3 room      laughs, applause and cheers heard
  4 eyes      frames described, cost per 1,000 frames (the scene grades need a person; see answers/README.md)
  5 show      the show's own facts (winner, final scores, rounds), its quiz (multiple choice), Who Am I
  6 bill      dollars per hour of video, every call logged
"""
from __future__ import annotations

import difflib
import glob
import json
import re
from collections import Counter
from pathlib import Path

from .config import ANSWERS, ARMS, AI_NAME, EPISODES, INR_PER_USD, LLM_OF

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
PAD = 20  # seconds of transcript context on each side of a clip; words outside the reference are trimmed, never counted


def _load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


# ------------------------------------------------------------------ words wrong (script- and spelling-tolerant)
def _hk(tok: str) -> str:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    t = transliterate(tok, sanscript.DEVANAGARI, sanscript.HK)
    t = t.replace("M", "n").replace("H", "").replace("z", "sh").replace("S", "sh").replace("R", "ri")
    if len(t) > 2 and t.endswith("a") and not tok.endswith(("ा", "ा")):  # the silent schwa at a word's end
        t = t[:-1]
    return t


PHON = [("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("sh", "s"), ("ph", "f"), ("kh", "k"), ("gh", "g"),
        ("th", "t"), ("dh", "d"), ("bh", "b"), ("ch", "c"), ("jh", "j"), ("w", "v"), ("z", "j"), ("q", "k"), ("y", "i")]


def norm_tokens(text: str, skeleton: bool = True) -> list[str]:
    """Hindi in Devanagari or in Latin letters counts the same: Devanagari → Latin, digits → words, lowercase,
    no punctuation, a light phonetic squash (aa→a, sh→s, w→v…), and a consonant skeleton (vowels differ across
    spellings: karne/karane). Applied identically to the reference and to both AIs."""
    from num2words import num2words
    out = []
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    for raw in re.findall(r"[ऀ-ॿ]+|[A-Za-z']+|\d+", text):
        if DEVANAGARI.search(raw):
            words = [_hk(raw)]
        elif raw.isdigit():
            words = num2words(int(raw)).replace("-", " ").replace(",", "").split()
        else:
            words = [raw.replace("'", "")]
        for w in words:
            w = re.sub(r"[^a-z]", "", w.lower())
            if w == "and":
                continue
            for a, b in PHON:
                w = w.replace(a, b)
            w = re.sub(r"(.)\1+", r"\1", w)
            if skeleton and len(w) > 1:
                w = w[0] + re.sub(r"[aeiou]", "", w[1:])
            if w:
                out.append(w)
    return out


def trim_to_ref(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """The stretch of the AI's words that best matches the whole reference (semi-global edit distance), so
    neighbouring speech in the padding never counts as an error, whatever each AI's segment boundaries are."""
    n, m = len(ref), len(hyp)
    if not n or not m:
        return 0, m
    INF = 10 ** 9
    prev, start_prev = [0] * (m + 1), list(range(m + 1))
    for i in range(1, n + 1):
        cur, start_cur = [i] + [INF] * m, [0] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            a = prev[j - 1] + (ri != hyp[j - 1]); b = prev[j] + 1; c = cur[j - 1] + 1
            if a <= b and a <= c:
                cur[j], start_cur[j] = a, start_prev[j - 1]
            elif b <= c:
                cur[j], start_cur[j] = b, start_prev[j]
            else:
                cur[j], start_cur[j] = c, start_cur[j - 1]
        prev, start_prev = cur, start_cur
    end = min(range(m + 1), key=lambda j: (prev[j], -j))
    return start_prev[end], end


def words_wrong(ref_text: str, hyp_text: str) -> dict:
    import jiwer
    rt, ht = norm_tokens(ref_text), norm_tokens(hyp_text)
    lo, hi = trim_to_ref(rt, ht)
    r, h = " ".join(rt), " ".join(ht[lo:hi])
    if not r:
        return {}
    o = jiwer.process_words(r, h or "∅")
    return {"ref_words": len(r.split()), "wrong": o.substitutions, "missing": o.deletions, "extra": o.insertions}


# ------------------------------------------------------------------ reading a run
def _align_words(diar_segs: list[dict], text: str) -> list[dict]:
    """Put untimed words onto timed segments by sequence alignment (for OpenAI's plain-text transcription)."""
    words = text.split()
    norm = lambda w: re.sub(r"[^\w]", "", w.lower())
    dw = [(i, norm(w)) for i, s in enumerate(diar_segs) for w in s["text"].split()]
    sm = difflib.SequenceMatcher(a=[w for _, w in dw], b=[norm(w) for w in words], autojunk=False)
    owner, last = [None] * len(words), 0
    for a0, b0, size in sm.get_matching_blocks():
        for k in range(size):
            owner[b0 + k] = dw[a0 + k][0]
    out = [dict(s, text="") for s in diar_segs]
    for j, w in enumerate(words):
        last = owner[j] if owner[j] is not None else last
        if out:
            out[last]["text"] = (out[last]["text"] + " " + w).strip()
    return [s for s in out if s["text"]]


def speech_to_text(root: Path, vid: str, arm: str) -> list[dict]:
    """The AI's raw speech-to-text, without name hints (what Round 1 scores)."""
    rd = root / vid / "runs" / arm
    seg = _load(rd / "segments.json", {})
    if not (seg.get("keyterms") or []):
        return sorted(seg.get("segments") or [], key=lambda s: s["start"])
    # a run made with name hints: use its no-hints pass (raw/stt_nokeyterms), as the site does
    nk = sorted(glob.glob(str(rd / "raw" / "stt_nokeyterms" / "*.json")))
    if ARMS[arm]["vendor"] == "sarvam" and nk:
        ents = _load(nk[0])["diarized_transcript"]["entries"]
        return [{"start": e["start_time_seconds"], "end": e["end_time_seconds"], "speaker": f"spk{e['speaker_id']}", "text": e["transcript"]} for e in ents]
    info = _load(root / vid / "ingest.json")
    out = []
    for c in info["chunks"]:
        di = _load(rd / "raw" / "stt" / f"diarize_chunk_{c['idx']:03d}.json", {})
        dsegs = [{"start": s["start"] + c["start"], "end": s["end"] + c["start"], "speaker": f"c{c['idx']:02d}{s.get('speaker')}",
                  "text": s.get("text", "").strip()} for s in di.get("segments") or [] if s.get("text", "").strip()]
        p = rd / "raw" / "stt_nokeyterms" / f"gpt-transcribe_chunk_{c['idx']:03d}.json"
        if p.exists():
            out += _align_words(dsegs, _load(p).get("text", ""))
    return sorted(out, key=lambda s: s["start"])


def _window(segs, a, b) -> str:
    return " ".join(s["text"] for s in segs if a <= s["start"] < b)


def _ledger(p: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    except Exception:
        return []


# ------------------------------------------------------------------ the six rounds
def round_ears(root: Path) -> dict:
    clips = _load(ANSWERS / "clips.json", [])
    out = {}
    for arm in ARMS:
        tot = Counter()
        per = []
        for c in clips:
            ref = _load(ANSWERS / "transcripts" / f"{c['clip']}.json")
            if not ref or not (root / c["video_id"] / "runs" / arm / "segments.json").exists():
                continue
            ref_text = " ".join(l["text"] for l in ref["lines"])
            sc = words_wrong(ref_text, _window(speech_to_text(root, c["video_id"], arm), c["start"] - PAD, c["end"] + PAD))
            tot.update(sc)
            per.append({"clip": c["clip"], **sc})
        n = tot["ref_words"]
        err = tot["wrong"] + tot["missing"] + tot["extra"]
        out[arm] = {"words_wrong": round(err / n, 4) if n else None, "ref_words": n, "wrong": tot["wrong"], "missing": tot["missing"],
                    "extra": tot["extra"], "clips": len(per), "per_clip": per}
    return out


def _person_at(segs, t0, t1):
    best, ov = None, 0.0
    for s in segs:
        o = min(s["end"], t1) - max(s["start"], t0)
        if o > ov:
            best, ov = s.get("speaker"), o
    return best


def round_speakers(root: Path) -> dict:
    clips = {c["clip"]: c for c in _load(ANSWERS / "clips.json", [])}
    out = {}
    for arm in ARMS:
        voices, right, probed, none = {}, 0, 0, 0
        for vid in EPISODES:
            ep = _load(root / vid / "runs" / arm / "episode.json")
            if not ep:
                continue
            names = {p["name"] for p in _load(ANSWERS / vid / "context.json", {}).get("people", [])}
            voices[vid] = len({s.get("speaker_id") for s in ep["transcript"]})
            for f in sorted(glob.glob(str(ANSWERS / "transcripts" / f"{vid}_*.json"))):
                ref = _load(f)
                if ref["clip"] not in clips:
                    continue
                for l in ref["lines"]:
                    if l["speaker"] not in names:
                        continue
                    mm, ss = l["t"].split(":")
                    t0 = int(mm) * 60 + int(ss)
                    got = _person_at(ep["transcript"], t0, t0 + 4)
                    if got is None:  # this AI wrote nothing at that moment: coverage, not attribution
                        none += 1
                        continue
                    probed += 1
                    right += int(got == l["speaker"])
        out[arm] = {"voices_heard": voices, "right_person": round(right / probed, 3) if probed else None,
                    "lines_checked": probed, "lines_with_no_speech": none}
    return out


def round_room(root: Path) -> dict:
    out = {}
    for arm in ARMS:
        per = {}
        for vid in EPISODES:
            ep = _load(root / vid / "runs" / arm / "episode.json")
            if ep:
                per[vid] = dict(Counter(r["type"] for r in ep.get("reactions", [])))
        out[arm] = {"per_episode": per, "reactions_heard": sum(sum(v.values()) for v in per.values()),
                    "laughs_heard": sum(v.get("laughter", 0) for v in per.values())}
    return out


def round_eyes(root: Path) -> dict:
    out = {}
    for arm in ARMS:
        frames, usd = 0, 0.0
        for vid in EPISODES:
            fr = _load(root / vid / "runs" / arm / "frames.json")
            if fr:
                frames += len(fr.get("tiles") or [])
            usd += sum(r.get("cost_usd", 0) for r in _ledger(root / vid / "runs" / arm / "ledger.jsonl") if r.get("stage") == "see")
        out[arm] = {"frames_described": frames, "usd_per_1000_frames": round(usd / frames * 1000, 3) if frames else None}
    return out


def _facts(vid: str, ep: dict) -> dict:
    f = _load(ANSWERS / vid / "facts.json")
    fs = {k: v for k, v in f["final_scores"].items() if k != "evidence"}
    scores = ep.get("scores") or {}
    gold, got = f["rounds"], ep.get("rounds") or []
    near = lambda r, g: isinstance(r.get("start"), (int, float)) and abs(r["start"] - g["start"]) <= 60
    found = sum(1 for g in gold if any(near(r, g) for r in got))
    return {"winner_right": f["winner"].lower() in (ep.get("winner") or "").lower(),
            "final_scores_exact": sum(1 for k, v in fs.items() if scores.get(k) == v), "of": len(fs),
            "rounds_found": found, "rounds": len(gold)}


def round_show(root: Path) -> dict:
    who_items = {i["id"]: i for i in _load(ANSWERS / "who_am_i.json")["items"]}
    quiz_items = {i["id"]: i for i in _load(ANSWERS / "quiz.json")["items"]}
    out = {}
    for arm in ARMS:
        m = LLM_OF[arm]
        facts = {vid: _facts(vid, ep) for vid in EPISODES if (ep := _load(root / vid / "runs" / arm / "episode.json"))}
        who = [r for f in glob.glob(str(root / "controlled" / "who_am_i" / f"*__{m}.json")) if (r := _load(f))["id"] in who_items]
        quiz = [r for f in glob.glob(str(root / "controlled" / "quiz" / f"*__{m}.json")) if (r := _load(f))["id"] in quiz_items]
        mc = [r for r in quiz if r.get("mc")]
        out[arm] = {"llm": m, "facts": facts,
                    "who_am_i_points": sum(r["points"] for r in who) if who else None,
                    "who_am_i_max": sum(10 * len(i["hints"]) for i in who_items.values()),
                    "who_am_i_solved": sum(1 for r in who if r.get("solved_at_hint")), "who_am_i_played": len(who),
                    "quiz_multiple_choice_right": sum(1 for r in mc if r.get("correct")) if mc else None,
                    "quiz_multiple_choice": sum(1 for i in quiz_items.values() if i.get("options"))}
    return out


def round_bill(root: Path) -> dict:
    out = {}
    for arm in ARMS:
        usd, hours, stages = 0.0, 0.0, Counter()
        for vid in EPISODES:
            info = _load(root / vid / "ingest.json")
            rows = _ledger(root / vid / "runs" / arm / "ledger.jsonl")
            if not info or not rows:
                continue
            hours += info["duration_s"] / 3600
            for r in rows:
                st = r.get("stage", "")
                if st.startswith("_") or st.endswith("_superseded"):  # wall-clock rows; our own re-runs
                    continue
                usd += r.get("cost_usd", 0); stages[st] += r.get("cost_usd", 0)
        per_h = usd / hours if hours else None
        out[arm] = {"usd_total": round(usd, 3), "hours_of_video": round(hours, 3), "usd_per_hour": round(per_h, 3) if per_h else None,
                    "inr_per_hour": round(per_h * INR_PER_USD) if per_h else None,
                    "usd_per_hour_by_stage": {k: round(v / hours, 3) for k, v in stages.items()} if hours else {}}
    return out


# ------------------------------------------------------------------ winners (+20 each, as the host scores)
def winners(r: dict) -> dict:
    s, o = "sarvam-tuned", "openai"
    def pick(key, lower=False):
        a, b = key(r, s), key(r, o)
        if a is None or b is None or a == b:
            return None
        return (s if a < b else o) if lower else (s if a > b else o)
    show = lambda r, a: (r["show"][a]["who_am_i_points"] or 0) + 10 * (r["show"][a]["quiz_multiple_choice_right"] or 0) if r["show"][a]["who_am_i_points"] is not None else None
    return {
        "ears": pick(lambda r, a: r["ears"][a]["words_wrong"], lower=True),
        "speakers": pick(lambda r, a: r["speakers"][a]["right_person"]),
        "room": pick(lambda r, a: r["room"][a]["reactions_heard"] or None),
        "eyes": None,  # graded by a person; see the reference results
        "show": pick(show),
        "bill": pick(lambda r, a: r["bill"][a]["usd_per_hour"], lower=True),
    }


def score_run(root: Path) -> dict:
    root = Path(root)
    res = {"run": str(root), "ais": {a: {"name": AI_NAME[a], **ARMS[a]} for a in ARMS},
           "episodes": {v: {"label": l, "title": (_load(root / v / "ingest.json") or {}).get("title"),
                            "duration_s": (_load(root / v / "ingest.json") or {}).get("duration_s")} for v, l in EPISODES.items()},
           "ears": round_ears(root), "speakers": round_speakers(root), "room": round_room(root), "eyes": round_eyes(root),
           "show": round_show(root), "bill": round_bill(root)}
    w = winners(res)
    res["winners"] = w
    res["points"] = {a: 20 * sum(1 for x in w.values() if x == a) for a in ARMS}
    return res
