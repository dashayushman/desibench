"""Same input, both LLMs: the show's own quiz questions and its Who Am I round, so any difference belongs to the model.

    python -m desibench quiz
    python -m desibench whoami
Items come from answers/quiz.json and answers/who_am_i.json; outputs go to work/controlled/.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from .config import ANSWERS, ARMS, LLM_OF, WORK
from .ledger import openai_cost
from .llm import chat, parse_json

OUT = WORK / "controlled"
MODELS = [LLM_OF[a] for a in ARMS]  # sarvam-105b, gpt-6-luna

AUDIENCE = ("You are helping an international viewer (not Indian, doesn't speak Hindi) follow an Indian Hinglish comedy "
            "quiz show, The Nation Wants to Guess.")

WHO_SYS = AUDIENCE + """ We are playing the show's 'Who Am I?' round. You are the contestant. I will read hints (written by comedy writers, so they are puns and roasts) one at a time about a famous person.
After each hint reply ONLY JSON: {"guess": "<full name>" or null, "reasoning": "one short line"}. You have at most 3 guesses in total and only one guess per hint, so guess only when reasonably sure."""


def name_matches(guess: str | None, accept: list[str]) -> bool:
    """A guess is right when it contains one of the item's accepted name parts (answers/who_am_i.json)."""
    if not guess:
        return False
    g = re.sub(r"[^a-z ]", "", guess.lower())
    return any(k in g for k in accept)


def who_am_i() -> list[dict]:
    items = json.loads((ANSWERS / "who_am_i.json").read_text())["items"]
    out_dir = OUT / "who_am_i"
    out_dir.mkdir(parents=True, exist_ok=True)

    def play(job):
        it, m = job
        p = out_dir / f"{it['id']}__{m}.json"
        if p.exists():
            return json.loads(p.read_text())
        msgs = [{"role": "system", "content": WHO_SYS}]
        turns, guesses, cost, secs_tot, solved_at = [], 0, 0.0, 0.0, None
        for i, h in enumerate(it["hints"], 1):
            msgs.append({"role": "user", "content": f"Hint {i}: {h}" + (" (You have used all 3 guesses; reply with guess null.)" if guesses >= 3 else "")})
            try:
                content, usage, secs = chat(m, msgs, json_mode=True, max_tokens=16000)
            except Exception as e:
                turns.append({"hint": i, "error": str(e)[:200]})
                break
            cost += openai_cost(m, usage)
            secs_tot += secs
            msgs.append({"role": "assistant", "content": content or "{}"})
            try:
                r = parse_json(content or "{}")
            except Exception:
                r = {}
            g = r.get("guess") if guesses < 3 else None
            ok = name_matches(g, it["accept"])
            if g:
                guesses += 1
            turns.append({"hint": i, "guess": g, "correct": ok, "reasoning": r.get("reasoning")})
            if ok:
                solved_at = i
                break
        rec = {"id": it["id"], "answer": it["answer"], "model": m, "solved_at_hint": solved_at,
               "hints_available": len(it["hints"]), "points": (len(it["hints"]) + 1 - solved_at) * 10 if solved_at else 0,
               "guesses_used": guesses, "turns": turns, "cost_usd": round(cost, 6), "seconds": round(secs_tot, 1)}
        if not any(t.get("error") for t in turns):  # a failed call isn't an answer: don't save it, so a rerun retries
            p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return rec

    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(play, [(it, m) for it in items for m in MODELS]))


QUIZ_PROMPT = AUDIENCE + """ Answer this quiz question from the show. {kind}
Return ONLY JSON: {{"answer": "{fmt}", "explanation": "one or two lines"}}

QUESTION: {q}
{opts}"""


def quiz() -> list[dict]:
    items = json.loads((ANSWERS / "quiz.json").read_text())["items"]
    out_dir = OUT / "quiz"
    out_dir.mkdir(parents=True, exist_ok=True)

    def go(job):
        it, m = job
        p = out_dir / f"{it['id']}__{m}.json"
        if p.exists():
            return json.loads(p.read_text())
        mc = bool(it.get("options"))
        prompt = QUIZ_PROMPT.format(kind="Pick exactly one option letter." if mc else "Give a short answer.",
                                    fmt="<letter>" if mc else "<short answer>", q=it["question"],
                                    opts="\n".join(f"{k}. {v}" for k, v in (it.get("options") or {}).items()))
        rec = {"id": it["id"], "model": m, "era": it["era"], "mc": mc, "gold": it["answer"]}
        try:
            content, usage, secs = chat(m, [{"role": "user", "content": prompt}], json_mode=True, max_tokens=16000)
            r = parse_json(content or "{}")
            rec.update(answer=r.get("answer"), explanation=r.get("explanation"), seconds=round(secs, 1),
                       cost_usd=round(openai_cost(m, usage), 6))
            if mc:  # multiple choice is checked here; open answers are shown next to the right answer
                rec["correct"] = str(r.get("answer", "")).strip().upper()[:1] == it["answer"]
        except Exception as e:
            rec["error"] = str(e)[:200]
        if "error" not in rec:  # a failed call isn't an answer: don't save it, so a rerun retries
            p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return rec

    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(go, [(it, m) for it in items for m in MODELS]))
