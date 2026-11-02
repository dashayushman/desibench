"""AI Wants to Guess: any model plays the show's 42 Who Am I rounds, hint by hint, once.

Same rules as the reels: one hint at a time, at most 3 guesses and 1 per hint, no feedback on a wrong guess. A round
already played is never replayed, so a model can't be re-rolled for a better score. Claude models go to Anthropic,
everything else to OpenAI or whatever OPENAI_BASE_URL points at.
"""
from __future__ import annotations

import json
import statistics as st
import time

from .config import ANSWERS, RESULTS, WORK, cred
from .llm import _http, chat, parse_json

SYSTEM = """You are a blindfolded contestant on The Nation Wants to Guess, an Indian Hinglish comedy quiz show. This is the 'Who Am I?' round. The host reads hints, written by comedy writers as puns and roasts, one at a time about a famous person.
After each hint reply ONLY JSON: {"guess": "<full name>" or null, "reasoning": "one short line"}. You have at most 3 guesses in total and only one guess per hint, so you may pass by replying with guess null. You see nothing but the hints."""


def rounds(only: list[str] | None = None) -> list[dict]:
    items = json.loads((ANSWERS / "wants_to_guess.json").read_text())["rounds"]
    return [r for r in items if not only or r["id"] in only]


def _claude(model: str, system: str, messages: list[dict]) -> tuple[str, float]:
    t0 = time.time()
    r = _http().post("https://api.anthropic.com/v1/messages", json={"model": model, "max_tokens": 4000, "system": system, "messages": messages},
                     headers={"x-api-key": cred("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        raise RuntimeError(f"{model} call failed: HTTP {r.status_code}: {r.text[:300]}")
    return "".join(b.get("text", "") for b in r.json()["content"]), time.time() - t0


def turn(model: str, system: str, messages: list[dict]) -> tuple[str, float]:
    if model.startswith("claude"):
        return _claude(model, system, messages)
    text, _, secs = chat(model, [{"role": "system", "content": system}, *messages], json_mode=True, max_tokens=16000)
    return text, secs


def right(guess: str | None, r: dict) -> bool:
    return bool(guess) and any(k in guess.lower() for k in r["accept"])


def play(r: dict, model: str) -> dict:
    p = WORK / "guess" / model.replace("/", "_") / f"{r['id']}.json"
    if p.exists():
        return json.loads(p.read_text())
    system = SYSTEM + (f"\n{r['rule']}" if r.get("rule") else "")
    messages, turns, guesses, solved_at = [], [], 0, None
    for i, h in enumerate(r["hints"], 1):
        out_of_guesses = " (You have used all 3 guesses; reply with guess null.)" if guesses >= 3 else ""
        messages.append({"role": "user", "content": f"Hint {i}: {h}{out_of_guesses} Reply in JSON."})
        text, secs = turn(model, system, messages)
        messages.append({"role": "assistant", "content": text})
        got = parse_json(text or "{}")
        guess = got.get("guess") if guesses < 3 else None
        ok = right(guess, r)
        guesses += 1 if guess else 0
        turns.append({"hint": i, "guess": guess, "correct": ok, "reasoning": got.get("reasoning"), "seconds": round(secs, 1)})
        if ok:
            solved_at = i
            break
    rec = {"id": r["id"], "answer": r["answer"], "model": model, "solved_at_hint": solved_at, "guesses_used": guesses, "turns": turns}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def verdict(rec: dict, r: dict) -> str:
    """Whoever says it first on the episode's clock: won, beaten or nobody."""
    if rec["solved_at_hint"]:
        said = r["hint_read_end"][rec["solved_at_hint"] - 1] + rec["turns"][-1]["seconds"]
        return "won" if r["comedian_said_at"] is None or said < r["comedian_said_at"] else "beaten"
    return "nobody" if r["comedian_said_at"] is None else "beaten"


def summary(recs: list[tuple[dict, dict]]) -> dict:
    got = [x["solved_at_hint"] for x, _ in recs if x["solved_at_hint"]]
    v = [verdict(x, r) for x, r in recs]
    return {"rounds": len(recs), "solved": len(got), "avg_hint": round(st.mean(got), 2) if got else None, "solved_on_hint_1": got.count(1),
            "wrong_guesses": sum(x["guesses_used"] - bool(x["solved_at_hint"]) for x, _ in recs), "beat_comedian": v.count("won"), "lost_to_comedian": v.count("beaten")}


def main(model: str, only: list[str] | None, yes: bool) -> None:
    items = rounds(only)
    if not yes and input(f"Play {len(items)} rounds with {model}? This calls a paid API, up to {sum(len(r['hints']) for r in items)} requests. [y/N] ").strip().lower() != "y":
        print("Stopped. Nothing was called.")
        return
    recs = []
    for r in items:
        rec = play(r, model)
        recs.append((rec, r))
        print(f"{r['answer']}: {'hint ' + str(rec['solved_at_hint']) if rec['solved_at_hint'] else 'never'}, {verdict(rec, r)}", flush=True)
    out = {"model": model, **summary(recs)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"yours_guess_{model.replace('/', '_')}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    mine = json.loads((RESULTS / "wants_to_guess.json").read_text())["summary"]
    print(f"For comparison, across all 42: ChatGPT {mine['chatgpt']['solved']}, Claude {mine['claude']['solved']}, comedians {mine['comedians']['solved']}.")
