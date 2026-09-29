# desibench: which AI is more desi?

I made Sarvam and OpenAI watch 165 minutes of an Indian comedy quiz show, **The Nation Wants to Guess** (hosted by
Gursimran Khamba), and do everything a viewer does: write down every word of the Hinglish, know who is talking, hear
the audience laugh, watch the screen, get the jokes and the pop culture, and play along with the quiz. Then I scored
both, round by round, and counted every rupee.

This repo lets you run the whole thing yourself, end to end, on your laptop. You bring two API keys. That's it.

**The story version, with the clips:** [desibench.aiwithayushman.com](https://desibench.aiwithayushman.com)

## My result

| Round | The question | Sarvam | OpenAI | +20 |
|---|---|---|---|---|
| 1 | Who understands the video better? (people on screen counted right) | 51% | 88% | OpenAI |
| 2 | Who hears Hinglish better? (words wrong, lower is better) | **18.7%** | 21.1% | Sarvam |
| 3 | Who hears the audience laugh? (laughs and applause heard) | 1 | **70** | OpenAI |
| 4 | Who knows who's talking? (lines put on the right person) | **64%** | 41% | Sarvam |
| 5 | Who understands Indian pop culture better? (Who Am I points, of 270) | 0 | **90** | OpenAI |
| 6 | Who's cheaper to run? (per hour of video) | $5.50 | **$1.84** | OpenAI |
| | **Final score** | **40** | **80** | |

Sarvam hears Indian speech better, and cheaper. OpenAI understands it better, sees better, and costs a third as much overall.

## The two contestants

Each stack uses one vendor only, and the two language models are priced alike.

| Job | Sarvam | OpenAI |
|---|---|---|
| Writing down every word | Saaras v4 (codemix, hi-IN, tuned for Hinglish) | GPT Transcribe |
| Who's talking | Saaras v4 (speaker labels) | GPT-4o Transcribe Diarize |
| Clean-up, and hearing the room | Sarvam 105B (words only; Sarvam has no model that listens to sound) | GPT Audio Mini |
| Watching the screen | Sarvam Vision | GPT-6 Luna* |
| Making sense of the show, the quiz, Who Am I | Sarvam 105B | GPT-6 Luna* |

*GPT-6 Luna: I picked the GPT-6 variant priced like Sarvam 105B (both under $1 per million tokens). Putting OpenAI's
top model against it would be an unfair fight.

## Run it yourself

You need Python 3.10+, [ffmpeg](https://ffmpeg.org), about 6 GB of disk, a Sarvam API key and an OpenAI API key.

```bash
git clone https://github.com/dashayushman/desibench && cd desibench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your two keys in .env
python -m desibench doctor  # checks everything
python -m desibench all     # download → run → score → open the results page
```

Or one step at a time:

```bash
python -m desibench download   # the two episodes from YouTube → videos/ (about 830 MB)
python -m desibench estimate   # what it will cost and how long it takes
python -m desibench run        # both AIs, both episodes; asks before it spends anything
python -m desibench score      # → results/yours.json
python -m desibench serve      # → http://localhost:8000/viewer/
```

**Cost and time:** a full run costs about **$20** (Sarvam about $15, OpenAI about $5, from my logged costs) and takes
about **4 hours**, almost all of it Sarvam Vision reading frames one at a time. The two AIs run side by side. Try one
episode first with `--episode _3UvCy7FMTg`, or one AI with `--ai openai`. Every step is resumable: run it again and it
picks up where it stopped.

**With a coding agent:** open Claude Code or Codex in this folder and say "set this up and run it for me". The agent
follows [AGENTS.md](AGENTS.md): it asks for your keys, tells you the cost, waits for your go, and hands you the results page.

## The results page

`python -m desibench serve` opens a simple local page:

- **The scoreboard**: all six rounds, Sarvam and OpenAI side by side, my run or yours.
- **Round by round**: the numbers behind each round (every clip's words wrong, voices heard, the bill by step).
- **Watch an episode**: the YouTube video with both transcripts next to each other, following it line by line. Search both.
- **Frame by frame** (your run): every frame the AIs looked at, with what each one wrote.

It shows my run straight away, before you run anything.

## How it's scored

Everything is checked against the ground truth in [`answers/`](answers/README.md), with the same code for my run and yours:

1. **Words wrong**: each AI's speech-to-text (no name hints) against 24 one-minute reference transcripts, 3,130 words.
   Hindi counts the same in Devanagari or Latin letters, and spelling variants don't count as errors.
2. **Who's talking**: every reference line spoken by the host or a comedian, checked against whoever the AI put there.
3. **The room**: laughs, applause and cheers each AI heard.
4. **The screen**: frames described and cost per 1,000 frames. How well each AI described the scene was graded frame by
   frame for my run; those grades are in the results, and your run shows you each AI's descriptions side by side.
5. **Pop culture**: the show's own facts (the winner, final scores, rounds), its quiz (multiple choice is checked
   automatically), and its Who Am I round, played hint by hint.
6. **The bill**: every API call is logged with its cost.

## What's in here

| Path | What |
|---|---|
| `desibench/` | The pipeline: ingest → transcribe → clean up → watch → make sense → export; the quiz and Who Am I; scoring |
| `answers/` | The ground truth, and each episode's cast and YouTube markers |
| `results/reference.json` | My run, scored. `results/reference/` holds my run's episode outputs for the results page |
| `viewer/` | The results page (plain HTML, no build step) |
| `videos/`, `work/` | Created when you run it (git-ignored) |

## Credits

The show is [The Nation Wants to Guess](https://www.youtube.com/@GursimranKhamba) by Gursimran Khamba. The videos are
downloaded from YouTube for your own run and never stored in this repo; the reference transcripts are short excerpts
used for research and commentary. Sarvam and OpenAI names belong to their owners.

Built by [Ayushman Dash](https://aiwithayushman.com). Code under the MIT license.
