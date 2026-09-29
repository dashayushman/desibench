# The ground truth

Everything in this folder is the ground truth I created for this benchmark. It is the answer key every score is
checked against, for my run and for yours.

It might have some flaws. I did my best. If you find a mistake, please open an issue or a pull request with the fix,
and the scores will follow it.

| File | What it is |
|---|---|
| `transcripts/<clip>.json` | 24 one-minute reference transcripts: what was actually said, line by line, with who said it. Used for words wrong and who's talking |
| `clips.json` | Where each of those minutes sits in its episode |
| `<episode>/facts.json` | The show's real facts for that episode: the rounds, the final scores, the winner |
| `<episode>/context.json` | The host and the three comedians, given the same way to both AIs |
| `<episode>/youtube_markers.json` | YouTube's most-replayed curve and chapters for the episode, saved from its page |
| `quiz.json` | The questions the show actually asked, with the answers confirmed on screen and by the host |
| `who_am_i.json` | The show's Who Am I rounds: the hints as Gursimran read them, the answer, and the name parts that count as right |

The two episodes: `_3UvCy7FMTg` (Video 1, 75 minutes) and `2pxmtBus3f8` (Video 2, the finale, 90 minutes).
