// desibench results page. Reads results/reference.json (my run) and results/yours.json (your run, after
// `python -m desibench run`), plus each run's episode files, and shows them side by side. No build step.
const AIS = ['sarvam-tuned', 'openai'];
const NAME = { 'sarvam-tuned': 'Sarvam', openai: 'OpenAI' };
const CLS = { 'sarvam-tuned': 's', openai: 'o' };
const EPS = [['_3UvCy7FMTg', 'Video 1'], ['2pxmtBus3f8', 'Video 2']];
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const mmss = (s) => { s = Math.max(0, Math.floor(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
const pct = (x) => (x == null ? '—' : `${(x * 100).toFixed(1).replace(/\.0$/, '')}%`);
const usd = (x) => (x == null ? '—' : `$${x.toFixed(2)}`);
const getJSON = (u) => fetch(u, { cache: 'no-store' }).then((r) => (r.ok ? r.json() : null)).catch(() => null);

const state = { run: 'reference', R: {}, ep: EPS[1][0], q: '', t: 0 };

// the six rounds, in the site's order; each picks the number that decides it
const ROUNDS = [
  { k: 'eyes', q: 'Who understands the video better?', m: (r, a) => r.graded?.eyes?.[a]?.people_counted_right, f: pct, hi: true,
    label: 'people on screen counted right', alt: (r, a) => r.eyes[a].frames_described, altLabel: 'frames described (grading needs a person)' },
  { k: 'ears', q: 'Who hears Hinglish better?', m: (r, a) => r.ears[a].words_wrong, f: pct, hi: false, label: 'words wrong (lower is better)' },
  { k: 'room', q: 'Who hears the audience laugh?', m: (r, a) => r.room[a].reactions_heard, f: String, hi: true, label: 'laughs and applause heard' },
  { k: 'speakers', q: "Who knows who's talking?", m: (r, a) => r.speakers[a].right_person, f: pct, hi: true, label: 'lines put on the right person' },
  { k: 'show', q: 'Who understands Indian pop culture better?', m: (r, a) => r.show[a].who_am_i_points, f: (v) => (v == null ? '—' : String(v)), hi: true,
    label: 'Who Am I points, of 270' },
  { k: 'bill', q: "Who's cheaper to run?", m: (r, a) => r.bill[a].usd_per_hour, f: usd, hi: false, label: 'dollars per hour of video (lower is better)' },
];

async function boot() {
  state.R.reference = await getJSON('../results/reference.json');
  state.R.yours = await getJSON('../results/yours.json');
  const hasYours = !!(state.R.yours && Object.values(state.R.yours.bill).some((b) => b.usd_total));
  if (!hasYours) state.R.yours = null;
  document.querySelectorAll('[data-run]').forEach((b) => {
    b.disabled = b.dataset.run === 'yours' && !hasYours;
    b.onclick = () => { state.run = b.dataset.run; render(); };
  });
  if (new URLSearchParams(location.search).get('run') === 'yours' && hasYours) state.run = 'yours';
  $('[data-eps]').innerHTML = EPS.map(([v, l]) => `<button data-ep="${v}">${l}</button>`).join('');
  $('[data-eps]').onclick = (e) => { const b = e.target.closest('[data-ep]'); if (b) { state.ep = b.dataset.ep; loadEpisode(); } };
  $('[data-q]').oninput = (e) => { state.q = e.target.value.trim().toLowerCase(); drawTx(); };
  render();
}

function render() {
  const r = state.R[state.run];
  document.querySelectorAll('[data-run]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.run === state.run)));
  $('[data-note]').innerHTML = state.run === 'reference'
    ? (state.R.yours ? 'Showing my run. Switch to <b>Your run</b> to see yours.' : 'Showing my run. Run it yourself with <code>python -m desibench run</code>, then this page shows yours next to mine.')
    : 'Showing your run, scored against the same ground truth. Rounds that need a person to grade (the video descriptions) show what each AI wrote, frame by frame, below.';
  if (!r) return;
  // your run may cover one episode only: open on one it has
  const have = EPS.map(([v]) => v).filter((v) => r.episodes?.[v]?.duration_s);
  if (have.length && !have.includes(state.ep)) state.ep = have[0];
  AIS.forEach((a) => { $(`[data-pts="${a}"]`).textContent = r.points[a]; $(`[data-pts="${a}"]`).parentElement.classList.toggle('win', r.points[a] > r.points[AIS.find((x) => x !== a)]); });
  $('[data-rounds]').innerHTML = ROUNDS.map((R, i) => card(R, r, i)).join('');
  $('[data-details]').innerHTML = details(r);
  loadEpisode();
}

function card(R, r, i) {
  let vals = AIS.map((a) => R.m(r, a)), label = R.label, f = R.f;
  if (vals.every((v) => v == null) && R.alt) { vals = AIS.map((a) => R.alt(r, a)); label = R.altLabel; f = String; }
  const mx = Math.max(...vals.map((v) => +v || 0), 1e-9);
  const w = r.winners[R.k];
  return `<article class="rc"><h3><small>Round ${i + 1}</small>${R.q}</h3>
    <div class="bars">${AIS.map((a, j) => `<div class="bar ${CLS[a]}"><span>${NAME[a]}</span><i><span style="width:${((+vals[j] || 0) / mx) * 100}%"></span></i><b>${f(vals[j])}</b></div>`).join('')}</div>
    <p class="m">${label}</p>
    ${w ? `<span class="win-tag">${NAME[w]} +20</span>` : '<span class="win-tag none">no winner from checkable scores</span>'}</article>`;
}

function table(head, rows) {
  return `<table><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr>${rows.map((row) => `<tr>${row.map((c) => (typeof c === 'object' && c ? `<td class="${c.c || ''}">${c.v}</td>` : `<td>${c}</td>`)).join('')}</tr>`).join('')}</table>`;
}
const both = (fn) => AIS.map((a) => ({ v: fn(a), c: 'n' }));

function details(r) {
  const out = [];
  const e = r.ears;
  out.push(`<details open><summary>Who hears Hinglish better? Words wrong against my reference transcripts</summary>
    <p class="sub">${e['openai'].clips} one-minute clips, ${e['openai'].ref_words} reference words. Hindi counts the same in Devanagari or Latin letters.</p>
    ${table(['', 'Sarvam', 'OpenAI'], [['Words wrong', ...both((a) => pct(e[a].words_wrong))], ['Heard a different word', ...both((a) => e[a].wrong)],
      ['Missed a word', ...both((a) => e[a].missing)], ['Added a word', ...both((a) => e[a].extra)]])}
    ${table(['Clip', 'Sarvam', 'OpenAI'], (e['sarvam-tuned'].per_clip || []).map((c, i) => {
      const o = (e.openai.per_clip || [])[i] || {}; const w = (x) => (x.ref_words ? pct((x.wrong + x.missing + x.extra) / x.ref_words) : '—');
      return [c.clip, { v: w(c), c: 'n' }, { v: w(o), c: 'n' }];
    }))}</details>`);
  const s = r.speakers;
  out.push(`<details><summary>Who knows who's talking?</summary>${table(['', 'Sarvam', 'OpenAI'], [
    ['Lines put on the right person', ...both((a) => pct(s[a].right_person))], ['Lines checked', ...both((a) => s[a].lines_checked)],
    ...EPS.map(([v, l]) => [`Voices heard, ${l} (4 people talk)`, ...both((a) => s[a].voices_heard?.[v] ?? '—')])])}</details>`);
  const rm = r.room;
  out.push(`<details><summary>Who hears the audience laugh?</summary>${table(['', 'Sarvam', 'OpenAI'], [
    ['Laughs, applause and cheers heard', ...both((a) => rm[a].reactions_heard)], ['Laughs heard', ...both((a) => rm[a].laughs_heard)]])}</details>`);
  const ey = r.eyes, g = r.graded?.eyes;
  out.push(`<details><summary>Who understands the video better?</summary>${table(['', 'Sarvam', 'OpenAI'], [
    ['Frames described', ...both((a) => ey[a].frames_described)], ['Cost per 1,000 frames', ...both((a) => usd(ey[a].usd_per_1000_frames))],
    ...(g ? [['People counted right (graded)', ...both((a) => pct(g[a].people_counted_right))], ['Scene described right (graded)', ...both((a) => pct(g[a].scene_described_right))],
      ['Things made up (graded)', ...both((a) => g[a].made_things_up)], ['Named the people right (graded)', ...both((a) => pct(g[a].named_people_right))]] : [])])}
    ${g ? `<p class="sub">Graded frame by frame against my ground truth (${g.frames} frames).</p>` : '<p class="sub">These need a person to grade. Compare what each AI wrote, frame by frame, further down.</p>'}</details>`);
  const sh = r.show, gq = r.graded;
  out.push(`<details><summary>Who understands Indian pop culture better?</summary>${table(['', 'Sarvam (Sarvam 105B)', 'OpenAI (GPT-6 Luna*)'], [
    ['Who Am I points (of 270)', ...both((a) => sh[a].who_am_i_points ?? '—')], ['Who Am I solved', ...both((a) => `${sh[a].who_am_i_solved} of ${sh[a].who_am_i_played}`)],
    ['Quiz, multiple choice right', ...both((a) => (sh[a].quiz_multiple_choice_right == null ? '—' : `${sh[a].quiz_multiple_choice_right} of ${sh[a].quiz_multiple_choice}`))],
    ...(gq ? [['Quiz, all 15 (graded)', ...both((a) => `${gq.quiz_all_15[a].points} of ${gq.quiz_all_15[a].of}`)], ['Slang and references explained right (graded)', ...both((a) => pct(gq.slang[a].score))]] : []),
    ...EPS.map(([v, l]) => [`${l}: winner right, final scores exact`, ...both((a) => { const f = sh[a].facts?.[v]; return f ? `${f.winner_right ? '✓' : '✗'} · ${f.final_scores_exact} of ${f.of}` : '—'; })])])}
    <p class="sub">*GPT-6 Luna: the GPT-6 variant priced like Sarvam 105B (both under $1 per million tokens). Putting OpenAI's top model against it would be an unfair fight.</p>
    ${whoTable(r)}</details>`);
  const b = r.bill;
  const stages = [...new Set(AIS.flatMap((a) => Object.keys(b[a].usd_per_hour_by_stage || {})))];
  const SW = { transcribe: 'Listening (every word, who is talking)', refine: 'Clean-up (and hearing the room)', see: 'Watching the screen', fuse: 'Making sense of the show', export: 'Writing the results' };
  out.push(`<details><summary>Who's cheaper to run?</summary>${table(['Per hour of video', 'Sarvam', 'OpenAI'], [
    ...stages.map((st) => [SW[st] || st, ...both((a) => usd(b[a].usd_per_hour_by_stage?.[st]))]),
    [{ v: '<b>Total</b>' }, ...both((a) => `<b>${usd(b[a].usd_per_hour)}</b> (₹${b[a].inr_per_hour ?? '—'})`)]])}</details>`);
  return out.join('');
}

function whoTable(r) {
  const d = r.details?.who_am_i;
  if (!d) return '';
  const ids = [...new Set(d.map((x) => x.id))];
  const cell = (x) => (!x ? '—' : x.withheld ? `${x.points} pts` : `${x.solved_at_hint ? `✓ at hint ${x.solved_at_hint}` : '✗'} · ${x.points} pts${x.turns?.length ? `<br><small>${x.turns.filter((t) => t.guess).map((t) => esc(t.guess)).join(', ') || 'no guesses'}</small>` : ''}`);
  return table(['Who Am I', 'Sarvam', 'OpenAI'], ids.map((id) => {
    const s = d.find((x) => x.id === id && x.model === 'sarvam-105b'), o = d.find((x) => x.id === id && x.model === 'gpt-6-luna');
    return [esc((s || o).answer), { v: cell(s), c: 'n' }, { v: cell(o), c: 'n' }];
  }));
}

// ---------- the episode: YouTube + both transcripts, following the video
let E = {}, yt = null, ytReady = false;
async function loadEpisode() {
  document.querySelectorAll('[data-ep]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.ep === state.ep)));
  const src = (a) => (state.run === 'yours' ? `../work/${state.ep}/runs/${a}/episode.json` : `../results/reference/${state.ep}/${a}.json`);
  const [s, o] = await Promise.all(AIS.map((a) => getJSON(src(a))));
  E = { 'sarvam-tuned': s, openai: o };
  $('[data-meta-s]').textContent = s ? `${s.transcript.length} lines` : 'not run yet';
  $('[data-meta-o]').textContent = o ? `${o.transcript.length} lines` : 'not run yet';
  drawTx();
  shown = 24; drawFrames();   // the frames follow the episode picker
  if (!yt) initYT(); else if (ytReady) yt.cueVideoById(state.ep);
}
function drawTx() {
  for (const a of AIS) {
    const box = $(`[data-tx-${CLS[a]}]`), ep = E[a];
    if (!ep) { box.innerHTML = '<p class="sub" style="padding:1rem">Nothing yet for this run.</p>'; continue; }
    const q = state.q;
    box.innerHTML = ep.transcript.filter((l) => !q || l.text.toLowerCase().includes(q)).map((l) => {
      const text = q ? esc(l.text).replace(new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), (m) => `<mark>${m}</mark>`) : esc(l.text);
      return `<div class="ln" data-t="${l.start}"><span class="t">${mmss(l.start)}</span><span><span class="sp">${esc(l.speaker)}</span>${text}</span></div>`;
    }).join('');
  }
  follow(state.t, true);
}
function follow(t, force) {
  for (const a of AIS) {
    const box = $(`[data-tx-${CLS[a]}]`); let hit = null;
    for (const l of box.querySelectorAll('.ln')) { if (+l.dataset.t <= t + 0.2) hit = l; else break; }
    const old = box.querySelector('.ln.now');
    if (hit !== old || force) { old?.classList.remove('now'); if (hit) { hit.classList.add('now'); box.scrollTo({ top: hit.offsetTop - box.clientHeight * 0.3, behavior: force ? 'auto' : 'smooth' }); } }
  }
}
document.addEventListener('click', (e) => {
  const l = e.target.closest('.ln'); if (!l) return;
  state.t = +l.dataset.t; if (ytReady) { yt.seekTo(state.t, true); yt.playVideo(); } follow(state.t, true);
});
function initYT() {
  const tag = document.createElement('script'); tag.src = 'https://www.youtube.com/iframe_api'; document.head.appendChild(tag);
  window.onYouTubeIframeAPIReady = () => {
    yt = new YT.Player('yt', { videoId: state.ep, host: 'https://www.youtube-nocookie.com', playerVars: { rel: 0, playsinline: 1 },
      events: { onReady: () => { ytReady = true; setInterval(() => { if ([1, 3].includes(yt.getPlayerState())) { state.t = yt.getCurrentTime(); follow(state.t); } }, 400); } } });
  };
}

// ---------- frame by frame (your run: the frames are on your disk)
let shown = 24;
async function drawFrames() {
  const box = $('[data-frames]');
  if (state.run !== 'yours') { box.innerHTML = '<p class="sub">Switch to <b>Your run</b> after running it: every frame the AIs looked at, with what each one wrote next to it.</p>'; return; }
  const [s, o] = await Promise.all(AIS.map((a) => getJSON(`../work/${state.ep}/runs/${a}/frames.json`)));
  if (!s && !o) { box.innerHTML = '<p class="sub">No frames yet for this episode.</p>'; return; }
  const by = (f) => new Map((f?.tiles || []).map((t) => [Math.round(t.t), t]));
  const S = by(s), O = by(o), ts = [...new Set([...S.keys(), ...O.keys()])].sort((a, b) => a - b);
  const d = (t) => (t ? `${esc(t.desc || '')}${t.visible ? `<br><small>on screen: ${esc(Array.isArray(t.visible) ? t.visible.join(', ') : t.visible)}</small>` : ''}${t.text ? `<br><small>text: ${esc(t.text)}</small>` : ''}` : '<small>—</small>');
  box.innerHTML = ts.slice(0, shown).map((t) => `<div class="fr"><div><img loading="lazy" src="../work/${state.ep}/frames/frame_${String(Math.round(t / 5) + 1).padStart(5, '0')}.jpg" alt=""><span class="t">${mmss(t)}</span></div>
    <div class="d s"><b>Sarvam</b><br>${d(S.get(t))}</div><div class="d o"><b>OpenAI</b><br>${d(O.get(t))}</div></div>`).join('')
    + (ts.length > shown ? `<button class="more" data-more>Show more (${ts.length - shown} left)</button>` : '');
  box.querySelector('[data-more]')?.addEventListener('click', () => { shown += 48; drawFrames(); });
}

boot();
