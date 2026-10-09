// Depends on js/config.js (TR namespace)

// Replay — a game's sets laid out on its clock, and where the ball is at any
// moment of it. Pure functions over parsed events; replay.html draws them.
//
// An event here is { t, type, name, owner, actor, x, y }: t in game seconds,
// owner the team in possession ('Team 1' / 'Team 2'), actor who did it, and
// x / y the tagged position (0-100, attack-normalised — y 100 is the try line
// the team in possession is attacking), or null when untagged.
TR.Replay = (() => {

  // "12:34" or "1:02:03" → seconds.
  function toSecs(s) {
    const p = String(s || '').trim().split(':').map(Number);
    if (p.some(n => !Number.isFinite(n))) return null;
    return p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2] : p.length === 2 ? p[0] * 60 + p[1] : null;
  }

  // How a set ended, from its last event.
  function outcomeOf(last) {
    if (!last) return { key: 'none', label: '' };
    if (last.type === 'Try') return { key: 'try', label: 'Try' };
    if (last.type === 'Turnover' && last.name === '6th Touch') return { key: 'sixth', label: '6th touch' };
    if (last.type === 'Turnover') return { key: 'error', label: 'Lost — ' + (last.name || 'error').toLowerCase() };
    if (last.type === 'Penalty Attack') return { key: 'pen', label: 'Penalty against — ' + (last.name || '').toLowerCase() };
    return { key: 'none', label: '' };
  }

  // Sets in game order. A set is one team's run of possession: it starts with
  // the first event after the ball changes hands (or a Game Start) and lasts
  // to its last event. Steps are the events in it that carry a position, the
  // points the ball is drawn through. Sets without one are dropped.
  function buildTimeline(events) {
    const evs = events.filter(e => e.t != null).map((e, i) => Object.assign({ i }, e))
      .sort((a, b) => a.t - b.t || a.i - b.i);
    const sets = [], halves = [], tries = [];
    let cur = null, half = 0;
    for (const e of evs) {
      if (e.type === 'Game Event') {
        if (e.name === 'Game Start') { half++; halves.push({ n: half, start: e.t, end: null }); cur = null; }
        else if (e.name === 'Game End') { if (halves.length) halves[halves.length - 1].end = e.t; cur = null; }
        continue;
      }
      if (!half) continue;                       // before kick-off
      if (e.type === 'Try') tries.push({ t: e.t, team: e.actor });
      if (!cur || cur.owner !== e.owner) {
        cur = { owner: e.owner, half, events: [], steps: [] };
        sets.push(cur);
      }
      cur.events.push(e);
      if (e.x != null && e.y != null) cur.steps.push(e);
    }
    const kept = sets.filter(s => s.steps.length);
    kept.forEach((s, n) => {
      s.n = n;
      s.start = s.steps[0].t;
      s.end = s.steps[s.steps.length - 1].t;
      s.outcome = outcomeOf(s.events[s.events.length - 1]);
    });
    const start = halves.length ? halves[0].start : (kept[0] ? kept[0].start : 0);
    const lastHalf = halves[halves.length - 1];
    const end = Math.max(lastHalf && lastHalf.end != null ? lastHalf.end : 0, kept.length ? kept[kept.length - 1].end : 0);
    return { sets: kept, halves, tries, start, end };
  }

  // The set being played at time T: the one whose span holds T, else the
  // last one that has started (the ball rests where it ended until the next
  // set starts). -1 before the first.
  function setIndexAt(tl, T) {
    let idx = -1;
    for (let i = 0; i < tl.sets.length; i++) {
      if (tl.sets[i].start <= T) idx = i; else break;
    }
    return idx;
  }

  // Where the ball is in a set at T, gliding between steps by their times.
  // `step` is the index of the last step reached (-1 before the first).
  function ballAt(set, T) {
    const st = set.steps;
    if (T <= st[0].t) return { x: st[0].x, y: st[0].y, step: T < st[0].t ? -1 : 0, done: st.length === 1 && T >= st[0].t };
    const last = st[st.length - 1];
    if (T >= last.t) return { x: last.x, y: last.y, step: st.length - 1, done: true };
    let i = 0;
    while (i < st.length - 1 && st[i + 1].t <= T) i++;
    const a = st[i], b = st[i + 1];
    const f = b.t > a.t ? (T - a.t) / (b.t - a.t) : 1;
    return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, step: i, done: false };
  }

  // The touch count at T: the latest "Touch n" in the set, back to zero after
  // a six-again or a defensive penalty (both restart the count).
  function touchAt(set, T) {
    let n = 0;
    for (const e of set.events) {
      if (e.t > T) break;
      if ((e.type === 'Turnover' && e.name === '6 Again') || e.type === 'Penalty Defence') { n = 0; continue; }
      const m = e.type === 'Touch' && /(\d+)/.exec(e.name || '');
      if (m) n = +m[1];
    }
    return n;
  }

  function scoreAt(tl, T) {
    const s = { 'Team 1': 0, 'Team 2': 0 };
    tl.tries.forEach(tr => { if (tr.t <= T && s[tr.team] != null) s[tr.team]++; });
    return s;
  }

  function halfAt(tl, T) {
    let h = tl.halves.length ? 1 : 0;
    tl.halves.forEach(hv => { if (hv.start <= T) h = hv.n; });
    return h;
  }

  // Dead time isn't worth watching: past a gap longer than `maxGap` seconds
  // (half time, a stoppage, the walk back after a try) the clock jumps to just
  // before the next set. Returns T unchanged when it's in play or a short gap.
  function skipDead(tl, T, maxGap = 4, lead = 1) {
    const i = setIndexAt(tl, T);
    const next = tl.sets[i + 1];
    if (!next) return T;
    const prevEnd = i >= 0 ? tl.sets[i].end : tl.start;
    if (T > prevEnd && next.start - prevEnd > maxGap && T < next.start - lead) return next.start - lead;
    return T;
  }

  // Positions as drawn: Team 1 always attacks up the screen, Team 2 down, so a
  // replay never flips at a change of possession. Positions are recorded from
  // the attacking team's view, so Team 2's are turned through 180°.
  function screenPos(owner, x, y) {
    return owner === 'Team 2' ? { x: 100 - x, y: 100 - y } : { x, y };
  }

  return { toSecs, buildTimeline, setIndexAt, ballAt, touchAt, scoreAt, halfAt, skipDead, screenPos, outcomeOf };
})();
