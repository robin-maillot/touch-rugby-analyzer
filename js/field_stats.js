// Depends on js/config.js (TR namespace)

// ══════════════════════════════════════════════════════════════════
// Field stats — what the pitch positions are for
//
// Shared by the field annotator's live Stats sheet and Game Analysis's Field
// view, so the two always draw the same numbers the same way.
//
// Everything is derived from attack-normalised coordinates, so both teams are
// measured in the same frame however they were pointing: y = 0 is a team's own
// try line and y = 100 the one they're attacking, over 70m; x runs 0–100 from
// that team's left touchline.
//
// Events are plain objects: { type, name, possessionOwner, actionOwner, x, y }
// with x/y null when the event has no position. Owners are 'Team 1' / 'Team 2'.
// ══════════════════════════════════════════════════════════════════

TR.FieldStats = (() => {
  const Y_TO_M   = 0.7;                       // one y unit, in metres
  const RED_ZONE = 100 - (10 / 70 * 100);     // inside the opposition 10m line
  const MAP_LEN  = 140;                       // drawing units, try line to try line (70m × 2)

  const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  // One entry per possession, with the positions that possession covered.
  // Possessions with nothing positioned (a v1 game, or a set tagged entirely from
  // the buttons) are dropped rather than counted as zero-metre sets.
  function possessionSets(events) {
    const sets = [];
    let cur = null;
    for (const a of events) {
      if (a.type === 'To Review') continue;
      if (a.type === 'Game Event') {
        // A half boundary closes whatever was open; Ball Live belongs to the set.
        // A Set Break (see eventsStartingIn) closes it too.
        if (a.name === 'Game Start' || a.name === 'Game End' || a.name === SET_BREAK) { cur = null; }
        continue;
      }
      if (!cur || cur.owner !== a.possessionOwner) {
        cur = { owner: a.possessionOwner, touches: 0, pts: [], endType: null, endName: null };
        sets.push(cur);
      }
      if (a.type === 'Touch') cur.touches++;
      if (a.x != null) cur.pts.push({ x: a.x, y: a.y });
      cur.endType = a.type;
      cur.endName = a.name;
    }
    return sets.filter(s => s.pts.length).map(s => {
      const ys = s.pts.map(p => p.y);
      return Object.assign(s, {
        startY: ys[0],
        endY:   ys[ys.length - 1],
        maxY:   Math.max(...ys),
        gain:   ys[ys.length - 1] - ys[0],
      });
    });
  }

  function computeFieldStats(events) {
    const sets    = possessionSets(events);
    const touches = events.filter(a => a.type === 'Touch' && a.x != null);

    const forTeam = key => {
      const mine = sets.filter(s => s.owner === key);
      const myTouches = touches.filter(a => a.possessionOwner === key);
      // "Red zone" = sets that got inside the opposition 10m. Converting those is
      // a different question from scoring at all, and the more useful one: it
      // separates getting there from finishing.
      const red      = mine.filter(s => s.maxY >= RED_ZONE);
      const redTries = red.filter(s => s.endType === 'Try');
      const chan = [0, 0, 0];
      myTouches.forEach(a => chan[a.x < 33.3 ? 0 : a.x < 66.7 ? 1 : 2]++);
      const chanTotal = myTouches.length || 1;
      return {
        sets:        mine.length,
        startM:      mean(mine.map(s => s.startY)) * Y_TO_M,
        gainM:       mean(mine.map(s => s.gain))   * Y_TO_M,
        endM:        mean(mine.map(s => s.endY))   * Y_TO_M,
        perTouchM:   myTouches.length ? mean(mine.map(s => s.gain)) * Y_TO_M * mine.length / myTouches.length : 0,
        touchesPer:  mean(mine.map(s => s.touches)),
        territory:   myTouches.length ? myTouches.filter(a => a.y > 50).length / myTouches.length * 100 : 0,
        redSets:     red.length,
        redPct:      mine.length ? red.length / mine.length * 100 : 0,
        redTries:    redTries.length,
        redConvPct:  red.length ? redTries.length / red.length * 100 : null,
        channels:    chan.map(c => Math.round(c / chanTotal * 100)),
        touchPts:    myTouches.map(a => ({ x: a.x, y: a.y })),
        // The third and fourth touch are where a set is decided — the shape is
        // set by then and the break comes off one of them — so those are the two
        // the map plots, rather than every touch at once.
        t3Pts:       myTouches.filter(a => a.name === 'Touch 3').map(a => ({ x: a.x, y: a.y })),
        t4Pts:       myTouches.filter(a => a.name === 'Touch 4').map(a => ({ x: a.x, y: a.y })),
        t3M:         mean(myTouches.filter(a => a.name === 'Touch 3').map(a => a.y)) * Y_TO_M,
        t4M:         mean(myTouches.filter(a => a.name === 'Touch 4').map(a => a.y)) * Y_TO_M,
        tryPts:      events.filter(a => a.type === 'Try' && a.actionOwner === key && a.x != null).map(a => ({ x: a.x, y: a.y })),
        lostPts:     events.filter(a => a.x != null && a.possessionOwner === key &&
                       (a.type === 'Turnover' || a.type === 'Penalty Attack')).map(a => ({ x: a.x, y: a.y })),
      };
    };
    return { t1: forTeam('Team 1'), t2: forTeam('Team 2'), any: sets.length > 0, sets };
  }

  // A small attack-normalised pitch per team, plotting the two touches that decide
  // a set — the third and the fourth — plus where the ball was lost and where they
  // scored, and a line at the average position of each of those two touches. Every
  // touch at once was just a cloud; two touches and two lines is a read.
  //
  // Both teams attack upwards here whichever ends they actually played, because
  // the whole point of the map is to compare them. Touch 3 and touch 4 are told
  // apart by fill and by dash, never by hue, so the team colour stays free to
  // mean the team.
  // Touch and try markers sit at 0.8 opacity, so where several land on the
  // same spot the overlap shows as a deeper mark instead of one hiding another.
  const POINT_OPACITY = 0.8;

  function fieldMapSVG(stat, color, name) {
    const vy = y => (100 - y) / 100 * MAP_LEN;
    const t3 = stat.t3Pts.map(p =>
      `<circle cx="${p.x}" cy="${vy(p.y)}" r="2.6" fill="none" stroke="${color}" stroke-width="1.1" opacity="${POINT_OPACITY}"/>`).join('');
    const t4 = stat.t4Pts.map(p =>
      `<circle cx="${p.x}" cy="${vy(p.y)}" r="2.6" fill="${color}" opacity="${POINT_OPACITY}"/>`).join('');
    const lost = stat.lostPts.map(p => {
      const d = 2.4;
      return `<path d="M${p.x - d} ${vy(p.y) - d}L${p.x + d} ${vy(p.y) + d}M${p.x + d} ${vy(p.y) - d}L${p.x - d} ${vy(p.y) + d}"
               stroke="var(--orange)" stroke-width="1.1" fill="none"/>`;
    }).join('');
    const tries = stat.tryPts.map(p => `<circle cx="${p.x}" cy="${vy(p.y)}" r="3.4" fill="var(--green)" opacity="${POINT_OPACITY}"/>`).join('');

    const avgLine = (m, n, dash) => !m ? '' :
      `<line x1="2" y1="${vy(m / Y_TO_M)}" x2="90" y2="${vy(m / Y_TO_M)}" stroke="${color}"
             stroke-width="1.2"${dash ? ' stroke-dasharray="4 3"' : ''}/>` +
      `<text x="93" y="${vy(m / Y_TO_M) + 2.6}" fill="${color}" font-size="6" font-weight="800">${n}</text>`;

    const foot = stat.t3M || stat.t4M
      ? `T3 ${stat.t3M.toFixed(0)}m · T4 ${stat.t4M.toFixed(0)}m`
      : `${stat.sets} set${stat.sets === 1 ? '' : 's'}`;

    return `<div class="fmap">
      <div class="fmap-name"><i style="background:${color}"></i>${name}</div>
      <svg viewBox="-3 -3 106 ${MAP_LEN + 6}" preserveAspectRatio="xMidYMid meet">
        <rect x="0" y="0" width="100" height="${MAP_LEN}" rx="1.5" fill="#12301f" stroke="rgba(255,255,255,0.14)"/>
        <line x1="0" y1="${vy(100)}" x2="100" y2="${vy(100)}" stroke="rgba(255,255,255,0.6)" stroke-width="1.4"/>
        <line x1="0" y1="${vy(50)}"  x2="100" y2="${vy(50)}"  stroke="rgba(255,255,255,0.22)" stroke-dasharray="3 3"/>
        <line x1="0" y1="${vy(RED_ZONE)}" x2="100" y2="${vy(RED_ZONE)}" stroke="rgba(255,255,255,0.14)" stroke-dasharray="2 3"/>
        ${avgLine(stat.t3M, '3', true)}${avgLine(stat.t4M, '4', false)}
        ${t3}${t4}${lost}${tries}
      </svg>
      <div class="fmap-foot">${stat.sets} set${stat.sets === 1 ? '' : 's'} · ${foot}</div>
    </div>`;
  }

  // The key under the pair of maps.
  const fieldMapKey = () => `<div class="fmap-key">
      <span><i class="k-t3"></i>touch 3</span>
      <span><i class="k-t4"></i>touch 4</span>
      <span><i class="k-x"></i>ball lost</span>
      <span><i class="k-try"></i>try</span>
      <span><i class="k-avg"></i>avg T3 / T4</span>
    </div>`;

  // ── Charts ─────────────────────────────────────────────────────
  // Inline SVG and flexbox, no libraries — the annotator has to work on a pitch
  // with no signal. The outcome palette was validated for colour-vision
  // separation: everything passes except green↔orange, which sits in the 6–8 ΔE
  // band for deuteranopia. That's why every segment also carries a 2px gap, a
  // legend entry and a direct label — identity is never left to hue alone.
  const OUTCOMES = [
    { key: 'Try',       label: 'Try',       color: '#16a34a' },
    { key: 'Turnover',  label: 'Turnover',  color: '#ea580c' },
    { key: '6th Touch', label: '6th touch', color: '#3b82f6' },
    { key: 'Penalty',   label: 'Penalty',   color: '#ef4444' },
  ];

  function outcomeOf(set) {
    if (set.endType === 'Try')      return 'Try';
    if (set.endType === 'Penalty Attack' || set.endType === 'Penalty Defence') return 'Penalty';
    if (set.endType === 'Turnover') return set.endName === '6th Touch' ? '6th Touch' : 'Turnover';
    return null;
  }

  // A 100%-wide bar of one team's set outcomes. Segments are gapped and labelled,
  // so the split survives both a colour-vision difference and a black-and-white
  // print of the screen.
  function outcomeBar(sets, name, color) {
    const counts = {};
    sets.forEach(s => { const o = outcomeOf(s); if (o) counts[o] = (counts[o] || 0) + 1; });
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    if (!total) return '';
    const segs = OUTCOMES.filter(o => counts[o.key]).map(o => {
      const pct = counts[o.key] / total * 100;
      return `<div class="cseg" style="width:${pct}%;background:${o.color}"
                title="${name} — ${counts[o.key]} of ${total} sets ended in a ${o.label.toLowerCase()}">` +
             `${pct >= 13 ? counts[o.key] : ''}</div>`;
    }).join('');
    return `<div class="cfig-row">
      <div class="cfig-name"><i style="background:${color}"></i>${name}</div>
      <div class="cbar">${segs}</div>
      <div class="cfig-val">${total}</div>
    </div>`;
  }

  // One team's touches split left / middle / right. A single hue with gaps and
  // direct labels: the three parts are one team's whole, not three identities, so
  // giving them three colours would say something untrue.
  function channelBar(stat, name, color) {
    if (!stat.touchPts.length) return '';
    const names = ['Left', 'Middle', 'Right'];
    const segs = stat.channels.map((pct, i) =>
      `<div class="cseg" style="width:${pct}%;background:${color};opacity:${[0.68, 1, 0.68][i]}"
         title="${name} — ${pct}% of touches down the ${names[i].toLowerCase()}">${pct >= 13 ? pct + '%' : ''}</div>`
    ).join('');
    return `<div class="cfig-row">
      <div class="cfig-name"><i style="background:${color}"></i>${name}</div>
      <div class="cbar">${segs}</div>
      <div class="cfig-val">${stat.touchPts.length}</div>
    </div>`;
  }

  // Every set in the order it happened, drawn as a bar from where it started to
  // where it ended, in metres up that team's own half of the field. Bars that rise
  // are territory won; bars that fall gave ground away. The dot marks a try.
  function territorySVG(sets, names, colors) {
    if (sets.length < 2) return '';
    const W = 320, H = 132, L = 26, R = 6, T = 10, B = 18;
    const plotW = W - L - R, plotH = H - T - B;
    const yOf = m => T + plotH - (Math.max(0, Math.min(70, m)) / 70) * plotH;
    const step = plotW / sets.length;
    const bw   = Math.max(2.5, Math.min(9, step - 2.5));

    const grid = [[0, '0'], [35, '35m'], [70, '70m']].map(([m, lab]) =>
      `<line x1="${L}" y1="${yOf(m)}" x2="${W - R}" y2="${yOf(m)}" class="cgrid"/>` +
      `<text x="${L - 5}" y="${yOf(m) + 3}" class="cax" text-anchor="end">${lab}</text>`).join('') +
      `<line x1="${L}" y1="${yOf(60)}" x2="${W - R}" y2="${yOf(60)}" class="cgrid cgrid-10"/>` +
      `<text x="${W - R}" y="${yOf(60) - 3}" class="cax" text-anchor="end">opp. 10m</text>`;

    const bars = sets.map((s, i) => {
      const x  = L + step * i + (step - bw) / 2;
      const y1 = yOf(s.startY * Y_TO_M), y2 = yOf(s.endY * Y_TO_M);
      const c  = colors[s.owner];
      const top = Math.min(y1, y2), h = Math.max(2, Math.abs(y2 - y1));
      const tryDot = outcomeOf(s) === 'Try' ? `<circle cx="${x + bw / 2}" cy="${yOf(70) + 1}" r="2.6" fill="#16a34a"/>` : '';
      return `<g><title>${names[s.owner]} — set ${i + 1}: ${(s.startY * Y_TO_M).toFixed(0)}m → ` +
        `${(s.endY * Y_TO_M).toFixed(0)}m${outcomeOf(s) ? ', ' + outcomeOf(s).toLowerCase() : ''}</title>` +
        `<rect x="${x}" y="${top}" width="${bw}" height="${h}" rx="${Math.min(2, bw / 2)}" fill="${c}" fill-opacity="0.85"/>` +
        tryDot + `</g>`;
    }).join('');

    return `<svg class="cchart" viewBox="0 0 ${W} ${H}" role="img"
        aria-label="Field position reached by each set, in order">
      ${grid}${bars}
      <text x="${L}" y="${H - 4}" class="cax">first set</text>
      <text x="${W - R}" y="${H - 4}" class="cax" text-anchor="end">last set</text>
    </svg>`;
  }

  function chartLegend(items) {
    return `<div class="cleg">` + items.map(i =>
      `<span><i style="background:${i.color}${i.opacity ? ';opacity:' + i.opacity : ''}"></i>${i.label}</span>`).join('') + `</div>`;
  }

  // ══════════════════════════════════════════════════════════════
  // Possessions as paths — where each set started, every touch it went
  // through, and how it ended. Used by Game Analysis's Possessions panel.
  // ══════════════════════════════════════════════════════════════

  // Where the next set starts after each kind of ending. A turnover, a 6th
  // touch or a penalty hands the ball over on the spot, so the new set starts
  // there — seen from the other end, hence the mirror. After a try (and at
  // kick-off) play restarts with a tap on halfway.
  const handsOverOnTheSpot = e => e.type === 'Turnover' || e.type === 'Penalty Attack';

  // One path per possession, in order: { owner, half, steps, end, outcome }.
  // steps[] is { k, x, y, type, name } where k is 'start' (where the ball was
  // won or tapped), a touch number, or 'end'; a set with nothing positioned
  // is left out.
  function possessionPaths(events) {
    const sets = [];
    let cur = null, half = 0, last = null;     // last: the set that just ended
    for (let ei = 0; ei < events.length; ei++) {
      const a = events[ei];
      if (a.type === 'To Review') continue;
      if (a.type === 'Game Event') {
        if (a.name === 'Game Start') { half++; cur = null; last = { restart: true }; }
        else if (a.name === 'Game End' || a.name === SET_BREAK) { cur = null; last = null; }
        continue;
      }
      if (!cur || cur.owner !== a.possessionOwner) {
        cur = { owner: a.possessionOwner, half: half || 1, steps: [], end: null, idx: [] };
        if (last && last.restart) {
          cur.steps.push({ k: 'start', how: 'tap', x: 50, y: 50 });
        } else if (last && last.end && handsOverOnTheSpot(last.end) && last.lastPt) {
          cur.steps.push({ k: 'start', how: 'won', x: 100 - last.lastPt.x, y: 100 - last.lastPt.y });
        }
        sets.push(cur);
      }
      cur.idx.push(ei);
      if (a.x != null) {
        const n = a.type === 'Touch' ? parseInt(String(a.name).replace(/\D+/g, ''), 10) : NaN;
        cur.steps.push({ k: a.type === 'Touch' && n ? n : 'end', type: a.type, name: a.name, x: a.x, y: a.y });
        cur.lastPt = { x: a.x, y: a.y };
      }
      cur.end = { type: a.type, name: a.name };
      // A defensive penalty or a 6 Again keeps the ball with the same team, so
      // the set carries on; anything else that changes hands closes it.
      last = a.type === 'Try' ? { restart: true } : cur;
    }
    // Only the last step ends the set. A positioned event before it — a
    // defensive penalty, a 6 Again — was the set carrying on with a fresh
    // count, so it's named for what it was rather than read as an ending.
    sets.forEach(s => s.steps.forEach((p, i) => {
      if (p.k === 'end' && i < s.steps.length - 1) p.k = p.type === 'Penalty Defence' ? 'pen' : p.name === '6 Again' ? 'again' : 'event';
    }));
    return sets
      .filter(s => s.steps.some(p => p.k !== 'start'))
      .map(s => Object.assign(s, { outcome: outcomeOf({ endType: s.end.type, endName: s.end.name }) || 'Other' }));
  }

  // ── Where a set started ───────────────────────────────────────
  // Split at the two 10m lines either side of halfway: before your own 10m
  // (0–25m from your try line), between the 10m lines (25–45m), or past the
  // opposition's 10m (45–70m). A set starts where it was won, or on halfway
  // after a try, so a tap-off always counts as the middle.
  const SET_BREAK = 'Set Break';      // a fence between sets, never shown or stored
  const START_ZONES = [
    { key: 'own', label: 'Own end',  sub: '0–25m',  lo: 0,  hi: 25 },
    { key: 'mid', label: 'Middle',   sub: '25–45m', lo: 25, hi: 45 },
    { key: 'opp', label: 'Opp end',  sub: '45–70m', lo: 45, hi: 71 },
  ];
  function startZone(set) {
    const m = set.steps[0].y * Y_TO_M;
    return (START_ZONES.find(z => m >= z.lo && m < z.hi) || START_ZONES[2]).key;
  }

  // The events of only those sets that started in `zone`, plus every game
  // event, so halves and kick-offs still read the same. 'all' changes nothing.
  // Possessions with nothing positioned have no start and are left out once a
  // zone is picked — there's no way to place them.
  //
  // Dropping the sets in between would leave two of one team's sets side by
  // side, and anything grouping by possession would read them as one; so each
  // kept set is fenced off with a Set Break, which the groupers here honour.
  function eventsStartingIn(events, zone) {
    if (!zone || zone === 'all') return events;
    const setOf = new Map();
    possessionPaths(events).forEach((s, si) => { if (startZone(s) === zone) s.idx.forEach(i => setOf.set(i, si)); });
    const out = [];
    let lastSet = null;
    events.forEach((a, i) => {
      if (a.type === 'Game Event') { out.push(a); if (a.name === 'Game Start' || a.name === 'Game End') lastSet = null; return; }
      if (!setOf.has(i)) return;
      const si = setOf.get(i);
      if (lastSet != null && si !== lastSet) out.push({ type: 'Game Event', name: SET_BREAK });
      out.push(a);
      lastSet = si;
    });
    return out;
  }

  // Metres gained on each step of a set, in order: [{ from, to, label, m }].
  function pathGains(set) {
    const lab = k => typeof k === 'number' ? 'T' + k : ({ start: 'start', end: 'end', pen: 'Pen', again: '6A', event: '·' })[k] || k;
    return set.steps.slice(1).map((p, i) => {
      const q = set.steps[i];
      return { from: q.k, to: p.k, label: `${lab(q.k)}→${lab(p.k)}`, m: (p.y - q.y) * Y_TO_M, dx: (p.x - q.x) * 0.5 };
    });
  }

  // The steps B compares across sets: into touch 1 from wherever the set
  // started, touch to touch up to 5, and from the last touch to the end.
  const GAIN_BUCKETS = ['→T1', 'T1→T2', 'T2→T3', 'T3→T4', 'T4→T5', 'T5→end'];
  function gainBuckets(sets) {
    const out = GAIN_BUCKETS.map(label => ({ label, m: [], dx: [] }));
    sets.forEach(s => pathGains(s).forEach(g => {
      let i = -1;
      if (typeof g.to === 'number' && g.to >= 1 && g.to <= 5) i = g.from === 'start' || g.to === 1 ? 0 : g.to - 1;
      else if (g.to === 'end' && typeof g.from === 'number' && g.from >= 5) i = 5;
      if (i < 0) return;
      // Only consecutive touches count, so a 6 Again restart can't read as T4→T1.
      if (i > 0 && i < 5 && g.from !== g.to - 1) return;
      out[i].m.push(g.m); out[i].dx.push(g.dx);
    }));
    return out.map(b => ({ label: b.label, n: b.m.length, mean: mean(b.m), drift: mean(b.dx), values: b.m }));
  }

  // The typical set: the average position of each touch, 1 to 5, over the sets
  // that reached it (at least 3, or the average would be one set's position).
  function typicalSet(sets) {
    return [1, 2, 3, 4, 5].map(n => {
      const pts = sets.map(s => s.steps.find(p => p.k === n)).filter(Boolean);
      return pts.length >= 3 ? { k: n, x: mean(pts.map(p => p.x)), y: mean(pts.map(p => p.y)), n: pts.length } : null;
    }).filter(Boolean);
  }

  // ── Drawing ────────────────────────────────────────────────────
  const OUTCOME_STYLE = {
    'Try':       { color: '#22c55e', label: 'Try' },
    'Turnover':  { color: '#f97316', label: 'Ball lost' },
    '6th Touch': { color: '#93c5fd', label: '6th touch' },
    'Penalty':   { color: '#ef4444', label: 'Penalty' },
    'Other':     { color: '#9ba6b9', label: 'Ended' },
  };
  function endMarker(outcome, x, y, r) {
    const c = (OUTCOME_STYLE[outcome] || OUTCOME_STYLE.Other).color;
    if (outcome === 'Try')       return `<circle cx="${x}" cy="${y}" r="${r + 0.6}" fill="${c}"/>`;
    if (outcome === 'Turnover')  return `<path d="M${x - r} ${y - r}L${x + r} ${y + r}M${x + r} ${y - r}L${x - r} ${y + r}" stroke="${c}" stroke-width="1.4"/>`;
    if (outcome === '6th Touch') return `<path d="M${x} ${y - r}L${x + r} ${y + r}L${x - r} ${y + r}Z" fill="${c}"/>`;
    if (outcome === 'Penalty')   return `<rect x="${x - r * 0.8}" y="${y - r * 0.8}" width="${r * 1.6}" height="${r * 1.6}" fill="${c}"/>`;
    return `<circle cx="${x}" cy="${y}" r="${r * 0.7}" fill="${c}"/>`;
  }
  const pitchBase = vy => `<rect x="0" y="0" width="100" height="${MAP_LEN}" rx="1.5" fill="#12301f" stroke="rgba(255,255,255,0.14)"/>
    <line x1="0" y1="${vy(100)}" x2="100" y2="${vy(100)}" stroke="rgba(255,255,255,0.6)" stroke-width="1.4"/>
    <line x1="0" y1="${vy(50)}"  x2="100" y2="${vy(50)}"  stroke="rgba(255,255,255,0.22)" stroke-dasharray="3 3"/>
    <line x1="0" y1="${vy(RED_ZONE)}" x2="100" y2="${vy(RED_ZONE)}" stroke="rgba(255,255,255,0.14)" stroke-dasharray="2 3"/>`;

  // A metres label beside a step, pushed off the line along its normal so it
  // never sits on the touch numbers at either end.
  function gainLabel(a, b, m, vy) {
    const ax = a.x, ay = vy(a.y), bx = b.x, by = vy(b.y);
    const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1;
    let nx = -dy / len, ny = dx / len;
    const mx = (ax + bx) / 2, my = (ay + by) / 2;
    if ((mx + nx) > 50 === mx > 50) { nx = -nx; ny = -ny; }     // lean towards the middle of the pitch
    const off = len < 10 ? 9 : 7;
    return `<text x="${(mx + nx * off).toFixed(1)}" y="${(my + ny * off + 1.7).toFixed(1)}" text-anchor="middle" font-size="5" font-weight="800"
      fill="${m >= 0 ? '#bbf7d0' : '#fecaca'}" stroke="#0b1a12" stroke-width="1.8" paint-order="stroke">${m >= 0 ? '+' : ''}${m.toFixed(0)}m</text>`;
  }
  const node = (p, color, label) => `<circle cx="${p.x}" cy="${p.vy}" r="3.4" fill="${color}" stroke="#fff" stroke-width="0.8"/>` +
    `<text x="${p.x}" y="${p.vy + 1.6}" text-anchor="middle" font-size="4.3" font-weight="800" fill="#fff">${label}</text>`;

  // A — every set faint, ◯ where it started, its ending as a marker that can be
  // clicked (data-set = its index). With `pick`, that set is drawn on top with
  // numbered touches and its metres; without, the typical set is.
  function pathsSVG(sets, color, pick) {
    const vy = y => (100 - y) / 100 * MAP_LEN;
    const d  = s => s.steps.map((p, i) => `${i ? 'L' : 'M'}${p.x} ${vy(p.y)}`).join(' ');
    const picked = pick != null ? sets[pick] : null;
    let g = pitchBase(vy);
    sets.forEach((s, i) => {
      const dim = picked && i !== pick, a = s.steps[0];
      g += `<g opacity="${dim ? 0.3 : 1}"><path d="${d(s)}" fill="none" stroke="${color}" stroke-width="0.8" stroke-opacity="${picked ? 0.14 : 0.22}" stroke-linejoin="round"/>` +
           `<circle cx="${a.x}" cy="${vy(a.y)}" r="1.5" fill="none" stroke="#fff" stroke-opacity="0.5" stroke-width="0.6"/></g>`;
    });
    const hero = picked ? picked.steps : typicalSet(sets);
    if (hero.length > 1 || picked) {
      const hd = hero.map((p, i) => `${i ? 'L' : 'M'}${p.x} ${vy(p.y)}`).join(' ');
      g += `<path d="${hd}" fill="none" stroke="#fff" stroke-width="2.8" stroke-opacity="0.95" stroke-linejoin="round"/>` +
           `<path d="${hd}" fill="none" stroke="${color}" stroke-width="1.7" stroke-linejoin="round"${picked ? '' : ' stroke-dasharray="5 2"'}/>`;
      hero.forEach((p, i) => { if (i) g += gainLabel(hero[i - 1], p, (p.y - hero[i - 1].y) * Y_TO_M, vy); });
      hero.forEach(p => {
        if (p.k === 'pen' || p.k === 'again') g += node({ x: p.x, vy: vy(p.y) }, '#475569', p.k === 'pen' ? 'P' : '6A');
        if (typeof p.k === 'number') g += node({ x: p.x, vy: vy(p.y) }, color, p.k);
        else if (p.k === 'start') g += `<circle cx="${p.x}" cy="${vy(p.y)}" r="2.6" fill="#0b1a12" stroke="#fff" stroke-width="1"/>`;
      });
    }
    sets.forEach((s, i) => {
      const z = s.steps[s.steps.length - 1], dim = picked && i !== pick;
      g += `<g class="pp-end" data-set="${i}" style="cursor:pointer" opacity="${dim ? 0.4 : 1}">` +
           `<circle cx="${z.x}" cy="${vy(z.y)}" r="5.5" fill="transparent"/>${endMarker(s.outcome, z.x, vy(z.y), i === pick ? 3.4 : 2.4)}</g>`;
    });
    return `<svg viewBox="-4 -4 108 ${MAP_LEN + 8}" preserveAspectRatio="xMidYMid meet" class="pp-pitch">${g}</svg>`;
  }

  // B — average metres gained on each step, one bar per team, every set as a dot.
  function gainChartSVG(teams) {
    const W = 470, H = 172, L = 34, R = 8, T = 16, B = 36, lo = -15, hi = 35;
    const gw = (W - L - R) / GAIN_BUCKETS.length;
    const yOf = v => T + (H - T - B) * (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo));
    let g = [-10, 0, 10, 20, 30].map(v =>
      `<line x1="${L}" y1="${yOf(v)}" x2="${W - R}" y2="${yOf(v)}" stroke="rgba(255,255,255,${v === 0 ? 0.35 : 0.08})"/>` +
      `<text x="${L - 5}" y="${yOf(v) + 3}" text-anchor="end" font-size="8" fill="#6c7689">${v}m</text>`).join('');
    teams.forEach((t, ti) => {
      gainBuckets(t.sets).forEach((b, i) => {
        if (!b.n) return;
        const bw = gw * 0.3, x0 = L + gw * i + gw * 0.5 + (ti ? bw * 0.15 : -bw * 1.15);
        g += `<g><title>${t.name} — ${b.label}: ${b.mean >= 0 ? '+' : ''}${b.mean.toFixed(1)}m on average over ${b.n} set${b.n === 1 ? '' : 's'}</title>` +
             `<rect x="${x0}" y="${Math.min(yOf(b.mean), yOf(0))}" width="${bw}" height="${Math.abs(yOf(b.mean) - yOf(0))}" rx="2" fill="${t.color}" fill-opacity="0.85"/></g>`;
        b.values.forEach((v, k) => { g += `<circle cx="${x0 + bw / 2 + ((k * 37) % 9 - 4) * 1.1}" cy="${yOf(v)}" r="1.4" fill="#fff" fill-opacity="0.45"/>`; });
        g += `<text x="${x0 + bw / 2}" y="${yOf(Math.max(b.mean, 0)) - 4}" text-anchor="middle" font-size="8" font-weight="800" fill="${t.color}">${b.mean >= 0 ? '+' : ''}${b.mean.toFixed(0)}</text>`;
        const arrow = Math.abs(b.drift) < 1.5 ? '↑' : b.drift < 0 ? '↖' : '↗';
        g += `<text x="${x0 + bw / 2}" y="${H - B + 26}" text-anchor="middle" font-size="7" fill="${t.color}" fill-opacity="0.85">${arrow}${Math.abs(b.drift).toFixed(0)}m</text>`;
      });
    });
    GAIN_BUCKETS.forEach((lab, i) => { g += `<text x="${L + gw * i + gw / 2}" y="${H - B + 13}" text-anchor="middle" font-size="8.5" font-weight="700" fill="#9ba6b9">${lab}</text>`; });
    return `<svg viewBox="0 0 ${W} ${H}" class="pp-chart" role="img" aria-label="Average metres gained on each touch">${g}</svg>`;
  }

  // C — every set as a column, in order: from where it started up to its
  // furthest point, one segment per step (later = darker), red for ground
  // lost, the ending on top. Columns are clickable (data-set).
  function setColumnsSVG(sets, color, pick) {
    const W = 330, H = 122, L = 26, R = 6, T = 12, B = 6;
    const step = (W - L - R) / Math.max(1, sets.length), bw = Math.max(3, Math.min(9, step - 2));
    const yOf = m => T + (H - T - B) * (1 - Math.max(0, Math.min(70, m)) / 70);
    let g = [[0, '0'], [35, '35m'], [70, '70m']].map(([m, l]) =>
      `<line x1="${L}" y1="${yOf(m)}" x2="${W - R}" y2="${yOf(m)}" stroke="rgba(255,255,255,0.1)"/>` +
      `<text x="${L - 4}" y="${yOf(m) + 3}" text-anchor="end" font-size="7.5" fill="#6c7689">${l}</text>`).join('') +
      `<line x1="${L}" y1="${yOf(60)}" x2="${W - R}" y2="${yOf(60)}" stroke="rgba(255,255,255,0.18)" stroke-dasharray="3 3"/>`;
    sets.forEach((s, i) => {
      const x = L + step * i + (step - bw) / 2, dim = pick != null && i !== pick;
      let col = '';
      for (let k = 1; k < s.steps.length; k++) {
        const a = s.steps[k - 1].y * Y_TO_M, b = s.steps[k].y * Y_TO_M;
        const top = Math.min(yOf(a), yOf(b)), h = Math.max(1.2, Math.abs(yOf(b) - yOf(a)));
        const op = [0.32, 0.45, 0.58, 0.72, 0.86, 1][Math.min(5, k - 1)];
        col += `<rect x="${x}" y="${top}" width="${bw}" height="${h}" fill="${b >= a ? color : '#ef4444'}" fill-opacity="${b >= a ? op : 0.85}" stroke="#0b0e14" stroke-width="0.6"/>`;
      }
      const topY = Math.min(...s.steps.map(p => yOf(p.y * Y_TO_M)));
      g += `<g class="pp-col" data-set="${i}" style="cursor:pointer" opacity="${dim ? 0.35 : 1}">` +
           `<rect x="${x - 1}" y="${T - 8}" width="${bw + 2}" height="${H - T + 2}" fill="transparent"/>` +
           (i === pick ? `<rect x="${x - 1.5}" y="${topY - 9}" width="${bw + 3}" height="${yOf(0) - topY + 10}" rx="2" fill="none" stroke="#fff" stroke-opacity="0.7"/>` : '') +
           col + endMarker(s.outcome, x + bw / 2, topY - 5, 2.3) + `</g>`;
    });
    return `<svg viewBox="0 0 ${W} ${H}" class="pp-chart" role="img" aria-label="Every set as a column, in order">${g}</svg>`;
  }

  // The one-line read-out of a picked set.
  function describeSet(s, i, total) {
    const st = OUTCOME_STYLE[s.outcome] || OUTCOME_STYLE.Other;
    const a = s.steps[0], z = s.steps[s.steps.length - 1];
    const touches = s.steps.filter(p => typeof p.k === 'number').length;
    const start = a.k === 'start' ? (a.how === 'tap' ? 'from the tap on halfway' : `won at ${(a.y * Y_TO_M).toFixed(0)}m`) : `from ${(a.y * Y_TO_M).toFixed(0)}m`;
    const gain = (z.y - a.y) * Y_TO_M;
    const move = s.end.type === 'Try' && s.end.name && s.end.name !== 'Other' ? ` · ${s.end.name}` : '';
    const how = touches ? `over ${touches} touch${touches === 1 ? '' : 'es'}` : 'straight from the turnover, no touch';
    return {
      title: `${st.label}${move}`, color: st.color,
      summary: `Set ${i + 1} of ${total}: ${start} to ${(z.y * Y_TO_M).toFixed(0)}m, ${gain >= 0 ? '+' : ''}${gain.toFixed(0)}m ${how}`,
      steps: pathGains(s).map(g => `${g.label} ${g.m >= 0 ? '+' : ''}${g.m.toFixed(0)}m`),
    };
  }

  const pathsKey = () => `<div class="pp-key">` +
    ['Try', 'Turnover', '6th Touch', 'Penalty'].map(o =>
      `<span><svg viewBox="-4 -4 8 8" width="10" height="10">${endMarker(o, 0, 0, 2.6)}</svg>${OUTCOME_STYLE[o].label}</span>`).join('') +
    `<span><svg viewBox="-4 -4 8 8" width="10" height="10"><circle r="2.2" fill="none" stroke="#fff" stroke-opacity="0.7" stroke-width="0.8"/></svg>where it started</span></div>`;

  // ── Tries by side and channel ─────────────────────────────────
  // From the side:/ch: tags on each try (events carry them as `tags`, parsed
  // from the Detail column). Untagged tries are counted, not dropped, so the
  // shares are read against how many tries were actually tagged.
  const SIDES    = ['open', 'blind'];
  const CHANNELS = ['MM', 'ML', 'LW', 'W+'];

  function tryTagStats(events) {
    const forTeam = key => {
      const tries = events.filter(a => a.type === 'Try' && a.actionOwner === key);
      const side = Object.fromEntries(SIDES.map(k => [k, 0]));
      const ch   = Object.fromEntries(CHANNELS.map(k => [k, 0]));
      let sideTagged = 0, chTagged = 0;
      tries.forEach(a => {
        const t = a.tags || {};
        if (side[t.side] != null) { side[t.side]++; sideTagged++; }
        if (ch[t.ch]     != null) { ch[t.ch]++;     chTagged++; }
      });
      return { tries: tries.length, side, ch, sideTagged, chTagged };
    };
    return { t1: forTeam('Team 1'), t2: forTeam('Team 2') };
  }

  // One team's tagged tries as a single-hue bar: the parts are one team's
  // whole, so they share its colour and are told apart by shade, gap and a
  // direct label — the same treatment as the channel bar above.
  function tagBar(counts, order, name, color, labels) {
    const total = order.reduce((n, k) => n + counts[k], 0);
    if (!total) return `<div class="cfig-row"><div class="cfig-name"><i style="background:${color}"></i>${name}</div>` +
                       `<div class="cbar cbar-empty">no tries tagged</div><div class="cfig-val">0</div></div>`;
    const shades = order.length === 2 ? [1, 0.55] : [1, 0.8, 0.6, 0.42];
    const segs = order.filter(k => counts[k]).map(k => {
      const i = order.indexOf(k), pct = counts[k] / total * 100, lab = (labels && labels[k]) || k;
      return `<div class="cseg" style="width:${pct}%;background:${color};opacity:${shades[i]}"
                title="${name} — ${counts[k]} of ${total} tagged tries ${lab}">${pct >= 16 ? `${lab} ${counts[k]}` : counts[k]}</div>`;
    }).join('');
    return `<div class="cfig-row">
      <div class="cfig-name"><i style="background:${color}"></i>${name}</div>
      <div class="cbar">${segs}</div>
      <div class="cfig-val">${total}</div>
    </div>`;
  }

  const SIDE_LABELS = { open: 'Open', blind: 'Blind' };
  const sideBar    = (stat, name, color) => tagBar(stat.side, SIDES, name, color, SIDE_LABELS);
  const channelTagBar = (stat, name, color) => tagBar(stat.ch, CHANNELS, name, color);

  return {
    Y_TO_M, RED_ZONE, MAP_LEN, OUTCOMES, mean,
    tryTagStats, sideBar, channelTagBar, SIDES, CHANNELS,
    START_ZONES, startZone, eventsStartingIn,
    possessionSets, computeFieldStats, outcomeOf,
    fieldMapSVG, fieldMapKey, outcomeBar, channelBar, territorySVG, chartLegend,
    possessionPaths, pathGains, gainBuckets, typicalSet, GAIN_BUCKETS, OUTCOME_STYLE,
    pathsSVG, gainChartSVG, setColumnsSVG, describeSet, pathsKey,
  };
})();
