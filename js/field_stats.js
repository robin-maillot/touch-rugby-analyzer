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
        if (a.name === 'Game Start' || a.name === 'Game End') { cur = null; }
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

  return {
    Y_TO_M, RED_ZONE, MAP_LEN, OUTCOMES, mean,
    possessionSets, computeFieldStats, outcomeOf,
    fieldMapSVG, fieldMapKey, outcomeBar, channelBar, territorySVG, chartLegend,
  };
})();
