/* Shared demo engine for the Field Annotator v2 mockups.
   Not production code — just enough state to make the three layouts tappable so
   they can be judged on feel rather than on a screenshot. */
(function (global) {

  // ── Pitch geometry (attack-normalised) ───────────────────────
  // x: 0..100 left→right.  y: 0 = the try line being attacked (top of screen),
  // 100 = your own try line (bottom). The team in possession always attacks up,
  // so "forward" is always the same direction on screen.
  // The viewBox adds an in-goal band at each end: a tap above y=0 is a try, and
  // a tap below y=100 is a play-the-ball inside your own in-goal.
  const BAND = 15;                       // in-goal depth, in y units
  const VB   = { x: 0, y: -BAND, w: 100, h: 100 + BAND * 2 };

  function pitchSVG(opts) {
    opts = opts || {};
    return opts.landscape ? landscapeSVG() : portraitSVG();
  }

  // Attack is up: the try line being attacked is at the top of the screen.
  function portraitSVG() {
    const line = (y, cls) => `<line x1="0" y1="${y}" x2="100" y2="${y}" class="${cls}"/>`;
    return `
<svg class="pitch" viewBox="0 ${-BAND} 100 ${100 + BAND * 2}" preserveAspectRatio="none">
  ${TURF}
  <rect x="0" y="${-BAND}" width="100" height="${100 + BAND * 2}" fill="url(#turf)"/>
  <rect x="0" y="${-BAND}" width="100" height="${BAND}" class="ingoal try-band"/>
  <rect x="0" y="100"      width="100" height="${BAND}" class="ingoal own-band"/>
  ${line(0, 'ln try-ln')}${line(100, 'ln try-ln')}
  ${line(7.1, 'ln five')}${line(92.9, 'ln five')}
  ${line(50, 'ln half')}
  <text x="11" y="${-BAND / 2 + 2}" class="band-label">TRY</text>
  <text x="89" y="${-BAND / 2 + 2}" class="band-label">TRY</text>
  <text x="50" y="${100 + 4.6}" class="band-label dim">own in-goal</text>
</svg>`;
  }

  // Attack is to the right: the same pitch turned a quarter, for holding the
  // phone in two hands. Pitch coordinates are unchanged — only the drawing and
  // the pointer mapping know about the rotation.
  function landscapeSVG() {
    const line = (x, cls) => `<line x1="${x}" y1="0" x2="${x}" y2="100" class="${cls}"/>`;
    return `
<svg class="pitch" viewBox="${-BAND} 0 ${100 + BAND * 2} 100" preserveAspectRatio="none">
  ${TURF}
  <rect x="${-BAND}" y="0" width="${100 + BAND * 2}" height="100" fill="url(#turf)"/>
  <rect x="100"      y="0" width="${BAND}" height="100" class="ingoal try-band"/>
  <rect x="${-BAND}" y="0" width="${BAND}" height="100" class="ingoal own-band"/>
  ${line(0, 'ln try-ln')}${line(100, 'ln try-ln')}
  ${line(7.1, 'ln five')}${line(92.9, 'ln five')}
  ${line(50, 'ln half')}
  <text x="${100 + BAND / 2}" y="52" class="band-label" transform="rotate(90 ${100 + BAND / 2} 50)">T R Y</text>
  <text x="${-BAND / 2}" y="52" class="band-label dim" transform="rotate(-90 ${-BAND / 2} 50)">own in-goal</text>
</svg>`;
  }

  const TURF = `<defs><linearGradient id="turf" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#12301f"/><stop offset="100%" stop-color="#0d2417"/>
    </linearGradient></defs>`;

  // ── Demo state ───────────────────────────────────────────────
  function makeState() {
    return {
      possession: 'FRA',
      other:      'ENG',
      touch:      0,
      score:      { FRA: 3, ENG: 2 },
      pins:       [],      // {x,y,n,kind}  kind: touch|turnover|try|pen
      log:        [],      // {t,label,sub,team,kind}
      armed:      null,    // 'Pen Atk' | 'Pen Def' | '6 Again'
      clock:      742,
    };
  }

  // Convert a pointer event on the <svg> into pitch coordinates
  // (x = across the pitch, y = 0 at the try line being attacked).
  function pointAt(svg, ev, landscape) {
    const r  = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width;
    const py = (ev.clientY - r.top) / r.height;
    if (landscape) {
      const X = -BAND + px * (100 + BAND * 2);
      return { x: py * 100, y: 100 - X };
    }
    return { x: px * 100, y: -BAND + py * (100 + BAND * 2) };
  }

  // Pitch coordinates → viewBox coordinates.
  const toView = (p, landscape) => landscape ? { vx: 100 - p.y, vy: p.x } : { vx: p.x, vy: p.y };

  const fmt = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

  // Screen-space y for a pitch y (the SVG is stretched, so pins are drawn in
  // viewBox units and simply inherit the stretch).
  // Pins are HTML, not SVG: the pitch stretches to fill the screen
  // (preserveAspectRatio="none"), which would squash SVG circles into ellipses.
  // Percentage-positioned divs stretch their *position* with the pitch while
  // staying perfectly round, and they hit-test more reliably under a thumb.
  function renderPins(layer, st, landscape) {
    const D = 100 + BAND * 2;
    layer.innerHTML = st.pins.map(p => {
      const { vx, vy } = toView(p, landscape);
      const left = landscape ? (vx + BAND) / D * 100 : vx;
      const top  = landscape ? vy : (vy + BAND) / D * 100;
      const label = p.kind === 'touch' ? p.n : p.kind === 'try' ? '✓' : p.kind === 'pen' ? 'P' : '✕';
      return `<button class="pin pin-${p.kind}${p.stale ? ' stale' : ''}" data-i="${p.i}"
        style="left:${left.toFixed(2)}%;top:${top.toFixed(2)}%">${label}</button>`;
    }).join('');
  }

  // One tap on the pitch, resolved against the current arm/touch state.
  function tap(st, pt) {
    st.clock += 7 + Math.floor(Math.random() * 12);

    if (st.armed) {
      const kind = st.armed;
      push(st, pt, 'pen', kind, kind === 'Pen Def' ? st.other : st.possession);
      if (kind === 'Pen Def') flip(st);
      else st.touch = 0;
      st.armed = null;
      return;
    }

    if (pt.y < 0) {                                   // try band
      st.score[st.possession]++;
      push(st, { x: pt.x, y: 0 }, 'try', 'Try', st.possession);
      flip(st);
      return;
    }

    const y = Math.min(100, Math.max(0, pt.y));
    st.touch++;
    if (st.touch >= 6) {                              // handover on the 6th
      push(st, { x: pt.x, y }, 'turnover', '6th Touch', st.possession);
      flip(st);
      return;
    }
    push(st, { x: pt.x, y }, 'touch', 'Touch ' + st.touch, st.possession);
  }

  // Tapping a pin promotes it from a touch to a turnover.
  function promote(st, i) {
    const p = st.pins.find(x => x.i === i);
    if (!p || p.kind !== 'touch') return false;
    p.kind = 'turnover';
    const e = st.log.find(l => l.i === i);
    if (e) { e.label = 'Turnover'; e.sub = 'Ball Down'; e.kind = 'turnover'; }
    flip(st);
    return true;
  }

  let seq = 0;
  function push(st, pt, kind, label, team) {
    const i = ++seq;
    st.pins.push({ i, x: pt.x, y: pt.y, n: st.touch, kind });
    st.log.unshift({ i, t: fmt(st.clock), label, sub: '', team, kind });
    if (st.pins.length > 8) st.pins.shift();
  }

  // Possession over: the last set fades to grey rather than vanishing, so the
  // pin you just tapped stays visible while the new set builds on top of it.
  function flip(st) {
    const p = st.possession;
    st.possession = st.other;
    st.other = p;
    st.touch = 0;
    st.pins.forEach(x => { x.stale = true; });
    st.pins = st.pins.slice(-6);
  }

  function undo(st) {
    const e = st.log.shift();
    if (!e) return;
    st.pins = st.pins.filter(p => p.i !== e.i);
    if (e.kind === 'touch') st.touch = Math.max(0, st.touch - 1);
    if (e.kind === 'try') st.score[e.team] = Math.max(0, st.score[e.team] - 1);
  }

  global.Mock = { pitchSVG, toView, makeState, pointAt, renderPins, tap, promote, undo, fmt, BAND, VB };
})(window);
