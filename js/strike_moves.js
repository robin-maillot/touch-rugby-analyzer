// Depends on js/events.js (TR.isAttackEnd, TR.strikeMoveOf, TR.MIN_MOVE_ATTEMPTS)

// Try rate per strike move, over a list of normalised events.
//
// An "attempt" is any event where the ball changes hands off an attack — a Try
// (success), a Turnover, or a Penalty Attack (both failures). 6 Again and
// Penalty Defence are not attempts: the attack keeps the ball, so the move is
// still running. Attempts with no move tagged are excluded from every rate,
// which is why `coverage` is reported alongside — a 40% rate over 12% coverage
// is a different claim from the same rate over 90%.
//
// Events are {type, name, strikeMove, actionOwner}. Callers that want a single
// team's numbers filter by actionOwner first: on a Try, Turnover and Penalty
// Attack alike, TR.inferActionOwner returns the possession owner, so the action
// owner is always the attacking team that ran the move.
TR.strikeMoveStats = (events) => {
  const byMove = new Map();
  const cov = { tries: { tagged: 0, total: 0 }, fails: { tagged: 0, total: 0 } };

  (events || []).forEach(e => {
    if (!e || !TR.isAttackEnd(e.type, e.name)) return;
    const isTry = e.type === 'Try';
    const side  = isTry ? cov.tries : cov.fails;
    side.total++;
    // Re-derived, never read from the stored column: the viewer's inline edit
    // writes Name without touching Strike Move, so the column goes stale.
    const move = TR.strikeMoveOf(e.type, e.name, e.strikeMove);
    // 'Other' and 'Interception' count as untagged. On a Try they are what
    // "the annotator skipped the picker" looks like, while on a failure that
    // same skip yields ''. Left in, they would sit at a 100% artefact rate.
    if (!move || TR.EXCLUDED_MOVES.includes(move)) return;
    side.tagged++;
    if (!byMove.has(move)) byMove.set(move, { move, tries: 0, fails: 0, attempts: 0, rate: 0 });
    const m = byMove.get(move);
    m.attempts++;
    if (isTry) m.tries++; else m.fails++;
  });

  const tagged = cov.tries.tagged + cov.fails.tagged;
  const total  = cov.tries.total  + cov.fails.total;

  const moves = [...byMove.values()];
  moves.forEach(m => { m.rate = m.attempts ? m.tries / m.attempts : 0; });
  moves.sort((a, b) => b.rate - a.rate || b.attempts - a.attempts || a.move.localeCompare(b.move));

  // Efficiency needs a floor or a lone 1-for-1 tops the board on 100%.
  const eligible   = moves.filter(m => m.attempts >= TR.MIN_MOVE_ATTEMPTS);
  const topByRate  = eligible.length ? eligible[0] : null;

  // Volume has no floor, but a move that never scored isn't a "top scorer".
  const byTries    = moves.slice().sort((a, b) =>
    b.tries - a.tries || b.rate - a.rate || a.move.localeCompare(b.move));
  const topByTries = byTries.length && byTries[0].tries > 0 ? byTries[0] : null;

  const pct = c => (c.total ? c.tagged / c.total : 0);
  return {
    moves,
    // Per side as well as combined: a Try always has a Name, so its coverage is
    // 100% by construction and hides a sparse failure side when averaged in.
    coverage: {
      tagged, total, pct: total ? tagged / total : 0,
      tries: { tagged: cov.tries.tagged, total: cov.tries.total, pct: pct(cov.tries) },
      fails: { tagged: cov.fails.tagged, total: cov.fails.total, pct: pct(cov.fails) },
    },
    topByTries,
    topByRate,
  };
};
