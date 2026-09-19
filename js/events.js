// Depends on js/config.js (TR namespace)

// Canonical event-type → sub-type map (source of truth for all pages).
// viewer.html previously had a divergent NAMES_BY_TYPE; TR.MENU is the canonical version.
TR.MENU = {
  'Penalty Attack':  ['Forward Pass', 'Touch and Pass', 'Off the Mark', 'Delay of Play', 'Hard Touch', 'Backchat', '7 on the field', 'Other'],
  'Penalty Defence': ['Offside', 'Hard Touch', 'In the Ruck', 'Not Moving Forward', 'Delay of Play', 'Backchat', '7 on the field', 'Other'],
  'Turnover':        ['Ball Down', '6th Touch', 'Dummy Touch', 'Bad Roll', 'In Touch', '6 Again', 'Interception', 'Other'],
  'Game Event':      ['Game Start', 'Game End'],
  'Try':             ['Scoop', '21', '32', '23', '33', '32 - Cut', '32 - Long', '32 - Quicky', '32 - Scoop', '23 - Backdoor', '23 - Quicky', '23 - Scoop', '33 - Backdoor', '33 - Cut', '33 - Quicky', '33 - Scoop', 'French Flair', 'Interception', 'Other'],
  'To Review':       [],
};

TR.NAMES_BY_TYPE = TR.MENU;

// Returns true if this event causes a possession switch.
TR.isTurnover = (type, name) => {
  if (type === 'Try')                              return true;
  if (type === 'Penalty Attack')                   return true;
  if (type === 'Penalty Defence')                  return false;
  if (type === 'Turnover' && name !== '6 Again')   return true;
  return false;
};

// ── Strike moves ───────────────────────────────────────────────
// The attacking move an attempt was running. A Try already records it in Name;
// these let a Turnover or Penalty Attack record the move that failed, so a try
// rate per move can be computed. Sliced so a caller can't mutate TR.MENU.
TR.STRIKE_MOVES      = TR.MENU['Try'].slice();
TR.STRIKE_MOVE_TYPES = ['Try', 'Turnover', 'Penalty Attack'];
TR.MIN_MOVE_ATTEMPTS = 2;

// Selectable in the annotators, but never rate-bearing. On a Try these are what
// "the annotator skipped the picker" looks like — annotator_field2 filters
// 'Other' out of its sub-type strip, and Simple Mode names every Try 'Other' —
// while on a failure that same skip yields ''. Counted as untagged so they
// cannot sit at a 100% artefact rate and top both leaderboards. 'Interception'
// on a try means a defensive intercept, not a called move off the tap, and has
// no failure counterpart at all.
TR.EXCLUDED_MOVES = ['Other', 'Interception'];

// An attempt ends precisely when the ball changes hands, which TR.isTurnover
// already encodes: Try, Penalty Attack and Turnover all end it — except
// '6 Again' (and Penalty Defence), where the attack keeps the ball and the
// move is still running.
TR.isAttackEnd = (type, name) =>
  TR.STRIKE_MOVE_TYPES.includes(type) && TR.isTurnover(type, name);

// A touch does not end the attack, but a called move that ends in a touch did
// not break the line — so a TAGGED touch is a failure of that move, and counts.
// Untagged touches never count: touches 1-3 are meant to be touched (settle,
// drive, yards), so an untagged touch is ordinary play, not a missed tag. That
// asymmetry is deliberate — were untagged touches in the denominator, coverage
// would collapse the moment touch uploading is switched on.
//
// Tag the touch where the called move was stopped, and not also the turnover
// that ends the same possession: two tags on one attempt count it twice.
TR.isTouchFailure = (type, strikeMove) => type === 'Touch' && !!strikeMove;

// The move this event was an attempt at, or '' when it isn't one or wasn't
// tagged. A Try's move IS its Name — derived rather than stored twice, so
// renaming a Try can't leave a stale move behind.
//
// Deliberately narrower than TR.recordedMoveOf below: this is the stats-facing
// answer, so a defensive penalty yields '' even when a move is recorded on it.
TR.strikeMoveOf = (type, name, strikeMove) =>
    type === 'Try'                            ? (name || '')
  : TR.isAttackEnd(type, name)                ? (strikeMove || '')
  : TR.isTouchFailure(type, strikeMove)       ? strikeMove
  : '';

// ── Recording a move vs. counting one ──────────────────────────
// Which events the annotators offer the move picker on. An attack penalty or a
// turnover ends the attempt, so its move feeds the try rate. A DEFENSIVE
// penalty does not end anything — the attack keeps the ball — but the move it
// was running when the defence infringed is still worth recording: it is how
// you find the moves that pressure a defence into conceding. So the picker is
// offered, and the value is stored and exported, while every rate stays
// untouched (TR.isAttackEnd, which the stats gate on, still says false).
//
// A Try is excluded because its move IS its name, not a separate field.
// Touch is offered too: unlike a defensive penalty, a touch that stopped a
// called move IS counted as that move failing (see TR.isTouchFailure).
TR.offersStrikeMove = (type, name) =>
  (type !== 'Try' && TR.isAttackEnd(type, name)) ||
  type === 'Penalty Defence' || type === 'Touch';

// The move to SHOW and EXPORT for an event — the annotator's move column, the
// CSV, the Strike Move cell pushed to the sheet, the Events viewer's tag.
// Use TR.strikeMoveOf instead for anything that counts attempts or rates: that
// one drops a defensive penalty's move on purpose, so one attack can never land
// in the denominator twice (the penalty, and then the Try or Turnover that
// actually ended the same attack).
TR.recordedMoveOf = (type, name, strikeMove) =>
    type === 'Try'                    ? (name || '')
  : TR.offersStrikeMove(type, name)   ? (strikeMove || '')
  : '';
