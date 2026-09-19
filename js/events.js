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

// Every event type that can carry a called move. Wider than STRIKE_MOVE_TYPES,
// which is only the set whose members END a possession — a defensive penalty
// and a 6 Again both leave the attack the ball and a fresh count, and a Touch
// leaves the set running entirely, yet a move was still called and still did
// not score. Touch is handled separately below because only a TAGGED touch
// counts; the rest count tagged or not.
TR.MOVE_BEARING_TYPES = ['Try', 'Turnover', 'Penalty Attack', 'Penalty Defence'];
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
// A possession is not one attempt. A move can be called at every touch, so one
// set legitimately produces several — tag each touch that carried a called
// move, and the turnover or try that ends the set as well. They are separate
// attempts, not one attempt tagged twice.
TR.isTouchFailure = (type, strikeMove) => type === 'Touch' && !!strikeMove;

// Which events enter a move's record. Deliberately NOT expressed in terms of
// isAttackEnd: whether the ball changed hands turned out to be the wrong
// question. A set can end in a try, a turnover, a penalty either way, a fresh
// count, or the runner simply being touched — a move was called in each case,
// and in every case but the try it did not score.
//
// So the rate is tries ÷ attempts and "fails" means "did not score", not "the
// defence stopped it". A defensive penalty and a 6 Again are good attacking
// outcomes that read as fails here. Worth knowing before judging a move that
// reliably wins penalties.
//
// Only Game Event and To Review are never attempts — and an untagged touch,
// which is ordinary play rather than a called move.
TR.countsAsAttempt = (type, name, strikeMove) =>
     TR.MOVE_BEARING_TYPES.includes(type)
  || TR.isTouchFailure(type, strikeMove);

// The move this event was an attempt at, or '' when it isn't one or wasn't
// tagged. A Try's move IS its Name — derived rather than stored twice, so
// renaming a Try can't leave a stale move behind.
TR.strikeMoveOf = (type, name, strikeMove) =>
    type === 'Try'                              ? (name || '')
  : TR.countsAsAttempt(type, name, strikeMove)  ? (strikeMove || '')
  : '';

// ── Where the picker appears ───────────────────────────────────
// Every event that can carry a move offers one. A Try is excluded because its
// move IS its name, not a separate field.
// Touch is offered too: unlike a defensive penalty, a touch that stopped a
// called move IS counted as that move failing (see TR.isTouchFailure).
TR.offersStrikeMove = (type, name) =>
  (type !== 'Try' && TR.MOVE_BEARING_TYPES.includes(type)) || type === 'Touch';

// The move to SHOW and EXPORT — the annotator's move column, the CSV, the
// Strike Move cell pushed to the sheet, the Events viewer's tag.
//
// This used to be deliberately wider than TR.strikeMoveOf, back when a
// defensive penalty's move was recorded but never counted. Now that a defensive
// penalty and a tagged touch both count, the two rules agree exactly, so this
// delegates rather than restating them and risking a silent drift. Splitting
// them again means giving this its own body once more, not editing one of the
// two in place.
TR.recordedMoveOf = (type, name, strikeMove) =>
  TR.strikeMoveOf(type, name, strikeMove);
