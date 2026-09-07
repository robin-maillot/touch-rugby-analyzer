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

// The move this event was an attempt at, or '' when it isn't one or wasn't
// tagged. A Try's move IS its Name — derived rather than stored twice, so
// renaming a Try can't leave a stale move behind.
TR.strikeMoveOf = (type, name, strikeMove) =>
    type === 'Try'             ? (name || '')
  : TR.isAttackEnd(type, name) ? (strikeMove || '')
  : '';
