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

// ── Detail tags ────────────────────────────────────────────────
// The Detail column holds "key:value; key:value" tags rather than a column per
// stat, most of which would be empty on most rows:
//
//   pos:62,48            pitch position, x,y 0-100, attack-normalised
//   player:7             the try scorer's shirt number
//   side:open | blind    which side of the ruck the try went
//   ch:MM | ML | LW | W+ the channel it was scored in
//
// "; " separates tags because a value may contain spaces; the first ":" ends
// the key, so a value may contain colons. Unknown keys survive a round trip, so
// a stat added later never needs a new column or a reader change to be kept.
TR.DETAIL_SIDES    = ['open', 'blind'];
TR.DETAIL_CHANNELS = ['MM', 'ML', 'LW', 'W+'];
const DETAIL_ORDER = ['pos', 'player', 'side', 'ch'];

TR.parseDetail = (str) => {
  const out = {};
  String(str || '').split(';').forEach(part => {
    const i = part.indexOf(':');
    if (i <= 0) return;
    const key = part.slice(0, i).trim().toLowerCase();
    const val = part.slice(i + 1).trim();
    if (key && val) out[key] = val;
  });
  return out;
};

// Known keys first in a fixed order, the rest alphabetically, so the same tags
// always write the same cell. Empty values are dropped; separators that would
// corrupt the cell are stripped from values rather than escaped.
TR.formatDetail = (obj) => {
  const tags = {};
  Object.entries(obj || {}).forEach(([k, v]) => {
    const val = v == null ? '' : String(v).replace(/[;\r\n]+/g, ' ').trim();
    if (val) tags[k.trim().toLowerCase()] = val;
  });
  const rest = Object.keys(tags).filter(k => !DETAIL_ORDER.includes(k)).sort();
  return [...DETAIL_ORDER.filter(k => k in tags), ...rest]
    .map(k => `${k}:${tags[k]}`)
    .join('; ');
};

// Before Detail existed, field annotator v2 appended the position to Comment as
// "@62,48". Read from Detail first and fall back to that, so older games keep
// their positions.
const LEGACY_POS = /(?:^|\s)@(\d{1,3}(?:\.\d+)?),(\d{1,3}(?:\.\d+)?)(?=\s|$)/;

TR.detailPos = (detail, comment) => {
  const d = typeof detail === 'string' ? TR.parseDetail(detail) : (detail || {});
  const m = d.pos ? /^(\d{1,3}(?:\.\d+)?),(\d{1,3}(?:\.\d+)?)$/.exec(d.pos) : LEGACY_POS.exec(String(comment || ''));
  if (!m) return null;
  const x = +m[1], y = +m[2];
  return x <= 100 && y <= 100 ? { x, y } : null;
};

// The comment as a person wrote it — without a legacy "@x,y" position token.
TR.stripLegacyPos = (comment) =>
  String(comment || '').replace(LEGACY_POS, '').trim();

// A short human label for the try tags, e.g. "#7 · Open · ML". Position is left
// out: it reads as a number pair, not as a fact about the try.
TR.detailLabel = (detail) => {
  const d = typeof detail === 'string' ? TR.parseDetail(detail) : (detail || {});
  return [
    d.player ? '#' + d.player : '',
    d.side ? d.side.charAt(0).toUpperCase() + d.side.slice(1) : '',
    d.ch || '',
  ].filter(Boolean).join(' · ');
};

// Which side of the last ruck a try went, from where each was on the pitch.
// Both x values are attack-normalised (0 = the attacking team's left touchline,
// 100 = its right), and both belong to the scoring team's set, so they share a
// frame. The blind side is the narrower one — between the ruck and its nearer
// touchline — and anything across the ruck from it is open. Returns '' when it
// can't be told: no ruck, a ruck dead centre (both sides equal), or a try
// straight through the ruck's own line.
TR.inferTrySide = (ruckX, tryX) => {
  if (ruckX == null || tryX == null || isNaN(ruckX) || isNaN(tryX)) return '';
  if (ruckX === 50 || tryX === ruckX) return '';
  const blindIsLeft = ruckX < 50;
  return (tryX < ruckX) === blindIsLeft ? 'blind' : 'open';
};
