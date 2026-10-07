#!/usr/bin/env node
// Runs TR.* shared-utility assertions in Node.js — no npm required.
// Uses the vm module to load browser-targeted JS files with minimal stubs.
const vm     = require('vm');
const fs     = require('fs');
const assert = require('assert/strict');

// In-memory localStorage stand-in so the field-annotator game store can be
// exercised here the same way it runs in the browser.
function memStorage() {
  const map = new Map();
  return {
    getItem:    k => (map.has(String(k)) ? map.get(String(k)) : null),
    setItem:    (k, v) => map.set(String(k), String(v)),
    removeItem: k => map.delete(String(k)),
    clear:      () => map.clear(),
  };
}

const ctx = vm.createContext({
  sessionStorage: { getItem: () => null },
  localStorage:   memStorage(),
  window:         { location: { replace() {} } },
});

for (const f of ['js/config.js', 'js/utils.js', 'js/events.js', 'js/possession.js', 'js/consistency.js', 'js/player.js', 'js/field_games.js', 'js/strike_moves.js', 'js/playlists.js', 'js/field_stats.js']) {
  vm.runInContext(fs.readFileSync(f, 'utf8'), ctx);
}

const { TR } = ctx;

let passed = 0, failed = 0;
function test(name, fn) {
  try   { fn(); console.log(`  ✓ ${name}`); passed++; }
  catch (e) { console.error(`  ✗ ${name}\n    ${e.message}`); failed++; }
}

// ── TR.fmt ────────────────────────────────────────────────────
console.log('TR.fmt');
test('zero/falsy',  () => { assert.equal(TR.fmt(0), '0:00:00'); assert.equal(TR.fmt(null), '0:00:00'); assert.equal(TR.fmt(undefined), '0:00:00'); assert.equal(TR.fmt(NaN), '0:00:00'); });
test('sub-minute',  () => { assert.equal(TR.fmt(5), '0:00:05'); assert.equal(TR.fmt(59), '0:00:59'); });
test('minutes',     () => { assert.equal(TR.fmt(60), '0:01:00'); assert.equal(TR.fmt(65), '0:01:05'); assert.equal(TR.fmt(599), '0:09:59'); assert.equal(TR.fmt(600), '0:10:00'); });
test('hours',       () => { assert.equal(TR.fmt(3600), '1:00:00'); assert.equal(TR.fmt(3661), '1:01:01'); assert.equal(TR.fmt(36000), '10:00:00'); });
test('fractional seconds floor', () => { assert.equal(TR.fmt(5.9), '0:00:05'); assert.equal(TR.fmt(59.9), '0:00:59'); });

// ── TR.enc ────────────────────────────────────────────────────
console.log('TR.enc');
test('encodes special chars', () => { assert.equal(TR.enc('a b'), 'a%20b'); assert.equal(TR.enc('a&b=c'), 'a%26b%3Dc'); });
test('plain strings pass through', () => assert.equal(TR.enc('m30-staff'), 'm30-staff'));

// ── TR.extractVideoId ─────────────────────────────────────────
console.log('TR.extractVideoId');
test('watch URL',  () => assert.equal(TR.extractVideoId('https://www.youtube.com/watch?v=dQw4w9WgXcQ'), 'dQw4w9WgXcQ'));
test('youtu.be',   () => assert.equal(TR.extractVideoId('https://youtu.be/dQw4w9WgXcQ'), 'dQw4w9WgXcQ'));
test('embed',      () => assert.equal(TR.extractVideoId('https://www.youtube.com/embed/dQw4w9WgXcQ'), 'dQw4w9WgXcQ'));
test('live',       () => assert.equal(TR.extractVideoId('https://www.youtube.com/live/dQw4w9WgXcQ'), 'dQw4w9WgXcQ'));
test('invalid',    () => { assert.equal(TR.extractVideoId('https://example.com'), null); assert.equal(TR.extractVideoId(null), null); assert.equal(TR.extractVideoId(''), null); });

// ── TR.normalizeYoutubeUrl ────────────────────────────────────
console.log('TR.normalizeYoutubeUrl');
const CANON = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ';
test('bare 11-char video ID → watch URL', () => assert.equal(TR.normalizeYoutubeUrl('dQw4w9WgXcQ'), CANON));
test('live/ stream URL → watch URL',      () => assert.equal(TR.normalizeYoutubeUrl('https://www.youtube.com/live/dQw4w9WgXcQ'), CANON));
test('watch URL → canonical',             () => assert.equal(TR.normalizeYoutubeUrl('https://www.youtube.com/watch?v=dQw4w9WgXcQ'), CANON));
test('youtu.be with extra params stripped', () => assert.equal(TR.normalizeYoutubeUrl('https://youtu.be/dQw4w9WgXcQ?t=90'), CANON));
test('watch URL with playlist param',     () => assert.equal(TR.normalizeYoutubeUrl('https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123'), CANON));
test('embed URL → watch URL',             () => assert.equal(TR.normalizeYoutubeUrl('https://www.youtube.com/embed/dQw4w9WgXcQ'), CANON));
test('surrounding whitespace trimmed',    () => assert.equal(TR.normalizeYoutubeUrl('  dQw4w9WgXcQ  '), CANON));
test('empty / null / undefined → empty string', () => { assert.equal(TR.normalizeYoutubeUrl(''), ''); assert.equal(TR.normalizeYoutubeUrl(null), ''); assert.equal(TR.normalizeYoutubeUrl(undefined), ''); });
test('too-short / non-ID text → empty string', () => { assert.equal(TR.normalizeYoutubeUrl('hello'), ''); assert.equal(TR.normalizeYoutubeUrl('too-short'), ''); });
test('exactly 11 valid chars treated as ID', () => assert.equal(TR.normalizeYoutubeUrl('abcdefghijk'), 'https://www.youtube.com/watch?v=abcdefghijk'));
test('12-char bare string is not an ID',  () => assert.equal(TR.normalizeYoutubeUrl('abcdefghijkl'), ''));

// ── TR.substituteTeams ────────────────────────────────────────
console.log('TR.substituteTeams');
test('replaces Team 1/2', () => {
  const rows = [['Team 1', 'Team 2'], ['Team 2', 'Team 1']];
  TR.substituteTeams(rows, 'France', 'England', [0, 1]);
  assert.deepEqual(rows[0], ['France', 'England']);
  assert.deepEqual(rows[1], ['England', 'France']);
});
test('skips negative index', () => {
  const rows = [['Team 1', 'Team 2']];
  TR.substituteTeams(rows, 'France', 'England', [-1, 1]);
  assert.equal(rows[0][0], 'Team 1');
  assert.equal(rows[0][1], 'England');
});
test('leaves non-placeholder values', () => {
  const rows = [['Try', 'France']];
  TR.substituteTeams(rows, 'France', 'England', [0, 1]);
  assert.deepEqual(rows[0], ['Try', 'France']);
});

// ── TR.MENU ───────────────────────────────────────────────────
console.log('TR.MENU');
test('canonical types present',   () => { ['Penalty Attack','Penalty Defence','Turnover','Game Event','Try','To Review'].forEach(t => assert.ok(Array.isArray(TR.MENU[t]), `missing ${t}`)); });
test('Turnover has 6 Again',      () => assert.ok(TR.MENU['Turnover'].includes('6 Again')));
test('Game Event has Start/End',  () => { assert.ok(TR.MENU['Game Event'].includes('Game Start')); assert.ok(TR.MENU['Game Event'].includes('Game End')); });
test('Try has 21 and Interception', () => { assert.ok(TR.MENU['Try'].includes('21')); assert.ok(TR.MENU['Try'].includes('Interception')); });
test('NAMES_BY_TYPE alias',       () => assert.equal(TR.NAMES_BY_TYPE, TR.MENU));

// ── TR.isTurnover ─────────────────────────────────────────────
console.log('TR.isTurnover');
test('Try → true',              () => { assert.equal(TR.isTurnover('Try', 'Scoop'), true); assert.equal(TR.isTurnover('Try', 'Other'), true); });
test('Penalty Attack → true',   () => assert.equal(TR.isTurnover('Penalty Attack', 'Forward Pass'), true));
test('Penalty Defence → false', () => assert.equal(TR.isTurnover('Penalty Defence', 'Offside'), false));
test('Turnover 6th Touch → true', () => assert.equal(TR.isTurnover('Turnover', '6th Touch'), true));
test('Turnover Ball Down → true', () => assert.equal(TR.isTurnover('Turnover', 'Ball Down'), true));
test('Turnover 6 Again → false',  () => assert.equal(TR.isTurnover('Turnover', '6 Again'), false));
test('Game Event Game Start → false', () => assert.equal(TR.isTurnover('Game Event', 'Game Start'), false));
test('Game Event Game End → false',   () => assert.equal(TR.isTurnover('Game Event', 'Game End'),   false));
test('Game Event Ball Live → false',  () => assert.equal(TR.isTurnover('Game Event', 'Ball Live'),  false));
test('To Review → false',             () => assert.equal(TR.isTurnover('To Review', ''), false));

// ── TR.STRIKE_MOVES ───────────────────────────────────────────
console.log('TR.STRIKE_MOVES');
test('matches the Try menu',   () => assert.deepEqual(TR.STRIKE_MOVES, TR.MENU['Try']));
test('is a copy, not the same array', () => assert.notEqual(TR.STRIKE_MOVES, TR.MENU['Try']));
test('keeps Other and Interception', () => {
  assert.ok(TR.STRIKE_MOVES.includes('Other'));
  assert.ok(TR.STRIKE_MOVES.includes('Interception'));
});
test('min attempts is 2',      () => assert.equal(TR.MIN_MOVE_ATTEMPTS, 2));

// ── TR.isAttackEnd ────────────────────────────────────────────
console.log('TR.isAttackEnd');
test('Try ends an attempt',        () => assert.equal(TR.isAttackEnd('Try', '32 - Cut'), true));
test('Penalty Attack ends it',     () => assert.equal(TR.isAttackEnd('Penalty Attack', 'Forward Pass'), true));
test('Turnover ends it',           () => assert.equal(TR.isAttackEnd('Turnover', 'Ball Down'), true));
test('6 Again does not',           () => assert.equal(TR.isAttackEnd('Turnover', '6 Again'), false));
test('Penalty Defence does not',   () => assert.equal(TR.isAttackEnd('Penalty Defence', 'Offside'), false));
test('Game Event does not',        () => assert.equal(TR.isAttackEnd('Game Event', 'Game Start'), false));
test('To Review does not',         () => assert.equal(TR.isAttackEnd('To Review', ''), false));

// ── Touches as move failures ──────────────────────────────────
console.log('TR.isTouchFailure');
test('a tagged touch is a failure',   () => assert.equal(TR.isTouchFailure('Touch', '32 - Cut'), true));
test('an untagged touch is not',      () => assert.equal(TR.isTouchFailure('Touch', ''), false));
test('undefined move is not',         () => assert.equal(TR.isTouchFailure('Touch', undefined), false));
test('only a Touch qualifies',        () => assert.equal(TR.isTouchFailure('Turnover', '32 - Cut'), false));
test('a touch never ends the attack', () => assert.equal(TR.isAttackEnd('Touch', 'Touch 3'), false));
test('the picker is offered on it',   () => assert.equal(TR.offersStrikeMove('Touch', 'Touch 3'), true));
test('a tagged touch has a move',     () => assert.equal(TR.strikeMoveOf('Touch', 'Touch 3', '32 - Cut'), '32 - Cut'));
test('an untagged touch has none',    () => assert.equal(TR.strikeMoveOf('Touch', 'Touch 3', ''), ''));
test('it is recorded as well as counted',
  () => assert.equal(TR.recordedMoveOf('Touch', 'Touch 3', '32 - Cut'), '32 - Cut'));

// ── TR.strikeMoveOf ───────────────────────────────────────────
console.log('TR.strikeMoveOf');
test('Try returns its own name',    () => assert.equal(TR.strikeMoveOf('Try', '33 - Quicky', ''), '33 - Quicky'));
test('Try ignores a stored move',   () => assert.equal(TR.strikeMoveOf('Try', '33 - Quicky', 'Scoop'), '33 - Quicky'));
test('Turnover returns its move',   () => assert.equal(TR.strikeMoveOf('Turnover', 'Ball Down', '32 - Cut'), '32 - Cut'));
test('Pen Attack returns its move', () => assert.equal(TR.strikeMoveOf('Penalty Attack', 'Forward Pass', '23'), '23'));
test('untagged returns empty',      () => assert.equal(TR.strikeMoveOf('Turnover', 'Ball Down', ''), ''));
test('undefined move returns empty',() => assert.equal(TR.strikeMoveOf('Turnover', 'Ball Down', undefined), ''));
test('6 Again keeps its move',      () => assert.equal(TR.strikeMoveOf('Turnover', '6 Again', '32'), '32'));
test('Pen Defence keeps its move',  () => assert.equal(TR.strikeMoveOf('Penalty Defence', 'Offside', '32'), '32'));
test('Game Event drops its move',   () => assert.equal(TR.strikeMoveOf('Game Event', 'Game Start', '32'), ''));

// ── TR.otherTeam ──────────────────────────────────────────────
console.log('TR.otherTeam');
test('Team 1 → Team 2',  () => assert.equal(TR.otherTeam('Team 1'), 'Team 2'));
test('Team 2 → Team 1',  () => assert.equal(TR.otherTeam('Team 2'), 'Team 1'));

// ── TR.inferPossessionAfter ───────────────────────────────────
console.log('TR.inferPossessionAfter');
test('Try → other',             () => { assert.equal(TR.inferPossessionAfter('Team 1', 'Try', 'Scoop'), 'Team 2'); assert.equal(TR.inferPossessionAfter('Team 2', 'Try', 'Scoop'), 'Team 1'); });
test('Penalty Attack → other',  () => assert.equal(TR.inferPossessionAfter('Team 1', 'Penalty Attack', 'Forward Pass'), 'Team 2'));
test('Penalty Defence → same',  () => assert.equal(TR.inferPossessionAfter('Team 1', 'Penalty Defence', 'Offside'), 'Team 1'));
test('Turnover → other',        () => assert.equal(TR.inferPossessionAfter('Team 1', 'Turnover', '6th Touch'), 'Team 2'));
test('6 Again → same',          () => assert.equal(TR.inferPossessionAfter('Team 1', 'Turnover', '6 Again'), 'Team 1'));
test('Ball Live → same',        () => { assert.equal(TR.inferPossessionAfter('Team 1', 'Game Event', 'Ball Live'), 'Team 1'); assert.equal(TR.inferPossessionAfter('Team 2', 'Game Event', 'Ball Live'), 'Team 2'); });
test('Game Start → same',       () => assert.equal(TR.inferPossessionAfter('Team 1', 'Game Event', 'Game Start'), 'Team 1'));

// ── TR.inferActionOwner ───────────────────────────────────────
console.log('TR.inferActionOwner');
test('Try → possession owner',       () => assert.equal(TR.inferActionOwner('Team 1', 'Try', 'Scoop'), 'Team 1'));
test('Turnover → possession owner',  () => assert.equal(TR.inferActionOwner('Team 1', 'Turnover', '6th Touch'), 'Team 1'));
test('6 Again → other (defence)',    () => assert.equal(TR.inferActionOwner('Team 1', 'Turnover', '6 Again'), 'Team 2'));
test('Penalty Attack → same',        () => assert.equal(TR.inferActionOwner('Team 1', 'Penalty Attack', 'Forward Pass'), 'Team 1'));
test('Penalty Defence → other',      () => { assert.equal(TR.inferActionOwner('Team 1', 'Penalty Defence', 'Offside'), 'Team 2'); assert.equal(TR.inferActionOwner('Team 2', 'Penalty Defence', 'Offside'), 'Team 1'); });
test('Ball Live → possession owner', () => assert.equal(TR.inferActionOwner('Team 1', 'Game Event', 'Ball Live'), 'Team 1'));

// ── TR.player.providerFor ────────────────────────────────────
console.log('TR.player.providerFor');
test('null meta',              () => assert.equal(TR.player.providerFor(null), null));
test('empty meta',             () => assert.equal(TR.player.providerFor({}), null));
test('youtubelink → youtube',  () => assert.equal(TR.player.providerFor({ youtubelink: 'https://youtu.be/abc' }), 'youtube'));
test('no youtubelink → null',  () => assert.equal(TR.player.providerFor({ gcsObject: 'x.mp4' }), null));

// ── TR.player.hasClip ────────────────────────────────────────
console.log('TR.player.hasClip');
test('null meta',         () => assert.equal(TR.player.hasClip(null), false));
test('no gcsObject',      () => assert.equal(TR.player.hasClip({ youtubelink: 'x' }), false));
test('with gcsObject',    () => assert.equal(TR.player.hasClip({ gcsObject: 'a.mp4' }), true));
test('empty gcsObject',   () => assert.equal(TR.player.hasClip({ gcsObject: '' }), false));

// ── TR.player.clampWindow ────────────────────────────────────
console.log('TR.player.clampWindow');
test('missing value → fallback',   () => { assert.equal(TR.player.clampWindow(null, 5), 5); assert.equal(TR.player.clampWindow(undefined, 3), 3); assert.equal(TR.player.clampWindow('', 5), 5); });
test('non-numeric → fallback',     () => { assert.equal(TR.player.clampWindow('abc', 5), 5); assert.equal(TR.player.clampWindow(NaN, 3), 3); assert.equal(TR.player.clampWindow({}, 5), 5); });
test('plain values pass through',  () => { assert.equal(TR.player.clampWindow(0, 5), 0); assert.equal(TR.player.clampWindow(7, 5), 7); assert.equal(TR.player.clampWindow('12', 5), 12); });
test('clamped at both ends',       () => { assert.equal(TR.player.clampWindow(-4, 5), 0); assert.equal(TR.player.clampWindow(500, 5), 120); assert.equal(TR.player.clampWindow(120, 5), 120); });
test('floats truncate',            () => { assert.equal(TR.player.clampWindow(7.9, 5), 7); assert.equal(TR.player.clampWindow('3.5', 5), 3); });
test('fallback used as-is',        () => assert.equal(TR.player.clampWindow(undefined, 0), 0));

// ── TR.player.seekLink ───────────────────────────────────────
console.log('TR.player.seekLink');
test('youtube URL',         () => assert.equal(TR.player.seekLink({ youtubelink: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' }, 65), 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=60s'));
test('youtube 5s lookback', () => assert.equal(TR.player.seekLink({ youtubelink: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' }, 3),  'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0s'));
test('no provider → empty', () => assert.equal(TR.player.seekLink({}, 10), ''));
test('null meta → empty',   () => assert.equal(TR.player.seekLink(null, 10), ''));

// ── Possession-chain consistency ──────────────────────────────
// Helper: build an annotations array from a compact spec [time, type, name, possessionOwner].
let _nextId = 1;
function mkAnns(rows) {
  return rows.map(([time, type, name, possessionOwner]) => ({
    id: _nextId++, time, type, name: name || '', possessionOwner: possessionOwner || '',
    actionOwner: '', comment: '', timeStr: '',
  }));
}
function ownersById(anns) {
  return Object.fromEntries(anns.map(a => [a.id, a.possessionOwner]));
}

console.log('TR.computeExpectedOwners');

test('empty input → empty map', () => {
  assert.equal(TR.computeExpectedOwners([]).size, 0);
});

test('non-array input → empty map', () => {
  assert.equal(TR.computeExpectedOwners(null).size, 0);
  assert.equal(TR.computeExpectedOwners(undefined).size, 0);
});

test('events with no possessionOwner anywhere → empty map', () => {
  const anns = mkAnns([[0, 'Try', 'Scoop', ''], [10, 'Try', 'Scoop', '']]);
  assert.equal(TR.computeExpectedOwners(anns).size, 0);
});

test('first event is seed → its expected equals itself', () => {
  const anns = mkAnns([[0, 'Game Event', 'Game Start', 'Team 1']]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[0].id), 'Team 1');
});

test('Try flips possession in chain', () => {
  const anns = mkAnns([
    [0, 'Game Event', 'Game Start', 'Team 1'],
    [10, 'Try', 'Scoop',            'Team 1'],
    [20, 'Game Event', 'Ball Live', 'Team 2'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[0].id), 'Team 1');
  assert.equal(exp.get(anns[1].id), 'Team 1');  // Try is by Team 1, possession was Team 1
  assert.equal(exp.get(anns[2].id), 'Team 2');  // After Try, kickoff to Team 2
});

test('Penalty Defence does NOT flip possession', () => {
  const anns = mkAnns([
    [0,  'Game Event',      'Game Start', 'Team 1'],
    [10, 'Penalty Defence', 'Offside',    'Team 1'],
    [20, 'Try',             'Scoop',      'Team 1'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[2].id), 'Team 1');  // Team 1 still has possession after pen defence
});

test('6 Again keeps possession with attacking team', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Turnover',   '6 Again',    'Team 1'],
    [20, 'Try',        'Scoop',      'Team 1'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[2].id), 'Team 1');  // 6 Again does not flip
});

test('Turnover (not 6 Again) flips possession', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Turnover',   '6th Touch',  'Team 1'],
    [20, 'Try',        'Scoop',      'Team 2'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[2].id), 'Team 2');
});

test('Penalty Attack flips possession (attacking team gave away penalty)', () => {
  const anns = mkAnns([
    [0,  'Game Event',     'Game Start',   'Team 1'],
    [10, 'Penalty Attack', 'Forward Pass', 'Team 1'],
    [20, 'Game Event',     'Ball Live',    'Team 2'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[2].id), 'Team 2');
});

test('Ball Live does NOT flip possession', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Game Event', 'Ball Live',  'Team 1'],
    [20, 'Penalty Defence', 'Offside', 'Team 1'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[1].id), 'Team 1');
  assert.equal(exp.get(anns[2].id), 'Team 1');
});

test('To Review events are skipped (do not break the chain)', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'To Review',  '',           'Team 2'],  // skipped entirely
    [20, 'Try',        'Scoop',      'Team 1'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.has(anns[1].id), false);      // To Review has no expected
  assert.equal(exp.get(anns[2].id), 'Team 1');   // chain ignores To Review
});

test('chain advances from EXPECTED not RECORDED (single wrong override does not infect downstream)', () => {
  // Seed Team 1, then Try Team 1 → expected Team 2 next.
  // But Ball Live recorded as Team 1 (wrong).
  // The Try after that recorded as Team 2 — should still match expected because chain uses expected.
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Try',        'Scoop',      'Team 1'],
    [20, 'Game Event', 'Ball Live',  'Team 1'],   // wrong: should be Team 2
    [30, 'Game Event', 'Ball Live',  'Team 2'],   // correct under expected chain
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[2].id), 'Team 2');    // expected says Team 2 for the wrong row
  assert.equal(exp.get(anns[3].id), 'Team 2');    // chain advanced from expected, still Team 2
});

test('seed is the first event with a possessionOwner (earlier blanks are skipped)', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', ''],         // no owner — chain not yet seeded
    [10, 'Try',        'Scoop',      'Team 2'],   // becomes the seed
    [20, 'Game Event', 'Ball Live',  'Team 1'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.has(anns[0].id), false);
  assert.equal(exp.get(anns[1].id), 'Team 2');
  assert.equal(exp.get(anns[2].id), 'Team 1');    // Try Team 2 → Team 1 next
});

test('events out of chronological order get sorted before chaining', () => {
  const anns = mkAnns([
    [20, 'Try',        'Scoop',      'Team 1'],   // listed first but later in time
    [0,  'Game Event', 'Game Start', 'Team 1'],   // listed second but earlier
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[1].id), 'Team 1');    // Game Start is the seed
  assert.equal(exp.get(anns[0].id), 'Team 1');    // Try at t=20 expects Team 1
});

test('Game Start re-seeds the chain (any team can start a half)', () => {
  // First half ends with Team 1 in possession; Team 1 also starts the second
  // half. Without the per-half re-seed the chain would expect Team 1 anyway —
  // so use a scenario where the continuous chain would expect Team 2.
  const anns = mkAnns([
    [0,   'Game Event', 'Game Start', 'Team 1'],
    [10,  'Try',        'Scoop',      'Team 1'],   // chain → Team 2
    [600, 'Game Event', 'Game End',   'Team 2'],
    [700, 'Game Event', 'Game Start', 'Team 1'],   // 2nd half: Team 1 starts — must not be flagged
    [710, 'Game Event', 'Ball Live',  'Team 1'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.get(anns[3].id), 'Team 1');   // expected equals itself (new seed)
  assert.equal(exp.get(anns[4].id), 'Team 1');   // chain continues from the new seed
});

test('Game Start without an owner leaves the new half unseeded until the next owned event', () => {
  const anns = mkAnns([
    [0,   'Game Event', 'Game Start', 'Team 1'],
    [600, 'Game Event', 'Game End',   'Team 1'],
    [700, 'Game Event', 'Game Start', ''],         // owner unknown — no expectation
    [710, 'Try',        'Scoop',      'Team 2'],   // becomes the second-half seed
    [720, 'Game Event', 'Ball Live',  'Team 1'],
  ]);
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.has(anns[2].id), false);
  assert.equal(exp.get(anns[3].id), 'Team 2');
  assert.equal(exp.get(anns[4].id), 'Team 1');    // Try Team 2 → Team 1 next
});

test('null/empty annotations are tolerated', () => {
  const anns = [null, ...mkAnns([[0, 'Try', 'Scoop', 'Team 1']]), undefined];
  const exp = TR.computeExpectedOwners(anns);
  assert.equal(exp.size, 1);
});


console.log('TR.getInconsistentIds');

test('all consistent → empty set', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Try',        'Scoop',      'Team 1'],
    [20, 'Game Event', 'Ball Live',  'Team 2'],
  ]);
  assert.equal(TR.getInconsistentIds(anns).size, 0);
});

test('detects Ball Live with wrong possessionOwner', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Try',        'Scoop',      'Team 1'],
    [20, 'Game Event', 'Ball Live',  'Team 1'],   // wrong: chain says Team 2
  ]);
  const bad = TR.getInconsistentIds(anns);
  assert.equal(bad.size, 1);
  assert.ok(bad.has(anns[2].id));
});

test('detects Try with wrong possessionOwner', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Try',        'Scoop',      'Team 2'],   // wrong: nobody handed it to Team 2
  ]);
  const bad = TR.getInconsistentIds(anns);
  assert.equal(bad.size, 1);
  assert.ok(bad.has(anns[1].id));
});

test('seed event itself is never flagged', () => {
  const anns = mkAnns([
    [0, 'Game Event', 'Game Start', 'Team 1'],
  ]);
  assert.equal(TR.getInconsistentIds(anns).size, 0);
});

test('only the bad row is flagged (no cascade) when downstream re-aligns to expected', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Try',        'Scoop',      'Team 1'],
    [20, 'Game Event', 'Ball Live',  'Team 1'],   // wrong (expected Team 2)
    [30, 'Game Event', 'Ball Live',  'Team 2'],   // matches expected chain
  ]);
  const bad = TR.getInconsistentIds(anns);
  assert.equal(bad.size, 1);
  assert.ok(bad.has(anns[2].id));
  assert.equal(bad.has(anns[3].id), false);
});

test('empty possessionOwner after the seed is flagged', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Game Event', 'Ball Live',  ''],         // missing owner — expected Team 1
  ]);
  const bad = TR.getInconsistentIds(anns);
  assert.equal(bad.size, 1);
  assert.ok(bad.has(anns[1].id));
});

test('multiple inconsistencies in different parts of the game', () => {
  const anns = mkAnns([
    [0,  'Game Event',     'Game Start',   'Team 1'],
    [10, 'Try',            'Scoop',        'Team 1'],   // ok
    [20, 'Game Event',     'Ball Live',    'Team 1'],   // bad (expected Team 2)
    [30, 'Penalty Attack', 'Forward Pass', 'Team 2'],   // ok (uses expected Team 2)
    [40, 'Game Event',     'Ball Live',    'Team 2'],   // bad (expected Team 1)
  ]);
  const bad = TR.getInconsistentIds(anns);
  assert.equal(bad.size, 2);
  assert.ok(bad.has(anns[2].id));
  assert.ok(bad.has(anns[4].id));
});

test('Penalty Defence chain (possession stays with attacker)', () => {
  const anns = mkAnns([
    [0,  'Game Event',      'Game Start', 'Team 1'],
    [10, 'Penalty Defence', 'Offside',    'Team 1'],   // Team 1 keeps the ball
    [20, 'Try',             'Scoop',      'Team 1'],   // ok
    [30, 'Game Event',      'Ball Live',  'Team 2'],   // ok
  ]);
  assert.equal(TR.getInconsistentIds(anns).size, 0);
});

test('second-half Game Start is never flagged regardless of which team starts', () => {
  const anns = mkAnns([
    [0,   'Game Event', 'Game Start', 'Team 1'],
    [10,  'Try',        'Scoop',      'Team 1'],   // continuous chain → Team 2
    [600, 'Game Event', 'Game End',   'Team 2'],
    [700, 'Game Event', 'Game Start', 'Team 1'],   // Team 1 starts again — valid
    [710, 'Try',        'Scoop',      'Team 2'],   // bad: second-half chain says Team 1
  ]);
  const bad = TR.getInconsistentIds(anns);
  assert.equal(bad.size, 1);
  assert.ok(bad.has(anns[4].id));                  // only the Try, not the Game Start
});

test('field-annotator scenario: manual possession swap with no tagged event', () => {
  // User manually clicked the possession button to swap teams without
  // tagging a Turnover. The next event is recorded with the new team and
  // should be flagged because no chain-advancing event explains the swap.
  const anns = mkAnns([
    [0,  'Game Event',      'Game Start', 'Team 1'],
    [10, 'Penalty Defence', 'Offside',    'Team 1'],   // possession stays Team 1
    [20, 'Try',             'Scoop',      'Team 2'],   // bad: chain says Team 1
  ]);
  const bad = TR.getInconsistentIds(anns);
  assert.equal(bad.size, 1);
  assert.ok(bad.has(anns[2].id));
});


console.log('TR.applyConsistencyFix');

test('all-consistent input is unchanged and returns 0', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Try',        'Scoop',      'Team 1'],
    [20, 'Game Event', 'Ball Live',  'Team 2'],
  ]);
  const before = ownersById(anns);
  const changed = TR.applyConsistencyFix(anns);
  assert.equal(changed, 0);
  assert.deepEqual(ownersById(anns), before);
});

test('rewrites Ball Live override to chain value', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Try',        'Scoop',      'Team 1'],
    [20, 'Game Event', 'Ball Live',  'Team 1'],   // wrong
  ]);
  const changed = TR.applyConsistencyFix(anns);
  assert.equal(changed, 1);
  assert.equal(anns[2].possessionOwner, 'Team 2');
});

test('rewrites a wrong Try (cascade-correcting downstream too)', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'Try',        'Scoop',      'Team 2'],   // wrong
    [20, 'Game Event', 'Ball Live',  'Team 1'],   // wrong (would-be after wrong Try)
  ]);
  TR.applyConsistencyFix(anns);
  assert.equal(anns[1].possessionOwner, 'Team 1');  // fixed
  assert.equal(anns[2].possessionOwner, 'Team 2');  // fixed (after Try Team 1 → Team 2)
});

test('preserves the seed event (first with a recorded owner)', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 2'],   // seed
    [10, 'Try',        'Scoop',      'Team 1'],   // wrong relative to seed
  ]);
  TR.applyConsistencyFix(anns);
  assert.equal(anns[0].possessionOwner, 'Team 2');  // seed kept
  assert.equal(anns[1].possessionOwner, 'Team 2');  // fixed (expected Team 2 after Game Start)
});

test('fix preserves each half\'s Game Start owner and rewrites from it', () => {
  const anns = mkAnns([
    [0,   'Game Event', 'Game Start', 'Team 1'],
    [10,  'Try',        'Scoop',      'Team 1'],   // continuous chain → Team 2
    [600, 'Game Event', 'Game End',   'Team 2'],
    [700, 'Game Event', 'Game Start', 'Team 1'],   // 2nd-half seed — must be preserved
    [710, 'Game Event', 'Ball Live',  'Team 2'],   // wrong: should follow new seed (Team 1)
  ]);
  const changed = TR.applyConsistencyFix(anns);
  assert.equal(anns[3].possessionOwner, 'Team 1');  // seed kept, not rewritten to Team 2
  assert.equal(anns[4].possessionOwner, 'Team 1');  // fixed against the new seed
  assert.equal(changed, 1);
});

test('actionOwner is recomputed after fixing possessionOwner', () => {
  const anns = mkAnns([
    [0,  'Game Event',      'Game Start', 'Team 1'],
    [10, 'Penalty Defence', 'Offside',    'Team 1'],   // expected Team 1; actionOwner should be Team 2
  ]);
  anns[1].actionOwner = 'Team 1';  // stale
  TR.applyConsistencyFix(anns);
  assert.equal(anns[1].possessionOwner, 'Team 1');
  assert.equal(anns[1].actionOwner, 'Team 2');  // defending team's foul
});

test('To Review rows are left alone', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', 'Team 1'],
    [10, 'To Review',  '',           'Team 2'],
    [20, 'Try',        'Scoop',      'Team 2'],   // wrong; should be Team 1 (To Review doesn't break chain)
  ]);
  TR.applyConsistencyFix(anns);
  assert.equal(anns[1].possessionOwner, 'Team 2');  // To Review untouched
  assert.equal(anns[2].possessionOwner, 'Team 1');  // fixed
});

test('after applyConsistencyFix, getInconsistentIds returns empty', () => {
  const anns = mkAnns([
    [0,  'Game Event',     'Game Start',   'Team 1'],
    [10, 'Try',            'Scoop',        'Team 1'],
    [20, 'Game Event',     'Ball Live',    'Team 1'],   // wrong
    [30, 'Penalty Attack', 'Forward Pass', 'Team 1'],   // wrong
    [40, 'Game Event',     'Ball Live',    'Team 2'],   // wrong after chain reapply
  ]);
  TR.applyConsistencyFix(anns);
  assert.equal(TR.getInconsistentIds(anns).size, 0);
});

test('applyConsistencyFix on input with no seed is a no-op', () => {
  const anns = mkAnns([
    [0,  'Game Event', 'Game Start', ''],
    [10, 'Try',        'Scoop',      ''],
  ]);
  const changed = TR.applyConsistencyFix(anns);
  assert.equal(changed, 0);
  assert.equal(anns[0].possessionOwner, '');
  assert.equal(anns[1].possessionOwner, '');
});

test('long realistic game: 8 events stay aligned end-to-end', () => {
  const anns = mkAnns([
    [0,    'Game Event',      'Game Start',   'Team 1'],
    [30,   'Penalty Defence', 'Offside',      'Team 1'],
    [60,   'Try',             'Scoop',        'Team 1'],   // Team 1 scores, kickoff to Team 2
    [90,   'Game Event',      'Ball Live',    'Team 2'],
    [120,  'Penalty Attack',  'Forward Pass', 'Team 2'],   // Team 2 loses possession
    [150,  'Game Event',      'Ball Live',    'Team 1'],
    [180,  'Turnover',        '6 Again',      'Team 1'],   // 6 Again — Team 1 keeps it
    [210,  'Try',             'Scoop',        'Team 1'],
  ]);
  assert.equal(TR.getInconsistentIds(anns).size, 0);
});


// ── Offline mode ──────────────────────────────────────────────
console.log('TR offline mode');

test('off by default', () => {
  ctx.localStorage.clear();
  assert.equal(TR.isOfflineMode(), false);
});

test('enter / exit toggles the flag', () => {
  ctx.localStorage.clear();
  TR.enterOfflineMode();
  assert.equal(ctx.localStorage.getItem('trl2_offline'), '1');
  assert.equal(TR.isOfflineMode(), true);
  TR.exitOfflineMode();
  assert.equal(ctx.localStorage.getItem('trl2_offline'), null);
  assert.equal(TR.isOfflineMode(), false);
});

test('a real session wins over the offline flag', () => {
  ctx.localStorage.clear();
  TR.enterOfflineMode();
  const realSecret = TR.secret;
  TR.secret = () => 'm30-staff';
  assert.equal(TR.isOfflineMode(), false, 'signed in — not offline mode');
  TR.secret = realSecret;
});

test('saveSession leaves offline mode', () => {
  ctx.localStorage.clear();
  TR.enterOfflineMode();
  TR.saveSession('m30-staff', 'staff', 'm30');
  assert.equal(ctx.localStorage.getItem('trl2_offline'), null);
});

test('exiting is idempotent', () => {
  ctx.localStorage.clear();
  TR.exitOfflineMode();
  TR.exitOfflineMode();
  assert.equal(TR.isOfflineMode(), false);
});

// ── TR.FieldGames ─────────────────────────────────────────────
console.log('TR.FieldGames');

const FG = TR.FieldGames;
const resetStore = () => ctx.localStorage.clear();

// Minimal annotation shaped like the ones the field annotator writes.
const ann = (time, type, name, actionOwner) =>
  ({ id: time, type, name, possessionOwner: actionOwner, actionOwner, comment: '', time });

test('create registers the game in the index', () => {
  resetStore();
  const g = FG.create({ team1: 'France', team2: 'England' });
  assert.deepEqual([...FG.ids()], [g.id]);
  assert.equal(FG.get(g.id).meta.team1, 'France');
  assert.equal(FG.get(g.id).status, 'active');
});

test('create fills missing meta keys', () => {
  resetStore();
  const g = FG.create({ team1: 'France' });
  assert.equal(g.meta.team2, '');
  assert.equal(g.meta.youtubelink, '');
});

test('games are isolated from each other', () => {
  resetStore();
  const a = FG.create({ team1: 'A' });
  const b = FG.create({ team1: 'B' });
  a.annotations.push(ann(0, 'Try', 'Scoop', 'Team 1'));
  FG.save(a);
  assert.equal(FG.get(a.id).annotations.length, 1);
  assert.equal(FG.get(b.id).annotations.length, 0);
  assert.equal(FG.ids().length, 2);
});

test('save bumps the revision; touch:false does not', () => {
  resetStore();
  const g = FG.create({});
  assert.equal(g.revision, 0);
  FG.save(g);
  assert.equal(g.revision, 1);
  FG.save(g, { touch: false });
  assert.equal(g.revision, 1);
});

test('a new game is dirty and not uploaded', () => {
  resetStore();
  const g = FG.create({});
  assert.equal(FG.isDirty(g), true);
  assert.equal(FG.isUploaded(g), false);
  assert.equal(FG.isSynced(g), false);
});

test('markSynced clears dirty; a later edit sets it again', () => {
  resetStore();
  const g = FG.create({});
  g.annotations.push(ann(0, 'Try', 'Scoop', 'Team 1'));
  FG.save(g);
  FG.markSynced(g, '2026_m30_cup_a_b');
  assert.equal(FG.isDirty(g), false);
  assert.equal(FG.isSynced(g), true);
  assert.equal(FG.get(g.id).sync.sheetName, '2026_m30_cup_a_b');

  // An event-type correction keeps the count but must still re-push.
  g.annotations[0].type = 'Turnover';
  FG.save(g);
  assert.equal(FG.isDirty(g), true);
  assert.equal(FG.isUploaded(g), true, 'still uploaded once, just out of date');
  assert.equal(FG.isSynced(g), false);
});

test('markSynced honours an explicit revision', () => {
  resetStore();
  const g = FG.create({});
  FG.save(g);                       // revision 1 — what a push would send
  const sent = g.revision;
  FG.save(g);                       // revision 2 — tagged while in flight
  FG.markSynced(g, 'tab', sent);
  assert.equal(FG.isDirty(g), true, 'the in-flight event still needs pushing');
  assert.equal(FG.isUploaded(g), true);
});

test('remove drops the record and the index entry', () => {
  resetStore();
  const a = FG.create({ team1: 'A' });
  const b = FG.create({ team1: 'B' });
  FG.remove(a.id);
  assert.deepEqual([...FG.ids()], [b.id]);
  assert.equal(FG.get(a.id), null);
});

test('removeSynced spares unsynced and dirty games', () => {
  resetStore();
  const clean = FG.create({ team1: 'clean' });
  FG.markSynced(clean, 'tab-clean');
  const dirty = FG.create({ team1: 'dirty' });
  FG.markSynced(dirty, 'tab-dirty');
  dirty.annotations.push(ann(0, 'Try', 'Scoop', 'Team 1'));
  FG.save(dirty);
  const never = FG.create({ team1: 'never' });

  assert.equal(FG.removeSynced(), 1);
  const left = FG.ids();
  assert.equal(left.length, 2);
  assert.ok(left.includes(dirty.id) && left.includes(never.id));
});

test('list is ordered by most recently edited', () => {
  resetStore();
  const a = FG.create({ team1: 'A' });
  const b = FG.create({ team1: 'B' });
  a.updatedAt = 5000; FG.save(a, { touch: false });
  b.updatedAt = 9000; FG.save(b, { touch: false });
  assert.deepEqual([...FG.list().map(g => g.meta.team1)], ['B', 'A']);
});

test('get returns null for an unknown id', () => {
  resetStore();
  assert.equal(FG.get('g_nope'), null);
});

test('normalize repairs a record from an older build', () => {
  resetStore();
  const g = FG.create({ team1: 'A' });
  // Simulate a record written before status/sync/revision existed.
  ctx.localStorage.setItem(FG.gameKey(g.id), JSON.stringify({ id: g.id, meta: { team1: 'A' } }));
  const back = FG.get(g.id);
  assert.equal(back.annotations.length, 0);
  assert.equal(back.possession, 'Team 1');
  assert.equal(back.status, 'active');
  assert.equal(back.revision, 0);
  assert.equal(back.sync.pushedAt, null);
  assert.equal(typeof back.createdAt, 'number');
});

test('collisions finds other games sharing a tab name', () => {
  resetStore();
  const nameOf = g => [g.meta.year, g.meta.team1, g.meta.team2].filter(Boolean).join('_');
  const a = FG.create({ year: '2026', team1: 'fra', team2: 'eng' });
  const b = FG.create({ year: '2026', team1: 'fra', team2: 'eng' });
  const c = FG.create({ year: '2026', team1: 'fra', team2: 'wal' });
  assert.deepEqual([...FG.collisions(a, nameOf).map(g => g.id)], [b.id]);
  assert.equal(FG.collisions(c, nameOf).length, 0);
  // A game with no metadata yet can't collide with anything.
  assert.equal(FG.collisions(FG.create({}), nameOf).length, 0);
  // An explicit name wins over the record's stored metadata, so an unsaved
  // keystroke in the Setup panel is reflected immediately.
  assert.equal(FG.collisions(c, nameOf, '2026_fra_eng').length, 2);
  assert.equal(FG.collisions(a, nameOf, '2026_fra_wal').length, 1);
  assert.equal(FG.collisions(a, nameOf, '').length, 0);
});

test('summarize reports score, status and duration', () => {
  resetStore();
  const g = FG.create({ team1: 'France', team2: 'England', year: '2026', division: 'M30' });
  g.annotations = [
    ann(0,   'Game Event', 'Game Start', 'Team 1'),
    ann(60,  'Try',        'Scoop',      'Team 1'),
    ann(120, 'Try',        'Scoop',      'Team 2'),
    ann(180, 'Try',        'Scoop',      'Team 1'),
    ann(300, 'Game Event', 'Game End',   'Team 1'),
  ];
  g.status = 'finished';
  FG.save(g);
  const s = FG.summarize(g);
  assert.equal(s.score1, 2);
  assert.equal(s.score2, 1);
  assert.equal(s.events, 5);
  assert.equal(s.duration, 300);
  assert.equal(s.status, 'finished');
  assert.equal(s.subtitle, '2026 · M30');
  assert.equal(s.titled, true);
});

test('summarize distinguishes new / running / paused', () => {
  resetStore();
  const g = FG.create({});
  assert.equal(FG.summarize(g).status, 'new');

  g.wallStart = Date.now();
  g.annotations = [ann(0, 'Game Event', 'Game Start', 'Team 1')];
  assert.equal(FG.summarize(g).status, 'running');

  g.annotations.push(ann(600, 'Game Event', 'Game End', 'Team 1'));
  assert.equal(FG.summarize(g).status, 'paused');

  g.annotations.push(ann(700, 'Game Event', 'Game Start', 'Team 1'));
  assert.equal(FG.summarize(g).status, 'running', 'second half restarts the clock');
});

test('summarize falls back to placeholder team names', () => {
  resetStore();
  const s = FG.summarize(FG.create({}));
  assert.equal(s.team1, 'Team 1');
  assert.equal(s.team2, 'Team 2');
  assert.equal(s.titled, false);
  assert.equal(s.duration, 0);
});

test('migrateLegacy imports the old single session once', () => {
  resetStore();
  ctx.localStorage.setItem('fieldAnnotatorSession', JSON.stringify({
    annotations: [ann(0, 'Game Event', 'Game Start', 'Team 1'), ann(30, 'Try', 'Scoop', 'Team 1')],
    possession: 'Team 2',
    wallStart: 1234567890,
    teamsSwapped: true,
    creatorToken: 'tok-1',
    meta: { team1: 'France', team2: 'England' },
  }));
  const g = FG.migrateLegacy();
  assert.ok(g);
  assert.equal(g.annotations.length, 2);
  assert.equal(g.possession, 'Team 2');
  assert.equal(g.wallStart, 1234567890);
  assert.equal(g.teamsSwapped, true);
  assert.equal(g.creatorToken, 'tok-1');
  assert.equal(g.meta.team1, 'France');
  assert.equal(FG.isDirty(g), true, 'imported games are assumed un-pushed');
  assert.equal(ctx.localStorage.getItem('fieldAnnotatorSession'), null);
  assert.equal(FG.migrateLegacy(), null, 'second run is a no-op');
});

test('migrateLegacy discards an empty session', () => {
  resetStore();
  ctx.localStorage.setItem('fieldAnnotatorSession', JSON.stringify({ annotations: [], meta: {} }));
  assert.equal(FG.migrateLegacy(), null);
  assert.equal(FG.ids().length, 0);
  assert.equal(ctx.localStorage.getItem('fieldAnnotatorSession'), null);
});

// ── TR.strikeMoveStats ────────────────────────────────────────
console.log('TR.strikeMoveStats');
const ev = (type, name, strikeMove, actionOwner) => ({ type, name, strikeMove, actionOwner: actionOwner || 'Team 1' });
// TR.strikeMoveStats executes inside the vm context loaded above, so the
// plain objects/arrays it returns carry that context's Object/Array
// prototypes. assert/strict's deepEqual is a strict deepStrictEqual that
// checks prototype identity, so it rejects those results against this
// file's own object/array literals even when every field matches —
// structuredClone re-realizes the value in this (the main) realm first.
const stats = (events) => structuredClone(TR.strikeMoveStats(events));

test('a tagged touch counts as a failure of that move', () => {
  // The distortion this exists to fix: without touches, a move that is run
  // constantly and almost never breaks the line reads as a perfect one.
  const tries  = [ev('Try', '32 - Cut', ''), ev('Try', '32 - Cut', '')];
  const before = stats(tries);
  assert.equal(before.moves[0].rate, 1);
  const after = stats(tries.concat(
    Array.from({ length: 18 }, () => ev('Touch', 'Touch 3', '32 - Cut'))));
  assert.equal(after.moves[0].tries, 2);
  assert.equal(after.moves[0].attempts, 20);
  assert.equal(after.moves[0].rate, 0.1);
  assert.equal(after.moves[0].rateKnown, true);
  assert.equal(after.ratesMeaningful, true);
});

test('untagged touches never dilute coverage', () => {
  // Touches 1-3 are meant to be touched, so an untagged touch is ordinary play,
  // not a missed tag. Were they counted, switching touch upload on would drop
  // coverage through the floor overnight.
  const base = [ev('Try', '32', ''), ev('Touch', 'Touch 2', '32')];
  const noisy = base.concat(Array.from({ length: 500 }, () => ev('Touch', 'Touch 1', '')));
  assert.deepEqual(stats(noisy).coverage, stats(base).coverage);
  assert.equal(stats(noisy).moves[0].attempts, 2);
});

test('counting no longer asks whether the ball changed hands', () => {
  // None of these three ends the attack, and all three count. Whether the ball
  // changed hands turned out to be the wrong question: a move was called and,
  // try aside, it did not score.
  ['Touch', 'Penalty Defence', 'Turnover'].forEach(t => {
    const name = t === 'Touch' ? 'Touch 4' : t === 'Turnover' ? '6 Again' : 'Offside';
    assert.equal(TR.isAttackEnd(t, name), false, t + ' should not end the attack');
    assert.equal(stats([ev(t, name, '32')]).moves.length, 1, t + ' should still count');
  });
  // Only these are never attempts, plus an untagged touch.
  assert.deepEqual(stats([ev('Game Event', 'Game Start', '32')]).moves, []);
  assert.deepEqual(stats([ev('To Review', '', '32')]).moves, []);
  assert.deepEqual(stats([ev('Touch', 'Touch 2', '')]).moves, []);
});


test('empty input', () => {
  const s = stats([]);
  assert.deepEqual(s.moves, []);
  // Per-side breakdown is always present, even at zero (Phase 2 design: coverage
  // is reported per side as well as combined, never just combined).
  assert.deepEqual(s.coverage, {
    tagged: 0, total: 0, pct: 0,
    tries: { tagged: 0, total: 0, pct: 0 },
    fails: { tagged: 0, total: 0, pct: 0 },
  });
  assert.equal(s.topByTries, null);
  assert.equal(s.topByRate, null);
  assert.equal(s.ratesMeaningful, false, 'no events at all - nothing to be meaningful about');
});

test('null input is tolerated', () => assert.equal(TR.strikeMoveStats(null).moves.length, 0));

test('ratesMeaningful is false when tries are tagged but no failure ever is', () => {
  // Every existing game: tries carry the move via Name, no failure ever does.
  const s = stats([
    ev('Try', '32 - Cut', ''), ev('Try', '32 - Cut', ''),
    ev('Try', '23 - Scoop', ''),
  ]);
  assert.equal(s.moves.length, 2, 'moves is still populated');
  s.moves.forEach(m => assert.equal(m.rate, 1, 'every rate computes to 1 by construction'));
  assert.equal(s.ratesMeaningful, false);
});

test('ratesMeaningful is true once at least one failure is tagged', () => {
  const s = stats([
    ev('Try', '32 - Cut', ''),
    ev('Turnover', 'Ball Down', '32 - Cut'),
  ]);
  assert.equal(s.ratesMeaningful, true);
});

test('ratesMeaningful is a dataset-level gate, not per-move', () => {
  // The tagged failure sits on a different move from the tries - still true,
  // because the gate asks "can any rate here be trusted", not "is this move's".
  const s = stats([
    ev('Try', '32 - Cut', ''), ev('Try', '32 - Cut', ''),
    ev('Turnover', 'Ball Down', '23 - Scoop'),
  ]);
  assert.equal(s.ratesMeaningful, true);
});

test('ratesMeaningful is false when failures exist but none are tagged', () => {
  const s = stats([
    ev('Try', '32 - Cut', ''),
    ev('Turnover', 'Ball Down', ''),
    ev('Penalty Attack', 'Forward Pass', ''),
  ]);
  assert.equal(s.moves.length, 1, 'the tagged try still gets a row');
  assert.equal(s.ratesMeaningful, false);
});

test('excluded moves are Other and Interception',
  // TR.EXCLUDED_MOVES is a vm-context array literal too — same realm fix.
  () => assert.deepEqual(structuredClone(TR.EXCLUDED_MOVES), ['Other', 'Interception']));
test('excluded moves are real entries of the picker list',
  () => TR.EXCLUDED_MOVES.forEach(m => assert.ok(TR.STRIKE_MOVES.includes(m), m)));

test('a try and a turnover on the same move', () => {
  const s = stats([
    ev('Try', '32 - Cut', ''),
    ev('Turnover', 'Ball Down', '32 - Cut'),
  ]);
  assert.equal(s.moves.length, 1);
  assert.deepEqual(s.moves[0], { move: '32 - Cut', tries: 1, fails: 1, attempts: 2, rate: 0.5, rateKnown: true });
});

test('a pen attack counts as a failure', () => {
  const s = stats([ev('Penalty Attack', 'Forward Pass', '23')]);
  assert.deepEqual(s.moves[0], { move: '23', tries: 0, fails: 1, attempts: 1, rate: 0, rateKnown: true });
});

test('coverage counts attack-ends only', () => {
  const s = stats([
    ev('Try', 'Scoop', ''),                    // attack end, tagged (name is the move)
    ev('Turnover', 'Ball Down', '32'),         // attack end, tagged
    ev('Turnover', 'Ball Down', ''),           // attack end, untagged
    ev('Turnover', '6 Again', '32'),           // counts: a fresh count, still no try
    ev('Penalty Defence', 'Offside', '32'),    // counts: an attempt that didn't score
    ev('Game Event', 'Game Start', ''),        // never an attempt
  ]);
  // 1 Try (tagged) + 4 fail-side attempts: 3 turnovers including the 6 Again
  // (2 tagged) and the defensive penalty (tagged). Only Game Event is out.
  assert.deepEqual(s.coverage, {
    tagged: 4, total: 5, pct: 4 / 5,
    tries: { tagged: 1, total: 1, pct: 1 },
    fails: { tagged: 3, total: 4, pct: 3 / 4 },
  });
});

test('untagged attempts are excluded from every move row', () => {
  const s = stats([
    ev('Try', '32', ''),
    ev('Turnover', 'Ball Down', ''),
  ]);
  assert.equal(s.moves.length, 1);
  assert.equal(s.moves[0].attempts, 1);
});

test('Other and Interception never get a row', () => {
  const s = stats([
    ev('Try', 'Other', ''),
    ev('Turnover', 'Ball Down', 'Interception'),
  ]);
  assert.deepEqual(s.moves, []);
  assert.equal(s.coverage.tagged, 0);
  assert.equal(s.coverage.total, 2);
  assert.equal(s.topByTries, null);
  assert.equal(s.topByRate, null);
});

test('a Simple Mode game cannot top the board on Other', () => {
  const s = stats([
    ev('Try', 'Other', ''), ev('Try', 'Other', ''), ev('Try', 'Other', ''),
    ev('Try', '32', ''), ev('Turnover', 'Ball Down', '32'),
  ]);
  assert.equal(s.topByTries.move, '32');
  assert.equal(s.topByRate.move, '32');
});

test('a stale stored move loses to the Name on a Try', () => {
  // What a viewer Name edit leaves behind: Name corrected, column not.
  const s = stats([ev('Try', '32 - Cut', 'Other')]);
  assert.deepEqual(s.moves.map(m => m.move), ['32 - Cut']);
  assert.equal(s.moves[0].tries, 1);
});

test('a 6 Again carries its move like any other turnover', () => {
  // It used to be dropped, on the reading that a 6 Again continued the same
  // attempt. It does not: the count restarts, so the move that was called is
  // over, and it did not score.
  const s = stats([ev('Turnover', '6 Again', '32')]);
  assert.equal(s.moves.length, 1);
  assert.equal(s.moves[0].fails, 1);
  assert.equal(s.coverage.total, 1);
});

test('coverage is reported per side', () => {
  const s = stats([
    ev('Try', '32', ''),                 // try side, tagged
    ev('Try', 'Other', ''),              // try side, excluded -> untagged
    ev('Turnover', 'Ball Down', '32'),   // fail side, tagged
    ev('Turnover', 'Ball Down', ''),     // fail side, untagged
    ev('Penalty Attack', 'Forward Pass', ''),
  ]);
  assert.deepEqual(s.coverage.tries, { tagged: 1, total: 2, pct: 0.5 });
  assert.deepEqual(s.coverage.fails, { tagged: 1, total: 3, pct: 1 / 3 });
  assert.equal(s.coverage.total, 5);
});

test('moves are sorted by rate descending', () => {
  const s = stats([
    ev('Turnover', 'Ball Down', 'Scoop'), ev('Turnover', 'Ball Down', 'Scoop'),
    ev('Try', '32', ''),                  ev('Try', '32', ''),
  ]);
  assert.deepEqual(s.moves.map(m => m.move), ['32', 'Scoop']);
});

test('topByTries ignores the attempts threshold', () => {
  const s = stats([
    ev('Try', '32', ''), ev('Try', '32', ''), ev('Try', '32', ''),
    ev('Turnover', 'Ball Down', '32'), ev('Turnover', 'Ball Down', '32'),
    ev('Try', 'Scoop', ''),
  ]);
  assert.equal(s.topByTries.move, '32');
  assert.equal(s.topByTries.tries, 3);
});

test('topByRate needs MIN_MOVE_ATTEMPTS', () => {
  const s = stats([
    ev('Try', 'Scoop', ''),                                     // 1/1 = 100%, only 1 attempt
    ev('Try', '32', ''), ev('Try', '32', ''),                   // 2/3 = 67%, 3 attempts
    ev('Turnover', 'Ball Down', '32'),
  ]);
  assert.equal(s.topByRate.move, '32');
  assert.equal(s.topByTries.move, '32');
});

test('topByRate is null when nothing clears the threshold', () => {
  const s = stats([ev('Try', 'Scoop', '')]);
  assert.equal(s.topByRate, null);
  assert.equal(s.topByTries.move, 'Scoop');
});

test('topByTries is null when no move ever scored', () => {
  const s = stats([ev('Turnover', 'Ball Down', '32')]);
  assert.equal(s.topByTries, null);
});

// ── rateKnown: a per-move gate, not the dataset-level ratesMeaningful ──
// The moment ANY move anywhere gets a tagged failure, ratesMeaningful flips
// true - but every OTHER move that has never itself failed still computes to
// rate 1 by construction. rateKnown asks the question per move.
test('a move with fails > 0 has rateKnown true', () => {
  const s = stats([
    ev('Try', '32 - Cut', ''),
    ev('Turnover', 'Ball Down', '32 - Cut'),
  ]);
  assert.equal(s.moves[0].rateKnown, true);
});

test('a move with 0 fails has rateKnown false even when ratesMeaningful is true', () => {
  const s = stats([
    ev('Try', '32 - Cut', ''), ev('Try', '32 - Cut', ''),   // 2/2 = 100%, never failed
    ev('Turnover', 'Ball Down', '23 - Scoop'),               // flips the dataset-level gate
  ]);
  assert.equal(s.ratesMeaningful, true, 'sanity: the dataset-level gate is on');
  const cut = s.moves.find(m => m.move === '32 - Cut');
  assert.equal(cut.rate, 1);
  assert.equal(cut.rateKnown, false, 'this move itself has never been seen to fail');
});

test('topByRate skips a 0-fail 100% move for a lower-rated move that has actually failed', () => {
  const s = stats([
    ev('Try', '32 - Cut', ''), ev('Try', '32 - Cut', ''),        // 2/2 = 100%, never failed
    ev('Try', '23 - Scoop', ''), ev('Try', '23 - Scoop', ''),    // 2/3 = 67%, has failed once
    ev('Turnover', 'Ball Down', '23 - Scoop'),
  ]);
  assert.equal(s.topByRate.move, '23 - Scoop', 'the untested 100% must not crown');
  assert.equal(s.topByRate.rate, 2 / 3);
});

test('topByRate is null when every qualifying move has 0 fails', () => {
  const s = stats([
    ev('Try', '32 - Cut', ''), ev('Try', '32 - Cut', ''),
    ev('Try', '23 - Scoop', ''), ev('Try', '23 - Scoop', ''),
  ]);
  assert.equal(s.topByRate, null);
});

test('topByTries still returns the 0-fail move when it leads on volume', () => {
  const s = stats([
    ev('Try', '32 - Cut', ''), ev('Try', '32 - Cut', ''), ev('Try', '32 - Cut', ''), // 3 tries, never failed
    ev('Try', '23 - Scoop', ''),
    ev('Turnover', 'Ball Down', '23 - Scoop'),
  ]);
  assert.equal(s.topByTries.move, '32 - Cut');
  assert.equal(s.topByTries.tries, 3);
  const cut = s.moves.find(m => m.move === '32 - Cut');
  assert.equal(cut.rateKnown, false, 'volume leader can still be untested');
});

test('ties break on attempts then alphabetically', () => {
  const s = stats([
    ev('Try', '23', ''), ev('Turnover', 'Ball Down', '23'),
    ev('Try', '21', ''), ev('Turnover', 'Ball Down', '21'),
  ]);
  assert.deepEqual(s.moves.map(m => m.move), ['21', '23']);
});

// ── TR.filterSummary ──────────────────────────────────────────
console.log('TR.filterSummary');
test('none',        () => { assert.equal(TR.filterSummary([]), 'Any'); });
test('one',         () => { assert.equal(TR.filterSummary(['Try']), 'Try'); });
test('two',         () => { assert.equal(TR.filterSummary(['Try', 'Turnover']), 'Try, Turnover'); });
test('three+',      () => { assert.equal(TR.filterSummary(['Try', 'Turnover', 'Penalty Attack']), '3 selected'); });
test('from a Set',  () => { assert.equal(TR.filterSummary([...new Set(['Try'])]), 'Try'); });
test('numbers',     () => { assert.equal(TR.filterSummary([2025, 2026]), '2025, 2026'); });

// ── TR.optMatch ───────────────────────────────────────────────
console.log('TR.optMatch');
test('empty needle', () => { assert.equal(TR.optMatch('Wiggle', ''), true); assert.equal(TR.optMatch('Wiggle', null), true); assert.equal(TR.optMatch('Wiggle', undefined), true); });
test('hit',          () => { assert.equal(TR.optMatch('Wiggle', 'wig'), true); });
test('case',         () => { assert.equal(TR.optMatch('wiggle', 'WIG'), true); });
test('mid-string',   () => { assert.equal(TR.optMatch('Dummy Switch', 'switch'), true); });
test('miss',         () => { assert.equal(TR.optMatch('Wiggle', 'zzz'), false); });
test('non-string',   () => { assert.equal(TR.optMatch(2026, '26'), true); assert.equal(TR.optMatch(null, 'x'), false); });

// ── TR.evId / TR.evKey / TR.refKey ────────────────────────────
console.log('TR.evId / TR.evKey / TR.refKey');
const EV1 = { game: '2025_m30_cup_fra_eng', time: 134, type: 'Try', name: '32 - Cut' };
test('evId is all four parts', () => { assert.equal(TR.evId(EV1), '2025_m30_cup_fra_eng#134#Try#32 - Cut'); });
test('evKey is game + time',   () => { assert.equal(TR.evKey(EV1), '2025_m30_cup_fra_eng#134'); });
test('refKey trims a ref to its key', () => {
  assert.equal(TR.refKey('2025_m30_cup_fra_eng#134#Try#32 - Cut'), '2025_m30_cup_fra_eng#134');
});
test('refKey survives a rename', () => {
  const renamed = { ...EV1, type: 'Turnover', name: 'Ball Down' };
  assert.equal(TR.refKey(TR.evId(EV1)), TR.evKey(renamed));
});
test('refKey on junk', () => {
  assert.equal(TR.refKey(''), '');
  assert.equal(TR.refKey(null), '');
  assert.equal(TR.refKey('onlygame'), 'onlygame');
});

// ── TR.playlists.resolve ──────────────────────────────────────
console.log('TR.playlists.resolve');
const PROWS = [
  { game: 'g1', time: 10,  type: 'Try',      name: 'Scoop'    },
  { game: 'g1', time: 90,  type: 'Turnover', name: 'Ball Down'},
  { game: 'g2', time: 30,  type: 'Try',      name: '32 - Cut' },
];
// TR.playlists.resolve builds its `events` array and result object with
// vm-context literals, so they carry that context's Object/Array prototypes
// even though this test code calls in from the main realm — same fix as
// TR.strikeMoveStats above: structuredClone before deepEqual.
test('all resolve', () => {
  const r = TR.playlists.resolve(['g1#10#Try#Scoop', 'g2#30#Try#32 - Cut'], PROWS);
  assert.equal(r.missing, 0);
  assert.deepEqual(structuredClone(r.events.map(e => e.game)), ['g1', 'g2']);
});
test('playlist order wins over clock order', () => {
  const r = TR.playlists.resolve(['g2#30#Try#32 - Cut', 'g1#10#Try#Scoop'], PROWS);
  assert.deepEqual(structuredClone(r.events.map(e => e.time)), [30, 10]);
});
test('a renamed event still resolves', () => {
  const r = TR.playlists.resolve(['g1#90#Turnover#6th Touch'], PROWS);
  assert.equal(r.missing, 0);
  assert.equal(r.events[0].name, 'Ball Down');
});
test('missing refs are counted, survivors kept', () => {
  const r = TR.playlists.resolve(['g1#10#Try#Scoop', 'gone#1#Try#x'], PROWS);
  assert.equal(r.missing, 1);
  assert.equal(r.events.length, 1);
});
test('empty inputs', () => {
  assert.deepEqual(structuredClone(TR.playlists.resolve([], PROWS)), { events: [], missing: 0 });
  assert.deepEqual(structuredClone(TR.playlists.resolve(null, null)), { events: [], missing: 0 });
});
test('duplicate game+time: first row in sheet order wins', () => {
  const dupes = [{ game: 'g1', time: 10, type: 'Try', name: 'first' },
                 { game: 'g1', time: 10, type: 'Try', name: 'second' }];
  assert.equal(TR.playlists.resolve(['g1#10#Try#anything'], dupes).events[0].name, 'first');
});

// ── TR.playlists.reorder ──────────────────────────────────────
console.log('TR.playlists.reorder');
// Unlike resolve, reorder builds its return array via .slice()/.splice() on
// the `refs` argument, so the result inherits whichever realm that argument
// came from. R4 is a main-realm array, so most of these compare clean; only
// the null-input case falls through to a vm-context `[]` fallback inside
// playlists.js and needs the same structuredClone fix.
const R4 = ['a', 'b', 'c', 'd'];
test('move down',      () => { assert.deepEqual(TR.playlists.reorder(R4, 0, 2), ['b', 'c', 'a', 'd']); });
test('move up',        () => { assert.deepEqual(TR.playlists.reorder(R4, 3, 1), ['a', 'd', 'b', 'c']); });
test('to the top',     () => { assert.deepEqual(TR.playlists.reorder(R4, 2, 0), ['c', 'a', 'b', 'd']); });
test('to the end',     () => { assert.deepEqual(TR.playlists.reorder(R4, 1, 3), ['a', 'c', 'd', 'b']); });
test('same index',     () => { assert.deepEqual(TR.playlists.reorder(R4, 1, 1), R4); });
test('out of range',   () => { assert.deepEqual(TR.playlists.reorder(R4, -1, 2), R4);
                               assert.deepEqual(TR.playlists.reorder(R4, 0, 9), R4); });
test('does not mutate',() => { TR.playlists.reorder(R4, 0, 3); assert.deepEqual(R4, ['a', 'b', 'c', 'd']); });
test('empty',          () => { assert.deepEqual(TR.playlists.reorder([], 0, 0), []);
                               assert.deepEqual(structuredClone(TR.playlists.reorder(null, 0, 1)), []); });

// ── TR.parseApiResponse ───────────────────────────────────────
// The Apps Script /exec endpoint answers a POST with a 302 to a one-shot
// googleusercontent URL. That second hop intermittently comes back as an HTML
// error or sign-in page instead of the handler's JSON, and a bare resp.json()
// then surfaces the raw "Unexpected token '<'" parser error to the user.
console.log('TR.parseApiResponse');
test('parses a normal JSON body', () => {
  assert.deepEqual(structuredClone(TR.parseApiResponse(200, '{"ok":true,"updated":3}')), { ok: true, updated: 3 });
});
test('parses a JSON error body',  () => {
  assert.deepEqual(structuredClone(TR.parseApiResponse(200, '{"ok":false,"error":"Unauthorized"}')), { ok: false, error: 'Unauthorized' });
});
test('HTML body throws, not a parser error', () => {
  assert.throws(() => TR.parseApiResponse(405, '<!DOCTYPE html><html><body>nope</body></html>'),
    err => !/Unexpected token/.test(err.message) && /405/.test(err.message));
});
test('HTML body names it a page, not data', () => {
  assert.throws(() => TR.parseApiResponse(405, '<!DOCTYPE html><html></html>'), /page instead of data/i);
});
test('HTML body is flagged retryable',  () => {
  try { TR.parseApiResponse(405, '<!DOCTYPE html>'); assert.fail('should throw'); }
  catch (e) { assert.equal(e.retryable, true); }
});
test('leading whitespace before HTML',  () => {
  assert.throws(() => TR.parseApiResponse(500, '\n  <!DOCTYPE html>'), /page instead of data/i);
});
test('an empty body throws',            () => {
  assert.throws(() => TR.parseApiResponse(200, ''), /empty/i);
});
test('non-HTML junk throws too',        () => {
  assert.throws(() => TR.parseApiResponse(200, 'not json at all'), /200/);
});
test('junk is retryable as well',       () => {
  try { TR.parseApiResponse(200, 'not json at all'); assert.fail('should throw'); }
  catch (e) { assert.equal(e.retryable, true); }
});

// ── TR.offersStrikeMove ───────────────────────────────────────
// Which events the annotators offer the move picker on. Wider than
// TR.isAttackEnd: a defensive penalty keeps the ball, so it is not an attempt,
// but the move being run when the defence infringed is still worth recording.
console.log('TR.offersStrikeMove');
test('Turnover offers one',        () => assert.equal(TR.offersStrikeMove('Turnover', 'Ball Down'), true));
test('Penalty Attack offers one',  () => assert.equal(TR.offersStrikeMove('Penalty Attack', 'Forward Pass'), true));
test('Penalty Defence offers one', () => assert.equal(TR.offersStrikeMove('Penalty Defence', 'Offside'), true));
test('Try does not — name is the move', () => assert.equal(TR.offersStrikeMove('Try', '32 - Cut'), false));
test('6 Again offers one',         () => assert.equal(TR.offersStrikeMove('Turnover', '6 Again'), true));
test('Game Event does not',        () => assert.equal(TR.offersStrikeMove('Game Event', 'Game Start'), false));
test('To Review does not',         () => assert.equal(TR.offersStrikeMove('To Review', ''), false));

// ── TR.recordedMoveOf ─────────────────────────────────────────
// What the move column shows and exports, as opposed to TR.strikeMoveOf, which
// answers the narrower "what move was this an attempt at" for the rate maths.
console.log('TR.recordedMoveOf');
test('Try uses its own name',      () => assert.equal(TR.recordedMoveOf('Try', '33 - Quicky', ''), '33 - Quicky'));
test('Turnover uses its move',     () => assert.equal(TR.recordedMoveOf('Turnover', 'Ball Down', '32 - Cut'), '32 - Cut'));
test('Pen Defence keeps its move', () => assert.equal(TR.recordedMoveOf('Penalty Defence', 'Offside', '32'), '32'));
test('Pen Defence untagged is empty', () => assert.equal(TR.recordedMoveOf('Penalty Defence', 'Offside', ''), ''));
test('6 Again keeps its move',     () => assert.equal(TR.recordedMoveOf('Turnover', '6 Again', '32'), '32'));
test('Game Event drops its move',  () => assert.equal(TR.recordedMoveOf('Game Event', 'Game Start', '32'), ''));

// A defensive penalty counts as an attempt that did not score. It leaves the
// attack the ball and a fresh count, so it is a good attacking outcome reading
// as a fail here — the rate is tries ÷ attempts, and this was not a try.
console.log('Pen Defence counts as an attempt');
test('strikeMoveOf keeps it',  () => assert.equal(TR.strikeMoveOf('Penalty Defence', 'Offside', '32'), '32'));
test('still not an attack end',() => assert.equal(TR.isAttackEnd('Penalty Defence', 'Offside'), false));
test('adds one failed attempt', () => {
  const without = TR.strikeMoveStats([
    { type: 'Try',      name: '32', strikeMove: '',   actionOwner: 'Team 1' },
    { type: 'Turnover', name: 'Ball Down', strikeMove: '32', actionOwner: 'Team 1' },
  ]);
  const withPenDef = TR.strikeMoveStats([
    { type: 'Try',      name: '32', strikeMove: '',   actionOwner: 'Team 1' },
    { type: 'Turnover', name: 'Ball Down', strikeMove: '32', actionOwner: 'Team 1' },
    { type: 'Penalty Defence', name: 'Offside', strikeMove: '32', actionOwner: 'Team 1' },
  ]);
  assert.equal(without.moves[0].attempts, 2);
  assert.equal(withPenDef.moves[0].attempts, 3);
  assert.equal(withPenDef.moves[0].tries, 1);
  assert.equal(withPenDef.moves[0].fails, 2);
  assert.equal(withPenDef.coverage.total, without.coverage.total + 1);
});
test('an untagged one is a missed tag, unlike an untagged touch', () => {
  // A defensive penalty is a discrete, notable event, so leaving it untagged is
  // a gap in coverage. An untagged touch is just ordinary play.
  const s = TR.strikeMoveStats([
    { type: 'Penalty Defence', name: 'Offside', strikeMove: '', actionOwner: 'T1' },
    { type: 'Touch',           name: 'Touch 2', strikeMove: '', actionOwner: 'T1' },
  ]);
  assert.equal(s.coverage.fails.total, 1);
  assert.equal(s.coverage.fails.tagged, 0);
});

// ── TR.csvCell ────────────────────────────────────────────────
console.log('TR.csvCell');
test('plain value',      () => assert.equal(TR.csvCell('Try'), 'Try'));
test('empty and nullish',() => { assert.equal(TR.csvCell(''), ''); assert.equal(TR.csvCell(null), ''); assert.equal(TR.csvCell(undefined), ''); });
test('number',           () => assert.equal(TR.csvCell(3), '3'));
test('zero is not blank',() => assert.equal(TR.csvCell(0), '0'));
test('comma quotes',     () => assert.equal(TR.csvCell('a,b'), '"a,b"'));
test('quote doubles',    () => assert.equal(TR.csvCell('say "hi"'), '"say ""hi"""'));
test('newline quotes',   () => assert.equal(TR.csvCell('a\nb'), '"a\nb"'));
test('CR quotes',        () => assert.equal(TR.csvCell('a\rb'), '"a\rb"'));
test('formula =',        () => assert.equal(TR.csvCell('=1+1'), "'=1+1"));
test('formula +',        () => assert.equal(TR.csvCell('+5'), "'+5"));
test('formula -',        () => assert.equal(TR.csvCell('-5m clips'), "'-5m clips"));
test('formula @',        () => assert.equal(TR.csvCell('@here'), "'@here"));
test('formula number',   () => assert.equal(TR.csvCell(-5), "'-5"));
// The guard checks a trimmed copy, but prefixes the ORIGINAL string, or a
// leading space/tab would slip the formula past a naive first-char check.
test('leading space before formula', () => assert.equal(TR.csvCell(' =1+1'), "' =1+1"));
test('leading tab before formula',   () => assert.equal(TR.csvCell('\t=1+1'), "'\t=1+1"));
// The guard runs BEFORE quoting, so a formula carrying a comma is both
// neutralised and quoted — quoting first would bury the apostrophe inside.
test('formula + comma',  () => assert.equal(TR.csvCell('=A1,B1'), `"'=A1,B1"`));
test('apostrophe mid-string is untouched', () => assert.equal(TR.csvCell("Dad's Army"), "Dad's Army"));
test('leading apostrophe is doubled', () => assert.equal(TR.csvCell("'19 season"), "''19 season"));
test('leading apostrophe before formula', () => assert.equal(TR.csvCell("'=1+1"), "''=1+1"));

// ── TR.toCSV ──────────────────────────────────────────────────
console.log('TR.toCSV');
test('header and rows', () => assert.equal(
  TR.toCSV([['#', 'Name'], [1, 'Scoop'], [2, 'a,b']]),
  '#,Name\r\n1,Scoop\r\n2,"a,b"'));
test('single row',   () => assert.equal(TR.toCSV([['a', 'b']]), 'a,b'));
test('empty input',  () => { assert.equal(TR.toCSV([]), ''); assert.equal(TR.toCSV(null), ''); });
test('no trailing newline', () => assert.equal(TR.toCSV([['a'], ['b']]).endsWith('\n'), false));

// ── TR.slugify ────────────────────────────────────────────────
console.log('TR.slugify');
test('spaces to dashes', () => assert.equal(TR.slugify('Backdoor teaching set', 'playlist'), 'backdoor-teaching-set'));
test('punctuation collapses', () => assert.equal(TR.slugify('France v England!! (2025)', 'playlist'), 'france-v-england-2025'));
test('diacritics stripped',   () => assert.equal(TR.slugify('Équipe Française', 'playlist'), 'equipe-francaise'));
test('path characters',       () => assert.equal(TR.slugify('a/b\\c', 'playlist'), 'a-b-c'));
test('fallback when empty',   () => { assert.equal(TR.slugify('', 'playlist'), 'playlist'); assert.equal(TR.slugify('!!!', 'playlist'), 'playlist'); assert.equal(TR.slugify(null, 'playlist'), 'playlist'); });
// Truncation happens before the trim, so a cut landing mid-separator cannot
// leave a trailing dash.
test('truncates to 60',  () => assert.equal(TR.slugify('a'.repeat(80), 'playlist').length, 60));
test('no trailing dash after truncation', () => {
  const s = TR.slugify('a'.repeat(59) + ' bbbb', 'playlist');
  assert.equal(s.length <= 60, true);
  assert.equal(s.endsWith('-'), false);
});


// ── TR.fromCSV ────────────────────────────────────────────────
console.log('TR.fromCSV');
const rt = (t) => structuredClone(TR.fromCSV(t));
test('simple rows',        () => assert.deepEqual(rt('a,b\r\nc,d'), [['a','b'],['c','d']]));
test('LF endings too',     () => assert.deepEqual(rt('a,b\nc,d'), [['a','b'],['c','d']]));
test('quoted comma',       () => assert.deepEqual(rt('a,"b,c"'), [['a','b,c']]));
test('doubled quote',      () => assert.deepEqual(rt('a,"say ""hi"""'), [['a','say "hi"']]));
test('embedded newline',   () => assert.deepEqual(rt('a,"line1\nline2"\r\nb,c'), [['a','line1\nline2'],['b','c']]));
test('embedded CRLF',      () => assert.deepEqual(rt('a,"l1\r\nl2"'), [['a','l1\r\nl2']]));
test('empty fields',       () => assert.deepEqual(rt('a,,c'), [['a','','c']]));
test('trailing newline',   () => assert.deepEqual(rt('a,b\r\n'), [['a','b']]));
test('blank lines skipped',() => assert.deepEqual(rt('a,b\r\n\r\nc,d'), [['a','b'],['c','d']]));
test('BOM stripped',       () => assert.deepEqual(rt('﻿a,b'), [['a','b']]));
test('quote mid-field is literal', () => assert.deepEqual(rt(`a,b"c`), [['a','b"c']]));
test('empty input',        () => { assert.deepEqual(rt(''), []); assert.deepEqual(rt(null), []); });
test('quoted empty field', () => assert.deepEqual(rt('a,""'), [['a','']]));
// A lone "" is one field that is empty, and the blank-line skip (deliberately,
// see the 'blank lines skipped' test above) can't tell that apart from a truly
// blank line — so it vanishes too. Known and intentional, not a bug: harmless
// here because the annotator always writes 9-10 columns, never one.
// Same shape trade-off as everywhere else in this file now: a single-field
// row is a degenerate ("ragged" in the loosest sense — there's nothing to be
// uniform WITH) one-column table, which this app never legitimately writes.
// The strict parse here (2 rows) and the loose parse (3 rows, since a raw
// '""' line isn't blank to a dumb splitter) disagree on row count, so the
// same rule that recovers a real legacy file applies — except the loose
// result is one column wide, which the "never single-column" guard rejects,
// so strict is kept after all. Pinned to nail down that the guard is doing
// real work here, not just in the fallback cases.
test('quoted empty field alone on a line vanishes, like a blank line', () => assert.deepEqual(rt('"a"\r\n""\r\n"b"'), [['a'],['b']]));
test('non-string input is stringified', () => { assert.deepEqual(rt(42), [['42']]); assert.deepEqual(rt(true), [['true']]); });
test('file of only newlines',          () => assert.deepEqual(rt('\r\n\r\n\n\n'), []));
// A closing quote immediately followed by another character (not a comma,
// newline or doubled quote) has nowhere to go per RFC 4180; the parser drops
// the closing quote and treats what follows as ordinary text in the same
// field. Pinned as known, not a bug: malformed input, not toCSV's output.
test('quote immediately after a closing quote is dropped', () => assert.deepEqual(rt('"a"b,c'), [['ab','c']]));
// A bare CR with no following LF must still end a row (old exports before
// toCSV existed, and files touched by classic-Mac tools, use CR alone).
// Mutation testing found that deleting the `|| c === '\r'` row-ending check
// leaves every other test passing, so pin it directly.
//
// The plain unquoted case alone no longer catches that mutation now that
// ── TR.fromCSV is a PURE strict parser ───────────────────────
// Three attempts to auto-detect a legacy file from its content were rejected:
// every content heuristic has counterexamples, because a legacy file and a
// valid RFC 4180 file can be shape-identical. The format marker decides
// instead — the fixed export writes a UTF-8 BOM and nothing older did — and
// that choice lives in the caller, which reads the file's bytes. So fromCSV
// has no modes: strict, always, with a round trip that can be proven.
console.log('TR.fromCSV — strict, no modes');
test('an unterminated quote consumes to EOF, as RFC 4180 says', () => {
  // Deliberately NOT "recovered". A legacy file never reaches this parser.
  assert.deepEqual(rt('a,"b\r\nc,d'), [['a', 'b\r\nc,d']]);
});
test('a quoted field spanning rows stays one field', () => {
  assert.deepEqual(rt('a,"x\r\ny",z'), [['a', 'x\r\ny', 'z']]);
});

// ── toCSV → fromCSV is identity, fuzzed ──────────────────────
// A hand-picked corpus proves only that the cases someone thought of survive.
// This generates them, because the failures found in review were all values
// nobody had thought to write down.
console.log('CSV round trip (fuzz)');
test('10000 random tables survive toCSV → fromCSV → csvUnguard', () => {
  const atoms = ['a', 'Z', '1', '', ' ', ',', '"', "'", '=', '+', '-', '@', '\n', '\r\n', '\r',
                 'é', 'ü', 'ß', 'tab\there', 'a, b', 'say "hi"', '-5m', "'19", '=1+1', '  lead'];
  let seed = 20260915;
  const rnd = (n) => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
  for (let t = 0; t < 10000; t++) {
    const cols = 2 + rnd(8);                       // ≥2: a 1-column empty row can't round-trip
    const rows = 1 + rnd(6);
    const table = [];
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < cols; c++) {
        let v = '';
        for (let k = 0, n = rnd(3); k <= n; k++) v += atoms[rnd(atoms.length)];
        row.push(v);
      }
      // A row of entirely empty fields is indistinguishable from a blank line,
      // which the parser skips by design. Keep one field non-empty.
      if (row.every(f => f === '')) row[0] = 'x';
      table.push(row);
    }
    const back = structuredClone(TR.fromCSV(TR.toCSV(table))).map(r => r.map(f => TR.csvUnguard(f)));
    assert.deepEqual(back, table);
  }
});

// ── TR.csvUnguard ─────────────────────────────────────────────
// The exact inverse of csvCell's formula guard. Only strips an apostrophe the
// guard could have written, so a hand-authored leading apostrophe survives.
console.log('TR.csvUnguard');
test('undoes a guarded dash',      () => assert.equal(TR.csvUnguard("'-5m"), '-5m'));
test('undoes a guarded equals',    () => assert.equal(TR.csvUnguard("'=1+1"), '=1+1'));
test('undoes a guarded apostrophe',() => assert.equal(TR.csvUnguard("''19"), "'19"));
test('undoes across whitespace',   () => assert.equal(TR.csvUnguard("' =1+1"), ' =1+1'));
test('keeps a hand-typed apostrophe', () => assert.equal(TR.csvUnguard("'19 season"), "'19 season"));
test('leaves plain values alone',  () => { assert.equal(TR.csvUnguard('Try'), 'Try'); assert.equal(TR.csvUnguard(''), ''); });
test('nullish',                    () => { assert.equal(TR.csvUnguard(null), ''); assert.equal(TR.csvUnguard(undefined), ''); });

// ── CSV round trip ────────────────────────────────────────────
// The half a write-only test can't prove: what comes back out.
console.log('CSV round trip');
test('nasty values survive a write then read', () => {
  const original = [
    ['Time','Type','Comment','Team'],
    ['0:10','Try','a comma, inside','France'],
    ['0:20','Turnover','say "play on"','Éire'],
    ['0:30','Try','-5m from the line','België'],
    ['0:40','Try',"'19 season squad",'Ünited'],
    ['0:50','Try','line one\nline two','X'],
    ['1:00','Try','','Y'],
  ];
  const back = structuredClone(TR.fromCSV(TR.toCSV(original)))
    .map(row => row.map(f => TR.csvUnguard(f)));
  assert.deepEqual(back, original);
});

// ── Detail tags ───────────────────────────────────────────────
console.log('TR.parseDetail / TR.formatDetail');
test('parses the try tags', () =>
  assert.deepEqual({ ...TR.parseDetail('pos:62,48; player:7; side:open; ch:W+') },
    { pos: '62,48', player: '7', side: 'open', ch: 'W+' }));
test('a value may hold spaces and colons', () =>
  assert.deepEqual({ ...TR.parseDetail('note:32 - Cut: late; player:7') }, { note: '32 - Cut: late', player: '7' }));
test('keys are case-insensitive, junk is skipped', () =>
  assert.deepEqual({ ...TR.parseDetail(' Player : 9 ;; nokey; :novalue; side:') }, { player: '9' }));
test('empty and nullish read as no tags', () => {
  assert.deepEqual({ ...TR.parseDetail('') }, {});
  assert.deepEqual({ ...TR.parseDetail(null) }, {});
});
test('writes known keys in a fixed order, the rest sorted', () =>
  assert.equal(TR.formatDetail({ zeta: 1, ch: 'ML', alpha: 'x', player: 7, pos: '50,99' }),
    'pos:50,99; player:7; ch:ML; alpha:x; zeta:1'));
test('drops empty values', () =>
  assert.equal(TR.formatDetail({ player: '', side: null, ch: undefined, pos: '1,2' }), 'pos:1,2'));
test('a separator in a value cannot corrupt the cell', () =>
  assert.equal(TR.formatDetail({ note: 'a;b\nc' }), 'note:a b c'));
test('round trip keeps every tag, unknown ones included', () => {
  const tags = { pos: '62,48', player: '11', side: 'blind', ch: 'LW', assist: '4' };
  assert.deepEqual({ ...TR.parseDetail(TR.formatDetail(tags)) }, tags);
});
test('nothing to write is an empty cell', () => assert.equal(TR.formatDetail({}), ''));

console.log('TR.detailPos / TR.stripLegacyPos');
test('position from Detail', () => assert.deepEqual({ ...TR.detailPos('pos:62,48; player:7', '') }, { x: 62, y: 48 }));
test('Detail wins over a legacy comment token', () =>
  assert.deepEqual({ ...TR.detailPos('pos:10,20', 'note @62,48') }, { x: 10, y: 20 }));
test('falls back to the legacy "@x,y" in Comment', () =>
  assert.deepEqual({ ...TR.detailPos('', 'great line break @62,48') }, { x: 62, y: 48 }));
test('a bare legacy token', () => assert.deepEqual({ ...TR.detailPos('', '@0,100') }, { x: 0, y: 100 }));
test('no position anywhere', () => {
  assert.equal(TR.detailPos('player:7', 'no pos here'), null);
  assert.equal(TR.detailPos('', 'email me @ 5,6'), null);
  assert.equal(TR.detailPos('', 'x@5,6'), null);
});
test('out-of-range positions are ignored', () => assert.equal(TR.detailPos('pos:150,20', ''), null));
test('strips the legacy token, keeps the note', () => {
  assert.equal(TR.stripLegacyPos('great line break @62,48'), 'great line break');
  assert.equal(TR.stripLegacyPos('@62,48'), '');
  assert.equal(TR.stripLegacyPos('? @5,6'), '?');
  assert.equal(TR.stripLegacyPos('no token'), 'no token');
  assert.equal(TR.stripLegacyPos(null), '');
});

console.log('TR.detailLabel');
test('labels the try tags', () => assert.equal(TR.detailLabel('pos:1,2; player:7; side:open; ch:ML'), '#7 · Open · ML'));
test('partial tags', () => {
  assert.equal(TR.detailLabel('ch:W+'), 'W+');
  assert.equal(TR.detailLabel('side:blind'), 'Blind');
  assert.equal(TR.detailLabel('pos:1,2'), '');
});

console.log('TR.inferTrySide');
test('ruck near the left touchline: a try further left is blind', () => assert.equal(TR.inferTrySide(20, 8), 'blind'));
test('ruck near the left touchline: a try to its right is open', () => assert.equal(TR.inferTrySide(20, 60), 'open'));
test('ruck near the right touchline: a try further right is blind', () => assert.equal(TR.inferTrySide(75, 95), 'blind'));
test('ruck near the right touchline: a try to its left is open', () => assert.equal(TR.inferTrySide(75, 30), 'open'));
test('just off centre still has a narrower side', () => {
  assert.equal(TR.inferTrySide(49, 40), 'blind');
  assert.equal(TR.inferTrySide(51, 40), 'open');
});
test('no call when it cannot be told', () => {
  assert.equal(TR.inferTrySide(50, 20), '');
  assert.equal(TR.inferTrySide(30, 30), '');
  assert.equal(TR.inferTrySide(null, 30), '');
  assert.equal(TR.inferTrySide(30, undefined), '');
});

console.log('TR.FieldGames.pausedSeconds');
{
  const ps = TR.FieldGames.pausedSeconds;
  test('no stoppages', () => { assert.equal(ps([], 0, 100, 100), 0); assert.equal(ps(undefined, 0, 100, 100), 0); });
  test('a finished stoppage inside the window', () => assert.equal(ps([{ from: 10, to: 40 }], 0, 100, 100), 30));
  test('clipped to the window at both ends', () => {
    assert.equal(ps([{ from: 10, to: 40 }], 20, 100, 100), 20);
    assert.equal(ps([{ from: 10, to: 40 }], 0, 25, 100), 15);
    assert.equal(ps([{ from: 10, to: 40 }], 50, 100, 100), 0);
  });
  test('one still running counts up to now', () => {
    assert.equal(ps([{ from: 60, to: null }], 0, 90, 90), 30);
    assert.equal(ps([{ from: 60, to: null }], 0, 120, 120), 60);
  });
  test('several add up', () => assert.equal(ps([{ from: 10, to: 20 }, { from: 50, to: 55 }, { from: 80, to: null }], 0, 100, 100), 35));
  test('the clock window minus its stoppages is the match time', () => {
    // kick-off at 5, a 30s injury stoppage, now 125 → 90s of match time
    const pauses = [{ from: 40, to: 70 }];
    assert.equal(125 - 5 - ps(pauses, 5, 125, 125), 90);
  });
}
test('a record from an older build gets an empty pause list', () => {
  const FGs = TR.FieldGames;
  const rec = FGs.create({ team1: 'A' });
  delete rec.pauses; FGs.save(rec);
  assert.deepEqual([...FGs.get(rec.id).pauses], []);
  FGs.remove(rec.id);
});
test('malformed pauses are dropped on load', () => {
  const FGs = TR.FieldGames;
  const rec = FGs.create({ team1: 'A' });
  rec.pauses = [{ from: 1, to: 2 }, { from: 'x' }, null, { from: 5, to: null }];
  FGs.save(rec);
  assert.equal(FGs.get(rec.id).pauses.length, 2);
  FGs.remove(rec.id);
});

// ── Every page's inline scripts compile ───────────────────────
// A syntax error in a page's own <script> takes the whole page down, and
// nothing above loads the pages — so compile each inline block (without
// running it). Modules and data blocks are skipped.
console.log('Inline page scripts');
for (const page of fs.readdirSync('.').filter(f => f.endsWith('.html')).sort()) {
  const html   = fs.readFileSync(page, 'utf8');
  const blocks = [...html.matchAll(/<script(\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
    .filter(m => !/\bsrc\s*=/.test(m[1] || '') && !/type\s*=\s*["']?(module|application\/(ld\+)?json|text\/template)/i.test(m[1] || ''))
    .map(m => m[2]).filter(code => code.trim());
  if (!blocks.length) continue;
  test(`${page} (${blocks.length} block${blocks.length === 1 ? '' : 's'})`, () => {
    blocks.forEach((code, i) => {
      try { new vm.Script(code, { filename: `${page}#script${i + 1}` }); }
      catch (e) { throw new Error(`script ${i + 1}: ${e.message}`); }
    });
  });
}

console.log('TR.FieldStats');
{
  const FS = TR.FieldStats;
  const ev = (type, name, owner, x, y) => ({ type, name, possessionOwner: owner, actionOwner: owner, x: x ?? null, y: y ?? null });
  const game = [
    ev('Game Event', 'Game Start', 'Team 1'),
    ev('Touch', 'Touch 1', 'Team 1', 50, 20), ev('Touch', 'Touch 2', 'Team 1', 40, 45),
    ev('Touch', 'Touch 3', 'Team 1', 30, 70), ev('Touch', 'Touch 4', 'Team 1', 20, 90),
    ev('Try', '32 - Cut', 'Team 1', 15, 100),
    ev('Touch', 'Touch 1', 'Team 2', 50, 10), ev('Touch', 'Touch 2', 'Team 2', 70, 25),
    ev('Turnover', 'Ball Down', 'Team 2', 80, 30),
    ev('Game Event', 'Game End', 'Team 1'),
  ];
  test('one set per possession, positioned only', () => {
    const sets = FS.possessionSets(game);
    assert.equal(sets.length, 2);
    assert.equal(sets[0].owner, 'Team 1'); assert.equal(sets[0].touches, 4); assert.equal(sets[0].endType, 'Try');
    assert.equal(sets[0].startY, 20); assert.equal(sets[0].endY, 100); assert.equal(sets[0].maxY, 100);
    assert.equal(sets[1].endType, 'Turnover'); assert.equal(sets[1].gain, 20);
  });
  test('a set with no positions is left out', () =>
    assert.equal(FS.possessionSets([ev('Touch', 'Touch 1', 'Team 1'), ev('Try', 'Other', 'Team 1')]).length, 0));
  test('per-team numbers', () => {
    const { t1, t2, any } = FS.computeFieldStats(game);
    assert.equal(any, true);
    assert.equal(t1.sets, 1); assert.equal(t1.redSets, 1); assert.equal(t1.redTries, 1); assert.equal(t1.redConvPct, 100);
    assert.equal(Math.round(t1.gainM), 56);                 // 80 y-units × 0.7
    assert.deepEqual([...t1.channels], [50, 50, 0]);          // touches at x 30, 20 (left) and 50, 40 (middle)
    assert.equal(t1.t3Pts.length, 1); assert.equal(t1.tryPts.length, 1);
    assert.equal(t2.lostPts.length, 1); assert.equal(t2.redSets, 0); assert.equal(t2.redConvPct, null);
  });
  test('no positions at all → nothing to show', () =>
    assert.equal(FS.computeFieldStats([ev('Touch', 'Touch 1', 'Team 1'), ev('Try', 'Other', 'Team 1')]).any, false));
  test('outcomes', () => {
    assert.equal(FS.outcomeOf({ endType: 'Turnover', endName: '6th Touch' }), '6th Touch');
    assert.equal(FS.outcomeOf({ endType: 'Penalty Defence' }), 'Penalty');
    assert.equal(FS.outcomeOf({ endType: 'Touch' }), null);
  });
  test('the drawing functions return markup', () => {
    const { t1, sets } = FS.computeFieldStats(game);
    assert.match(FS.fieldMapSVG(t1, '#3b82f6', 'A'), /<svg/);
    assert.match(FS.outcomeBar(sets, 'A', '#3b82f6'), /cseg/);
    assert.match(FS.territorySVG(sets, { 'Team 1': 'A', 'Team 2': 'B' }, { 'Team 1': '#3b82f6', 'Team 2': '#f59e0b' }), /<rect/);
  });
}

console.log('TR.FieldStats possession paths');
{
  const FS = TR.FieldStats;
  const ev = (type, name, owner, x, y) => ({ type, name, possessionOwner: owner, actionOwner: owner, x: x ?? null, y: y ?? null });
  const game = [
    ev('Game Event', 'Game Start', 'Team 1'),
    ev('Touch', 'Touch 1', 'Team 1', 50, 30), ev('Touch', 'Touch 2', 'Team 1', 40, 50),
    ev('Turnover', 'Ball Down', 'Team 1', 30, 60),                     // lost at 30,60
    ev('Turnover', 'Other', 'Team 2', 70, 45),                          // straight back, no touch
    ev('Try', '33 - Cut', 'Team 1', 20, 100),                           // try from the turnover
    ev('Touch', 'Touch 1', 'Team 2', 55, 60), ev('Penalty Defence', 'Offside', 'Team 2', 55, 65),
    ev('Touch', 'Touch 1', 'Team 2', 50, 75), ev('Turnover', '6th Touch', 'Team 2', 45, 90),
    ev('Game Event', 'Game End', 'Team 1'),
  ];
  const sets = FS.possessionPaths(game);
  test('one path per possession', () => assert.deepEqual([...sets.map(s => s.owner)], ['Team 1', 'Team 2', 'Team 1', 'Team 2']));
  test('kick-off starts on halfway', () => assert.deepEqual({ ...sets[0].steps[0] }, { k: 'start', how: 'tap', x: 50, y: 50 }));
  test('a turnover hands over on the spot, seen from the other end', () =>
    assert.deepEqual({ ...sets[1].steps[0] }, { k: 'start', how: 'won', x: 70, y: 40 }));
  test('a try straight from a turnover still has a path', () => {
    const s = sets[2];
    assert.equal(s.outcome, 'Try');
    assert.deepEqual([...s.steps.map(p => p.k)], ['start', 'end']);
    assert.deepEqual([s.steps[0].x, s.steps[0].y], [30, 55]);           // mirror of 70,45
  });
  test('after a try the next set taps off on halfway', () => assert.equal(sets[3].steps[0].how, 'tap'));
  test('a defensive penalty keeps the set going', () => {
    const s = sets[3];
    assert.equal(s.outcome, '6th Touch');
    assert.deepEqual([...s.steps.map(p => p.k)], ['start', 1, 'pen', 1, 'end']);
    assert.equal(FS.pathGains(s).map(g => g.label).join(' '), 'start→T1 T1→Pen Pen→T1 T1→end');
  });
  test('metres per step', () =>
    assert.equal(JSON.stringify(FS.pathGains(sets[0]).map(g => [g.label, Math.round(g.m)])), JSON.stringify([['start→T1', -14], ['T1→T2', 14], ['T2→end', 7]])));
  test('gain buckets only join consecutive touches', () => {
    const b = FS.gainBuckets(sets);
    assert.equal(b[0].label, '→T1'); assert.equal(b[0].n, 3);          // three sets reached a touch 1
    assert.equal(b[1].n, 1); assert.equal(Math.round(b[1].mean), 14);  // T1→T2 once
  });
  test('the typical set needs 3 sets at a touch', () => {
    assert.equal(FS.typicalSet(sets).length, 0);                       // only 2 sets reached touch 1
    const three = FS.possessionPaths([...game, ev('Game Event', 'Game Start', 'Team 1'), ev('Touch', 'Touch 1', 'Team 1', 50, 40), ev('Try', 'Other', 'Team 1', 50, 100)]);
    assert.deepEqual([...FS.typicalSet(three).map(p => p.k)], [1]);
  });
  test('a picked set reads back', () => {
    const d = FS.describeSet(sets[2], 2, sets.length);
    assert.equal(d.title, 'Try · 33 - Cut');
    assert.match(d.summary, /won at 39m to 70m, \+31m straight from the turnover, no touch/);
  });
  test('drawing returns markup with clickable sets', () => {
    assert.match(FS.pathsSVG(sets.filter(s => s.owner === 'Team 1'), '#3b82f6', 1), /data-set="1"/);
    assert.match(FS.setColumnsSVG(sets, '#3b82f6', null), /pp-col/);
    assert.match(FS.gainChartSVG([{ name: 'A', color: '#3b82f6', sets }]), /<rect/);
  });
}

console.log('TR.FieldStats tries by side and channel');
{
  const FS = TR.FieldStats;
  const tr = (owner, tags) => ({ type: 'Try', name: 'Other', possessionOwner: owner, actionOwner: owner, x: 50, y: 100, tags });
  const evs = [
    tr('Team 1', { side: 'open', ch: 'ML' }), tr('Team 1', { side: 'blind' }), tr('Team 1', {}),
    tr('Team 2', { ch: 'W+' }), tr('Team 2', { side: 'open', ch: 'W+' }), tr('Team 2', { side: 'sideways', ch: 'XX' }),
    { type: 'Touch', name: 'Touch 1', possessionOwner: 'Team 1', actionOwner: 'Team 1', x: 1, y: 1, tags: { side: 'open' } },
  ];
  const { t1, t2 } = FS.tryTagStats(evs);
  test('counts each tag per team, only on tries', () => {
    assert.equal(t1.tries, 3); assert.equal(t1.side.open, 1); assert.equal(t1.side.blind, 1); assert.equal(t1.ch.ML, 1);
    assert.equal(t2.ch['W+'], 2); assert.equal(t2.side.open, 1);
  });
  test('untagged and unknown values are counted as untagged, not dropped', () => {
    assert.equal(t1.sideTagged, 2); assert.equal(t1.chTagged, 1);
    assert.equal(t2.tries, 3); assert.equal(t2.sideTagged, 1); assert.equal(t2.chTagged, 2);
  });
  test('bars label their parts, and say so when nothing was tagged', () => {
    assert.match(FS.sideBar(t1, 'A', '#3b82f6'), /Open 1/);
    assert.match(FS.channelTagBar(t2, 'B', '#f59e0b'), /W\+ 2/);
    assert.match(FS.channelTagBar({ ch: { MM: 0, ML: 0, LW: 0, 'W+': 0 } }, 'C', '#000'), /no tries tagged/);
  });
}

console.log('TR.FieldStats start zones');
{
  const FS = TR.FieldStats;
  const ev = (type, name, owner, x, y) => ({ type, name, possessionOwner: owner, actionOwner: owner, x: x ?? null, y: y ?? null });
  const game = [
    ev('Game Event', 'Game Start', 'Team 1'),
    ev('Touch', 'Touch 1', 'Team 1', 50, 60), ev('Turnover', 'Ball Down', 'Team 1', 40, 80),   // tap on halfway → middle
    ev('Touch', 'Touch 1', 'Team 2', 50, 30), ev('Turnover', 'Ball Down', 'Team 2', 60, 40),   // won at 100-80=20 → 14m, own end
    ev('Touch', 'Touch 1', 'Team 1', 50, 70), ev('Try', 'Other', 'Team 1', 50, 100),           // won at 100-40=60 → 42m, middle
    ev('Touch', 'Touch 1', 'Team 2', 50, 60), ev('Turnover', 'Ball Down', 'Team 2', 50, 20),   // tap → middle
    ev('Touch', 'Touch 1', 'Team 1', 50, 85), ev('Try', 'Other', 'Team 1', 50, 100),           // won at 100-20=80 → 56m, opp end
    ev('Game Event', 'Game End', 'Team 1'),
  ];
  const sets = FS.possessionPaths(game);
  test('each set is placed by where it started', () =>
    assert.equal(sets.map(FS.startZone).join(' '), 'mid own mid mid opp'));
  test('the 10m lines are the boundaries', () => {
    const at = m => FS.startZone({ steps: [{ y: m / FS.Y_TO_M }] });
    assert.equal(at(0), 'own'); assert.equal(at(24.9), 'own'); assert.equal(at(25), 'mid');
    assert.equal(at(44.9), 'mid'); assert.equal(at(45), 'opp'); assert.equal(at(70), 'opp');
  });
  test('filtering keeps only those sets, and every game event', () => {
    const opp = FS.eventsStartingIn(game, 'opp');
    assert.equal(opp.filter(e => e.type === 'Game Event').length, 2);
    assert.equal(opp.filter(e => e.type !== 'Game Event').length, 2);           // the last set's touch + try
    assert.equal(FS.eventsStartingIn(game, 'all'), game);
    assert.equal(FS.computeFieldStats(FS.eventsStartingIn(game, 'own')).t2.sets, 1);
  });
  test('two of one team\'s sets left side by side stay two sets', () => {
    const mid = FS.eventsStartingIn(game, 'mid');                                // sets 1, 3 and 4
    const t1 = FS.computeFieldStats(mid).t1;
    assert.equal(t1.sets, 2);                                                    // sets 1 and 3 are both Team 1
    assert.equal(FS.possessionSets(mid).length, 3);
    assert.equal(mid.filter(e => e.name === 'Set Break').length, 2);
  });
}

// ── Code.gs cache: gzip + chunking ────────────────────────────
// Apps Script can't run here, so CacheService and Utilities are stood in for
// with a Map and Node's zlib — enough to prove a value too big for one entry
// comes back byte-for-byte, whichever path it took.
console.log('Code.gs cache helpers');
{
  const zlib = require('zlib');
  const gs   = fs.readFileSync('apps_script/Code.gs', 'utf8');
  const src  = gs.slice(gs.indexOf('// ── Cache helpers'), gs.indexOf('// ── Tab ownership tokens'));
  const store = new Map(), puts = [];
  const cache = {
    get: k => (store.has(k) ? store.get(k) : null),
    getAll: ks => Object.fromEntries(ks.filter(k => store.has(k)).map(k => [k, store.get(k)])),
    put: (k, v) => { if (v.length > 100000) throw new Error('entry too big'); store.set(k, v); puts.push(k); },
    putAll: o => Object.entries(o).forEach(([k, v]) => cache.put(k, v)),
  };
  const blob = bytes => ({ getBytes: () => bytes, getDataAsString: () => Buffer.from(bytes).toString('utf8') });
  const ctx = vm.createContext({
    CacheService: { getScriptCache: () => cache },
    Utilities: {
      newBlob: (data) => blob(typeof data === 'string' ? Buffer.from(data, 'utf8') : Buffer.from(data)),
      gzip: b => blob(zlib.gzipSync(Buffer.from(b.getBytes()))),
      ungzip: b => blob(zlib.gunzipSync(Buffer.from(b.getBytes()))),
      base64Encode: bytes => Buffer.from(bytes).toString('base64'),
      base64Decode: str => [...Buffer.from(str, 'base64')],
    },
  });
  const selfTestSrc = gs.slice(gs.indexOf('function cacheSelfTest()'), gs.indexOf('function doGet('));
  vm.runInContext(src + selfTestSrc + '; this.cacheGet = cacheGet; this.cachePut = cachePut; this.cacheSelfTest = cacheSelfTest;', ctx);
  const { cacheGet, cachePut } = ctx;

  test('a small value is stored as is', () => {
    cachePut('small', '{"ok":true}'); assert.equal(store.get('small'), '{"ok":true}'); assert.equal(cacheGet('small'), '{"ok":true}');
  });
  test('a 1 MB feed that compresses well fits one gzipped entry', () => {
    const rows = Array.from({ length: 8000 }, (_, i) => ['0:01:23', 'Team 1', 'Touch', 'Touch ' + (i % 5 + 1), '', '', 'Team 1', '', 'pos:41,58', '2026_m35_euros_england_south-africa', 'England', 'South Africa', 'Euros', '2026', 'M35']);
    const big = JSON.stringify({ ok: true, version: 7, rows });
    assert.ok(big.length > 900000);
    cachePut('all', big);
    assert.ok(store.get('all').startsWith('gz:'));
    assert.equal(cacheGet('all'), big);
  });
  test('something that won\'t compress enough is split across chunks', () => {
    let seed = 1; const noise = Array.from({ length: 200000 }, () => String.fromCharCode(33 + ((seed = seed * 16807 % 2147483647) % 90))).join('');
    cachePut('noisy', noise);
    assert.match(store.get('noisy'), /^chunks:\d+$/);
    assert.equal(cacheGet('noisy'), noise);
  });
  test('a missing chunk reads as a miss, not as broken data', () => {
    const n = +store.get('noisy').split(':')[1];
    store.delete('noisy#' + (n - 1));
    assert.equal(cacheGet('noisy'), null);
  });
  test('the deploy self-test passes on a working cache', () => {
    const r = ctx.cacheSelfTest();
    assert.equal(r.small, true); assert.equal(r.large, true); assert.ok(r.largeKB > 100);
    assert.ok(store.get('selftest:big').startsWith('gz:'));
  });
  test('unicode survives the round trip', () => {
    const v = JSON.stringify({ team: 'Côte d’Ivoire — Équipe', rows: Array.from({ length: 4000 }, () => ['é', 'ü', '—']) });
    cachePut('uni', v); assert.equal(cacheGet('uni'), v);
  });
}

// ── Server copy of the strike-move rule ───────────────────────
// Code.gs can't load js/events.js, so it carries its own deriveStrikeMove for
// the inline-edit path. It drifted once — 6 Again, Penalty Defence and tagged
// touches lost their move on every edit — so the two are checked against each
// other over every type and name rather than by a few hand-picked cases.
console.log('Code.gs deriveStrikeMove');
{
  const gs   = fs.readFileSync('apps_script/Code.gs', 'utf8');
  const grab = (start, end) => gs.slice(gs.indexOf(start), gs.indexOf(end, gs.indexOf(start)));
  const src  = grab('var MOVE_BEARING_TYPES', 'function updateRow(');
  const gsCtx = vm.createContext({});
  vm.runInContext(src, gsCtx);
  const derive = gsCtx.deriveStrikeMove;

  test('its move-bearing types match the client', () =>
    assert.deepEqual([...vm.runInContext('MOVE_BEARING_TYPES', gsCtx)], [...TR.MOVE_BEARING_TYPES]));

  const cases = [];
  for (const type of [...Object.keys(TR.MENU), 'Touch']) {
    const names = type === 'Touch' ? ['Touch 1', 'Touch 4'] : (TR.MENU[type].length ? TR.MENU[type] : ['']);
    for (const name of names) for (const stored of ['', '32 - Cut']) cases.push([type, name, stored]);
  }
  test(`agrees with TR.strikeMoveOf on all ${cases.length} combinations`, () => {
    const off = cases.filter(([t, n, m]) => derive(t, n, m) !== TR.strikeMoveOf(t, n, m))
      .map(([t, n, m]) => `${t}/${n}/${m || '∅'}: server "${derive(t, n, m)}" client "${TR.strikeMoveOf(t, n, m)}"`);
    assert.deepEqual(off, []);
  });
  test('keeps the move on the events it used to wipe', () => {
    assert.equal(derive('Turnover', '6 Again', 'Scoop'), 'Scoop');
    assert.equal(derive('Penalty Defence', 'Offside', 'Scoop'), 'Scoop');
    assert.equal(derive('Touch', 'Touch 3', 'Scoop'), 'Scoop');
  });
  test('still clears it where none belongs', () => {
    assert.equal(derive('Game Event', 'Game Start', 'Scoop'), '');
    assert.equal(derive('To Review', '', 'Scoop'), '');
    assert.equal(derive('Try', '33 - Cut', 'Scoop'), '33 - Cut');
  });
}

// ─────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
