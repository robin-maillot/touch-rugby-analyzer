#!/usr/bin/env node
// Runs TR.* shared-utility assertions in Node.js — no npm required.
// Uses the vm module to load browser-targeted JS files with minimal stubs.
const vm     = require('vm');
const fs     = require('fs');
const assert = require('assert/strict');

const ctx = vm.createContext({
  sessionStorage: { getItem: () => null },
  window:         { location: { replace() {} } },
});

for (const f of ['js/config.js', 'js/utils.js', 'js/events.js', 'js/possession.js', 'js/consistency.js', 'js/player.js']) {
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


// ─────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
