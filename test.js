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

for (const f of ['js/config.js', 'js/utils.js', 'js/events.js', 'js/possession.js']) {
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
test('zero/falsy',  () => { assert.equal(TR.fmt(0), '0:00:00'); assert.equal(TR.fmt(null), '0:00:00'); assert.equal(TR.fmt(NaN), '0:00:00'); });
test('sub-minute',  () => { assert.equal(TR.fmt(5), '0:00:05'); assert.equal(TR.fmt(59), '0:00:59'); });
test('minutes',     () => { assert.equal(TR.fmt(65), '0:01:05'); assert.equal(TR.fmt(600), '0:10:00'); });
test('hours',       () => { assert.equal(TR.fmt(3600), '1:00:00'); assert.equal(TR.fmt(3661), '1:01:01'); assert.equal(TR.fmt(36000), '10:00:00'); });

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

// ── TR.isTurnover ─────────────────────────────────────────────
console.log('TR.isTurnover');
test('Try → true',              () => assert.equal(TR.isTurnover('Try', 'Scoop'), true));
test('Penalty Attack → true',   () => assert.equal(TR.isTurnover('Penalty Attack', 'Forward Pass'), true));
test('Penalty Defence → false', () => assert.equal(TR.isTurnover('Penalty Defence', 'Offside'), false));
test('Turnover 6th Touch → true', () => assert.equal(TR.isTurnover('Turnover', '6th Touch'), true));
test('Turnover 6 Again → false',  () => assert.equal(TR.isTurnover('Turnover', '6 Again'), false));
test('Game Event → false',       () => assert.equal(TR.isTurnover('Game Event', 'Game Start'), false));

// ── TR.inferPossessionAfter ───────────────────────────────────
console.log('TR.inferPossessionAfter');
test('Try → other',             () => { assert.equal(TR.inferPossessionAfter('Team 1', 'Try', 'Scoop'), 'Team 2'); assert.equal(TR.inferPossessionAfter('Team 2', 'Try', 'Scoop'), 'Team 1'); });
test('Penalty Attack → other',  () => assert.equal(TR.inferPossessionAfter('Team 1', 'Penalty Attack', 'Forward Pass'), 'Team 2'));
test('Penalty Defence → same',  () => assert.equal(TR.inferPossessionAfter('Team 1', 'Penalty Defence', 'Offside'), 'Team 1'));
test('Turnover → other',        () => assert.equal(TR.inferPossessionAfter('Team 1', 'Turnover', '6th Touch'), 'Team 2'));
test('6 Again → same',          () => assert.equal(TR.inferPossessionAfter('Team 1', 'Turnover', '6 Again'), 'Team 1'));

// ── TR.inferActionOwner ───────────────────────────────────────
console.log('TR.inferActionOwner');
test('Try → possession owner',       () => assert.equal(TR.inferActionOwner('Team 1', 'Try', 'Scoop'), 'Team 1'));
test('Turnover → possession owner',  () => assert.equal(TR.inferActionOwner('Team 1', 'Turnover', '6th Touch'), 'Team 1'));
test('6 Again → other (defence)',    () => assert.equal(TR.inferActionOwner('Team 1', 'Turnover', '6 Again'), 'Team 2'));
test('Penalty Attack → same',        () => assert.equal(TR.inferActionOwner('Team 1', 'Penalty Attack', 'Forward Pass'), 'Team 1'));
test('Penalty Defence → other',      () => { assert.equal(TR.inferActionOwner('Team 1', 'Penalty Defence', 'Offside'), 'Team 2'); assert.equal(TR.inferActionOwner('Team 2', 'Penalty Defence', 'Offside'), 'Team 1'); });

// ─────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
