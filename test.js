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

for (const f of ['js/config.js', 'js/utils.js', 'js/events.js', 'js/possession.js', 'js/player.js']) {
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
test('null meta',                  () => assert.equal(TR.player.providerFor(null), null));
test('empty meta',                 () => assert.equal(TR.player.providerFor({}), null));
test('youtubelink → youtube',      () => assert.equal(TR.player.providerFor({ youtubelink: 'https://youtu.be/abc' }), 'youtube'));
test('streamuid → stream',         () => assert.equal(TR.player.providerFor({ streamuid: 'abc' }), 'stream'));
test('streamuid wins over youtube',() => assert.equal(TR.player.providerFor({ youtubelink: 'x', streamuid: 'y' }), 'stream'));
test('explicit override: youtube', () => assert.equal(TR.player.providerFor({ videoprovider: 'youtube', streamuid: 'y' }), 'youtube'));
test('explicit override: stream',  () => assert.equal(TR.player.providerFor({ videoprovider: 'stream', youtubelink: 'x' }), 'stream'));
test('garbage override ignored',   () => assert.equal(TR.player.providerFor({ videoprovider: 'vimeo', youtubelink: 'x' }), 'youtube'));

// ── TR.player.seekLink ───────────────────────────────────────
console.log('TR.player.seekLink');
test('youtube URL',         () => assert.equal(TR.player.seekLink({ youtubelink: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' }, 65), 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=60s'));
test('youtube 5s lookback', () => assert.equal(TR.player.seekLink({ youtubelink: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' }, 3),  'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0s'));
test('stream URL',          () => assert.equal(TR.player.seekLink({ streamuid: 'ea95132c15732ca6abd6cc11696c3e2c' }, 65), 'https://iframe.videodelivery.net/ea95132c15732ca6abd6cc11696c3e2c?startTime=60'));
test('no provider → empty', () => assert.equal(TR.player.seekLink({}, 10), ''));
test('null meta → empty',   () => assert.equal(TR.player.seekLink(null, 10), ''));

// ─────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
