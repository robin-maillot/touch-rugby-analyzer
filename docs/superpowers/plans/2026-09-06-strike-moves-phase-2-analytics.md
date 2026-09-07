# Strike Moves — Phase 2 (Analytics) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report try rate per strike move and the top scoring move — by volume and by efficiency — across the dashboard, per-game analysis, the analytics explorer, and a Strike Moves sheet reached from both field annotators' Stats sheets.

**Architecture:** One pure module, `js/strike_moves.js`, owns the entire calculation. Each of the four surfaces adapts its own row shape to a common `{type, name, strikeMove, actionOwner}` at the boundary and calls it. None of them re-implements the formula.

**Tech Stack:** Vanilla ES2020 in static HTML pages, no build step. Shared `TR.*` modules under `js/`. Vega-Lite for charts on `games.html` / `analytics.html`. Node assertions in `test.js`, mirrored as QUnit in `tests.html`.

**Spec:** `docs/superpowers/specs/2026-09-06-strike-moves-design.md`
**Depends on:** `docs/superpowers/plans/2026-09-06-strike-moves-phase-1-capture.md` — must be shipped, deployed, and **used to tag real games** first.

## Global Constraints

- No build step. Plain `<script src>` tags; every shared helper hangs off the global `TR` namespace.
- `js/strike_moves.js` must be registered in **four** places: `test.js` load list, `tests.html` script tags, `sw.js` `ASSETS`, and each consuming page's script tags. `dashboard.html` and `games.html` currently load only `js/config.js` — they need `js/events.js` **before** `js/strike_moves.js`, since it depends on `TR.isAttackEnd`.
- The formula lives in `TR.strikeMoveStats` and nowhere else. A surface that needs a variant passes a filtered event list; it does not compute rates itself.
- Untagged attempts are excluded from rates. Every surface that shows a rate must also show `coverage`.
- `coverage.total` counts attack-ending events only — never all events.
- `'Other'` and `'Interception'` are **excluded from every rate**: no row in `moves`, never `topByRate` or `topByTries`, counted as untagged in `coverage`. They remain selectable in the annotators. See the spec's amended decision for why — the two sides are asymmetric and both would otherwise sit at a 100% artefact rate and top both leaderboards.
- **Never read the stored `Strike Move` column directly.** Always re-derive through `TR.strikeMoveOf(type, name, storedMove)` at the input boundary. `Code.gs`'s `updateRow` (the viewer's inline edit) writes only Name and Comment, so the stored column goes stale on edited rows.
- `coverage` is reported **per side** — Try-side and failure-side separately. A Try always has a Name, so its coverage is 100% by construction and would inflate a combined figure.
- `topByRate` requires `attempts >= TR.MIN_MOVE_ATTEMPTS` (2). `topByTries` has no threshold.
- `rate` is a 0–1 number. Formatting to a percentage is each surface's job.

---

### Task 1: The `TR.strikeMoveStats` module

**Files:**
- Modify: `js/events.js` (add `TR.EXCLUDED_MOVES` beside `TR.STRIKE_MOVES`)
- Create: `js/strike_moves.js`
- Modify: `test.js:26` (module load list)
- Modify: `tests.html:15` (script tags)
- Modify: `sw.js:21-27` (`ASSETS`)
- Test: `test.js` (append a new section at the end)
- Test: `tests.html` (append a QUnit module at the end)

**Interfaces:**
- Consumes: `TR.isAttackEnd`, `TR.strikeMoveOf`, `TR.MIN_MOVE_ATTEMPTS` (Phase 1 Task 1).
- Produces:

```js
TR.strikeMoveStats(events: {type, name, strikeMove, actionOwner}[]) => {
  moves:      { move: string, tries: number, fails: number, attempts: number, rate: number }[],
  coverage:   { tagged: number, total: number, pct: number,
                tries:  { tagged: number, total: number, pct: number },
                fails:  { tagged: number, total: number, pct: number } },
  topByTries: { move, tries, fails, attempts, rate } | null,
  topByRate:  { move, tries, fails, attempts, rate } | null,
}

`TR.EXCLUDED_MOVES = ['Other', 'Interception']` — add it to `js/events.js` beside
`TR.STRIKE_MOVES` in Task 1, so the annotators and the analytics agree on one list.
```

- [ ] **Step 1: Write the failing tests**

Append to `test.js`:

```js
// ── TR.strikeMoveStats ────────────────────────────────────────
console.log('TR.strikeMoveStats');
const ev = (type, name, strikeMove, actionOwner) => ({ type, name, strikeMove, actionOwner: actionOwner || 'Team 1' });

test('empty input', () => {
  const s = TR.strikeMoveStats([]);
  assert.deepEqual(s.moves, []);
  assert.deepEqual(s.coverage, { tagged: 0, total: 0, pct: 0 });
  assert.equal(s.topByTries, null);
  assert.equal(s.topByRate, null);
});

test('null input is tolerated', () => assert.equal(TR.strikeMoveStats(null).moves.length, 0));

test('excluded moves are Other and Interception',
  () => assert.deepEqual(TR.EXCLUDED_MOVES, ['Other', 'Interception']));
test('excluded moves are real entries of the picker list',
  () => TR.EXCLUDED_MOVES.forEach(m => assert.ok(TR.STRIKE_MOVES.includes(m), m)));

test('a try and a turnover on the same move', () => {
  const s = TR.strikeMoveStats([
    ev('Try', '32 - Cut', ''),
    ev('Turnover', 'Ball Down', '32 - Cut'),
  ]);
  assert.equal(s.moves.length, 1);
  assert.deepEqual(s.moves[0], { move: '32 - Cut', tries: 1, fails: 1, attempts: 2, rate: 0.5 });
});

test('a pen attack counts as a failure', () => {
  const s = TR.strikeMoveStats([ev('Penalty Attack', 'Forward Pass', '23')]);
  assert.deepEqual(s.moves[0], { move: '23', tries: 0, fails: 1, attempts: 1, rate: 0 });
});

test('coverage counts attack-ends only', () => {
  const s = TR.strikeMoveStats([
    ev('Try', 'Scoop', ''),                    // attack end, tagged (name is the move)
    ev('Turnover', 'Ball Down', '32'),         // attack end, tagged
    ev('Turnover', 'Ball Down', ''),           // attack end, untagged
    ev('Turnover', '6 Again', '32'),           // NOT an attack end
    ev('Penalty Defence', 'Offside', '32'),    // NOT an attack end
    ev('Game Event', 'Game Start', ''),        // NOT an attack end
  ]);
  assert.deepEqual(s.coverage, { tagged: 2, total: 3, pct: 2 / 3 });
});

test('untagged attempts are excluded from every move row', () => {
  const s = TR.strikeMoveStats([
    ev('Try', '32', ''),
    ev('Turnover', 'Ball Down', ''),
  ]);
  assert.equal(s.moves.length, 1);
  assert.equal(s.moves[0].attempts, 1);
});

test('Other and Interception never get a row', () => {
  const s = TR.strikeMoveStats([
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
  const s = TR.strikeMoveStats([
    ev('Try', 'Other', ''), ev('Try', 'Other', ''), ev('Try', 'Other', ''),
    ev('Try', '32', ''), ev('Turnover', 'Ball Down', '32'),
  ]);
  assert.equal(s.topByTries.move, '32');
  assert.equal(s.topByRate.move, '32');
});

test('a stale stored move loses to the Name on a Try', () => {
  // What a viewer Name edit leaves behind: Name corrected, column not.
  const s = TR.strikeMoveStats([ev('Try', '32 - Cut', 'Other')]);
  assert.deepEqual(s.moves.map(m => m.move), ['32 - Cut']);
  assert.equal(s.moves[0].tries, 1);
});

test('a stale stored move is dropped when the name stops ending an attack', () => {
  const s = TR.strikeMoveStats([ev('Turnover', '6 Again', '32')]);
  assert.deepEqual(s.moves, []);
  assert.equal(s.coverage.total, 0);
});

test('coverage is reported per side', () => {
  const s = TR.strikeMoveStats([
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
  const s = TR.strikeMoveStats([
    ev('Turnover', 'Ball Down', 'Scoop'), ev('Turnover', 'Ball Down', 'Scoop'),
    ev('Try', '32', ''),                  ev('Try', '32', ''),
  ]);
  assert.deepEqual(s.moves.map(m => m.move), ['32', 'Scoop']);
});

test('topByTries ignores the attempts threshold', () => {
  const s = TR.strikeMoveStats([
    ev('Try', '32', ''), ev('Try', '32', ''), ev('Try', '32', ''),
    ev('Turnover', 'Ball Down', '32'), ev('Turnover', 'Ball Down', '32'),
    ev('Try', 'Scoop', ''),
  ]);
  assert.equal(s.topByTries.move, '32');
  assert.equal(s.topByTries.tries, 3);
});

test('topByRate needs MIN_MOVE_ATTEMPTS', () => {
  const s = TR.strikeMoveStats([
    ev('Try', 'Scoop', ''),                                     // 1/1 = 100%, only 1 attempt
    ev('Try', '32', ''), ev('Try', '32', ''),                   // 2/3 = 67%, 3 attempts
    ev('Turnover', 'Ball Down', '32'),
  ]);
  assert.equal(s.topByRate.move, '32');
  assert.equal(s.topByTries.move, '32');
});

test('topByRate is null when nothing clears the threshold', () => {
  const s = TR.strikeMoveStats([ev('Try', 'Scoop', '')]);
  assert.equal(s.topByRate, null);
  assert.equal(s.topByTries.move, 'Scoop');
});

test('topByTries is null when no move ever scored', () => {
  const s = TR.strikeMoveStats([ev('Turnover', 'Ball Down', '32')]);
  assert.equal(s.topByTries, null);
});

test('ties break on attempts then alphabetically', () => {
  const s = TR.strikeMoveStats([
    ev('Try', '23', ''), ev('Turnover', 'Ball Down', '23'),
    ev('Try', '21', ''), ev('Turnover', 'Ball Down', '21'),
  ]);
  assert.deepEqual(s.moves.map(m => m.move), ['21', '23']);
});
```

- [ ] **Step 2: Register the module so the tests can load it**

In `test.js:26`, add `'js/strike_moves.js'` to the end of the load list:

```js
for (const f of ['js/config.js', 'js/utils.js', 'js/events.js', 'js/possession.js', 'js/consistency.js', 'js/player.js', 'js/field_games.js', 'js/strike_moves.js']) {
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
node test.js
```

Expected: the run aborts with `ENOENT: no such file or directory, open 'js/strike_moves.js'`.

- [ ] **Step 4: Add `TR.EXCLUDED_MOVES`, then write the module**

First, in `js/events.js`, immediately after `TR.MIN_MOVE_ATTEMPTS`:

```js
// Selectable in the annotators, but never rate-bearing. On a Try these are what
// "the annotator skipped the picker" looks like — annotator_field2 filters
// 'Other' out of its sub-type strip, and Simple Mode names every Try 'Other' —
// while on a failure that same skip yields ''. Counted as untagged so they
// cannot sit at a 100% artefact rate and top both leaderboards. 'Interception'
// on a try means a defensive intercept, not a called move off the tap, and has
// no failure counterpart at all.
TR.EXCLUDED_MOVES = ['Other', 'Interception'];
```

Then create `js/strike_moves.js`:

```js
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
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
node test.js
```

Expected: PASS — every new assertion green, no regressions in the existing count.

- [ ] **Step 6: Register the module in the browser and the service worker**

In `tests.html`, after the `js/player.js` script tag, add:

```html
  <script src="js/strike_moves.js"></script>
```

In `sw.js`, add to `ASSETS` after `'js/consistency.js',`:

```js
  'js/strike_moves.js',
```

- [ ] **Step 7: Mirror the key assertions in the browser harness**

Append to `tests.html`, inside the `<script>` block:

```js
  QUnit.module('TR.strikeMoveStats', () => {
    const ev = (type, name, strikeMove) => ({ type, name, strikeMove, actionOwner: 'Team 1' });
    QUnit.test('rate over tries and fails', t => {
      const s = TR.strikeMoveStats([ev('Try', '32 - Cut', ''), ev('Turnover', 'Ball Down', '32 - Cut')]);
      t.equal(s.moves.length, 1);
      t.equal(s.moves[0].rate, 0.5);
      t.equal(s.moves[0].attempts, 2);
    });
    QUnit.test('coverage counts attack-ends only', t => {
      const s = TR.strikeMoveStats([
        ev('Try', 'Scoop', ''), ev('Turnover', 'Ball Down', '32'), ev('Turnover', 'Ball Down', ''),
        ev('Turnover', '6 Again', '32'), ev('Penalty Defence', 'Offside', '32'),
      ]);
      t.equal(s.coverage.total, 3);
      t.equal(s.coverage.tagged, 2);
    });
    QUnit.test('topByRate needs two attempts, topByTries does not', t => {
      const s = TR.strikeMoveStats([
        ev('Try', 'Scoop', ''),
        ev('Try', '32', ''), ev('Try', '32', ''), ev('Turnover', 'Ball Down', '32'),
      ]);
      t.equal(s.topByRate.move, '32');
      t.equal(s.topByTries.move, '32');
    });
  });
```

- [ ] **Step 8: Verify the browser harness passes**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/tests.html`. Expected: 0 failed. Stop the server afterwards.

- [ ] **Step 9: Commit**

```bash
git add js/strike_moves.js test.js tests.html sw.js
git commit -m "feat(analytics): add TR.strikeMoveStats"
```

---

### Task 2: Dashboard — the Strike Moves section

**Files:**
- Modify: `dashboard.html:170` (script tags)
- Modify: `dashboard.html:394-396` (column indices)
- Modify: `dashboard.html:422-441` (per-game accumulation)
- Modify: `dashboard.html:455-463` (per-game return value)
- Modify: `dashboard.html:293` area (new section markup)
- Modify: `dashboard.html:509-517` (`renderDashboard`)

**Interfaces:**
- Consumes: `TR.strikeMoveStats` (Task 1).
- Produces: `renderStrikeMoves()`; each entry in `gameStats` gains `moveEvents: {type, name, strikeMove, actionOwner}[]`.

Carrying a normalised `moveEvents` array on each game — rather than pre-aggregating counts — means the existing analysable-game filtering applies for free, and any later per-team or per-competition cut is a `filter` rather than a second accumulator.

- [ ] **Step 1: Load the modules**

Replace `dashboard.html:170`:

```html
<script src="js/config.js"></script>
```

with:

```html
<script src="js/config.js"></script>
<script src="js/events.js"></script>
<script src="js/strike_moves.js"></script>
```

- [ ] **Step 2: Resolve the new column**

At `dashboard.html:395`, replace:

```js
  const AOI = col('action owner'), GI = col('game');
```

with:

```js
  const AOI = col('action owner'), GI = col('game');
  const SMI = col('strike move');   // -1 on data pushed before Phase 1
```

- [ ] **Step 3: Collect the attempts per game**

At `dashboard.html:422` (with the other accumulator declarations), add:

```js
    const moveEvents    = [];
```

Then inside the `gRows.forEach` loop at `dashboard.html:426`, immediately after `const owner  = res(row[POI]);`, add:

```js
      // Every attack-ending event, normalised for TR.strikeMoveStats. Action
      // owner is the attacking team on a Try, Turnover and Pen Attack alike.
      if (TR.isAttackEnd(type, name)) {
        moveEvents.push({ type, name, strikeMove: SMI >= 0 ? (row[SMI] || '').trim() : '', actionOwner: action });
      }
```

And add `moveEvents,` to the returned object alongside `penTypes, turnTypes, tryTypes, teamTryTypes,` at `dashboard.html:462`.

- [ ] **Step 4: Add the section markup**

Immediately after the Team Detail section's closing markup around `dashboard.html:293`, add:

```html
      <div class="section">
        <div class="section-header" onclick="toggleSection('strike-moves-body')">
          <h2>Strike Moves</h2>
        </div>
        <div id="strike-moves-body" class="section-body">
          <div id="strike-move-tops" class="strike-move-tops"></div>
          <div id="strike-move-coverage" class="strike-move-coverage"></div>
          <table class="strike-move-table">
            <thead><tr>
              <th>Move</th><th>Tries</th><th>Fails</th><th>Attempts</th><th>Try rate</th><th></th>
            </tr></thead>
            <tbody id="strike-move-rows"></tbody>
          </table>
          <div id="strike-move-empty" class="strike-move-empty" style="display:none">
            No strike moves tagged yet. Tag the move on turnovers and attack penalties
            in any annotator, and these rates fill in.
          </div>
        </div>
      </div>
```

Add alongside the existing `.team-detail-*` rules:

```css
  .strike-move-tops { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }
  .strike-move-top { background: #141925; border-radius: 6px; padding: 8px 12px; flex: 1 1 200px; }
  .strike-move-top .label { font-size: 0.68rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.04em; }
  .strike-move-top .value { font-size: 1.05rem; font-weight: 600; }
  .strike-move-top .detail { font-size: 0.72rem; color: var(--dim); }
  .strike-move-coverage { font-size: 0.72rem; color: var(--dim); margin-bottom: 8px; }
  .strike-move-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  .strike-move-table th { text-align: left; color: var(--dim); font-weight: 500; padding: 4px 6px; }
  .strike-move-table td { padding: 4px 6px; border-top: 1px solid #1e2433; }
  .strike-move-bar { height: 6px; border-radius: 3px; background: #22c55e; }
  .strike-move-empty { font-size: 0.78rem; color: var(--dim); padding: 8px 0; }
```

- [ ] **Step 5: Render it**

Add `renderStrikeMoves();` to `renderDashboard` (`dashboard.html:509-517`), after `renderTeamDetail();`. Then add the function after `renderTeamDetail` ends:

```js
// ── Strike Moves ──────────────────────────────────────────────────
function renderStrikeMoves() {
  const stats  = TR.strikeMoveStats(gameStats.flatMap(g => g.moveEvents || []));
  const tops   = document.getElementById('strike-move-tops');
  const cover  = document.getElementById('strike-move-coverage');
  const body   = document.getElementById('strike-move-rows');
  const empty  = document.getElementById('strike-move-empty');
  const pct    = r => (r * 100).toFixed(1) + '%';

  if (!stats.moves.length) {
    tops.innerHTML = ''; cover.innerHTML = ''; body.innerHTML = '';
    empty.style.display = '';
    return;
  }
  empty.style.display = 'none';

  const tile = (label, m, detail) => !m ? '' : `<div class="strike-move-top">
      <div class="label">${label}</div>
      <div class="value">${m.move}</div>
      <div class="detail">${detail(m)}</div>
    </div>`;

  tops.innerHTML =
    tile('🏆 Most tries', stats.topByTries, m => `${m.tries} ${m.tries === 1 ? 'try' : 'tries'} from ${m.attempts} · ${pct(m.rate)}`) +
    tile('⚡ Best try rate', stats.topByRate, m => `${pct(m.rate)} · ${m.tries} from ${m.attempts}`);

  // Failure-side coverage is the number that matters: the Try side is 100% by
  // construction, so quoting only the combined figure flatters thin tagging.
  const f = stats.coverage.fails;
  cover.textContent =
    `Coverage: ${f.tagged} of ${f.total} failed attempts tagged ` +
    `(${Math.round(f.pct * 100)}%) · ` +
    `ranked by rate over at least ${TR.MIN_MOVE_ATTEMPTS} attempts · ` +
    `Other and Interception are not ranked`;

  const maxAttempts = Math.max(...stats.moves.map(m => m.attempts), 1);
  body.innerHTML = stats.moves.map(m => `<tr>
      <td>${m.move}</td>
      <td>${m.tries}</td>
      <td>${m.fails}</td>
      <td>${m.attempts}</td>
      <td>${m.attempts >= TR.MIN_MOVE_ATTEMPTS ? pct(m.rate) : `<span style="color:var(--dim)">${pct(m.rate)}</span>`}</td>
      <td style="width:35%"><div class="strike-move-bar" style="width:${Math.round(m.attempts / maxAttempts * 100)}%;opacity:${0.35 + m.rate * 0.65}"></div></td>
    </tr>`).join('');
}
```

The rate is dimmed rather than hidden below the threshold: the number is real, it just shouldn't be read as a ranking.

- [ ] **Step 6: Verify manually**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/dashboard.html`, log in with `m30-admin`. Expected:

1. A collapsible **Strike Moves** section that expands and collapses like its neighbours.
2. If no moves are tagged yet: the empty-state message, no table rows, no console errors.
3. If moves are tagged: two tiles, a coverage line, and one row per move with the rate dimmed on any move under 2 attempts.
4. The row count equals the number of distinct moves — no zero-attempt rows.

- [ ] **Step 7: Commit**

```bash
git add dashboard.html
git commit -m "feat(dashboard): strike move try rates and top scoring move"
```

---

### Task 3: Dashboard — add the rate to the Team Detail card

**Files:**
- Modify: `dashboard.html:483-491` (`aggregateTeams`)
- Modify: `dashboard.html:527-558` (`renderTeamDetail`)

**Interfaces:**
- Consumes: `TR.strikeMoveStats` (Task 1); `moveEvents` on each game (Task 2).
- Produces: each entry in `teamStats` gains `moveEvents`.

The card **keeps its existing top-3-by-tries ranking**. Re-ranking it by rate was considered and rejected in the spec: with per-team splits most teams won't clear 2 attempts on a move and the card would go empty. The rate is appended as a second figure — purely additive, so nothing that works today regresses.

- [ ] **Step 1: Carry the attempts through to each team**

In `aggregateTeams`, add `moveEvents: [],` to the initialiser at `dashboard.html:483` alongside `tryTypes: {},`. Then after the `Object.entries(g.teamTryTypes[team] || {})` block at `dashboard.html:489-491`, add:

```js
      // Only this team's own attempts: on a Try, Turnover and Pen Attack alike
      // the action owner is the attacking team that ran the move.
      (g.moveEvents || []).forEach(e => { if (e.actionOwner === team) t.moveEvents.push(e); });
```

- [ ] **Step 2: Show the rate next to each move**

In `renderTeamDetail`, replace the `const top3` line and the `rows` expression (`dashboard.html:534-551`) with:

```js
    const moveStats = TR.strikeMoveStats(t.moveEvents || []);
    const rateOf    = new Map(moveStats.moves.map(m => [m.move, m]));
    const top3  = Object.entries(t.tryTypes)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    const maxCount = top3.length ? top3[0][1] : 1;

    const rows = top3.length
      ? top3.map(([name, count], i) => {
          const pct     = total ? Math.round(count / total * 100) : 0;
          const barPct  = Math.round(count / maxCount * 100);
          const m       = rateOf.get(name);
          // Only shown once this team has enough attempts on the move for the
          // rate to mean anything; the share-of-tries figure is always there.
          const rateTxt = m && m.attempts >= TR.MIN_MOVE_ATTEMPTS
            ? `<span class="td-rate" title="${m.tries} from ${m.attempts} attempts">${Math.round(m.rate * 100)}% scored</span>`
            : '';
          return `<div class="td-row">
            <span class="td-rank">${i + 1}</span>
            <span class="td-try-name">${name}</span>
            <div class="td-bar-wrap"><div class="td-bar" style="width:${barPct}%"></div></div>
            <span class="td-pct">${pct}%</span>
            ${rateTxt}
          </div>`;
        }).join('')
      : `<div class="td-empty">No try data</div>`;
```

Add alongside the other `.td-*` rules:

```css
  .td-rate { font-size: 0.66rem; color: #22c55e; white-space: nowrap; }
```

`.td-row` at `dashboard.html:50` is `display: flex` with `gap: 6px`, so the extra span flows on the end with no layout change needed — do not add a grid column.

- [ ] **Step 2b: Run the tests**

```bash
node test.js
```

Expected: PASS.

- [ ] **Step 3: Verify manually**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/dashboard.html`. Expected:

1. The Team Detail cards still list each team's top 3 try types in the same order as before this change.
2. Teams with ≥2 tagged attempts on a listed move show a green `N% scored` figure; teams without show nothing extra.
3. With no moves tagged anywhere, the cards look exactly as they did before Phase 2.

- [ ] **Step 4: Commit**

```bash
git add dashboard.html
git commit -m "feat(dashboard): show try rate on team detail moves"
```

---

### Task 4: Game Analysis — per-game move breakdown

**Files:**
- Modify: `games.html:201` (script tags)
- Modify: `games.html:532-542` (row normalisation)
- Modify: `games.html:281-284` (new card after Team Statistics)
- Modify: `games.html:1144` (`renderStats`, the per-half render coordinator)

**Interfaces:**
- Consumes: `TR.strikeMoveStats` (Task 1).
- Produces: `renderGameStrikeMoves(events, team1, team2)`; each event object gains a `'Strike Move'` key.

Named `renderGameStrikeMoves`, not `renderStrikeMoves`: `dashboard.html` defines a zero-argument `renderStrikeMoves` in Task 2, and the two are read together often enough that the collision would mislead.

Single-game samples are small, so this leads with counts and treats the rate as secondary. The coverage line is always present rather than conditional — a per-game table that hides its own thinness invites over-reading.

- [ ] **Step 1: Load the modules**

Replace `games.html:201`:

```html
<script src="js/config.js"></script>
```

with:

```html
<script src="js/config.js"></script>
<script src="js/events.js"></script>
<script src="js/strike_moves.js"></script>
```

- [ ] **Step 2: Carry the column through normalisation**

At `games.html:533`, replace:

```js
  const pi = col('possession owner'), ai = col('action owner'), ci = col('comment');
```

with:

```js
  const pi = col('possession owner'), ai = col('action owner'), ci = col('comment');
  const si = col('strike move');   // -1 on games pushed before Phase 1
```

and add to the returned object at `games.html:541`, after `Comment:`:

```js
    'Strike Move':      si >= 0 ? (r[si] || '') : '',
```

- [ ] **Step 3: Add the section markup**

Insert a new card immediately after the Team Statistics card, which ends at `games.html:284`:

```html
    <div class="card">
      <div class="card-label">Strike Moves</div>
      <div id="game-move-coverage" class="game-move-coverage"></div>
      <div id="game-move-grid" class="game-move-grid"></div>
    </div>
```

Note it uses `<div class="card-label">`, matching its neighbours at `games.html:282` — this page has no `<h3>` convention.

with:

```css
  .game-move-coverage { font-size: 0.72rem; color: #8a93a6; margin-bottom: 8px; }
  .game-move-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .game-move-grid table { width: 100%; border-collapse: collapse; font-size: 0.76rem; }
  .game-move-grid th { text-align: left; color: #8a93a6; font-weight: 500; padding: 3px 5px; }
  .game-move-grid td { padding: 3px 5px; border-top: 1px solid #1e2433; }
  .game-move-rate { color: #8a93a6; }
  @media (max-width: 700px) { .game-move-grid { grid-template-columns: 1fr; } }
```

- [ ] **Step 4: Render it**

Add this function, and call it from `renderStats` (`games.html:1144`) — near where `metricsTable` is populated, passing `filtered` (the half-scoped event list that function already computes) plus `team1` and `team2`:

```js
  renderGameStrikeMoves(filtered, team1, team2);
```

The function itself:

```js
// Per-game move breakdown, split by the team that ran the move. Counts lead;
// the rate is deliberately secondary, because one game rarely gives a move
// enough attempts for a percentage to mean much.
function renderGameStrikeMoves(events, team1, team2) {
  const all   = TR.strikeMoveStats(events);
  const cover = document.getElementById('game-move-coverage');
  const grid  = document.getElementById('game-move-grid');

  if (!all.moves.length) {
    cover.textContent = '';
    grid.innerHTML = '<div style="color:#8a93a6;font-size:0.78rem">No strike moves tagged in this game.</div>';
    return;
  }

  const f = all.coverage.fails;
  cover.textContent =
    `Coverage: ${f.tagged} of ${f.total} failed attempts tagged ` +
    `(${Math.round(f.pct * 100)}%). Single-game rates are noisy — read the counts first.`;

  grid.innerHTML = [team1, team2].map(team => {
    const s = TR.strikeMoveStats(events.filter(e => e['Action Owner'] === team));
    const rows = s.moves.length
      ? s.moves.map(m => `<tr>
          <td>${m.move}</td><td>${m.tries}</td><td>${m.fails}</td>
          <td class="game-move-rate">${(m.rate * 100).toFixed(0)}%</td>
        </tr>`).join('')
      : `<tr><td colspan="4" style="color:#8a93a6">No moves tagged</td></tr>`;
    return `<div>
      <table>
        <thead><tr><th>${team}</th><th>Tries</th><th>Fails</th><th>Rate</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  }).join('');
}
```

- [ ] **Step 5: Verify manually**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/games.html`, log in with `m30-admin`, and select a game. Expected:

1. A **Strike Moves** card with one table per team.
2. On a game tagged before Phase 1: the "No strike moves tagged in this game." message, no errors.
3. On a tagged game: each team's own moves only, with the coverage line above.
4. Narrow the window below 700px. Expected: the two tables stack.

- [ ] **Step 6: Commit**

```bash
git add games.html
git commit -m "feat(games): per-game strike move breakdown by team"
```

---

### Task 5: Analytics — strike move as a filter dimension

**Files:**
- Modify: `analytics.html:219` (script tags)
- Modify: `analytics.html:577-589` (index map)
- Modify: `analytics.html:605-616` (event normalisation)
- Modify: `analytics.html:967-990` (`renderFilteredTypeChart`)

**Interfaces:**
- Consumes: `TR.strikeMoveStats` (Task 1).
- Produces: every event in `ALL_EVENTS` gains `strikeMove: string`.

- [ ] **Step 1: Load the module**

After `analytics.html:219` (`<script src="js/events.js"></script>`), add:

```html
<script src="js/strike_moves.js"></script>
```

- [ ] **Step 2: Resolve and carry the column**

In the index map at `analytics.html:582`, after `action: idx('action owner'),`, add:

```js
    strikeMove: idx('strike move'),
```

Then in the returned event object at `analytics.html:606-607`, after `type, name,`, add:

```js
      strikeMove: I.strikeMove >= 0 ? (r[I.strikeMove] || '').trim() : '',
```

- [ ] **Step 3: Break down by move when a strike-move type is selected**

In `renderFilteredTypeChart`, replace the `if (activeType)` block's first two lines (`analytics.html:970-972`):

```js
  if (activeType) {
    const byName = countBy(evs, e => e.name || '(blank)');
    const data = Object.entries(byName).map(([name, count]) => ({ name, count }));
```

with:

```js
  if (activeType) {
    // For the three types that end an attack, breaking down by strike move
    // answers a different and more useful question than breaking down by name.
    const byMove = document.getElementById('fBreakdownByMove')?.checked;
    const useMove = byMove && TR.STRIKE_MOVE_TYPES.includes(activeType);
    const counts = countBy(evs, e => useMove
      ? (TR.strikeMoveOf(e.type, e.name, e.strikeMove) || '(untagged)')
      : (e.name || '(blank)'));
    const data = Object.entries(counts).map(([name, count]) => ({ name, count }));
```

and change the chart title on the following lines from `` `${activeType} — by name` `` to:

```js
      title: { text: `${activeType} — by ${useMove ? 'strike move' : 'name'}`, color: '#aaa', fontSize: 11, fontWeight: 600, anchor: 'start' },
```

- [ ] **Step 4: Add the toggle**

Beside the existing type-filter controls, add:

```html
      <label class="filter-toggle" title="Only applies to Try, Turnover and Penalty Attack">
        <input type="checkbox" id="fBreakdownByMove" onchange="renderBottom()"> by strike move
      </label>
```

`renderBottom()` at `analytics.html:931` is the function that re-runs the whole filtered panel — it calls `renderFilteredTypeChart(evs, byType, t)` at `analytics.html:960`, so routing the checkbox to it re-renders the chart with the new setting.

- [ ] **Step 5: Verify manually**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/analytics.html`, log in with `m30-admin`. Expected:

1. Select type `Turnover`, tick **by strike move**. Expected: the chart re-titles to `Turnover — by strike move` and bars are moves, with `(untagged)` for the rest.
2. Untick it. Expected: the chart returns to the by-name breakdown.
3. Select `Penalty Defence` with the box ticked. Expected: it stays on the by-name breakdown, since that type never carries a move.
4. No console errors on data with no moves tagged.

- [ ] **Step 6: Commit**

```bash
git add analytics.html
git commit -m "feat(analytics): break failed attempts down by strike move"
```

---

### Task 6: Field annotators — a Strike Moves sheet off the Stats sheet

**Files:**
- Modify: `annotator_field.html:718` and `annotator_field2.html:982` (script tags)
- Modify: `annotator_field.html:929-947` and the matching `#statsOverlay` markup in `annotator_field2.html`
- Modify: `annotator_field.html:1828` and the matching `refreshStats` in `annotator_field2.html`

**Interfaces:**
- Consumes: `TR.strikeMoveStats` (Task 1).
- Produces: `openMoves()`, `closeMoves()`, `renderMoves()` local to each page.

**Owner's decision (2026-09-07), which replaces an earlier inline-block design:**
the moves do **not** sit inline in the Stats sheet. The Stats sheet gets a
**tappable row** which opens a **second overlay on top of it**, listing every move
tried **in order of success**. Rationale: the Stats sheet is already dense, and the
move table is a different question ("what is working?") from the team comparison
("who is winning?") — it deserves its own surface rather than another block to
scroll past.

Both files get the same code. Repeat it rather than sharing — neither page loads
the other, and the existing stats code is already duplicated between them.

- [ ] **Step 1: Load the module in both pages**

In `annotator_field.html` after line 719 (`js/possession.js`) and in
`annotator_field2.html` after line 983, add:

```html
<script src="js/strike_moves.js"></script>
```

- [ ] **Step 2: Add the tappable row to both Stats sheets**

`annotator_field.html:944-946` is the tail of the stats card:

```html
    <div class="stats-status" id="statsStatus"></div>
    <div id="statsBody"></div>
    <button class="stats-close-btn" onclick="closeStats()">Close</button>
```

Insert the row between `statsBody` and the Close button, in both files:

```html
      <button class="stats-drill" id="movesDrill" onclick="openMoves()">
        <span class="stats-drill-label">Strike moves</span>
        <span class="stats-drill-meta" id="movesDrillMeta"></span>
        <span class="stats-drill-chev">›</span>
      </button>
```

`#movesDrillMeta` carries the summary so the row is worth tapping — the best move
and the tagged count — set in Step 4. Style it as a row, not a card, so it reads
as a way further in rather than as another statistic:

```css
  .stats-drill { display: flex; align-items: center; gap: 10px; width: 100%;
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
    color: var(--text); font: inherit; text-align: left; padding: 12px 14px;
    margin-top: 10px; cursor: pointer; min-height: 52px; }
  .stats-drill:active { background: var(--surface); }
  .stats-drill-label { font-weight: 600; font-size: 0.9rem; }
  .stats-drill-meta { margin-left: auto; font-size: 0.72rem; color: var(--text-dim);
    text-align: right; }
  .stats-drill-chev { color: var(--text-dim); font-size: 1.1rem; }
```

- [ ] **Step 3: Add the second overlay**

Add a sibling overlay after `#statsOverlay` closes, in both files. It layers above
the Stats sheet — opening it does not close Stats, so Back returns there.

```html
<div id="movesOverlay" onclick="if(event.target===this)closeMoves()">
  <div id="movesCard">
    <div class="stats-handle"></div>
    <div class="moves-head">
      <button class="moves-back" onclick="closeMoves()">‹ Stats</button>
      <div class="moves-title">Strike moves</div>
    </div>
    <div class="moves-cov" id="movesCov"></div>
    <div id="movesBody"></div>
    <button class="stats-close-btn" onclick="closeMoves()">Close</button>
  </div>
</div>
```

Match `#statsOverlay`'s own positioning rules (read them first — they differ
between phone and the wider breakpoint) and give it a higher `z-index` so it sits
above the Stats sheet rather than beside it:

```css
  #movesOverlay { position: fixed; inset: 0; z-index: 60; display: none;
    background: rgba(0,0,0,0.55); align-items: flex-end; justify-content: center; }
  #movesOverlay.open { display: flex; }
  #movesCard { background: var(--surface); width: 100%; max-width: 560px;
    border-radius: 16px 16px 0 0; padding: 0 16px 16px; max-height: 88dvh;
    overflow-y: auto; }
  .moves-head { display: flex; align-items: center; gap: 10px; padding: 4px 0 10px; }
  .moves-back { background: none; border: none; color: var(--accent); font: inherit;
    font-size: 0.86rem; padding: 6px 0; cursor: pointer; }
  .moves-title { font-weight: 600; font-size: 1rem; margin-left: auto;
    margin-right: auto; padding-right: 48px; }
  .moves-cov { font-size: 0.7rem; color: var(--text-dim); padding-bottom: 8px; }
  .moves-row { display: grid; grid-template-columns: 22px 1fr auto 46px; gap: 10px;
    align-items: center; padding: 9px 0; border-top: 1px solid var(--border);
    font-size: 0.84rem; }
  .moves-rank { color: var(--text-dim); font-size: 0.72rem;
    font-variant-numeric: tabular-nums; }
  .moves-bar { grid-column: 2 / 5; height: 4px; border-radius: 2px;
    background: var(--border); margin-top: -4px; }
  .moves-bar span { display: block; height: 100%; border-radius: 2px;
    background: var(--try, #22c55e); }
  .moves-count { color: var(--text-dim); font-size: 0.76rem;
    font-variant-numeric: tabular-nums; }
  .moves-rate { text-align: right; font-weight: 600;
    font-variant-numeric: tabular-nums; }
  .moves-rate.thin { color: var(--text-dim); font-weight: 400; }
  .moves-empty { color: var(--text-dim); font-size: 0.82rem; padding: 20px 0;
    text-align: center; }
```

- [ ] **Step 4: Render both surfaces**

`TR.strikeMoveStats` already returns `moves` sorted by rate descending, which is
"in order of success" — do not re-sort. Add `renderMoves();` at the end of
`refreshStats` in each file (it refreshes the drill row's summary, and the overlay
too when it is open, so the 5-second tick keeps both live), then add:

```js
// The Stats sheet answers "who is winning"; this answers "what is working", so it
// gets its own surface rather than another block to scroll past. Sorted by rate
// descending straight from the module — that is the "order of success".
// In-game samples are tiny, so a rate below MIN_MOVE_ATTEMPTS is shown but
// de-emphasised, and the raw tries/attempts count always sits beside it.
function renderMoves() {
  const s = TR.strikeMoveStats(annotations);

  const meta = document.getElementById('movesDrillMeta');
  if (meta) {
    const best = s.topByRate || s.topByTries;
    meta.textContent = best
      ? `${best.move} · ${Math.round(best.rate * 100)}%`
      : (s.coverage.fails.total ? 'none tagged yet' : '—');
  }

  const cov  = document.getElementById('movesCov');
  const body = document.getElementById('movesBody');
  if (!cov || !body) return;

  if (!s.moves.length) {
    cov.textContent = '';
    body.innerHTML = '<div class="moves-empty">No moves tagged yet. Pick a move '
      + 'after a turnover or attack penalty and they show up here.</div>';
    return;
  }

  // Failure-side coverage is the honest number: every try has a name, so the
  // try side reads 100% by construction.
  const f = s.coverage.fails;
  cov.textContent = `${f.tagged} of ${f.total} failed attempts tagged`
    + ` · best first · ${TR.MIN_MOVE_ATTEMPTS}+ attempts to rank`;

  body.innerHTML = s.moves.map((m, i) => {
    const thin = m.attempts < TR.MIN_MOVE_ATTEMPTS;
    return `<div class="moves-row">
        <span class="moves-rank">${i + 1}</span>
        <span>${m.move}</span>
        <span class="moves-count">${m.tries}/${m.attempts}</span>
        <span class="moves-rate${thin ? ' thin' : ''}">${Math.round(m.rate * 100)}%</span>
      </div>
      <div class="moves-bar"><span style="width:${Math.round(m.rate * 100)}%"></span></div>`;
  }).join('');
}

function openMoves() {
  renderMoves();
  document.getElementById('movesOverlay').classList.add('open');
}

function closeMoves() {
  document.getElementById('movesOverlay').classList.remove('open');
}
```

- [ ] **Step 5: Verify manually**

Serve the repo and open `annotator_field.html` at a 375px width, logged in as
`m30-admin`. Confirm:

1. Open Stats on a game with no moves tagged. Expected: the **Strike moves** row is
   present showing `—`, and tapping it opens the overlay with the empty-state text.
2. Tag a turnover, set its move to `32 - Cut`, then score a try with `32 - Cut`.
   Reopen. Expected: the drill row reads `32 - Cut · 50%`, and the overlay lists
   `1  32 - Cut  1/2  50%` with a half-width bar.
3. Tag a second move with a worse rate. Expected: it sorts **below** the better one.
4. A move with a single attempt. Expected: its rate is de-emphasised, its count
   still legible.
5. Tap **‹ Stats** and the scrim. Expected: both return to the Stats sheet, which is
   still open behind — not to the game.
6. Leave the overlay open for 5+ seconds after tagging. Expected: it refreshes.
7. Repeat all of it in `annotator_field2.html`.

- [ ] **Step 6: Commit**

```bash
git add annotator_field.html annotator_field2.html
git commit -m "feat(field-annotators): a strike moves sheet off the stats sheet"
```

---

### Task 7: Refresh the cache and document the analytics

**Files:**
- Modify: `sw.js:8` (`CACHE_VERSION`)
- Modify: `README.md`
- Modify: `FIELD_ANNOTATOR.md`, `FIELD_ANNOTATOR_V2.md`

- [ ] **Step 1: Bump the cache**

In `sw.js:8`, increment `CACHE_VERSION` by one (e.g. `'trl-shell-v13'` → `'trl-shell-v14'`). Confirm `js/strike_moves.js` is in `ASSETS` from Task 1 Step 6.

- [ ] **Step 2: Document each surface**

In `README.md`:

- Under **Dashboard**, add: a **Strike Moves** section showing try rate per move across all analysable games, with the top scoring move by volume and by rate (rate needs at least 2 attempts), plus a coverage figure for how many attempts were tagged.
- Under **Game Analysis**, add: a per-game **Strike Moves** card, split by team.
- Under **Event Viewer**, note that `Strike Move` is filterable.

In `FIELD_ANNOTATOR.md` and `FIELD_ANNOTATOR_V2.md`, add to the Stats-sheet sections: the sheet carries a tappable **Strike moves** row showing the best move at a glance, which opens a second sheet listing every move tried in order of success — tries-over-attempts beside a rate, de-emphasised below 2 attempts. Say why it is a separate sheet: the Stats sheet answers who is winning, this answers what is working.

Add to the top of the Dashboard entry in `README.md`:

```markdown
> Try rates only count attempts where a move was actually tagged. The coverage
> figure says how many that is — a 60% rate over 15% coverage is a much weaker
> claim than the same rate over 80%.
```

- [ ] **Step 3: Run the full suite**

```bash
node test.js
```

Expected: PASS.

- [ ] **Step 4: Verify the browser harness**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/tests.html`. Expected: 0 failed.

- [ ] **Step 5: Commit**

```bash
git add sw.js README.md FIELD_ANNOTATOR.md FIELD_ANNOTATOR_V2.md
git commit -m "docs: document strike move analytics; bump shell cache"
```

---

## Phase 2 exit criteria

- [ ] `node test.js` passes, including every `TR.strikeMoveStats` assertion.
- [ ] `tests.html` reports 0 failures.
- [ ] Dashboard shows the Strike Moves section with both top-move tiles and a coverage line.
- [ ] Team Detail cards keep their existing ranking and gain a rate where the sample allows.
- [ ] Game Analysis shows a per-game, per-team breakdown.
- [ ] Analytics can break Try / Turnover / Penalty Attack down by strike move.
- [ ] Both field annotators have a tappable Strike moves row in Stats that opens a second sheet listing every move in order of success, and Back returns to Stats rather than to the game.
- [ ] Every surface degrades cleanly to an empty state on data tagged before Phase 1.
- [ ] The formula exists only in `js/strike_moves.js` — `grep -rn "tries.*attempts\|/ *m.attempts" *.html` finds no second implementation.
