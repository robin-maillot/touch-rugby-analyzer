# Strike Moves — Phase 1 (Capture) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record which attacking strike move was being run when a turnover or attack penalty ends an attack, so Phase 2 can compute try rate per move.

**Architecture:** One new `Strike Move` column on the Google Sheet, appended last. Two new pure predicates in `js/events.js` that all three annotators share. An optional second-stage picker in each annotator, skippable everywhere, that writes an optional `strikeMove` field onto the in-memory event.

**Tech Stack:** Vanilla ES2020 in static HTML pages, no build step. Shared `TR.*` modules under `js/`. Google Apps Script backend (`apps_script/Code.gs`). Node-based assertions in `test.js`, mirrored as QUnit in `tests.html`.

**Spec:** `docs/superpowers/specs/2026-09-06-strike-moves-design.md`

## Global Constraints

- No build step. Plain `<script src>` tags; every shared helper hangs off the global `TR` namespace declared in `js/config.js`.
- New shared JS files must be added to **four** places: `test.js` load list, `tests.html` script tags, `sw.js` `ASSETS`, and every consuming page's script tags.
- `TR.STRIKE_MOVES` is exactly `TR.MENU['Try']` — all 19 entries, including `Other` and `Interception`. Do not filter it.
- `TR.MIN_MOVE_ATTEMPTS = 2`.
- The strike move is **always optional**. Every picker must be skippable and skipping must leave `strikeMove` as `''`.
- Never offer or store a move on `Turnover → 6 Again` or on `Penalty Defence`: possession does not change, so the attack is still running.
- A `Try`'s move is its own `Name`, derived at write time. Never store it twice on the event.
- All reads of `strikeMove` coalesce with `|| ''` — games already in `localStorage` have no such field and must not be migrated.
- **Deploy order is load-bearing:** Task 2 (Apps Script) must be deployed to production *before* any of Tasks 3–8 ship.

---

### Task 1: Shared strike-move constants and predicates

**Files:**
- Modify: `js/events.js:23` (append after `TR.isTurnover`)
- Test: `test.js` (append a new section after the `TR.isTurnover` section)
- Test: `tests.html` (append a QUnit module after the `TR.isTurnover` module)

**Interfaces:**
- Consumes: `TR.MENU`, `TR.isTurnover` (both already in `js/events.js`).
- Produces:
  - `TR.STRIKE_MOVES: string[]` — a copy of `TR.MENU['Try']`.
  - `TR.STRIKE_MOVE_TYPES: string[]` — `['Try', 'Turnover', 'Penalty Attack']`.
  - `TR.MIN_MOVE_ATTEMPTS: number` — `2`.
  - `TR.isAttackEnd(type: string, name: string) => boolean`
  - `TR.strikeMoveOf(type: string, name: string, strikeMove: string|undefined) => string`

- [ ] **Step 1: Write the failing tests**

Append to `test.js`, after the `TR.isTurnover` section:

```js
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

// ── TR.strikeMoveOf ───────────────────────────────────────────
console.log('TR.strikeMoveOf');
test('Try returns its own name',    () => assert.equal(TR.strikeMoveOf('Try', '33 - Quicky', ''), '33 - Quicky'));
test('Try ignores a stored move',   () => assert.equal(TR.strikeMoveOf('Try', '33 - Quicky', 'Scoop'), '33 - Quicky'));
test('Turnover returns its move',   () => assert.equal(TR.strikeMoveOf('Turnover', 'Ball Down', '32 - Cut'), '32 - Cut'));
test('Pen Attack returns its move', () => assert.equal(TR.strikeMoveOf('Penalty Attack', 'Forward Pass', '23'), '23'));
test('untagged returns empty',      () => assert.equal(TR.strikeMoveOf('Turnover', 'Ball Down', ''), ''));
test('undefined move returns empty',() => assert.equal(TR.strikeMoveOf('Turnover', 'Ball Down', undefined), ''));
test('6 Again drops its move',      () => assert.equal(TR.strikeMoveOf('Turnover', '6 Again', '32'), ''));
test('Pen Defence drops its move',  () => assert.equal(TR.strikeMoveOf('Penalty Defence', 'Offside', '32'), ''));
test('Game Event drops its move',   () => assert.equal(TR.strikeMoveOf('Game Event', 'Game Start', '32'), ''));
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
node test.js
```

Expected: FAIL — `✗ matches the Try menu` with `Expected values to be loosely deep-equal` (`TR.STRIKE_MOVES` is `undefined`), and `✗ Try ends an attempt` with `TR.isAttackEnd is not a function`.

- [ ] **Step 3: Write the implementation**

Append to `js/events.js`, after the `TR.isTurnover` definition:

```js
// ── Strike moves ───────────────────────────────────────────────
// The attacking move an attempt was running. A Try already records it in Name;
// these let a Turnover or Penalty Attack record the move that failed, so a try
// rate per move can be computed. Sliced so a caller can't mutate TR.MENU.
TR.STRIKE_MOVES      = TR.MENU['Try'].slice();
TR.STRIKE_MOVE_TYPES = ['Try', 'Turnover', 'Penalty Attack'];
TR.MIN_MOVE_ATTEMPTS = 2;

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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
node test.js
```

Expected: PASS — all new assertions green, and the pre-existing count unchanged (no regressions).

- [ ] **Step 5: Mirror the assertions in the browser harness**

Append to `tests.html`, inside the `<script>` block after the `TR.isTurnover` module:

```js
  QUnit.module('TR.strikeMoveOf', () => {
    QUnit.test('a Try uses its own name', t => {
      t.equal(TR.strikeMoveOf('Try', '33 - Quicky', ''), '33 - Quicky');
      t.equal(TR.strikeMoveOf('Try', '33 - Quicky', 'Scoop'), '33 - Quicky');
    });
    QUnit.test('a failed attempt uses its stored move', t => {
      t.equal(TR.strikeMoveOf('Turnover', 'Ball Down', '32 - Cut'), '32 - Cut');
      t.equal(TR.strikeMoveOf('Penalty Attack', 'Forward Pass', '23'), '23');
    });
    QUnit.test('events that keep possession carry no move', t => {
      t.equal(TR.strikeMoveOf('Turnover', '6 Again', '32'), '');
      t.equal(TR.strikeMoveOf('Penalty Defence', 'Offside', '32'), '');
      t.equal(TR.strikeMoveOf('Game Event', 'Game Start', '32'), '');
    });
    QUnit.test('untagged is empty', t => {
      t.equal(TR.strikeMoveOf('Turnover', 'Ball Down', ''), '');
      t.equal(TR.strikeMoveOf('Turnover', 'Ball Down', undefined), '');
    });
  });

  QUnit.module('TR.isAttackEnd', () => {
    QUnit.test('true only when the ball changes hands', t => {
      t.equal(TR.isAttackEnd('Try', '32 - Cut'), true);
      t.equal(TR.isAttackEnd('Penalty Attack', 'Forward Pass'), true);
      t.equal(TR.isAttackEnd('Turnover', 'Ball Down'), true);
      t.equal(TR.isAttackEnd('Turnover', '6 Again'), false);
      t.equal(TR.isAttackEnd('Penalty Defence', 'Offside'), false);
      t.equal(TR.isAttackEnd('Game Event', 'Game Start'), false);
    });
  });
```

- [ ] **Step 6: Verify the browser harness passes**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/tests.html`. Expected: 0 failed. Stop the server afterwards.

- [ ] **Step 7: Commit**

```bash
git add js/events.js test.js tests.html
git commit -m "feat(events): add strike move constants and predicates"
```

---

### Task 2: Add the Strike Move column to the sheet backend

**Files:**
- Modify: `apps_script/Code.gs:14`

**Interfaces:**
- Consumes: nothing.
- Produces: an 8th canonical column named `Strike Move`, returned by `action=all` and written into every newly created game tab.

**Why this ships alone and first:** the client pushes 8-column rows, but a new tab takes its header row from `HEADERS` at `apps_script/Code.gs:597`. Ship the client first and `setValues` writes an 8th column of data under a 7-column header — the data lands but `action=all` silently drops it, because `Code.gs:377` maps columns by header name. Deploy this, then ship the rest.

- [ ] **Step 1: Add the column**

In `apps_script/Code.gs`, replace line 14:

```js
const HEADERS = ['Time', 'Possession Owner', 'Type', 'Name', 'To Review', 'Comment', 'Action Owner'];
```

with:

```js
// Strike Move is appended LAST so the Python pipeline's positional reads of
// columns 0-6 are unaffected. Every read path maps by header name, so tabs
// written before this column existed simply report '' for it.
const HEADERS = ['Time', 'Possession Owner', 'Type', 'Name', 'To Review', 'Comment', 'Action Owner', 'Strike Move'];
```

- [ ] **Step 2: Commit**

```bash
git add apps_script/Code.gs
git commit -m "feat(sheet): add Strike Move column to canonical headers"
```

- [ ] **Step 3: Deploy and verify manually**

Open the Apps Script editor → **Deploy → Manage deployments → edit the existing deployment → New version → Deploy**. The web app URL is unchanged.

Then verify the header reaches clients:

```bash
curl -s "$(grep -o 'https://script.google.com[^'"'"']*' js/config.js | head -1)?action=all&secret=m30-admin" | head -c 400
```

Expected: the first row of `rows` ends `…,"Action Owner","Strike Move","Game",…`.

- [ ] **Step 4: Check the Python pipeline**

`apps_script/Code.gs:13` says the column order "must match what the Python pipeline reads". The pipeline lives in the gitignored `experiments/` directory, so it is not verifiable from the repo. Confirm it reads columns by header name or by index 0-6 only:

```bash
grep -rn "Action Owner\|iloc\[\|columns\[" experiments/ --include=*.py | head -20
```

Expected: either header-name lookups (safe) or positional reads bounded at index 6 (safe, since `Strike Move` is index 7). If anything reads a fixed **column count** or slices to the end of the row, update it to tolerate the extra column before proceeding.

**Do not start Task 3 until this deployment is live.**

---

### Task 3: `annotator.html` — persist the strike move

**Files:**
- Modify: `annotator.html:830-831` (the `tag` function)
- Modify: `annotator.html:1207-1218` (push rows)
- Modify: `annotator.html:1070` and `annotator.html:1083-1085` (load from sheet)
- Modify: `annotator.html:1285-1296` (CSV export)
- Modify: `annotator.html:1319` and `annotator.html:1327-1333` (CSV import)

**Interfaces:**
- Consumes: `TR.strikeMoveOf` (Task 1).
- Produces: annotation objects in `annotator.html` carry an optional `strikeMove: string` field; `tag()` gains a fifth parameter.

This task is persistence only — nothing sets a non-empty move yet, so the round trip is provably lossless before any UI exists to exercise it.

- [ ] **Step 1: Widen `tag()` to carry a move**

In `annotator.html`, replace the signature and push at lines 825 and 831:

```js
function tag(type, name, possessionOwner, comment = '') {
```

with:

```js
function tag(type, name, possessionOwner, comment = '', strikeMove = '') {
```

and:

```js
  annotations.push({ id: Date.now() + Math.random(), type, name, possessionOwner, actionOwner, comment, time: t, timeStr: TR.fmt(t) });
```

with:

```js
  annotations.push({ id: Date.now() + Math.random(), type, name, possessionOwner, actionOwner, comment, strikeMove, time: t, timeStr: TR.fmt(t) });
```

- [ ] **Step 2: Add the column to the push payload**

Replace lines 1207-1218:

```js
  // Column order must match HEADERS in Code.gs: Time, Possession Owner, Type, Name, To Review, Comment, Action Owner
  const rows = annotations.map(a => {
    const outType = a.type;
    return [
      a.timeStr,
      a.possessionOwner || '',
      outType,
      a.name || '',
      a.type === 'To Review' ? 'Yes' : '',
      a.comment || '',
      a.actionOwner || '',
    ];
  });
```

with:

```js
  // Column order must match HEADERS in Code.gs: Time, Possession Owner, Type, Name, To Review, Comment, Action Owner, Strike Move
  const rows = annotations.map(a => {
    const outType = a.type;
    return [
      a.timeStr,
      a.possessionOwner || '',
      outType,
      a.name || '',
      a.type === 'To Review' ? 'Yes' : '',
      a.comment || '',
      a.actionOwner || '',
      TR.strikeMoveOf(a.type, a.name, a.strikeMove),
    ];
  });
```

- [ ] **Step 3: Read the column back from the sheet**

At line 1070, replace:

```js
    const commentIdx   = col('comment'), actionOwnerIdx = col('action owner');
```

with:

```js
    const commentIdx   = col('comment'), actionOwnerIdx = col('action owner');
    const strikeIdx    = col('strike move');   // -1 on tabs written before this column existed
```

Then at lines 1083-1085, replace:

```js
      return { id: Date.now() + Math.random(), type, name, possessionOwner, actionOwner,
               comment: commentIdx >= 0 ? (r[commentIdx] || '') : '',
               time, timeStr };
```

with:

```js
      return { id: Date.now() + Math.random(), type, name, possessionOwner, actionOwner,
               comment: commentIdx >= 0 ? (r[commentIdx] || '') : '',
               strikeMove: strikeIdx >= 0 ? (r[strikeIdx] || '') : '',
               time, timeStr };
```

- [ ] **Step 4: Add the column to CSV export**

Replace lines 1285-1296:

```js
  const headers = ['Time','Type','Name','Possession Owner'];
  if (fullGame) headers.push('Turnover');
  headers.push('To Review','Comment','Action Owner','YouTube Link');
  const rows = [headers.join(','),
    ...annotations.map(a => {
      const toReview = a.type === 'To Review' ? 'Yes' : '';
      const outType = a.type;
      const cols = [a.timeStr, outType, a.name, a.possessionOwner || ''];
      if (fullGame) cols.push(TR.isTurnover(a.type, a.name) ? 'Yes' : '');
      // Per-event YouTube deep link (timestamped to the event, 5s lookback);
      // empty when no video URL is loaded.
      cols.push(toReview, a.comment || '', a.actionOwner || '', ytLink(a.time));
      return cols.join(',');
    })
  ];
```

with:

```js
  const headers = ['Time','Type','Name','Possession Owner'];
  if (fullGame) headers.push('Turnover');
  headers.push('To Review','Comment','Action Owner','Strike Move','YouTube Link');
  const rows = [headers.join(','),
    ...annotations.map(a => {
      const toReview = a.type === 'To Review' ? 'Yes' : '';
      const outType = a.type;
      const cols = [a.timeStr, outType, a.name, a.possessionOwner || ''];
      if (fullGame) cols.push(TR.isTurnover(a.type, a.name) ? 'Yes' : '');
      // Per-event YouTube deep link (timestamped to the event, 5s lookback);
      // empty when no video URL is loaded.
      cols.push(toReview, a.comment || '', a.actionOwner || '',
                TR.strikeMoveOf(a.type, a.name, a.strikeMove), ytLink(a.time));
      return cols.join(',');
    })
  ];
```

- [ ] **Step 5: Read the column back from CSV**

At line 1319, replace:

```js
    const commentIdx = col('comment'), actionOwnerIdx = col('action owner');
```

with:

```js
    const commentIdx = col('comment'), actionOwnerIdx = col('action owner');
    const strikeIdx  = col('strike move');   // -1 in CSVs exported before this column existed
```

Then at line 1333, replace:

```js
      loaded.push({ id: Date.now()+Math.random(), type, name, possessionOwner, actionOwner, comment, time, timeStr: cleanTime });
```

with:

```js
      const strikeMove = strikeIdx >= 0 && parts[strikeIdx] ? parts[strikeIdx].trim() : '';
      loaded.push({ id: Date.now()+Math.random(), type, name, possessionOwner, actionOwner, comment, strikeMove, time, timeStr: cleanTime });
```

- [ ] **Step 6: Verify the round trip manually**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/annotator.html`, log in with `m30-admin`, load any YouTube video, tag a Game Start, a Try, a Turnover and a Pen Attack. Then:

1. **⬇ Export CSV** — open the file. Expected: a `Strike Move` column between `Action Owner` and `YouTube Link`; the Try row carries its own name there; the Turnover and Pen Attack rows are blank.
2. **📂 Load CSV** on that same file. Expected: all events reload, no console errors.
3. **⬇ Load from Sheet** on an existing pre-change game. Expected: it loads normally with every `strikeMove` empty — no errors, nothing missing.

- [ ] **Step 7: Commit**

```bash
git add annotator.html
git commit -m "feat(annotator): persist strike move through sheet and CSV"
```

---

### Task 4: `annotator.html` — the optional second-stage picker

**Files:**
- Modify: `annotator.html:109` and `annotator.html:205` (grid columns)
- Modify: `annotator.html:113-124` (add `.ann-move` styling)
- Modify: `annotator.html:499-501` (pending state)
- Modify: `annotator.html:713` (`selectTeam`)
- Modify: `annotator.html:741-745` (`quickTag`) and `annotator.html:759-765` (`selectType`)
- Modify: `annotator.html:803-825` (`selectSubType`)
- Modify: `annotator.html:842` (`tag` reset)
- Modify: `annotator.html:907-916` (`renderList` row)
- Modify: `annotator.html:936-968` (add `editAnnotationMove` after `editAnnotationName`)
- Modify: `annotator.html:1413-1437` (keyboard handler)

**Interfaces:**
- Consumes: `TR.STRIKE_MOVES`, `TR.isAttackEnd` (Task 1); `tag(type, name, possessionOwner, comment, strikeMove)` (Task 3).
- Produces: `offersStrikeMove(type, name) => boolean`, `renderMoveMenu(type, name)`, `selectStrikeMove(move)`, `finishSubType(type, name, comment, strikeMove)`, `editAnnotationMove(id)`.

**Two hazards this task must avoid.** First, `needsTeamSelection` returns `true` for *every* type when Simple Mode is on (`annotator.html:710`), and the keydown handler at `annotator.html:1429` binds `1`/`2` to team selection whenever `pendingType !== null` — so the move menu must be up *before* `pendingType` is ever set, or the number keys are hijacked. Second, Simple Mode's whole purpose is one click, and `quickTag` routes the `6` shortcut through `selectSubType`; the move stage is therefore gated off entirely when Simple Mode is on.

- [ ] **Step 1: Add the move state and helpers**

At `annotator.html:501`, after `let pendingComment = null;`, add:

```js
let pendingStrikeMove = null;
// The optional second stage: which failed attempt is currently awaiting a move.
// Deliberately NOT pendingType — that variable arms the 1/2 team-select hotkeys.
let moveType = null, moveName = null, moveComment = null;
```

Then add these three functions immediately after `renderSubmenu` (i.e. after `annotator.html:801`):

```js
// ── Strike move (optional second stage) ────────────────────────
// Turnovers and attack penalties can record the move that was being run when
// the attack failed. Offered after the sub-type is picked and always skippable.
// Simple Mode is excluded: being one click is the entire point of it.
function offersStrikeMove(type, name) {
  return !isDefaultOther() && type !== 'Try' && TR.isAttackEnd(type, name);
}

function renderMoveMenu(type, name) {
  document.getElementById('submenu').innerHTML =
    `<span class="submenu-label" style="color:${COLORS[type]}">${name} — move:</span>` +
    TR.STRIKE_MOVES.map((m, i) =>
      `<button class="sub-btn" onclick="selectStrikeMove('${m.replace(/'/g, "\\'")}')">` +
      (i < 9 ? `<span class="hotkey">${i + 1}</span>` : '') + m + `</button>`).join('') +
    `<button class="sub-btn" onclick="selectStrikeMove('')">✕ Skip</button>`;
}

function selectStrikeMove(move) {
  const type = moveType, name = moveName, comment = moveComment || '';
  moveType = null; moveName = null; moveComment = null;
  if (type === null) return;
  finishSubType(type, name, comment, move);
}
```

- [ ] **Step 2: Split `selectSubType` so the move stage comes first**

Replace `annotator.html:803-825` in full:

```js
function selectSubType(type, name, comment = '') {
  if (needsTeamSelection(type)) { pendingType = type; pendingName = name; pendingComment = comment; showTeamMenu(); }
  else {
    if (isFullGame()) {
      const inferred = inferPossessionOwner();
      if (!inferred) {
        setStatus('Cannot infer possession owner — tag a Game Start event first.', true);
        alert('Cannot infer possession owner.\n\nNo prior annotation with a known possession owner was found.\n\nTag a Game Start (Game Event) with a possession owner first.');
        selectedType = null;
        document.querySelectorAll('.tag-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('submenu').innerHTML = '<span class="submenu-empty">Select an action above — then choose a sub-type (or press 1–9)</span>';
        if (wasPlaying) { wasPlaying = false; doPlay(); }
        return;
      }
      tag(type, name, inferred, comment);
    } else {
      tag(type, name, '', comment);
    }
  }
}
```

with:

```js
function selectSubType(type, name, comment = '') {
  // Interpose the optional move stage before anything else, so pendingType is
  // still null and the 1-9 keys belong to the move menu rather than to team
  // selection.
  if (offersStrikeMove(type, name)) {
    moveType = type; moveName = name; moveComment = comment;
    renderMoveMenu(type, name);
    return;
  }
  finishSubType(type, name, comment, '');
}

function finishSubType(type, name, comment, strikeMove) {
  if (needsTeamSelection(type)) {
    pendingType = type; pendingName = name; pendingComment = comment; pendingStrikeMove = strikeMove;
    showTeamMenu();
  } else {
    if (isFullGame()) {
      const inferred = inferPossessionOwner();
      if (!inferred) {
        setStatus('Cannot infer possession owner — tag a Game Start event first.', true);
        alert('Cannot infer possession owner.\n\nNo prior annotation with a known possession owner was found.\n\nTag a Game Start (Game Event) with a possession owner first.');
        selectedType = null;
        document.querySelectorAll('.tag-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('submenu').innerHTML = '<span class="submenu-empty">Select an action above — then choose a sub-type (or press 1–9)</span>';
        if (wasPlaying) { wasPlaying = false; doPlay(); }
        return;
      }
      tag(type, name, inferred, comment, strikeMove);
    } else {
      tag(type, name, '', comment, strikeMove);
    }
  }
}
```

- [ ] **Step 3: Carry the move through team selection and clear it on every reset**

Replace `annotator.html:713`:

```js
function selectTeam(team) { if (pendingType !== null) tag(pendingType, pendingName, team, pendingComment || ''); }
```

with:

```js
function selectTeam(team) { if (pendingType !== null) tag(pendingType, pendingName, team, pendingComment || '', pendingStrikeMove || ''); }
```

Then clear the new state at all three existing reset sites. Insert this pair of lines immediately after each one:

```js
  pendingStrikeMove = null;
  moveType = null; moveName = null; moveComment = null;
```

The three sites, verbatim as they read today:

- `annotator.html:743`, in `quickTag` — `  pendingType = null; pendingName = null; pendingComment = null;`
- `annotator.html:761-763`, in `selectType` — the same three assignments on separate lines, ending with `  pendingComment = null;`
- `annotator.html:842`, in `tag` — `  selectedType = null; pendingType = null; pendingName = null;`

- [ ] **Step 4: Bind number keys to the move menu**

In the keydown handler, insert this **before** the existing team-selection branch at `annotator.html:1429`:

```js
  // Strike move by number — must precede both the team-select and sub-type
  // branches, since selectedType is still set while the move menu is up.
  if (moveType !== null && /^[1-9]$/.test(e.key)) {
    const idx = parseInt(e.key) - 1;
    if (idx < TR.STRIKE_MOVES.length) { e.preventDefault(); selectStrikeMove(TR.STRIKE_MOVES[idx]); }
    return;
  }
  // Enter skips the move stage, leaving the event untagged for move.
  if (moveType !== null && e.key === 'Enter') { e.preventDefault(); selectStrikeMove(''); return; }
```

And extend the `Escape` branch at line 1414 so it also clears the move state — replace:

```js
    selectedType = null; pendingType = null; pendingName = null;
```

with:

```js
    selectedType = null; pendingType = null; pendingName = null; pendingStrikeMove = null;
    moveType = null; moveName = null; moveComment = null;
```

- [ ] **Step 5: Show the move in the annotation list and make it editable**

Replace `annotator.html:109`:

```css
  .annotation-row { display: grid; grid-template-columns: 55px 90px 90px 110px 1fr 1fr auto auto auto; align-items: center; gap: 8px; background: #141925; border-radius: 6px; padding: 5px 10px; border-left: 3px solid; font-size: 0.77rem; cursor: pointer; transition: background 0.1s; }
```

with:

```css
  .annotation-row { display: grid; grid-template-columns: 55px 90px 90px 90px 110px 1fr 1fr auto auto auto; align-items: center; gap: 8px; background: #141925; border-radius: 6px; padding: 5px 10px; border-left: 3px solid; font-size: 0.77rem; cursor: pointer; transition: background 0.1s; }
```

Add after line 124:

```css
  .ann-move { color: #7dd3fc; font-size: 0.72rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
```

And inside the mobile media query at line 205, add alongside the existing rule:

```css
    .ann-move { display: none; }
```

Then in `renderList`, insert this cell immediately after the `ann-name` span (line 909):

```js
      <span class="ann-move${TR.isAttackEnd(a.type, a.name) && a.type !== 'Try' ? ' ann-name-editable' : ''}" id="ann-move-${a.id}" onclick="event.stopPropagation();editAnnotationMove(${a.id})" title="${TR.isAttackEnd(a.type, a.name) && a.type !== 'Try' ? 'Click to set the strike move' : ''}">${TR.isAttackEnd(a.type, a.name) && a.type !== 'Try' ? (a.strikeMove || '—') : ''}</span>
```

- [ ] **Step 6: Add the inline move editor**

Add immediately after `editAnnotationName` ends (after `annotator.html:968`):

```js
// Mirrors editAnnotationName, but for the strike move, and with a blank option
// because leaving an attempt untagged is always allowed.
function editAnnotationMove(id) {
  const ann = annotations.find(a => a.id === id);
  if (!ann || ann.type === 'Try' || !TR.isAttackEnd(ann.type, ann.name)) return;

  const span = document.getElementById(`ann-move-${id}`);
  if (!span) return;

  const sel = document.createElement('select');
  sel.className = 'ann-name-select';
  ['', ...TR.STRIKE_MOVES].forEach(o => {
    const opt = document.createElement('option');
    opt.value = o;
    opt.textContent = o || '— none —';
    if (o === (ann.strikeMove || '')) opt.selected = true;
    sel.appendChild(opt);
  });

  // The move has no bearing on possession or action owner, so nothing is
  // recomputed here — unlike editAnnotationName.
  const commit = () => { ann.strikeMove = sel.value; renderList(); };
  sel.onchange = commit;
  sel.onblur = () => renderList();
  sel.onkeydown = e => { if (e.key === 'Escape') { e.stopPropagation(); renderList(); } if (e.key === 'Enter') { e.stopPropagation(); commit(); } };
  sel.onclick = e => e.stopPropagation();

  span.replaceWith(sel);
  sel.focus();
}
```

- [ ] **Step 7: Verify manually**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/annotator.html`, log in with `m30-admin`, load a video, and confirm each of these:

1. Simple Mode **off**, tag a Turnover → pick `Ball Down`. Expected: the submenu becomes `Ball Down — move:` with 19 buttons and a `✕ Skip`; pressing `3` tags it with `32`.
2. Same, but press `Escape` at the move stage. Expected: everything clears, no event is tagged.
3. Same, but click `✕ Skip`. Expected: the event is tagged with a blank move.
4. Tag a Turnover → `6 Again`. Expected: **no** move menu; it tags immediately.
5. Tag a Penalty Defence → `Offside`. Expected: **no** move menu.
6. Tag a Try. Expected: **no** move menu; the list shows a blank move cell for it.
7. Simple Mode **on**, press `6`. Expected: one click tags `Turnover / 6th Touch` with no move menu.
8. Simple Mode **on**, Full Game **off**, tag a Turnover. Expected: the team menu appears and `1`/`2` still select the team.
9. Click the move cell on a tagged turnover. Expected: a dropdown with `— none —` first; picking a value updates the cell; `Escape` cancels.
10. Export CSV. Expected: the moves you picked appear in the `Strike Move` column.

- [ ] **Step 8: Commit**

```bash
git add annotator.html
git commit -m "feat(annotator): optional strike move picker on failed attempts"
```

---

### Task 5: `annotator_field.html` — persist and pick the strike move

**Files:**
- Modify: `annotator_field.html:1312-1317` (`tag`)
- Modify: `annotator_field.html:2087` (push rows)
- Modify: `annotator_field.html` — the edit sheet markup containing `editSubtypeWrap`
- Modify: `annotator_field.html:1446-1473` (`renderSubtypeMenu`, `applySubtype`)
- Modify: `annotator_field.html:1425-1440` (`applyEditType`)

**Interfaces:**
- Consumes: `TR.STRIKE_MOVES`, `TR.isAttackEnd`, `TR.strikeMoveOf` (Task 1).
- Produces: `renderMoveSelect(a)` and `applyStrikeMove()` local to this page; annotations carry an optional `strikeMove`.

- [ ] **Step 1: Carry the field on new events**

Replace `annotator_field.html:1317`:

```js
  annotations.push({ id: Date.now() + Math.random(), type, name, possessionOwner, actionOwner, comment: '', time: t, timeStr: TR.fmt(t) });
```

with:

```js
  annotations.push({ id: Date.now() + Math.random(), type, name, possessionOwner, actionOwner, comment: '', strikeMove: '', time: t, timeStr: TR.fmt(t) });
```

- [ ] **Step 2: Add the column to the push payload**

Replace `annotator_field.html:2087`:

```js
  const rows = annotations.map(a => [a.timeStr, a.possessionOwner, a.type, a.name, '', a.comment, a.actionOwner]);
```

with:

```js
  // Column order must match HEADERS in Code.gs, Strike Move last.
  const rows = annotations.map(a => [a.timeStr, a.possessionOwner, a.type, a.name, '', a.comment,
                                     a.actionOwner, TR.strikeMoveOf(a.type, a.name, a.strikeMove)]);
```

- [ ] **Step 3: Add the picker to the edit sheet markup**

At `annotator_field.html:895-897` the sub-type select sits inside the edit head row:

```html
        <span class="edit-subtype-wrap" id="editSubtypeWrap">
          <select id="editSubtype" class="edit-subtype" aria-label="Sub-type" onchange="applySubtype()"></select>
        </span>
```

Add a sibling immediately after that closing `</span>`, reusing the same classes so it inherits the sheet's styling:

```html
        <span class="edit-subtype-wrap" id="editMoveWrap" style="display:none">
          <select id="editMove" class="edit-subtype" aria-label="Strike move" onchange="applyStrikeMove()"></select>
        </span>
```

- [ ] **Step 4: Render and apply the move**

Add after `applySubtype` (`annotator_field.html:1473`):

```js
// Only a failed attempt carries a move — a Try's move is its own sub-type, and
// 6 Again / Penalty Defence keep the ball so the move is still running.
// '' is a real option: leaving it untagged is always allowed.
// Named renderMoveSelect, not renderMoveMenu: annotator.html already has a
// renderMoveMenu(type, name) with a different signature, and the two pages are
// read side by side often enough that the collision would mislead.
function renderMoveSelect(a) {
  const wrap = document.getElementById('editMoveWrap');
  const sel  = document.getElementById('editMove');
  const show = a.type !== 'Try' && TR.isAttackEnd(a.type, a.name);
  wrap.style.display = show ? '' : 'none';
  if (!show) return;
  const opts = ['', ...TR.STRIKE_MOVES];
  const cur  = a.strikeMove || '';
  if (cur && !opts.includes(cur)) opts.splice(1, 0, cur);
  sel.innerHTML = opts.map(o =>
    `<option value="${o}"${o === cur ? ' selected' : ''}>${o || '— none —'}</option>`).join('');
}

// Applies straight away rather than waiting on Save, matching applySubtype.
// The move has no bearing on possession, so nothing is replayed.
function applyStrikeMove() {
  if (editingIndex === null) return;
  annotations[editingIndex].strikeMove = document.getElementById('editMove').value;
  updateUI();
  saveSession();
}
```

- [ ] **Step 5: Keep the picker in sync when type or sub-type changes**

Three call sites, all exact:

- `annotator_field.html:1402`, inside `openEdit`: the line reads `renderSubtypeMenu(a);`. Add `renderMoveSelect(a);` on the line after it.
- `annotator_field.html:1471`, inside `applySubtype`: add `renderMoveSelect(a);` immediately before `markSubtypeState();`.
- `annotator_field.html:1436`, inside `applyEditType`: add `renderMoveSelect(a);` immediately before `replayPossessionFrom(editingIndex + 1);`.

Changing a Turnover to a Penalty Defence must hide the picker; changing a sub-type to `6 Again` must hide it too. Both are covered by re-running `renderMoveSelect`, which recomputes `show` from the current type and name.

- [ ] **Step 6: Verify manually**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/annotator_field.html` on a phone-sized viewport, log in with `m30-admin`, start a new game and confirm:

1. Tag a Turnover, tap it to edit. Expected: a `Strike move` select below `Sub-type`, defaulting to `— none —`.
2. Pick `32 - Cut`. Expected: it sticks after closing and reopening the sheet.
3. Change the sub-type to `6 Again`. Expected: the move select disappears.
4. Change the type to `Penalty Defence`. Expected: the move select disappears.
5. Change the type to `Try`. Expected: the move select disappears.
6. Reload the page and reopen the game from the picker. Expected: the move survived `localStorage`.
7. Open a game saved **before** this change. Expected: it loads with no move set and no console errors.

- [ ] **Step 7: Commit**

```bash
git add annotator_field.html
git commit -m "feat(field-annotator): strike move on failed attempts"
```

---

### Task 6: `annotator_field2.html` — persist, chip strip, and edit sheet

**Files:**
- Modify: `annotator_field2.html:1592-1594` (`tag`)
- Modify: `annotator_field2.html:2439-2441` (push rows)
- Modify: `annotator_field2.html:2984-3020` (`offerSubtype`, `closeSubtype`, `pickSubtype`)
- Modify: `annotator_field2.html` — the edit sheet markup containing `editSubtypeWrap`
- Modify: `annotator_field2.html:1709` area (`renderSubtypeMenu`) and its `applySubtype`

**Interfaces:**
- Consumes: `TR.STRIKE_MOVES`, `TR.isAttackEnd`, `TR.strikeMoveOf` (Task 1).
- Produces: `offerMoveStrip(idx)`, `closeMoveStrip()`, `pickStrikeMove(move)`, `renderMoveSelect(a)`, `applyStrikeMove()` local to this page.

- [ ] **Step 1: Carry the field on new events**

Replace `annotator_field2.html:1593-1594`:

```js
  annotations.push({ id: Date.now() + Math.random(), type, name, possessionOwner, actionOwner, comment: '', time: t, timeStr: TR.fmt(t),
                     x: pos ? pos.x : null, y: pos ? pos.y : null });
```

with:

```js
  annotations.push({ id: Date.now() + Math.random(), type, name, possessionOwner, actionOwner, comment: '', strikeMove: '', time: t, timeStr: TR.fmt(t),
                     x: pos ? pos.x : null, y: pos ? pos.y : null });
```

- [ ] **Step 2: Add the column to the push payload**

Replace `annotator_field2.html:2439-2441`:

```js
  const rows = annotations
    .filter(a => uploadTouches || a.type !== 'Touch')
    .map(a => [a.timeStr, a.possessionOwner, a.type, a.name, '', commentWithPos(a), a.actionOwner]);
```

with:

```js
  // Column order must match HEADERS in Code.gs, Strike Move last. The pitch
  // position still rides in Comment; the move gets its own column.
  const rows = annotations
    .filter(a => uploadTouches || a.type !== 'Touch')
    .map(a => [a.timeStr, a.possessionOwner, a.type, a.name, '', commentWithPos(a),
               a.actionOwner, TR.strikeMoveOf(a.type, a.name, a.strikeMove)]);
```

- [ ] **Step 3: Chain a move strip after the sub-type strip**

Replace `pickSubtype` (`annotator_field2.html:3005-3020`) so it hands off to a move strip instead of closing outright:

```js
function pickSubtype(name) {
  if (subtypeIdx === null) return;
  const a = annotations[subtypeIdx];
  const i = subtypeIdx;
  closeSubtype();
  if (!a) return;
  a.name = name;
  a.actionOwner = TR.inferActionOwner(a.possessionOwner, a.type, a.name);
  // Some sub-types change whether the ball turns over at all (6 Again keeps it),
  // so re-derive possession from the corrected name instead of leaving the
  // guess made when the event was first tagged.
  if (i === annotations.length - 1) setPossession(TR.inferPossessionAfter(a.possessionOwner, a.type, a.name));
  else replayPossessionFrom(i + 1);
  updateUI();
  saveSession();
  // A failed attempt can also record which move was being run. Same strip, same
  // timeout — ignoring it simply leaves the move blank.
  if (a.type !== 'Try' && TR.isAttackEnd(a.type, a.name)) offerMoveStrip(i);
}

// ── Strike move strip ──────────────────────────────────────────
let moveIdx = null, moveTimer = null;

function offerMoveStrip(idx) {
  const a = annotations[idx];
  if (!a) return;
  moveIdx = idx;
  document.getElementById('pitchStatus').innerHTML =
    `<span class="sub-chip head">Move —</span>` +
    TR.STRIKE_MOVES.map(m => `<button class="sub-chip" data-move="${m}">${m}</button>`).join('');
  clearTimeout(moveTimer);
  moveTimer = setTimeout(closeMoveStrip, 10000);
}

function closeMoveStrip() {
  clearTimeout(moveTimer);
  moveIdx = null;
  const bar = document.getElementById('pitchStatus');
  if (bar) bar.innerHTML = '<span id="pitchHint"></span>';
  renderPitch();
}

function pickStrikeMove(move) {
  if (moveIdx === null) return;
  const a = annotations[moveIdx];
  closeMoveStrip();
  if (!a) return;
  a.strikeMove = move;   // no possession impact, so nothing is replayed
  updateUI();
  saveSession();
}
```

- [ ] **Step 4: Route chip taps to the new handler**

Replace the delegated handler at `annotator_field2.html:3298-3301`:

```js
  document.getElementById('pitchStatus').addEventListener('click', ev => {
    const chip = ev.target.closest('[data-sub]');
    if (chip) pickSubtype(chip.dataset.sub);
  });
```

with:

```js
  document.getElementById('pitchStatus').addEventListener('click', ev => {
    const chip = ev.target.closest('[data-sub]');
    if (chip) { pickSubtype(chip.dataset.sub); return; }
    const mv = ev.target.closest('[data-move]');
    if (mv) pickStrikeMove(mv.dataset.move);
  });
```

- [ ] **Step 5: Make sure the strip can't outlive its event**

`moveIdx` is an index into `annotations`, so it must not survive that array shrinking or the game being swapped out. Add `closeMoveStrip();` immediately after the existing `closeSubtype();` at both of these sites:

- `annotator_field2.html:1465` — inside the game-load routine.
- `annotator_field2.html:2988` — the guard in `offerSubtype`.

(The call at `annotator_field2.html:3009` is inside `pickSubtype` itself, which Step 3 already rewrote — do not add another there.)

Also call `closeMoveStrip()` from the undo and delete handlers, alongside whatever cleanup they already do.

- [ ] **Step 6: Add the same select to the edit sheet**

Apply Task 5 Steps 3, 4 and 5 to `annotator_field2.html` — the same markup, the same `renderMoveSelect(a)` and `applyStrikeMove()` functions, and the same sync calls. The two edit sheets are structurally identical, so the code is byte-for-byte the same; repeat it rather than trying to share it, since neither page loads the other.

The line numbers differ:

- Markup: `annotator_field2.html:1166-1168` (same `editSubtypeWrap` span) — add the `editMoveWrap` sibling after its closing `</span>`.
- `renderSubtypeMenu` is at `annotator_field2.html:1706`; add `renderMoveSelect` and `applyStrikeMove` after `applySubtype` ends.
- Sync call in `openEdit`: `annotator_field2.html:1662`, the line reading `renderSubtypeMenu(a);` — add `renderMoveSelect(a);` after it.
- Add the other two sync calls in this file's `applySubtype` and `applyEditType`, in the same positions Task 5 Step 5 describes.

- [ ] **Step 7: Verify manually**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/annotator_field2.html` on a phone-sized viewport, log in with `m30-admin`, start a new game and confirm:

1. Tap out five touches, then tap the pin again to make a turnover. Pick `Ball Down` from the sub-type strip. Expected: the strip immediately becomes `Move — (Scoop)(21)(32)…`.
2. Tap `32 - Cut`. Expected: the strip closes and the move is stored.
3. Repeat but ignore the move strip for 10 seconds. Expected: it closes on its own; the event keeps a blank move.
4. Make a turnover and pick `6 Again`. Expected: **no** move strip appears.
5. Score a try. Expected: the sub-type strip appears, but **no** move strip after it.
6. Undo an event while its move strip is showing. Expected: the strip closes; no console errors.
7. Tap an event to edit. Expected: the `Strike move` select behaves exactly as in v1.
8. Reload and reopen from the picker. Expected: moves survived `localStorage`.
9. Open a game created in v1 (both annotators share the store). Expected: it loads cleanly.

- [ ] **Step 8: Commit**

```bash
git add annotator_field2.html
git commit -m "feat(field-annotator-v2): strike move strip and edit-sheet picker"
```

---

### Task 7: Surface the column in the viewer, refresh the cache, update the docs

**Files:**
- Modify: `viewer.html:585` (`FILTERABLE`)
- Modify: `sw.js:8` (`CACHE_VERSION`)
- Modify: `README.md` (event-type table and sheet-columns section)
- Modify: `FIELD_ANNOTATOR.md`, `FIELD_ANNOTATOR_V2.md`

**Interfaces:**
- Consumes: the `Strike Move` column from Task 2; nothing produces new API.

The viewer is entirely header-driven (`viewer.html:796`), so the column already renders once Task 2 is deployed. This only adds it to the filter bar — which is how you check tagging coverage before trusting any Phase 2 number.

- [ ] **Step 1: Make the column filterable**

Replace `viewer.html:585`:

```js
const FILTERABLE  = ['Type', 'Name', 'Action Owner', 'Division', 'Competition'];
```

with:

```js
const FILTERABLE  = ['Type', 'Name', 'Strike Move', 'Action Owner', 'Division', 'Competition'];
```

- [ ] **Step 2: Force a service-worker refresh**

Replace `sw.js:8`:

```js
const CACHE_VERSION = 'trl-shell-v12';
```

with:

```js
const CACHE_VERSION = 'trl-shell-v13';
```

- [ ] **Step 3: Document it**

In `README.md`, under **Game tabs**, replace the columns sentence with:

```markdown
Each game is a separate sheet tab named `YEAR_DIVISION_COMPETITION_TEAM1_TEAM2`. Columns: `Time`, `Possession Owner`, `Type`, `Name`, `To Review`, `Comment`, `Youtube Link`, `Action Owner`, `Strike Move`.

**Strike Move** — which attacking move the attempt was running. A Try's move is its own `Name`; a Turnover or Pen Attack can record the move that failed, which is what makes a try rate per move possible. Always optional, and never set on `6 Again` or Penalty Defence, since the attack keeps the ball there.
```

Add a row to the **Event types** table after `Turnover`:

```markdown
| Strike move | `1`–`9` | Optional second stage after a Turnover or Pen Attack sub-type — which move was being run. Skippable with `Escape`, `Enter`, or **✕ Skip**. Not offered in Simple Mode. |
```

In `FIELD_ANNOTATOR.md`, add to the editing section: the edit sheet now carries a **Strike move** picker below **Sub-type**, shown only on turnovers and attack penalties, defaulting to `— none —`.

In `FIELD_ANNOTATOR_V2.md`, add to the sub-type strip section: picking a sub-type on a turnover or attack penalty chains straight into a **Move** strip on the same line, with the same 10-second timeout, and ignoring it leaves the move blank.

- [ ] **Step 4: Run the full test suite**

```bash
node test.js
```

Expected: PASS, no failures.

- [ ] **Step 5: Verify the viewer filter**

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/viewer.html`, log in with `m30-admin`. Expected: a `Strike Move` filter dropdown in the toolbar, and a `Strike Move` column in the table. On games pushed before this change the column is blank throughout — that is correct, not a bug.

- [ ] **Step 6: Commit**

```bash
git add viewer.html sw.js README.md FIELD_ANNOTATOR.md FIELD_ANNOTATOR_V2.md
git commit -m "feat(viewer): filter by strike move; document the new column"
```

---

## Phase 1 exit criteria

- [ ] `node test.js` passes with the new `TR.isAttackEnd` / `TR.strikeMoveOf` / `TR.STRIKE_MOVES` assertions.
- [ ] `tests.html` reports 0 failures.
- [ ] The Apps Script deployment is live and `action=all` returns a `Strike Move` header.
- [ ] All three annotators can tag a move, skip a move, and round-trip both through the sheet.
- [ ] No annotator offers a move on `Try`, `6 Again`, or `Penalty Defence`.
- [ ] A game pushed before this change still loads in every tool without errors.
- [ ] `viewer.html` can filter by `Strike Move`.

**Do not start Phase 2 until real games have been tagged with moves.** Rate analytics built against an empty column cannot be validated, which is the whole reason the work is split.
