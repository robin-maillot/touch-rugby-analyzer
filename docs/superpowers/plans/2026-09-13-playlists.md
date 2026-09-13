# Saved Playlists Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an account save named, ordered lists of hand-picked events that span games and filters, and play them in the Event Viewer.

**Architecture:** A `_playlists` tab in the control spreadsheet stores one row per playlist, owned by the caller's secret. Three new Apps Script actions read and write it. A new `js/playlists.js` holds the pure resolve/reorder logic plus thin fetch wrappers. `game.html` gains a collect mode for gathering events across filters, a playlists bottom sheet, and a reorder screen — all reusing the two-level drill pattern the filter sheet already has.

**Tech Stack:** Vanilla JS, no build step. Google Apps Script backend. Node `test.js` harness (`node test.js`, no npm).

## Global Constraints

- **No build step.** Plain `<script src>` tags; `js/*.js` files attach to the global `TR` object declared with `var` in `js/config.js`.
- **Refs are resolved on `game` + `time` only.** Type and name are stored but advisory. Never match on all four segments.
- **Cap: 500 refs per playlist**, enforced server-side with the message `A playlist holds at most 500 events.`
- **Playlist reads and writes bypass `cacheGet`/`cachePut`** and never call `bumpVersion()`.
- **`_playlists` is never added to `ADMIN_SHEETS`.**
- **Every playlist write is ownership-checked server-side**, never trusted from the client.
- Comment style matches the codebase: explain *why*, not *what*. See `js/utils.js` for the register.
- Commit messages: imperative mood, a body explaining the reasoning, ending with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

**Spec:** `docs/superpowers/specs/2026-09-13-playlists-design.md`

---

### Task 1: Move `evId` into utils as `TR.evId` / `TR.evKey` / `TR.refKey`

The playlist module needs the ref format, and a second copy in `js/playlists.js` would drift from `game.html`'s the first time either changed. Moving it first means every later task consumes one definition.

**Files:**
- Modify: `js/utils.js` (append at end)
- Modify: `game.html` (delete `evId`, update 3 call sites)
- Test: `test.js`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TR.evId(e) -> string` — `"game#time#type#name"`, the form written to the sheet.
  - `TR.evKey(e) -> string` — `"game#time"`, what resolution matches on.
  - `TR.refKey(ref) -> string` — the first two `#`-separated segments of a stored ref string.

- [ ] **Step 1: Write the failing tests**

Append to `test.js`, immediately before the final `console.log(\`\n${passed} passed...\`)` summary block:

```js
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
node test.js
```

Expected: five `✗` lines reading `TR.evId is not a function` (and similar), and a non-zero exit.

- [ ] **Step 3: Add the three helpers to `js/utils.js`**

Append to the end of `js/utils.js`:

```js
// Identity for an event row. Index-free on purpose: an id built from the render
// position goes stale the moment a filter changes the list.
TR.evId = (e) => `${e.game}#${e.time}#${e.type}#${e.name}`;

// The part of an id that survives an Event Editor rename. Playlists store the
// full id (readable in the sheet) but resolve on this, so correcting a Name
// can't orphan a saved entry. Safe to split on '#': sheet tab names and event
// names never contain one.
TR.evKey = (e) => `${e.game}#${e.time}`;

// The same key, taken from a stored ref string rather than a live event.
TR.refKey = (ref) => String(ref == null ? '' : ref).split('#').slice(0, 2).join('#');
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
node test.js
```

Expected: five new `✓` lines, `0 failed`, exit 0.

- [ ] **Step 5: Delete `evId` from `game.html` and repoint its call sites**

Delete this function and its comment block (around line 566–572):

```js
// Index-free on purpose: an id built from the render position went stale the
// moment a filter changed the list, losing the active row and breaking
// prev/next. Game + time + type + name is unique in practice.
function evId(e){return e.game+'#'+e.time+'#'+e.type+'#'+e.name}
```

Then replace each remaining `evId(` with `TR.evId(`. There are exactly three:

- in `render()`, `const id=evId(e);` → `const id=TR.evId(e);`
- in `step()`, `let i=shown.findIndex(e=>evId(e)===activeId);` → `...e=>TR.evId(e)===activeId);`
- in `step()`, `playEvent(e.time,evId(e),e.game);` → `playEvent(e.time,TR.evId(e),e.game);`

Verify none are left:

```bash
grep -n "[^.]evId(" game.html
```

Expected: no output.

- [ ] **Step 6: Check the page still works**

With `python3 -m http.server 8765` running, open `http://localhost:8765/game.html`, pick a game, tap an event (it should play and highlight), then tap **next** (it should advance). A `TR.evId is not a function` in the console means `js/utils.js` isn't loaded before the inline script — it is, via the existing `<script src="js/utils.js">`.

- [ ] **Step 7: Commit**

```bash
git add js/utils.js game.html test.js
git commit -m "$(cat <<'EOF'
refactor: evId moves to utils as TR.evId, with a loose key beside it

Playlists need the same ref format game.html synthesises, and a second copy
would drift from this one the first time either changed. TR.evKey and
TR.refKey name the part that survives an Event Editor rename — game and time —
which is what a saved playlist has to resolve on.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `js/playlists.js` — the pure resolve and reorder helpers

Pure logic first, with no network in the file yet, so the Node harness can load and exercise it.

**Files:**
- Create: `js/playlists.js`
- Modify: `test.js` (harness file list + assertions)
- Modify: `sw.js` (shell list)
- Modify: `game.html` (script tag)

**Interfaces:**
- Consumes: `TR.evKey`, `TR.refKey` from Task 1.
- Produces:
  - `TR.playlists.resolve(refs, rows) -> { events: Array, missing: number }`
  - `TR.playlists.reorder(refs, from, to) -> Array` (a new array; out-of-range or equal indices are a no-op copy)
  - `TR.playlists.MAX_REFS -> 500`

- [ ] **Step 1: Write the failing tests**

Append to `test.js`, before the summary block:

```js
// ── TR.playlists.resolve ──────────────────────────────────────
console.log('TR.playlists.resolve');
const PROWS = [
  { game: 'g1', time: 10,  type: 'Try',      name: 'Scoop'    },
  { game: 'g1', time: 90,  type: 'Turnover', name: 'Ball Down'},
  { game: 'g2', time: 30,  type: 'Try',      name: '32 - Cut' },
];
test('all resolve', () => {
  const r = TR.playlists.resolve(['g1#10#Try#Scoop', 'g2#30#Try#32 - Cut'], PROWS);
  assert.equal(r.missing, 0);
  assert.deepEqual(r.events.map(e => e.game), ['g1', 'g2']);
});
test('playlist order wins over clock order', () => {
  const r = TR.playlists.resolve(['g2#30#Try#32 - Cut', 'g1#10#Try#Scoop'], PROWS);
  assert.deepEqual(r.events.map(e => e.time), [30, 10]);
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
  assert.deepEqual(TR.playlists.resolve([], PROWS), { events: [], missing: 0 });
  assert.deepEqual(TR.playlists.resolve(null, null), { events: [], missing: 0 });
});
test('duplicate game+time: first row in sheet order wins', () => {
  const dupes = [{ game: 'g1', time: 10, type: 'Try', name: 'first' },
                 { game: 'g1', time: 10, type: 'Try', name: 'second' }];
  assert.equal(TR.playlists.resolve(['g1#10#Try#anything'], dupes).events[0].name, 'first');
});

// ── TR.playlists.reorder ──────────────────────────────────────
console.log('TR.playlists.reorder');
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
                               assert.deepEqual(TR.playlists.reorder(null, 0, 1), []); });
```

- [ ] **Step 2: Add `js/playlists.js` to the Node harness**

In `test.js`, add `'js/playlists.js'` to the end of the file list on line 26:

```js
for (const f of ['js/config.js', 'js/utils.js', 'js/events.js', 'js/possession.js', 'js/consistency.js', 'js/player.js', 'js/field_games.js', 'js/strike_moves.js', 'js/playlists.js']) {
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
node test.js
```

Expected: a thrown `ENOENT: no such file or directory, open 'js/playlists.js'` — the harness cannot load a file that does not exist yet.

- [ ] **Step 4: Create `js/playlists.js` with the pure helpers**

```js
// Depends on js/config.js (TR namespace) and js/utils.js (TR.evKey / TR.refKey).
//
// A playlist is a named, ordered list of hand-picked events belonging to one
// account. It spans games and filters freely, because each entry is a concrete
// event ref rather than a filter that has to re-derive one.
TR.playlists = {};

// The sheet cap, mirrored client-side so the UI can refuse before the round
// trip. Kept in step with PLAYLIST_MAX_REFS in apps_script/Code.gs.
TR.playlists.MAX_REFS = 500;

// Turn stored refs into live event objects, in the PLAYLIST's order rather than
// the clock's.
//
// Matching is on game + time alone (TR.refKey). Refs are synthetic — built from
// the row's own values — so an Event Editor rename changes the type and name
// segments of every ref pointing at that event. Matching loosely survives that;
// matching on all four would silently empty a playlist after a correction.
//
// Where two rows somehow share a game and a timestamp, the first in sheet order
// wins, deterministically.
TR.playlists.resolve = (refs, rows) => {
  const by = new Map();
  (rows || []).forEach(e => {
    const k = TR.evKey(e);
    if (!by.has(k)) by.set(k, e);
  });
  const events = [];
  let missing = 0;
  (refs || []).forEach(r => {
    const e = by.get(TR.refKey(r));
    if (e) events.push(e); else missing++;
  });
  return { events, missing };
};

// Move one entry, returning a new array. An out-of-range or unchanged index is
// a no-op copy rather than a throw — the reorder screen's ↑ on the first row
// and ↓ on the last both land here.
TR.playlists.reorder = (refs, from, to) => {
  const out = (refs || []).slice();
  if (!(from >= 0) || from >= out.length || !(to >= 0) || to >= out.length || from === to) return out;
  const [moved] = out.splice(from, 1);
  out.splice(to, 0, moved);
  return out;
};
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
node test.js
```

Expected: 14 new `✓` lines, `0 failed`, exit 0.

- [ ] **Step 6: Load the file in `game.html` and the service worker shell**

In `game.html`, add the script tag after the existing `js/utils.js` one (keep the order — `playlists.js` reads `TR.evKey` at call time, but grouping them reads better):

```html
<script src="js/playlists.js"></script>
```

In `sw.js`, add `'js/playlists.js'` to the `SHELL` array after `'js/strike_moves.js'`, and bump the cache version by one. Check the current value first:

```bash
grep -n "CACHE_VERSION" sw.js
```

If it reads `'trl-shell-v20'`, change it to `'trl-shell-v21'`. If it has moved on, bump whatever is there by one.

- [ ] **Step 7: Confirm the browser loads it**

Reload `http://localhost:8765/game.html` and run in the console:

```js
TR.playlists.reorder(['a','b','c'], 0, 2)
```

Expected: `['b', 'c', 'a']`.

- [ ] **Step 8: Commit**

```bash
git add js/playlists.js test.js sw.js game.html
git commit -m "$(cat <<'EOF'
feat(playlists): resolve and reorder helpers

resolve() maps stored refs onto live events in the playlist's order, matching
on game and time alone. Refs are synthetic, so an Event Editor rename rewrites
the type and name segments of every ref pointing at that event — a strict match
would silently empty a playlist after a correction. What still can't resolve is
counted rather than dropped, so a broken playlist can explain itself.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Apps Script — the `_playlists` tab and three actions

**Files:**
- Modify: `apps_script/Code.gs`

**Interfaces:**
- Consumes: existing `authFor`, `json`, `CONTROL_SHEET_ID`.
- Produces the HTTP contract every later task calls:
  - `GET  ?secret=…&action=playlists` → `{ok:true, playlists:[{id,name,note,refs:[string],updated}]}`
  - `POST {secret, action:'save_playlist', id?, name, note, refs:[string]}` → `{ok:true, id}`
  - `POST {secret, action:'delete_playlist', id}` → `{ok:true}`
  - Failures: `{ok:false, error:'Playlist not found.'}`, `{ok:false, error:'A playlist needs a name.'}`, `{ok:false, error:'A playlist holds at most 500 events.'}`

There is no Node harness for `.gs` — it is verified against the deployed script in Step 5.

- [ ] **Step 1: Add the constants**

In `apps_script/Code.gs`, in the `── Configuration ──` block, after the `ADMIN_SHEETS` line:

```js
// User content, not control plane. Deliberately absent from ADMIN_SHEETS: the
// admin sheet editor exists to repair the control tabs by hand, and putting
// every account's playlists in one editable grid buys nothing.
const PLAYLISTS_SHEET  = '_playlists';
const PLAYLIST_HEADERS = ['Id', 'Owner', 'Name', 'Note', 'Refs', 'Updated At'];
const PLAYLIST_MAX_REFS = 500;
```

- [ ] **Step 2: Add the sheet helpers**

Add after `isAdminSecret()` (around line 70):

```js
// ── Playlist helpers ───────────────────────────────────────────
// The _playlists tab, created on first use so a fresh control spreadsheet needs
// no manual setup.
function playlistsSheet() {
  const ss = SpreadsheetApp.openById(CONTROL_SHEET_ID);
  let sh = ss.getSheetByName(PLAYLISTS_SHEET);
  if (!sh) {
    sh = ss.insertSheet(PLAYLISTS_SHEET);
    sh.appendRow(PLAYLIST_HEADERS);
  }
  return sh;
}

// Every playlist row. `row` is the 1-based sheet row, so a write can address it
// directly rather than searching again. Never cached: the tab is small, it
// changes on every edit, and the action=version fast-path must not let a client
// skip a playlist change.
function readPlaylists() {
  const values = playlistsSheet().getDataRange().getDisplayValues();
  if (values.length < 2) return [];
  const h  = values[0].map(s => String(s).toLowerCase().trim());
  const ii = h.indexOf('id'), oi = h.indexOf('owner'), ni = h.indexOf('name');
  const ti = h.indexOf('note'), ri = h.indexOf('refs'), ui = h.indexOf('updated at');
  if (ii < 0 || oi < 0) return [];
  const out = [];
  for (let i = 1; i < values.length; i++) {
    const id = String(values[i][ii] || '').trim();
    if (!id) continue;
    out.push({
      row:     i + 1,
      id:      id,
      owner:   String(values[i][oi] || '').trim(),
      name:    ni >= 0 ? String(values[i][ni] || '') : '',
      note:    ti >= 0 ? String(values[i][ti] || '') : '',
      refs:    ri >= 0 ? String(values[i][ri] || '').split('\n').map(s => s.trim()).filter(Boolean) : [],
      updated: ui >= 0 ? String(values[i][ui] || '') : '',
    });
  }
  return out;
}

// The playlist row this secret owns, or null. Ownership is checked here on
// every write and never trusted from the client, the same way canEditGame
// guards a game tab.
function ownedPlaylist(secret, id) {
  if (!secret || !id) return null;
  const want = String(id);
  return readPlaylists().find(p => p.id === want && p.owner === String(secret)) || null;
}
```

- [ ] **Step 3: Add the read action to `doGet`**

In `doGet`, immediately after the `action=whoami` block (which ends `return json({ ok: true, role: auth.role, group: auth.group });`), insert:

```js
    // action=playlists → the caller's own saved playlists. Never cached, and
    // deliberately above the cacheKeySuffix line below: a playlist changes no
    // game data, so it must not ride on the version-keyed game cache.
    if (e.parameter.action === 'playlists') {
      const mine = readPlaylists()
        .filter(p => p.owner === String(e.parameter.secret))
        .map(p => ({ id: p.id, name: p.name, note: p.note, refs: p.refs, updated: p.updated }));
      return json({ ok: true, playlists: mine });
    }
```

- [ ] **Step 4: Add the two write actions to `doPost`**

In `doPost`, immediately after the `const callerGroup = ...` line and before the `action=update_rows` block, insert:

```js
    // action=save_playlist → create, or replace one the caller owns. Any role
    // may own playlists, viewer included: a viewer is exactly the person
    // building a teaching set, and a playlist grants no access to anything — a
    // ref only resolves against events the caller could already see.
    //
    // No bumpVersion(): a playlist changes no game data, and bumping would make
    // every client refetch the heavy action=all payload for nothing.
    if (data.action === 'save_playlist') {
      const name = String(data.name == null ? '' : data.name).trim();
      if (!name) return json({ ok: false, error: 'A playlist needs a name.' });
      const refs = (data.refs || []).map(r => String(r).trim()).filter(Boolean);
      if (refs.length > PLAYLIST_MAX_REFS) {
        return json({ ok: false, error: 'A playlist holds at most ' + PLAYLIST_MAX_REFS + ' events.' });
      }
      const note    = String(data.note == null ? '' : data.note).trim();
      const updated = new Date().toISOString();
      const sh      = playlistsSheet();
      if (data.id) {
        const owned = ownedPlaylist(data.secret, data.id);
        // A miss is reported rather than silently creating a second row — the
        // client asked to replace something specific.
        if (!owned) return json({ ok: false, error: 'Playlist not found.' });
        sh.getRange(owned.row, 1, 1, PLAYLIST_HEADERS.length)
          .setValues([[owned.id, owned.owner, name, note, refs.join('\n'), updated]]);
        return json({ ok: true, id: owned.id });
      }
      const id = Utilities.getUuid();
      sh.appendRow([id, String(data.secret), name, note, refs.join('\n'), updated]);
      return json({ ok: true, id: id });
    }

    // action=delete_playlist → remove one the caller owns.
    if (data.action === 'delete_playlist') {
      const owned = ownedPlaylist(data.secret, data.id);
      if (!owned) return json({ ok: false, error: 'Playlist not found.' });
      playlistsSheet().deleteRow(owned.row);
      return json({ ok: true });
    }
```

- [ ] **Step 5: Deploy and verify against the live script**

Paste `Code.gs` into the Apps Script editor, then **Deploy → Manage deployments → edit the existing deployment → New version**. The web app URL does not change.

Then, in the browser console on any logged-in page of the app, run each of these and check the stated result:

```js
const U = TR.APPS_SCRIPT_URL, S = TR.secret();
const post = b => fetch(U, {method:'POST', headers:{'Content-Type':'text/plain'},
  body: JSON.stringify({secret:S, ...b})}).then(r => r.json());
const list = () => fetch(`${U}?secret=${TR.enc(S)}&action=playlists`).then(r => r.json());

// 1. create
const made = await post({action:'save_playlist', name:'throwaway', note:'test', refs:['g#1#Try#x']});
made        // → {ok: true, id: "<uuid>"}

// 2. list
(await list()).playlists.find(p => p.id === made.id)
            // → {id, name: "throwaway", note: "test", refs: ["g#1#Try#x"], updated: "<iso>"}

// 3. edit
await post({action:'save_playlist', id:made.id, name:'renamed', note:'', refs:['g#1#Try#x','g#2#Try#y']})
            // → {ok: true, id: "<same uuid>"}

// 4. a name is required
await post({action:'save_playlist', name:'  '})
            // → {ok: false, error: "A playlist needs a name."}

// 5. the cap
await post({action:'save_playlist', name:'big', refs:Array(501).fill('g#1#Try#x')})
            // → {ok: false, error: "A playlist holds at most 500 events."}

// 6. someone else's playlist is invisible and unwritable
await post({action:'save_playlist', id:'no-such-id', name:'hijack'})
            // → {ok: false, error: "Playlist not found."}

// 7. delete
await post({action:'delete_playlist', id:made.id})   // → {ok: true}
(await list()).playlists.some(p => p.id === made.id)  // → false
```

Also confirm the `_playlists` tab now exists in the control spreadsheet with the six headers, and that **`_playlists` does not appear** in the admin page's sheet picker.

For check 6 to be meaningful, repeat step 3 while logged in as a *different* secret and confirm it also returns `Playlist not found.` — that is the ownership guard, not just a missing row.

- [ ] **Step 6: Commit**

```bash
git add apps_script/Code.gs
git commit -m "$(cat <<'EOF'
feat(api): _playlists tab with read, save and delete actions

One row per playlist, owned by the caller's secret — the same plaintext secrets
_groups already holds in this spreadsheet, so no new exposure. Ownership is
checked server-side on every write; a replace that doesn't match is reported
rather than quietly creating a second row.

Reads and writes bypass the cache and never bumpVersion(): a playlist changes
no game data, and the version fast-path must neither skip a playlist edit nor
force every client to refetch action=all because of one.

Deliberately absent from ADMIN_SHEETS — this is user content, and the admin
editor would put every account's lists in one grid for no benefit.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `TR.playlists.load` / `save` / `remove`

**Files:**
- Modify: `js/playlists.js` (append)

**Interfaces:**
- Consumes: Task 3's HTTP contract; `TR.APPS_SCRIPT_URL`, `TR.enc` from `js/config.js` / `js/utils.js`.
- Produces:
  - `TR.playlists.load(secret) -> Promise<Array<{id,name,note,refs,updated}>>`
  - `TR.playlists.save(secret, {id?, name, note, refs}) -> Promise<{id}>`
  - `TR.playlists.remove(secret, id) -> Promise<void>`
  - All three reject with an `Error` carrying the server's message.

These wrap `fetch`, so they are not covered by the Node harness — `test.js` has no `fetch` stub and should not grow one for three one-line wrappers. They are verified in the browser in Step 3.

- [ ] **Step 1: Append the wrappers to `js/playlists.js`**

```js
// ── Network ───────────────────────────────────────────────────
// Apps Script web apps reject a CORS preflight, so every POST here sends
// text/plain — the same shape every other write in the app uses.
TR.playlists._post = async (body) => {
  const r = await fetch(TR.APPS_SCRIPT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || 'Playlist save failed.');
  return j;
};

TR.playlists.load = async (secret) => {
  const r = await fetch(`${TR.APPS_SCRIPT_URL}?secret=${TR.enc(secret)}&action=playlists`);
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || 'Could not load playlists.');
  return j.playlists || [];
};

// pl: {id?, name, note, refs}. Omitting id creates; passing one the caller
// doesn't own is refused server-side.
TR.playlists.save = (secret, pl) => TR.playlists._post({
  secret, action: 'save_playlist',
  id: pl.id || '', name: pl.name, note: pl.note || '', refs: pl.refs || [],
});

TR.playlists.remove = (secret, id) =>
  TR.playlists._post({ secret, action: 'delete_playlist', id }).then(() => undefined);
```

- [ ] **Step 2: Confirm the harness still passes**

```bash
node test.js
```

Expected: `0 failed`. (The new code references `fetch` only inside function bodies, so loading the file in the vm context is safe.)

- [ ] **Step 3: Verify a round trip in the browser**

On `http://localhost:8765/game.html`, in the console:

```js
const S = TR.secret();
const p = await TR.playlists.save(S, {name:'wrapper test', refs:['g#1#Try#x']});
(await TR.playlists.load(S)).find(x => x.id === p.id)   // → the saved playlist
await TR.playlists.remove(S, p.id);
(await TR.playlists.load(S)).some(x => x.id === p.id)   // → false
await TR.playlists.save(S, {name:''}).catch(e => e.message)  // → "A playlist needs a name."
```

- [ ] **Step 4: Commit**

```bash
git add js/playlists.js
git commit -m "$(cat <<'EOF'
feat(playlists): load, save and remove against the sheet

Thin wrappers over the three actions. POSTs send text/plain because Apps Script
web apps reject a preflight, matching every other write in the app. A server
error surfaces as a thrown Error carrying its message, so callers can show the
real reason rather than a generic failure.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: The playlists sheet, and playing a playlist

The read half of the feature: you can open, see and play playlists, but not yet make one. Deliverable is testable on its own using a playlist created from the console.

**Files:**
- Modify: `game.html` (CSS, state, `openSheet`, `render`, `pickGame`, `applyHash`, load IIFE)

**Interfaces:**
- Consumes: `TR.playlists.load`, `TR.playlists.resolve`, `TR.evId`.
- Produces (for Tasks 6 and 7):
  - `PL` — module-level array of loaded playlists.
  - `plOpen` — id of the playing playlist, or `null`.
  - `plMissing` — count of refs the current playlist could not resolve.
  - `playPlaylist(id)` — sets `plOpen`, closes the sheet, re-renders.
  - `openSheet('playlists')` — the sheet kind.
  - `plById(id)` — lookup helper.

- [ ] **Step 1: Add the state and the loader**

In `game.html`, beside the other module-level `let` declarations (after the `let F={…}` block):

```js
// Saved playlists for this account, and which one is currently playing.
// A playlist is a fixed, ordered list of hand-picked events; while one plays it
// IS the list, so the filters are suppressed rather than allowed to narrow it.
let PL=[], plOpen=null, plMissing=0;

function plById(id){return PL.find(p=>p.id===id)||null}
```

In the top-level load IIFE, after the existing `if(lr.ok){…}` line inside the `try`, add:

```js
    try{ PL=await TR.playlists.load(TR.secret()); }catch(e){ PL=[]; }
```

A playlist load failure must not take the page down with it — the event list is the page's job, playlists are an extra.

- [ ] **Step 2: Source `shown` from the playlist in `render()`**

At the top of `render()`, replace the single `shown=events.filter(...)` assignment with a branch. The existing code begins:

```js
function render(keepWindow){
  if(!keepWindow) renderLimit=200;
  const q=F.q.toLowerCase();
  shown=events.filter(e=>
```

Change it to:

```js
function render(keepWindow){
  if(!keepWindow) renderLimit=200;
  const q=F.q.toLowerCase();
  const pl=plOpen?plById(plOpen):null;
  if(pl){
    // A playlist is already a hand-picked set. Letting the filters narrow it
    // further would leave the chips describing a list that isn't on screen.
    const r=TR.playlists.resolve(pl.refs,rows);
    shown=r.events; plMissing=r.missing;
  }else{
    plMissing=0;
    shown=events.filter(e=>
```

…keeping the existing predicate body, and closing the `else` after the predicate's `);`. The line after the predicate currently reads:

```js
  const n=F.comp.size+F.div.size+...
```

so insert a `}` on its own line just before it.

- [ ] **Step 3: Paint the playlist chip instead of the filter chips**

Still in `render()`, the filter bar is built starting at `const bar=$('filterbar');`. Wrap the whole existing filter-chip section in the same `pl` branch. Replace from `const bar=$('filterbar');` down to (and including) the `if(!n){ … }` quick-access block with:

```js
  const bar=$('filterbar');
  if(pl){
    bar.innerHTML=`<button class="chip pl" onclick="playPlaylist(null)">
        ${esc(pl.name)} <span class="x">✕</span></button>`+
      `<span class="count">${shown.length} events</span>`+
      (plMissing?`<span class="count warn">${plMissing} no longer in the data</span>`:'');
    $('fCount').hidden=true;
  }else{
```

…then the existing body unchanged, and a closing `}` after the `if(!n){ … }` block. The `$('fCount')` and `$('count')` assignments that currently sit above `const bar=` move inside the `else` branch, since a playing playlist has no filter count.

- [ ] **Step 4: Add the chip styles**

In `game.html`'s `<style>`, after the `.chip.drill b{…}` rule:

```css
  /* Playing a playlist: its own accent (the --review purple), distinct from
     both the blue "filter applied" chips and the drill chip's cyan. */
  .chip.pl{background:#231733;border-color:var(--review);color:#e9d5ff}
  .count.warn{color:var(--turn)}
```

- [ ] **Step 5: Add `playPlaylist` and the sheet**

Add near the other sheet functions:

```js
// null leaves the playlist and returns to the ordinary filtered view.
function playPlaylist(id){
  plOpen=id; activeId=null; closeSheet(); render();
  if(id) LS.set('lastplaylist',id);
}

function playlistSheet(){
  if(!PL.length)return '<div class="empty">No playlists yet.<br>'+
    'Tap ☑ in the filter bar to start collecting events into one.</div>';
  return '<div class="glist">'+PL.map(p=>{
    const r=TR.playlists.resolve(p.refs,rows);
    const gone=r.missing?` · ${r.missing} missing`:'';
    return `<button class="${p.id===plOpen?'on':''}" onclick="playPlaylist('${p.id}')">
      ${esc(p.name)}
      <span class="gs">${r.events.length} events${gone}${p.note?' · '+esc(p.note):''}</span>
    </button>`}).join('')+'</div>';
}
```

- [ ] **Step 6: Register the sheet kind**

In `openSheet(kind)`, extend both ternaries:

```js
  $('sheetTitle').textContent = kind==='game'?'Choose a game':kind==='moves'?'Strike moves'
    :kind==='window'?'Playback window':kind==='playlists'?'Playlists':'Filter events';
  $('sheetBody').innerHTML = kind==='game'?gameSheet():kind==='moves'?movesSheetBody()
    :kind==='window'?windowSheet():kind==='playlists'?playlistSheet():filterSheet();
```

- [ ] **Step 7: Add the topbar button**

In the `.topbar`, before the existing filter button, add:

```html
  <button class="icon-btn" onclick="openSheet('playlists')" title="Playlists">📋</button>
```

- [ ] **Step 8: Clear `plOpen` where the source changes, and add the hash**

In `pickGame(k)`, add `plOpen=null;` to the line that already resets `activeId`:

```js
  activeId=null; plOpen=null; closeSheet(); LS.set('lastgame',k);
```

In `applyHash()`, add a line beside the others:

```js
  if(h==='playlists' && picked) openSheet('playlists');
```

- [ ] **Step 9: Verify in the browser**

With the server running, on `http://localhost:8765/game.html`:

1. In the console, make a playlist out of three real events from two different games:
   ```js
   const S=TR.secret();
   const some=[rows[0],rows[1],rows.find(r=>r.game!==rows[0].game)];
   await TR.playlists.save(S,{name:'browser check',refs:some.map(TR.evId)});
   PL=await TR.playlists.load(S);
   ```
2. Tap **📋** → the playlist is listed with `3 events`.
3. Tap it → the sheet closes, the filter bar shows one purple chip named `browser check`, the count reads `3 events`, and the list holds exactly those three rows **in the order saved**, including the one from the other game.
4. Tap an event → it plays, and the video switches games where it should.
5. Tap **next** → it advances within the playlist.
6. Tap the chip's **✕** → back to the ordinary filtered view with the filter chips returned.
7. Add a junk ref and confirm the warning:
   ```js
   const p=PL[0];
   await TR.playlists.save(S,{id:p.id,name:p.name,refs:[...p.refs,'gone#1#Try#x']});
   PL=await TR.playlists.load(S); playPlaylist(p.id);
   ```
   The bar should read `3 events` plus an orange `1 no longer in the data`.
8. Open the game picker and choose a game → the playlist chip is gone.
9. Clean up: `await TR.playlists.remove(S, PL[0].id)`.

- [ ] **Step 10: Commit**

```bash
git add game.html
git commit -m "$(cat <<'EOF'
feat: playlists are playable in the Event Viewer

Selecting one replaces shown with its resolved refs, so prev/next, loop, the
scrub marks and the playback window all follow it untouched — they read shown
already.

Filters are suppressed while a playlist plays. A playlist is a hand-picked set;
leaving the chips live would let them narrow it while still claiming to describe
what's on screen. Refs that no longer resolve are counted in the bar rather than
quietly shortening the list.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Collect mode

**Files:**
- Modify: `game.html` (CSS, state, `.listhead`, list rendering, cinema toggle)

**Interfaces:**
- Consumes: `PL`, `plOpen`, `plById` from Task 5; `TR.evId`, `TR.playlists.save`, `TR.playlists.MAX_REFS`.
- Produces: `collecting` (bool), `sel` (`Set` of `TR.evId` refs), `toggleCollect()`, `addToSheet()` — Task 7 touches none of these.

- [ ] **Step 1: Add the state**

Beside the playlist state from Task 5:

```js
// Collect mode: tick events into a pending selection, then commit the lot to a
// playlist in one write.
//
// sel holds full TR.evId refs, never list indices. An index is a position in
// `shown`, and `shown` is rebuilt by every filter change — so an index-based
// selection would silently repoint at different events the moment the user
// filtered, which is exactly the motion this mode exists to support.
let collecting=false;
const sel=new Set();
```

- [ ] **Step 2: Add the CSS**

In `game.html`'s `<style>`:

```css
  /* ── Collect mode ──────────────────────────────────────────── */
  /* The tick replaces the timestamp column rather than adding one, so a row
     gains no width here and loses none when not collecting. */
  body.collect .ev{grid-template-columns:34px 1fr}
  body.collect .list{padding-bottom:64px}
  .tick{width:22px;height:22px;border-radius:6px;border:2px solid var(--border);
    display:flex;align-items:center;justify-content:center;font-size:.75rem;color:transparent}
  .ev.sel .tick{background:var(--review);border-color:var(--review);color:#fff}
  .ev.sel{background:var(--surface2);border-left-color:var(--review)!important}
  .tray{display:none;background:#231733;border:1px solid var(--review);border-radius:8px;
    padding:8px 11px;margin:6px 8px 0;font-size:.72rem;color:#e9d5ff;line-height:1.45}
  body.collect .tray{display:block}
  .tray b{color:#fff}
  /* Pinned, so the count and the commit never scroll away from the rows being
     ticked. */
  .actionbar{position:fixed;left:0;right:0;bottom:0;z-index:30;display:none;
    align-items:center;gap:8px;padding:9px 10px;background:var(--surface);
    border-top:1px solid var(--review);padding-bottom:calc(9px + var(--safe-b))}
  body.collect .actionbar{display:flex}
  .actionbar .n{font-size:.75rem;font-weight:700;color:var(--review);white-space:nowrap}
  .actionbar .sp{flex:1 1 auto}
  .ab-go{background:var(--review);border:1px solid var(--review);color:#fff;border-radius:8px;
    height:36px;padding:0 13px;font-size:.78rem;font-weight:600;cursor:pointer}
  .ab-go:disabled{opacity:.35}
```

- [ ] **Step 3: Add the markup**

In `.listhead`, before the existing `collapse-btn`:

```html
    <button class="icon-btn" id="collectBtn" onclick="toggleCollect()"
            title="Collect events into a playlist">☑</button>
```

Immediately after the `.listhead` div closes and before `<div class="list" id="list">`:

```html
  <div class="tray" id="tray"></div>
```

At the end of `<body>`, before the scrim:

```html
<div class="actionbar">
  <span class="n" id="selN">0 selected</span>
  <button class="icon-btn" id="selAll" onclick="selectAll()">All</button>
  <span class="sp"></span>
  <button class="icon-btn" onclick="toggleCollect()">Cancel</button>
  <button class="ab-go" id="abGo" onclick="addToSheet()" disabled>Add to…</button>
</div>
```

- [ ] **Step 4: Make rows tick instead of play while collecting**

In `render()`, the list rendering currently reads:

```js
  L.innerHTML=windowed.map(e=>{
    const id=TR.evId(e);
    return `<div class="ev${activeId===id?' active':''}" style="border-left-color:${col(e.type)}"
      onclick="playEvent(${e.time},'${id.replace(/'/g,"\\'")}','${String(e.game).replace(/'/g,"\\'")}')">
      <span class="ev-t">${shortT(e.timeStr)}</span>
```

Replace those four lines with:

```js
  L.innerHTML=windowed.map(e=>{
    const id=TR.evId(e), esc1=s=>String(s).replace(/'/g,"\\'");
    const cls=collecting?(sel.has(id)?' sel':''):(activeId===id?' active':'');
    const tap=collecting?`tick('${esc1(id)}')`
      :`playEvent(${e.time},'${esc1(id)}','${esc1(e.game)}')`;
    return `<div class="ev${cls}" style="border-left-color:${col(e.type)}" onclick="${tap}">
      ${collecting?'<span class="tick">✓</span>':`<span class="ev-t">${shortT(e.timeStr)}</span>`}
```

…leaving the rest of the template literal as it is.

Then, at the end of `render()` just before `paintMarks();`, add:

```js
  paintCollect();
```

- [ ] **Step 5: Add the collect functions**

```js
// The ☑ is hidden while a playlist plays: a playing playlist is not a pool to
// collect from, and editing one has a home already in the reorder screen.
function paintCollect(){
  $('collectBtn').hidden = !!plOpen;
  $('selN').textContent=`${sel.size} selected`;
  $('selAll').textContent=`All ${shown.length}`;
  $('abGo').disabled=!sel.size;
  $('abGo').textContent=sel.size?`Add ${sel.size} to…`:'Add to…';
  $('tray').innerHTML=sel.size
    ? `<b>${sel.size} collected.</b> Change the filter and keep tapping — the collection is kept until you save it.`
    : 'Tap events to collect them. Filters can change while you do.';
}

function toggleCollect(){
  collecting=!collecting;
  document.body.classList.toggle('collect',collecting);
  $('collectBtn').classList.toggle('on',collecting);
  if(!collecting) sel.clear();
  render();
}

function tick(id){ sel.has(id)?sel.delete(id):sel.add(id); render(true); }
function selectAll(){ shown.forEach(e=>sel.add(TR.evId(e))); render(true); }
```

- [ ] **Step 6: Add the "Add to…" sheet**

```js
// Committing is one write, however many filters the collection was gathered
// across.
function addToSheet(){
  openSheet('addto');
  // A naming field rather than prompt(): prompt() appears nowhere in this
  // codebase, and iOS standalone web apps handle it badly.
  $('sheetBody').innerHTML=
    `<p class="win-note">Adding <b>${sel.size}</b> event${sel.size===1?'':'s'}.</p>`+
    '<div class="glist">'+PL.map(p=>`<button onclick="addTo('${p.id}')">
        ${esc(p.name)}<span class="gs">${p.refs.length} events</span></button>`).join('')+
    '</div>'+
    `<div class="grp"><h3>New playlist</h3>
      <input class="srch" id="newPlName" placeholder="Name it…"
             onkeydown="if(event.key==='Enter')addTo(null)">
      <button class="icon-btn" style="width:100%;height:38px" onclick="addTo(null)">
        ＋ Create and add ${sel.size}</button></div>`;
}

async function addTo(id){
  const pl=id?plById(id):null;
  const name=pl?pl.name:($('newPlName').value||'').trim();
  if(!pl&&!name){ $('newPlName').focus(); return; }
  // Existing refs first, new ones appended in the order they were ticked — the
  // reorder screen is where an order is chosen deliberately. The Set drops a
  // re-tick of something already in the playlist.
  const refs=[...new Set([...(pl?pl.refs:[]),...sel])];
  if(refs.length>TR.playlists.MAX_REFS){
    alert(`A playlist holds at most ${TR.playlists.MAX_REFS} events.`); return;
  }
  const n=sel.size;
  try{
    await TR.playlists.save(TR.secret(),{id:pl?pl.id:'',name,note:pl?pl.note:'',refs});
    PL=await TR.playlists.load(TR.secret());
  }catch(e){ alert(e.message); return; }
  closeSheet(); toggleCollect();
  $('count').textContent=`Added ${n} to ${name}`;
}
```

Add the sheet kind to `openSheet`'s two ternaries:

```js
    :kind==='playlists'?'Playlists':kind==='addto'?'Add to playlist':'Filter events';
```

```js
    :kind==='playlists'?playlistSheet():kind==='addto'?$('sheetBody').innerHTML:filterSheet();
```

`addToSheet()` writes the body itself after `openSheet` runs, so the `addto` branch must not clobber it — returning the current `innerHTML` keeps `openSheet` generic without a special case.

- [ ] **Step 7: Exit collect when entering cinema**

In `toggleCinema()`, at the top of the function:

```js
  // The pinned action bar would float over a full-screen video with no list
  // behind it.
  if(collecting) toggleCollect();
```

- [ ] **Step 8: Verify in the browser**

On `http://localhost:8765/game.html`, with **All games** picked:

1. Tap **☑** → rows show empty ticks, the tray appears, the bottom bar reads `0 selected` and `Add to…` is disabled.
2. Tap three rows → they turn purple, the bar reads `3 selected`, `Add 3 to…` is enabled.
3. **The key check:** open the filter sheet, filter to a different Type, close it. The tray still reads `3 collected`, and the bar still says `3 selected`. Tick two more from this filter → `5 selected`.
4. Tap **Add 5 to…** → the sheet lists your playlists, then a **New playlist** name field. Type a name and tap **＋ Create and add 5** → the sheet closes, collect mode exits, and the count line reads `Added 5 to <name>`. Tapping create with the field empty focuses it rather than saving.
5. Tap **📋** → the new playlist is listed with 5 events. Play it → all five are there, spanning both filters.
6. Tap **All N** with a filter applied → every row in the filtered list ticks.
7. While a playlist plays, confirm the **☑** button is hidden; leave the playlist and confirm it returns.
8. Enter cinema while collecting → collect mode exits and the action bar is gone.
9. Clean up the test playlist from the console.

- [ ] **Step 9: Commit**

```bash
git add game.html
git commit -m "$(cat <<'EOF'
feat: collect mode gathers events across filters into a playlist

Ticking holds TR.evId refs rather than list indices. An index is a position in
shown, and shown is rebuilt by every filter change — an index-based selection
would silently repoint at different events the moment the user filtered, which
is the one motion this mode exists to support. The tray keeps the running count
visible while the list changes underneath it, and the whole collection commits
in one write.

The tick replaces the timestamp column instead of adding one, so a row costs
nothing when not collecting. Cinema exits the mode rather than floating a
pinned bar over a full-screen video.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: The reorder screen

**Files:**
- Modify: `game.html` (CSS, `playlistSheet`, new drill level, `openSheet`)

**Interfaces:**
- Consumes: `PL`, `plOpen`, `plById`, `playPlaylist` (Task 5); `TR.playlists.reorder`, `TR.playlists.save`, `TR.playlists.remove`, `TR.playlists.resolve`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Add the drill state**

Beside `fCat`:

```js
// null at the playlist list, otherwise the playlist id being edited. Reset on
// every open, for the same reason fCat is: reopening the sheet must never
// strand the user inside a playlist.
let plEdit=null, plDraft=null;
```

`plDraft` holds the pending `{name, note, refs}` so five drag moves are one write on leaving, not five.

- [ ] **Step 2: Turn the sheet into a two-way switch**

Rename the existing `playlistSheet()` body to `plList()` and add the switch, mirroring `filterSheet()`:

```js
function playlistSheet(){ return plEdit?plEditBody():plList(); }
```

In `plList()`, add a ⠿ to each row. Replace the button template with:

```js
    return `<div class="pl-row${p.id===plOpen?' on':''}">
      <button class="pl-go" onclick="playPlaylist('${p.id}')">
        ${esc(p.name)}
        <span class="gs">${r.events.length} events${gone}${p.note?' · '+esc(p.note):''}</span>
      </button>
      <button class="pl-edit" onclick="plDrill('${p.id}')" title="Edit">⠿</button>
    </div>`
```

and drop the wrapping `'<div class="glist">'+ … +'</div>'` in favour of `'<div class="pl-list">'+ … +'</div>'`.

- [ ] **Step 3: Add the CSS**

```css
  /* ── Playlists sheet ───────────────────────────────────────── */
  .pl-row{display:grid;grid-template-columns:1fr 44px;gap:6px;align-items:stretch;margin-bottom:6px}
  .pl-row button{background:var(--surface2);border:1px solid var(--border);color:var(--text);
    border-radius:8px;font-size:.8rem;cursor:pointer;font-family:inherit}
  .pl-go{text-align:left;padding:10px 12px}
  .pl-go .gs{display:block;font-size:.66rem;color:var(--dim);margin-top:2px}
  .pl-row.on .pl-go{border-color:var(--review);background:#231733}
  .pl-edit{font-size:1rem;color:var(--dim)}
  /* Reorder rows: the handle takes the timestamp's column, ↑↓ take the right. */
  .ev.ord{grid-template-columns:26px 1fr 70px;cursor:default}
  .ev.ord .handle{color:var(--dim);font-size:1rem;text-align:center}
  .ord-btns{display:flex;gap:3px;justify-self:end}
  .ord-btns button{width:30px;height:30px;border-radius:7px;background:var(--surface2);
    border:1px solid var(--border);color:var(--dim);font-size:.75rem;cursor:pointer}
  .pl-danger{width:100%;height:38px;margin-top:12px;background:var(--surface2);
    border:1px solid var(--pdef);color:#fca5a5;border-radius:8px;font-size:.78rem;
    cursor:pointer;font-family:inherit}
```

- [ ] **Step 4: Add the edit screen**

```js
function plDrill(id){
  plEdit=id;
  // A draft, so five moves are one write on leaving rather than five.
  if(id){const p=plById(id); plDraft={name:p.name,note:p.note,refs:p.refs.slice()};}
  else plDraft=null;
  $('sheetBody').innerHTML=playlistSheet();
  paintPlHead();
}

// The title doubles as the back control when drilled in, the same shape the
// filter sheet uses.
function paintPlHead(){
  const h=$('sheetTitle');
  h.innerHTML=plEdit?'‹ Playlists':'Playlists';
  h.className=plEdit?'back':'';
  h.onclick=plEdit?()=>plSaveAndBack():null;
}

function plEditBody(){
  const r=TR.playlists.resolve(plDraft.refs,rows);
  return `<input class="srch" id="plName" value="${esc(plDraft.name)}"
      placeholder="Playlist name" oninput="plDraft.name=this.value">`+
    (r.missing?`<p class="win-note">${r.missing} event${r.missing===1?'':'s'} in this playlist `+
      `${r.missing===1?'is':'are'} no longer in the data.</p>`:'')+
    (r.events.length?r.events.map((e,n)=>
      `<div class="ev ord" style="border-left-color:${col(e.type)}">
        <span class="handle">⠿</span>
        <span class="ev-main">
          <span class="ev-name">${esc(e.name||e.type)}</span>
          <span class="ev-meta">
            <span class="tag" style="background:${col(e.type)}22;color:${col(e.type)}">${esc(e.type)}</span>
            <span class="gmatch">${esc(shortLabel(e.game))}</span>
            ${e.team?`<span class="team">${esc(e.team)}</span>`:''}
          </span>
        </span>
        <span class="ord-btns">
          <button onclick="plMove(${n},-1)" ${n===0?'disabled':''}>↑</button>
          <button onclick="plMove(${n},1)" ${n===r.events.length-1?'disabled':''}>↓</button>
          <button onclick="plDrop(${n})">✕</button>
        </span>
      </div>`).join('')
      :'<div class="empty">Nothing in here yet.</div>')+
    `<button class="pl-danger" onclick="plDelete()">Delete this playlist</button>`;
}
```

Note the ↑↓ act on the *resolved* index, which equals the ref index only when nothing is missing. Keep them in step by rebuilding the draft's refs from what resolved:

```js
// Missing refs are dropped from the draft the moment it is edited, so the
// buttons' indices and the stored refs can't drift apart. A playlist you edit
// is a playlist you've seen, so silently shedding a ref you were just told
// about is honest.
function plLive(){
  return TR.playlists.resolve(plDraft.refs,rows).events.map(TR.evId);
}
function plMove(n,d){ plDraft.refs=TR.playlists.reorder(plLive(),n,n+d);
  $('sheetBody').innerHTML=playlistSheet(); }
function plDrop(n){ const r=plLive(); r.splice(n,1); plDraft.refs=r;
  $('sheetBody').innerHTML=playlistSheet(); }
```

- [ ] **Step 5: Save on leaving, and delete**

```js
async function plSaveAndBack(){
  const id=plEdit, d=plDraft;
  plEdit=null; plDraft=null;
  $('sheetBody').innerHTML=playlistSheet(); paintPlHead();
  const p=plById(id);
  if(!p||!d) return;
  if(d.name===p.name&&d.note===p.note&&d.refs.join('\n')===p.refs.join('\n')) return;
  if(!d.name.trim()){ alert('A playlist needs a name.'); return; }
  try{
    await TR.playlists.save(TR.secret(),{id,name:d.name.trim(),note:d.note,refs:d.refs});
    PL=await TR.playlists.load(TR.secret());
  }catch(e){ alert(e.message); return; }
  $('sheetBody').innerHTML=playlistSheet();
  if(plOpen===id) render();
}

async function plDelete(){
  const id=plEdit;
  if(!confirm('Delete this playlist? This cannot be undone.')) return;
  try{
    await TR.playlists.remove(TR.secret(),id);
    PL=await TR.playlists.load(TR.secret());
  }catch(e){ alert(e.message); return; }
  // Never leave the page playing a list that no longer exists.
  if(plOpen===id){ plOpen=null; render(); }
  plEdit=null; plDraft=null;
  $('sheetBody').innerHTML=playlistSheet(); paintPlHead();
}
```

- [ ] **Step 6: Reset the drill on open and paint the header**

In `openSheet(kind)`, beside the existing `if(kind==='filter')fCat=null;`:

```js
  if(kind==='playlists'){plEdit=null;plDraft=null;}
```

and after the two ternaries, beside the `if(kind==='window')paintWin();` line:

```js
  if(kind==='playlists')paintPlHead();
```

- [ ] **Step 7: Verify in the browser**

1. Create a five-event playlist via collect mode (Task 6).
2. Tap **📋** → the row shows the name, the count, and a ⠿.
3. Tap ⠿ → the title becomes `‹ Playlists`, the name is editable, five rows show with ⠿ and ↑↓✕.
4. ↑ on the first row and ↓ on the last are disabled.
5. Move the last row to the top with repeated ↑ → the order updates each tap.
6. ✕ a row → four remain.
7. Edit the name.
8. Tap `‹ Playlists` → back at the list, the new name and `4 events` show. Reload the page and tap 📋 → the change persisted.
9. Play the playlist, reopen 📋, drill in, reorder, go back → the event list behind the sheet reflects the new order.
10. Drill in and tap **Delete this playlist** → confirm → it's gone from the list, and if it was playing the page returns to the filtered view.
11. Close and reopen the sheet while drilled in → it opens at the list, not stranded inside a playlist.

- [ ] **Step 8: Commit**

```bash
git add game.html
git commit -m "$(cat <<'EOF'
feat: a reorder screen for playlists

The playlists sheet becomes two steps, reusing the drill the filter sheet
already has: a state variable, a switch between renderers, and the header title
as the back control. Edits are held in a draft and written once on leaving, so
five moves are one request rather than five.

Buttons ship alongside the handle rather than drag alone: a 52px row inside a
scrolling sheet is a poor drag target on a phone, and the buttons are what makes
the ordering checkable without synthesising pointer events. Deleting the
playlist that's playing clears it, rather than leaving the page playing a list
that no longer exists.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Documentation and the cache bump

**Files:**
- Modify: `README.md`
- Modify: `sw.js`

- [ ] **Step 1: Document the feature in the Event Viewer section**

In `README.md`, in the **Event Viewer (`game.html`)** section, after the paragraph describing the playback window, add:

```markdown
**Playlists** are saved, hand-picked sets of events that span games and filters —
the eleven backdoors you want to show on Tuesday, in the order you want to show
them. Tap **☑** in the filter bar to start collecting: rows tick instead of
playing, and the collection survives a filter change, so you can gather tries
here and turnovers there and save the lot in one go. **📋** lists what you've
saved; tapping one makes it the event list, so prev/next, loop and the playback
window all follow it. The ⠿ beside a playlist opens rename, reorder and delete.

Playlists belong to the account that made them, not the device, and are stored
in a `_playlists` tab. An event is remembered by its game and timestamp, so
correcting a Name in the Event Editor won't break a saved playlist; anything
that genuinely can't be found any more is reported in the filter bar rather than
silently dropped.
```

- [ ] **Step 2: Document the sheet**

In the **Apps Script backend** section, after the `_live` sheet subsection, add:

```markdown
### `_playlists` sheet

One row per saved playlist. Columns: `Id`, `Owner`, `Name`, `Note`, `Refs`,
`Updated At`.

- **Owner** — the account secret that owns it. Every read and write is filtered
  and checked against this server-side.
- **Refs** — the playlist's events, one `game#time#type#name` ref per line, in
  playback order. Only the game and time are matched on; the type and name are
  stored so the row stays readable, and go stale harmlessly after a rename.

Capped at 500 events per playlist. Not editable from the admin sheet editor —
it's user content, not control plane.
```

- [ ] **Step 3: Bump the service worker cache**

`game.html`, `README.md` and `js/playlists.js` have all changed since Task 2's bump. Check and increment once more:

```bash
grep -n "CACHE_VERSION" sw.js
```

Bump the number by one (e.g. `'trl-shell-v21'` → `'trl-shell-v22'`).

- [ ] **Step 4: Run the full test suite**

```bash
node test.js
```

Expected: `0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add README.md sw.js
git commit -m "$(cat <<'EOF'
docs: playlists in the Event Viewer and the _playlists sheet

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes

**Spec coverage.** §1 sheet → Task 3 Steps 1–2. §2 loose resolution → Task 1 (`refKey`), Task 2 (`resolve`), Task 5 Step 3 (the warning line). §3 API → Task 3 Steps 3–4. §4 module → Tasks 2 and 4. §5 `evId` move → Task 1. §6 collect mode → Task 6 (cinema exit Step 7, hidden-while-playing in `paintCollect`). §7 sheet and reorder → Tasks 5 and 7. §8 playing → Task 5 (filters suppressed Step 3, `pickGame` Step 8, `#playlists` Step 8, delete-while-playing Task 7 Step 5). Testing → Tasks 1, 2 (Node) and per-task browser checks; `sw.js` → Tasks 2 and 8.

**Naming.** `TR.evId` / `TR.evKey` / `TR.refKey`, `TR.playlists.{resolve,reorder,load,save,remove,MAX_REFS}`, and the `game.html` globals `PL / plOpen / plMissing / plEdit / plDraft / collecting / sel` are used with these exact spellings in every task that references them.

**Known sharp edge, handled deliberately:** Task 7's ↑↓ act on resolved indices while the draft stores refs. `plLive()` rebuilds the draft from what resolved, so a playlist with missing refs sheds them on its first edit rather than letting the two lists drift. That is a real behaviour change and is commented as such in the code.
