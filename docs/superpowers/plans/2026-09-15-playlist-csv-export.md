# Playlist CSV Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the playlist you are editing as a CSV, one row per event, each carrying a YouTube link timed to that moment.

**Architecture:** Three pure helpers go in `js/utils.js` (CSV field quoting, row joining, filename slugging) where the Node harness can test them. `game.html` gains an export button on the playlist edit screen that resolves the draft, builds rows, and hands a Blob to a synthetic `<a download>`. No network call, no sheet write — this is read-only.

**Tech Stack:** Vanilla JS, no build step. Node `test.js` harness (`node test.js`, no npm).

## Global Constraints

- **No build step, no npm.** Plain `<script src>`; `js/*.js` attach to the global `TR` declared with `var` in `js/config.js`.
- **Never build an event handler, or a CSS selector, by interpolating data.** Values ride in `data-` attributes through `esc()`, read off `dataset` by ONE delegated listener per container, every match bounded with the existing `within(ev, el)` helper. Commits `00b52a3` / `6892607` removed exactly this bug class from `game.html`.
- **`TR.csvCell` must neutralise a leading `=`, `+`, `-` or `@`** by prefixing an apostrophe. Excel and Google Sheets execute such a cell on open.
- **CSV is RFC 4180:** fields joined with `,`, rows joined with `\r\n`, embedded `"` doubled, field quoted when it contains `"`, `,`, CR or LF.
- **The file is prefixed with a UTF-8 BOM** (`\uFEFF`).
- **Filename slug order:** lowercase → strip diacritics → collapse non-alphanumerics to `-` → truncate to 60 → trim leading/trailing `-`. Trimming is last.
- Comment style: explain WHY, not what. Match the register of `js/utils.js`.
- Commit messages: imperative mood, a body explaining the reasoning, ending with exactly:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

**Spec:** `docs/superpowers/specs/2026-09-15-playlist-csv-export-design.md`

---

### Task 1: The three pure helpers

**Files:**
- Modify: `js/utils.js` (append at end)
- Test: `test.js`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TR.csvCell(v) -> string` — one RFC 4180 field, formula-guarded.
  - `TR.toCSV(rows) -> string` — array of arrays → CRLF-joined text, no trailing newline.
  - `TR.slugify(s, fallback) -> string` — filename-safe slug.

- [ ] **Step 1: Write the failing tests**

Append to `test.js`, immediately before the final summary block (`// ─────` then `console.log(\`\n${passed} passed, ${failed} failed\`)`):

```js
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
// The guard runs BEFORE quoting, so a formula carrying a comma is both
// neutralised and quoted — quoting first would bury the apostrophe inside.
test('formula + comma',  () => assert.equal(TR.csvCell('=A1,B1'), `"'=A1,B1"`));
test('apostrophe mid-string is untouched', () => assert.equal(TR.csvCell("Dad's Army"), "Dad's Army"));

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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
node test.js
```

Expected: a run of `✗` lines reading `TR.csvCell is not a function` and similar, and a non-zero exit.

- [ ] **Step 3: Append the three helpers to `js/utils.js`**

```js
// One RFC 4180 CSV field.
//
// Two jobs. The quoting is ordinary — double every embedded quote, wrap the
// field when it carries a quote, a comma, CR or LF.
//
// The second job is the one that matters. Excel and Google Sheets EXECUTE a
// cell whose first character is = + - or @, and an event Comment is free text
// an annotator typed, so an exported file could run code on whoever opens it.
// The leading apostrophe is the conventional spreadsheet text marker and is
// consumed on display. Same class of bug sheetSafe() closes server-side, except
// this payload would land in the recipient's spreadsheet rather than ours.
//
// Guard first, quote second: quoting first would bury the apostrophe inside the
// quotes, where it protects nothing.
TR.csvCell = (v) => {
  let s = v == null ? '' : String(v);
  if (/^[=+\-@]/.test(s)) s = "'" + s;
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
};

// Rows of fields → RFC 4180 text. CRLF between rows, which is what the format
// says and what Excel expects; no trailing newline, so the file has no phantom
// final row.
TR.toCSV = (rows) => (rows || [])
  .map(r => (r || []).map(c => TR.csvCell(c)).join(','))
  .join('\r\n');

// A filename-safe slug. Diacritics are folded rather than dropped, because this
// dataset is largely French and "Équipe" deserves better than "quipe".
//
// Order matters: truncate BEFORE trimming separators. A cut at the cap can land
// mid-separator, and trimming first would leave the dash behind.
TR.slugify = (s, fallback) => {
  const out = String(s == null ? '' : s)
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .slice(0, 60)
    .replace(/^-+|-+$/g, '');
  return out || fallback || 'file';
};
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
node test.js
```

Expected: the new `✓` lines, `0 failed`, exit 0. The suite was at 238 passed before this task; it should now read 263 passed.

- [ ] **Step 5: Commit**

```bash
git add js/utils.js test.js
git commit -m "$(cat <<'EOF'
feat(utils): csvCell, toCSV and slugify for exporting a playlist

csvCell does RFC 4180 quoting and neutralises a leading = + - @, because Excel
and Sheets execute such a cell on open and an event Comment is free text an
annotator typed. The guard runs before the quoting; quoting first would bury the
apostrophe inside the quotes where it protects nothing.

slugify folds diacritics rather than dropping them - this dataset is largely
French - and truncates before trimming separators, so a cut landing mid-word
can't leave a trailing dash.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: The export button

**Files:**
- Modify: `game.html` (CSS, `plEditBody`, new `plExport`, the `$('sheetBody')` listener)

**Interfaces:**
- Consumes: `TR.csvCell`, `TR.toCSV`, `TR.slugify` from Task 1; and existing `TR.playlists.resolve(refs, rows)`, `TR.player.seekLink(meta, seconds)`, plus `game.html`'s module-level `rows`, `meta`, `labels`, `plDraft`.
- Produces: `plExport()` and the `data-plexport` hook. Nothing later depends on them.

- [ ] **Step 1: Add the button style**

In `game.html`'s `<style>`, immediately after the `.pl-danger{...}` rule:

```css
  /* Same footprint as .pl-danger below it, but this is not destructive, so it
     keeps the ordinary button colours. */
  .pl-export{width:100%;height:38px;margin-top:12px}
```

- [ ] **Step 2: Add the button to `plEditBody()`**

In `plEditBody()`, the return currently ends:

```js
      :'<div class="empty">Nothing in here yet.</div>')+
    `<button class="pl-danger" onclick="plDelete()">Delete this playlist</button>`;
```

Change it to:

```js
      :'<div class="empty">Nothing in here yet.</div>')+
    `<button class="icon-btn pl-export" data-plexport="1"
       ${r.events.length?'':'disabled'}>⬇ Export CSV</button>`+
    `<button class="pl-danger" onclick="plDelete()">Delete this playlist</button>`;
```

`r` is already in scope — `plEditBody` computes `const r=TR.playlists.resolve(plDraft.refs,rows)` on its first line. Do not resolve a second time.

- [ ] **Step 3: Add `plExport()`**

Add immediately after `plEditBody()`:

```js
// Exports the DRAFT, not the saved row: what is on screen is what comes out, so
// a reorder the user hasn't left the screen with still exports in the order
// they're looking at.
function plExport(){
  if(!plDraft) return;
  const r=TR.playlists.resolve(plDraft.refs,rows);
  if(!r.events.length) return;          // the button is disabled too; belt and braces
  const head=['#','Game','Time','Type','Name','Team','Move','Comment','YouTube Link'];
  const body=r.events.map((e,n)=>[
    // Position in the FILE, not in the stored ref list — a ref that no longer
    // resolves is absent here, and a # that skipped 4 would send the reader
    // hunting for a row that isn't coming.
    n+1,
    // The full label, not shortLabel: a playlist spans matches, and two
    // "France vs England" rows from different years must be tellable apart.
    labels[e.game]||e.game,
    e.timeStr, e.type, e.name, e.team, e.move, e.comment,
    // seekLink applies the game's videoOffset and the 5s lookback, and returns
    // '' when the game has no video or the moment predates the recording. The
    // row still exports, so the file's length always matches the list above it.
    TR.player.seekLink(meta[e.game],e.time)
  ]);
  // BOM: without it Excel reads the file as the system codepage and mangles
  // every accented team name, and this dataset is largely French.
  const blob=new Blob(['\uFEFF'+TR.toCSV([head,...body])],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=TR.slugify(plDraft.name,'playlist')+'.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}
```

- [ ] **Step 4: Wire the delegated listener**

In the `$('sheetBody').addEventListener('click', …)` handler, add as the last branch, after the existing `data-plnew` line:

```js
  if(within(ev,ev.target.closest('[data-plexport]'))){plExport();return}
```

Do NOT add an `onclick` to the button — `game.html` routes every data-carrying control through this one delegated listener, and `00b52a3` removed the inline-handler pattern from this file deliberately.

- [ ] **Step 5: Confirm the harness still passes**

```bash
node test.js
```

Expected: 263 passed, 0 failed. (This task changes no file the harness loads; run it to confirm nothing broke.)

- [ ] **Step 6: Syntax-check the inline script**

Extract `game.html`'s inline `<script>` block to a temporary `.js` file and run `node --check` on it. Never execute it — it references browser globals. Delete the temp file afterwards.

Expected: no output (syntax OK).

- [ ] **Step 7: Verify in a browser**

The page needs a server and a session. Start one on a free port and seed `sessionStorage` before navigating, because `TR.auth('viewer')` redirects to `index.html` without it:

```js
sessionStorage.setItem('password','m30'); sessionStorage.setItem('role','viewer');
```

Then on `game.html`:

1. Pick **All games**, enter collect mode (☑), tick four events **from at least two different matches**, and save them to a new playlist.
2. Open 📋, tap ⠿ on that playlist. The **⬇ Export CSV** button is present, above *Delete this playlist*, and enabled.
3. Tap it. A file downloads named after the playlist, slugified, ending `.csv`.
4. Open the file. Confirm: a header row, then exactly four rows; `#` runs 1–4; `Game` names both matches distinguishably; `YouTube Link` cells are populated for games that have video.
5. Click one of the links. It opens YouTube at roughly five seconds before that event.
6. Reorder two rows with ↑/↓ **without leaving the screen**, export again, and confirm the new file reflects the on-screen order — this is the draft-not-saved-row behaviour.
7. Rename the playlist to something with a comma and an accent (e.g. `Équipe, review`), export, and confirm the filename is `equipe-review.csv` and the file still parses.
8. Add an event whose game has no YouTube link, export, and confirm that row is present with an **empty** link cell rather than being dropped.
9. Empty the playlist with ✕ and confirm the button renders `disabled`.

- [ ] **Step 8: Commit**

```bash
git add game.html
git commit -m "$(cat <<'EOF'
feat: export a playlist as CSV with YouTube links

A playlist was watchable but not shareable. The edit screen gains an export
button that writes one row per event - position, match, clock time, type, name,
team, move, comment - and a link timed to that moment.

It exports the draft rather than the saved row, so a reorder you haven't left
the screen with still comes out in the order you're looking at; plEditBody has
already resolved the refs, so there is one resolution, not two.

Links come from TR.player.seekLink, which already applies the game's videoOffset
and returns empty for a moment that predates the recording. An event with no
footage still exports, with an empty link cell - an export that silently has
fewer rows than the list it came from is the kind of mismatch nobody notices
until it matters.

The file carries a UTF-8 BOM, without which Excel mangles every accented team
name, and this dataset is largely French.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Documentation and the cache bump

**Files:**
- Modify: `README.md`
- Modify: `sw.js`

- [ ] **Step 1: Document it in the Event Viewer section**

In `README.md`, find the paragraph in the **Event Viewer (`game.html`)** section that describes playlists — it ends with the sentence about the ⠿ opening rename, reorder and delete. Extend that section with:

```markdown
A playlist can be exported as a CSV from its edit screen: one row per event with
the match, clock time, type, name, team, move and comment, plus a YouTube link
that opens the video about five seconds before the moment. Events whose game has
no video still get a row, with an empty link cell, so the file is never quietly
shorter than the playlist. The export reflects what is on screen, so a reorder
you have not saved yet still comes out in the order you are looking at.
```

Verify each claim against the code before writing it — a previous task in this
repo shipped two plan-mandated documentation sentences that were factually
wrong. Specifically confirm: the button's label and location, the exact column
list in `plExport`, the lookback `TR.player.seekLink` applies, and what happens
to an event with no video.

- [ ] **Step 2: Bump the service worker cache**

`game.html`, `js/utils.js` and `README.md` are all shell assets. Read the current value first:

```bash
grep -n "CACHE_VERSION" sw.js
```

Bump the number by one (at the time of writing it reads `'trl-shell-v24'`, so it becomes `'trl-shell-v25'`; if it has moved on, bump whatever is there).

- [ ] **Step 3: Run the full suite**

```bash
node test.js
```

Expected: 263 passed, 0 failed.

- [ ] **Step 4: Commit**

```bash
git add README.md sw.js
git commit -m "$(cat <<'EOF'
docs: playlist CSV export in the Event Viewer

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes

**Spec coverage.** §1 where it lives → Task 2 Steps 1, 2, 4 (button, style, delegation) and Step 3's draft-not-saved-row comment. §2 columns → Task 2 Step 3, including the `#`-is-file-position and full-label decisions. §3 the two helpers → Task 1 (plus `slugify`, which §4 requires). §4 the download → Task 2 Step 3 (Blob, BOM, slug filename). §5 unchanged → nothing in this plan touches the data model, the Apps Script actions, `resolve`/`reorder`, collect mode or playback. Testing → Task 1 (Node) and Task 2 Step 7 (browser). The `sw.js` bump and README, which the spec implies but does not name, are Task 3.

**Naming.** `TR.csvCell`, `TR.toCSV`, `TR.slugify(s, fallback)`, `plExport()`, `data-plexport`, `.pl-export` — used with these exact spellings in every task that references them.

**One thing the implementer must not "tidy":** `plExport` calls `TR.playlists.resolve` again rather than receiving `r` from `plEditBody`. That is deliberate — the button fires from a delegated listener long after `plEditBody` returned, and closing over its local `r` would export a stale resolution after an ↑/↓ or ✕. Resolving from `plDraft.refs` at click time is what makes the export match the screen.
