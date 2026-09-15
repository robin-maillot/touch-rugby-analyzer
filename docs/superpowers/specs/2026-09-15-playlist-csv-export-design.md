# Export a playlist as CSV

Date: 2026-09-15

## Problem

A playlist is watchable in the Event Viewer but not shareable. A coach who has
built "the eleven backdoors for Tuesday" can only hand it over by sitting
someone down in front of the same phone. There is no way to take the list
somewhere else — a spreadsheet, a team chat, a session plan.

## Goal

Export the playlist you are editing as a CSV, one row per event, each carrying a
YouTube link that opens the video at that moment.

## Design

### 1. Where it lives

A full-width **⬇ Export CSV** button in `plEditBody()`, directly above **Delete
this playlist** — the screen that already owns rename, note, reorder and delete.
Styled as a normal `.icon-btn`, not `.pl-danger`; this is not a destructive
action.

It exports the **draft**, not the saved row. `plEditBody()` already computes

```js
const r = plResolved = TR.playlists.resolve(plDraft.refs, rows);
```

stashing the result in a module-level `plResolved`, and the export reuses that
stash rather than resolving again. Resolving a second time at click time would
read whatever `rows` holds *then* — and `rows` is reassigned whole when a
fetch lands, independently of what the playlists sheet is showing — so a click
some time after paint could silently export something other than what is on
screen. The stash can't go stale on order either: `plMove`/`plDrop`/`plDrill`
all repaint through `plEditBody()`, which recomputes it on every reorder. What
is on screen is what comes out. One resolution, reused, not two.

Disabled when the playlist resolves to no events, so the button can never
produce a file with a header and nothing under it.

Wiring is a `data-plexport` branch on the existing `$('sheetBody')` delegated
listener, bounded with `within()`. No inline handler — see Global constraints.

### 2. Columns

```
#, Game, Time, Type, Name, Team, Move, Comment, YouTube Link
```

- **#** — 1-based position **in the exported file**, so it always runs
  `1..N` with no gaps. It is not the position in the stored ref list: when a ref
  no longer resolves it is absent from the file, and a `#` column that skipped
  4 would invite the reader to hunt for a row that is not coming. The order
  survives a sort in Excel either way.
- **Game** — `labels[e.game]` (the full `TR.sheetNameToLabel`, carrying year,
  division and competition), falling back to the raw sheet name. Deliberately
  *not* `shortLabel()`: a playlist spans matches, and two `France vs England`
  rows from different years would otherwise be indistinguishable.
- **Time** — `e.timeStr`, the original `H:MM:SS` from the sheet, not raw
  seconds.
- **Type / Name / Team / Move / Comment** — as the event list shows them.
- **YouTube Link** — `TR.player.seekLink(meta[e.game], e.time)`.

`seekLink` already does the hard part and is unit-tested: it applies the game's
`videoOffset` (confirmed present in the `action=list` payload), subtracts the
5-second lookback, and returns `''` when the game has no video or the moment
precedes the recording. An event with no footage still exports, with an empty
link cell — the row count always matches the playlist, because an export that
silently has fewer rows than the list it came from is the kind of mismatch
nobody notices until it matters.

Refs that no longer resolve are absent from `r.events` and so from the file.
The screen already says how many those are, immediately above the button.

### 3. Two pure helpers in `js/utils.js`

```js
TR.csvCell(v)    // → a single RFC 4180 field
TR.toCSV(rows)   // → array of arrays, CRLF-joined
```

**`csvCell` does two jobs.**

*Quoting.* Double every `"`, and wrap the field in quotes when it contains `"`,
`,`, CR or LF. `null` and `undefined` become an empty field; numbers stringify.

*Formula-injection guard.* A field whose first character is `=`, `+`, `-` or
`@` is prefixed with an apostrophe. Excel and Google Sheets **execute** such a
cell on open, and a Comment is free text an annotator types — so this is a real
path, not a theoretical one. It is the same class of bug closed twice
server-side in `sheetSafe()`, except the payload would land in the recipient's
spreadsheet rather than ours.

The visible cost is that a comment of `-5m` exports as `'-5m`. That is accepted:
a leading apostrophe is the conventional spreadsheet text marker and is consumed
on display by Sheets, and the alternative is shipping a file that can run code
on someone else's machine.

`toCSV` joins fields with `,` and rows with `\r\n`, per RFC 4180.

This deliberately does **not** reuse `annotator.html`'s existing export, which
does a bare `cols.join(',')` — a comment containing a comma silently corrupts
that file today. Fixing it is out of scope here; see Out of scope.

### 4. The download

`Blob` + a synthetic `<a download>`, the same idiom as `annotator.html`'s
`exportCSV()`.

The file is prefixed with a UTF-8 **BOM** (`﻿`). Without it Excel reads the
file as the system codepage and mangles every accented team name, and this
dataset is largely French and European.

The filename is slugified from the playlist name, in this order: fold
diacritics (NFD normalize, strip the combining marks), lowercase, collapse
every run of non-alphanumerics to a single `-`, truncate to 60 characters, then
trim leading and trailing `-`. Trimming last matters — cutting
at 60 can land mid-separator and would otherwise leave `some-name-.csv`. A name
that slugifies to nothing falls back to `playlist`.

`Backdoor teaching set` → `backdoor-teaching-set.csv`. Slugification also keeps
a name carrying `/`, `\` or a control character from producing a filename the
browser or OS refuses.

### 5. Unchanged

The playlist data model, the three Apps Script actions, `resolve`/`reorder`,
collect mode, and playback. This is a read-only addition: it writes no sheet
data and makes no network call.

## Testing

`js/utils.js` is already loaded by `game.html` and already on the Node harness's
list in `test.js`, so no new wiring. `test.js` covers:

- `TR.csvCell` — a plain value; a value containing a comma; one containing a
  quote (doubled *and* wrapped); one containing a newline; each of `=`, `+`,
  `-`, `@` as the leading character; `null`, `undefined`, a number, and the
  empty string.
- `TR.toCSV` — a header plus two rows, joined with CRLF; an empty input; a row
  whose fields need mixed treatment.

By hand in the browser: the button appears and is disabled for an empty
playlist; the file downloads with the expected name; it opens in a spreadsheet
with the right row count and columns; a link from the file opens YouTube at the
right moment; and an event whose game has no video exports with an empty link
cell rather than being dropped.

## Out of scope

- **`annotator.html`'s CSV export.** Its unescaped `cols.join(',')` is a real
  pre-existing bug, but fixing it means touching the annotator's round-trip with
  its own `loadCSV`, which is a different change with a different risk profile.
- **Importing a CSV back into a playlist.** No demand, and the refs a playlist
  needs are not reconstructible from the human-readable columns above.
- **Exporting the current filter rather than a playlist.** Considered and set
  aside: a second entry point for the same file, with a fuzzier answer to "what
  did I just export".
- **Clip export via the Cloud Run service.** Still the obvious follow-on, still
  gated on GCS-backed source video that not every game has.
