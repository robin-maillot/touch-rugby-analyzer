# Strike moves on turnovers and attack penalties

**Date:** 2026-09-06
**Status:** Approved, not yet implemented

## Problem

A Try records *which* attacking move scored it, in the `Name` column
(`32 - Cut`, `23 - Backdoor`, `Scoop`, …). Nothing records which move was
being run when an attack *failed*. So we can count how often a move scores,
but not how often it is tried — and a raw try count rewards moves that are
run a lot, not moves that work.

We want two numbers:

- **Try rate per move** — how often a given strike move ends in a try.
- **Top scoring move** — both by volume and by efficiency.

That needs the move recorded on the failures too.

## Domain model

`TR.inferActionOwner` (js/possession.js) returns the possession owner for
Try, Penalty Attack and Turnover alike. So on all three, **Action Owner is
the attacking team that ran the move** — no special-casing is needed to
attribute an attempt to a team.

An attempt **ends exactly when the ball changes hands**, which
`TR.isTurnover(type, name)` already encodes:

| Event | Ends the attempt? | Outcome |
|---|---|---|
| Try | yes | success |
| Penalty Attack | yes | failure |
| Turnover (not `6 Again`) | yes | failure |
| Turnover → `6 Again` | no | attack retains the ball, move continues |
| Penalty Defence | no | attack retains the ball, move continues |

`6 Again` and Penalty Defence are therefore **excluded**, and the annotators
must not offer the move picker on them — one less tap on the sideline, and no
junk in the denominator.

Try rate for move M:

```
tries(M) / (tries(M) + turnovers(M) + penAttacks(M))
```

## Data model

### Sheet

One new column, `Strike Move`, appended to `HEADERS` in `apps_script/Code.gs`
as the 8th column:

```
Time | Possession Owner | Type | Name | To Review | Comment | Action Owner | Strike Move
```

This is safe because every read path resolves columns by header *name*, not
index. `Code.gs:377` already yields `''` for a column a tab lacks, and
`viewer.html:796` builds its table from whatever headers arrive. **Old tabs
need no backfill** — they report blank moves and 0% coverage.

### Shared constants — `js/events.js`

Single source of truth for all three annotators and all four analytics surfaces.

```js
TR.STRIKE_MOVES      = TR.MENU['Try'];   // the exact same 19 options
TR.STRIKE_MOVE_TYPES = ['Try', 'Turnover', 'Penalty Attack'];
TR.MIN_MOVE_ATTEMPTS = 2;

// An attempt ends precisely when the ball changes hands.
TR.isAttackEnd = (type, name) =>
  TR.STRIKE_MOVE_TYPES.includes(type) && TR.isTurnover(type, name);

// A Try's move IS its Name — derived, never stored twice, so editing a Try's
// name cannot leave a stale move behind.
TR.strikeMoveOf = (type, name, strikeMove) =>
    type === 'Try'                  ? (name || '')
  : TR.isAttackEnd(type, name)      ? (strikeMove || '')
  : '';
```

The in-memory event gains one optional field, `strikeMove`. Push writes
`TR.strikeMoveOf(a.type, a.name, a.strikeMove)`, so a Try's row carries its
move without the annotator storing it twice. Every read coalesces with
`|| ''`, so **no localStorage migration** is needed for games already on a phone.

### Stats module — `js/strike_moves.js` (new)

Pure, so it is unit-testable under `node test.js` like the rest of `TR.*`.

```js
TR.strikeMoveStats(events) → {
  moves:      [{ move, tries, fails, attempts, rate }],  // rate desc
  coverage:   { tagged, total, pct },
  topByTries: { move, tries, attempts, rate } | null,
  topByRate:  { move, tries, attempts, rate } | null,    // >= MIN_MOVE_ATTEMPTS
}
```

Takes a normalised `{type, name, strikeMove, actionOwner}[]`. Each page adapts
its own row shape at the boundary (`games.html` uses `e.Type`, `analytics.html`
uses `e.type`, `dashboard.html` uses raw arrays), so the module stays clean and
none of them grows a copy of the formula.

**Untagged attempts are excluded from rates**, and `coverage` is reported
alongside every table so the numbers can be trusted proportionally.

Precise definitions, so no surface has to guess:

- `coverage.total` counts **attack-ending events only** (those where
  `TR.isAttackEnd` is true) — not every event in the input.
- `coverage.tagged` counts those with a non-empty move.
- `moves` contains only moves with at least one attempt; a move that was never
  run does not appear as a zero row.
- `'Other'` and `'Interception'` are **real move values**, since the list is
  exactly `TR.MENU['Try']`. A Try named `Other` counts as tagged with the move
  `Other`, and gets its own row. They are not treated as untagged.
- `rate` is `tries / attempts` as a 0–1 number; formatting is each surface's job.

## Capture

Optional in all three tools. Skipping always leaves the move blank.

### `annotator.html`

After a Turnover / Pen Attack sub-type is picked, `renderSubmenu` renders a
second row:

```
Move (optional): [1 Scoop] [2 21] [3 32] … [x Skip]
```

Hotkeys `1`–`9` cover the first nine — the same limit the existing 19-item Try
picker already lives with. `Escape` or `Enter` skips. It slots in as a
`pendingStrikeMove` **before** the existing team-selection path, so
`needsTeamSelection` and possession inference are untouched. Not offered when
the sub-type is `6 Again`.

For correcting a tag afterwards, an `ann-move` cell in the annotation row gets
the same inline-`<select>` swap that `editAnnotationName` (annotator.html:936)
already uses.

### `annotator_field.html`

A second `<select>` under the existing sub-type one in the edit sheet,
defaulting to `— none —`, applying immediately on change the way
`applySubtype` (annotator_field.html:1465) does. The native phone wheel handles
19 items comfortably. Hidden when the sub-type is `6 Again`.

### `annotator_field2.html`

Picking from the sub-type chip strip re-arms the strip as a *move* strip on the
same status line (`Move — (Scoop)(21)(32)…`), reusing the existing 10s timeout
and `closeSubtype`. Plus the same `<select>` in the edit sheet, so a missed
strip is always recoverable. Not offered after `6 Again`.

### `viewer.html`

`'Strike Move'` joins `FILTERABLE` (viewer.html:585), so tagging can be
eyeballed before any rate is trusted.

## Analytics

### `dashboard.html` — headline home

A new collapsible "Strike Moves" section using the existing pattern at
dashboard.html:520:

- Two highlight tiles: **Most tries** (volume) and **Best try rate**
  (efficiency, `>= MIN_MOVE_ATTEMPTS`).
- Sortable table: Move / Tries / Fails / Attempts / Rate, with bars.
- Coverage line.

Aggregation slots into the existing per-game reducer: a
`teamMoves = {[t1]:{}, [t2]:{}}` accumulator alongside `teamTryTypes`
(dashboard.html:422), keyed on Action Owner, summed in `aggregateTeams`
exactly as `tryTypes` already is.

The per-team Team Detail card **keeps its top-3-by-tries ranking** and appends
the rate as a second figure. Re-ranking it by rate was considered and rejected:
with per-team splits most teams will not clear 2 attempts on a move, and the
card would go empty. Appending is purely additive — nothing regresses.

### `games.html` — per-game

The same table scoped to one game, split by team, near the existing penalty
breakdown. Counts prominent, rate secondary, coverage line always present —
single-game samples are noisy and the layout should say so rather than imply
precision.

### `analytics.html` — filter dimension

`strikeMove` joins the event normalisation, gains a "Strike Move" breakdown in
`renderFilteredTypeChart`'s machinery, and becomes a filter chip so it crosses
with team, time and v2's pitch positions like any other field.

### Field annotators — live Stats

A "Strike Moves" block in the Stats overlay of both v1 and v2, fed from the
local `annotations` array through the same module, riding the existing 5s
`refreshStats` interval. In-game samples are tiny, so this shows counts with
the rate greyed until 2 attempts.

## Phasing

**Phase 1 — capture.** `events.js` constants, `Code.gs` `HEADERS`, push/load in
all three annotators, the three pickers, the viewer filter.

**Phase 2 — analytics.** `js/strike_moves.js` plus tests, then the four surfaces.

Split this way because **analytics on zero data cannot be validated**. Phase 1
needs to be in use for at least a tournament before any rate means anything.
Phase 1 also carries all the risk (schema change + redeploy); Phase 2 is purely
additive.

### Sequencing hazard

The client will push 8-column rows, but new tabs get their header row from
`newSheet.appendRow(HEADERS)` (Code.gs:597). If the client ships before the
Apps Script redeploy, `setValues` writes an 8th column of data under a
7-column header — data present, header missing, and `action=all` silently drops
it.

**Redeploy the Apps Script first, then ship the client.**

### Open item for the user to verify

`Code.gs:13` states the column order "must match what the Python pipeline
reads". The new column is appended last, so any positional read of columns 0–6
is unaffected — but the pipeline lives in the gitignored `experiments/` and
could not be verified from the repo. Worth a check before Phase 1 ships.

## Testing

In `test.js`, matching the existing pure-`TR.*` style:

- `TR.strikeMoveOf` — Try returns its own Name; `6 Again` returns `''`;
  Penalty Defence returns `''`; an untagged Turnover returns `''`.
- `TR.isAttackEnd` — true for Try / Pen Attack / Turnover-not-`6 Again`;
  false for `6 Again`, Penalty Defence, Game Event, To Review.
- `TR.strikeMoveStats` — rate maths; coverage percentage; the 2-attempt
  threshold applying to `topByRate` but **not** `topByTries`; empty input;
  ties.

Plus:

- A round-trip assertion that a pushed row has 8 columns and loads back with
  its move intact.
- A manual check that a pre-change tab loads with blank moves and 0% coverage
  without breaking.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Move list | Exactly `TR.MENU['Try']`, all 19 | One constant, zero divergence, rates compare like-for-like with Try tags |
| Storage | New `Strike Move` sheet column | First-class and queryable; header-name mapping makes it backward compatible without backfill |
| Untagged attempts | Excluded, with coverage shown | Honest, and degrades gracefully as tagging improves |
| Top scoring move | Both volume and rate, side by side | Volume rewards what is run; rate rewards what works |
| Rate threshold | `MIN_MOVE_ATTEMPTS = 2` | Stops a 1-for-1 move showing 100% and topping the board |
| Capture UX | Optional second stage, per-tool native | Skipping costs nothing; each tool keeps its own idiom |
| `6 Again` / Pen Defence | Excluded from attempts entirely | Possession does not change, so the move is not over |
