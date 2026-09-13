# Saved playlists of events

Date: 2026-09-13

## Problem

The Event Viewer (`game.html`) can reach any event, but only through a filter.
A coach preparing a session wants a *set* — eleven backdoors from four different
matches, in the order they want to show them — and there is no way to express
that. A filter cannot, by construction: the set spans several different filters,
and its order is the coach's, not the clock's.

Today the closest thing is re-deriving the filter each time and scrolling, which
loses the order and cannot be resumed on another device.

## Goal

A **playlist**: a named, ordered list of hand-picked events, saved per account
in the sheet, able to span games and filters freely, and playable in the Event
Viewer with the transport that already exists.

Three mockups of the interaction were compared first — see
[`mockups/playlists/`](../../../mockups/playlists/index.html). This spec builds
option B (collect mode) with option C's reorder screen.

## Design

### 1. Storage — the `_playlists` tab

A new tab in `CONTROL_SHEET_ID`, alongside `_groups` and `_metadata`:

```
Id | Owner | Name | Note | Refs | Updated At
```

- **Id** — a generated key (`Utilities.getUuid()`), never reused.
- **Owner** — the caller's secret. `_groups` already holds every secret in
  plaintext in the same spreadsheet, so this is the existing security posture,
  not a widening of it.
- **Refs** — the ordered event refs, newline-separated, each in full
  `game#time#type#name` form. Full rather than minimal so a row stays readable
  to a human debugging it; see §2 for why only part of it is matched on.
- **Updated At** — an ISO timestamp, for a "last edited" line and for ordering
  the sheet by hand.

Every role owns playlists, `viewer` included. A viewer is exactly the person
building a teaching set, and playlists grant no access to anything — a ref only
resolves against events the caller could already see.

**Cap: 500 refs per playlist**, enforced client-side with a plain message. A ref
runs about 60 characters, so 500 sits well inside the ~50,000-character Sheets
cell limit with room for long game names.

`_playlists` is **not** added to `ADMIN_SHEETS`. The admin sheet editor exists
for control sheets an admin needs to repair by hand; a playlist is user content,
and exposing it there would put every account's lists in one editable grid for
no benefit.

### 2. Resolution — refs are matched loosely

A ref is stored in full but **resolved on `game` + `time` only**. Type and name
are advisory.

This is the answer to the one real hazard in the feature: event refs are
synthetic, not stored IDs. `evId()` builds them from the row's own values, so an
admin renaming an event in the Event Editor changes its ref and would otherwise
break every playlist holding it. Matching on game + time survives the rename.
The pair is unique in practice — the annotator stamps one event per timestamp
per game — and where it somehow is not, the first match wins deterministically
because `rows` is in sheet order.

Anything that still cannot resolve (the game was deleted, the event removed) is
**not dropped silently**. `resolve()` reports the count, and the list header
carries a single line:

```
2 events are no longer in the data
```

rendered in the filter bar (`#filterbar`) beside the playlist chip, where the
event count already sits. The remaining events play normally. A broken playlist
explains itself rather than quietly getting shorter.

Adding real stored IDs to every game tab would be more correct and was
considered; it is a migration across every sheet plus an annotator change, a
larger project than this feature, and nothing here forecloses it.

### 3. API — three new actions in `Code.gs`

| Action | Method | Payload | Returns |
|---|---|---|---|
| `playlists` | GET | `secret` | `{ok, playlists:[{id,name,note,refs,updated}]}` |
| `save_playlist` | POST | `{secret, id?, name, note, refs}` | `{ok, id}` |
| `delete_playlist` | POST | `{secret, id}` | `{ok}` |

`action=playlists` returns only rows whose `Owner` equals the caller's secret.

Writes are guarded by a small helper mirroring the shape of `canEditGame`:

```js
function ownsPlaylist(secret, id) {
  // → the sheet row index for a playlist this secret owns, or -1
}
```

`save_playlist` with no `id` creates; with an `id` it replaces that row only if
`ownsPlaylist` finds it. A miss returns `{ok:false, error:'Playlist not found.'}`
rather than silently creating a second row.

**These three actions bypass `cacheGet` / `cachePut`.** The `_playlists` tab is
small, it changes on every edit, and the `action=version` fast-path that lets
clients skip `action=all` must never let a client skip a playlist change. Each
call reads the tab directly.

`bumpVersion()` is **not** called on a playlist write — a playlist changes no
game data, and bumping would force every client to refetch the heavy
`action=all` payload for nothing.

### 4. Client module — `js/playlists.js`

New file, loaded by `game.html`. The network calls are thin; the logic worth
testing is pure.

```js
TR.playlists.load(secret)              // → Promise<[playlist]>
TR.playlists.save(secret, pl)          // → Promise<{id}>
TR.playlists.remove(secret, id)        // → Promise<void>

TR.playlists.resolve(refs, rows)       // → {events:[…], missing:N}
TR.playlists.reorder(refs, from, to)   // → a new array
```

`resolve` maps each ref to an event by `game#time`, preserving the playlist's
order (not the clock's), and counts what it could not place. `reorder` moves one
entry and returns a new array, clamping out-of-range indices to a no-op.

### 5. One targeted tidy — `evId` moves to `utils.js`

`evId()` currently lives inside `game.html`, around line 572. The playlist
module needs the same ref format, and a second copy would drift the
first time either changes.

It moves to `js/utils.js` as `TR.evId(e)`, with the loose key beside it:

```js
TR.evId(e)    // game#time#type#name — the stored ref
TR.evKey(e)   // game#time — what resolution matches on
```

`game.html` calls `TR.evId` at its three existing call sites. `js/utils.js` is
already loaded by `game.html` and already on the Node harness's list in
`test.js`, so no new wiring. This is the only refactor in the change.

### 6. Collect mode

A ☑ toggle in `.listhead`, beside the existing collapse button, sets
`body.collect` on the page.

**State:**

```js
let collecting = false;
const sel = new Set();   // TR.evId refs, never list indices
```

`sel` holds full `TR.evId` refs — the same form written to the sheet, so
committing a collection is a copy rather than a conversion.

Refs rather than indices is deliberate. An index is a position in
`shown`, and `shown` is rebuilt by every filter change — so an index-based
selection would silently repoint at different events the moment the user
filtered, which is precisely the motion the feature exists to support.

**Behaviour:**

- Rows tick instead of playing. `.ev.sel` gets the accent treatment; the
  timestamp column is replaced by the tick, so the row gains nothing in width
  and loses nothing when not collecting.
- A tray above the list holds the running count. It stays put while the list
  changes underneath it, which is what makes a cross-filter collection legible:
  tick three tries, change the filter to turnovers, tick two more, save once.
- A pinned action bar: `N selected` · `All N` (everything currently in `shown`)
  · `Cancel` · `Add N to…`.
- `Add N to…` opens the sheet as an `addto` kind — the playlists to add to, plus
  **＋ New playlist**. Choosing one fires a single `save_playlist`.
- **Entering cinema exits collect**, or the pinned action bar floats over a
  full-screen video with no list behind it.
- **Collect is unavailable while a playlist is playing** — the ☑ toggle is
  hidden. A playing playlist is not a pool to collect from, and editing one has
  a home already: the reorder screen in §7. The toggle returns as soon as the
  playlist chip's ✕ is tapped.

### 7. Playlists sheet and the reorder screen

A fourth `openSheet` kind, `'playlists'`, beside `game` / `filter` / `moves`.

The sheet is a two-level drill, reusing the mechanism the filter sheet already
has (`fCat` / `catList()` / `catOpts()` / `paintFilterHead()`, landed in
8734e08) rather than inventing a second one: a `plEdit` state variable (`null` =
the list, otherwise a playlist id), a `filterSheet`-style switch between the two
renderers, and the sheet header's title as the back control. `plEdit` resets on
`openSheet('playlists')` for the same reason `fCat` does — reopening the sheet
must never strand the user inside a playlist.

**Level 1 — the list.** A row per playlist: name, note, event count, chevron.
Tapping the row plays it. A ⠿ at the row's end drills in. Below them, **＋ New
playlist**.

**Level 2 — reorder.** Name field, then a row per event with a ⠿ handle and ↑↓
buttons, an ✕ to remove, and **Delete playlist** at the foot. Changes are
batched and written with one `save_playlist` on leaving the screen, so dragging
five rows is one request rather than five.

↑↓ buttons ship alongside the handle rather than drag alone: a 52px row on a
phone inside a scrolling sheet is a poor drag target, and the buttons are also
what makes the reordering testable without synthesising pointer events.

### 8. Playing a playlist

Selecting a playlist sets `plOpen` to its id and rebuilds `shown` from
`resolve()` rather than from the filter predicate. Everything downstream is
untouched — prev/next, loop, the scrub marks and the playback window all read
`shown` already, so they follow the playlist with no change.

A `.chip.pl` in the filter bar names the playing playlist and carries an ✕ to
leave it.

**Filters are suppressed while a playlist plays.** A playlist is already a
hand-picked set; leaving the filter chips live would let them narrow it while
claiming to describe the whole list. The filter bar shows the playlist chip and
the count, nothing else, until the ✕ returns to the normal filtered view.

`pickGame()` already resets `F` and `activeId`; it additionally clears `plOpen`
and exits collect mode, so picking a game from the picker always lands in the
ordinary filtered view. Deleting the playlist that is currently playing clears
`plOpen` the same way, rather than leaving the page playing a list that no
longer exists.

`#playlists` joins `#cinema` / `#filter` / `#moves` in `applyHash()`, so the
sheet is linkable and survives a reload like the others.

## Testing

`test.js` — pure helpers, no browser:

- `TR.evId` / `TR.evKey` — the two formats, and that `evKey` ignores a renamed
  type and name.
- `TR.playlists.resolve` — every ref resolves; some refs missing (count
  reported, survivors kept); a ref whose stored type and name are stale still
  resolves on game + time; playlist order is preserved rather than clock order.
- `TR.playlists.reorder` — move up, move down, both ends, out-of-range no-op.

`sw.js` — the service worker's cached asset list gains `js/playlists.js`, and
its cache version is bumped, as 8734e08 did for the filter change.

By hand in the browser: the collect toggle and its action bar, a collection held
across a filter change, the two-level playlists sheet and its back control, the
reorder screen's single save on leaving, the missing-events line, and cinema
exiting collect.

The three actions in `Code.gs` are verified against the deployed script with a
throwaway playlist: create, list, edit, delete, and a `save_playlist` for an id
owned by another secret (must be refused).

## Out of scope

- **Sharing.** A playlist belongs to one account. Group-visible playlists — a
  coach's set reaching the squad — are a real feature and deliberately deferred;
  a `Visibility` column is additive when it is wanted.
- **Deep links.** `?playlist=<id>` would only resolve for the owner, so it is
  not a sharing mechanism and there is no other reason for it.
- **Smart playlists.** A saved *filter* that stays live is a different feature
  with a different data model, and it cannot span filters, which is the
  requirement here.
- **Clip export.** Handing an ordered playlist to the Cloud Run clipper is the
  obvious follow-on, but it needs GCS-backed source video, which not every game
  has.
- **The Event Editor** (`viewer.html`). It gets no playlist UI.
