# Filter sheet: two-step drill-down

Date: 2026-09-13

## Problem

The Event Viewer's Filter sheet (`game.html`, `filterSheet()`) renders every
category expanded at once: a search box, then Competition, Division, Year, Type,
Team, Name and Move, each with all of its options as chips. Across all games
that is a long scroll on a phone, and the categories that matter get buried
under the ones that do not. The Name group already concedes the point by
truncating to `.slice(0,30)`, which silently hides options rather than making
them reachable.

## Goal

Replace the single long scroll with two steps: a list of categories showing what
is currently picked, and a per-category screen showing that category's options.
Drop the Name cap, since the drill-down is what makes a long list affordable.

## Design

### 1. State

One new module-level variable:

```js
let fCat = null;   // null = category list; otherwise a filter key ('type', 'team', …)
```

`openSheet('filter')` resets `fCat = null`, so the sheet always opens at the
category list. Closing and reopening never strands the user inside a category,
and the `#filter` hash restore needs no drill state of its own.

`filterSheet()` becomes a two-way switch returning either `catList()` or
`catOpts(fCat)`.

### 2. Category metadata

A single ordered table replaces the repeated `grp(...)` calls, so the two
renderers agree on titles, keys and option sources without duplicating the list:

```js
const FCATS = [
  {key:'comp', title:'Competition', of:e=>e.comp},
  {key:'div',  title:'Division',    of:e=>e.div},
  {key:'year', title:'Year',        of:e=>e.year},
  {key:'type', title:'Type',        of:e=>e.type},
  {key:'team', title:'Team',        of:e=>e.team},
  {key:'name', title:'Name',        of:e=>e.name},
  {key:'move', title:'Move',        of:e=>e.move},
];
```

Order is unchanged from today: the cross-game narrowers first, then the
per-event groups.

### 3. Navigation

The back affordance lives in the **sheet header**, not the body. `#sheetTitle`
becomes `‹ Team` and is itself the tappable back control; **Done** stays where
it already is on the right. A pinned header target survives a scrolling option
list, and no body row competes with the options for the first screenful.

`openSheet` currently sets the title from a `kind` expression. That moves into a
small `paintFilterHead()` so the title can change on drill without reopening the
sheet — the same shape `paintWin()` already uses for the window sheet.

At the category list the title reads `Filter events` and is inert.

### 4. Category list — `catList()`

A row per category: title on the left, selection summary on the right, chevron
at the end. Reuses the `.glist` button styling from the game sheet, which is
already the page's drill-in idiom.

```
Competition   Any                    ›
Type          Try, Turnover          ›
Team          Any                    ›
Name          3 selected             ›
```

The summary comes from a pure helper so it can be tested:

```js
TR.filterSummary(values)   // [] → 'Any'; 1–2 values → joined with ', '; more → 'N selected'
```

The existing "a group with fewer than 2 options cannot narrow anything" rule
carries over to rows, including its exception: a category that is actively
filtered keeps its row, so a live filter never becomes invisible and
unremovable.

The search box (`F.q`) stays pinned above the rows — it is a cross-cutting
free-text filter, not a category — and **Clear all filters** stays pinned below
them.

### 5. Option screen — `catOpts(key)`

The same `.opts` chip grid as today, with two changes:

- **No Name cap.** `uniq(e=>e.name)` is no longer sliced.
- **Type-to-narrow.** When a category offers more than 12 options, a `.srch`
  input appears above the chips. It filters which chips are drawn and nothing
  else — it never touches `F`, and its value is discarded on leaving the
  category.

Narrowing uses a pure, case-insensitive substring predicate:

```js
TR.optMatch(value, needle)   // true when needle is empty or value contains it, case-insensitively
```

Chips keep their current behaviour: `tog()` updates `F`, flips the chip class
and re-renders the event list live behind the sheet. Multi-select within a
category is unchanged. The user leaves via back or Done.

### 6. Unchanged

`render()`, the shape of `F`, the removable active-filter chips in the bar above
the list, the quick-access type chips shown when nothing is filtered, and the
`#filter` hash restore all stay as they are. `clearF()` keeps its semantics and
additionally returns to the category list.

## Testing

Both helpers live in `js/utils.js` — already loaded by `game.html` and already
on the Node harness's list in `test.js`, so neither file needs a new entry.
`test.js` covers:

- `TR.filterSummary` — none, one, two, and many values.
- `TR.optMatch` — empty needle, case-insensitive hit, miss.

Drill navigation, the header back control and the live re-render behind the
sheet are verified by hand in the browser.

## Out of scope

- The Event Editor's (`viewer.html`) own filter row. It is a desktop-width
  table toolbar with a different layout problem, and nothing here forces a
  change to it.
- Per-category "select all" / "invert" controls. No demand for them yet.
