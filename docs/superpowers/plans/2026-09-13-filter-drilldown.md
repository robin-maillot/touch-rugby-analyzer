# Filter Drill-Down Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Event Viewer's single long Filter sheet with two steps — a category list showing what's picked, and a per-category option screen reached by tapping a row.

**Architecture:** `game.html` is a single self-contained page; its filter UI is three functions near the bottom of one inline `<script>`. A new `fCat` variable decides whether `filterSheet()` renders the category list or one category's options, and an `FCATS` table gives both renderers one source of titles and option accessors. Two pure string helpers move to `js/utils.js` so the Node harness in `test.js` can exercise them.

**Tech Stack:** Vanilla ES2015+ browser JS, no build step, no framework. Tests run with `node test.js` — a `vm`-based harness with no npm dependencies.

## Global Constraints

- No build step, no npm dependencies, no framework. Plain `<script>` tags and `TR.*` globals.
- `js/utils.js` depends only on `js/config.js` (the `TR` namespace) and must stay free of DOM references — the Node harness loads it with no `document`.
- All user-supplied strings rendered into HTML go through the page's `esc()` helper; values interpolated into inline `onclick` handlers additionally have `'` escaped as `\'`, matching the existing code.
- Bump `CACHE_VERSION` in `sw.js` whenever `game.html` changes, or returning users keep the cached shell.
- Attribution: end every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

### Task 1: Pure helpers in `js/utils.js`

**Files:**
- Modify: `js/utils.js` (append at end)
- Test: `test.js` (append a new section before the final summary block)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TR.filterSummary(values)` → `string`. `values` is an array (or any iterable spread into one) of already-`String`-able option values. Returns `'Any'` for empty, the values joined with `', '` for 1–2 values, and `` `${n} selected` `` for 3 or more.
  - `TR.optMatch(value, needle)` → `boolean`. Case-insensitive substring test. Returns `true` when `needle` is empty, null or undefined. Handles non-string `value` (e.g. a numeric year) by coercing with `String()`.

- [ ] **Step 1: Write the failing tests**

Append to `test.js`, immediately before the closing summary block (the lines that print `passed`/`failed` and call `process.exit`):

```js
// ── TR.filterSummary ──────────────────────────────────────────
console.log('TR.filterSummary');
test('none',        () => { assert.equal(TR.filterSummary([]), 'Any'); });
test('one',         () => { assert.equal(TR.filterSummary(['Try']), 'Try'); });
test('two',         () => { assert.equal(TR.filterSummary(['Try', 'Turnover']), 'Try, Turnover'); });
test('three+',      () => { assert.equal(TR.filterSummary(['Try', 'Turnover', 'Penalty Attack']), '3 selected'); });
test('from a Set',  () => { assert.equal(TR.filterSummary([...new Set(['Try'])]), 'Try'); });
test('numbers',     () => { assert.equal(TR.filterSummary([2025, 2026]), '2025, 2026'); });

// ── TR.optMatch ───────────────────────────────────────────────
console.log('TR.optMatch');
test('empty needle', () => { assert.equal(TR.optMatch('Wiggle', ''), true); assert.equal(TR.optMatch('Wiggle', null), true); assert.equal(TR.optMatch('Wiggle', undefined), true); });
test('hit',          () => { assert.equal(TR.optMatch('Wiggle', 'wig'), true); });
test('case',         () => { assert.equal(TR.optMatch('wiggle', 'WIG'), true); });
test('mid-string',   () => { assert.equal(TR.optMatch('Dummy Switch', 'switch'), true); });
test('miss',         () => { assert.equal(TR.optMatch('Wiggle', 'zzz'), false); });
test('non-string',   () => { assert.equal(TR.optMatch(2026, '26'), true); assert.equal(TR.optMatch(null, 'x'), false); });
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node test.js`

Expected: FAIL — twelve new `✗` lines reading `TR.filterSummary is not a function` and `TR.optMatch is not a function`, and a non-zero exit code.

- [ ] **Step 3: Write the implementation**

Append to `js/utils.js`:

```js
// Describes a filter category's current selection for the Filter sheet's
// category list. Two values still fit a row; beyond that a count reads better
// than a truncated list.
TR.filterSummary = (values) => {
  const v = Array.from(values || []);
  if (!v.length) return 'Any';
  return v.length <= 2 ? v.map(String).join(', ') : `${v.length} selected`;
};

// Case-insensitive substring test for the Filter sheet's type-to-narrow box.
// An empty needle matches everything, so an untouched box hides nothing.
TR.optMatch = (value, needle) => {
  if (!needle) return true;
  return String(value == null ? '' : value).toLowerCase().includes(String(needle).toLowerCase());
};
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node test.js`

Expected: PASS — the twelve new lines show `✓`, and the run ends with `0 failed` and exit code 0.

- [ ] **Step 5: Commit**

```bash
git add js/utils.js test.js
git commit -m "$(cat <<'EOF'
feat(utils): filterSummary and optMatch for the filter sheet

Two pure string helpers the two-step Filter sheet needs: one describes a
category's current selection for its row, the other backs the type-to-narrow
box on long option lists.

They live in js/utils.js rather than inline in game.html so test.js can reach
them -- the Node harness loads js/*.js and cannot see page script.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: The `FCATS` table and the category list

**Files:**
- Modify: `game.html` — the `filterSheet()` / `tog()` / `clearF()` block at the end of the inline `<script>` (currently ~lines 931–957), plus `openSheet()` (~line 901) and the `.glist` CSS block (~line 213).

**Interfaces:**
- Consumes: `TR.filterSummary` from Task 1.
- Produces, for Task 3:
  - `FCATS` — module-level `const`, an array of `{key, title, of}` where `of` is an event accessor.
  - `fCat` — module-level `let`, `null` at the category list or an `FCATS` key when drilled in.
  - `catVals(c)` — returns the sorted-by-first-appearance unique truthy values of category `c` across `events`.
  - `catList()` — returns the category-list HTML.
  - `paintFilterHead()` — sets `#sheetTitle`'s text and back affordance from `fCat`.
  - `drill(key)` — sets `fCat` and repaints head + body.

This task leaves `catOpts()` unwritten; `drill()` is added in Task 3. Ending here would give a category list whose rows do nothing, so **Tasks 2 and 3 land as one commit** — Task 2's step 5 is a checkpoint, not a commit.

- [ ] **Step 1: Add the category-row CSS**

In `game.html`, immediately after the `.glist .gs` rule (~line 217), add:

```css
  /* Filter sheet category rows: a .glist button with the selection summary
     pulled onto the same line as the title, and a chevron at the end. The
     selector carries `.glist button` because the block rule above it is a
     class-plus-element and would otherwise win over a bare `.fcat`. */
  .glist button.fcat{display:flex;align-items:center;gap:8px}
  .fcat .fc-t{flex:1;min-width:0}
  .fcat .fc-v{color:var(--dim);font-size:.72rem;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;max-width:52%}
  .fcat .fc-v.set{color:var(--accent)}
  .fcat .fc-x{color:var(--dim);font-size:.8rem;flex-shrink:0}
```

- [ ] **Step 2: Replace `filterSheet()` with the table plus the two renderers' entry point**

Replace the whole `filterSheet()` function (the comment block starting `// Groups ordered by how much they narrow:` through the closing `}` of `filterSheet`) with:

```js
// Categories ordered by how much they narrow: Competition/Division/Year first
// (the ones that matter most once you're browsing across every game), then the
// original per-event groups. One table so the category list and the option
// screen can't drift apart.
const FCATS=[
  {key:'comp', title:'Competition', of:e=>e.comp},
  {key:'div',  title:'Division',    of:e=>e.div},
  {key:'year', title:'Year',        of:e=>e.year},
  {key:'type', title:'Type',        of:e=>e.type},
  {key:'team', title:'Team',        of:e=>e.team},
  {key:'name', title:'Name',        of:e=>e.name},
  {key:'move', title:'Move',        of:e=>e.move},
];
// null at the category list, otherwise the FCATS key being browsed. Reset on
// every open, so the sheet never reopens stranded inside a category.
let fCat=null, fNarrow='';

function catVals(c){return [...new Set(events.map(c.of).filter(Boolean))]}

// A category with one option can't narrow anything — every event already shares
// that value — so it's noise. Kept if it's actively filtered, so a live filter
// never becomes invisible and unremovable.
function catShown(c){return catVals(c).length>1||F[c.key].size>0}

function filterSheet(){
  if(!picked)return '<div class="empty">Pick a game first.</div>';
  return fCat?catOpts(fCat):catList();
}

function catList(){
  return `<input class="srch" id="q" placeholder="Search name, comment, move…" value="${esc(F.q)}"
      oninput="F.q=this.value;render()">`+
    '<div class="glist">'+FCATS.filter(catShown).map(c=>{
      const set=F[c.key].size>0;
      return `<button class="fcat${set?' on':''}" onclick="drill('${c.key}')">
        <span class="fc-t">${c.title}</span>
        <span class="fc-v${set?' set':''}">${esc(TR.filterSummary([...F[c.key]]))}</span>
        <span class="fc-x">›</span>
      </button>`}).join('')+'</div>'+
    `<button class="icon-btn" style="width:100%;height:40px" onclick="clearF()">Clear all filters</button>`;
}
```

- [ ] **Step 3: Add the header painter and the drill control**

Immediately after `catList()`, add:

```js
// The title doubles as the back control when drilled in. Painted rather than
// re-rendered so drilling doesn't have to reopen the sheet.
function paintFilterHead(){
  const c=FCATS.find(x=>x.key===fCat), h=$('sheetTitle');
  h.innerHTML=c?`‹ ${c.title}`:'Filter events';
  h.className=c?'back':'';
  h.onclick=c?()=>drill(null):null;
}

function drill(key){
  fCat=key; fNarrow='';
  $('sheetBody').innerHTML=filterSheet();
  paintFilterHead();
}
```

And add the `.back` style beside the other `.sheet-head` rules (~line 205):

```css
  .sheet-head h2.back{cursor:pointer;color:var(--accent)}
```

- [ ] **Step 4: Reset drill state and paint the head on open**

In `openSheet()`, replace the `$('sheetTitle').textContent = ...` assignment and the body assignment with:

```js
function openSheet(kind){
  $('scrim').classList.add('open'); $('sheet').classList.add('open');
  if(kind==='filter')fCat=null;               // always open at the category list
  $('sheetTitle').className=''; $('sheetTitle').onclick=null;
  $('sheetTitle').textContent = kind==='game'?'Choose a game':kind==='moves'?'Strike moves'
    :kind==='window'?'Playback window':'Filter events';
  $('sheetBody').innerHTML = kind==='game'?gameSheet():kind==='moves'?movesSheetBody()
    :kind==='window'?windowSheet():filterSheet();
  // The window sheet is painted rather than rendered with its state inlined, so
  // that a keystroke can update the chips without re-rendering the field.
  if(kind==='window')paintWin();
}
```

- [ ] **Step 5: Checkpoint — do not commit yet**

Rows render but `catOpts()` does not exist, so tapping one throws. Continue straight to Task 3.

---

### Task 3: The option screen, and rewiring `tog()` / `clearF()`

**Files:**
- Modify: `game.html` — same block as Task 2, plus `sw.js`.
- Test: `node test.js` (regression only — this task is DOM code with no new pure logic).

**Interfaces:**
- Consumes: `FCATS`, `fCat`, `fNarrow`, `catVals`, `drill`, `paintFilterHead` from Task 2; `TR.optMatch` from Task 1.
- Produces: `catOpts(key)` returning one category's option-screen HTML; `optChips(key, vals)` rendering the chip markup shared by the first paint and every keystroke; `narrow(v)` updating the type-to-narrow box.

- [ ] **Step 1: Add `catOpts()`, `optChips()` and `narrow()`**

In `game.html`, immediately after `drill()`, add:

```js
// Twelve chips is about two phone rows — past that, scanning for one costs
// more than typing three letters of it.
const NARROW_AT=12;

function catOpts(key){
  const c=FCATS.find(x=>x.key===key);
  if(!c)return '<div class="empty">No such filter.</div>';
  const all=catVals(c);
  const vals=all.filter(v=>TR.optMatch(v,fNarrow));
  const box=all.length>NARROW_AT
    ? `<input class="srch" id="narrow" placeholder="Narrow ${c.title.toLowerCase()}…"
        value="${esc(fNarrow)}" oninput="narrow(this.value)">` : '';
  return box+`<div class="grp"><div class="opts" id="optWrap">${optChips(key,vals)}</div></div>`;
}

// Shared by the first paint and every keystroke, so the two can't drift.
function optChips(key,vals){
  if(!vals.length)return `<div class="empty">Nothing matches “${esc(fNarrow)}”.</div>`;
  return vals.map(v=>`<button class="chip ${F[key].has(v)?'on':''}"
    onclick="tog('${key}','${String(v).replace(/'/g,"\\'")}',this)">${esc(v)}</button>`).join('');
}

// Redraws only the chips: re-rendering the body would blow away focus and the
// caret mid-keystroke, the same reason paintWin exists.
function narrow(v){
  fNarrow=v;
  const c=FCATS.find(x=>x.key===fCat);
  $('optWrap').innerHTML=optChips(fCat,catVals(c).filter(x=>TR.optMatch(x,fNarrow)));
}
```

- [ ] **Step 2: Point `clearF()` back at the category list**

Replace `clearF()` with:

```js
function clearF(){F={type:new Set(),team:new Set(),name:new Set(),move:new Set(),
    comp:new Set(),div:new Set(),year:new Set(),q:''};
  drill(null); render()}
```

`tog()` is unchanged — it already toggles `F`, flips the chip class and calls `render()`.

- [ ] **Step 3: Bump the service-worker cache**

In `sw.js` line 8, change `trl-shell-v19` to `trl-shell-v20`.

- [ ] **Step 4: Run the regression tests**

Run: `node test.js`

Expected: PASS — `0 failed`, exit code 0. Nothing in this task touches `js/*.js`, so this only confirms Task 1 is still green.

- [ ] **Step 5: Verify in the browser**

Open `game.html` with a dev server and check, in order:

1. Pick a game. Tap **Filter** — the sheet shows the search box, then one row per category reading `Any`, then **Clear all filters**.
2. Tap **Type** — the header reads `‹ Type` in the accent colour; the body shows Type's chips.
3. Tap `Try` — the chip lights, and the event list behind the sheet re-filters immediately.
4. Tap `‹ Type` — back at the category list, with the Type row now reading `Try` in the accent colour.
5. Tap **Done**, then **Filter** again — it reopens at the category list, not inside Type.
6. Drill into **Name** with a game that has more than 12 distinct names — a "Narrow name…" box appears. Type three letters: chips filter live, the caret stays in the box, and a no-match query shows `Nothing matches "…"`.
7. Confirm a single-option category has no row, and that the removable chips in the bar above the list still clear filters.

- [ ] **Step 6: Commit**

```bash
git add game.html sw.js
git commit -m "$(cat <<'EOF'
feat: the filter sheet becomes two steps

Seven categories rendered fully expanded is a long scroll on a phone, and Name
already conceded the point by truncating to 30 options. The sheet now opens on a
row per category showing what's picked, and tapping one drills into that
category's chips.

Back lives in the sheet header rather than the body, so it survives a scrolling
option list and doesn't cost the first screenful of options. Drill state resets
on open, which leaves the #filter hash restore untouched.

The Name cap is gone: categories past twelve options get a type-to-narrow box
instead, redrawing only the chips so the caret survives a keystroke. The
one-option-category rule and its actively-filtered exception carry over from the
old groups to the new rows.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Update the README

**Files:**
- Modify: `README.md` — the Event Viewer section (~line 93).

**Interfaces:**
- Consumes: the shipped behaviour from Task 3.
- Produces: nothing.

- [ ] **Step 1: Rewrite the filtering sentence**

In the **Event Viewer (`game.html`)** section, replace the first paragraph's `filter the event list` clause so the paragraph reads:

```markdown
The main way to watch events. Pick a game (or all games), filter the event list, and tap any row to play it — the video follows the event, loading a different game's footage when the tap calls for it. Prev/next step through the *filtered* list, so after filtering to tries "next" is the next try.

The **Filter** sheet works in two steps: it opens on a row per category — Competition, Division, Year, Type, Team, Name, Move — each showing what's currently picked, and tapping one drills into that category's options. Categories offering more than twelve options get a type-to-narrow box. A category every event already agrees on is hidden unless it's actively filtered. Free-text search sits above the rows and spans name, comment and move; picked filters also appear as removable chips above the event list.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: the Event Viewer's filters are a two-step sheet

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```
