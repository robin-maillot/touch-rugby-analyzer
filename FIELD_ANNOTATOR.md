# Field Annotator — Guide

The **Field Annotator** (`annotator_field.html`) is a phone-first tool for tagging a
touch rugby game **live from the sideline, with no video**. You tap what happens as it
happens; the page timestamps each event against a wall clock, infers possession and the
score, shows running stats, and pushes everything to the Google Sheet in the same format
as the video annotator — so the game shows up in Game Analysis, Dashboard, and Live just
like any other.

> Access: requires the **staff** or **admin** password (entered on `index.html`). The
> password is reused as the API secret and also decides which **access group** your
> uploads are tagged to.

It's a PWA — on iOS/Android you can "Add to Home Screen" and run it full-screen.

---

## The 30-second version

1. **⚙ Setup** → type the two team names (+ year / division / competition).
2. Set **possession** (which team has the ball), then tap **Start** when the whistle goes
   for kickoff.
3. Tap events as they happen: **Try**, **Turnover**, **6th Touch**, **Pen Attack**,
   **Pen Defence**, **6 Again**.
4. Possession and the scoreboard update themselves. Tap a **Recent** row to fix a mistake.
5. Tap **Start** again at half time (it becomes **End**), and again for the second half.
6. When the game is over, tap **End**, then **⬆ Push** to save it to the sheet.

Everything below is detail on top of that loop.

---

## Layout

```
┌────────────────────────────────────────────────┐
│ TR  Field Annotator   ←Back ⬆Push ⇆Swap 📊Stats │   header
│                              ⚫Live  ⚙Setup       │
├────────────────────────────────────────────────┤
│  France        00:00 (half)        England       │   scoreboard
│    3       –   12:34 total    –       2           │
├────────────────────────────────────────────────┤
│ [ France ]   [ Start ]   [ England ]             │   possession + clock toggle
│                                                  │
│  EVENTS   (only visible once the game is running)│
│  [ Try ] [ Turnover ] [ 6th Touch ]              │
│  [ Pen Atk ] [ Pen Def ] [ 6 Again ]             │
│                                                  │
│  GAME EVENTS                                     │
│  [ Drive + ]      [ Drive − ]                    │
│  [ Ball Live ] [ To Review ] [ ↩ Undo ]          │
│                                                  │
│  RECENT                                          │
│   12:30  Try · France            ›               │
│   11:58  Turnover · England      ›               │
└────────────────────────────────────────────────┘
```

The **Events** grid and the **Drive / Ball Live / To Review** buttons stay hidden until a
game is running (i.e. you've tapped **Start**). Before that, only the possession toggle,
**Start**, and **Undo** are shown — there's nothing meaningful to tag yet.

---

## Setup panel (⚙ Setup)

Open before kickoff and fill in:

| Field | Notes |
|---|---|
| **Team 1 / Team 2** | Free text. As soon as you type, the possession buttons and scoreboard relabel themselves. |
| **Year** | Defaults to the current year. |
| **Division** | Dropdown (MXO, MO, WO, M30, W27, M40, W35, or NONE). |
| **Competition** | Free text, e.g. "Seniors Cup". |
| **ID** | *Optional.* A disambiguator for repeat fixtures — when the same two teams in the same comp/year/division play more than once, set a different ID on each so they get distinct sheet tabs. |
| **Click delay (s)** | Subtracted from every tag **after the first**, to compensate for your reaction time. `0`–`9`. Set it to ~1–2 if you find yourself always tapping a beat late. Persists across sessions. |

A live **preview** shows the Google Sheet tab name that will be generated, e.g.
`2025_m30_seniors-cup_france_england`. If you set an ID it's suffixed
(`…_england_rematch`).

**Group line:** below the preview, the panel tells you which access group your push will
be tagged to (derived from your password). If it warns *"No group on your login"*, your
upload will be admin-only — re-login or check your access before you rely on it. Admins
upload ungrouped and set visibility later in Backfill.

---

## Tagging events

### 1. Set possession first
The **possession toggle** (left/right of the Start button) marks which team currently has
the ball. Tag this *before* the event so ownership is attributed correctly. After most
events possession flips automatically (see below), so in practice you mostly just correct
it when the auto-inference is wrong.

### 2. Start / End the game (and halves)
The centre **Start** button tags a `Game Start`. It then reads **End** — tap it at half
time to tag `Game End`. Tap **Start** again for the second half, and so on. The scoreboard
shows the current half (`1st Half`, `2nd Half`, or `Break`) plus two clocks:

- **Half clock** — time since the last `Game Start`.
- **Total clock** — time since the first `Game Start`.

A game is only marked **Analyzable** (and so appears in Game Analysis / Dashboard) if it
has **both** a Game Start and a Game End.

### 3. Event buttons

| Button | What it records | Possession after |
|---|---|---|
| **Try** | Try for the team in possession | **switches** |
| **6th Touch** | Possession completed cleanly — ball handed over on the 6th touch | **switches** |
| **Turnover** | Possession lost to an error (ball down, dummy, bad roll, in touch, intercepted…) | **switches** |
| **Pen Attack** | Penalty won by the attacking team | **stays** |
| **Pen Defence** | Penalty conceded by the attacking team (defence benefits) | **switches** |
| **6 Again** | Penalty restart — attacking team keeps the ball | **stays** |

#### 6th Touch vs Turnover — the key distinction

These two both end a possession and both flip the ball over, but they mean opposite things:

- **6th Touch** = the possession was **completed**. The team used all six touches *without
  a real handling error* and simply ran out of touches, so the ball changes hands. Use it
  for every clean set that ends naturally on the sixth touch.
- **Turnover** = the possession was **lost to an actual error** — ball down, dummy touch,
  bad roll, kicked into touch, intercepted, and so on. Use it whenever the set ended
  because someone made a mistake.

Getting this right matters because it drives the **Completions** stat (below). When in
doubt: *did they earn the handover by surviving all six touches (6th Touch), or did they
cough it up (Turnover)?*

Possession inference is the shared rule used everywhere in the app
(`js/possession.js`): it **switches** after Try, Pen Defence, and Turnover (except 6
Again), and **stays** otherwise. **Action owner** (who gets credit/blame) is also derived
automatically — e.g. a Pen Defence or a 6 Again is attributed to the *other* team.

### 4. Game events (optional richness)

- **Drive + / Drive −** — a per-possession quality rating of the attacking set. Only **one
  drive can be tagged per possession**; after you tag one, both buttons lock until the next
  play-ending event, so you can't double-count.
- **Ball Live** — marks the ball coming back into play after a stoppage. The gap between an
  end-of-play event and the next Ball Live is counted as **dead-ball time**, which feeds the
  Dead Ball and Avg Turnover stats. If you never tag Ball Live, those stats just stay at
  zero — everything else still works.
- **To Review** — flags the current moment for later review (shows up as a To-Review row in
  the viewer). Doesn't affect score or possession.

### 5. Undo and edit
- **↩ Undo** removes the most recent event and restores the possession that preceded it.
- Tap any row in **Recent** to open the **Change Event** sheet and re-pick its type
  (Try → Turnover, etc.). Possession for every event *after* the edited one is
  recomputed automatically, so a correction near the start of the game ripples through
  cleanly.

---

## Live stats (📊 Stats)

Opens a bottom-sheet with the running score and a breakdown that **refreshes every 5
seconds** while open:

- **Scoring** — Tries and **Completions** per possession. A possession counts as
  *completed* when it ends in a **Try** or a **6th Touch** — i.e. the team kept the ball
  through the set without conceding an error. **Completions = (tries + 6th-touch
  possessions) ÷ total possessions**, shown as a count and a percentage. Possessions that
  end in a Turnover or a penalty are *not* completions — which is exactly why tagging
  6th Touch vs Turnover correctly matters.
- **Possession** — Avg / Total possession time. (Dead Ball and Avg Turnover appear *only*
  if you've tagged Ball Live events.)
- **Discipline** — Pen Attack and Pen Defence counts.
- **Drives** — Drive + / Drive − (only if you tagged any).

Green/red highlighting and the proportion bar show who's ahead on each metric. All of this
is computed locally from your taps — no network needed.

---

## Broadcasting live (⚫ Live)

Tap **⚫ Live** to start pushing the current score, possession counts, completions, and an
event log to the **Live** page every 30 seconds (the button turns 🔴). Spectators on
`live.html` then see the game appear with a running scoreline. Tapping it off — or closing
the tab — clears the live entry. Live broadcasting needs the team/competition fields filled
in (so it knows which game it is).

---

## Other controls

- **⇆ Swap** — flips which team is shown on the left/right in the scoreboard, possession
  toggle, and stats. **Display only** — it doesn't change any recorded data, just matches
  the on-screen sides to where the teams actually are on the pitch.
- **← Back** — returns to the hub. If you have unsaved events it confirms first (they're
  restored when you come back).

---

## Saving (⬆ Push)

**⬆ Push** uploads every event to the game's sheet tab in the same column layout as the
video annotator (`Time, Possession Owner, Type, Name, To Review, Comment, Action Owner`),
so the game is immediately usable in Game Analysis, Dashboard, and the Viewer.

- The tab is marked **Analyzable = TRUE** automatically when both a Game Start and a Game
  End are present.
- If a non-admin login has **no access group**, Push warns you (the game would be
  admin-only) before continuing.
- If a tab with the same name already exists, Push reports it. An **⚠ Override** button
  appears for **admins** to replace it; non-admins need admin help.

### Video later
There's no video here by design. To attach footage afterwards, an admin uses
**Backfill** (`backfill.html`) to link a YouTube URL and a start offset — the offset aligns
your wall-clock timestamps to the video timeline so the viewer's seek links land on the
right moments.

---

## Your session is saved automatically

Everything (events, metadata, possession, swap state) is stored in the browser's
`localStorage` after every tap. If you close the tab, lose signal, or your phone sleeps,
re-open the page and it offers to **restore the previous session** — so a dropped
connection mid-game never loses your annotations. **🗑 Clear all data** in Setup wipes the
saved session (double-confirmed; irreversible).

> Tip: pushing to the sheet is the durable save. `localStorage` is per-device and
> per-browser — don't switch phones mid-game expecting the session to follow you.

---

## How timing works

Times are **wall-clock based**: the very first tap is `0:00`, and every later event is
measured from that instant. The `Game Start` you tag becomes the reference for the
half/total clocks and the live elapsed time. The **click delay** (Setup) shaves a fixed
number of seconds off each tag after the first to account for the gap between *seeing* a
play and *tapping* it.
