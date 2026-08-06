# Field Annotator — Guide

The **Field Annotator** (`annotator_field.html`) is a phone-first tool for tagging a
touch rugby game **live from the sideline, with no video**. You tap what happens as it
happens; the page timestamps each event against a wall clock, infers possession and the
score, shows running stats, and pushes everything to the Google Sheet in the same format
as the video annotator — so the game shows up in Game Analysis, Dashboard, and Live just
like any other.

> Access: requires the **staff** or **admin** password (entered on `index.html`). The
> password is reused as the API secret and also decides which **access group** your
> uploads are tagged to. No signal at the pitch? Use **📴 Use offline** on the login
> screen — see [Offline mode](#offline-mode) below.

It's a PWA — on iOS/Android you can "Add to Home Screen" and run it full-screen.

---

## The 30-second version

1. Open the annotator → tap **＋ New Game** (or an existing game to carry on with it).
2. **⚙ Setup** → type the two team names (+ year / division / competition).
3. Set **possession** (which team has the ball), then tap **Start** when the whistle goes
   for kickoff.
4. Tap events as they happen: **Try**, **Turnover**, **6th Touch**, **Pen Attack**,
   **Pen Defence**, **6 Again**.
5. Possession and the scoreboard update themselves. Tap a **Recent** row to fix a mistake.
6. Tap **Start** again at half time (it becomes **End**), and again for the second half.
7. At full time tap **End**, then **⏹ Stop Game** and choose **⬆ Finish & Upload**.

Everything below is detail on top of that loop.

---

## The game picker

The annotator opens on a list of every game held on this device — you don't lose one game
by starting another, so a tournament day is just a stack of tiles.

```
┌────────────────────────────────────────────┐
│            ＋ New Game                      │
├────────────────────────────────────────────┤
│ GAMES ON THIS DEVICE      🗑 Delete uploaded │
├────────────────────────────────────────────┤
│ France vs England                      ✕   │
│ 2026 · M30 · SENIORS CUP                   │
│ 3–2 · 47 events · 24:10                    │
│ 🏁 FINISHED   ⚠ NOT UPLOADED     ↺ Reopen  │
└────────────────────────────────────────────┘
```

Each tile carries two badges:

| Status | Meaning |
|---|---|
| **● Live** | The clock is running — a half is in progress. |
| **⏸ Paused** | Started, currently between halves. |
| **🏁 Finished** | Stopped for good; frozen (see below). |
| **Not started** | Created, no kickoff tagged yet. |

| Upload | Meaning |
|---|---|
| **☁ Synced** | In the sheet, with no changes since. |
| **⚠ Unsynced changes** | Uploaded once, but you've corrected something since. |
| **⚠ Not uploaded** | This device is the only copy. |

- Tap a tile to open that game exactly where you left it — clock, possession, teams and all.
- **✕** deletes a game. If it was never uploaded you're asked twice, because nothing else has it.
- **🗑 Delete uploaded** clears out games that are safely in the sheet. Anything still
  waiting to sync is deliberately left alone.
- **← Games** in a game's header brings you back here; **← Hub** leaves for the main menu.

> Games live in this browser's storage on this phone. Uploading is still the durable save.

---

## Layout

```
┌────────────────────────────────────────────────┐
│ TR  Field Annotator  ←Games ⬆Push ⇆Swap 📊Stats │   header
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
│  [ ⏹ Stop Game ]   (only between halves)         │
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

### 6. Finishing the game (⏹ Stop Game)

**⏹ Stop Game** appears only **between halves** — i.e. once you've tagged **End**. That's
deliberate: it means the recorded game always closes on the whistle rather than mid-play.

Tapping it asks one question, which doubles as the upload prompt:

- **⬆ Finish & Upload** — freezes the game and pushes it to the sheet in one go.
- **Finish without uploading** — freezes it now, upload later.
- With **no internet**, the upload option is replaced by a warning: *"No internet — this
  game has not been uploaded. It's saved on this device only. Upload it later from the
  games list."*

Either way the game is **frozen**:

- Both clocks stop for good and can't be restarted.
- The event buttons, possession toggle, Start/End, Undo and ⚫ Live all disappear.
- **Recent** expands to the *whole* event list, because correcting a type is the one thing
  left to do — tap any row to change it (or add a comment). Game Start / Game End /
  Ball Live / Drive conversions are hidden here, since changing those would break the
  frozen clock and the Analyzable flag.

If a game is frozen by mistake, **↺ Reopen** on its picker tile puts it back to paused and
the clock picks up from real time again. You'll need to upload it again afterwards.

### Don't forget to sync

A finished game that isn't in the sheet shows a standing amber banner and keeps an
**⬆ Upload now** button, and its tile reads **⚠ Not uploaded** — the warning doesn't go
away until the push succeeds. If you correct an event type after uploading, the banner
comes back as **⚠ Unsynced changes**, because that correction is only on the phone until
you push again. A finished game that's still waiting also retries by itself the moment the
device comes back online.

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
- **← Games** — returns to the game picker. Nothing is lost: the game keeps its clock and
  events, and tapping its tile picks straight up again. From the picker, **← Hub** leaves
  for the main menu.

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
- Two saved games with the same teams, competition, year and division would resolve to the
  same tab and fight over it. Setup warns you when that's the case — give one of them an
  **ID** to keep them apart.

### Video later
There's no video here by design. To attach footage afterwards, an admin uses
**Backfill** (`backfill.html`) to link a YouTube URL and a start offset — the offset aligns
your wall-clock timestamps to the video timeline so the viewer's seek links land on the
right moments.

---

## Offline mode

Passwords are checked against the server, so with no connection there's nothing to check
against. **📴 Use offline** on the login screen exists for that: it opens a local-only mode
where the Field Annotator is the *only* page available, because it's the only one that
works entirely on the device.

In offline mode:

- Tagging, the clock, live stats, editing and Stop Game all behave exactly as normal.
- **⬆ Push** and **⚫ Live** are greyed out — there's no account to upload with.
- **⏹ Stop Game** offers only *Finish game*, with a note that it hasn't been uploaded.
- Setup shows *"Offline mode — this game stays on the device. It will be tagged to
  whichever account uploads it later."*

### Uploading afterwards

Games tagged offline carry **no identity**, so they're attributed to whoever is signed in
when they're finally pushed:

1. Tag the game(s) offline and finish them. They sit in the picker marked **⚠ Not uploaded**.
2. When you have signal, go to the hub and **log in** normally. (Logging in automatically
   leaves offline mode; your games are untouched.)
3. Open the Field Annotator, tap the game, and hit **⬆ Sync** — it uploads under your
   account and is tagged to your group, exactly as if you'd been logged in all along.

> Nothing expires and nothing is lost in the meantime — but the games only exist on that
> one phone until you do step 3.

---

## Your games are saved automatically

Everything (events, metadata, possession, swap state) is written to the browser's
`localStorage` after every tap, under its own key per game. If you close the tab, lose
signal, or your phone sleeps, just re-open the annotator and tap the game in the picker —
it resumes exactly where it was, so a dropped connection mid-game never loses your
annotations. Delete a game with the **✕** on its
picker tile (double-confirmed when it hasn't been uploaded; irreversible).

> Tip: pushing to the sheet is the durable save. `localStorage` is per-device and
> per-browser — don't switch phones mid-game expecting the games to follow you.

---

## How timing works

Times are **wall-clock based**: the very first tap is `0:00`, and every later event is
measured from that instant. The `Game Start` you tag becomes the reference for the
half/total clocks and the live elapsed time. The **click delay** (Setup) shaves a fixed
number of seconds off each tag after the first to account for the gap between *seeing* a
play and *tapping* it.
