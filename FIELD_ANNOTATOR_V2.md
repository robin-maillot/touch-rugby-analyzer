# Field Annotator v2 — Guide

`annotator_field2.html` is the Field Annotator with the event grid replaced by a
**pitch you tap**. Everything else — the game picker, the clock, live stats, live
broadcasting, editing, Stop Game, and the push to the Google Sheet — is the same
tool, so a game tagged here lands in Game Analysis exactly like any other.

> **Access: admin only.** It's hidden from staff and viewers, and `TR.auth('admin')`
> turns anyone else away at the door. See [Who can see it](#who-can-see-it).

The original stays at [`annotator_field.html`](FIELD_ANNOTATOR.md) and is untouched.
Both read and write the same on-device game store, so a game can be opened in either.

---

## The idea

Tagging live is a speed-and-accuracy problem, so the pitch gets the whole screen:

- **Every touch is a tap** where it happened. The touch counter fills 1…6.
- **Tap the pin again** and that touch becomes a **turnover** — the ball was lost
  right there.
- **The sixth tap is the handover.** It's tagged `Turnover / 6th Touch`
  automatically and the ball changes hands.
- **Tap the green try band** at the attacking end to score.
- **Penalties stay as buttons** along the top, where they can be hit without
  looking down.
- The **last four events** sit along the bottom; tap one to correct it.

### Attack-normalised: the pitch never turns round

Whoever has the ball **always attacks up the screen**. When possession flips, the
pitch relabels itself rather than reorienting, so *forward* is the same direction
all game and one tap always means the same thing. Positions are recorded the same
way:

| Axis | 0 | 100 |
|---|---|---|
| **y** | the attacking team's own try line | the try line they're attacking |
| **x** | the attacking team's left touchline | their right touchline |

That's the form you want for analysis anyway — a heat map of where sets die reads
the same whichever end a team happened to be kicking towards.

> **⇆ Swap** still only flips the scoreboard for the sideline you're standing on.
> It never touches the pitch or the recorded data.

---

## The tagging loop

1. **＋ New Game** → **⚙ Setup** → team names, year, division, competition.
2. Set **possession**, then **Start** on the kickoff whistle. The pitch appears.
3. Tap each touch. Watch the six dots down the left edge.
4. When the set ends:
   - it survived six touches → the **6th tap** ends it for you;
   - the ball went down → **tap the last pin**;
   - they scored → **tap the try band**;
   - a penalty → **tap the penalty button, then tap the spot**.
5. Possession, score and the touch counter all reset themselves.
6. **End** at half time, **Start** again for the second half, **End** at full time,
   then **⏹ Stop Game** → **⬆ Finish & Upload**.

### Penalties

The three buttons at the top are the same three events v1 tags, with the same
effect on possession (it comes from `js/possession.js`, shared with every other
page):

| Button | Tags | Possession |
|---|---|---|
| **Pen Attack** | `Penalty Attack` | **switches** — the ball goes over |
| **Pen Defence** | `Penalty Defence` | **stays** — the attack keeps it |
| **6 Again** | `Turnover / 6 Again` | **stays**, and the touch count restarts |

Tapping one **arms** it (it turns yellow) and the next tap on the pitch places it.
Tap the armed button a second time to drop it where the ball already is — so a
penalty never costs two taps unless the placement is worth one.

> ⚠️ The v1 guide's table lists Pen Attack and Pen Defence the other way round.
> The behaviour above is what `TR.inferPossessionAfter` has always done, and is
> what both annotators produce; the discrepancy is in that document, not the code.

### Sub-types

After a try, turnover or penalty a strip of sub-types slides in for ten seconds —
`Ball Down`, `32 - Scoop`, `Offside`, and so on, from the same canonical list the
video annotator uses. Tapping one applies it immediately. Ignoring it leaves the
event as `Other`, exactly as v1's buttons do.

Picking `6 Again` or `6th Touch` after the fact re-derives possession from the
corrected name, so a late correction doesn't leave the ball on the wrong team.

### Fixing a mistake

- **↩ Undo** drops the newest event.
- **Tap any row** in the strip along the bottom for the full edit card: change the
  type, pick a sub-type, add a comment — or **Delete**, which is new in v2 and is
  the only way to remove a mis-tap that isn't the most recent event. Possession is
  recalculated for everything after it, and the touches in that set renumber.

Touch numbers are always derived from the event list, so undo, edit, delete and a
mid-game reload all leave the counter and the pins agreeing with each other.

---

## What reaches the Google Sheet

The sheet keeps its seven-column shape (`Time, Possession Owner, Type, Name,
To Review, Comment, Action Owner`). The pitch position rides in the **Comment**
column — the one free-text column — as `@x,y`:

```
0:02:44   Team 1   Try        Other   ""   @50,100          Team 1
0:01:54   Team 1   Turnover   Other   ""   dropped it @62,48  Team 1
```

Anything you actually typed as a comment is kept in front of it.

**The individual touches are not uploaded by default.** Setup has a switch,
*"Upload the individual touches too"*. Off, only the events v1 would have produced
go up — with their positions. On, every touch goes up as well, as type `Touch`,
name `Touch 1`…`Touch 5`.

That default is deliberate: `Touch` is inert for every possession and stat
calculation in the app (it isn't a possession trigger, so `TR.isTurnover` leaves
the chain alone), but it multiplies the row count of a sheet tab and makes the
Event Viewer's list a lot longer. Turn it on once you've looked at a tab and
decided you want the detail. Touches are always kept on the device regardless.

---

## Who can see it

| Where | Rule |
|---|---|
| The page | `TR.auth('admin')` — staff and viewers are redirected to the hub |
| The hub card | Shown to admins only |
| Offline mode | Shown only on a device that has signed in as admin before |

That last one exists so the tool is reachable at a pitch with no signal, where
there's no password to check, without putting it on anyone else's phone. The flag
is `localStorage.trl2_admin_device`, set at admin login and local to that browser.

Offline mode otherwise behaves exactly as it does in v1: tagging, the clock, stats,
editing and Stop Game all work; **⬆ Push** and **⚫ Live** are greyed out; games
tagged offline are attributed to whoever finally uploads them.

---

## Layout notes

- The pitch is stretched to fill the space rather than letterboxed, so a phone in
  portrait gets a tall pitch. Positions stay a faithful linear mapping — only the
  drawing is stretched — so the recorded data is unaffected.
- The previous set stays on the pitch as faded pins until the next one starts, so
  a turnover doesn't blank the screen you were reading.
- Only the newest pin is tappable. That tap means "the ball was lost here", and
  letting older pins take it would make the gesture ambiguous.
- While a game is running the header drops its labels to icons and the scoreboard
  compresses, so the pitch keeps the height.

---

## Mockups

Three layouts were built and compared before this one was picked; they're at
[`mockups/`](mockups/index.html) and are all tappable:

- **A — full-bleed pitch** (built): the pitch is the screen, everything floats over it.
- **B — split deck**: pitch on top, a labelled button deck below. More legible, half the pitch.
- **C — landscape two-thumb**: attack runs left→right, rails either side. Better
  pitch geometry, but landscape-only is a worse home-screen app.
