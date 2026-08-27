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

Tagging live is a speed-and-accuracy problem, so the pitch gets the screen and
everything pressable sits in a rail beside it:

- **Every touch is a tap** where it happened. The counter fills 1…6.
- **Tap the pin again** and that touch becomes a **turnover** — the ball was lost
  right there.
- **The sixth tap is the handover.** It's tagged `Turnover / 6th Touch`
  automatically and the ball changes hands.
- **Tap the try band** at the attacking end to score. It carries the attacking
  team's name and colour, so it always says whose try it would be.
- **After a try, and at each kick-off**, a hollow **0** appears on halfway: the
  tap-off. Nothing is recorded until you tap the first touch, but that's where
  the ball is, so a penalty placed without aiming lands there.
- **Penalties, Ball Live, Review and Undo** sit in a rail, under one thumb.
- The **last four events** sit below the pitch; tap one to correct it.

The pitch is drawn to its real 50 m × 70 m proportions, with the full Touch
markings — a **7 m line** at each end, **10 m lines either side of halfway**, and
halfway itself solid — so a position on it means what it looks like. The whole
field is always on screen: it's fitted to whatever space is left rather than
sized from the width, and the page never scrolls while a game is running.

### The pitch turns, and the controls follow it

A fixed-shape pitch can only ever fill one dimension of a screen; the other has
slack. So rather than guess, the page works out how big the pitch would come out
in each combination — **upright or turned**, with the buttons **beneath it,
beside it, or beside it with the last four beside that** — and picks the biggest.
The controls therefore never eat the dimension that is limiting the pitch, and
the answer differs by device:

| Screen | Pitch | Controls | Pitch drawn |
|---|---|---|---|
| Phone portrait | upright | rail beside | 233 × 372 |
| Phone landscape | **turned** | rail + panel beside | 333 × 208 |
| Tablet portrait | upright | rail beside | 495 × 793 |
| Tablet landscape | **turned** | rail + panel beside | 775 × 484 |

**⟳ Turn** overrides the choice and pins it for that device — the controls still
rearrange around whichever way you pinned it. On a tablet held landscape, turning
the pitch is worth about 50% more field than leaving it upright (375k px² against
251k), which is the whole reason it's there.

No button is smaller than **46 px** on its short side, and where six of them
won't fit in a column — a phone held sideways — the rail goes to two columns
rather than pushing Undo off the bottom.

The last four has a **fixed** height rather than a growing one, so the pitch is
exactly as big at full time as it was at kick-off.

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

Because that flip is invisible in the geometry, it's made loud in the colour.
Each team owns one — **Team 1 blue, Team 2 amber** — and it tints the try band,
the possession button, the touch counter and every pin in the current set at
once. When the ball turns over, the whole screen changes colour.

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
5. Possession, score and the touch counter all reset themselves. After a try the
   ball goes back to halfway for the tap-off.
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

> **No Drive +/−.** v2 doesn't tag drive ratings and doesn't offer them as an
> edit target. A game tagged in v1 that has them still shows and counts them.

### Sub-types

After a try, turnover or penalty a strip of sub-types slides in for ten seconds —
`Ball Down`, `32 - Scoop`, `Offside`, and so on, from the same canonical list the
video annotator uses. Tapping one applies it immediately. Ignoring it leaves the
event as `Other`, exactly as v1's buttons do.

Picking `6 Again` or `6th Touch` after the fact re-derives possession from the
corrected name, so a late correction doesn't leave the ball on the wrong team.

### Fixing a mistake

- **↩ Undo**, in the rail, drops the newest event.
- **Tap any row** in the strip along the bottom for the full edit card: change the
  type, pick a sub-type, add a comment — or **Delete**, which is new in v2 and is
  the only way to remove a mis-tap that isn't the most recent event. Possession is
  recalculated for everything after it, and the touches in that set renumber.

Touch numbers are always derived from the event list, so undo, edit, delete and a
mid-game reload all leave the counter and the pins agreeing with each other.

---

## Live stats — what the positions buy you

The 📊 Stats sheet keeps everything v1 reported and opens with the thing you
actually read first: **two pitch maps**, one per team, both drawn attacking
upwards whichever ends they were really playing — because the point of the map
is to compare them.

Each map plots the two touches that decide a set — **touch 3 hollow, touch 4
filled** — with a **line at the average position of each**, dashed for T3 and
solid for T4. An ✕ marks where that team lost the ball and a green dot marks a
try. Plotting every touch was just a cloud; two touches and two lines is a read.
T3 and T4 are told apart by fill and by dash rather than by colour, so the team
colour stays free to mean the team.

Then three figures, all inline SVG and flexbox — no chart library, because the
page has to work at a pitch with no signal.

**How sets end** — one bar per team, split into Try / Turnover / 6th touch /
Penalty. The share of sets a team simply gives away is the fastest read of the
game there is.

**How far each set got** — every set in the order it happened, drawn as a bar
from where it started to where it ended, in metres up that team's half. Bars
that rise won territory; bars that fall gave it away. A green dot marks a try.

**Which channel** — each team's touches split into left / middle / right thirds.

> The four outcome colours were run through a colour-vision check against this
> sheet's surface. Every pair separates cleanly except green↔orange, which lands
> in the marginal band for deuteranopia — so every segment also carries a gap, a
> legend entry and a number on its face. No figure here asks you to tell two
> things apart by hue alone.

Underneath, two sections that only exist because the taps carry a position.
Every distance is in metres up a 70 m field, measured from the team's own line.

**Field position**

| Row | What it tells you |
|---|---|
| **Avg start** | Where their sets begin. The single biggest driver of everything else. |
| **Avg gain per set** | Metres from the first touch to wherever the set ended. Territory earned, not territory held. |
| **Territory** | Share of their touches in the opposition half. |
| **Touches per set** | Whether they're building sets or coughing them up early. |

**Inside the 10m**

| Row | What it tells you |
|---|---|
| **Sets that got there** | How often they reach the opposition 10 m line at all. |
| **Converted to a try** | Of those, how many scored — getting there and finishing are different problems, and this separates them. |
| **Left / mid / right** | Share of touches in each third across the pitch. A lopsided split is a pattern worth knowing about. |

Both sections disappear for a game with no positions, the same way Drives and
Dead Ball sit out when they weren't tagged.

In the **Recent** strip each event carries its position as a channel letter and
a distance — `R 58m`, `M 22m` — rather than raw coordinates, which read as
nothing at a glance.

### Ball Live after every touch

v1 had few enough events that a **Ball Live** could only ever follow the end of a
possession. Here you can tag one after each individual touch — the roll-ball — and
the arithmetic holds: each gap is charged to the team that was in possession, and
**ball-in-play time comes out the same either way**, because the extra elapsed
time you record is exactly the extra dead time subtracted from it. Checked on a
two-half game: tagging roll-balls took Dead Ball from 35s to 107s for one team
while its average possession stayed at 28.8s, and Avg Turnover was untouched at
7s in both.

What does change is what **Dead Ball** means. Tagged only at the end of sets it's
stoppage time; tagged per touch it's stoppage time *plus every roll-ball*, which
is several times larger. Either is defensible, but they aren't comparable — so
pick one habit and keep to it, or the number moves between games for reasons that
have nothing to do with the games.

> One known gap, inherited from v1: a Ball Live tagged after the **last** play of
> a game belongs to no possession, since nobody receives the ball afterwards, so
> those few seconds appear in neither team's Dead Ball. It's one interval per
> game and it is left out rather than assigned to a team that never got the ball.

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

- The previous set stays on the pitch as faded pins until the next one starts, so
  a turnover doesn't blank the screen you were reading.
- The pitch keeps a true aspect rather than stretching to fill a portrait phone,
  so one dimension always has slack. The possession toggle and the last-four list
  sit below it and take it; the last four is capped and scrollable so a long
  history can never push the field off the bottom.
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
