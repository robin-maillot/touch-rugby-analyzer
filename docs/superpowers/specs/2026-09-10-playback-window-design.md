# Selectable playback window (pre/post event)

Date: 2026-09-10

## Problem

The Event Viewer (`game.html`) hardcodes how much video surrounds an event:
`const LOOP_BEFORE=5, LOOP_AFTER=2`. The pre-roll decides where playback starts
on every event tap; the post-roll decides where a loop turns around. Neither can
be changed without editing the page. Different events want different windows — a
strike move needs a longer run-up than a penalty restart.

The older table view (`viewer.html`) already exposes both values as chips plus a
custom number input, defaulting to 5 and 2.

## Goal

Make the window selectable in the Event Viewer, defaulting to **5 s before** and
**3 s after**, and move the table view's post-roll default to 3 s so the two
pages agree.

## Design

### 1. Shared clamp helper — `js/player.js`

`TR.player.clampWindow(val, fallback)` returns an integer in 0–120, or
`fallback` when the input is null, undefined or non-numeric. Floats truncate.

One definition serves both pages. It lives in `js/player.js` rather than inline
so `test.js` can exercise it — the Node harness loads `js/*.js` and cannot see
page script.

### 2. Event Viewer — `game.html`

- Replace the two constants with mutable `winBefore` / `winAfter`, seeded from
  `LS.get('winBefore')` / `LS.get('winAfter')` through `clampWindow`, defaulting
  to 5 and 3.
- `playEvent` uses them in place of the constants, and records `activeSecs` /
  `activeGame` so the window can be re-derived without re-tapping the event.
- A new transport button sits after the loop button, labelled with the live
  values (`5/3`), opening `openSheet('window')`.
- A new sheet kind, **"Playback window"**: a Before group (chips 3/5/10/15) and
  an After group (chips 2/3/5/10), each with a custom number input, built from
  the existing `.grp` / `.opts` / `.chip` classes.
- Setting a value clamps it, persists it, relabels the button, re-derives
  `loopStart` / `loopEnd` for the active event, and — when looping is on and the
  playhead now sits outside the window — seeks to the new start. The sheet stays
  open so both values can be set in one visit.
- `progress()` gains a `loopEnd > loopStart` guard, so a 0/0 window plays
  straight through instead of seeking back on every tick.

**Layout.** The transport currently holds 7 buttons. An 8th overflows a 375 px
phone: 8x42 + 7x6 + 16 = 394. Under `max-width:400px`, `.t-btn` min-width drops
to 38 px and the transport gap to 4 px, giving 348 px. This is deliberate — the
CSS comment above the landscape-phone media query documents how little vertical
and horizontal slack the viewer has on a phone.

### 3. Table view — `viewer.html`

Defaults only:

- `linkOffsetAfter` initialises to 3 instead of 2, and the number input's
  `value` attribute matches.
- The after-chips become 3/5/10 rather than 2/5/10, so the new default
  highlights a chip instead of reading as a custom value. 2 s stays reachable
  through the number input.
- `setLinkOffset` and `setLinkOffsetAfter` adopt `clampWindow`, which makes the
  `max="120"` already on those inputs actually enforced.

The before value stays 5. Anyone who has already used the picker has a saved
`localStorage` value and keeps it; only new users see the changed default.

### 4. Service worker — `sw.js`

`CACHE_VERSION` goes v18 -> v19, since both page shells change.

## Testing

`test.js` gains `clampWindow` assertions: the default path, clamping at 0 and at
120, non-numeric input, and float truncation.

The UI wiring — sheet, chips, button label, live re-derivation — is not reachable
from the Node harness and is verified in the browser preview.

## Edge cases

- Before 0: playback starts exactly on the event.
- Before and After both 0: the loop guard keeps the video playing rather than
  freezing on a zero-length window.
- A Before large enough to precede the recording: already handled by the
  existing `Math.max(0, ...)` in `playEvent`.

## Out of scope

- Per-event or per-event-type windows. One window applies to all events.
- Changing the fixed 5 s lookback baked into `TR.player.seekLink`, which the
  table view's copy-link path deliberately cancels out.
