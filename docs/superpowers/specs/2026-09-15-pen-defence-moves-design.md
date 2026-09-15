# Attacking moves on defensive penalties

**Date:** 2026-09-15
**Status:** Implemented

## Problem

[2026-09-06-strike-moves-design.md](2026-09-06-strike-moves-design.md) gave
Turnovers and attack penalties a Strike Move, so a move's try rate could be
computed over every attempt. It deliberately excluded Penalty Defence:

> `6 Again` and Penalty Defence are therefore **excluded**, and the annotators
> must not offer the move picker on them — one less tap on the sideline, and no
> junk in the denominator.

That reasoning holds for the *denominator*. It does not hold for *recording*.
A defensive penalty is often the direct result of the move being run — a 32
that stretches the defence draws the offside. Right now that is unrecorded, so
there is no way to ask which moves pressure a defence into conceding.

## Decision

Offer the move picker on Penalty Defence, and keep its value out of every rate.

The move is **context, not an attempt**. The attack keeps the ball, and the
same attack goes on to end in a Try, Turnover or Penalty Attack carrying its
own move. Counting the penalty as well would put one attack in the denominator
twice and depress the rate of exactly the moves that work.

## Domain model

The one gate becomes two, because "should this event record a move?" and "does
this event's move count as an attempt?" stopped being the same question.

| Helper | Answers | Penalty Defence |
|---|---|---|
| `TR.isAttackEnd` | did the ball change hands? | `false` (unchanged) |
| `TR.offersStrikeMove` | show the picker / allow an edit? | **`true`** (new) |
| `TR.strikeMoveOf` | the move for the rate maths | `''` (unchanged) |
| `TR.recordedMoveOf` | the move to show and export | **the stored value** (new) |

`TR.isAttackEnd` and `TR.strikeMoveOf` are untouched, which is what makes
"context only" hold by construction: `TR.strikeMoveStats` gates on
`isAttackEnd` before it reads anything, so a defensive penalty cannot reach a
rate, a coverage figure, or either leaderboard — no matter what is stored on
it, and no matter which surface hands it over.

`6 Again` stays excluded. It keeps the ball for the same reason, so the move is
still running and will be recorded on whatever ends the attack. Only defensive
penalties were asked for, and they are the ones a defence concedes.

## Surfaces

Recording (`offersStrikeMove`): all three annotators —
`annotator.html` (second stage after the sub-type, still skippable, still
suppressed in Simple Mode), `annotator_field.html` and `annotator_field2.html`
(edit sheet, plus v2's post-tag move strip) — and `viewer.html`'s inline edit.

Display and export (`recordedMoveOf`): the annotator's move column, CSV export,
Push to Sheet, the Editor's move cell, and `game.html`'s event cards, tags and
filters.

Untouched (`strikeMoveOf`): `js/strike_moves.js`, `dashboard.html` and
`analytics.html` — `analytics.html`'s breakdown-by-move is gated on
`TR.STRIKE_MOVE_TYPES`, which Penalty Defence is not in.

## Data model

No change. The existing `Strike Move` column carries it, so no sheet migration
and no backfill — old tabs simply have blanks, as before.

## Testing

`test.js` and `tests.html` cover both new helpers, and assert directly that
adding a Penalty Defence carrying a move to an event list leaves
`TR.strikeMoveStats` byte-for-byte identical — attempts, rate and coverage all
unmoved. That assertion is the spec: if a later change lets a defensive penalty
into the rate maths, it fails.

## Not included

A "penalties drawn per move" statistic. The data is now being recorded, which
is the prerequisite; what it says is worth seeing before a surface is built on
it.
