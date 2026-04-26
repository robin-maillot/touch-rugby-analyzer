# Touch Rugby Analyzer

A web-based platform for annotating, reviewing, and analysing touch rugby matches.

**Deployment:** [touch-rugby-analyzer](https://robin-maillot.github.io/touch-rugby-analyzer/)

---

## Access levels

All pages are behind a password prompt on `index.html`. Three passwords exist:

| Password | Access |
|---|---|
| `m30` | Viewer — Game Analysis, Dashboard, Event Viewer, Live |
| `m30-staff` | Staff — everything above + Annotator + Field Annotator |
| `m30-admin` | Admin — everything above + Video Backfill + inline event editing in the viewer + sheet override |

The password is stored in `sessionStorage` and used as the API secret for all calls to the Apps Script backend.

---

## Development

No build step. Open pages directly in a browser, or use a local HTTP server (required now that shared JS lives in `js/`):

```bash
python3 -m http.server
# → http://localhost:8000
```

Any equivalent works: `npx serve .`, `php -S localhost:8000`, etc.

**Unit tests** (pure JS utilities, no browser needed):

```bash
node test.js
```

**Browser tests** (same assertions via QUnit, with the server running):

```
http://localhost:8000/tests.html
```

Tests also run automatically in CI and must pass before each deployment.

---

## Pages

### Game Analysis (`games.html`)
*Available to all*

Charts and statistics for a single game. Select a game from the dropdown to load it. Only games marked **Analyzable** in `_metadata` are shown. Displays possession charts, try timelines, penalty breakdowns, and half-by-half stats.

---

### Dashboard (`dashboard.html`)
*Available to all*

Aggregate view across all analysable games. Shows team rankings, cumulative try charts, and cross-game comparisons.

---

### Event Viewer (`viewer.html`)
*Available to all*

Searchable, filterable table of every event across all games. Clicking a row with a YouTube link plays the video at that timestamp. Supports filtering by Type, Name, Possession Owner, Action Owner, and Game. Events without a YouTube link are hidden.

**Admin extras:** Name and Comment cells become editable inline. Name shows a dropdown with the same options as the annotator (depends on Type). A **💾 Save N changes** button appears in the toolbar when edits are pending and sends all changes to the sheet in one request.

---

### Live (`live.html`)
*Available to all — uses `m30` as fallback if not logged in*

Displays live scores for games currently being annotated in live mode. Cards show team names, score, and elapsed game time. Updates every 30 seconds. Games that haven't pushed an update in the last 2 minutes are hidden automatically.

---

### Annotator (`annotator.html`)
*Staff and admin only*

The main tool for tagging events against a match video (local file or YouTube).

#### Loading a video
- **Local file** — click **Open** and select a video. You'll be prompted to load a matching CSV.
- **YouTube** — paste a URL into the header field (supports `/watch`, `/live`, `youtu.be` formats) and click **▶ Load YT**.

#### Event types

| Button | Key | Description |
|---|---|---|
| Try | `T` | Try scored by the possession team. Sub-types: Scoop, 32-Long, 33 Quicky, 33, 32 Cut, French Flair, Other |
| Turnover | `U` | Ball changes hands. Sub-types: Ball Down, 6th Touch, Dummy Touch, Bad Roll, 6 Again, Interception, Other |
| Pen Attack | `P` | Penalty against the defence (attacking team benefits) |
| Pen Defence | `Q` | Penalty against the attack (defending team benefits, possession switches) |
| Game Event | `G` | Game Start or Game End — tag first so possession can be inferred |
| To Review | `R` | Marks a moment for later review |
| 6th Touch | `6` | Shortcut for Turnover → 6th Touch (Simple Mode only) |

#### Modes
- **Full Game Analysis** — tracks possession for every event. Inferred automatically after a Game Start is tagged.
- **Simple Mode** — sub-type defaults to Other on a single click; a 6th Touch shortcut button appears.

#### Possession inference
- Possession **stays** after: Try, Penalty Attack, Turnover (all except 6 Again)
- Possession **switches** after: Penalty Defence, Turnover → 6 Again

#### Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `←` / `→` | Skip −5s / +5s |
| `T` `U` `P` `Q` `G` `R` | Tag event type |
| `6` | Tag 6th Touch (Simple Mode) |
| `1`–`9` | Select sub-type by position |
| `Escape` | Cancel selection |

#### Metadata
Fill in Team 1, Team 2, Year, Division, Competition before pushing. The Google Sheet tab name is generated automatically (e.g. `2025_m30_seniors-cup_france_england`).

#### Saving
- **⬆ Push to Sheet** — uploads all annotations. If the tab already exists, an Override button appears (admin only; no extra password needed).
- **⬇ Load from Sheet** — downloads annotations for the selected game.
- **⬇ Export CSV / 📂 Load CSV** — local CSV backup.
- **⚫ Go Live** — starts broadcasting the current score every 30 seconds to the Live page. Turns 🔴 when active. Clears the live entry when turned off or the tab is closed.

---

### Field Annotator (`field.html`)
*Staff and admin only*

Mobile-friendly annotation without a video. Designed for use on the pitch during live games. Timestamps are wall-clock based: the first event is `0:00:00` and all subsequent events are offset from that.

#### Layout
- **Possession toggle** — set which team has the ball before tagging. Labels update when team names are filled in.
- **Event buttons** (two-column grid):

| Left | Right |
|---|---|
| Try | Turnover |
| Pen Attack | 6th Touch |
| Pen Defence | 6 Again |

- **Game Start / Game End** — structural events; the game clock counts up from Game Start.
- **Recent events** — last 5 events shown, with an Undo button to remove the last one.

#### Possession rules
- Try and Turnover (including 6th Touch) auto-switch possession to the other team.
- 6 Again keeps possession with the attacking team.
- Try ownership is inferred from possession at the time of tagging.

#### Setup panel
Tap **⚙ Setup** to enter Team 1, Team 2, Year, Division, Competition. The Google Sheet tab name preview updates as you type.

#### Saving
- **⬆ Push** — same format as the video annotator, fully compatible with the Games and Dashboard pages.
- **⚫ Live** — same live broadcast as the video annotator. Uses real elapsed time since Game Start.

---

### Video Backfill (`backfill.html`)
*Admin only*

Links a YouTube video to a game that was annotated without one (e.g. annotated on the field). Shows games missing a YouTube URL at the top and already-linked games below.

For each game, enter:
- **YouTube URL** — the video to link (any standard YouTube URL format).
- **Start offset (MM:SS)** — the timestamp in the video that corresponds to annotation time `0:00:00`. For example, if the game starts 2 minutes and 15 seconds into the video, enter `2:15`.

On submit, every event row in the sheet gets a YouTube link computed as `video_offset + event_time − 5s` (5-second lookback). The YouTube URL is also saved to `_metadata` so the game won't appear in the missing list again.

---

## Apps Script backend

All data is stored in a single Google Sheet. The Apps Script web app (`apps_script/Code.gs`) serves as the API.

### `_metadata` sheet

One row per game. Columns: `Sheet Name`, `Team 1`, `Team 2`, `Competition`, `Year`, `Division`, `Video Name`, `Analyzable`, `Youtube URL`.

- **Analyzable** — `TRUE` if the game has both a Game Start and Game End event. Set automatically on push. Only `TRUE` games appear in Game Analysis and Dashboard.
- **Youtube URL** — set by the Video Backfill page. Used to detect which games need backlinking.

### `_live` sheet

Transient live game state. One row per active live session. Columns: `Sheet Name`, `Team 1`, `Team 2`, `Score 1`, `Score 2`, `Time Seconds`, `Updated At`. Managed automatically by the annotators.

### Game tabs

Each game is a separate sheet tab named `YEAR_DIVISION_COMPETITION_TEAM1_TEAM2`. Columns: `Time`, `Possession Owner`, `Type`, `Name`, `To Review`, `Comment`, `Youtube Link`, `Action Owner`.

### Deploying updates

After editing `Code.gs`, go to the Apps Script editor → **Deploy → Manage deployments → edit the existing deployment → New version**. The web app URL stays the same.
