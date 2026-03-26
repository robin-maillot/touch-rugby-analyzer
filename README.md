# Touch Rugby Analyzer

A web-based tool for annotating touch rugby match footage with events, possession tracking, and statistics.

**Deployment:** [touch-rugby-analyzer](https://robin-maillot.github.io/touch-rugby-analyzer/)

Run locally:
```
python main.py
```

---

## Annotator (viewer.html)

The annotator is the main tool for tagging events in a match video. Access it via the **Annotator** link in the nav bar (staff login required).

### Loading a video

- **Local file** — click **Open** or **Open Local Video** and select a video file from your machine. You'll be prompted to load a matching CSV if one is available.
- **YouTube** — paste a YouTube URL into the field in the header and click **▶ Load YT** (or press Enter). The video will load in the player and annotations can be pushed directly to Google Sheets.

---

### Modes

Two toggles control how the annotator behaves. They can be combined.

#### Full Game Analysis
When **on**, the tool tracks which team has possession for every event. Most events infer the possession owner automatically from the previous event — you only need to manually select a team for **Game Start** and **To Review** events.

When **off**, possession is not tracked at all. Use this for lightweight tagging where you only care about event types and timestamps.

#### Simple Mode
When **on**, the sub-type for Try, Turnover, Penalty Attack and Penalty Defence is automatically set to **Other** — so tagging any of those requires only a single click. A dedicated **6th Touch** button also appears for fast turnover tagging.

When **off**, you must explicitly choose a sub-type from the submenu before the event is saved.

> **Simple Mode + Full Game Analysis**: possession is inferred automatically. If it cannot be inferred (i.e. no Game Start has been tagged yet), an error will appear — tag a Game Start event first.

---

### Action buttons

Press the keyboard shortcut or click the button, then select a sub-type if prompted.

#### Try — `T`
A try is scored by the attacking team. The possession owner is the team that scored.

Sub-types: `Scoop`, `32 - Long`, `33 Quicky`, `33`, `32 Cut`, `French Flair`, `Other`

#### Turnover — `U`
Possession changes from the attacking team to the defending team (except **6 Again**, which keeps possession).

| Sub-type | When to use |
|---|---|
| Ball Down | The ball hits the ground |
| 6th Touch | The attacking team uses all 6 touches |
| Dummy Touch | Referee calls a dummy touch |
| Bad Roll | The roll ball goes off-line or is touched incorrectly |
| 6 Again | Referee awards 6 more touches — possession **stays** with the attacking team |
| Interception | Defending team intercepts the ball |
| Other | Any other turnover |

#### Pen Attack — `P`
A penalty awarded **against the defence** (the attacking team benefits). Use this when the defending team commits a foul.

Sub-types: `Forward Pass`, `Touch and Pass`, `Off the Mark`, `Not Moving Forward`, `Delay of Play`, `Hard Touch`, `Other`

#### Pen Defence — `Q`
A penalty awarded **against the attack** (the defending team benefits). Use this when the attacking team commits a foul. Possession transfers to the defending team.

Sub-types: `Offside`, `Hard Touch`, `In the Ruck`, `Not Moving Forward`, `Other`

#### Game Event — `G`
Structural events that mark the boundaries of play. Always requires a possession owner to be selected (which team kicks off or restarts).

| Sub-type | When to use |
|---|---|
| Game Start | Start of the game or a half — tag this **first** so possession can be inferred for all subsequent events |
| Game End | End of the game or a half |

#### To Review — `R`
Marks a moment in the video for later review. An optional free-text note can be added. Always requires a possession owner to be selected.

#### 6th Touch — `6` *(Simple Mode only)*
A shortcut that tags a Turnover with sub-type **6th Touch** in a single click. Equivalent to clicking Turnover → 6th Touch in the submenu. Only visible when Simple Mode is on.

---

### Possession inference

In Full Game Analysis mode, the annotator automatically determines which team has possession when you tag an event, based on the most recent event **before the current video position**:

- **Possession stays** with the same team after: Try, Penalty Attack, Turnover (all sub-types except 6 Again)
- **Possession switches** to the other team after: Penalty Defence, Turnover → 6 Again

This means you can go back and tag events in the middle of a match and the inference will still be correct relative to where you are in the video.

---

### Editing annotations

The annotations list at the bottom shows every tagged event in chronological order.

- **Seek** — click anywhere on a row to jump to that timestamp in the video.
- **Change sub-type** — click on the sub-type name (shown with a dashed underline) to open an inline dropdown and select a different option.
- **Add / edit comment** — click **✎** (or **💬** if a comment exists) to open the comment editor.
- **Delete** — click **✕** on the right of the row.
- **Clear all** — the **Clear All** button removes every annotation after confirmation.

The **⏮ Last Tag** button jumps the video back to the most recently tagged event, useful for reviewing the last tag without losing your place.

---

### Metadata

Fill in the fields in the metadata bar before pushing to Google Sheets:

| Field | Example |
|---|---|
| Team 1 | France |
| Team 2 | England |
| Year | 2025 |
| Division | M30 |
| Competition | Seniors Cup |

The tab name in Google Sheets is generated automatically from these fields (e.g. `2025_m30_seniors-cup_france_england`). The preview updates as you type.

---

### Saving and loading

#### Export CSV
Downloads all annotations as a `.csv` file. In Full Game Analysis mode the CSV includes a **Turnover** column.

#### Load CSV
Imports annotations from a previously exported CSV file. If the CSV contains a YouTube link, the video is loaded automatically.

#### Push to Sheet — `⬆`
Uploads annotations to the configured Google Sheet. Requires metadata to be filled in. If the tab already exists, you'll be prompted for an admin override password before the existing data is replaced.

#### Load from Sheet — `⬇`
Downloads annotations from a Google Sheet tab. Select the game from the dropdown and click the button. The video (if YouTube) and metadata are loaded automatically.

---

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `←` / `→` | Skip −5s / +5s |
| `T` | Tag Try |
| `U` | Tag Turnover |
| `P` | Tag Penalty Attack |
| `Q` | Tag Penalty Defence |
| `G` | Tag Game Event |
| `R` | Tag To Review |
| `6` | Tag 6th Touch *(Simple Mode only)* |
| `1`–`9` | Select sub-type by position in the submenu |
| `1` / `2` | Select Team 1 / Team 2 when possession menu is open |
| `Escape` | Cancel current selection |
