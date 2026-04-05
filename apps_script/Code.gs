// ── Configuration ──────────────────────────────────────────────
const SHEET_ID        = '1B9DwThGoINgicevtjoUDkeQBoGBCD6QFjL0oL7mAU-k';
const VALID_SECRETS   = new Set(['m30', 'm30-admin', 'm30-staff']);
const ADMIN_SECRET    = 'm30-admin';
const METADATA_SHEET  = '_metadata';  // reserved tab name for game metadata
const LIVE_SHEET      = '_live';      // reserved tab name for live game state

// Expected column order (must match what the Python pipeline reads)
const HEADERS = ['Time', 'Possession Owner', 'Type', 'Name', 'To Review', 'Comment', 'Action Owner'];

// Metadata columns appended to action=all rows
const META_COLS = ['Team 1', 'Team 2', 'Competition', 'Year', 'Division', 'Video Name', 'Analyzable'];

// ── Metadata helper ────────────────────────────────────────────
// Reads _metadata sheet and returns { [sheetName]: { team1, team2, competition, year, division } }
function getMetadata() {
  const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(METADATA_SHEET);
  if (!sheet) return {};

  const values = sheet.getDataRange().getDisplayValues();
  if (values.length < 2) return {};

  const h    = values[0].map(s => s.toLowerCase().trim());
  const col  = name => h.indexOf(name);
  const ni   = col('sheet name'), t1i = col('team 1'), t2i = col('team 2');
  const ci   = col('competition'), yi = col('year'), di = col('division');
  const vi   = col('video name'), ai = col('analyzable'), yui = col('youtube link');

  const meta = {};
  values.slice(1).forEach(row => {
    const name = row[ni];
    if (!name) return;
    meta[name] = {
      team1:       t1i >= 0 ? row[t1i] : '',
      team2:       t2i >= 0 ? row[t2i] : '',
      competition: ci  >= 0 ? row[ci]  : '',
      year:        yi  >= 0 ? row[yi]  : '',
      division:    di  >= 0 ? row[di]  : '',
      video:       vi  >= 0 ? row[vi]  : '',
      analyzable:  ai  >= 0 ? row[ai]  : '',
      youtubelink: yui >= 0 ? row[yui] : '',
    };
  });
  return meta;
}

// ── Cache helpers ───────────────────────────────────────────────
const CACHE_TTL = 300; // seconds (5 min)

function cacheGet(key) {
  try { return CacheService.getScriptCache().get(key); } catch(e) { return null; }
}

function cachePut(key, str) {
  try {
    // CacheService limit is 100 KB per entry
    if (str && str.length < 95000) CacheService.getScriptCache().put(key, str, CACHE_TTL);
  } catch(e) {}
}

function cacheClear() {
  try { CacheService.getScriptCache().removeAll(['list', 'all']); } catch(e) {}
}

function rawJson(str) {
  return ContentService.createTextOutput(str).setMimeType(ContentService.MimeType.JSON);
}

// ── GET — list sheets or fetch rows from a tab ─────────────────
function doGet(e) {
  try {
    if (!VALID_SECRETS.has(e.parameter.secret)) {
      return json({ ok: false, error: 'Unauthorized' });
    }

    // action=live → current live game states
    if (e.parameter.action === 'live') {
      const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(LIVE_SHEET);
      if (!sheet) return json({ ok: true, games: [] });
      const values = sheet.getDataRange().getDisplayValues();
      if (values.length < 2) return json({ ok: true, games: [] });
      const h   = values[0].map(s => s.toLowerCase().trim());
      const col = name => h.indexOf(name);
      const ni  = col('sheet name'), t1i = col('team 1'), t2i = col('team 2');
      const s1i = col('score 1'),    s2i = col('score 2');
      const tsi = col('time seconds'), uai = col('updated at');
      const games = values.slice(1)
        .map(row => ({
          name:        row[ni]  || '',
          team1:       t1i >= 0 ? row[t1i] : '',
          team2:       t2i >= 0 ? row[t2i] : '',
          score1:      s1i >= 0 ? Number(row[s1i]) : 0,
          score2:      s2i >= 0 ? Number(row[s2i]) : 0,
          timeSeconds: tsi >= 0 ? Number(row[tsi]) : 0,
          updatedAt:   uai >= 0 ? row[uai] : '',
        }))
        .filter(g => g.name);
      return json({ ok: true, games });
    }

    // action=list → sheet names + metadata for each game
    if (e.parameter.action === 'list') {
      const hit = cacheGet('list');
      if (hit) return rawJson(hit);

      const meta   = getMetadata();
      const result = JSON.stringify({ ok: true, sheets: Object.entries(meta).map(([name, m]) => ({ name, ...m })) });
      cachePut('list', result);
      return rawJson(result);
    }

    // action=all → every row from every game sheet listed in _metadata, metadata columns appended
    if (e.parameter.action === 'all') {
      const hit = cacheGet('all');
      if (hit) return rawJson(hit);

      const meta   = getMetadata();
      const ss     = SpreadsheetApp.openById(SHEET_ID);
      const metaNames = new Set(Object.keys(meta));
      const sheets = ss.getSheets().filter(s => metaNames.has(s.getName()));
      const allRows = [];
      let headerSent = false;

      for (const sheet of sheets) {
        const values = sheet.getDataRange().getDisplayValues();
        if (values.length < 2) continue;
        if (!headerSent) {
          allRows.push([...values[0], 'Game', ...META_COLS]);
          headerSent = true;
        }
        const name = sheet.getName();
        const m    = meta[name] || {};
        const metaValues = [m.team1 || '', m.team2 || '', m.competition || '', m.year || '', m.division || '', m.video || '', m.analyzable || ''];
        values.slice(1).forEach(row => allRows.push([...row, name, ...metaValues]));
      }

      const result = JSON.stringify({ ok: true, rows: allRows });
      cachePut('all', result);
      return rawJson(result);
    }

    const sheetName = e.parameter.sheetName;
    if (!sheetName) {
      return json({ ok: false, error: 'Missing sheetName parameter' });
    }

    const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(sheetName);
    if (!sheet) {
      return json({ ok: false, error: `Tab "${sheetName}" not found.` });
    }

    // getDisplayValues avoids Date-object conversion on time-formatted cells
    const values = sheet.getDataRange().getDisplayValues();
    const meta   = getMetadata();
    return json({ ok: true, rows: values, meta: meta[sheetName] || {} });

  } catch (err) {
    return json({ ok: false, error: err.toString() });
  }
}

// ── POST — write rows to a tab ─────────────────────────────────
function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);

    if (!VALID_SECRETS.has(data.secret)) {
      return json({ ok: false, error: 'Unauthorized' });
    }

    // action=update_rows → update Name/Comment for specific rows (admin only)
    if (data.action === 'update_rows') {
      if (data.secret !== ADMIN_SECRET) return json({ ok: false, error: 'Admin access required.' });
      let updated = 0;
      for (const change of (data.changes || [])) {
        if (updateRow(change.sheetName, change.time, change.name, change.comment)) updated++;
      }
      cacheClear();
      return json({ ok: true, updated });
    }

    // action=backfill_youtube → write YouTube links into a game tab
    if (data.action === 'backfill_youtube') {
      if (data.secret !== ADMIN_SECRET) return json({ ok: false, error: 'Admin access required.' });
      const updated = backfillYoutubeLinks(data.sheetName, data.youtubeUrl, Number(data.offsetSeconds) || 0);
      cacheClear();
      return json({ ok: true, updated });
    }

    // action=live_update → upsert a row in _live
    if (data.action === 'live_update') {
      writeLiveRow(data.sheetName, data.team1, data.team2, data.score1, data.score2, data.timeSeconds);
      return json({ ok: true });
    }

    // action=live_clear → remove a row from _live
    if (data.action === 'live_clear') {
      clearLiveRow(data.sheetName);
      return json({ ok: true });
    }

    const ss    = SpreadsheetApp.openById(SHEET_ID);
    const sheet = ss.getSheetByName(data.sheetName);

    if (sheet) {
      if (!data.override) {
        return json({ ok: false, error: `Tab "${data.sheetName}" already exists.`, tabExists: true });
      }
      if (data.secret !== ADMIN_SECRET) {
        return json({ ok: false, error: 'Admin access required to override.' });
      }
      ss.deleteSheet(sheet);
    }

    const newSheet = ss.insertSheet(data.sheetName);
    newSheet.appendRow(HEADERS);

    const rows = data.rows;
    if (rows && rows.length > 0) {
      newSheet.getRange(newSheet.getLastRow() + 1, 1, rows.length, rows[0].length)
              .setValues(rows);
    }

    if (data.meta) writeMeta(data.sheetName, data.meta);

    cacheClear(); // invalidate list + all caches after any write

    return json({ ok: true, appended: rows ? rows.length : 0 });

  } catch (err) {
    return json({ ok: false, error: err.toString() });
  }
}

// ── Write / upsert a row in _metadata ──────────────────────────
// If a row for sheetName already exists → overwrite it.
// If not → append to the bottom. No extra password needed here
// because the caller has already passed the override check above.
function writeMeta(sheetName, meta) {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  let sheet = ss.getSheetByName(METADATA_SHEET);
  if (!sheet) {
    sheet = ss.insertSheet(METADATA_SHEET);
    sheet.appendRow(['Sheet Name', 'Team 1', 'Team 2', 'Competition', 'Year', 'Division', 'Video Name', 'Analyzable', 'Youtube Link']);
  }

  const values  = sheet.getDataRange().getValues();
  const headers = values[0].map(h => String(h).toLowerCase().trim());
  const col     = name => headers.indexOf(name); // returns -1 if missing

  const ni   = col('sheet name');
  const t1i  = col('team 1');
  const t2i  = col('team 2');
  const ci   = col('competition');
  const yi   = col('year');
  const di   = col('division');
  const vi   = col('video name');
  const ai   = col('analyzable');
  const yui  = col('youtube link');

  // Find existing row, if any
  const numCols = headers.length;
  let existingRow = -1;
  let existingData = null;
  for (let i = 1; i < values.length; i++) {
    if (ni >= 0 && values[i][ni] === sheetName) { existingRow = i + 1; existingData = values[i]; break; }
  }

  // Start from existing data (to preserve fields not in this update) or a blank row
  const row = existingData ? existingData.map(String) : Array(numCols).fill('');
  while (row.length < numCols) row.push('');

  // Only overwrite fields that are explicitly provided in meta
  if (ni  >= 0)                             row[ni]  = sheetName;
  if (t1i >= 0 && meta.team1       != null) row[t1i] = meta.team1;
  if (t2i >= 0 && meta.team2       != null) row[t2i] = meta.team2;
  if (ci  >= 0 && meta.competition != null) row[ci]  = meta.competition;
  if (yi  >= 0 && meta.year        != null) row[yi]  = meta.year;
  if (di  >= 0 && meta.division    != null) row[di]  = meta.division;
  if (vi  >= 0 && meta.video       != null) row[vi]  = meta.video;
  if (ai  >= 0 && meta.analyzable  != null) row[ai]  = meta.analyzable;
  if (yui >= 0 && meta.youtubelink != null) row[yui] = meta.youtubelink;

  if (existingRow > 0) {
    sheet.getRange(existingRow, 1, 1, numCols).setValues([row]);
  } else {
    sheet.appendRow(row);
  }
}

// ── One-time utility: create/backfill _metadata from tab names ─
// Run this once from the Apps Script editor (not exposed via HTTP).
// Tab name format expected: YEAR_DIVISION_COMPETITION_TEAM1_TEAM2
// e.g. 2025_m30_seniorscup_france_england
function backfillMetadata() {
  const ss     = SpreadsheetApp.openById(SHEET_ID);
  const sheets = ss.getSheets()
    .map(s => s.getName())
    .filter(n => n !== METADATA_SHEET);

  // Delete and recreate _metadata sheet
  const existing = ss.getSheetByName(METADATA_SHEET);
  if (existing) ss.deleteSheet(existing);
  const meta = ss.insertSheet(METADATA_SHEET);

  const header = ['Sheet Name', 'Team 1', 'Team 2', 'Competition', 'Year', 'Division', 'Video Name'];
  meta.appendRow(header);

  sheets.forEach(name => {
    // Split on underscores; last two parts are team names, first is year,
    // second is division, everything in between is competition.
    const parts = name.split('_');
    if (parts.length < 5) {
      // Can't parse — add row with sheet name only so it's visible for manual fill
      meta.appendRow([name, '', '', '', '', '']);
      return;
    }
    const year        = parts[0];
    const division    = parts[1];
    const team1       = parts[parts.length - 2];
    const team2       = parts[parts.length - 1];
    const competition = parts.slice(2, parts.length - 2).join('_');
    meta.appendRow([name, team1, team2, competition, year, division]);
  });

  Logger.log('_metadata backfilled with ' + sheets.length + ' games.');
}

// ── Live game state helpers ─────────────────────────────────────
function writeLiveRow(sheetName, team1, team2, score1, score2, timeSeconds) {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  let sheet = ss.getSheetByName(LIVE_SHEET);
  if (!sheet) {
    sheet = ss.insertSheet(LIVE_SHEET);
    sheet.appendRow(['Sheet Name', 'Team 1', 'Team 2', 'Score 1', 'Score 2', 'Time Seconds', 'Updated At']);
  }

  const now    = new Date().toISOString();
  const newRow = [sheetName, team1 || '', team2 || '', score1 || 0, score2 || 0, timeSeconds || 0, now];
  const values = sheet.getDataRange().getValues();

  for (let i = 1; i < values.length; i++) {
    if (String(values[i][0]) === sheetName) {
      sheet.getRange(i + 1, 1, 1, newRow.length).setValues([newRow]);
      return;
    }
  }
  sheet.appendRow(newRow);
}

function clearLiveRow(sheetName) {
  const ss    = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheetByName(LIVE_SHEET);
  if (!sheet) return;
  const values = sheet.getDataRange().getValues();
  for (let i = 1; i < values.length; i++) {
    if (String(values[i][0]) === sheetName) {
      sheet.deleteRow(i + 1);
      return;
    }
  }
}

// ── Update Name/Comment on a specific row ──────────────────────
function updateRow(sheetName, time, name, comment) {
  const ss    = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheetByName(sheetName);
  if (!sheet) return false;

  const values  = sheet.getDataRange().getValues();
  const headers = values[0].map(h => String(h).toLowerCase().trim());
  const timeIdx    = headers.indexOf('time');
  const nameIdx    = headers.indexOf('name');
  const commentIdx = headers.indexOf('comment');
  if (timeIdx < 0) return false;

  for (let i = 1; i < values.length; i++) {
    if (String(values[i][timeIdx]) === String(time)) {
      if (nameIdx    >= 0 && name    !== undefined) sheet.getRange(i + 1, nameIdx    + 1).setValue(name);
      if (commentIdx >= 0 && comment !== undefined) sheet.getRange(i + 1, commentIdx + 1).setValue(comment);
      return true;
    }
  }
  return false;
}

// ── Write YouTube link into _metadata for a game ───────────────
// offsetSeconds: if non-zero, shifts every Time value in the sheet by this amount.
function backfillYoutubeLinks(sheetName, youtubeUrl, offsetSeconds) {
  const m = String(youtubeUrl).match(/(?:v=|youtu\.be\/|embed\/|live\/)([A-Za-z0-9_-]{11})/);
  if (!m) throw new Error('Invalid YouTube URL — could not extract video ID.');

  const ss    = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheetByName(sheetName);
  if (!sheet) throw new Error(`Tab "${sheetName}" not found.`);

  writeMeta(sheetName, { youtubelink: youtubeUrl });

  if (offsetSeconds && offsetSeconds !== 0) {
    const values  = sheet.getDataRange().getDisplayValues();
    if (values.length < 2) return 1;
    const headers = values[0].map(s => s.toLowerCase().trim());
    const timeIdx = headers.indexOf('time');
    if (timeIdx < 0) return 1;

    const newTimes = [];
    for (let i = 1; i < values.length; i++) {
      const timeStr = values[i][timeIdx];
      const parts   = timeStr ? timeStr.split(':').map(Number) : [];
      let secs;
      if      (parts.length === 3) secs = parts[0]*3600 + parts[1]*60 + parts[2];
      else if (parts.length === 2) secs = parts[0]*60   + parts[1];
      else { newTimes.push([timeStr]); continue; }

      const s   = Math.max(0, secs + offsetSeconds);
      const h   = Math.floor(s / 3600);
      const min = Math.floor((s % 3600) / 60);
      const sec = Math.floor(s % 60);
      newTimes.push([`${h}:${String(min).padStart(2,'0')}:${String(sec).padStart(2,'0')}`]);
    }

    sheet.getRange(2, timeIdx + 1, newTimes.length, 1).setValues(newTimes);
  }

  return 1;
}

function json(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
