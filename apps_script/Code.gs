// ── Configuration ──────────────────────────────────────────────
const SHEET_ID        = '1B9DwThGoINgicevtjoUDkeQBoGBCD6QFjL0oL7mAU-k';
const SECRET          = 'm30-test-key';
const OVERRIDE_PASS   = 'm30-admin';
const METADATA_SHEET  = '_metadata';  // reserved tab name for game metadata

// Expected column order (must match what the Python pipeline reads)
const HEADERS = ['Time', 'Possession Owner', 'Type', 'Name', 'Video Name', 'To Review', 'Comment', 'Youtube Link'];

// Metadata columns appended to action=all rows
const META_COLS = ['Team 1', 'Team 2', 'Competition', 'Year', 'Division'];

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
    };
  });
  return meta;
}

// ── GET — list sheets or fetch rows from a tab ─────────────────
function doGet(e) {
  try {
    if (e.parameter.secret !== SECRET) {
      return json({ ok: false, error: 'Unauthorized' });
    }

    // action=list → sheet names + metadata for each game
    if (e.parameter.action === 'list') {
      const meta   = getMetadata();
      const sheets = SpreadsheetApp.openById(SHEET_ID).getSheets()
        .map(s => s.getName())
        .filter(n => n !== METADATA_SHEET);
      return json({ ok: true, sheets: sheets.map(name => ({ name, ...(meta[name] || {}) })) });
    }

    // action=all → every row from every game sheet, metadata columns appended
    if (e.parameter.action === 'all') {
      const meta   = getMetadata();
      const ss     = SpreadsheetApp.openById(SHEET_ID);
      const sheets = ss.getSheets().filter(s => s.getName() !== METADATA_SHEET);
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
        const metaValues = [m.team1 || '', m.team2 || '', m.competition || '', m.year || '', m.division || ''];
        values.slice(1).forEach(row => allRows.push([...row, name, ...metaValues]));
      }

      return json({ ok: true, rows: allRows });
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

    if (data.secret !== SECRET) {
      return json({ ok: false, error: 'Unauthorized' });
    }

    const ss    = SpreadsheetApp.openById(SHEET_ID);
    const sheet = ss.getSheetByName(data.sheetName);

    if (sheet) {
      if (!data.overridePassword) {
        return json({ ok: false, error: `Tab "${data.sheetName}" already exists.`, tabExists: true });
      }
      if (data.overridePassword !== OVERRIDE_PASS) {
        return json({ ok: false, error: 'Incorrect override password.' });
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
    sheet.appendRow(['Team 1', 'Team 2', 'Competition', 'Year', 'Division', 'Sheet Name']);
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

  // Build a row the same width as the header, filling known columns by position
  const numCols = headers.length;
  const makeRow = () => Array(numCols).fill('');
  const row = makeRow();
  if (ni  >= 0) row[ni]  = sheetName;
  if (t1i >= 0) row[t1i] = meta.team1       || '';
  if (t2i >= 0) row[t2i] = meta.team2       || '';
  if (ci  >= 0) row[ci]  = meta.competition || '';
  if (yi  >= 0) row[yi]  = meta.year        || '';
  if (di  >= 0) row[di]  = meta.division    || '';

  let existingRow = -1;
  for (let i = 1; i < values.length; i++) {
    if (ni >= 0 && values[i][ni] === sheetName) { existingRow = i + 1; break; } // 1-indexed
  }

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

  const header = ['Sheet Name', 'Team 1', 'Team 2', 'Competition', 'Year', 'Division'];
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

function json(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
