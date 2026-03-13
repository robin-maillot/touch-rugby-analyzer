// ── Configuration ──────────────────────────────────────────────
const SHEET_ID        = '1B9DwThGoINgicevtjoUDkeQBoGBCD6QFjL0oL7mAU-k';
const SECRET          = 'm30-test-key';   // shared key for all read/write access
const OVERRIDE_PASS   = 'm30-admin';      // extra password required to overwrite an existing tab

// Expected column order (must match what the Python pipeline reads)
const HEADERS = ['Time', 'Possession Owner', 'Type', 'Name', 'Video Name', 'To Review', 'Comment', 'Youtube Link'];

// ── GET — fetch rows from an existing tab ──────────────────────
function doGet(e) {
  try {
    if (e.parameter.secret !== SECRET) {
      return json({ ok: false, error: 'Unauthorized' });
    }

    const sheetName = e.parameter.sheetName;
    if (!sheetName) {
      return json({ ok: false, error: 'Missing sheetName parameter' });
    }

    const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(sheetName);
    if (!sheet) {
      return json({ ok: false, error: `Tab "${sheetName}" not found.` });
    }

    const values = sheet.getDataRange().getValues();
    // values[0] is the header row — return all rows so the client can parse
    return json({ ok: true, rows: values });

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
      // Tab exists — only allow overwrite with the admin password
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

    return json({ ok: true, appended: rows ? rows.length : 0 });

  } catch (err) {
    return json({ ok: false, error: err.toString() });
  }
}

function json(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
