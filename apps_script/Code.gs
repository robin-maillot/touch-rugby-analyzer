// ── Configuration ──────────────────────────────────────────────
// Replace these two values before deploying.
const SHEET_ID = '1B9DwThGoINgicevtjoUDkeQBoGBCD6QFjL0oL7mAU-k';  // e.g. '1B9DwThGoINgicevtjoUDkeQBoGBCD6QFjL0oL7mAU-k'
const SECRET   = 'm30-test-key';    // any shared passphrase you choose

// Expected column order (must match what the Python pipeline reads)
const HEADERS = ['Time', 'Possession Owner', 'Type', 'Name', 'Video Name', 'To Review', 'Comment', 'Youtube Link'];

// ── Entry point ────────────────────────────────────────────────
function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);

    if (data.secret !== SECRET) {
      return json({ ok: false, error: 'Unauthorized' });
    }

    const ss    = SpreadsheetApp.openById(SHEET_ID);
    const sheet = ss.getSheetByName(data.sheetName);

    if (sheet) {
      return json({ ok: false, error: `Tab "${data.sheetName}" already exists. Rename it or choose a different tab name.` });
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
