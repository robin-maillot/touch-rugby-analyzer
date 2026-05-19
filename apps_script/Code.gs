// ── Configuration ──────────────────────────────────────────────
const SHEET_ID        = '1B9DwThGoINgicevtjoUDkeQBoGBCD6QFjL0oL7mAU-k';
const VALID_SECRETS   = new Set(['m30', 'm30-admin', 'm30-staff']);
const ADMIN_SECRET    = 'm30-admin';
const METADATA_SHEET  = '_metadata';  // reserved tab name for game metadata
const LIVE_SHEET      = '_live';      // reserved tab name for live game state

// Expected column order (must match what the Python pipeline reads)
const HEADERS = ['Time', 'Possession Owner', 'Type', 'Name', 'To Review', 'Comment', 'Action Owner'];

// Metadata columns appended to action=all rows
const META_COLS = ['Team 1', 'Team 2', 'Competition', 'Year', 'Division', 'Video Name', 'Analyzable', 'ID'];

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
  const idi  = col('id');
  const vpi  = col('video provider');  // 'youtube' | 'stream' (optional override)
  const sui  = col('stream uid');      // Cloudflare Stream video UID
  const ssi  = col('stream signed');   // '1' / 'true' if signed URLs required

  const meta = {};
  values.slice(1).forEach(row => {
    const name = row[ni];
    if (!name) return;
    const youtubelink = yui >= 0 ? row[yui] : '';
    const streamuid   = sui >= 0 ? row[sui] : '';
    let provider = vpi >= 0 ? String(row[vpi] || '').toLowerCase().trim() : '';
    if (provider !== 'youtube' && provider !== 'stream') {
      provider = streamuid ? 'stream' : (youtubelink ? 'youtube' : '');
    }
    const signedRaw = ssi >= 0 ? String(row[ssi] || '').toLowerCase().trim() : '';
    const streamsigned = signedRaw === '1' || signedRaw === 'true' || signedRaw === 'yes';
    meta[name] = {
      team1:         t1i >= 0 ? row[t1i] : '',
      team2:         t2i >= 0 ? row[t2i] : '',
      competition:   ci  >= 0 ? row[ci]  : '',
      year:          yi  >= 0 ? row[yi]  : '',
      division:      di  >= 0 ? row[di]  : '',
      video:         vi  >= 0 ? row[vi]  : '',
      analyzable:    ai  >= 0 ? row[ai]  : '',
      youtubelink:   youtubelink,
      id:            idi >= 0 ? row[idi] : '',
      videoprovider: provider,
      streamuid:     streamuid,
      streamsigned:  streamsigned,
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

// Spreadsheet-wide last-modified timestamp (ms since epoch). Drive updates this
// on every cell change, so it's a reliable "did anything change" signal.
// Must be read BEFORE the data it stamps — see doGet handlers.
function getSheetVersion() {
  try { return DriveApp.getFileById(SHEET_ID).getLastUpdated().getTime(); }
  catch (e) { return 0; }
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

    // action=version → live spreadsheet-modified timestamp (never cached server-side).
    // Clients use this to skip the heavy action=all / action=list refetches when nothing changed.
    if (e.parameter.action === 'version') {
      return json({ ok: true, version: getSheetVersion() });
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
      const po1i = col('poss 1'),  po2i = col('poss 2');
      const cm1i = col('comps 1'), cm2i = col('comps 2');
      const fini = col('finished');
      const trji = col('tries json');
      const games = values.slice(1)
        .map(row => ({
          name:        row[ni]  || '',
          team1:       t1i  >= 0 ? row[t1i]  : '',
          team2:       t2i  >= 0 ? row[t2i]  : '',
          score1:      s1i  >= 0 ? Number(row[s1i])  : 0,
          score2:      s2i  >= 0 ? Number(row[s2i])  : 0,
          timeSeconds: tsi  >= 0 ? Number(row[tsi])  : 0,
          updatedAt:   uai  >= 0 ? row[uai]  : '',
          poss1:       po1i >= 0 ? Number(row[po1i]) : 0,
          poss2:       po2i >= 0 ? Number(row[po2i]) : 0,
          comps1:      cm1i >= 0 ? Number(row[cm1i]) : 0,
          comps2:      cm2i >= 0 ? Number(row[cm2i]) : 0,
          finished:    fini >= 0 ? row[fini] === 'true' : false,
          tries:       trji >= 0 && row[trji] ? (() => { try { return JSON.parse(row[trji]); } catch(e) { return []; } })() : [],
        }))
        .filter(g => g.name);
      return json({ ok: true, games });
    }

    // action=list → sheet names + metadata for each game
    if (e.parameter.action === 'list') {
      const hit = cacheGet('list');
      if (hit) return rawJson(hit);

      const version = getSheetVersion();   // capture BEFORE reading data
      const meta    = getMetadata();
      const result  = JSON.stringify({ ok: true, version, sheets: Object.entries(meta).map(([name, m]) => ({ name, ...m })) });
      cachePut('list', result);
      return rawJson(result);
    }

    // action=all → every row from every game sheet listed in _metadata, metadata columns appended
    if (e.parameter.action === 'all') {
      const hit = cacheGet('all');
      if (hit) return rawJson(hit);

      const version = getSheetVersion();   // capture BEFORE reading data
      const meta    = getMetadata();
      const ss     = SpreadsheetApp.openById(SHEET_ID);
      const metaNames = new Set(Object.keys(meta));
      const sheets = ss.getSheets().filter(s => metaNames.has(s.getName()));
      const allRows = [];

      // Always emit a canonical header regardless of each sheet's column order
      allRows.push([...HEADERS, 'Game', ...META_COLS]);

      for (const sheet of sheets) {
        const values = sheet.getDataRange().getDisplayValues();
        if (values.length < 2) continue;

        // Map each canonical header to its index in this sheet (by name, case-insensitive)
        const sheetHeaders = values[0].map(h => h.trim().toLowerCase());
        const colIndices   = HEADERS.map(h => sheetHeaders.indexOf(h.toLowerCase()));

        const name = sheet.getName();
        const m    = meta[name] || {};
        const metaValues = [m.team1 || '', m.team2 || '', m.competition || '', m.year || '', m.division || '', m.video || '', m.analyzable || '', m.id || ''];
        values.slice(1).forEach(row => {
          const mapped = colIndices.map(i => i >= 0 ? row[i] : '');
          allRows.push([...mapped, name, ...metaValues]);
        });
      }

      const result = JSON.stringify({ ok: true, version, rows: allRows });
      cachePut('all', result);
      return rawJson(result);
    }

    // action=stream_token → mint a short-lived signed JWT for a private
    // Cloudflare Stream video. Only meaningful when the game's metadata has
    // streamsigned=1; for public Stream videos the frontend uses the raw UID.
    if (e.parameter.action === 'stream_token') {
      const name = e.parameter.sheetName;
      if (!name) return json({ ok: false, error: 'Missing sheetName parameter' });
      const meta = getMetadata()[name];
      if (!meta || !meta.streamuid) return json({ ok: false, error: 'No Stream UID configured for ' + name });
      if (!meta.streamsigned)        return json({ ok: false, error: 'This video is public — use the raw UID' });
      try {
        const token = mintStreamToken(meta.streamuid, 2 * 60 * 60); // 2h
        return json({ ok: true, token, expiresAt: Math.floor(Date.now() / 1000) + 2 * 60 * 60 });
      } catch (err) {
        return json({ ok: false, error: err.toString() });
      }
    }

    // action=stream_upload_url → admin-only: request a one-time Cloudflare
    // direct creator upload URL. The browser uploads straight to Cloudflare;
    // the file never touches Apps Script (which has a 6-min execution cap).
    if (e.parameter.action === 'stream_upload_url') {
      if (e.parameter.secret !== ADMIN_SECRET) return json({ ok: false, error: 'Admin access required.' });
      try {
        const maxSec = Math.min(21600, Math.max(60, Number(e.parameter.maxDurationSeconds) || 7200));
        const result = requestStreamUpload(maxSec);
        return json({ ok: true, uploadURL: result.uploadURL, uid: result.uid });
      } catch (err) {
        return json({ ok: false, error: err.toString() });
      }
    }

    // action=stream_clip_status → poll the encoding + MP4-download state of a
    // clip we previously created. Drives the "Download clip" button's spinner.
    // Side effect: once the clip is ready-to-stream we auto-trigger the
    // /downloads endpoint so the caller doesn't need a second action.
    if (e.parameter.action === 'stream_clip_status') {
      const clipUid = e.parameter.clipUid;
      if (!clipUid) return json({ ok: false, error: 'Missing clipUid parameter' });
      try {
        return json({ ok: true, ...getStreamClipStatus(clipUid) });
      } catch (err) {
        return json({ ok: false, error: err.toString() });
      }
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
    const version = getSheetVersion();   // capture BEFORE reading data
    const values  = sheet.getDataRange().getDisplayValues();
    const meta    = getMetadata();
    return json({ ok: true, version, rows: values, meta: meta[sheetName] || {} });

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
      const updated = backfillVideoLink(data.sheetName, 'youtube', data.youtubeUrl, false, Number(data.offsetSeconds) || 0);
      cacheClear();
      return json({ ok: true, updated });
    }

    // action=backfill_video → attach either a YouTube URL or a Cloudflare
    // Stream UID to a game, with optional time-offset. Replaces backfill_youtube.
    if (data.action === 'backfill_video') {
      if (data.secret !== ADMIN_SECRET) return json({ ok: false, error: 'Admin access required.' });
      const provider = String(data.provider || '').toLowerCase();
      if (provider !== 'youtube' && provider !== 'stream') {
        return json({ ok: false, error: 'provider must be "youtube" or "stream"' });
      }
      const updated = backfillVideoLink(
        data.sheetName,
        provider,
        data.value,
        !!data.signed,
        Number(data.offsetSeconds) || 0
      );
      cacheClear();
      return json({ ok: true, updated });
    }

    // action=live_update → upsert a row in _live
    if (data.action === 'live_update') {
      writeLiveRow(data.sheetName, data.team1, data.team2, data.score1, data.score2, data.timeSeconds, data.poss1, data.poss2, data.comps1, data.comps2, data.triesJson);
      return json({ ok: true });
    }

    // action=live_clear → remove a row from _live
    if (data.action === 'live_clear') {
      clearLiveRow(data.sheetName);
      return json({ ok: true });
    }

    // action=stream_clip → cut a clip out of a Cloudflare Stream video and
    // return the new clip's UID. Polling stream_clip_status follows.
    if (data.action === 'stream_clip') {
      const sheetName = data.sheetName;
      if (!sheetName) return json({ ok: false, error: 'Missing sheetName parameter' });
      const meta = getMetadata()[sheetName];
      if (!meta || !meta.streamuid) return json({ ok: false, error: 'No Stream UID configured for ' + sheetName });
      const start = Number(data.startSeconds);
      const end   = Number(data.endSeconds);
      if (!isFinite(start) || !isFinite(end) || end <= start) {
        return json({ ok: false, error: 'startSeconds and endSeconds required, end > start.' });
      }
      try {
        const clipUid = createStreamClip(meta.streamuid, start, end, !!meta.streamsigned);
        return json({ ok: true, clipUid });
      } catch (err) {
        return json({ ok: false, error: err.toString() });
      }
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
    sheet.appendRow(['Sheet Name', 'Team 1', 'Team 2', 'Competition', 'Year', 'Division', 'Video Name', 'Analyzable', 'Youtube Link', 'ID']);
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
  let yui    = col('youtube link');
  let idi    = col('id');
  let vpi    = col('video provider');
  let sui    = col('stream uid');
  let ssi    = col('stream signed');

  // Add 'Youtube Link' column if missing and we have a value to write
  if (yui < 0 && meta.youtubelink != null) {
    sheet.getRange(1, headers.length + 1).setValue('Youtube Link');
    headers.push('youtube link');
    yui = headers.length - 1;
    // Refresh values to include the new column in subsequent reads
    values.forEach(row => { while (row.length < headers.length) row.push(''); });
  }

  // Add 'ID' column if missing and we have a value to write
  if (idi < 0 && meta.id != null) {
    sheet.getRange(1, headers.length + 1).setValue('ID');
    headers.push('id');
    idi = headers.length - 1;
    values.forEach(row => { while (row.length < headers.length) row.push(''); });
  }

  // Add Cloudflare Stream columns lazily — only when first written to.
  if (vpi < 0 && meta.videoprovider != null) {
    sheet.getRange(1, headers.length + 1).setValue('Video Provider');
    headers.push('video provider');
    vpi = headers.length - 1;
    values.forEach(row => { while (row.length < headers.length) row.push(''); });
  }
  if (sui < 0 && meta.streamuid != null) {
    sheet.getRange(1, headers.length + 1).setValue('Stream UID');
    headers.push('stream uid');
    sui = headers.length - 1;
    values.forEach(row => { while (row.length < headers.length) row.push(''); });
  }
  if (ssi < 0 && meta.streamsigned != null) {
    sheet.getRange(1, headers.length + 1).setValue('Stream Signed');
    headers.push('stream signed');
    ssi = headers.length - 1;
    values.forEach(row => { while (row.length < headers.length) row.push(''); });
  }

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
  if (idi >= 0 && meta.id          != null) row[idi] = meta.id;
  if (vpi >= 0 && meta.videoprovider != null) row[vpi] = meta.videoprovider;
  if (sui >= 0 && meta.streamuid     != null) row[sui] = meta.streamuid;
  if (ssi >= 0 && meta.streamsigned  != null) row[ssi] = meta.streamsigned ? '1' : '';

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
const LIVE_HEADERS = ['Sheet Name', 'Team 1', 'Team 2', 'Score 1', 'Score 2', 'Time Seconds', 'Updated At', 'Poss 1', 'Poss 2', 'Comps 1', 'Comps 2', 'Finished', 'Tries JSON'];

function writeLiveRow(sheetName, team1, team2, score1, score2, timeSeconds, poss1, poss2, comps1, comps2, triesJson) {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  let sheet = ss.getSheetByName(LIVE_SHEET);

  if (!sheet) {
    sheet = ss.insertSheet(LIVE_SHEET);
    sheet.appendRow(LIVE_HEADERS);
  } else if (sheet.getLastColumn() < LIVE_HEADERS.length) {
    const existing = sheet.getLastColumn();
    sheet.getRange(1, existing + 1, 1, LIVE_HEADERS.length - existing).setValues([LIVE_HEADERS.slice(existing)]);
  }

  // Clean up rows older than 1 week
  const ONE_WEEK = 7 * 24 * 60 * 60 * 1000;
  const now = Date.now();
  const allVals = sheet.getDataRange().getValues();
  const hdrs = allVals[0].map(h => String(h).toLowerCase().trim());
  const uaIdx = hdrs.indexOf('updated at');
  if (uaIdx >= 0) {
    for (let i = allVals.length - 1; i >= 1; i--) {
      const ua = allVals[i][uaIdx];
      if (ua && (now - new Date(ua).getTime()) > ONE_WEEK) sheet.deleteRow(i + 1);
    }
  }

  const nowStr = new Date().toISOString();
  const newRow = [sheetName, team1 || '', team2 || '', score1 || 0, score2 || 0, timeSeconds || 0, nowStr, poss1 || 0, poss2 || 0, comps1 || 0, comps2 || 0, '', triesJson || '[]'];
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
  const hdrs   = values[0].map(h => String(h).toLowerCase().trim());
  const fi     = hdrs.indexOf('finished');
  const uai    = hdrs.indexOf('updated at');
  for (let i = 1; i < values.length; i++) {
    if (String(values[i][0]) === sheetName) {
      if (fi  >= 0) sheet.getRange(i + 1, fi  + 1).setValue('true');
      if (uai >= 0) sheet.getRange(i + 1, uai + 1).setValue(new Date().toISOString());
      return;
    }
  }
}

// ── Update Name/Comment on a specific row ──────────────────────
function updateRow(sheetName, time, name, comment) {
  const ss    = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheetByName(sheetName);
  if (!sheet) return false;

  const values  = sheet.getDataRange().getDisplayValues();
  const headers = values[0].map(h => h.toLowerCase().trim());
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

// ── Attach a video (YouTube URL or Cloudflare Stream UID) to a game ──
// provider: 'youtube' | 'stream'
// value:    the YouTube URL OR the Stream UID (32-char hex)
// signed:   only meaningful for stream — sets the Stream Signed flag
// offsetSeconds: if non-zero, shifts every Time value in the game tab by
//                this amount so the sheet aligns with the video timeline.
function backfillVideoLink(sheetName, provider, value, signed, offsetSeconds) {
  const ss    = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheetByName(sheetName);
  if (!sheet) throw new Error(`Tab "${sheetName}" not found.`);

  const meta = { videoprovider: provider };
  if (provider === 'youtube') {
    if (!String(value || '').match(/(?:v=|youtu\.be\/|embed\/|live\/)([A-Za-z0-9_-]{11})/)) {
      throw new Error('Invalid YouTube URL — could not extract video ID.');
    }
    meta.youtubelink = value;
  } else if (provider === 'stream') {
    const uid = String(value || '').trim();
    if (!/^[a-f0-9]{32}$/i.test(uid)) {
      throw new Error('Invalid Stream UID — expected 32 hex characters.');
    }
    meta.streamuid    = uid;
    meta.streamsigned = !!signed;
  } else {
    throw new Error('Unknown provider: ' + provider);
  }
  writeMeta(sheetName, meta);

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

// Run this once from the Apps Script editor (Run ▶) to trigger the OAuth
// consent prompt for UrlFetchApp. Without this, the deployed web app gets:
//   "Vous n'êtes pas autorisé à appeler UrlFetchApp.fetch."
// After authorizing once, every action that calls Cloudflare will work.
function authorizeUrlFetch() {
  UrlFetchApp.fetch('https://api.cloudflare.com/client/v4/user/tokens/verify', {
    headers: { Authorization: 'Bearer ' + (PropertiesService.getScriptProperties().getProperty('CF_API_TOKEN') || '') },
    muteHttpExceptions: true,
  });
  Logger.log('UrlFetchApp authorized.');
}

// ── Cloudflare Stream integration ──────────────────────────────
// All Cloudflare credentials live in Apps Script Script Properties so they
// stay out of source control. Set them from the Apps Script editor:
//   File → Project Settings → Script Properties
// Required keys:
//   CF_ACCOUNT_ID       — Cloudflare account ID (for upload-url endpoint)
//   CF_API_TOKEN        — API token with Stream:Edit scope (for upload-url)
//   CF_STREAM_KEY_ID    — `id` returned by POST /stream/keys (the kid)
//   CF_STREAM_KEY_PEM   — the PEM private key, base64-DECODED from `pem`.
//                         Should include the BEGIN/END PRIVATE KEY lines.
function cfProps_() {
  const p = PropertiesService.getScriptProperties();
  return {
    accountId: p.getProperty('CF_ACCOUNT_ID') || '',
    apiToken:  p.getProperty('CF_API_TOKEN')  || '',
    keyId:     p.getProperty('CF_STREAM_KEY_ID')  || '',
    keyPem:    p.getProperty('CF_STREAM_KEY_PEM') || '',
  };
}

function base64Url_(bytes) {
  // Utilities.base64EncodeWebSafe pads with '=' which JWT spec forbids.
  return Utilities.base64EncodeWebSafe(bytes).replace(/=+$/, '');
}

// Mint a signed JWT for a Stream video UID. lifetimeSeconds defaults to 2h.
// Returns a JWT suitable for use as the path segment in
// https://iframe.videodelivery.net/<token>.
function mintStreamToken(uid, lifetimeSeconds) {
  const { keyId, keyPem } = cfProps_();
  if (!keyId)  throw new Error('CF_STREAM_KEY_ID is not set in Script Properties.');
  if (!keyPem) throw new Error('CF_STREAM_KEY_PEM is not set in Script Properties.');
  const now = Math.floor(Date.now() / 1000);
  const exp = now + (Number(lifetimeSeconds) || 7200);
  const header  = { alg: 'RS256', kid: keyId, typ: 'JWT' };
  const payload = { sub: uid, kid: keyId, exp: exp, nbf: now - 30 };
  const headerB64  = base64Url_(Utilities.newBlob(JSON.stringify(header)).getBytes());
  const payloadB64 = base64Url_(Utilities.newBlob(JSON.stringify(payload)).getBytes());
  const signingInput = headerB64 + '.' + payloadB64;
  const sigBytes = Utilities.computeRsaSha256Signature(signingInput, keyPem);
  return signingInput + '.' + base64Url_(sigBytes);
}

// Create a clip out of an existing Stream video. Returns the new clip's UID.
// requireSignedURLs is inherited from the source video so a private match
// produces a private clip.
function createStreamClip(sourceUid, startSeconds, endSeconds, requireSignedURLs) {
  const { accountId, apiToken } = cfProps_();
  if (!accountId) throw new Error('CF_ACCOUNT_ID is not set in Script Properties.');
  if (!apiToken)  throw new Error('CF_API_TOKEN is not set in Script Properties.');
  const resp = UrlFetchApp.fetch(
    'https://api.cloudflare.com/client/v4/accounts/' + accountId + '/stream/clip',
    {
      method: 'post',
      headers: { Authorization: 'Bearer ' + apiToken },
      contentType: 'application/json',
      payload: JSON.stringify({
        clippedFromVideoUID: sourceUid,
        startTimeSeconds:    Math.floor(startSeconds),
        endTimeSeconds:      Math.ceil(endSeconds),
        requireSignedURLs:   !!requireSignedURLs,
      }),
      muteHttpExceptions: true,
    }
  );
  const body = JSON.parse(resp.getContentText() || '{}');
  // Cloudflare returns the result at top level for /clip (not wrapped in {success, result}).
  if (body.uid) return body.uid;
  if (body.success && body.result && body.result.uid) return body.result.uid;
  const msg = (body.errors && body.errors.map(e => e.message).join('; ')) || resp.getContentText();
  throw new Error('Cloudflare /stream/clip failed: ' + msg);
}

// Poll the state of a clip we previously created. State machine:
//   'clipping'       — Cloudflare is still encoding the clip from the source
//   'mp4-encoding'   — clip is ready; /downloads has been triggered, MP4 is rendering
//   'ready'          — MP4 is available; downloadUrl is returned
//   'error'          — something failed; message in `error`
//
// `cfState` carries the raw Cloudflare sub-state ('queued', 'inprogress', ...)
// so the UI can show something useful when pctComplete isn't populated yet
// (Cloudflare often leaves it null during the queued phase).
function getStreamClipStatus(clipUid) {
  const { accountId, apiToken } = cfProps_();
  const auth = { Authorization: 'Bearer ' + apiToken };

  // 1. Check the clip's own encoding state.
  const vidResp = UrlFetchApp.fetch(
    'https://api.cloudflare.com/client/v4/accounts/' + accountId + '/stream/' + clipUid,
    { method: 'get', headers: auth, muteHttpExceptions: true }
  );
  const vidBody = JSON.parse(vidResp.getContentText() || '{}');
  if (!vidBody.success) {
    const msg = (vidBody.errors && vidBody.errors.map(e => e.message).join('; ')) || 'unknown';
    return { state: 'error', error: msg };
  }
  const result = vidBody.result || {};
  if (!result.readyToStream) {
    const s = result.status || {};
    return {
      state:   'clipping',
      cfState: s.state || 'queued',
      percent: parsePct_(s.pctComplete),
    };
  }

  // 2. Clip is encoded — check (or kick off) the MP4 download render.
  const dlGet = UrlFetchApp.fetch(
    'https://api.cloudflare.com/client/v4/accounts/' + accountId + '/stream/' + clipUid + '/downloads',
    { method: 'get', headers: auth, muteHttpExceptions: true }
  );
  const dlBody = JSON.parse(dlGet.getContentText() || '{}');
  let def = dlBody.success && dlBody.result && dlBody.result.default;

  // If no default render exists yet, trigger one.
  if (!def) {
    const dlPost = UrlFetchApp.fetch(
      'https://api.cloudflare.com/client/v4/accounts/' + accountId + '/stream/' + clipUid + '/downloads',
      { method: 'post', headers: auth, contentType: 'application/json', payload: '{}', muteHttpExceptions: true }
    );
    const postBody = JSON.parse(dlPost.getContentText() || '{}');
    if (!postBody.success) {
      const msg = (postBody.errors && postBody.errors.map(e => e.message).join('; ')) || 'unknown';
      return { state: 'error', error: 'Could not request MP4: ' + msg };
    }
    def = postBody.result && postBody.result.default;
  }

  if (!def) return { state: 'mp4-encoding', cfState: 'queued', percent: 0 };
  if (def.status === 'ready') return { state: 'ready', downloadUrl: def.url };
  return {
    state:   'mp4-encoding',
    cfState: def.status || 'inprogress',
    percent: parsePct_(def.percentComplete),
  };
}

// Cloudflare returns percent fields as strings ("45.123456"), numbers (45), or
// null depending on the endpoint and the lifecycle phase. Normalise to a
// rounded number, or null when no progress info is available yet.
function parsePct_(raw) {
  if (raw == null || raw === '') return null;
  const n = Number(raw);
  if (!isFinite(n)) return null;
  return Math.round(n);
}

// Ask Cloudflare for a one-time direct-creator upload URL.
// Browser then POSTs the video file to `uploadURL`; on success Cloudflare
// stores it under the returned `uid`.
function requestStreamUpload(maxDurationSeconds) {
  const { accountId, apiToken } = cfProps_();
  if (!accountId) throw new Error('CF_ACCOUNT_ID is not set in Script Properties.');
  if (!apiToken)  throw new Error('CF_API_TOKEN is not set in Script Properties.');
  const resp = UrlFetchApp.fetch(
    'https://api.cloudflare.com/client/v4/accounts/' + accountId + '/stream/direct_upload',
    {
      method:  'post',
      headers: { Authorization: 'Bearer ' + apiToken },
      contentType: 'application/json',
      payload: JSON.stringify({ maxDurationSeconds: maxDurationSeconds, requireSignedURLs: false }),
      muteHttpExceptions: true,
    }
  );
  const body = JSON.parse(resp.getContentText() || '{}');
  if (!body.success) {
    const msg = (body.errors && body.errors.map(e => e.message).join('; ')) || 'Unknown Cloudflare error';
    throw new Error('Cloudflare direct_upload failed: ' + msg);
  }
  return { uploadURL: body.result.uploadURL, uid: body.result.uid };
}
