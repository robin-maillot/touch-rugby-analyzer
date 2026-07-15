// ── Configuration ──────────────────────────────────────────────
// SHEET_ID holds the game-data tabs (one per game).
// CONTROL_SHEET_ID holds the three control-plane tabs: _groups, _metadata, _live.
const SHEET_ID          = '1B9DwThGoINgicevtjoUDkeQBoGBCD6QFjL0oL7mAU-k';
const CONTROL_SHEET_ID  = '1Km2QtYOKLW-HQudHcfvjEksAZykBbIFKhl4AuA-AEIo';
const VALID_ROLES     = new Set(['viewer', 'staff', 'admin']);
const GROUPS_SHEET    = '_groups';     // reserved tab name for group access table
const METADATA_SHEET  = '_metadata';   // reserved tab name for game metadata
const LIVE_SHEET      = '_live';       // reserved tab name for live game state
// Control-plane tabs the admin sheet editor may read/write (all in CONTROL_SHEET_ID).
const ADMIN_SHEETS    = [GROUPS_SHEET, METADATA_SHEET, LIVE_SHEET];

// Expected column order (must match what the Python pipeline reads)
const HEADERS = ['Time', 'Possession Owner', 'Type', 'Name', 'To Review', 'Comment', 'Action Owner'];

// Metadata columns appended to action=all rows
const META_COLS = ['Team 1', 'Team 2', 'Competition', 'Year', 'Division', 'Video Name', 'Analyzable', 'ID'];

// ── Auth helper ────────────────────────────────────────────────
// Reads _groups (Group | Secret | Role) → Map<secret, {group, role}>.
// Cached in script cache for CACHE_TTL so we don't hit the sheet on every
// request. The generic cacheClear() does NOT bust this cache; callers that edit
// _groups (the admin editor) must call clearGroupsCache() so a newly-added
// secret is recognised at login immediately instead of after the TTL lapses.
function getGroups() {
  const hit = cacheGet('groups');
  if (hit) {
    try { return new Map(JSON.parse(hit)); } catch (e) {}
  }
  const sheet = SpreadsheetApp.openById(CONTROL_SHEET_ID).getSheetByName(GROUPS_SHEET);
  const map = new Map();
  if (!sheet) return map;
  const values = sheet.getDataRange().getDisplayValues();
  if (values.length < 2) return map;
  const h  = values[0].map(s => String(s).toLowerCase().trim());
  const gi = h.indexOf('group');
  const si = h.indexOf('secret');
  const ri = h.indexOf('role');
  if (si < 0 || ri < 0) return map;
  for (let i = 1; i < values.length; i++) {
    const secret = String(values[i][si] || '').trim();
    const role   = String(values[i][ri] || '').trim().toLowerCase();
    const group  = gi >= 0 ? String(values[i][gi] || '').trim() : '';
    if (secret && VALID_ROLES.has(role)) map.set(secret, { group, role });
  }
  try { cachePut('groups', JSON.stringify([...map])); } catch (e) {}
  return map;
}

function authFor(secret) {
  if (!secret) return null;
  return getGroups().get(String(secret)) || null;
}

// Evict the cached _groups map so the next authFor()/getGroups() re-reads the
// sheet. Call after any write that changes _groups (secrets, roles, groups),
// otherwise login keeps rejecting a just-added secret until the cache expires.
function clearGroupsCache() {
  try { CacheService.getScriptCache().remove('groups'); } catch (e) {}
}

function isAdminSecret(secret) {
  const a = authFor(secret);
  return !!(a && a.role === 'admin');
}

// ── Metadata helper ────────────────────────────────────────────
// Reads _metadata sheet and returns { [sheetName]: { team1, team2, competition, year, division } }
function getMetadata() {
  const sheet = SpreadsheetApp.openById(CONTROL_SHEET_ID).getSheetByName(METADATA_SHEET);
  if (!sheet) return {};

  const values = sheet.getDataRange().getDisplayValues();
  if (values.length < 2) return {};

  const h    = values[0].map(s => s.toLowerCase().trim());
  const col  = name => h.indexOf(name);
  const ni   = col('sheet name'), t1i = col('team 1'), t2i = col('team 2');
  const ci   = col('competition'), yi = col('year'), di = col('division');
  const vi   = col('video name'), ai = col('analyzable'), yui = col('youtube link');
  const idi  = col('id');
  const goi  = col('gcs object');      // object path within the configured GCS bucket
  const ggi  = col('groups');          // space-separated list of group names allowed to see this game
  const cmi  = col('comment');         // free-form game-level note

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
      id:          idi >= 0 ? row[idi] : '',
      gcsObject:   goi >= 0 ? row[goi] : '',
      groups:      ggi >= 0 ? String(row[ggi] || '').trim().split(/\s+/).filter(Boolean) : [],
      comment:     cmi >= 0 ? row[cmi] : '',
    };
  });
  return meta;
}

// True when `auth` (from authFor) is allowed to see the given game's meta entry.
// Admins always pass. The literal group "ALL" (case-insensitive) is a wildcard —
// any authenticated user can see games tagged with it. Otherwise the caller's
// group must appear in the game's Groups cell.
function canSeeGame(auth, gameMeta) {
  if (!auth) return false;
  if (auth.role === 'admin') return true;
  const gs = (gameMeta && gameMeta.groups) || [];
  if (gs.some(g => String(g).toUpperCase() === 'ALL')) return true;
  if (!auth.group) return false;
  return gs.indexOf(auth.group) >= 0;
}

// True when `auth` may edit a game's video links (backfill_youtube/gcs). Admins
// always pass; staff may edit only games their group can already see, so they
// can't touch games outside their group by POSTing a raw sheetName.
function canEditGame(auth, sheetName) {
  if (auth && auth.role === 'admin') return true;
  return canSeeGame(auth, getMetadata()[sheetName]);
}

// True when `auth` may OVERWRITE an existing game tab. Admins always; staff only
// when their group is explicitly attached to the game. Stricter than canSeeGame:
// the "ALL" wildcard grants read visibility, not override rights, and viewers
// never qualify — overriding a tab is destructive, so it needs real ownership.
function canOverrideGame(auth, sheetName) {
  if (!auth) return false;
  if (auth.role === 'admin') return true;
  if (auth.role !== 'staff' || !auth.group) return false;
  const meta = getMetadata()[sheetName];
  const gs = (meta && meta.groups) || [];
  return gs.indexOf(auth.group) >= 0;
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

// ── Tab ownership tokens ───────────────────────────────────────
// Maps a game tab name → the client-minted token that created it, so the same
// session can re-push (overwrite) a freshly-created game without admin override.
// Stored in persistent Script Properties (survives cache eviction). Volume is
// low (one key per game tab) so unbounded growth is not a concern.
function creatorTokenKey(sheetName) { return 'creator:' + sheetName; }

function getCreatorToken(sheetName) {
  try { return PropertiesService.getScriptProperties().getProperty(creatorTokenKey(sheetName)); }
  catch (e) { return null; }
}

function setCreatorToken(sheetName, token) {
  try {
    const props = PropertiesService.getScriptProperties();
    if (token) props.setProperty(creatorTokenKey(sheetName), token);
    else       props.deleteProperty(creatorTokenKey(sheetName));
  } catch (e) {}
}

function cacheClear() {
  // The list/all caches are keyed by version (key = '<list|all>:v<version>:<group>'),
  // so bumping the version below makes every existing entry unreachable — old keys
  // simply age out via CACHE_TTL. A bare removeAll(['list','all']) here was a no-op:
  // those unsuffixed keys are never written. The 'groups' cache is intentionally NOT
  // busted on writes (it's refreshed on its own schedule). Every cacheClear() site is
  // a write, and the version bump is what reliably forces clients to refetch — Drive's
  // getLastUpdated lags SpreadsheetApp writes, so we use an explicit counter instead.
  try { SpreadsheetApp.flush(); } catch (e) {}
  bumpVersion();
}

// Monotonic write counter stored in ScriptProperties. Bumped on every write
// via cacheClear() and consulted by getSheetVersion(). Replaces the previous
// Drive-getLastUpdated approach, which could lag writes by several seconds
// and cause clients to skip refetches and serve stale localStorage caches.
const VERSION_PROP = 'sheetVersion';

function bumpVersion() {
  try {
    PropertiesService.getScriptProperties().setProperty(VERSION_PROP, String(Date.now()));
  } catch (e) {}
}

function getSheetVersion() {
  try {
    const props  = PropertiesService.getScriptProperties();
    const stored = props.getProperty(VERSION_PROP);
    if (stored) return Number(stored);
    // Cold start: seed from Drive so any clients with pre-existing caches
    // still validate on the first call. Max across both spreadsheets — game
    // tabs live in SHEET_ID, but _metadata (which the list reads) lives in
    // CONTROL_SHEET_ID, so either one being newer should bust stale clients.
    let seed = 0;
    try {
      const t1 = DriveApp.getFileById(SHEET_ID).getLastUpdated().getTime();
      const t2 = DriveApp.getFileById(CONTROL_SHEET_ID).getLastUpdated().getTime();
      seed = Math.max(t1, t2);
    } catch (e) {}
    if (!seed) seed = Date.now();
    props.setProperty(VERSION_PROP, String(seed));
    return seed;
  } catch (e) { return 0; }
}

// ── Installable onEdit trigger ─────────────────────────────────
// Invalidates the list/all caches when _metadata is edited directly in the
// Sheets UI. Apps-Script-driven writes already call cacheClear(), but manual
// edits (toggling Analyzable, updating YouTube Link, changing Groups, adding
// a new game row) bypass that path and would otherwise sit behind the 5-min
// CACHE_TTL.
//
// One-time install (in the Apps Script editor):
//   Triggers → Add Trigger:
//     Function:           onMetadataEdit
//     Event source:       From spreadsheet
//     Select spreadsheet: the spreadsheet whose ID matches CONTROL_SHEET_ID
//     Event type:         On edit
// Re-install if CONTROL_SHEET_ID changes.
function onMetadataEdit(e) {
  try {
    if (!e || !e.range) return;
    if (e.range.getSheet().getName() !== METADATA_SHEET) return;
    cacheClear();
  } catch (err) {}
}

function rawJson(str) {
  return ContentService.createTextOutput(str).setMimeType(ContentService.MimeType.JSON);
}

// ── GET — list sheets or fetch rows from a tab ─────────────────
function doGet(e) {
  try {
    // action=live → current live game states. PUBLIC endpoint: no secret
    // required and no group filtering, so the live scoreboard is shareable to
    // spectators without a login. Handled before the auth gate below.
    if (e.parameter.action === 'live') {
      const sheet = SpreadsheetApp.openById(CONTROL_SHEET_ID).getSheetByName(LIVE_SHEET);
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
      const yli  = col('youtube link');
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
          youtubelink: yli  >= 0 ? row[yli]  : '',
        }))
        .filter(g => g.name);
      return json({ ok: true, games });
    }

    const auth = authFor(e.parameter.secret);
    if (!auth) {
      return json({ ok: false, error: 'Unauthorized' });
    }

    // action=version → live spreadsheet-modified timestamp (never cached server-side).
    // Clients use this to skip the heavy action=all / action=list refetches when nothing changed.
    if (e.parameter.action === 'version') {
      return json({ ok: true, version: getSheetVersion() });
    }

    // action=whoami → return the role/group for the supplied secret. Used by
    // index.html to validate a cached password against the authoritative _groups
    // sheet (replacing a stale hardcoded password map).
    if (e.parameter.action === 'whoami') {
      return json({ ok: true, role: auth.role, group: auth.group });
    }

    // action=admin_sheet → raw grid (headers + data rows) of a control sheet,
    // for the admin sheet editor. Admin only; never cached (always live).
    if (e.parameter.action === 'admin_sheet') {
      if (auth.role !== 'admin') return json({ ok: false, error: 'Admin access required.' });
      const name = e.parameter.name;
      if (ADMIN_SHEETS.indexOf(name) < 0) return json({ ok: false, error: 'Sheet not editable.' });
      const sheet = SpreadsheetApp.openById(CONTROL_SHEET_ID).getSheetByName(name);
      if (!sheet) return json({ ok: true, name, headers: [], rows: [] });
      const values = sheet.getDataRange().getDisplayValues();
      return json({ ok: true, name, headers: values[0] || [], rows: values.slice(1) });
    }

    // action=admin_stale_sheets → game-data tabs in SHEET_ID that have no matching
    // _metadata row (orphans left behind when a metadata row is deleted). They're
    // invisible in the games list (which is metadata-driven) but still take up
    // space, so surface them for cleanup. Admin only; never cached.
    if (e.parameter.action === 'admin_stale_sheets') {
      if (auth.role !== 'admin') return json({ ok: false, error: 'Admin access required.' });
      const meta = getMetadata();
      const ss   = SpreadsheetApp.openById(SHEET_ID);
      const stale = ss.getSheets()
        .filter(sh => !Object.prototype.hasOwnProperty.call(meta, sh.getName()))
        .map(sh => ({ name: sh.getName(), rows: Math.max(0, sh.getLastRow() - 1) }));
      return json({ ok: true, stale });
    }

    // Cache key suffix: include version + caller's role/group so writes auto-invalidate
    // (version bumps on cacheClear) and group A never sees group B's cached payload.
    const cacheKeySuffix = ':v' + getSheetVersion() + ':' + (auth.role === 'admin' ? '*admin' : (auth.group || '*nogroup'));

    // action=list → sheet names + metadata for each game
    if (e.parameter.action === 'list') {
      const key = 'list' + cacheKeySuffix;
      const hit = cacheGet(key);
      if (hit) return rawJson(hit);

      const version = getSheetVersion();
      const meta    = getMetadata();
      const sheets  = Object.entries(meta)
        .filter(([_, m]) => canSeeGame(auth, m))
        .map(([name, m]) => ({ name, ...m }));
      const result  = JSON.stringify({ ok: true, version, sheets });
      cachePut(key, result);
      return rawJson(result);
    }

    // action=all → every row from every game sheet listed in _metadata, metadata columns appended
    if (e.parameter.action === 'all') {
      const key = 'all' + cacheKeySuffix;
      const hit = cacheGet(key);
      if (hit) return rawJson(hit);

      const version = getSheetVersion();
      const meta    = getMetadata();
      const ss     = SpreadsheetApp.openById(SHEET_ID);
      const allowed = new Set(Object.keys(meta).filter(name => canSeeGame(auth, meta[name])));
      const sheets = ss.getSheets().filter(s => allowed.has(s.getName()));
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
      cachePut(key, result);
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
    const version = getSheetVersion();   // capture BEFORE reading data
    const values  = sheet.getDataRange().getDisplayValues();
    const meta    = getMetadata();
    if (!canSeeGame(auth, meta[sheetName])) {
      return json({ ok: false, error: `Tab "${sheetName}" not found.` });
    }
    return json({ ok: true, version, rows: values, meta: meta[sheetName] || {} });

  } catch (err) {
    return json({ ok: false, error: err.toString() });
  }
}

// ── POST — write rows to a tab ─────────────────────────────────
function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);

    const auth = authFor(data.secret);
    if (!auth) {
      return json({ ok: false, error: 'Unauthorized' });
    }
    // Non-admin writers have their group auto-stamped onto the game's metadata
    // row (lazily creating the row + Groups column as needed). Admins skip this
    // — they should set Groups explicitly via backfill.
    const callerGroup = auth.role === 'admin' ? '' : (auth.group || '');

    // action=update_rows → update Name/Comment for specific rows (admin only)
    if (data.action === 'update_rows') {
      if (!isAdminSecret(data.secret)) return json({ ok: false, error: 'Admin access required.' });
      let updated = 0;
      for (const change of (data.changes || [])) {
        if (updateRow(change.sheetName, change.time, change.name, change.comment)) updated++;
      }
      cacheClear();
      return json({ ok: true, updated });
    }

    // action=backfill_youtube → write YouTube links into a game tab. Staff may
    // do this for games in their own group; admins for any game.
    if (data.action === 'backfill_youtube') {
      if (!data.sheetName) return json({ ok: false, error: 'Missing sheetName parameter' });
      if (!canEditGame(auth, data.sheetName)) return json({ ok: false, error: 'You can only edit games in your group.' });
      const updated = backfillYoutubeLink(data.sheetName, data.youtubeUrl, Number(data.offsetSeconds) || 0);
      cacheClear();
      return json({ ok: true, updated });
    }

    // action=backfill_gcs → attach a GCS object path to a game. Staff may do this
    // for games in their own group; admins for any game. Empty string clears it.
    if (data.action === 'backfill_gcs') {
      if (!data.sheetName) return json({ ok: false, error: 'Missing sheetName parameter' });
      if (!canEditGame(auth, data.sheetName)) return json({ ok: false, error: 'You can only edit games in your group.' });
      writeMeta(data.sheetName, { gcsObject: String(data.gcsObject || '').trim() });
      cacheClear();
      return json({ ok: true });
    }

    // action=backfill_groups → set the full Groups list for a game (admin only —
    // Groups control access, so staff must not be able to change visibility).
    // Empty list / empty string is valid — removes all groups (admin-only visibility).
    if (data.action === 'backfill_groups') {
      if (!isAdminSecret(data.secret)) return json({ ok: false, error: 'Admin access required.' });
      if (!data.sheetName) return json({ ok: false, error: 'Missing sheetName parameter' });
      const groups = String(data.groups || '').trim().split(/\s+/).filter(Boolean);
      writeMeta(data.sheetName, { groups });
      cacheClear();
      return json({ ok: true });
    }

    // action=backfill_comment → write a free-form game comment (staff + admin).
    // Empty string is valid — used to clear the comment.
    if (data.action === 'backfill_comment') {
      if (auth.role !== 'staff' && auth.role !== 'admin') return json({ ok: false, error: 'Staff access required.' });
      if (!data.sheetName) return json({ ok: false, error: 'Missing sheetName parameter' });
      writeMeta(data.sheetName, { comment: String(data.comment == null ? '' : data.comment) });
      cacheClear();
      return json({ ok: true });
    }

    // ── Admin sheet editor (admin only) ──────────────────────────
    // Edit individual cells / add / delete rows in a control sheet
    // (_groups, _metadata, _live). Every path calls cacheClear() so the change
    // propagates site-wide immediately — unlike a manual edit in the Sheets UI,
    // which only invalidates caches if the onMetadataEdit trigger is installed.
    function adminSheet(name) {
      if (ADMIN_SHEETS.indexOf(name) < 0) return null;
      return SpreadsheetApp.openById(CONTROL_SHEET_ID).getSheetByName(name);
    }

    if (data.action === 'admin_set_cell') {
      if (!isAdminSecret(data.secret)) return json({ ok: false, error: 'Admin access required.' });
      const sheet = adminSheet(data.sheet);
      if (!sheet) return json({ ok: false, error: 'Sheet not editable.' });
      const r = Number(data.row), c = Number(data.col); // 0-based data coords (row excludes header)
      if (!(r >= 0) || !(c >= 0)) return json({ ok: false, error: 'Bad cell coordinates.' });
      sheet.getRange(r + 2, c + 1).setValue(data.value == null ? '' : data.value);
      cacheClear();
      if (data.sheet === GROUPS_SHEET) clearGroupsCache();
      return json({ ok: true });
    }

    if (data.action === 'admin_add_row') {
      if (!isAdminSecret(data.secret)) return json({ ok: false, error: 'Admin access required.' });
      const sheet = adminSheet(data.sheet);
      if (!sheet) return json({ ok: false, error: 'Sheet not editable.' });
      const width = Math.max(sheet.getLastColumn(), 1);
      const row = [];
      for (let i = 0; i < width; i++) row.push((data.values && data.values[i] != null) ? data.values[i] : '');
      sheet.appendRow(row);
      cacheClear();
      if (data.sheet === GROUPS_SHEET) clearGroupsCache();
      return json({ ok: true, row: sheet.getLastRow() - 2 }); // 0-based data index of the new row
    }

    if (data.action === 'admin_delete_row') {
      if (!isAdminSecret(data.secret)) return json({ ok: false, error: 'Admin access required.' });
      const sheet = adminSheet(data.sheet);
      if (!sheet) return json({ ok: false, error: 'Sheet not editable.' });
      const r = Number(data.row);
      if (!(r >= 0)) return json({ ok: false, error: 'Bad row index.' });
      sheet.deleteRow(r + 2);
      cacheClear();
      if (data.sheet === GROUPS_SHEET) clearGroupsCache();
      return json({ ok: true });
    }

    // action=admin_delete_sheet → delete a stale game-data tab (one with no
    // _metadata row). Guarded to stale tabs only, so an active game's data can't
    // be nuked here — deleting a live game is a separate, deliberate flow. Also
    // clears the tab's creator token and finalises any leftover _live row.
    if (data.action === 'admin_delete_sheet') {
      if (!isAdminSecret(data.secret)) return json({ ok: false, error: 'Admin access required.' });
      const name = data.name;
      if (!name) return json({ ok: false, error: 'Missing sheet name.' });
      if (Object.prototype.hasOwnProperty.call(getMetadata(), name)) {
        return json({ ok: false, error: 'Sheet is not stale (it has a _metadata row).' });
      }
      const ss    = SpreadsheetApp.openById(SHEET_ID);
      const sheet = ss.getSheetByName(name);
      if (!sheet) return json({ ok: false, error: 'Sheet not found.' });
      if (ss.getSheets().length <= 1) return json({ ok: false, error: 'Cannot delete the only tab in the spreadsheet.' });
      ss.deleteSheet(sheet);
      setCreatorToken(name, ''); // drop leftover ownership token
      clearLiveRow(name);        // finalise any stray live row (no-op if absent)
      cacheClear();
      return json({ ok: true });
    }

    // action=live_update → upsert a row in _live
    if (data.action === 'live_update') {
      writeLiveRow(data.sheetName, data.team1, data.team2, data.score1, data.score2, data.timeSeconds, data.poss1, data.poss2, data.comps1, data.comps2, data.triesJson, data.youtubelink);
      // Ensure the caller's group is on the game's metadata row. writeMeta has
      // a fast no-op path for pure addGroup calls when the group is already
      // present, so high-frequency live_update calls don't repeatedly write.
      if (callerGroup) writeMeta(data.sheetName, { addGroup: callerGroup });
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
      // A push carrying the same creatorToken that first created this tab is
      // allowed to overwrite it — this lets the field annotator push the same
      // (newly-created) game repeatedly from one device without admin override.
      const ownsTab = data.creatorToken && getCreatorToken(data.sheetName) === data.creatorToken;
      if (!ownsTab) {
        if (!data.override) {
          // Echo the game's groups so the client can decide whether to offer the
          // Override button (admins always; staff only for their own group).
          const meta = getMetadata()[data.sheetName];
          return json({ ok: false, error: `Tab "${data.sheetName}" already exists.`, tabExists: true, tabGroups: (meta && meta.groups) || [] });
        }
        if (!canOverrideGame(auth, data.sheetName)) {
          return json({ ok: false, error: 'You can only override games in your group. Admin access is required otherwise.' });
        }
      }
      ss.deleteSheet(sheet);
    }

    const newSheet = ss.insertSheet(data.sheetName);
    // Record (or refresh) which token owns this tab so subsequent pushes from
    // the same session can overwrite it. Admin overrides without a token clear
    // any prior ownership.
    setCreatorToken(data.sheetName, data.creatorToken || '');
    newSheet.appendRow(HEADERS);

    const rows = data.rows;
    if (rows && rows.length > 0) {
      newSheet.getRange(newSheet.getLastRow() + 1, 1, rows.length, rows[0].length)
              .setValues(rows);
    }

    // Stamp the caller's group onto the new metadata row (non-admin writes only).
    const metaToWrite = data.meta || {};
    if (callerGroup) metaToWrite.addGroup = callerGroup;
    if (Object.keys(metaToWrite).length) writeMeta(data.sheetName, metaToWrite);

    cacheClear(); // invalidate list + all caches after any write

    // Echo the group we stamped so the client can confirm attribution.
    return json({ ok: true, appended: rows ? rows.length : 0, group: callerGroup });

  } catch (err) {
    return json({ ok: false, error: err.toString() });
  }
}

// ── Write / upsert a row in _metadata ──────────────────────────
// If a row for sheetName already exists → overwrite it.
// If not → append to the bottom. No extra password needed here
// because the caller has already passed the override check above.
function writeMeta(sheetName, meta) {
  const ss = SpreadsheetApp.openById(CONTROL_SHEET_ID);
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
  let goi    = col('gcs object');
  let ggi    = col('groups');

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

  // Add 'GCS Object' column lazily
  if (goi < 0 && meta.gcsObject != null) {
    sheet.getRange(1, headers.length + 1).setValue('GCS Object');
    headers.push('gcs object');
    goi = headers.length - 1;
    values.forEach(row => { while (row.length < headers.length) row.push(''); });
  }

  // Add 'Groups' column lazily — triggered by either a replace (meta.groups) or
  // an additive write (meta.addGroup).
  if (ggi < 0 && (meta.groups != null || meta.addGroup)) {
    sheet.getRange(1, headers.length + 1).setValue('Groups');
    headers.push('groups');
    ggi = headers.length - 1;
    values.forEach(row => { while (row.length < headers.length) row.push(''); });
  }

  let cmi = col('comment');
  if (cmi < 0 && meta.comment != null) {
    sheet.getRange(1, headers.length + 1).setValue('Comment');
    headers.push('comment');
    cmi = headers.length - 1;
    values.forEach(row => { while (row.length < headers.length) row.push(''); });
  }

  // Find existing row, if any
  const numCols = headers.length;
  let existingRow = -1;
  let existingData = null;
  for (let i = 1; i < values.length; i++) {
    if (ni >= 0 && values[i][ni] === sheetName) { existingRow = i + 1; existingData = values[i]; break; }
  }

  // Fast no-op short-circuit: a pure addGroup call (no other meta fields) where
  // the row already exists and the group is already listed. live_update fires
  // every event so we skip writes here aggressively.
  const isPureAddGroup = meta.addGroup && !Object.keys(meta).some(k => k !== 'addGroup' && meta[k] != null);
  if (isPureAddGroup && existingRow > 0 && ggi >= 0) {
    const cur = String(existingData[ggi] || '').trim().split(/\s+/).filter(Boolean);
    if (cur.indexOf(meta.addGroup) >= 0) return;
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
  if (goi >= 0 && meta.gcsObject   != null) row[goi] = meta.gcsObject;
  if (cmi >= 0 && meta.comment     != null) row[cmi] = meta.comment;

  // Groups: `groups` replaces the cell wholesale; `addGroup` appends to whatever
  // is already there (de-duped). The two are independent — `groups` wins if both
  // are passed.
  if (ggi >= 0) {
    let groupList = String(row[ggi] || '').trim().split(/\s+/).filter(Boolean);
    if (Array.isArray(meta.groups)) {
      groupList = meta.groups.map(s => String(s).trim()).filter(Boolean);
    }
    if (meta.addGroup) {
      const g = String(meta.addGroup).trim();
      if (g && groupList.indexOf(g) < 0) groupList.push(g);
    }
    row[ggi] = groupList.join(' ');
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
  const gamesSS = SpreadsheetApp.openById(SHEET_ID);
  const ctrlSS  = SpreadsheetApp.openById(CONTROL_SHEET_ID);
  const sheets  = gamesSS.getSheets().map(s => s.getName());

  // Delete and recreate _metadata sheet in the control spreadsheet
  const existing = ctrlSS.getSheetByName(METADATA_SHEET);
  if (existing) ctrlSS.deleteSheet(existing);
  const meta = ctrlSS.insertSheet(METADATA_SHEET);

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
const LIVE_HEADERS = ['Sheet Name', 'Team 1', 'Team 2', 'Score 1', 'Score 2', 'Time Seconds', 'Updated At', 'Poss 1', 'Poss 2', 'Comps 1', 'Comps 2', 'Finished', 'Tries JSON', 'Youtube Link'];

function writeLiveRow(sheetName, team1, team2, score1, score2, timeSeconds, poss1, poss2, comps1, comps2, triesJson, youtubelink) {
  const ss = SpreadsheetApp.openById(CONTROL_SHEET_ID);
  let sheet = ss.getSheetByName(LIVE_SHEET);

  if (!sheet) {
    sheet = ss.insertSheet(LIVE_SHEET);
    sheet.appendRow(LIVE_HEADERS);
  } else if (sheet.getLastColumn() < LIVE_HEADERS.length) {
    const existing = sheet.getLastColumn();
    sheet.getRange(1, existing + 1, 1, LIVE_HEADERS.length - existing).setValues([LIVE_HEADERS.slice(existing)]);
  }

  const nowStr = new Date().toISOString();
  const newRow = [sheetName, team1 || '', team2 || '', score1 || 0, score2 || 0, timeSeconds || 0, nowStr, poss1 || 0, poss2 || 0, comps1 || 0, comps2 || 0, '', triesJson || '[]', youtubelink || ''];
  const values = sheet.getDataRange().getValues();

  const ytIdx = LIVE_HEADERS.length - 1; // Youtube Link is the last column

  for (let i = 1; i < values.length; i++) {
    if (String(values[i][0]) === sheetName) {
      // Don't clobber a previously-set link with an empty update.
      if (!youtubelink && values[i][ytIdx]) newRow[ytIdx] = values[i][ytIdx];
      sheet.getRange(i + 1, 1, 1, newRow.length).setValues([newRow]);
      return;
    }
  }
  sheet.appendRow(newRow);
}

function clearLiveRow(sheetName) {
  const ss    = SpreadsheetApp.openById(CONTROL_SHEET_ID);
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

// ── Attach a YouTube URL to a game ─────────────────────────────
// offsetSeconds: if non-zero, shifts every Time value in the game tab by
//                this amount so the sheet aligns with the video timeline.
function backfillYoutubeLink(sheetName, youtubeUrl, offsetSeconds) {
  const ss    = SpreadsheetApp.openById(SHEET_ID);
  const sheet = ss.getSheetByName(sheetName);
  if (!sheet) throw new Error(`Tab "${sheetName}" not found.`);
  if (!String(youtubeUrl || '').match(/(?:v=|youtu\.be\/|embed\/|live\/)([A-Za-z0-9_-]{11})/)) {
    throw new Error('Invalid YouTube URL — could not extract video ID.');
  }
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
