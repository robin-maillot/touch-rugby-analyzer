// Depends on js/config.js (TR namespace)

// Multi-game store for the field annotator. One localStorage key per game, so a
// tap mid-game only rewrites that game, plus an index key holding the id list.
// Lives here rather than in annotator_field.html so it can be unit-tested.
//
// A game is "dirty" when its revision has moved past the revision that was last
// pushed — a counter rather than an event count, because editing an event type
// (allowed even on a finished game) doesn't change how many events there are.
TR.FieldGames = (function () {
  const INDEX_KEY  = 'fieldGamesIndex';
  const LEGACY_KEY = 'fieldAnnotatorSession';   // pre-multi-game single session
  const gameKey    = id => 'fieldGame:' + id;

  const EMPTY_META = { team1: '', team2: '', year: '', division: '', competition: '', id: '', youtubelink: '' };
  const EMPTY_SYNC = { pushedAt: null, pushedRevision: null, sheetName: '' };

  // localStorage can be absent (Node tests), disabled (private mode) or full —
  // every access is guarded so the annotator degrades to in-memory instead of
  // throwing mid-game.
  function read(key) {
    try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : null; }
    catch (e) { return null; }
  }
  function write(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); return true; }
    catch (e) { return false; }
  }
  function drop(key) { try { localStorage.removeItem(key); } catch (e) {} }

  function newId() {
    const rand = (typeof crypto !== 'undefined' && crypto.randomUUID)
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10);
    return 'g_' + rand + Date.now().toString(36).slice(-4);
  }

  function ids() {
    const v = read(INDEX_KEY);
    return Array.isArray(v) ? v.filter(x => typeof x === 'string') : [];
  }

  function blank(meta) {
    const now = Date.now();
    return {
      id: newId(),
      createdAt: now,
      updatedAt: now,
      revision: 0,
      status: 'active',            // 'active' | 'finished'
      finishedAt: null,
      annotations: [],
      possession: 'Team 1',
      wallStart: null,
      teamsSwapped: false,
      creatorToken: null,
      meta: Object.assign({}, EMPTY_META, meta || {}),
      sync: Object.assign({}, EMPTY_SYNC),
    };
  }

  // Fill in anything a record written by an older build is missing, so the
  // picker never has to null-check its way through a half-shaped game.
  function normalize(rec) {
    if (!rec || typeof rec !== 'object' || !rec.id) return null;
    rec.annotations  = Array.isArray(rec.annotations) ? rec.annotations : [];
    rec.possession   = rec.possession === 'Team 2' ? 'Team 2' : 'Team 1';
    rec.status       = rec.status === 'finished' ? 'finished' : 'active';
    rec.revision     = typeof rec.revision === 'number' ? rec.revision : 0;
    rec.teamsSwapped = !!rec.teamsSwapped;
    rec.meta         = Object.assign({}, EMPTY_META, rec.meta || {});
    rec.sync         = Object.assign({}, EMPTY_SYNC, rec.sync || {});
    if (typeof rec.createdAt !== 'number') rec.createdAt = rec.updatedAt || Date.now();
    if (typeof rec.updatedAt !== 'number') rec.updatedAt = rec.createdAt;
    return rec;
  }

  function get(id) { return normalize(read(gameKey(id))); }

  // Newest-edited first — the order the picker renders in.
  function list() {
    return ids().map(get).filter(Boolean).sort((a, b) => b.updatedAt - a.updatedAt);
  }

  // touch:false writes without counting as an edit — used when the change isn't
  // the annotator's data (e.g. recording a successful push).
  function save(rec, opts) {
    if (!rec || !rec.id) return false;
    if (!opts || opts.touch !== false) {
      rec.revision  = (rec.revision || 0) + 1;
      rec.updatedAt = Date.now();
    }
    const ok  = write(gameKey(rec.id), rec);
    const all = ids();
    if (all.indexOf(rec.id) < 0) write(INDEX_KEY, [rec.id].concat(all));
    return ok;
  }

  function create(meta) {
    const rec = blank(meta);
    save(rec, { touch: false });
    return rec;
  }

  function remove(id) {
    drop(gameKey(id));
    write(INDEX_KEY, ids().filter(x => x !== id));
  }

  function isDirty(rec) {
    if (!rec) return false;
    return rec.sync.pushedRevision == null || rec.sync.pushedRevision !== rec.revision;
  }
  function isUploaded(rec) { return !!(rec && rec.sync.pushedAt); }
  function isSynced(rec)   { return isUploaded(rec) && !isDirty(rec); }

  // `revision` lets the caller record the revision it actually uploaded, so
  // events tagged while the request was in flight stay marked as unsynced.
  function markSynced(rec, sheetName, revision) {
    if (!rec) return;
    rec.sync = {
      pushedAt: Date.now(),
      pushedRevision: revision == null ? rec.revision : revision,
      sheetName: sheetName || '',
    };
    save(rec, { touch: false });
  }

  // Bulk cleanup offered in the picker. Only games that are uploaded AND have no
  // edits since — anything unsynced is left alone, since this device is the only
  // copy of it.
  function removeSynced() {
    const gone = list().filter(isSynced);
    gone.forEach(r => remove(r.id));
    return gone.length;
  }

  // Two saved games that resolve to the same sheet tab would fight over it on
  // push (the second hits "tab exists"). Returns the other games colliding with
  // this one, so Setup can nudge for an ID.
  // `name` overrides the name computed for `rec` itself — the annotator passes
  // the name from the live Setup fields, which are a keystroke ahead of what has
  // been saved onto the record.
  function collisions(rec, sheetNameOf, name) {
    const mine = name == null ? sheetNameOf(rec) : name;
    if (!mine) return [];
    return list().filter(o => o.id !== rec.id && sheetNameOf(o) === mine);
  }

  // Denormalized view for the picker tiles.
  function summarize(rec) {
    const a          = rec.annotations;
    const triesFor   = t => a.filter(x => x.type === 'Try' && x.actionOwner === t).length;
    const starts     = a.filter(x => x.name === 'Game Start');
    const ends       = a.filter(x => x.name === 'Game End');
    const firstStart = starts[0];
    const lastStart  = starts[starts.length - 1];
    const lastEnd    = ends[ends.length - 1];
    const lastTime   = a.length ? a[a.length - 1].time : 0;
    const onBreak    = !!(lastEnd && lastStart && lastEnd.time > lastStart.time);
    const finished   = rec.status === 'finished';
    const running    = !finished && !!lastStart && !onBreak;

    // A running game's clock is still moving, so measure it against the wall
    // clock rather than the last tagged event (which would freeze between taps).
    let duration = 0;
    if (firstStart) {
      duration = (running && rec.wallStart)
        ? Math.max(0, (Date.now() - (rec.wallStart + firstStart.time * 1000)) / 1000)
        : Math.max(0, (finished && lastEnd ? lastEnd.time : lastTime) - firstStart.time);
    }

    return {
      team1:    rec.meta.team1 || 'Team 1',
      team2:    rec.meta.team2 || 'Team 2',
      titled:   !!(rec.meta.team1 || rec.meta.team2),
      subtitle: [rec.meta.year, rec.meta.division, rec.meta.competition].filter(Boolean).join(' · '),
      score1:   triesFor('Team 1'),
      score2:   triesFor('Team 2'),
      events:   a.length,
      duration,
      status:   finished ? 'finished' : running ? 'running' : starts.length ? 'paused' : 'new',
      dirty:    isDirty(rec),
      uploaded: isUploaded(rec),
      synced:   isSynced(rec),
    };
  }

  // One-time upgrade from the single-session build. Returns the imported game,
  // or null when there was nothing worth keeping. The legacy key is always
  // cleared so this only ever runs once.
  function migrateLegacy() {
    const old = read(LEGACY_KEY);
    if (!old) return null;
    drop(LEGACY_KEY);
    if (!Array.isArray(old.annotations) || !old.annotations.length) return null;
    const rec = blank(old.meta);
    rec.annotations  = old.annotations;
    rec.possession   = old.possession || 'Team 1';
    rec.wallStart    = old.wallStart || null;
    rec.teamsSwapped = !!old.teamsSwapped;
    rec.creatorToken = old.creatorToken || null;
    save(rec, { touch: false });
    return rec;
  }

  return {
    INDEX_KEY, LEGACY_KEY, gameKey,
    ids, list, get, create, save, remove,
    isDirty, isUploaded, isSynced, markSynced, removeSynced,
    collisions, summarize, migrateLegacy,
  };
})();
