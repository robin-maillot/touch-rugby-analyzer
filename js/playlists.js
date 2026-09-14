// Depends on js/config.js (TR namespace) and js/utils.js (TR.evKey / TR.refKey).
//
// A playlist is a named, ordered list of hand-picked events belonging to one
// account. It spans games and filters freely, because each entry is a concrete
// event ref rather than a filter that has to re-derive one.
TR.playlists = {};

// The sheet cap, mirrored client-side so the UI can refuse before the round
// trip. Kept in step with PLAYLIST_MAX_REFS in apps_script/Code.gs.
TR.playlists.MAX_REFS = 500;

// Turn stored refs into live event objects, in the PLAYLIST's order rather than
// the clock's.
//
// Matching is on game + time alone (TR.refKey). Refs are synthetic — built from
// the row's own values — so an Event Editor rename changes the type and name
// segments of every ref pointing at that event. Matching loosely survives that;
// matching on all four would silently empty a playlist after a correction.
//
// Where two rows somehow share a game and a timestamp, the first in sheet order
// wins, deterministically.
TR.playlists.resolve = (refs, rows) => {
  const by = new Map();
  (rows || []).forEach(e => {
    const k = TR.evKey(e);
    if (!by.has(k)) by.set(k, e);
  });
  const events = [];
  let missing = 0;
  (refs || []).forEach(r => {
    const e = by.get(TR.refKey(r));
    if (e) events.push(e); else missing++;
  });
  return { events, missing };
};

// Move one entry, returning a new array. An out-of-range or unchanged index is
// a no-op copy rather than a throw — the reorder screen's ↑ on the first row
// and ↓ on the last both land here.
TR.playlists.reorder = (refs, from, to) => {
  const out = (refs || []).slice();
  if (!(from >= 0) || from >= out.length || !(to >= 0) || to >= out.length || from === to) return out;
  const [moved] = out.splice(from, 1);
  out.splice(to, 0, moved);
  return out;
};
