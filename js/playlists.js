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

// Fetch the caller's saved playlists from the backend. Uses a GET-style parameter
// to avoid CORS preflight. Returns an array of playlists; throws if the request fails.
TR.playlists.load = async () => {
  const resp = await fetch(`${TR.APPS_SCRIPT_URL}?action=playlists&secret=${encodeURIComponent(TR.secret())}`);
  const res = await resp.json();
  if (!res.ok) throw new Error(res.error || 'Load failed');
  return res.playlists;
};

// Save a playlist (create a new one, or update an existing one identified by id).
// The playlist object should have {name, note, refs} for a new entry, or
// {id, name, note, refs} for an update. Returns the uuid; throws if the request fails.
TR.playlists.save = async (playlist) => {
  const resp = await fetch(TR.APPS_SCRIPT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: JSON.stringify({
      secret: TR.secret(),
      action: 'save_playlist',
      id: playlist.id,
      name: playlist.name,
      note: playlist.note,
      refs: playlist.refs
    })
  });
  const res = await resp.json();
  if (!res.ok) throw new Error(res.error || 'Save failed');
  return res.id;
};

// Delete a playlist by its id. Throws if the request fails or the playlist does not exist.
TR.playlists.remove = async (id) => {
  const resp = await fetch(TR.APPS_SCRIPT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: JSON.stringify({
      secret: TR.secret(),
      action: 'delete_playlist',
      id: id
    })
  });
  const res = await resp.json();
  if (!res.ok) throw new Error(res.error || 'Remove failed');
};
