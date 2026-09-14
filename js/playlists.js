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

// ── Network ───────────────────────────────────────────────────
// Apps Script web apps reject a CORS preflight, so every POST here sends
// text/plain — the same shape every other write in the app uses.
//
// resp.json() throws SyntaxError on a non-JSON body (e.g. the HTML error page
// Apps Script returns for a broken deployment or a permissions problem — a
// live risk while this backend is mid-redeploy). Guarding it means a caller
// always gets a showable Error, either the server's own message or a note
// that the response wasn't valid, never a raw parser exception.
TR.playlists._post = async (body) => {
  const r = await fetch(TR.APPS_SCRIPT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: JSON.stringify(body),
  });
  let j;
  try { j = await r.json(); } catch (e) { throw new Error('The server did not return a valid response.'); }
  if (!j.ok) throw new Error(j.error || 'Playlist save failed.');
  return j;
};

// Fetch the caller's saved playlists from the backend. Uses a GET-style
// parameter to avoid a CORS preflight.
TR.playlists.load = async (secret) => {
  const r = await fetch(`${TR.APPS_SCRIPT_URL}?secret=${TR.enc(secret)}&action=playlists`);
  let j;
  try { j = await r.json(); } catch (e) { throw new Error('The server did not return a valid response.'); }
  if (!j.ok) throw new Error(j.error || 'Could not load playlists.');
  return j.playlists || [];
};

// pl: {id?, name, note, refs}. Omitting id creates; passing one the caller
// doesn't own is refused server-side. Resolves to {id} so callers can
// destructure the id of a freshly-created playlist straight off the result.
TR.playlists.save = (secret, pl) => TR.playlists._post({
  secret, action: 'save_playlist',
  id: pl.id || '', name: pl.name, note: pl.note || '', refs: pl.refs || [],
});

TR.playlists.remove = (secret, id) =>
  TR.playlists._post({ secret, action: 'delete_playlist', id }).then(() => undefined);
