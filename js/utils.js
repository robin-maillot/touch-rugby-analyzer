// Depends on js/config.js (TR namespace)

// Format seconds as h:mm:ss (always shows hours).
TR.fmt = (s) => {
  if (!s || isNaN(s)) return '0:00:00';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
};

TR.enc = (s) => encodeURIComponent(s);

// Returns the 11-char YouTube video ID from any YouTube URL form, or null.
TR.extractVideoId = (url) => {
  if (!url) return null;
  const m = String(url).match(/(?:v=|youtu\.be\/|embed\/|live\/)([A-Za-z0-9_-]{11})/);
  return m ? m[1] : null;
};

// Accept a full YouTube URL OR a bare 11-char video ID and return a canonical
// watch URL — or '' if neither. The field annotator lets users paste either form;
// live.html / the metadata store always get a clean watch?v= link.
TR.normalizeYoutubeUrl = (input) => {
  const v = String(input == null ? '' : input).trim();
  if (!v) return '';
  const id = TR.extractVideoId(v) || (/^[A-Za-z0-9_-]{11}$/.test(v) ? v : null);
  return id ? `https://www.youtube.com/watch?v=${id}` : '';
};

// Human-readable label for a sheet entry (from action=list), preferring
// metadata over tab-name parsing. Used by the game pickers in games.html
// and viewer.html so both surfaces show the same pretty name.
TR.sheetNameToLabel = (entry) => {
  if (!entry) return '';
  const t1   = entry.team1;
  const t2   = entry.team2;
  const tail = [entry.year, entry.division, entry.competition].filter(Boolean).join(' ');
  const id   = entry.id ? ` #${entry.id}` : '';
  if (t1 && t2) return `${t1} vs ${t2}${id}${tail ? '  (' + tail + ')' : ''}`;
  const parts = (entry.name || '').split('_');
  if (parts.length < 5) return entry.name || '';
  return `${parts[parts.length - 2]} vs ${parts[parts.length - 1]}${id}  (${parts.slice(0, -2).join(' ')})`;
};

// Replace "Team 1"/"Team 2" placeholder strings with actual team names.
// rows: array of arrays; team1/team2: actual names; cols: array of column indices to check.
TR.substituteTeams = (rows, team1, team2, cols) => {
  rows.forEach(row => {
    cols.forEach(idx => {
      if (idx < 0) return;
      if (row[idx] === 'Team 1') row[idx] = team1;
      else if (row[idx] === 'Team 2') row[idx] = team2;
    });
  });
};

// Describes a filter category's current selection for the Filter sheet's
// category list. Two values still fit a row; beyond that a count reads better
// than a truncated list.
TR.filterSummary = (values) => {
  const v = Array.from(values || []);
  if (!v.length) return 'Any';
  return v.length <= 2 ? v.map(String).join(', ') : `${v.length} selected`;
};

// Case-insensitive substring test for the Filter sheet's type-to-narrow box.
// An empty needle matches everything, so an untouched box hides nothing.
TR.optMatch = (value, needle) => {
  if (!needle) return true;
  return String(value == null ? '' : value).toLowerCase().includes(String(needle).toLowerCase());
};

// Identity for an event row. Index-free on purpose: an id built from the render
// position goes stale the moment a filter changes the list.
TR.evId = (e) => `${e.game}#${e.time}#${e.type}#${e.name}`;

// The part of an id that survives an Event Editor rename. Playlists store the
// full id (readable in the sheet) but resolve on this, so correcting a Name
// can't orphan a saved entry. Safe to split on '#': sheet tab names and event
// names never contain one.
TR.evKey = (e) => `${e.game}#${e.time}`;

// The same key, taken from a stored ref string rather than a live event.
TR.refKey = (ref) => String(ref == null ? '' : ref).split('#').slice(0, 2).join('#');

// One RFC 4180 CSV field.
//
// Two jobs. The quoting is ordinary — double every embedded quote, wrap the
// field when it carries a quote, a comma, CR or LF.
//
// The second job is the one that matters. Excel and Google Sheets EXECUTE a
// cell whose first character is = + - or @, and an event Comment is free text
// an annotator typed, so an exported file could run code on whoever opens it.
// The leading apostrophe is the conventional spreadsheet text marker and is
// consumed on display. Same class of bug sheetSafe() closes server-side, except
// this payload would land in the recipient's spreadsheet rather than ours.
//
// Guard first, quote second: quoting first would bury the apostrophe inside the
// quotes, where it protects nothing.
TR.csvCell = (v) => {
  let s = v == null ? '' : String(v);
  if (/^[=+\-@]/.test(s)) s = "'" + s;
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
};

// Rows of fields → RFC 4180 text. CRLF between rows, which is what the format
// says and what Excel expects; no trailing newline, so the file has no phantom
// final row.
TR.toCSV = (rows) => (rows || [])
  .map(r => (r || []).map(c => TR.csvCell(c)).join(','))
  .join('\r\n');

// A filename-safe slug. Diacritics are folded rather than dropped, because this
// dataset is largely French and "Équipe" deserves better than "quipe".
//
// Order matters: truncate BEFORE trimming separators. A cut at the cap can land
// mid-separator, and trimming first would leave the dash behind.
TR.slugify = (s, fallback) => {
  const out = String(s == null ? '' : s)
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .slice(0, 60)
    .replace(/^-+|-+$/g, '');
  return out || fallback || 'file';
};
