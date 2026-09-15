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
// The same apps also treat a leading apostrophe as their "force as text"
// marker and consume it on read — so '19 season would silently come back as
// 19 season. Same class of bug sheetSafe() closes server-side. Doubling the
// apostrophe fixes both: their marker eats our extra one and the user's real
// character survives.
//
// Guard first, quote second: quoting first would bury the apostrophe inside the
// quotes, where it protects nothing.
//
// Write-only: applying this twice guards and quotes twice over. The only
// caller is toCSV(), whose joined string can't be fed back in by accident —
// but call it exactly once per value, never on an already-exported cell.
TR.csvCell = (v) => {
  let s = v == null ? '' : String(v);
  // Spreadsheet apps strip leading whitespace before checking for a formula
  // character, so the test must too — but prefix the ORIGINAL string, not the
  // trimmed one, or the user's own leading space/tab is silently eaten.
  if (/^['=+\-@]/.test(s.trim())) s = "'" + s;
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
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .slice(0, 60)
    .replace(/^-+|-+$/g, '');
  return out || fallback || 'file';
};

// Parse RFC 4180 text into rows of fields — the counterpart to TR.toCSV, and a
// real parser rather than a pair of splits. A field may legitimately hold a
// comma, a doubled quote or a newline; splitting on those turns one row into
// several and silently truncates the field that contained them.
//
// Bare LF is accepted alongside CRLF, because files this app wrote before
// toCSV existed joined rows with LF and must still import. A leading BOM is
// consumed rather than left glued to the first header, where it would make a
// column lookup miss.
TR.fromCSV = (text) => {
  let s = String(text == null ? '' : text);
  if (s.charCodeAt(0) === 0xFEFF) s = s.slice(1);
  const rows = [];
  let row = [], field = '', quoted = false, i = 0;
  const endField = () => { row.push(field); field = ''; };
  // A wholly empty line is skipped rather than becoming a row of one empty
  // field, so a trailing newline or a blank separator line doesn't invent data.
  const endRow = () => {
    endField();
    if (row.length > 1 || row[0] !== '') rows.push(row);
    row = [];
  };
  while (i < s.length) {
    const c = s[i];
    if (quoted) {
      if (c === '"') {
        if (s[i + 1] === '"') { field += '"'; i += 2; continue; }   // doubled = literal
        quoted = false; i++; continue;
      }
      field += c; i++; continue;
    }
    // Only opens a quoted field at the start of one; a quote later in the field
    // is an ordinary character, which is how malformed input stays readable.
    if (c === '"' && field === '') { quoted = true; i++; continue; }
    if (c === ',') { endField(); i++; continue; }
    if (c === '\r' && s[i + 1] === '\n') { endRow(); i += 2; continue; }
    if (c === '\n' || c === '\r') { endRow(); i++; continue; }
    field += c; i++;
  }
  if (field !== '' || row.length) endRow();

  // A quoted field that is still open at EOF means the text was never RFC 4180
  // in the first place — it's a legacy export (or a hand-edited file) written
  // before toCSV existed, with comments dumped raw and unquoted. There, a
  // comment that merely *begins* with a literal " (e.g. `"great try`) opens a
  // quoted field that nothing ever closes, and the parser above — reasonably,
  // for real RFC 4180 — consumes every comma and newline from there to EOF
  // into that one field. A three-row file becomes one row with two rows'
  // worth of data silently swallowed into a comment.
  //
  // The old split(',')/split('\n') importer this replaced had no concept of
  // quoting at all, so the same file just mangled the one field with the
  // stray quote and kept every row. That bounded, visible damage is strictly
  // better than losing the file, so when we detect this shape, re-parse in
  // that old, quote-blind mode instead: every " literal, rows on newlines,
  // fields on commas. Well-formed RFC 4180 text never ends mid-quote, so this
  // branch is a no-op for every file toCSV produced.
  if (quoted) {
    return s.split(/\r\n|\r|\n/)
      .filter(l => l.trim() !== '')
      .map(l => l.split(','));
  }
  return rows;
};

// The exact inverse of csvCell's formula guard.
//
// csvCell only ever prefixes an apostrophe when the TRIMMED value starts with
// one of ' = + - @, so stripping one is correct exactly when the remainder
// still does. That precision is the point: a hand-typed "'19 season" —
// apostrophe then a digit — is left alone, because csvCell could not have
// produced it. Stripping unconditionally would eat that apostrophe.
TR.csvUnguard = (v) => {
  const s = v == null ? '' : String(v);
  if (s[0] !== "'") return s;
  const rest = s.slice(1);
  return /^['=+\-@]/.test(rest.trim()) ? rest : s;
};
