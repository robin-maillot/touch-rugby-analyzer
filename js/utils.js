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

// Strip a leading byte-order mark. Shared by both parse modes below — the
// wrong place to do this is inside a parsing loop, where a stray BOM would
// otherwise glue itself to the first header cell.
function stripBOM(text) {
  const s = String(text == null ? '' : text);
  return s.charCodeAt(0) === 0xFEFF ? s.slice(1) : s;
}

// The strict RFC 4180 loop, factored out so TR.fromCSV can run it and then
// decide, rather than committing to its output unconditionally. Unchanged in
// substance from the parser this file has always had.
function fromCSVStrict(s) {
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
  return rows;
}

// True when every row has as many fields as the first. Every file this app
// has ever written — old unquoted exports and new quoted ones alike — has
// this shape, header included: it's what "a CSV" means here, independent of
// how the text got sliced into rows.
function isRectangular(rows) {
  return rows.length > 0 && rows.every(r => r.length === rows[0].length);
}

// The old importer this file replaced, before toCSV/fromCSV existed: no
// quoting at all, rows on any newline, fields on commas. Kept alive — not
// deleted — as the second half of the two-parse strategy in TR.fromCSV below,
// and exposed on its own so that half can be tested in isolation: feed it a
// quoted field and the quotes come back out as ordinary characters, which is
// exactly the property that makes it a safe fallback for text that was never
// RFC 4180 in the first place.
TR.fromCSVLoose = (text) => {
  const s = stripBOM(text);
  return s.split(/\r\n|\r|\n/)
    .filter(l => l.trim() !== '')
    .map(l => l.split(','));
};

// Parse CSV text into rows of fields — the counterpart to TR.toCSV.
//
// This runs BOTH a real RFC 4180 parser (fromCSVStrict, a real parser rather
// than a pair of splits — a field may legitimately hold a comma, a doubled
// quote or a newline, and splitting on those turns one row into several and
// truncates the field that held them) and the old quote-blind splitter
// (fromCSVLoose) that this file's exporter used before toCSV existed, then
// picks whichever result actually looks like the table this app writes.
//
// Why bother running both, instead of asking the strict parser "did something
// go wrong?" A prior version tried exactly that, by checking whether a quoted
// field was still open at EOF. That's a parity check — it only notices an
// ODD number of quotes in the whole file — not a correctness check, and it
// fails on completely ordinary input: a legacy file's first stray quote opens
// a field that swallows the rest of the document, but if any LATER row
// happens to contain a balanced quoted phrase (someone typing `he said "wow"
// nice` in a Comment), the quote count returns to even, the parser looks
// "closed" at EOF, and the merge happens silently — the exact bug this
// function exists to catch slips through the check meant to catch it.
//
// Quote counting can't tell a mangled file from a clean one because it never
// looks at the RESULT, only at how the scan ended. Shape does: files this app
// writes are rectangular, every row carrying exactly the header's field
// count, and a legacy file with a stray quote makes the strict parser
// produce something that ISN'T — a field balloons to swallow whole
// neighbouring rows. So: parse both ways, and if the strict parse lost rows
// relative to a dumb per-line split (or isn't even internally rectangular)
// while the loose parse IS a rectangular, genuinely multi-column table
// (single-column data never comes from this app — real exports are always
// several fields wide, so a one-column "rectangle" out of the loose parse
// isn't good evidence of anything), trust the loose parse instead. Otherwise
// keep the strict result — including when both parses are ragged, which
// means the file is just a ragged file and re-parsing it differently won't
// help.
//
// This restores, on purpose, exactly the behaviour `main` had before toCSV
// existed: every legacy file quote-splits the same way it always did. The
// fallback is a restoration, not a new risk.
TR.fromCSV = (text) => {
  const s = stripBOM(text);
  const strictRows = fromCSVStrict(s);
  const looseRows = TR.fromCSVLoose(s);

  const strictIntact = isRectangular(strictRows) && strictRows.length === looseRows.length;
  if (!strictIntact && isRectangular(looseRows) && looseRows[0].length > 1) return looseRows;
  return strictRows;
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
