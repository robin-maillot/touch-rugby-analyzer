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
