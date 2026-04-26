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
