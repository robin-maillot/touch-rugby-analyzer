// Depends on js/config.js (TR namespace)

// Canonical event-type → sub-type map (source of truth for all pages).
// viewer.html previously had a divergent NAMES_BY_TYPE; TR.MENU is the canonical version.
TR.MENU = {
  'Penalty Attack':  ['Forward Pass', 'Touch and Pass', 'Off the Mark', 'Delay of Play', 'Hard Touch', 'Other'],
  'Penalty Defence': ['Offside', 'Hard Touch', 'In the Ruck', 'Not Moving Forward', 'Delay the play', 'Other'],
  'Turnover':        ['Ball Down', '6th Touch', 'Dummy Touch', 'Bad Roll', '6 Again', 'Interception', 'Other'],
  'Game Event':      ['Game Start', 'Game End'],
  'Try':             ['Scoop', 'Other', '32 - Long', '33 Quicky', '33', '32 Cut', 'French Flair'],
  'To Review':       [],
};

TR.NAMES_BY_TYPE = TR.MENU;

// Returns true if this event causes a possession switch.
TR.isTurnover = (type, name) => {
  if (type === 'Try')                              return true;
  if (type === 'Penalty Attack')                   return true;
  if (type === 'Penalty Defence')                  return false;
  if (type === 'Turnover' && name !== '6 Again')   return true;
  return false;
};
