// Depends on js/events.js (TR.isTurnover, TR.MENU)

TR.otherTeam = (team) => team === 'Team 1' ? 'Team 2' : 'Team 1';

// Returns the possession owner after this event is applied.
TR.inferPossessionAfter = (currentOwner, type, name) =>
  TR.isTurnover(type, name) ? TR.otherTeam(currentOwner) : currentOwner;

// Returns who "owns" this action (i.e. who caused it).
// Penalty Defence: the defending team gave away the penalty → other team owns the action.
// Turnover / 6 Again: the defending team conceded → other team owns the action.
// Everything else: the possession-holding team owns the action.
TR.inferActionOwner = (possessionOwner, type, name) => {
  if (type === 'Penalty Defence')                  return TR.otherTeam(possessionOwner);
  if (type === 'Turnover' && name === '6 Again')   return TR.otherTeam(possessionOwner);
  return possessionOwner;
};
