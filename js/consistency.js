// Depends on js/events.js (TR.MENU, TR.isTurnover) and js/possession.js
// (TR.inferPossessionAfter, TR.inferActionOwner).
//
// Possession-chain consistency: every non-To-Review event's possessionOwner
// should equal inferPossessionAfter(prev.possessionOwner, prev.type, prev.name)
// where prev is the most recent non-To-Review event with a recorded owner.
// The first such event seeds the chain.
//
// Each `Game Start` re-seeds the chain (one per half): any team can start any
// half, so a Game Start's recorded owner is taken at face value, never flagged,
// and the chain for that half is validated from it.
//
// Crucially, the chain advances from the *expected* owner (not the recorded
// one). So a single wrong override is flagged only on that one row — it does
// not "infect" every downstream row.

// Returns Map<annotationId, expectedPossessionOwner>.
// Annotations that have no expected owner (e.g. before the seed is set, or
// To Review events) are not present in the map.
TR.computeExpectedOwners = (annotations) => {
  const map = new Map();
  if (!Array.isArray(annotations)) return map;
  const sorted = annotations.filter(Boolean).sort((a, b) => (a.time || 0) - (b.time || 0));
  let state = '';
  for (const ann of sorted) {
    if (!ann || ann.type === 'To Review') continue;
    // A Game Start opens a new half — any team can start it, so re-seed.
    if (ann.type === 'Game Event' && ann.name === 'Game Start') state = '';
    if (!state) {
      if (ann.possessionOwner) {
        state = ann.possessionOwner;
        map.set(ann.id, ann.possessionOwner);
      }
    } else {
      map.set(ann.id, state);
    }
    if (state) state = TR.inferPossessionAfter(state, ann.type, ann.name);
  }
  return map;
};

// Returns Set<annotationId> of events whose recorded possessionOwner differs
// from their chain-expected owner.
TR.getInconsistentIds = (annotations, expected) => {
  expected = expected || TR.computeExpectedOwners(annotations);
  const ids = new Set();
  if (!Array.isArray(annotations)) return ids;
  for (const a of annotations) {
    if (!a || !expected.has(a.id)) continue;
    if (a.possessionOwner !== expected.get(a.id)) ids.add(a.id);
  }
  return ids;
};

// Mutates `annotations` in place: rewrites every non-To-Review event's
// possessionOwner & actionOwner to follow the inferred chain. The first event
// with a recorded possessionOwner is preserved as the seed, and every Game
// Start re-seeds the chain for its half (its recorded owner is preserved).
// Returns the count of events whose possessionOwner changed (excluding seeds).
TR.applyConsistencyFix = (annotations) => {
  if (!Array.isArray(annotations)) return 0;
  const sorted = annotations.filter(Boolean).sort((a, b) => (a.time || 0) - (b.time || 0));
  let state = '';
  let changed = 0;
  for (const ann of sorted) {
    if (!ann || ann.type === 'To Review') continue;
    // A Game Start opens a new half — keep its recorded owner as the new seed.
    if (ann.type === 'Game Event' && ann.name === 'Game Start') state = '';
    if (!state) {
      if (ann.possessionOwner) state = ann.possessionOwner;
    } else {
      if (ann.possessionOwner !== state) changed++;
      ann.possessionOwner = state;
    }
    ann.actionOwner = TR.inferActionOwner(ann.possessionOwner, ann.type, ann.name);
    if (state) state = TR.inferPossessionAfter(state, ann.type, ann.name);
  }
  return changed;
};
