var TR = {}; // var so it's a property of the global object (window in browsers, vm context in tests)

TR.APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbzhMw4nQr2nVW0MLfPWVXRPXsvImerpzshQ5GnJ3873qQkGMz0bcJQdmTRXCxwygfFm/exec';
TR.CLIP_SERVICE_URL = 'https://m30-clipper-1070277967282.europe-west1.run.app';

// Persistent login: a rolling 30-day session is stored in localStorage and
// hydrated into sessionStorage on every page load so the rest of the app
// (which still reads from sessionStorage) keeps working unchanged.
TR.SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;

(function hydrateSession() {
  try {
    if (sessionStorage.getItem('password')) return;
    const raw = localStorage.getItem('trl2_session');
    if (!raw) return;
    const s = JSON.parse(raw);
    if (!s || !s.password || !s.expires || s.expires < Date.now()) {
      localStorage.removeItem('trl2_session');
      return;
    }
    sessionStorage.setItem('password', s.password);
    if (s.role) sessionStorage.setItem('role', s.role);
  } catch (e) {
    try { localStorage.removeItem('trl2_session'); } catch {}
  }
})();

TR.saveSession = (password, role) => {
  try {
    sessionStorage.setItem('password', password);
    if (role) sessionStorage.setItem('role', role);
    localStorage.setItem('trl2_session', JSON.stringify({
      password,
      role: role || null,
      expires: Date.now() + TR.SESSION_TTL_MS,
    }));
  } catch (e) {}
};

TR.logout = () => {
  try {
    sessionStorage.removeItem('password');
    sessionStorage.removeItem('role');
    localStorage.removeItem('trl2_session');
  } catch (e) {}
  window.location.replace('index.html');
};

TR.secret = () => sessionStorage.getItem('password') || '';

TR.role = () => {
  const p = sessionStorage.getItem('password');
  return p === 'm30-admin' ? 'admin'
       : p === 'm30-staff' ? 'staff'
       : p === 'm30'       ? 'viewer' : 'anon';
};

// Umami pages set data-auto-track="false" so we send pageviews manually with
// role merged in. identify() tags the session; track() emits the pageview.
TR.umamiIdentify = (extra) => {
  const data = Object.assign({ role: TR.role() }, extra || {});
  const send = () => {
    if (!window.umami) return;
    if (window.umami.identify) window.umami.identify(data);
    if (window.umami.track) window.umami.track(props => Object.assign({}, props, data));
  };
  if (window.umami && window.umami.track) send();
  else if (typeof window.addEventListener === 'function') window.addEventListener('load', send);
};

// Redirect to index.html if the stored password doesn't meet the required role.
// role: 'viewer' | 'staff' | 'admin'
TR.auth = (role) => {
  const p = sessionStorage.getItem('password');
  const ok = {
    viewer: p === 'm30' || p === 'm30-staff' || p === 'm30-admin',
    staff:  p === 'm30-staff' || p === 'm30-admin',
    admin:  p === 'm30-admin',
  };
  if (!ok[role]) window.location.replace('index.html');
};

// Wipe per-user localStorage caches (action=list / action=all / game_*) when
// the password changes from what was in use last time these caches were written.
// Server-side responses are now filtered by group, so caches from a prior user
// would otherwise leak through the version-skip fast-path. Preference keys
// (linkOffset, layout, etc.) are explicitly left alone.
(function wipeStaleCaches() {
  try {
    const cur  = sessionStorage.getItem('password') || '';
    if (!cur) return;
    const last = localStorage.getItem('trl2_user') || '';
    if (cur === last) return;
    Object.keys(localStorage)
      .filter(k => k === 'trl2_list' || k === 'trl2_all' || k.indexOf('trl2_game_') === 0)
      .forEach(k => localStorage.removeItem(k));
    localStorage.setItem('trl2_user', cur);
  } catch (e) {}
})();

// Emit a single role-tagged pageview as soon as config.js loads.
TR.umamiIdentify();
