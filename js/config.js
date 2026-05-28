var TR = {}; // var so it's a property of the global object (window in browsers, vm context in tests)

TR.APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbzhMw4nQr2nVW0MLfPWVXRPXsvImerpzshQ5GnJ3873qQkGMz0bcJQdmTRXCxwygfFm/exec';
TR.CLIP_SERVICE_URL = 'https://m30-clipper-1070277967282.europe-west1.run.app';

// Persistent login: a rolling 30-day session is stored in localStorage and
// hydrated into sessionStorage on every page load so the rest of the app
// (which still reads from sessionStorage) keeps working unchanged.
//
// iOS standalone (home-screen) web apps spin up a fresh sessionStorage on every
// navigation and don't reliably flush/share localStorage across those per-page
// webviews, so localStorage alone loses the session as soon as you leave a page.
// A cookie is the one store iOS standalone persists across navigations and
// relaunches, so we mirror the session into a cookie and treat it as the
// last-resort source when both sessionStorage and localStorage come back empty.
TR.SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;
const SESSION_COOKIE = 'trl2_session';

function readSessionCookie() {
  try {
    const m = document.cookie.match(/(?:^|;\s*)trl2_session=([^;]+)/);
    if (!m) return null;
    const s = JSON.parse(decodeURIComponent(m[1]));
    if (!s || !s.password || !s.expires || s.expires < Date.now()) return null;
    return s;
  } catch (e) { return null; }
}

function writeSessionCookie(s) {
  try {
    const secure = location.protocol === 'https:' ? '; Secure' : '';
    const maxAge = Math.max(0, Math.floor((s.expires - Date.now()) / 1000));
    document.cookie = `${SESSION_COOKIE}=${encodeURIComponent(JSON.stringify(s))}; ` +
      `path=/; max-age=${maxAge}; SameSite=Lax${secure}`;
  } catch (e) {}
}

function clearSessionCookie() {
  try {
    const secure = location.protocol === 'https:' ? '; Secure' : '';
    document.cookie = `${SESSION_COOKIE}=; path=/; max-age=0; SameSite=Lax${secure}`;
  } catch (e) {}
}

(function hydrateSession() {
  try {
    if (sessionStorage.getItem('password')) return;
    let s = null;
    try {
      const raw = localStorage.getItem('trl2_session');
      if (raw) s = JSON.parse(raw);
    } catch (e) { s = null; }
    if (!s || !s.password || !s.expires || s.expires < Date.now()) {
      // localStorage missing/expired — fall back to the cookie (the store that
      // survives iOS standalone navigations) and re-seed localStorage from it.
      try { localStorage.removeItem('trl2_session'); } catch {}
      s = readSessionCookie();
      if (!s) { clearSessionCookie(); return; }
      try { localStorage.setItem('trl2_session', JSON.stringify(s)); } catch {}
    }
    sessionStorage.setItem('password', s.password);
    if (s.role) sessionStorage.setItem('role', s.role);
  } catch (e) {
    try { localStorage.removeItem('trl2_session'); } catch {}
  }
})();

TR.saveSession = (password, role) => {
  const s = { password, role: role || null, expires: Date.now() + TR.SESSION_TTL_MS };
  try {
    sessionStorage.setItem('password', password);
    if (role) sessionStorage.setItem('role', role);
    localStorage.setItem('trl2_session', JSON.stringify(s));
  } catch (e) {}
  writeSessionCookie(s);
};

TR.logout = () => {
  try {
    sessionStorage.removeItem('password');
    sessionStorage.removeItem('role');
    localStorage.removeItem('trl2_session');
  } catch (e) {}
  clearSessionCookie();
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

// ── Shared brand mark ──────────────────────────────────────────
// <tr-logo> renders the standard top-left banner logo so every page's header
// shares one definition. Light DOM with self-contained inline styles, so it
// looks identical regardless of the host page's CSS.
//   <tr-logo></tr-logo>          → badge + "Touch Rugby Analyzer" wordmark
//   <tr-logo compact></tr-logo>  → badge only (for pages with their own <h1>)
//   attrs: home="…" (link target, default index.html), label="…" (wordmark)
if (typeof customElements !== 'undefined' && typeof HTMLElement !== 'undefined') {
  class TRLogo extends HTMLElement {
    connectedCallback() {
      if (this._built) return;
      this._built = true;
      const home    = this.getAttribute('home')  || 'index.html';
      const label   = this.getAttribute('label') || 'Touch Rugby Analyzer';
      const compact = this.hasAttribute('compact');
      const a = document.createElement('a');
      a.href = home;
      a.className = 'tr-logo';
      a.style.cssText = 'display:inline-flex;align-items:center;gap:8px;text-decoration:none;flex-shrink:0';
      a.innerHTML =
        '<img src="apple-touch-icon.png" alt="Touch Rugby" ' +
        'style="width:24px;height:24px;border-radius:5px;display:block;flex-shrink:0">' +
        (compact ? '' :
          '<span class="tr-logo-text" style="font-size:1rem;font-weight:600;color:#fff;white-space:nowrap">' +
          label + '</span>');
      this.appendChild(a);
    }
  }
  if (!customElements.get('tr-logo')) customElements.define('tr-logo', TRLogo);
}

// ── Service worker (offline app shell) ─────────────────────────
// Caches static assets so pages load offline; see sw.js. Relative path keeps it
// working under a project subpath on GitHub Pages.
if (typeof navigator !== 'undefined' && 'serviceWorker' in navigator &&
    typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  });
}
