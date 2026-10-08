// Touch Rugby Analyzer service worker.
// Caches the static app shell so pages (notably the field annotator, used
// pitchside on flaky connections) load offline. Only same-origin GET requests
// are handled — cross-origin calls (Apps Script backend, clip service, CDNs,
// YouTube) always go straight to the network so live data is never stale.
//
// Bump CACHE_VERSION whenever shell assets change to force a refresh.
const CACHE_VERSION = 'trl-shell-v46';

const SHELL = [
  'index.html',
  'viewer.html',
  'game.html',
  'games.html',
  'dashboard.html',
  'analytics.html',
  'annotator.html',
  'annotator_field.html',
  'annotator_field2.html',
  'backfill.html',
  'live.html',
  'js/config.js',
  'js/utils.js',
  'js/events.js',
  'js/possession.js',
  'js/field_games.js',
  'js/player.js',
  'js/consistency.js',
  'js/strike_moves.js',
  'js/playlists.js',
  'js/field_stats.js',
  'css/theme.css',
  'css/field_stats.css',
  'manifest.json',
  'favicon-16x16.png',
  'favicon-32x32.png',
  'apple-touch-icon.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      // Tolerate individual asset failures so one missing file doesn't abort install.
      // cache:'reload' skips the browser's HTTP cache: GitHub Pages serves with
      // max-age=600, so a plain add() in the minutes after a deploy could fill
      // the brand-new cache with the previous deploy's scripts.
      .then((cache) => Promise.allSettled(SHELL.map((url) => cache.add(new Request(url, { cache: 'reload' })))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  // Only manage our own origin's assets; let everything else hit the network.
  if (url.origin !== self.location.origin) return;

  // Navigations / HTML: network-first so deploys show up, cache as offline fallback.
  const isHTML = req.mode === 'navigate' ||
    (req.headers.get('accept') || '').includes('text/html');
  if (isHTML) {
    event.respondWith(
      // 'no-cache': GitHub Pages serves pages with max-age=600, so a plain
      // fetch could hand back the browser's copy of a page for up to ten minutes
      // after a deploy. Revalidating costs one cheap 304 when nothing changed.
      // A navigation can't be refetched with options as-is (its redirect mode
      // is 'manual'), so it's rebuilt from its URL; a redirect it meets — e.g.
      // the bare repo URL gaining its trailing slash — is passed back as one.
      fetch(new Request(req.url, { cache: 'no-cache', credentials: 'same-origin', redirect: 'follow' }))
        .then((res) => {
          if (res.redirected) return Response.redirect(res.url, 302);
          if (res.ok) {
            const copy = res.clone();
            caches.open(CACHE_VERSION).then((c) => c.put(req, copy));
          }
          return res;
        })
        // Offline fallback: cached page (ignoring ?game=… and the like), then the
        // app shell, then a synthetic response — respondWith() throws if it ever
        // resolves to undefined.
        .catch(() => caches.match(req, { ignoreSearch: true })
          .then((hit) => hit || caches.match('index.html'))
          .then((hit) => hit || new Response(
            '<h1>Offline</h1><p>No cached copy of this page is available.</p>',
            { status: 503, statusText: 'Offline', headers: { 'Content-Type': 'text/html; charset=utf-8' } }
          )))
    );
    return;
  }

  // Scripts: network-first, like pages. A page and the shared js it calls change
  // together, so serving a fresh page with a cached script breaks it — the page
  // calls functions the old script doesn't have, on the first load after every
  // deploy. 'no-cache' revalidates with the server (a cheap 304 when nothing
  // changed) instead of trusting the HTTP cache. Offline, the cached copy is
  // used, which matches the cached page it's paired with.
  if (url.pathname.endsWith('.js')) {
    event.respondWith(
      fetch(req, { cache: 'no-cache' })
        .then((res) => {
          if (res && res.ok) {
            const copy = res.clone();
            caches.open(CACHE_VERSION).then((c) => c.put(req, copy));
          }
          return res;
        })
        .catch(() => caches.match(req).then((hit) => hit || Response.error()))
    );
    return;
  }

  // Other static assets (icons, manifest): cache-first, refresh in the background.
  event.respondWith(
    caches.match(req).then((hit) => {
      const network = fetch(req)
        .then((res) => {
          if (res && res.ok) {
            const copy = res.clone();
            caches.open(CACHE_VERSION).then((c) => c.put(req, copy));
          }
          return res;
        })
        .catch(() => hit);
      // Never resolve to undefined: respondWith() would throw a TypeError.
      return hit || network.then((res) => res || Response.error());
    })
  );
});
