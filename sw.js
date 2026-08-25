// Touch Rugby Analyzer service worker.
// Caches the static app shell so pages (notably the field annotator, used
// pitchside on flaky connections) load offline. Only same-origin GET requests
// are handled — cross-origin calls (Apps Script backend, clip service, CDNs,
// YouTube) always go straight to the network so live data is never stale.
//
// Bump CACHE_VERSION whenever shell assets change to force a refresh.
const CACHE_VERSION = 'trl-shell-v6';

const SHELL = [
  'index.html',
  'viewer.html',
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
  'manifest.json',
  'favicon-16x16.png',
  'favicon-32x32.png',
  'apple-touch-icon.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      // Tolerate individual asset failures so one missing file doesn't abort install.
      .then((cache) => Promise.allSettled(SHELL.map((url) => cache.add(url))))
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
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE_VERSION).then((c) => c.put(req, copy));
          return res;
        })
        // Offline fallback: cached page, then app shell, then a synthetic
        // response — respondWith() throws if it ever resolves to undefined.
        .catch(() => caches.match(req)
          .then((hit) => hit || caches.match('index.html'))
          .then((hit) => hit || new Response(
            '<h1>Offline</h1><p>No cached copy of this page is available.</p>',
            { status: 503, statusText: 'Offline', headers: { 'Content-Type': 'text/html; charset=utf-8' } }
          )))
    );
    return;
  }

  // Static assets (js, icons): cache-first, refresh in the background.
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
