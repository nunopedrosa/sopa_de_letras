// Service worker: precache the app shell; network-only for version.json
// so the in-app update check is always honest.
const CACHE = 'sopa-v1.2.3';

const ASSETS = [
  './',
  'index.html',
  'manifest.webmanifest',
  'css/style.css',
  'js/app.js',
  'js/generator.js',
  'js/i18n.js',
  'js/pdf.js',
  'js/rng.js',
  'js/state.js',
  'js/ui-wordsearch.js',
  'js/ui-codeword.js',
  'js/updater.js',
  'js/vendor/jspdf.umd.min.js',
  'icons/icon-192.png',
  'icons/icon-512.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(ASSETS))
  );
  // Do NOT skipWaiting here — the app tells us when the user accepts the update.
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('message', e => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;

  // version checks always go to the network
  if (url.pathname.endsWith('/version.json') || url.pathname.endsWith('version.json')) {
    e.respondWith(fetch(e.request));
    return;
  }

  // navigations: cached shell first, network fallback
  if (e.request.mode === 'navigate') {
    e.respondWith(
      caches.match('index.html').then(r => r || fetch(e.request))
    );
    return;
  }

  // everything else: cache-first, then network (and populate cache)
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(res => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return res;
      });
    })
  );
});
