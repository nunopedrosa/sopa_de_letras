// Version check + service worker update flow.
// version.json is fetched network-only (the SW special-cases it),
// compared against APP_VERSION, and an Update button triggers
// skipWaiting -> reload (with a cache-bust fallback).
export const APP_VERSION = '1.2.2';

export async function checkForUpdate() {
  if (!('serviceWorker' in navigator)) return { status: 'unsupported' };
  try {
    const res = await fetch(`version.json?v=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) return { status: 'error' };
    const { version } = await res.json();
    if (version && version !== APP_VERSION) {
      return { status: 'available', version };
    }
    return { status: 'current', version: APP_VERSION };
  } catch {
    return { status: 'offline' };
  }
}

export async function applyUpdate() {
  const reg = await navigator.serviceWorker.getRegistration();

  // Force any newly installed SW to activate, then reload.
  let reloaded = false;
  const reload = () => { if (!reloaded) { reloaded = true; location.reload(); } };
  navigator.serviceWorker.addEventListener('controllerchange', reload, { once: true });

  if (reg) {
    await reg.update();
    if (reg.waiting) reg.waiting.postMessage({ type: 'SKIP_WAITING' });
    else if (reg.installing) {
      reg.installing.addEventListener('statechange', function onState() {
        if (this.state === 'installed' && navigator.serviceWorker.controller) {
          this.postMessage({ type: 'SKIP_WAITING' });
        }
      });
    }
    reg.addEventListener('updatefound', () => {
      const sw = reg.installing;
      if (!sw) return;
      sw.addEventListener('statechange', () => {
        if (sw.state === 'installed' && navigator.serviceWorker.controller) {
          sw.postMessage({ type: 'SKIP_WAITING' });
        }
      });
    });
  }

  // Fallback: if nothing activated within 2.5s, nuke caches and reload.
  setTimeout(async () => {
    if (reloaded) return;
    const keys = await caches.keys();
    await Promise.all(keys.map(k => caches.delete(k)));
    if (reg) await reg.unregister();
    reload();
  }, 2500);
}
