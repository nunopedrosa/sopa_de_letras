// Version check + service worker update flow.
// version.json is fetched network-only (the SW special-cases it),
// compared against APP_VERSION, and an Update button triggers
// skipWaiting -> reload (with a cache-bust fallback).
export const APP_VERSION = '1.2.3';

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

  // If the new worker is slow to install, keep waiting. Deleting caches
  // here raced that install and forced a full re-download.
  let extraWaits = 0;
  const giveUp = async () => {
    if (reloaded) return;
    const current = await navigator.serviceWorker.getRegistration();
    if (current && (current.installing || current.waiting) && extraWaits < 2) {
      extraWaits++;
      const worker = current.waiting || current.installing;
      if (worker && navigator.serviceWorker.controller) worker.postMessage({ type: 'SKIP_WAITING' });
      setTimeout(giveUp, 10000);
      return;
    }
    reload();
  };
  setTimeout(giveUp, 8000);
}
