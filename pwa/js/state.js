// Persistence layer: settings + saved puzzles in localStorage.
const SETTINGS_KEY = 'slp_settings';
const PUZZLES_KEY = 'slp_puzzles';

const DEFAULT_SETTINGS = {
  highlight: 'colors',   // 'colors' | 'single' | 'subtle'
  theme: 'auto',         // 'auto' | 'light' | 'dark'
  language: null,        // null = auto-detect from browser
  timer: true,
  codewordAutoCheck: false,
  sound: false,
};

export function loadSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {}) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    return true;
  } catch {
    return false;
  }
}

export function loadPuzzles() {
  try {
    return JSON.parse(localStorage.getItem(PUZZLES_KEY)) || [];
  } catch {
    return [];
  }
}

export function savePuzzles(list) {
  try {
    localStorage.setItem(PUZZLES_KEY, JSON.stringify(list));
    return true;
  } catch {
    return false;
  }
}

export function getPuzzle(id) {
  return loadPuzzles().find(p => p.id === id) || null;
}

export function upsertPuzzle(p) {
  const list = loadPuzzles();
  const i = list.findIndex(x => x.id === p.id);
  p.updatedAt = Date.now();
  if (i >= 0) list[i] = p; else list.push(p);
  return savePuzzles(list);
}

export function deletePuzzle(id) {
  savePuzzles(loadPuzzles().filter(p => p.id !== id));
}

export function newId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}
