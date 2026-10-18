import { createLetterSoup, createCodeword, normalizeWord } from './generator.js';
import { renderWordSearch } from './ui-wordsearch.js';
import { renderCodeword } from './ui-codeword.js';
import { loadSettings, saveSettings, loadPuzzles, upsertPuzzle, deletePuzzle, getPuzzle, newId } from './state.js';
import { APP_VERSION, checkForUpdate, applyUpdate } from './updater.js';
import { RNG } from './rng.js';
import { t, setLang, getLang, detectLanguage, dateLocale, LANGUAGES, applyStaticTranslations } from './i18n.js';

const $ = sel => document.querySelector(sel);
const settings = loadSettings();
let currentPuzzleId = null;

// Embedded example word lists (work fully offline)
const EXAMPLES = {
  palavras: `reino\nafonso\ncastelo\nleão\ncombate\nmuçulmano\nhenrique\nraimundo\nfrança\nnobre\nvalente\nrei\npresente\ncondado\ngaliza\nportucalense\nurraca\nteresa\nlei\nfilho\nforte\ncavaleiro\nportugal\ncastelhano\nbatalha\nferoz\nguimaraes\nzamora`,
  bluey: `Bluey\nBingo\nBandit\nChilli\nMuffin\nSocks\nNana\nGrandad\nCoco\nSnickers\nMackenzie\nChloe\nRusty\nIndy\nLucky\nWinton\nJudo\nMissy\nHoney\nBuddy\nAlfie\nCaptain\nDoreen\nCalypso\nPat\nJanelle\nRad\nFrisky\nMort\nMaynard\nWendy\nLila\nJack\nMia`,
  livros: `BIBLIOTECA\nHISTORIAS\nAVENTURAS\nHEROIS\nAPRENDER\nIMAGINAR\nLEITURA\nFOLHAS\nCAPA\nCONHECER`,
  martinho: `cavaleiro\nMartinho\ndia\nchuva\npobre\nfome\nfrio\nesmola\nespada\ncorte\ncapa\ncastanhas\nprova\nvinho`,
};

// ---------- helpers ----------
function toast(msg, ms = 2600) {
  const tt = $('#toast');
  tt.textContent = msg;
  tt.classList.add('show');
  clearTimeout(tt._h);
  tt._h = setTimeout(() => tt.classList.remove('show'), ms);
}

function applyTheme() {
  const dark = settings.theme === 'dark' ||
    (settings.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', dark);
}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);

const screens = ['home', 'setup', 'play', 'settings'];
let currentScreen = 'home';
function show(name) {
  currentScreen = name;
  for (const s of screens) $('#screen-' + s).classList.toggle('active', s === name);
}

// ---------- language ----------
function renderLangMenu() {
  const menu = $('#lang-menu');
  menu.innerHTML = '';
  for (const [code, info] of Object.entries(LANGUAGES)) {
    const b = document.createElement('button');
    b.className = 'lang-option' + (code === getLang() ? ' active' : '');
    b.innerHTML = `<span class="flag">${info.flag}</span> ${info.name}`;
    b.onclick = () => {
      $('#lang-menu').classList.add('hidden');
      setLanguage(code);
    };
    menu.appendChild(b);
  }
}

function setLanguage(code) {
  settings.language = code;
  saveSettings(settings);
  setLang(code);
  $('#btn-lang').textContent = LANGUAGES[getLang()].flag;
  renderLangMenu();
  applyStaticTranslations();
  // Re-render dynamic content for the visible screen
  if (currentScreen === 'home') renderHome();
  else if (currentScreen === 'play' && currentPuzzleId) openPuzzle(currentPuzzleId);
}

// ---------- HOME ----------
function renderHome() {
  const list = $('#puzzle-list');
  const puzzles = loadPuzzles().sort((a, b) => b.updatedAt - a.updatedAt);
  list.innerHTML = '';
  if (!puzzles.length) {
    list.innerHTML = `<li class="empty">${t('noPuzzles')}</li>`;
  }
  for (const p of puzzles) {
    const li = document.createElement('li');
    const done = p.state && p.state.done ? ' ✅' : '';
    const date = new Date(p.createdAt).toLocaleDateString(dateLocale());
    const typeLabel = p.type === 'codeword' ? t('codeword') : t('wordsearch');
    const dims = p.data.grid || p.data.numberGrid;
    li.innerHTML = `
      <button class="puzzle-item">
        <span class="puzzle-name">${escapeHtml(p.name)}${done}</span>
        <span class="puzzle-meta">${typeLabel} · ${dims.length}×${dims[0].length} · ${date}</span>
      </button>
      <button class="btn-icon del" title="🗑">🗑</button>`;
    li.querySelector('.puzzle-item').onclick = () => openPuzzle(p.id);
    li.querySelector('.del').onclick = () => {
      if (confirm(t('deleteConfirm', { name: p.name }))) { deletePuzzle(p.id); renderHome(); }
    };
    list.appendChild(li);
  }
}

// ---------- SETUP ----------
function parseWordInput(text) {
  return text.split('\n')
    .map(l => l.trim())
    .filter(l => l && !l.startsWith('#'));
}

function dedupe(words) {
  const seen = new Set();
  return words.filter(w => {
    const n = normalizeWord(w);
    if (!n || seen.has(n)) return false;
    seen.add(n);
    return true;
  });
}

function friendlyError(e) {
  if (e.message && e.message.startsWith('err')) return t(e.message, e.params || {});
  return e.message;
}

function buildPuzzles(form) {
  const words = dedupe(parseWordInput(form.wordsText));
  if (!words.length) { toast(t('errNeedWords')); return null; }

  const seed = form.seed !== '' ? +form.seed : Math.floor(Math.random() * 1e9);
  const name = form.name || `${t('puzzleName')} ${new Date().toLocaleDateString(dateLocale())}`;
  const created = [];

  const doWordSearch = form.type === 'wordsearch' || form.type === 'both';
  const doCodeword = form.type === 'codeword' || form.type === 'both';

  let messageWordsOrig = [];
  if (form.secretType === 'words') {
    messageWordsOrig = dedupe(parseWordInput(form.secretWordsText));
    if (!messageWordsOrig.length) { toast(t('errNeedSecretWords')); return null; }
  }

  if (doWordSearch) {
    let res;
    try {
      res = createLetterSoup(words, {
        rows: form.rows, cols: form.cols,
        allowDiagonals: form.diagonals,
        allowBackwards: form.backwards,
        enforceIntersections: form.intersections,
        seed,
        secretMessage: form.secretType === 'letters' ? form.secretText : null,
      });
    } catch (e) {
      toast(friendlyError(e), 4000);
      return null;
    }
    let wordsDisplay = words.filter(w => !res.unplaced.includes(normalizeWord(w)));
    let secretMessage = null;
    if (form.secretType === 'words' && messageWordsOrig.length) {
      wordsDisplay = dedupe([...wordsDisplay, ...messageWordsOrig]);
      new RNG(seed).shuffle(wordsDisplay);
      secretMessage = { type: 'words', words: messageWordsOrig };
    } else if (form.secretType === 'letters' && form.secretText.trim()) {
      secretMessage = { type: 'letters', text: form.secretText.trim() };
      if (!res.messageCells.length) {
        toast(t('errMsgTooLong'));
        secretMessage = null;
      }
    }
    created.push({
      id: newId(), type: 'wordsearch', name: doCodeword ? `${name} ${t('suffixSoup')}` : name,
      createdAt: Date.now(), updatedAt: Date.now(),
      config: form,
      data: { grid: res.grid, placements: res.placements, wordsDisplay, secretMessage, messageCells: res.messageCells, unplaced: res.unplaced },
      state: { found: [], elapsed: 0, done: false },
    });
    if (res.unplaced.length) toast(t('errUnplaced', { words: res.unplaced.join(', ') }), 4000);
  }

  if (doCodeword) {
    try {
      const res = createCodeword(words, {
        rows: form.cwRows, cols: form.cwCols,
        numHints: form.cwHints, seed: seed + 1,
      });
      const wordsDisplay = words.filter(w => !res.unplaced.includes(normalizeWord(w)));
      created.push({
        id: newId(), type: 'codeword', name: doWordSearch ? `${name} ${t('suffixCodeword')}` : name,
        createdAt: Date.now(), updatedAt: Date.now(),
        config: form,
        data: {
          numberGrid: res.numberGrid, letterGrid: res.letterGrid,
          placements: res.placements, wordsDisplay, hints: res.hints,
          unplaced: res.unplaced, missingLetters: res.missingLetters,
        },
        state: { entries: {}, elapsed: 0, done: false },
      });
      if (res.unplaced.length) toast(t('errCwUnplaced', { words: res.unplaced.join(', ') }), 4000);
    } catch (e) {
      toast(t('errCwFailed', { msg: friendlyError(e) }), 4000);
    }
  }

  return created;
}

function initSetup() {
  const f = $('#setup-form');
  f.elements.example.onchange = e => {
    const v = e.target.value;
    if (v && EXAMPLES[v]) f.elements.words.value = EXAMPLES[v];
  };
  f.elements.import.onchange = e => {
    const file = e.target.files[0];
    if (!file) return;
    const rd = new FileReader();
    rd.onload = () => { f.elements.words.value = rd.result; };
    rd.readAsText(file);
    e.target.value = '';
  };
  f.elements.secretType.onchange = e => {
    const v = e.target.value;
    $('#secret-letters-row').classList.toggle('hidden', v !== 'letters');
    $('#secret-words-row').classList.toggle('hidden', v !== 'words');
  };
  f.onsubmit = e => {
    e.preventDefault();
    const el = f.elements;
    const form = {
      name: el.pname.value.trim(),
      type: el.ptype.value,
      wordsText: el.words.value,
      rows: Math.min(25, Math.max(5, +el.rows.value || 15)),
      cols: Math.min(25, Math.max(5, +el.cols.value || 15)),
      diagonals: el.diagonals.checked,
      backwards: el.backwards.checked,
      intersections: el.intersections.checked,
      cwRows: Math.min(25, Math.max(5, +el.cwRows.value || 15)),
      cwCols: Math.min(25, Math.max(5, +el.cwCols.value || 15)),
      cwHints: Math.min(26, Math.max(0, +el.cwHints.value || 3)),
      seed: el.seed.value.trim(),
      secretType: el.secretType.value,
      secretText: el.secretText.value,
      secretWordsText: el.secretWords.value,
    };
    const created = buildPuzzles(form);
    if (!created || !created.length) return;
    created.forEach(upsertPuzzle);
    openPuzzle(created[0].id);
  };
}

// ---------- PLAY ----------
function openPuzzle(id) {
  const p = getPuzzle(id);
  if (!p) { toast(t('errPuzzleNotFound')); return; }
  currentPuzzleId = id;
  show('play');
  const area = $('#play-area');
  if (area._cleanup) area._cleanup();
  const ctx = { settings, onExit: () => { currentPuzzleId = null; renderHome(); show('home'); }, toast };
  if (p.type === 'codeword') renderCodeword(area, p, ctx);
  else renderWordSearch(area, p, ctx);
}

// ---------- SETTINGS ----------
function initSettings() {
  const s = settings;
  $('#set-highlight').value = s.highlight;
  $('#set-theme').value = s.theme;
  $('#set-timer').checked = s.timer;
  $('#set-autocheck').checked = s.codewordAutoCheck;

  $('#set-highlight').onchange = e => { s.highlight = e.target.value; saveSettings(s); };
  $('#set-theme').onchange = e => { s.theme = e.target.value; saveSettings(s); applyTheme(); };
  $('#set-timer').onchange = e => { s.timer = e.target.checked; saveSettings(s); };
  $('#set-autocheck').onchange = e => { s.codewordAutoCheck = e.target.checked; saveSettings(s); };

  $('#app-version').textContent = `v${APP_VERSION}`;
  $('#btn-check-update').onclick = async () => {
    const st = $('#update-status');
    st.textContent = t('checking');
    const res = await checkForUpdate();
    if (res.status === 'available') {
      st.textContent = t('newVersion', { v: res.version });
      $('#btn-update').classList.remove('hidden');
    } else if (res.status === 'current') {
      st.textContent = t('latestVersion');
    } else if (res.status === 'offline') {
      st.textContent = t('noConnection');
    } else {
      st.textContent = t('checkFailed');
    }
  };
  $('#btn-update').onclick = async () => {
    $('#update-status').textContent = t('updating');
    await applyUpdate();
  };
}

// ---------- boot ----------
function init() {
  applyTheme();
  setLang(settings.language || detectLanguage());
  applyStaticTranslations();
  $('#btn-lang').textContent = LANGUAGES[getLang()].flag;
  renderLangMenu();

  $('#btn-lang').onclick = e => {
    e.stopPropagation();
    $('#lang-menu').classList.toggle('hidden');
  };
  document.addEventListener('click', e => {
    if (!e.target.closest('.lang-wrap')) $('#lang-menu').classList.add('hidden');
  });

  $('#btn-new').onclick = () => show('setup');
  $('#btn-nav-settings').onclick = () => show('settings');
  $('#btn-setup-cancel').onclick = () => show('home');
  $('#btn-settings-back').onclick = () => show('home');
  initSetup();
  initSettings();
  renderHome();

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').then(() => {
      checkForUpdate().then(res => {
        if (res.status === 'available') {
          toast(t('updateAvailableToast', { v: res.version }), 5000);
        }
      });
    }).catch(() => {});
  }

  show('home');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

init();
