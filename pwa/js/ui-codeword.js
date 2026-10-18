// Codeword play UI: tap a cell, type letters via on-screen or physical
// keyboard. Letters propagate to all cells sharing the number.
import { upsertPuzzle } from './state.js';
import { t } from './i18n.js';
import { printPuzzle } from './pdf.js';

const KB_ROWS = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM'];

export function renderCodeword(container, puzzle, { settings, onExit, toast }) {
  const { numberGrid, letterGrid, wordsDisplay, hints } = puzzle.data;
  const rows = numberGrid.length, cols = numberGrid[0].length;
  const state = puzzle.state;
  state.entries = state.entries || {};

  // Prefill hint letters (locked — they come from the solution)
  for (const [letter, num] of hints) state.entries[num] = letter;
  const hintNums = new Set(hints.map(([, num]) => num));

  // solution lookup: number -> letter (derived from letterGrid + numberGrid)
  const solNumToLetter = {};
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      if (numberGrid[r][c]) solNumToLetter[numberGrid[r][c]] = letterGrid[r][c];

  const usedNums = Object.keys(solNumToLetter).map(Number).sort((a, b) => a - b);

  container.innerHTML = `
    <div class="play-head">
      <button class="btn ghost" id="btn-back">${t('menu')}</button>
      <div class="play-title">${escapeHtml(puzzle.name)}</div>
      <div class="play-stats"><span id="cw-count"></span>
        <span id="timer" class="timer ${settings.timer ? '' : 'hidden'}">0:00</span></div>
    </div>
    <div class="play-actions">
      <button class="btn" id="btn-check">✔ ${t('check')}</button>
      <button class="btn" id="btn-sol">${t('solution')}</button>
      <button class="btn push-right" id="btn-print">${t('printBtn')}</button>
    </div>
    <div id="cw-grid" class="cw-grid"></div>
    <div id="cw-kb" class="cw-kb"></div>
    <details class="cw-words"><summary>${t('wordsCount', { n: wordsDisplay.length })}</summary>
      <div id="cw-word-list" class="ws-words"></div>
    </details>
    <div id="msg-panel" class="msg-panel hidden"></div>
  `;

  const gridEl = container.querySelector('#cw-grid');
  const kbEl = container.querySelector('#cw-kb');
  const countEl = container.querySelector('#cw-count');
  const timerEl = container.querySelector('#timer');
  gridEl.style.setProperty('--cols', cols);

  const cells = [];
  let selected = null; // [r,c]

  for (let r = 0; r < rows; r++) {
    cells[r] = [];
    for (let c = 0; c < cols; c++) {
      const d = document.createElement('div');
      const num = numberGrid[r][c];
      if (!num) {
        d.className = 'cw-cell black';
      } else {
        d.className = 'cw-cell';
        d.dataset.r = r;
        d.dataset.c = c;
        d.dataset.num = num;
        d.innerHTML = `<span class="cw-num">${num}</span><span class="cw-letter"></span>`;
        d.tabIndex = 0;
      }
      gridEl.appendChild(d);
      cells[r][c] = d;
    }
  }

  // word list
  const wl = container.querySelector('#cw-word-list');
  for (const w of wordsDisplay) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = w;
    wl.appendChild(chip);
  }

  function updateCount() {
    const filled = usedNums.filter(n => state.entries[n]).length;
    countEl.textContent = t('lettersCount', { f: filled, t: usedNums.length });
  }

  function paint() {
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const el = cells[r][c];
        const num = el.dataset && el.dataset.num;
        if (!num) continue;
        const span = el.querySelector('.cw-letter');
        const letter = state.entries[num] || '';
        span.textContent = letter;
        el.classList.toggle('hint', hintNums.has(+num));
        el.classList.toggle('same', selected !== null && +num === +numberGrid[selected[0]][selected[1]]);
        el.classList.remove('wrong', 'right');
        if (state.checked && letter) {
          el.classList.add(letter === solNumToLetter[num] ? 'right' : 'wrong');
        }
      }
    for (const el of gridEl.querySelectorAll('.cw-cell')) el.classList.remove('selected');
    if (selected) cells[selected[0]][selected[1]].classList.add('selected');
    updateCount();
    // keyboard availability: a letter may only map to one number
    const usedLetters = new Set(Object.values(state.entries));
    for (const btn of kbEl.querySelectorAll('button[data-letter]')) {
      const L = btn.dataset.letter;
      const curNum = selected ? numberGrid[selected[0]][selected[1]] : null;
      btn.disabled = usedLetters.has(L) && state.entries[curNum] !== L;
    }
  }

  function assign(letter) {
    if (!selected) return;
    const num = numberGrid[selected[0]][selected[1]];
    if (!num || hintNums.has(+num)) return;
    if (letter === '') {
      delete state.entries[num];
    } else {
      // a letter can only map to one number: remove it elsewhere
      for (const [n, l] of Object.entries(state.entries)) {
        if (l === letter && +n !== +num && !hintNums.has(+n)) delete state.entries[n];
      }
      state.entries[num] = letter;
    }
    state.checked = false;
    paint();
    save();
    checkWin();
  }

  gridEl.addEventListener('pointerdown', e => {
    const t = e.target.closest('.cw-cell');
    if (!t || !t.dataset.num) return;
    selected = [+t.dataset.r, +t.dataset.c];
    paint();
  });

  // physical keyboard
  container.tabIndex = -1;
  document.addEventListener('keydown', onKey);
  function onKey(e) {
    if (!container.isConnected) { document.removeEventListener('keydown', onKey); return; }
    if (e.target && e.target.matches && e.target.matches('input, textarea, select')) return;
    if (/^[a-zA-ZçÇ]$/.test(e.key)) assign(e.key.toUpperCase().normalize('NFD').replace(/\p{M}/gu, ''));
    else if (e.key === 'Backspace' || e.key === 'Delete') assign('');
    else if (e.key.startsWith('Arrow') && selected) {
      const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
      let r = selected[0], c = selected[1];
      for (let i = 0; i < rows * cols; i++) {
        r = (r + d[0] + rows) % rows;
        c = (c + d[1] + cols) % cols;
        if (numberGrid[r][c]) break;
      }
      selected = [r, c];
      paint();
      e.preventDefault();
    }
  }

  // on-screen keyboard
  for (const row of KB_ROWS) {
    const div = document.createElement('div');
    div.className = 'cw-kb-row';
    for (const letter of row) {
      const b = document.createElement('button');
      b.textContent = letter;
      b.dataset.letter = letter;
      b.onclick = () => assign(letter);
      div.appendChild(b);
    }
    kbEl.appendChild(div);
  }
  const lastRow = kbEl.lastChild;
  const del = document.createElement('button');
  del.textContent = '⌫';
  del.onclick = () => assign('');
  lastRow.appendChild(del);

  container.querySelector('#btn-check').onclick = () => {
    state.checked = true;
    paint();
    save();
    checkWin(true);
  };
  container.querySelector('#btn-print').onclick = async e => {
    const btn = e.currentTarget;
    btn.disabled = true;
    try { await printPuzzle(puzzle); }
    finally { btn.disabled = false; }
  };
  container.querySelector('#btn-sol').onclick = () => {
    if (confirm(t('confirmSolution'))) {
      for (const num of usedNums) state.entries[num] = solNumToLetter[num];
      state.done = true;
      paint();
      save();
      stopTimer();
    }
  };
  container.querySelector('#btn-back').onclick = () => {
    document.removeEventListener('keydown', onKey);
    stopTimer();
    save();
    onExit();
  };

  function checkWin(fromCheck = false) {
    const solved = usedNums.every(n => state.entries[n] === solNumToLetter[n]);
    if (solved) {
      state.done = true;
      stopTimer();
      save();
      const mp = container.querySelector('#msg-panel');
      mp.classList.remove('hidden');
      mp.innerHTML = `<div class="win">${t('cwDone')}</div>`;
      toast && toast(t('puzzleSolved'));
    }
  }

  // ---- timer ----
  let timerId = null;
  function tick() {
    state.elapsed = (state.elapsed || 0) + 1;
    const m = Math.floor(state.elapsed / 60), s = state.elapsed % 60;
    timerEl.textContent = `${m}:${String(s).padStart(2, '0')}`;
  }
  if (settings.timer && !state.done) timerId = setInterval(tick, 1000);
  function stopTimer() { if (timerId) clearInterval(timerId); timerId = null; }

  // Called before this container is re-rendered/discarded
  container._cleanup = () => { stopTimer(); document.removeEventListener('keydown', onKey); };
  if (state.elapsed) {
    const m = Math.floor(state.elapsed / 60), s = state.elapsed % 60;
    timerEl.textContent = `${m}:${String(s).padStart(2, '0')}`;
  }

  function save() { puzzle.state = state; upsertPuzzle(puzzle); }

  if (settings.codewordAutoCheck) state.checked = true;
  paint();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}
