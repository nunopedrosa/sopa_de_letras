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
  const cellsByNum = {};
  const letterCells = [];
  const kbButtons = [];
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
        d._letter = d.querySelector('.cw-letter');
        d.tabIndex = 0;
        (cellsByNum[num] ||= []).push(d);
        letterCells.push(d);
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

  function paintLetters(nums) {
    for (const num of nums) {
      const key = String(num);
      const letter = state.entries[key] || '';
      const mark = state.checked && letter
        ? (letter === solNumToLetter[key] ? 'right' : 'wrong')
        : '';
      for (const el of cellsByNum[key] || []) {
        el._letter.textContent = letter;
        el.classList.toggle('hint', hintNums.has(+key));
        el.classList.toggle('right', mark === 'right');
        el.classList.toggle('wrong', mark === 'wrong');
      }
    }
  }

  function paintSelection() {
    const selNum = selected ? String(numberGrid[selected[0]][selected[1]]) : null;
    for (const el of letterCells) {
      el.classList.toggle('same', selNum !== null && el.dataset.num === selNum);
      el.classList.remove('selected');
    }
    if (selected) cells[selected[0]][selected[1]].classList.add('selected');
    updateCount();
    const usedLetters = new Set(Object.values(state.entries));
    for (const btn of kbButtons) {
      const L = btn.dataset.letter;
      btn.disabled = usedLetters.has(L) && state.entries[selNum] !== L;
    }
  }

  function assign(letter) {
    if (!selected) return;
    const num = String(numberGrid[selected[0]][selected[1]]);
    if (!num || hintNums.has(+num)) return;
    const affected = new Set([num]);
    if (letter === '') {
      delete state.entries[num];
    } else {
      // a letter can only map to one number: remove it elsewhere
      for (const [n, l] of Object.entries(state.entries)) {
        if (l === letter && n !== num && !hintNums.has(+n)) {
          delete state.entries[n];
          affected.add(n);
        }
      }
      state.entries[num] = letter;
    }
    // Manual check marks are cleared by an edit. Auto-check stays on.
    const wasChecked = !!state.checked;
    if (!settings.codewordAutoCheck) state.checked = false;
    if (wasChecked && !state.checked) paintLetters(usedNums);
    else paintLetters(affected);
    paintSelection();
    scheduleSave();
    checkWin();
  }

  gridEl.addEventListener('pointerdown', e => {
    const cell = e.target.closest('.cw-cell');
    if (!cell || !cell.dataset.num) return;
    selected = [+cell.dataset.r, +cell.dataset.c];
    paintSelection();
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
      paintSelection();
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
      kbButtons.push(b);
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
    paintLetters(usedNums);
    paintSelection();
    flushSave();
    checkWin();
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
      paintLetters(usedNums);
      paintSelection();
      flushSave();
      stopTimer();
    }
  };
  container.querySelector('#btn-back').onclick = () => onExit();

  function checkWin() {
    const solved = usedNums.every(n => state.entries[n] === solNumToLetter[n]);
    if (!solved || state.done) return;
    state.done = true;
    stopTimer();
    flushSave();
    const mp = container.querySelector('#msg-panel');
    mp.classList.remove('hidden');
    mp.innerHTML = `<div class="win">${t('cwDone')}</div>`;
    if (toast) toast(t('puzzleSolved'));
  }

  // ---- timer + save ----
  // Letter edits are batched. The write is flushed when the page hides,
  // the puzzle is torn down, or the player pauses for a moment.
  let timerId = null;
  let saveTimer = null;
  let saveErrorShown = false;
  function showElapsed() {
    const secs = state.elapsed || 0;
    const m = Math.floor(secs / 60), s = secs % 60;
    timerEl.textContent = `${m}:${String(s).padStart(2, '0')}`;
  }
  function tick() {
    state.elapsed = (state.elapsed || 0) + 1;
    showElapsed();
  }
  function startTimer() {
    if (timerId || state.done || !settings.timer || document.hidden) return;
    timerId = setInterval(tick, 1000);
  }
  function stopTimer() { if (timerId) clearInterval(timerId); timerId = null; }

  function flushSave() {
    if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
    puzzle.state = state;
    if (!upsertPuzzle(puzzle) && !saveErrorShown) {
      saveErrorShown = true;
      if (toast) toast(t('errSave'));
    }
  }
  function scheduleSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(flushSave, 400);
  }

  function onVisibility() {
    if (document.hidden) { stopTimer(); flushSave(); }
    else startTimer();
  }
  document.addEventListener('visibilitychange', onVisibility);
  container._cleanup = () => {
    stopTimer();
    document.removeEventListener('keydown', onKey);
    document.removeEventListener('visibilitychange', onVisibility);
    flushSave();
  };

  showElapsed();
  if (settings.codewordAutoCheck) state.checked = true;
  paintLetters(usedNums);
  paintSelection();
  if (state.done) {
    const mp = container.querySelector('#msg-panel');
    mp.classList.remove('hidden');
    mp.innerHTML = `<div class="win">${t('cwDone')}</div>`;
  }
  startTimer();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}
