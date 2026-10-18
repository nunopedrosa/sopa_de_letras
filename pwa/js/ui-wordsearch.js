// Word search play UI: drag/swipe + tap-tap + keyboard selection,
// configurable highlighting, hints, solution view, secret message flows.
import { normalizeWord } from './generator.js';
import { upsertPuzzle } from './state.js';
import { t } from './i18n.js';
import { printPuzzle } from './pdf.js';

const PALETTE = [
  '#ffd166', '#a0e7a0', '#a0c4ff', '#ffadad', '#cdb4db',
  '#9bf6ff', '#fdffb6', '#ffc6a5', '#bde0fe', '#d0f4de',
  '#e4c1f9', '#fcf6bd', '#ffb4a2', '#b8f2e6', '#fae1dd',
];

const ALL_DIRS = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];

export function renderWordSearch(container, puzzle, { settings, onExit, toast }) {
  const { grid, placements, wordsDisplay, secretMessage, messageCells } = puzzle.data;
  const rows = grid.length, cols = grid[0].length;
  const state = puzzle.state;

  // Precompute a lookup from sorted-cell-key -> word
  const placementKeys = {};
  for (const [word, coords] of Object.entries(placements)) {
    placementKeys[coords.map(c => c.join(',')).sort().join('|')] = word;
  }

  // Stable color per found word, in discovery order
  const colorOf = {};
  state.found.forEach((w, i) => { colorOf[w] = PALETTE[i % PALETTE.length]; });

  container.innerHTML = `
    <div class="play-head">
      <button class="btn ghost" id="btn-back">${t('menu')}</button>
      <div class="play-title">${escapeHtml(puzzle.name)}</div>
      <div class="play-stats">
        <span id="found-count">${state.found.length}/${Object.keys(placements).length}</span>
        <span id="timer" class="timer ${settings.timer ? '' : 'hidden'}">0:00</span>
      </div>
    </div>
    <div class="play-actions">
      <button class="btn" id="btn-hint">${t('hint')}</button>
      <button class="btn" id="btn-solution">${t('solution')}</button>
      <button class="btn push-right" id="btn-print">${t('printBtn')}</button>
    </div>
    <div id="ws-grid" class="ws-grid" role="grid" aria-label="${t('gridAria')}"></div>
    <div id="ws-words" class="ws-words" aria-label="${t('wordsAria')}"></div>
    <div id="msg-panel" class="msg-panel hidden"></div>
  `;

  const gridEl = container.querySelector('#ws-grid');
  const wordsEl = container.querySelector('#ws-words');
  const msgPanel = container.querySelector('#msg-panel');
  const foundCountEl = container.querySelector('#found-count');
  const timerEl = container.querySelector('#timer');

  gridEl.style.setProperty('--cols', cols);

  // ---- build cells ----
  const cells = [];
  for (let r = 0; r < rows; r++) {
    cells[r] = [];
    for (let c = 0; c < cols; c++) {
      const d = document.createElement('div');
      d.className = 'ws-cell';
      d.textContent = grid[r][c];
      d.dataset.r = r;
      d.dataset.c = c;
      d.setAttribute('role', 'gridcell');
      gridEl.appendChild(d);
      cells[r][c] = d;
    }
  }

  // ---- word chips ----
  for (const w of wordsDisplay) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = w;
    chip.dataset.word = normalizeWord(w);
    wordsEl.appendChild(chip);
  }

  // ---- helpers ----
  const cellKey = coords => coords.map(c => c.join(',')).sort().join('|');

  function snappedEnd(ar, ac, r, c) {
    const dr = r - ar, dc = c - ac;
    if (dr === 0 && dc === 0) return [ar, ac];
    const hyp = Math.hypot(dr, dc);
    let best = [0, 1], bestScore = -2;
    for (const [ur, uc] of ALL_DIRS) {
      const score = (dr * ur + dc * uc) / (hyp * Math.hypot(ur, uc));
      if (score > bestScore) { bestScore = score; best = [ur, uc]; }
    }
    const [ur, uc] = best;
    const len = Math.max(1, Math.round((dr * ur + dc * uc) / (ur * ur + uc * uc)));
    let er = ar + ur * len, ec = ac + uc * len;
    // clamp into bounds by stepping back
    while (!(er >= 0 && er < rows && ec >= 0 && ec < cols)) {
      er -= ur; ec -= uc;
      if (er === ar && ec === ac) break;
    }
    return [er, ec];
  }

  function lineCells(ar, ac, er, ec) {
    const dr = Math.sign(er - ar), dc = Math.sign(ec - ac);
    const out = [];
    let r = ar, c = ac;
    while (true) {
      out.push([r, c]);
      if (r === er && c === ec) break;
      r += dr; c += dc;
      if (out.length > rows * cols) break;
    }
    return out;
  }

  let preview = [];
  let anchor = null;      // tap-tap anchor
  let dragging = false;
  let focusCell = [0, 0];

  function clearPreview() {
    for (const [r, c] of preview) cells[r][c].classList.remove('sel', 'anchor');
    preview = [];
  }

  function showPreview(coords) {
    clearPreview();
    preview = coords;
    for (const [r, c] of coords) cells[r][c].classList.add('sel');
  }

  function markAnchor(r, c) {
    clearPreview();
    preview = [[r, c]];
    cells[r][c].classList.add('anchor');
  }

  function applyFound(word) {
    state.found.push(word);
    colorOf[word] = PALETTE[Object.keys(colorOf).length % PALETTE.length];
    const cls = settings.highlight;
    for (const [r, c] of placements[word]) {
      const el = cells[r][c];
      el.classList.remove('sel', 'anchor');
      el.classList.add('found', `h-${cls}`);
      if (cls === 'colors') el.style.setProperty('--hcol', colorOf[word]);
    }
    const chip = wordsEl.querySelector(`[data-word="${word}"]`);
    if (chip) chip.classList.add('done');
    foundCountEl.textContent = `${state.found.length}/${Object.keys(placements).length}`;
    flushSave();
    if (state.found.length === Object.keys(placements).length) win();
  }

  function tryMatch(coords) {
    if (!coords.length) return false;
    const word = placementKeys[cellKey(coords)];
    if (word && !state.found.includes(word)) {
      applyFound(word);
      return true;
    }
    // A one-cell tap that isn't a word just clears. Longer misses flash.
    if (coords.length < 2) return false;
    for (const [r, c] of coords) {
      const el = cells[r][c];
      el.classList.add('bad');
      setTimeout(() => el.classList.remove('bad'), 350);
    }
    return false;
  }

  // ---- pointer (drag) selection ----
  // The grid rect is cached for the gesture. Preview DOM updates are
  // coalesced to one frame and skipped when the cell didn't change,
  // so a drag doesn't force layout on every pointer event.
  gridEl.style.touchAction = 'none';
  let dragAnchor = null;
  let dragRect = null;
  let hoverCell = null;
  let previewFrame = 0;

  function previewKey(coords) {
    let key = '';
    for (const [r, c] of coords) key += r + ',' + c + '|';
    return key;
  }

  function previewLine(ar, ac, r, c) {
    const [er, ec] = snappedEnd(ar, ac, r, c);
    return lineCells(ar, ac, er, ec);
  }

  function paintPreview(coords) {
    if (previewKey(preview) === previewKey(coords)) return;
    showPreview(coords);
  }

  function cancelPreviewFrame() {
    if (previewFrame) cancelAnimationFrame(previewFrame);
    previewFrame = 0;
  }

  gridEl.addEventListener('pointerdown', e => {
    const cell = e.target.closest('.ws-cell');
    if (!cell) return;
    e.preventDefault();
    const r = +cell.dataset.r, c = +cell.dataset.c;
    if (anchor) {
      const coords = previewLine(anchor[0], anchor[1], r, c);
      clearPreview();
      tryMatch(coords);
      anchor = null;
      return;
    }
    dragAnchor = [r, c];
    dragRect = gridEl.getBoundingClientRect();
    hoverCell = [r, c];
    dragging = false;
    gridEl.setPointerCapture(e.pointerId);
    showPreview([[r, c]]);
  });

  gridEl.addEventListener('pointermove', e => {
    if (!dragAnchor || !dragRect) return;
    const cw = dragRect.width / cols, ch = dragRect.height / rows;
    const c = Math.floor((e.clientX - dragRect.left) / cw);
    const r = Math.floor((e.clientY - dragRect.top) / ch);
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;
    if (hoverCell && hoverCell[0] === r && hoverCell[1] === c) return;
    hoverCell = [r, c];
    dragging = true;
    if (previewFrame) return;
    previewFrame = requestAnimationFrame(() => {
      previewFrame = 0;
      if (!dragAnchor || !hoverCell) return;
      paintPreview(previewLine(dragAnchor[0], dragAnchor[1], hoverCell[0], hoverCell[1]));
    });
  });

  function finishDrag(commit) {
    cancelPreviewFrame();
    if (!dragAnchor) return;
    if (commit && dragging && hoverCell) {
      paintPreview(previewLine(dragAnchor[0], dragAnchor[1], hoverCell[0], hoverCell[1]));
    }
    const wasDrag = dragging;
    const coords = preview.slice();
    const tapCell = dragAnchor.slice();
    dragAnchor = null;
    dragRect = null;
    hoverCell = null;
    dragging = false;
    if (!commit) {
      clearPreview();
      return;
    }
    if (wasDrag && coords.length >= 2) {
      clearPreview();
      tryMatch(coords);
    } else if (tryMatch([tapCell])) {
      // A one-letter word is found with a single tap.
      clearPreview();
    } else if (anchor && anchor[0] === tapCell[0] && anchor[1] === tapCell[1]) {
      anchor = null;
      clearPreview();
    } else {
      anchor = tapCell;
      markAnchor(tapCell[0], tapCell[1]);
      focusCell = tapCell;
    }
  }

  gridEl.addEventListener('pointerup', () => finishDrag(true));
  gridEl.addEventListener('pointercancel', () => finishDrag(false));

  // ---- keyboard support ----
  gridEl.tabIndex = 0;
  let kbAnchor = null;
  gridEl.addEventListener('keydown', e => {
    const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    if (moves[e.key]) {
      e.preventDefault();
      const [dr, dc] = moves[e.key];
      focusCell = [
        Math.min(rows - 1, Math.max(0, focusCell[0] + dr)),
        Math.min(cols - 1, Math.max(0, focusCell[1] + dc)),
      ];
      for (const row of cells) for (const el of row) el.classList.remove('focus');
      cells[focusCell[0]][focusCell[1]].classList.add('focus');
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (!kbAnchor) {
        kbAnchor = focusCell.slice();
        markAnchor(focusCell[0], focusCell[1]);
      } else {
        const [er, ec] = snappedEnd(kbAnchor[0], kbAnchor[1], focusCell[0], focusCell[1]);
        const coords = lineCells(kbAnchor[0], kbAnchor[1], er, ec);
        clearPreview();
        tryMatch(coords);
        kbAnchor = null;
      }
    } else if (e.key === 'Escape') {
      kbAnchor = null; anchor = null; clearPreview();
    }
  });

  // ---- controls ----
  container.querySelector('#btn-back').onclick = () => onExit();

  container.querySelector('#btn-hint').onclick = () => {
    const remaining = Object.keys(placements).filter(w => !state.found.includes(w));
    if (!remaining.length) return;
    const word = remaining[Math.floor(Math.random() * remaining.length)];
    const [r, c] = placements[word][0];
    const el = cells[r][c];
    el.classList.add('hinted');
    setTimeout(() => el.classList.remove('hinted'), 1500);
    state.hints = (state.hints || 0) + 1;
    flushSave();
  };

  container.querySelector('#btn-print').onclick = async e => {
    const btn = e.currentTarget;
    btn.disabled = true;
    try { await printPuzzle(puzzle); }
    finally { btn.disabled = false; }
  };

  let solutionOn = false;
  container.querySelector('#btn-solution').onclick = () => {
    solutionOn = !solutionOn;
    gridEl.classList.toggle('solution', solutionOn);
    if (solutionOn) {
      Object.entries(placements).forEach(([w, coords]) => {
        coords.forEach(([r, c]) => cells[r][c].classList.add('sol'));
      });
    } else {
      for (const row of cells) for (const el of row) el.classList.remove('sol');
    }
  };

  // ---- restore found ----
  for (const w of state.found) {
    const cls = settings.highlight;
    for (const [r, c] of placements[w]) {
      const el = cells[r][c];
      el.classList.add('found', `h-${cls}`);
      if (cls === 'colors') el.style.setProperty('--hcol', colorOf[w]);
    }
    const chip = wordsEl.querySelector(`[data-word="${w}"]`);
    if (chip) chip.classList.add('done');
  }

  // ---- timer ----
  // Paused while the page is hidden, so background time is not counted
  // and the interval does not keep waking the device.
  let timerId = null;
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
    puzzle.state = state;
    if (!upsertPuzzle(puzzle) && !saveErrorShown) {
      saveErrorShown = true;
      if (toast) toast(t('errSave'));
    }
  }

  function onVisibility() {
    if (document.hidden) { stopTimer(); flushSave(); }
    else startTimer();
  }
  document.addEventListener('visibilitychange', onVisibility);
  container._cleanup = () => {
    stopTimer();
    document.removeEventListener('visibilitychange', onVisibility);
    flushSave();
  };

  // ---- win + secret message ----
  // Built the same way on a fresh win and on resume, so leaving the
  // puzzle does not skip the secret step or reveal it early.
  function renderSecretPanel(scroll) {
    msgPanel.classList.remove('hidden');
    showElapsed();
    const timeTxt = settings.timer ? t('inTime', { time: timerEl.textContent }) : '';
    let html = `<div class="win">${t('foundAll', { time: timeTxt })}</div>`;

    if (secretMessage && secretMessage.type === 'letters') {
      if (state.secretRevealed) {
        html += `<div id="secret-text" class="secret-text"></div>`;
        msgPanel.innerHTML = html;
        msgPanel.querySelector('#secret-text').textContent = secretMessage.text;
        if (messageCells) messageCells.forEach(([r, c]) => cells[r][c].classList.add('secret'));
      } else {
        html += `<button class="btn" id="btn-reveal">${t('revealSecret')}</button>
                 <div id="secret-text" class="secret-text hidden"></div>`;
        msgPanel.innerHTML = html;
        msgPanel.querySelector('#btn-reveal').onclick = () => {
          state.secretRevealed = true;
          messageCells.forEach(([r, c], i) => {
            setTimeout(() => cells[r][c].classList.add('secret'), 120 * i);
          });
          const st = msgPanel.querySelector('#secret-text');
          st.classList.remove('hidden');
          st.textContent = secretMessage.text;
          flushSave();
        };
      }
    } else if (secretMessage && secretMessage.type === 'words') {
      html += `<p>${t('sortInstructions')}</p>
               <div id="sort-words" class="sort-words"></div>
               <button class="btn" id="btn-check-msg">${t('check')}</button>
               <div id="msg-result" class="secret-text"></div>`;
      msgPanel.innerHTML = html;
      const box = msgPanel.querySelector('#sort-words');
      const order = state.msgOrder || [...secretMessage.words].sort(() => Math.random() - 0.5);
      state.msgOrder = order;
      let pick = null;
      function drawSort() {
        box.innerHTML = '';
        order.forEach((w, i) => {
          const chip = document.createElement('button');
          chip.className = 'chip sortable' + (pick === i ? ' picked' : '');
          chip.textContent = w;
          chip.onclick = () => {
            if (pick === null) { pick = i; }
            else { [order[pick], order[i]] = [order[i], order[pick]]; pick = null; flushSave(); }
            drawSort();
          };
          box.appendChild(chip);
        });
      }
      drawSort();
      msgPanel.querySelector('#btn-check-msg').onclick = () => {
        const goal = secretMessage.words.map(normalizeWord);
        const cur = order.map(normalizeWord);
        const res = msgPanel.querySelector('#msg-result');
        if (cur.join('|') === goal.join('|')) {
          res.textContent = t('secretIs', { msg: secretMessage.words.join(' ') });
          res.classList.add('ok');
        } else {
          res.textContent = t('wrongOrder');
          res.classList.remove('ok');
        }
      };
    } else {
      msgPanel.innerHTML = html;
    }
    if (scroll) msgPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function win() {
    state.done = true;
    stopTimer();
    flushSave();
    renderSecretPanel(true);
  }

  showElapsed();
  if (state.done) renderSecretPanel(false);
  startTimer();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}
