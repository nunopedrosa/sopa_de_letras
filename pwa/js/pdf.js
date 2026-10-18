// 2-page A4 PDF export: page 1 = puzzle, page 2 = solution.
// jsPDF is vendored in js/vendor/ and loaded lazily on first print.
import { t } from './i18n.js';

let pdfReady;
function loadJsPDF() {
  pdfReady ??= new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'js/vendor/jspdf.umd.min.js';
    s.onload = resolve;
    s.onerror = () => reject(new Error('jspdf load failed'));
    document.head.appendChild(s);
  });
  return pdfReady;
}

const PAGE_W = 210, PAGE_H = 297, MARGIN = 15;

function header(doc, title, subtitle) {
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(24);
  doc.text(title, PAGE_W / 2, 20, { align: 'center' });
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(12);
  doc.setTextColor(80);
  doc.text(subtitle, PAGE_W / 2, 28, { align: 'center' });
  doc.setTextColor(0);
}

// Draw a letter/number grid; returns bottom y in mm.
function drawGrid(doc, rows, cols, cell, opts = {}) {
  const { top = 36, maxH = 185, black = null } = opts;
  const size = Math.min((PAGE_W - 2 * MARGIN) / cols, maxH / rows);
  const x0 = (PAGE_W - size * cols) / 2;
  const y0 = top;
  doc.setDrawColor(30);
  doc.setLineWidth(0.25);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = x0 + c * size, y = y0 + r * size;
      const isBlack = black && black(r, c);
      if (isBlack) {
        doc.setFillColor(20);
        doc.rect(x, y, size, size, 'FD');
        continue;
      }
      doc.setFillColor(255);
      doc.rect(x, y, size, size);
      const info = cell(r, c);
      if (info.letter) {
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(Math.max(6, size * 2.83 * 0.5));
        if (info.color) doc.setTextColor(...info.color);
        doc.text(info.letter, x + size / 2, y + size / 2, { align: 'center', baseline: 'middle' });
        doc.setTextColor(0);
      }
      if (info.num) {
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(Math.max(5, size * 2.83 * 0.28));
        doc.setTextColor(110);
        doc.text(String(info.num), x + size * 0.12, y + size * 0.12, { baseline: 'top' });
        doc.setTextColor(0);
      }
    }
  }
  return y0 + rows * size;
}

// Wrapped paragraph block; returns bottom y.
function para(doc, text, y, { size = 11, style = 'normal', color = null, width = PAGE_W - 2 * MARGIN } = {}) {
  doc.setFont('helvetica', style);
  doc.setFontSize(size);
  if (color) doc.setTextColor(...color);
  const lines = doc.splitTextToSize(text, width);
  doc.text(lines, MARGIN, y);
  doc.setTextColor(0);
  doc.setFont('helvetica', 'normal');
  return y + lines.length * (size * 0.42);
}

function label(doc, text, y, size = 13) {
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(size);
  doc.text(text, MARGIN, y);
  doc.setFont('helvetica', 'normal');
  return y + size * 0.42;
}

// ---------- word search ----------
function buildWordSearch(doc, puzzle) {
  const { grid, placements, wordsDisplay, secretMessage, messageCells } = puzzle.data;
  const rows = grid.length, cols = grid[0].length;

  header(doc, puzzle.name, t('printWsSubtitle'));
  const bottom = drawGrid(doc, rows, cols, (r, c) => ({ letter: grid[r][c] }));

  let y = bottom + 10;
  y = label(doc, t('printWords'), y);
  y = para(doc, wordsDisplay.join(', '), y, { size: 11 });

  if (secretMessage) {
    const dica = secretMessage.type === 'letters' ? t('printDicaLetters') : t('printDicaWords');
    para(doc, dica, y + 4, { size: 10, style: 'bolditalic', color: [25, 60, 160] });
  }

  // ---- page 2: solution ----
  doc.addPage();
  header(doc, `${puzzle.name} — ${t('printSolution')}`, t('printWsSolutionSub'));
  const inWord = new Set();
  for (const coords of Object.values(placements)) for (const [r, c] of coords) inWord.add(`${r},${c}`);
  const msgSet = new Set((messageCells || []).map(([r, c]) => `${r},${c}`));
  const bottom2 = drawGrid(doc, rows, cols, (r, c) =>
    inWord.has(`${r},${c}`) || msgSet.has(`${r},${c}`)
      ? { letter: grid[r][c], color: msgSet.has(`${r},${c}`) && !inWord.has(`${r},${c}`) ? [25, 60, 160] : null }
      : { letter: '·', color: [170, 170, 170] }
  );

  if (secretMessage) {
    let y2 = bottom2 + 12;
    y2 = label(doc, t('printSecret'), y2);
    const msg = secretMessage.type === 'letters'
      ? secretMessage.text
      : secretMessage.words.join(' ');
    para(doc, msg, y2, { size: 13, style: 'bolditalic', color: [25, 60, 160] });
  }
}

// ---------- codeword ----------
function buildCodeword(doc, puzzle) {
  const { numberGrid, letterGrid, wordsDisplay, hints } = puzzle.data;
  const rows = numberGrid.length, cols = numberGrid[0].length;
  const black = (r, c) => !numberGrid[r][c];

  header(doc, puzzle.name, t('printCwSubtitle'));
  const bottom = drawGrid(doc, rows, cols,
    (r, c) => ({ num: numberGrid[r][c] }), { black });

  let y = bottom + 10;
  y = label(doc, t('printHintsLabel'), y);
  const hintTxt = hints
    .map(([l, n]) => `${l}=${n}`)
    .sort((a, b) => +a.split('=')[1] - +b.split('=')[1])
    .join(', ');
  y = para(doc, hintTxt, y, { size: 11 });
  y = label(doc, t('printWords'), y + 4);
  para(doc, wordsDisplay.join(', '), y, { size: 11 });

  // ---- page 2: solution ----
  doc.addPage();
  header(doc, `${puzzle.name} — ${t('printSolution')}`, t('printCwSolutionSub'));
  const bottom2 = drawGrid(doc, rows, cols,
    (r, c) => ({ letter: letterGrid[r][c], num: numberGrid[r][c] }), { black });

  let y2 = bottom2 + 10;
  y2 = label(doc, t('printFullCode'), y2);
  // Derive number -> letter from the solution grid
  const numToLetter = {};
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      if (numberGrid[r][c]) numToLetter[numberGrid[r][c]] = letterGrid[r][c];
  const pairs = Object.entries(numToLetter)
    .sort((a, b) => +a[0] - +b[0])
    .map(([n, l]) => `${l}=${n}`);
  const lines = [];
  for (let i = 0; i < pairs.length; i += 7) lines.push(pairs.slice(i, i + 7).join(', '));
  para(doc, lines.join('\n'), y2, { size: 10 });
}

export async function printPuzzle(puzzle) {
  await loadJsPDF();
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  if (puzzle.type === 'codeword') buildCodeword(doc, puzzle);
  else buildWordSearch(doc, puzzle);
  const fname = puzzle.name.replace(/[^\w\- áàâãéêíóôõúç.]/gi, '').trim() || 'puzzle';
  doc.save(`${fname}.pdf`);
}
