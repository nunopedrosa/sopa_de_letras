// Port of sopa_de_letras.py generation logic to JavaScript.
import { RNG } from './rng.js';

export function stripAccents(s) {
  return s.normalize('NFD').replace(/\p{M}/gu, '');
}

export function normalizeWord(w) {
  return stripAccents(w.trim().replace(/[ \-]/g, '')).toUpperCase();
}

// -----------------------------
// Word search ("sopa de letras")
// -----------------------------
export function createLetterSoup(wordsPt, {
  rows = 15,
  cols = 15,
  allowDiagonals = false,
  allowBackwards = false,
  enforceIntersections = false,
  seed = 42,
  secretMessage = null, // string placed letter-by-letter into empty tiles
} = {}) {
  const rng = new RNG(seed);
  const normalized = wordsPt.map(normalizeWord).filter(Boolean);
  const words = [...new Set(normalized)];
  if (!words.length) throw new Error('errNoValidWords');
  const tooLong = words.filter(w => w.length > Math.max(rows, cols));
  const wordList = words.filter(w => w.length <= Math.max(rows, cols));
  if (!wordList.length) {
    const e = new Error('errAllTooLong');
    e.params = { rows, cols };
    throw e;
  }

  // Right and down, plus downward diagonals. Upward / leftward
  // directions are backwards, including the up-left diagonal.
  const dirs = [[0, 1], [1, 0]];
  if (allowBackwards) dirs.push([0, -1], [-1, 0]);
  if (allowDiagonals) {
    dirs.push([1, 1], [1, -1]);
    if (allowBackwards) dirs.push([-1, -1], [-1, 1]);
  }

  const grid = Array.from({ length: rows }, () => Array(cols).fill(''));
  const placements = {}; // word -> [[r,c],...]
  // Same-axis words, with the cells just before and after cached so
  // placement attempts don't rebuild those sets.
  const blocksByAxis = new Map();

  function axisOf(dr, dc) {
    let ndr = dr === 0 ? 0 : Math.sign(dr);
    let ndc = dc === 0 ? 0 : Math.sign(dc);
    if (ndr < 0 || (ndr === 0 && ndc < 0)) { ndr = -ndr; ndc = -ndc; }
    return `${ndr},${ndc}`;
  }

  function rememberPlacement(coords, dr, dc) {
    if (coords.length < 2) return;
    const eDr = dr === 0 ? 0 : Math.sign(dr);
    const eDc = dc === 0 ? 0 : Math.sign(dc);
    const cells = new Set(coords.map(([er, ec]) => `${er},${ec}`));
    const s = coords[0], e = coords[coords.length - 1];
    cells.add(`${s[0] - eDr},${s[1] - eDc}`);
    cells.add(`${e[0] + eDr},${e[1] + eDc}`);
    const axis = axisOf(dr, dc);
    const list = blocksByAxis.get(axis);
    if (list) list.push(cells); else blocksByAxis.set(axis, [cells]);
  }

  function canPlace(word, r, c, dr, dc, mustIntersect) {
    // Bounds and letter clashes are the common reject, so they come
    // before the same-direction adjacency scan.
    let rr = r, cc = c;
    let hasIntersection = false;
    for (let i = 0; i < word.length; i++) {
      if (!(rr >= 0 && rr < rows && cc >= 0 && cc < cols)) return false;
      const existing = grid[rr][cc];
      if (existing !== '' && existing !== word[i]) return false;
      if (existing === word[i]) hasIntersection = true;
      rr += dr;
      cc += dc;
    }
    if (mustIntersect && !hasIntersection) return false;

    const blocks = blocksByAxis.get(axisOf(dr, dc));
    if (!blocks) return true;
    const newCells = [];
    for (let i = 0; i < word.length; i++) newCells.push(`${r + i * dr},${c + i * dc}`);
    newCells.push(`${r - dr},${c - dc}`);
    newCells.push(`${r + word.length * dr},${c + word.length * dc}`);
    for (const block of blocks) {
      for (const cell of newCells) if (block.has(cell)) return false;
    }
    return true;
  }

  function doPlace(word, r, c, dr, dc) {
    const coords = [];
    let rr = r, cc = c;
    for (const ch of word) {
      grid[rr][cc] = ch;
      coords.push([rr, cc]);
      rr += dr;
      cc += dc;
    }
    rememberPlacement(coords, dr, dc);
    return coords;
  }

  const unplaced = [];
  const sorted = [...wordList].sort((a, b) => b.length - a.length);
  sorted.forEach((w, idx) => {
    let placed = false;
    let attempts = 0;
    const maxAttempts = rows * cols * 10;
    const mustIntersect = enforceIntersections && idx > 0;
    while (!placed && attempts < maxAttempts) {
      const [dr, dc] = rng.choice(dirs);
      let r, c;
      if (dr === 0) r = rng.randrange(rows);
      else if (dr > 0) r = rng.randrange(rows - w.length + 1);
      else r = rng.randrange(w.length - 1, rows);
      if (dc === 0) c = rng.randrange(cols);
      else if (dc > 0) c = rng.randrange(cols - w.length + 1);
      else c = rng.randrange(w.length - 1, cols);
      if (canPlace(w, r, c, dr, dc, mustIntersect)) {
        placements[w] = doPlace(w, r, c, dr, dc);
        placed = true;
      }
      attempts++;
    }
    if (!placed) unplaced.push(w);
  });

  // Empty tiles, in reading order
  let emptyTiles = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      if (grid[r][c] === '') emptyTiles.push([r, c]);

  const messageCells = [];
  if (secretMessage) {
    let msg = normalizeWord(secretMessage).replace(/[^A-Z]/g, '');
    let i = 0;
    for (const [r, c] of emptyTiles) {
      if (i >= msg.length) break;
      grid[r][c] = msg[i];
      messageCells.push([r, c]);
      i++;
    }
    if (i < msg.length) {
      // Not enough room: treat as if no message was placed
      for (const [r, c] of messageCells) grid[r][c] = '';
      messageCells.length = 0;
    } else {
      const filled = new Set(messageCells.map(([r, c]) => `${r},${c}`));
      emptyTiles = emptyTiles.filter(([r, c]) => !filled.has(`${r},${c}`));
    }
  }

  // Fill remaining empty tiles with seeded random letters.
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  for (const [r, c] of emptyTiles) {
    grid[r][c] = alphabet[rng.randrange(alphabet.length)];
  }

  return { grid, placements, unplaced: [...tooLong, ...unplaced], messageCells };
}

// -----------------------------
// Codeword
// -----------------------------
export function createCodeword(wordsPt, {
  rows = 15,
  cols = 15,
  numHints = 3,
  seed = 42,
} = {}) {
  const rng = new RNG(seed);
  const normalized = wordsPt.map(normalizeWord).filter(Boolean);
  const words = [...new Set(normalized)];
  if (!words.length) throw new Error('errNoValidWords');
  const tooLong = words.filter(w => w.length > Math.max(rows, cols));
  const wordList = words.filter(w => w.length <= Math.max(rows, cols));
  if (!wordList.length) {
    const e = new Error('errAllTooLong');
    e.params = { rows, cols };
    throw e;
  }

  const dirs = [[0, 1], [1, 0]];
  const grid = Array.from({ length: rows }, () => Array(cols).fill(''));
  const placements = {};
  const wordsSet = new Set(wordList);

  function validateSequences(word, r, c, dr, dc) {
    const pDr = dc, pDc = dr; // perpendicular direction

    for (let i = 0; i < word.length; i++) {
      const rr = r + i * dr, cc = c + i * dc;
      // Walk backwards to the start of the perpendicular sequence
      let sr = rr, sc = cc;
      while (sr - pDr >= 0 && sr - pDr < rows && sc - pDc >= 0 && sc - pDc < cols) {
        if (!grid[sr - pDr][sc - pDc]) break;
        sr -= pDr; sc -= pDc;
      }
      const seq = [];
      let cr = sr, ccl = sc;
      while (cr >= 0 && cr < rows && ccl >= 0 && ccl < cols) {
        if (cr === rr && ccl === cc) seq.push(word[i]);
        else if (grid[cr][ccl]) seq.push(grid[cr][ccl]);
        else break;
        cr += pDr; ccl += pDc;
      }
      const seqStr = seq.join('');
      if (seqStr.length > 1 && !wordsSet.has(seqStr)) return false;
    }

    // Same-direction sequences
    let sr = r, sc = c;
    while (sr - dr >= 0 && sr - dr < rows && sc - dc >= 0 && sc - dc < cols) {
      if (!grid[sr - dr][sc - dc]) break;
      sr -= dr; sc -= dc;
    }
    let cr = sr, ccl = sc;
    let current = [];
    const inWord = (rr, cc) => {
      for (let i = 0; i < word.length; i++) {
        if (rr === r + i * dr && cc === c + i * dc) return i;
      }
      return -1;
    };
    while (cr >= 0 && cr < rows && ccl >= 0 && ccl < cols) {
      const pos = inWord(cr, ccl);
      if (pos >= 0) current.push(word[pos]);
      else if (grid[cr][ccl]) current.push(grid[cr][ccl]);
      else {
        if (current.length > 1 && !wordsSet.has(current.join(''))) return false;
        current = [];
      }
      cr += dr; ccl += dc;
    }
    if (current.length > 1 && !wordsSet.has(current.join(''))) return false;
    return true;
  }

  function canPlace(word, r, c, dr, dc, mustIntersect) {
    let rr = r, cc = c;
    let hasIntersection = false;
    for (const ch of word) {
      if (!(rr >= 0 && rr < rows && cc >= 0 && cc < cols)) return false;
      const existing = grid[rr][cc];
      if (existing !== '' && existing !== ch) return false;
      if (existing === ch) hasIntersection = true;
      rr += dr;
      cc += dc;
    }
    if (mustIntersect && !hasIntersection) return false;
    return validateSequences(word, r, c, dr, dc);
  }

  function doPlace(word, r, c, dr, dc) {
    const coords = [];
    let rr = r, cc = c;
    for (const ch of word) {
      grid[rr][cc] = ch;
      coords.push([rr, cc]);
      rr += dr;
      cc += dc;
    }
    return coords;
  }

  function findIntersectionPoints(word) {
    const candidates = [];
    for (const coords of Object.values(placements)) {
      if (coords.length < 2) continue;
      const isHoriz = coords[0][0] === coords[coords.length - 1][0];
      const isVert = coords[0][1] === coords[coords.length - 1][1];
      for (let i = 0; i < word.length; i++) {
        for (const [er, ec] of coords) {
          if (grid[er][ec] !== word[i]) continue;
          if (isVert) {
            const cc = ec - i;
            if (cc >= 0 && cc < cols && cc + word.length <= cols) candidates.push([er, cc, 0, 1]);
          }
          if (isHoriz) {
            const rr = er - i;
            if (rr >= 0 && rr < rows && rr + word.length <= rows) candidates.push([rr, ec, 1, 0]);
          }
        }
      }
    }
    const seen = new Set();
    const unique = candidates.filter(k => {
      const key = k.join(',');
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    rng.shuffle(unique);
    return unique.slice(0, Math.min(100, unique.length));
  }

  const unplaced = [];
  const sorted = [...wordList].sort((a, b) => b.length - a.length);
  sorted.forEach((w, idx) => {
    let placed = false;
    let attempts = 0;
    const maxAttempts = rows * cols * 30;
    const mustIntersect = idx > 0;

    if (mustIntersect && Object.keys(placements).length) {
      for (const [r, c, dr, dc] of findIntersectionPoints(w)) {
        if (canPlace(w, r, c, dr, dc, mustIntersect)) {
          placements[w] = doPlace(w, r, c, dr, dc);
          placed = true;
          break;
        }
        attempts++;
      }
    }

    while (!placed && attempts < maxAttempts) {
      const [dr, dc] = rng.choice(dirs);
      let r, c;
      if (dr === 0) {
        r = rng.randrange(rows);
        c = dc > 0 ? rng.randrange(cols - w.length + 1) : rng.randrange(w.length - 1, cols);
      } else {
        c = rng.randrange(cols);
        r = dr > 0 ? rng.randrange(rows - w.length + 1) : rng.randrange(w.length - 1, rows);
      }
      if (canPlace(w, r, c, dr, dc, mustIntersect)) {
        placements[w] = doPlace(w, r, c, dr, dc);
        placed = true;
      }
      attempts++;
    }
    if (!placed) unplaced.push(w);
  });

  // Letter <-> number mapping
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const usedLetters = new Set();
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      if (grid[r][c]) usedLetters.add(grid[r][c]);

  const allWordsLetters = new Set();
  for (const w of wordsPt) {
    for (const ch of normalizeWord(w)) allWordsLetters.add(ch);
  }
  const missingLetters = [...allWordsLetters].filter(l => !usedLetters.has(l)).sort();

  const letterToNum = {};
  const numToLetter = {};
  const available = Array.from({ length: 26 }, (_, i) => i + 1);
  rng.shuffle(available);
  for (const letter of [...usedLetters].sort()) {
    if (!available.length) break;
    const num = available.shift();
    letterToNum[letter] = num;
    numToLetter[num] = letter;
  }
  for (const letter of alphabet) {
    if (!(letter in letterToNum) && available.length) {
      const num = available.shift();
      letterToNum[letter] = num;
      numToLetter[num] = letter;
    }
  }

  const numberGrid = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => (grid[r][c] ? String(letterToNum[grid[r][c]]) : ''))
  );

  const usedList = [...usedLetters].sort();
  let hints = [];
  if (usedList.length) {
    const n = Math.min(numHints, usedList.length);
    hints = rng.sample(usedList, n).map(letter => [letter, letterToNum[letter]]);
  }

  return {
    numberGrid, letterGrid: grid, placements,
    letterToNum, numToLetter, hints,
    unplaced: [...tooLong, ...unplaced],
    missingLetters,
  };
}
