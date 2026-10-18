import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createLetterSoup } from '../pwa/js/generator.js';

const WORDS = [
  'CASTELO', 'CAVALO', 'CAMINHO', 'ESCOLA', 'JANELA', 'FLORESTA',
  'MONTANHA', 'PASSARO', 'COZINHA', 'ESTRELA', 'BRASIL', 'AMIGO',
];

function direction(coords) {
  const dr = Math.sign(coords[coords.length - 1][0] - coords[0][0]);
  const dc = Math.sign(coords[coords.length - 1][1] - coords[0][1]);
  return [dr, dc];
}

test('diagonals without backwards never place a word upwards or leftwards', () => {
  const forbidden = new Set(['0,-1', '-1,0', '-1,-1', '-1,1']);
  for (let seed = 0; seed < 20; seed++) {
    const { placements } = createLetterSoup(WORDS, {
      rows: 14, cols: 14, allowDiagonals: true, allowBackwards: false, seed,
    });
    for (const coords of Object.values(placements)) {
      if (coords.length < 2) continue;
      const key = direction(coords).join(',');
      assert.equal(forbidden.has(key), false, `seed ${seed} placed ${key}`);
    }
  }
});

test('empty cells are seeded random letters, not the alphabet from W', () => {
  const opts = { rows: 12, cols: 12, seed: 4 };
  const first = createLetterSoup(['CAT'], opts);
  const again = createLetterSoup(['CAT'], opts);
  assert.deepEqual(again.grid, first.grid);

  const used = new Set();
  for (const coords of Object.values(first.placements)) {
    for (const [r, c] of coords) used.add(`${r},${c}`);
  }
  let filler = '';
  for (let r = 0; r < 12; r++) {
    for (let c = 0; c < 12; c++) {
      if (!used.has(`${r},${c}`)) filler += first.grid[r][c];
    }
  }
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  let fromW = '';
  for (let i = 0; i < filler.length; i++) fromW += alphabet[(22 + i) % 26];
  assert.notEqual(filler, fromW);
  assert.match(filler, /^[A-Z]+$/);
});

test('intersection word search stays off the main thread budget', () => {
  const words = [];
  const stems = ['CASA', 'CASTELO', 'CAVALO', 'CAMINHO', 'CAMPO', 'CARRO', 'CARTA', 'ESCOLA', 'ESTRELA', 'JANELA'];
  for (let i = 0; i < 70; i++) words.push(stems[i % stems.length] + String.fromCharCode(65 + (i % 26)));
  const t0 = performance.now();
  createLetterSoup(words, {
    rows: 15, cols: 15, allowDiagonals: true, allowBackwards: true,
    enforceIntersections: true, seed: 3,
  });
  const ms = performance.now() - t0;
  assert.ok(ms < 250, `took ${ms.toFixed(0)} ms`);
});

test('saving puzzles reports a full storage instead of throwing', async () => {
  globalThis.localStorage = {
    getItem() { return '[]'; },
    setItem() { const err = new Error('quota'); err.name = 'QuotaExceededError'; throw err; },
  };
  const { savePuzzles, upsertPuzzle } = await import('../pwa/js/state.js');
  assert.equal(savePuzzles([{ id: 'a' }]), false);
  assert.equal(upsertPuzzle({ id: 'b', name: 'b' }), false);
});
