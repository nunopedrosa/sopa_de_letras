#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import unicodedata
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
import argparse
import sys

# -----------------------------
# Helpers
# -----------------------------
def strip_accents(s: str) -> str:
    return ''.join(
        ch for ch in unicodedata.normalize('NFD', s)
        if unicodedata.category(ch) != 'Mn'
    )

def normalize_word(w: str) -> str:
    w = w.strip().replace(" ", "").replace("-", "")
    w = strip_accents(w)
    return w.upper()

# -----------------------------
# Core generator
# -----------------------------
def create_letter_soup(
    words_pt: List[str],
    rows: int = 12,
    cols: int = 12,
    allow_diagonals: bool = False,
    allow_backwards: bool = False,
    seed: Optional[int] = 42
):
    rng = random.Random(seed)
    # Normalize and deduplicate words
    normalized_words = [normalize_word(w) for w in words_pt if normalize_word(w)]
    words = list(dict.fromkeys(normalized_words))  # Preserves order while removing duplicates
    if not words:
        raise ValueError("No valid words provided.")
    max_len = max(len(w) for w in words)
    if max_len > max(rows, cols):
        raise ValueError(f"Word '{max(words, key=len)}' is too long for a {rows}x{cols} grid.")

    dirs = []
    dirs.extend([(0, 1), (1, 0)])  # right, down
    if allow_backwards:
        dirs.extend([(0, -1), (-1, 0)])
    if allow_diagonals:
        diags = [(1, 1), (-1, -1)]
        if allow_backwards:
            diags.extend([(1, -1), (-1, 1)])
        dirs.extend(diags)

    grid = [['' for _ in range(cols)] for _ in range(rows)]
    placements = {}

    def can_place(word: str, r: int, c: int, dr: int, dc: int) -> bool:
        rr, cc = r, c
        for ch in word:
            if not (0 <= rr < rows and 0 <= cc < cols):
                return False
            existing = grid[rr][cc]
            # This line allows words to CROSS when letters match
            if existing not in ('', ch):
                return False
            rr += dr
            cc += dc
        return True

    def do_place(word: str, r: int, c: int, dr: int, dc: int) -> List[Tuple[int, int]]:
        coords = []
        rr, cc = r, c
        for ch in word:
            grid[rr][cc] = ch
            coords.append((rr, cc))
            rr += dr
            cc += dc
        return coords

    for w in sorted(words, key=len, reverse=True):
        placed = False
        attempts = 0
        max_attempts = rows * cols * 10
        while not placed and attempts < max_attempts:
            dr, dc = rng.choice(dirs)
            if dr == 0:
                r = rng.randrange(rows)
            elif dr > 0:
                r = rng.randrange(rows - len(w) + 1)
            else:
                r = rng.randrange(len(w) - 1, rows)

            if dc == 0:
                c = rng.randrange(cols)
            elif dc > 0:
                c = rng.randrange(cols - len(w) + 1)
            else:
                c = rng.randrange(len(w) - 1, cols)

            if can_place(w, r, c, dr, dc):
                coords = do_place(w, r, c, dr, dc)
                placements[w] = coords
                placed = True
            attempts += 1

        if not placed:
            raise RuntimeError(f"Couldn't place the word '{w}'. Try a larger grid.")

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '':
                grid[r][c] = rng.choice(alphabet)

    return grid, placements

# -----------------------------
# Rendering (PDF and PNG)
# -----------------------------
def render_worksheet(
    grid: List[List[str]],
    words_display: List[str],
    title: str = "Sopa de Letras",
    subtitle: str = "Encontra as palavras!",
    output_png: Path = Path("sopa_de_letras.png"),
    output_pdf: Path = Path("sopa_de_letras.pdf"),
):
    rows, cols = len(grid), len(grid[0])
    fig = plt.figure(figsize=(8.27, 11.69), dpi=200)  # A4 portrait
    ax = plt.gca()
    ax.axis('off')

    ax.text(0.5, 0.96, title, ha='center', va='center', fontsize=28, fontweight='bold')
    ax.text(0.5, 0.925, subtitle, ha='center', va='center', fontsize=14)

    cell_text = [[grid[r][c] for c in range(cols)] for r in range(rows)]
    table = plt.table(
        cellText=cell_text,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.28, 0.8, 0.6]
    )

    for cell in table.get_celld().values():
        cell.get_text().set_fontsize(12)

    words_txt = ", ".join(words_display)
    wrapped = textwrap.wrap(words_txt, width=60)
    ax.text(0.1, 0.2, "Palavras:", fontsize=14, fontweight='bold')
    y = 0.175
    for line in wrapped:
        ax.text(0.1, y, line, fontsize=12)
        y -= 0.03

    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

def render_solution(
    grid: List[List[str]],
    placements: dict,
    output_png: Path = Path("sopa_de_letras_solucao.png"),
    output_pdf: Path = Path("sopa_de_letras_solucao.pdf"),
):
    rows, cols = len(grid), len(grid[0])
    mask = [['.' for _ in range(cols)] for _ in range(rows)]
    for coords in placements.values():
        for r, c in coords:
            mask[r][c] = grid[r][c]

    fig = plt.figure(figsize=(8.27, 11.69), dpi=200)
    ax = plt.gca()
    ax.axis('off')

    ax.text(0.5, 0.96, "Sopa de Letras — Solução", ha='center', va='center', fontsize=26, fontweight='bold')
    ax.text(0.5, 0.925, "Letras das palavras (o resto fica em ponto)", ha='center', va='center', fontsize=12)

    table = plt.table(
        cellText=mask,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.2, 0.8, 0.7]
    )
    for cell in table.get_celld().values():
        cell.get_text().set_fontsize(12)

    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 'sopa de letras' worksheet and its solution.")
    parser.add_argument('--words-file', '-f', help='Path to file with words (one per line). Lines starting with # ignored.')
    parser.add_argument('--rows', type=int, default=12, help='Number of rows in the grid.')
    parser.add_argument('--cols', type=int, default=12, help='Number of columns in the grid.')
    parser.add_argument('--allow-diagonals', action='store_true', help='Allow diagonal placements.')
    parser.add_argument('--allow-backwards', action='store_true', help='Allow backwards placements.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output-prefix', default='sopa_de_letras', help='Output files prefix (no extension).')
    args = parser.parse_args()

    if args.words_file:
        p = Path(args.words_file)
        if not p.exists():
            print(f"Words file not found: {p}", file=sys.stderr)
            sys.exit(2)
        with p.open(encoding='utf-8') as fh:
            # keep original word forms for display; ignore blank lines and comments
            palavras_raw = [line.strip() for line in fh if line.strip() and not line.strip().startswith('#')]
            # Deduplicate while preserving order (based on normalized form)
            seen = set()
            palavras = []
            for word in palavras_raw:
                normalized = normalize_word(word)
                if normalized not in seen:
                    seen.add(normalized)
                    palavras.append(word)
    else:
        palavras = ["coração", "bola", "peixe", "casa", "gato", "sol", "pão", "fada", "mão"]

    grid, placements = create_letter_soup(
        palavras,
        rows=args.rows,
        cols=args.cols,
        allow_diagonals=args.allow_diagonals,
        allow_backwards=args.allow_backwards,
        seed=args.seed
    )

    out_prefix = Path(args.output_prefix)
    worksheet_png = out_prefix.with_suffix('.png')
    worksheet_pdf = out_prefix.with_suffix('.pdf')
    solution_png = out_prefix.with_name(out_prefix.stem + '_solucao.png')
    solution_pdf = out_prefix.with_name(out_prefix.stem + '_solucao.pdf')

    render_worksheet(grid, palavras, output_png=worksheet_png, output_pdf=worksheet_pdf)
    render_solution(grid, placements, output_png=solution_png, output_pdf=solution_pdf)