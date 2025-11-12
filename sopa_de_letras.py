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
    rows: int = 15,
    cols: int = 15,
    allow_diagonals: bool = False,
    allow_backwards: bool = False,
    enforce_intersections: bool = False,
    seed: Optional[int] = 42
):
    rng = random.Random(seed)
    # Normalize and deduplicate words
    normalized_words = [normalize_word(w) for w in words_pt if normalize_word(w)]
    words = list(dict.fromkeys(normalized_words))  # Preserves order while removing duplicates
    if not words:
        raise ValueError("No valid words provided.")
    # Filter out words that are too long
    too_long = [w for w in words if len(w) > max(rows, cols)]
    words = [w for w in words if len(w) <= max(rows, cols)]
    if not words:
        raise ValueError(f"All words are too long for a {rows}x{cols} grid.")

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

    def can_place(word: str, r: int, c: int, dr: int, dc: int, must_intersect: bool = False) -> bool:
        rr, cc = r, c
        has_intersection = False
        
        # Check for overlapping or adjacent words in the same direction (confusing for kids)
        # Get all cells this word would occupy, plus adjacent cells in the same direction
        new_word_cells = set()
        for i in range(len(word)):
            new_word_cells.add((r + i * dr, c + i * dc))
        # Also check cells immediately before and after (to catch adjacent words)
        if len(word) > 0:
            # Cell before the word
            new_word_cells.add((r - dr, c - dc))
            # Cell after the word
            new_word_cells.add((r + len(word) * dr, c + len(word) * dc))
        
        # Check each existing word to see if it overlaps or is adjacent in the same direction
        for existing_word, existing_coords in placements.items():
            if len(existing_coords) < 2:
                continue  # Skip single-letter words
            
            # Check if existing word is in the same direction
            existing_dr = existing_coords[-1][0] - existing_coords[0][0]
            existing_dc = existing_coords[-1][1] - existing_coords[0][1]
            # Normalize direction (handle backwards)
            if existing_dr != 0:
                existing_dr = existing_dr // abs(existing_dr)
            if existing_dc != 0:
                existing_dc = existing_dc // abs(existing_dc)
            
            # Normalize the new word's direction for comparison
            norm_dr, norm_dc = dr, dc
            if norm_dr != 0:
                norm_dr = norm_dr // abs(norm_dr) if norm_dr != 0 else 0
            if norm_dc != 0:
                norm_dc = norm_dc // abs(norm_dc) if norm_dc != 0 else 0
            
            # Check if directions match (same line/row/diagonal)
            if (norm_dr, norm_dc) == (existing_dr, existing_dc) or (norm_dr, norm_dc) == (-existing_dr, -existing_dc):
                # Same direction - check for overlap or adjacency
                existing_cells = set(existing_coords)
                # Also include cells immediately before and after existing word
                if len(existing_coords) > 0:
                    existing_start = existing_coords[0]
                    existing_end = existing_coords[-1]
                    existing_cells.add((existing_start[0] - existing_dr, existing_start[1] - existing_dc))
                    existing_cells.add((existing_end[0] + existing_dr, existing_end[1] + existing_dc))
                
                overlap = new_word_cells & existing_cells
                if len(overlap) > 0:
                    # They share cells or are adjacent in the same direction - this is confusing
                    return False
        
        # Now check the actual placement
        for i, ch in enumerate(word):
            if not (0 <= rr < rows and 0 <= cc < cols):
                return False
            existing = grid[rr][cc]
            # This line allows words to CROSS when letters match (perpendicular crossings are OK)
            if existing not in ('', ch):
                return False
            # Check if this position intersects with an existing word
            if existing == ch and existing != '':
                has_intersection = True
            rr += dr
            cc += dc
        # If intersections are enforced, the word must intersect with at least one existing word
        if must_intersect and not has_intersection:
            return False
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

    unplaced_words = []
    for idx, w in enumerate(sorted(words, key=len, reverse=True)):
        placed = False
        attempts = 0
        max_attempts = rows * cols * 10
        # First word doesn't need to intersect; subsequent words do if enforce_intersections is True
        must_intersect = enforce_intersections and idx > 0
        
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

            if can_place(w, r, c, dr, dc, must_intersect):
                coords = do_place(w, r, c, dr, dc)
                placements[w] = coords
                placed = True
            attempts += 1

        if not placed:
            unplaced_words.append(w)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '':
                grid[r][c] = rng.choice(alphabet)

    # Combine too_long words with unplaced_words for reporting
    all_unplaced = too_long + unplaced_words
    return grid, placements, all_unplaced

# -----------------------------
# Codeword puzzle generator
# -----------------------------
def create_codeword(
    words_pt: List[str],
    rows: int = 15,
    cols: int = 15,
    num_hints: int = 3,
    seed: Optional[int] = 42
):
    """
    Create a codeword puzzle where each letter is represented by a number (1-26).
    Words must intersect like in a crossword puzzle.
    """
    rng = random.Random(seed)
    # Normalize and deduplicate words
    normalized_words = [normalize_word(w) for w in words_pt if normalize_word(w)]
    words = list(dict.fromkeys(normalized_words))  # Preserves order while removing duplicates
    if not words:
        raise ValueError("No valid words provided.")
    # Filter out words that are too long
    too_long = [w for w in words if len(w) > max(rows, cols)]
    words = [w for w in words if len(w) <= max(rows, cols)]
    if not words:
        raise ValueError(f"All words are too long for a {rows}x{cols} grid.")

    # For codeword, we only allow horizontal and vertical (no diagonals, no backwards)
    dirs = [(0, 1), (1, 0)]  # right, down

    grid = [['' for _ in range(cols)] for _ in range(rows)]
    placements = {}
    words_set = set(words)  # For fast lookup

    def extract_sequence(start_r: int, start_c: int, dr: int, dc: int) -> str:
        """Extract a sequence of letters in a given direction until hitting empty cell or boundary."""
        seq = []
        rr, cc = start_r, start_c
        # Go backwards to find the start of the sequence
        while 0 <= rr - dr < rows and 0 <= cc - dc < cols and grid[rr - dr][cc - dc]:
            rr -= dr
            cc -= dc
        # Now go forward to extract the full sequence
        while 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc]:
            seq.append(grid[rr][cc])
            rr += dr
            cc += dc
        return ''.join(seq)

    def validate_sequences(word: str, r: int, c: int, dr: int, dc: int) -> bool:
        """Check that placing this word doesn't create invalid sequences."""
        # Check sequences perpendicular to the word being placed
        perp_dr, perp_dc = dc, dr  # Perpendicular direction
        
        for i, ch in enumerate(word):
            rr = r + i * dr
            cc = c + i * dc
            
            # Build the perpendicular sequence as if the word is already placed
            # Go backwards to find the start
            seq = []
            start_r, start_c = rr, cc
            while 0 <= start_r - perp_dr < rows and 0 <= start_c - perp_dc < cols:
                prev_r = start_r - perp_dr
                prev_c = start_c - perp_dc
                if not grid[prev_r][prev_c]:
                    break
                start_r, start_c = prev_r, prev_c
            
            # Now go forward to extract the full sequence
            curr_r, curr_c = start_r, start_c
            while 0 <= curr_r < rows and 0 <= curr_c < cols:
                # Use the letter from the word being placed if at current position
                if curr_r == rr and curr_c == cc:
                    seq.append(ch)
                elif grid[curr_r][curr_c]:
                    seq.append(grid[curr_r][curr_c])
                else:
                    break
                curr_r += perp_dr
                curr_c += perp_dc
            
            # Validate the sequence: if longer than 1, must be a valid word
            seq_str = ''.join(seq)
            if len(seq_str) > 1 and seq_str not in words_set:
                return False
        
        # Also check sequences in the SAME direction as the word being placed
        # This handles cases where a horizontal word is placed between vertical words
        # Go backwards to find the start of the sequence in the same direction
        start_r, start_c = r, c
        while 0 <= start_r - dr < rows and 0 <= start_c - dc < cols:
            prev_r = start_r - dr
            prev_c = start_c - dc
            if not grid[prev_r][prev_c]:
                break
            start_r, start_c = prev_r, prev_c
        
        # Now go forward to extract all sequences in this direction
        # We need to find all sequences separated by empty cells
        curr_r, curr_c = start_r, start_c
        current_seq = []
        
        while 0 <= curr_r < rows and 0 <= curr_c < cols:
            # Check if this position is part of the word being placed
            is_in_word = False
            word_pos = -1
            for i in range(len(word)):
                if curr_r == r + i * dr and curr_c == c + i * dc:
                    is_in_word = True
                    word_pos = i
                    break
            
            if is_in_word:
                # Add letter from word being placed
                current_seq.append(word[word_pos])
            elif grid[curr_r][curr_c]:
                # Add existing letter
                current_seq.append(grid[curr_r][curr_c])
            else:
                # Empty cell - validate the sequence we've built so far
                if len(current_seq) > 1:
                    seq_str = ''.join(current_seq)
                    if seq_str not in words_set:
                        return False
                current_seq = []
            
            curr_r += dr
            curr_c += dc
        
        # Validate the last sequence if any
        if len(current_seq) > 1:
            seq_str = ''.join(current_seq)
            if seq_str not in words_set:
                return False
        
        return True

    def can_place(word: str, r: int, c: int, dr: int, dc: int, must_intersect: bool = False) -> bool:
        rr, cc = r, c
        has_intersection = False
        # First check basic placement constraints
        for ch in word:
            if not (0 <= rr < rows and 0 <= cc < cols):
                return False
            existing = grid[rr][cc]
            # Words must share letters at intersections
            if existing not in ('', ch):
                return False
            # Check if this position intersects with an existing word
            if existing == ch and existing != '':
                has_intersection = True
            rr += dr
            cc += dc
        # All words except the first must intersect
        if must_intersect and not has_intersection:
            return False
        # Validate that placing this word doesn't create invalid sequences
        if not validate_sequences(word, r, c, dr, dc):
            return False
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

    # Place words with intersection enforcement
    unplaced_words = []
    
    def count_common_letters(word: str, placed_words: List[str]) -> int:
        """Count how many letters this word shares with already placed words."""
        word_letters = set(word)
        count = 0
        for placed_word in placed_words:
            placed_letters = set(placed_word)
            count += len(word_letters & placed_letters)
        return count
    
    def find_intersection_points(word: str) -> List[Tuple[int, int, int, int]]:
        """Find all valid positions where this word can intersect with existing words."""
        candidates = []
        for existing_word, existing_coords in placements.items():
            if len(existing_coords) < 2:
                continue  # Skip single-letter words
            
            # Determine if existing word is horizontal or vertical
            is_horizontal = existing_coords[0][0] == existing_coords[-1][0]
            is_vertical = existing_coords[0][1] == existing_coords[-1][1]
            
            for i, ch in enumerate(word):
                for er, ec in existing_coords:
                    if grid[er][ec] == ch:  # Found matching letter
                        # Try horizontal placement intersecting with vertical word
                        if is_vertical:
                            r = er
                            c = ec - i
                            if 0 <= c < cols and c + len(word) <= cols:
                                candidates.append((r, c, 0, 1))  # horizontal
                        
                        # Try vertical placement intersecting with horizontal word
                        if is_horizontal:
                            r = er - i
                            c = ec
                            if 0 <= r < rows and r + len(word) <= rows:
                                candidates.append((r, c, 1, 0))  # vertical
        
        # Remove duplicates and limit candidates to avoid checking too many
        candidates = list(dict.fromkeys(candidates))  # Preserves order while removing duplicates
        # Shuffle to add randomness, but limit to top candidates
        rng.shuffle(candidates)
        return candidates[:min(100, len(candidates))]  # Limit to 100 candidates
    
    # Sort words: longest first, but also prioritize words with common letters
    # This helps words that are more likely to intersect get placed earlier
    sorted_words = sorted(words, key=len, reverse=True)
    
    for idx, w in enumerate(sorted_words):
        placed = False
        attempts = 0
        max_attempts = rows * cols * 30  # Increased attempts
        must_intersect = idx > 0  # All words except first must intersect
        
        # If word must intersect, try intersection points first
        if must_intersect and placements:
            intersection_candidates = find_intersection_points(w)
            for r, c, dr, dc in intersection_candidates:
                if can_place(w, r, c, dr, dc, must_intersect):
                    coords = do_place(w, r, c, dr, dc)
                    placements[w] = coords
                    placed = True
                    break
                attempts += 1
        
        # If not placed yet, try random placements
        while not placed and attempts < max_attempts:
            dr, dc = rng.choice(dirs)
            if dr == 0:  # horizontal
                r = rng.randrange(rows)
                if dc > 0:
                    c = rng.randrange(cols - len(w) + 1)
                else:
                    c = rng.randrange(len(w) - 1, cols)
            else:  # vertical
                c = rng.randrange(cols)
                if dr > 0:
                    r = rng.randrange(rows - len(w) + 1)
                else:
                    r = rng.randrange(len(w) - 1, rows)

            if can_place(w, r, c, dr, dc, must_intersect):
                coords = do_place(w, r, c, dr, dc)
                placements[w] = coords
                placed = True
            attempts += 1

        if not placed:
            unplaced_words.append(w)

    # Create letter-to-number mapping
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Collect all unique letters used in the grid
    used_letters = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                used_letters.add(grid[r][c])
    
    # Collect all letters from the original word list (placed + unplaced)
    all_words_letters = set()
    for w in words_pt:
        normalized = normalize_word(w)
        all_words_letters.update(normalized)
    
    # Find missing letters (letters in word list but not in grid)
    missing_letters = sorted(all_words_letters - used_letters)
    
    # Create mapping: assign numbers 1-26 to letters
    # First assign to used letters, then fill remaining
    letter_to_num = {}
    num_to_letter = {}
    available_numbers = list(range(1, 27))
    rng.shuffle(available_numbers)
    
    # Assign numbers to used letters first
    for letter in sorted(used_letters):
        if available_numbers:
            num = available_numbers.pop(0)
            letter_to_num[letter] = num
            num_to_letter[num] = letter
    
    # Assign remaining numbers to unused letters (for hints)
    for letter in alphabet:
        if letter not in letter_to_num and available_numbers:
            num = available_numbers.pop(0)
            letter_to_num[letter] = num
            num_to_letter[num] = letter

    # Create number grid
    number_grid = [['' for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                number_grid[r][c] = str(letter_to_num[grid[r][c]])
            else:
                number_grid[r][c] = ''

    # Select hints (letter-number pairs to reveal)
    used_letters_list = sorted(list(used_letters))
    if len(used_letters_list) > 0:
        num_hints = min(num_hints, len(used_letters_list))
        hint_letters = rng.sample(used_letters_list, num_hints)
        hints = [(letter, letter_to_num[letter]) for letter in hint_letters]
    else:
        hints = []

    # Combine too_long words with unplaced_words for reporting
    all_unplaced = too_long + unplaced_words
    return number_grid, grid, placements, letter_to_num, num_to_letter, hints, all_unplaced, missing_letters

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
    has_secret_message: bool = False,
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
    
    # Add secret message instructions if applicable
    if has_secret_message:
        y -= 0.02
        dica_text = "DICA: Depois de encontrares todas as palavras, ordena as restantes para descobrires a mensagem secreta!"
        wrapped_dica = textwrap.wrap(dica_text, width=70)
        for line in wrapped_dica:
            ax.text(0.1, y, line, fontsize=11, style='italic', color='darkblue', fontweight='bold')
            y -= 0.025

    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

def render_solution(
    grid: List[List[str]],
    placements: dict,
    output_png: Path = Path("sopa_de_letras_solucao.png"),
    output_pdf: Path = Path("sopa_de_letras_solucao.pdf"),
    secret_message: Optional[List[str]] = None,
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
    
    # Display secret message if provided
    y = 0.15
    if secret_message:
        ax.text(0.1, y, "Mensagem Secreta:", fontsize=14, fontweight='bold', color='darkblue')
        y -= 0.025
        message_txt = " ".join(secret_message)
        wrapped_msg = textwrap.wrap(message_txt, width=70)
        for line in wrapped_msg:
            ax.text(0.1, y, line, fontsize=13, style='italic', color='darkblue', fontweight='bold')
            y -= 0.025

    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

def render_codeword_worksheet(
    number_grid: List[List[str]],
    words_display: List[str],
    hints: List[Tuple[str, int]],
    title: str = "Codeword",
    subtitle: str = "Descobre o código numérico!",
    output_png: Path = Path("codeword.png"),
    output_pdf: Path = Path("codeword.pdf"),
    has_secret_message: bool = False,
):
    rows, cols = len(number_grid), len(number_grid[0])
    fig = plt.figure(figsize=(8.27, 11.69), dpi=200)  # A4 portrait
    ax = plt.gca()
    ax.axis('off')

    ax.text(0.5, 0.96, title, ha='center', va='center', fontsize=28, fontweight='bold')
    ax.text(0.5, 0.925, subtitle, ha='center', va='center', fontsize=14)

    # Create display grid (show numbers, empty cells as blank)
    cell_text = [['' for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if number_grid[r][c]:
                cell_text[r][c] = number_grid[r][c]
            else:
                cell_text[r][c] = ''

    table = plt.table(
        cellText=cell_text,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.35, 0.8, 0.5]
    )

    # Style the table cells - paint empty cells black
    for (r, c), cell in table.get_celld().items():
        if c == -1:  # Skip column header cells
            continue
        if r < 0:  # Skip row header cells
            continue
        if r >= rows or c >= cols:  # Skip out of bounds
            continue
        cell.get_text().set_fontsize(11)
        if number_grid[r][c] == '':  # Empty cell
            cell.set_facecolor('black')
            cell.get_text().set_text('')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)

    # Display hints
    y = 0.28
    ax.text(0.1, y, "Dicas:", fontsize=14, fontweight='bold')
    y -= 0.025
    hints_text = ", ".join([f"{letter}={num}" for letter, num in sorted(hints, key=lambda x: x[1])])
    wrapped_hints = textwrap.wrap(hints_text, width=70)
    for line in wrapped_hints:
        ax.text(0.1, y, line, fontsize=11)
        y -= 0.02

    # Display words
    y -= 0.02
    words_txt = ", ".join(words_display)
    wrapped = textwrap.wrap(words_txt, width=60)
    ax.text(0.1, y, "Palavras:", fontsize=14, fontweight='bold')
    y -= 0.025
    for line in wrapped:
        ax.text(0.1, y, line, fontsize=11)
        y -= 0.02
    
    # Add secret message instructions if applicable
    if has_secret_message:
        y -= 0.02
        dica_text = "DICA: Depois de encontrares todas as palavras, ordena as restantes para descobrires a mensagem secreta!"
        wrapped_dica = textwrap.wrap(dica_text, width=70)
        for line in wrapped_dica:
            ax.text(0.1, y, line, fontsize=10, style='italic', color='darkblue', fontweight='bold')
            y -= 0.02

    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

def render_codeword_solution(
    letter_grid: List[List[str]],
    number_grid: List[List[str]],
    placements: dict,
    letter_to_num: dict,
    output_png: Path = Path("codeword_solucao.png"),
    output_pdf: Path = Path("codeword_solucao.pdf"),
    secret_message: Optional[List[str]] = None,
):
    rows, cols = len(letter_grid), len(letter_grid[0])
    fig = plt.figure(figsize=(8.27, 11.69), dpi=200)
    ax = plt.gca()
    ax.axis('off')

    ax.text(0.5, 0.96, "Codeword — Solução", ha='center', va='center', fontsize=26, fontweight='bold')
    ax.text(0.5, 0.925, "Letras reveladas (células vazias ficam em branco)", ha='center', va='center', fontsize=12)

    # Show letters in the grid
    cell_text = [['' for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if letter_grid[r][c]:
                cell_text[r][c] = letter_grid[r][c]
            else:
                cell_text[r][c] = ''

    table = plt.table(
        cellText=cell_text,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.2, 0.8, 0.6]
    )
    
    # Style the table cells - paint empty cells black
    for (r, c), cell in table.get_celld().items():
        if c == -1:  # Skip column header cells
            continue
        if r < 0:  # Skip row header cells
            continue
        if r >= rows or c >= cols:  # Skip out of bounds
            continue
        cell.get_text().set_fontsize(12)
        if letter_grid[r][c] == '':  # Empty cell
            cell.set_facecolor('black')
            cell.get_text().set_text('')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)

    # Display the full mapping
    y = 0.15
    ax.text(0.1, y, "Código completo:", fontsize=14, fontweight='bold')
    y -= 0.025
    # Group mapping by number ranges for better display
    mapping_lines = []
    current_line = []
    # Sort by number (value) for display
    sorted_mappings = sorted(letter_to_num.items(), key=lambda x: x[1])
    for letter, num in sorted_mappings:
        current_line.append(f"{letter}={num}")
        if len(current_line) >= 6:  # 6 pairs per line
            mapping_lines.append(", ".join(current_line))
            current_line = []
    if current_line:
        mapping_lines.append(", ".join(current_line))
    
    for line in mapping_lines:
        wrapped = textwrap.wrap(line, width=80)
        for wline in wrapped:
            ax.text(0.1, y, wline, fontsize=10)
            y -= 0.018
    
    # Display secret message if provided
    if secret_message:
        y -= 0.02
        ax.text(0.1, y, "Mensagem Secreta:", fontsize=14, fontweight='bold', color='darkblue')
        y -= 0.025
        message_txt = " ".join(secret_message)
        wrapped_msg = textwrap.wrap(message_txt, width=70)
        for line in wrapped_msg:
            ax.text(0.1, y, line, fontsize=13, style='italic', color='darkblue', fontweight='bold')
            y -= 0.025

    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate word search ('sopa de letras') and codeword puzzles from a word list.")
    parser.add_argument('--words-file', '-f', help='Path to file with words (one per line). Lines starting with # ignored.')
    parser.add_argument('--rows', type=int, default=12, help='Number of rows in the word search grid.')
    parser.add_argument('--cols', type=int, default=12, help='Number of columns in the word search grid.')
    parser.add_argument('--allow-diagonals', action='store_true', help='Allow diagonal placements in word search.')
    parser.add_argument('--allow-backwards', action='store_true', help='Allow backwards placements in word search.')
    parser.add_argument('--enforce-intersections', action='store_true', help='Enforce that words must intersect in word search.')
    parser.add_argument('--codeword-rows', type=int, default=15, help='Number of rows in the codeword grid.')
    parser.add_argument('--codeword-cols', type=int, default=15, help='Number of columns in the codeword grid.')
    parser.add_argument('--codeword-hints', type=int, default=3, help='Number of letter-number hints to provide in codeword.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output-prefix', default='sopa_de_letras', help='Output files prefix (no extension).')
    parser.add_argument('--message-file', '-m', help='Path to file with secret message words (one per line). These words will be added to the word list but NOT placed in the grid. Lines starting with # are ignored.')
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

    # Load secret message words if provided
    message_words = []
    message_words_original = []
    if args.message_file:
        p_msg = Path(args.message_file)
        if not p_msg.exists():
            print(f"Message file not found: {p_msg}", file=sys.stderr)
            sys.exit(2)
        with p_msg.open(encoding='utf-8') as fh:
            # keep original word forms for display; ignore blank lines and comments
            message_words_raw = [line.strip() for line in fh if line.strip() and not line.strip().startswith('#')]
            # Deduplicate while preserving order (based on normalized form)
            seen_msg = set()
            for word in message_words_raw:
                normalized = normalize_word(word)
                if normalized not in seen_msg:
                    seen_msg.add(normalized)
                    message_words_original.append(word)
                    message_words.append(normalized)
        
        if message_words:
            # Message words are added to the word list but NOT to the grid
            # Combine regular words and message words, then shuffle all together for display
            all_words_for_display = palavras + message_words_original
            rng_shuffle = random.Random(args.seed)
            rng_shuffle.shuffle(all_words_for_display)
            # Keep only regular words for grid placement (message words stay out of grid)
            # palavras stays as regular words only for grid generation
            print(f"Added {len(message_words_original)} secret message word(s) to the word list (not in grid). All words shuffled together.", file=sys.stderr)

    # Generate word search puzzle
    print("Generating word search puzzle...", file=sys.stderr)
    grid, placements, unplaced_words_ws = create_letter_soup(
        palavras,
        rows=args.rows,
        cols=args.cols,
        allow_diagonals=args.allow_diagonals,
        allow_backwards=args.allow_backwards,
        enforce_intersections=args.enforce_intersections,
        seed=args.seed
    )

    # Prepare word list for display
    # Message words are in the word list but NOT in the grid
    if message_words:
        # Combine regular words (from grid) with message words (not in grid) for display
        # Filter out unplaced regular words
        placed_regular_words = [w for w in palavras if normalize_word(w) not in unplaced_words_ws]
        # Shuffle all words together (placed regular words + message words)
        all_words_display = placed_regular_words + message_words_original
        rng_display = random.Random(args.seed)
        rng_display.shuffle(all_words_display)
        palavras_ws = all_words_display
    else:
        # No message words - just show placed regular words
        palavras_ws = [w for w in palavras if normalize_word(w) not in unplaced_words_ws]
    if unplaced_words_ws:
        print(f"Warning: Could not place {len(unplaced_words_ws)} word(s) in word search: {', '.join(unplaced_words_ws)}", file=sys.stderr)
        print(f"These words will not appear in the puzzle.", file=sys.stderr)

    # Message words are not in the grid, so they're always the secret message
    secret_message = None
    has_secret_message = False
    if message_words:
        # All message words form the secret message (they're not in the grid)
        secret_message = message_words_original
        has_secret_message = len(secret_message) > 0
        if has_secret_message:
            print(f"Secret message contains {len(secret_message)} word(s): {' '.join(secret_message)}", file=sys.stderr)

    out_prefix = Path(args.output_prefix)
    worksheet_png = out_prefix.with_suffix('.png')
    worksheet_pdf = out_prefix.with_suffix('.pdf')
    solution_png = out_prefix.with_name(out_prefix.stem + '_solucao.png')
    solution_pdf = out_prefix.with_name(out_prefix.stem + '_solucao.pdf')

    # Show secret message hint whenever there's a secret message
    render_worksheet(grid, palavras_ws, output_png=worksheet_png, output_pdf=worksheet_pdf, 
                    has_secret_message=has_secret_message)
    render_solution(grid, placements, output_png=solution_png, output_pdf=solution_pdf, 
                   secret_message=secret_message)
    print(f"Word search puzzle saved: {worksheet_png}, {worksheet_pdf}", file=sys.stderr)
    print(f"Word search solution saved: {solution_png}, {solution_pdf}", file=sys.stderr)

    # Generate codeword puzzle
    # Exclude message words from codeword puzzles
    palavras_codeword = palavras
    if message_words:
        message_words_set = set(message_words)
        palavras_codeword = [w for w in palavras if normalize_word(w) not in message_words_set]
        print(f"Excluding {len(message_words_original)} message word(s) from codeword puzzle.", file=sys.stderr)
    
    print("Generating codeword puzzle...", file=sys.stderr)
    try:
        number_grid, letter_grid, codeword_placements, letter_to_num, num_to_letter, hints, unplaced_words_cw, missing_letters = create_codeword(
            palavras_codeword,
            rows=args.codeword_rows,
            cols=args.codeword_cols,
            num_hints=args.codeword_hints,
            seed=args.seed + 1  # Use different seed for codeword
        )

        # Filter out unplaced words from display
        palavras_cw = [w for w in palavras_codeword if normalize_word(w) not in unplaced_words_cw]
        if unplaced_words_cw:
            print(f"Warning: Could not place {len(unplaced_words_cw)} word(s) in codeword: {', '.join(unplaced_words_cw)}", file=sys.stderr)
            print(f"These words will not appear in the puzzle.", file=sys.stderr)
        
        # Warn about missing letters
        if missing_letters:
            # Count unique letters actually in the grid
            used_letters_in_grid = set()
            for r in range(len(letter_grid)):
                for c in range(len(letter_grid[0])):
                    if letter_grid[r][c]:
                        used_letters_in_grid.add(letter_grid[r][c])
            
            # Count all unique letters from all words (placed + unplaced)
            all_word_letters = set()
            for w in palavras:
                all_word_letters.update(normalize_word(w))
            
            num_used = len(used_letters_in_grid)
            num_missing = len(missing_letters)
            total_letters = len(all_word_letters)
            coverage_pct = (num_used / total_letters * 100) if total_letters > 0 else 0
            
            print(f"Warning: {num_missing} letter(s) from the word list are missing from the codeword puzzle: {', '.join(missing_letters)}", file=sys.stderr)
            print(f"Letter coverage: {num_used}/{total_letters} unique letters from word list ({coverage_pct:.1f}%)", file=sys.stderr)
            
            if num_missing > 3 or coverage_pct < 70:
                print(f"Consider adding words containing these letters to improve puzzle coverage.", file=sys.stderr)

        codeword_prefix = out_prefix.with_name(out_prefix.stem + '_codeword')
        codeword_png = codeword_prefix.with_suffix('.png')
        codeword_pdf = codeword_prefix.with_suffix('.pdf')
        codeword_solution_png = codeword_prefix.with_name(codeword_prefix.stem + '_solucao.png')
        codeword_solution_pdf = codeword_prefix.with_name(codeword_prefix.stem + '_solucao.pdf')

        render_codeword_worksheet(
            number_grid, palavras_cw, hints,
            output_png=codeword_png, output_pdf=codeword_pdf,
            has_secret_message=False
        )
        render_codeword_solution(
            letter_grid, number_grid, codeword_placements, letter_to_num,
            output_png=codeword_solution_png, output_pdf=codeword_solution_pdf,
            secret_message=None
        )
        print(f"Codeword puzzle saved: {codeword_png}, {codeword_pdf}", file=sys.stderr)
        print(f"Codeword solution saved: {codeword_solution_png}, {codeword_solution_pdf}", file=sys.stderr)
    except RuntimeError as e:
        print(f"Warning: Could not generate codeword puzzle: {e}", file=sys.stderr)
        print("Word search puzzle was generated successfully.", file=sys.stderr)