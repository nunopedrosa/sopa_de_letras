# Sopa de Letras

A Python script to generate word search puzzles ("sopa de letras" in Portuguese) and codeword puzzles from a list of words. The script creates both puzzle worksheets and solution sheets in PNG and PDF formats.

## Features

- Generate word search puzzles from a custom word list
- Generate codeword puzzles (crossword-style with number-to-letter mapping)
- **Secret message feature** for scavenger hunts: add hidden message words that need to be sorted to reveal the message
- Support for Portuguese characters (accents are normalized)
- Configurable grid size for both puzzle types
- Optional diagonal and backward word placements (word search only)
- Optional intersection enforcement for word search
- Deterministic output via random seed
- Automatic word deduplication
- Words can cross each other when letters match
- Smart placement strategies to minimize word rejections
- Validates all sequences in codeword puzzles to ensure only valid words appear
- Generates both puzzle and solution files for each puzzle type

## Requirements

- Python 3.6 or higher
- matplotlib (see `requirements.txt`)

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Generate a puzzle with default settings (12x12 grid):

```bash
python sopa_de_letras.py
```

### Using a Word File

Provide a file with words (one per line):

```bash
python sopa_de_letras.py --words-file palavras.txt
```

Lines starting with `#` are treated as comments and ignored. Blank lines are also ignored.

### Secret Message Feature (Scavenger Hunts)

Add a secret message to your word search puzzle! Words from a message file are added to the word list (shuffled with regular words) but **NOT placed in the grid**. Players find all the words that are actually in the grid, and the remaining words from the list (the message words that aren't in the grid) need to be sorted to reveal the secret message.

**Note:** Secret messages are only used in word search puzzles, not in codeword puzzles. Message words are automatically excluded from codeword puzzles.

```bash
python sopa_de_letras.py --words-file palavras.txt --message-file message.txt
```

This is perfect for less challenging scavenger hunts for young children (5-year-olds). Players see all words in the list (regular words + message words, shuffled together), find the ones that are actually in the grid, and the words they can't find (the message words) form the secret message when sorted.

The message file format is the same as the word file (one word per line). The words in the message file will be:
1. Added to the word list and shuffled together with regular words (using the same seed as the puzzle)
2. **NOT placed in the grid** (they're only in the word list)
3. Revealed in the solution file (in the correct order)

Example `message.txt`:
```
encontra
o
tesouro
na
biblioteca
```

**How it works:** Players see all words in the list (shuffled). They find the words that are actually in the grid. The words they can't find in the grid are the message words. These need to be sorted to reveal the secret message: "encontra o tesouro na biblioteca"

### Custom Grid Size

Specify the number of rows and columns:

```bash
python sopa_de_letras.py --words-file palavras.txt --rows 15 --cols 15
```

### Advanced Options

Allow diagonal and backward word placements:

```bash
python sopa_de_letras.py --words-file palavras.txt --allow-diagonals --allow-backwards
```

Set a custom random seed for reproducible puzzles:

```bash
python sopa_de_letras.py --words-file palavras.txt --seed 123
```

Custom output file prefix:

```bash
python sopa_de_letras.py --words-file palavras.txt --output-prefix my_puzzle
```

### Codeword Puzzle Options

Generate both word search and codeword puzzles with custom settings:

```bash
python sopa_de_letras.py --words-file palavras.txt --codeword-rows 18 --codeword-cols 18 --codeword-hints 5
```

Increase the number of hints in the codeword puzzle (more hints make it easier to solve):

```bash
python sopa_de_letras.py --words-file palavras.txt --codeword-hints 8
```

Enable intersection enforcement for word search:

```bash
python sopa_de_letras.py --words-file palavras.txt --enforce-intersections
```

### Command-Line Options

**Word Search Options:**
- `--words-file`, `-f`: Path to file with words (one per line). Lines starting with `#` are ignored.
- `--message-file`, `-m`: Path to file with secret message words (one per line). These words will be added to the word list but NOT placed in the grid. Lines starting with `#` are ignored.
- `--rows`: Number of rows in the word search grid (default: 15)
- `--cols`: Number of columns in the word search grid (default: 15)
- `--allow-diagonals`: Allow diagonal word placements in word search
- `--allow-backwards`: Allow backward word placements (left and up) in word search
- `--enforce-intersections`: Enforce that words must intersect with at least one existing word

**Codeword Options:**
- `--codeword-rows`: Number of rows in the codeword grid (default: 15)
- `--codeword-cols`: Number of columns in the codeword grid (default: 15)
- `--codeword-hints`: Number of letter-number hints to provide in the codeword puzzle (default: 3). More hints make the puzzle easier to solve. The actual number of hints may be less if there are fewer unique letters in the placed words.

**General Options:**
- `--seed`: Random seed for reproducible puzzles (default: 42)
- `--output-prefix`: Output files prefix without extension (default: `sopa_de_letras`)

## Output Files

The script generates files for both puzzle types:

**Word Search Files:**
- `{prefix}.png` - Word search puzzle worksheet (PNG format)
- `{prefix}.pdf` - Word search puzzle worksheet (PDF format)
- `{prefix}_solucao.png` - Word search solution sheet (PNG format)
- `{prefix}_solucao.pdf` - Word search solution sheet (PDF format)

**Codeword Files:**
- `{prefix}_codeword.png` - Codeword puzzle worksheet (PNG format)
- `{prefix}_codeword.pdf` - Codeword puzzle worksheet (PDF format)
- `{prefix}_codeword_solucao.png` - Codeword solution sheet (PNG format)
- `{prefix}_codeword_solucao.pdf` - Codeword solution sheet (PDF format)

The word search solution shows only the letters that are part of words (other cells are shown as dots). If a secret message is included, it will be displayed at the bottom of the solution file. The codeword puzzle shows numbers instead of letters, with empty cells painted black. The codeword solution shows the letters and the complete number-to-letter mapping. Note that secret messages are only used in word search puzzles, not codeword puzzles.

## Word File Format

Create a text file with one word per line. Example (`palavras.txt`):

```
reino
afonso
castelo
leão
combate
muçulmano
```

- Words can contain spaces and hyphens (they will be removed)
- Accented characters are normalized (e.g., "leão" becomes "LEAO")
- Duplicate words (after normalization) are automatically removed
- Lines starting with `#` are treated as comments
- Blank lines are ignored

**Message File Format:**

The message file uses the same format as the word file. The words in the message file will be shuffled and added to the puzzle word list. When players solve the puzzle, they need to identify which words are the message words (the ones not in the regular word list) and sort them to reveal the secret message. The solution file shows the secret message in the correct order.

## How It Works

### Word Search Generation

1. Words are normalized: accents removed, converted to uppercase, spaces/hyphens removed
2. Duplicate words (after normalization) are removed
3. Words that are too long for the grid are filtered out (with a warning)
4. Words are placed in the grid starting with the longest words first
5. Words can cross each other when letters match
6. If `--enforce-intersections` is enabled, words (except the first) must intersect with at least one existing word
7. Empty cells are filled with random letters
8. The puzzle and solution are rendered as A4-sized images

### Codeword Generation

Codeword puzzles use a more sophisticated placement strategy:

1. **Word Normalization**: Same as word search (accents removed, uppercase, etc.)
2. **Word Ordering**: Words are sorted by length (longest first) to maximize intersection opportunities
3. **Smart Intersection Finding**: 
   - For words that must intersect, the algorithm first searches for all valid intersection points with existing words
   - Intersection candidates are tried before random placement attempts
   - This significantly improves placement success rate
4. **Sequence Validation**: 
   - All horizontal and vertical sequences (between black cells or grid boundaries) are validated
   - Sequences of 2+ letters must be valid words from the word list
   - Single letters are allowed (common in codeword puzzles)
   - This ensures no invalid letter combinations appear in the puzzle
5. **Intersection Enforcement**: All words except the first must intersect with at least one existing word
6. **Number Mapping**: Each letter is assigned a unique number (1-26), with numbers shuffled randomly
7. **Hints**: A configurable number of letter-number pairs are revealed to help solvers start. Use the `--codeword-hints` parameter to control how many hints are provided (default: 3). More hints make the puzzle easier, while fewer hints make it more challenging. The actual number of hints is limited by the number of unique letters in the placed words.
8. **Empty Cells**: Cells without letters are painted black in the puzzle
9. **Letter Coverage Analysis**: 
   - The system tracks which letters from the word list appear in the puzzle
   - Warnings are issued when letters from the word list are missing from the grid
   - Coverage percentage is calculated and displayed
   - Suggestions are provided to add words containing missing letters when coverage is low (<70%) or more than 3 letters are missing

### Placement Strategies

The codeword generator uses several strategies to minimize word rejections:

- **Intersection-First Placement**: When a word must intersect, intersection points are tried before random positions
- **Increased Attempts**: More placement attempts (30x grid size) to find valid positions
- **Candidate Limiting**: Intersection candidates are limited to top 100 to balance thoroughness and performance
- **Validation Order**: Basic constraints (bounds, letter matching) are checked before expensive sequence validation

## Notes

- If words are too long for the grid, they are automatically filtered out with a warning
- If words cannot be placed after many attempts, they are skipped with a warning (not an error)
- Unplaced words are automatically removed from the displayed word list
- The script preserves the original word forms (with accents) for display in the puzzle, but uses normalized forms for placement
- Codeword puzzles require more grid space than word search puzzles due to intersection requirements
- The validation in codeword puzzles ensures all sequences are valid words, which may result in more rejections but guarantees puzzle quality
- **Letter Coverage Warnings**: For codeword puzzles, the script monitors letter coverage:
  - Compares letters actually placed in the grid vs. all unique letters from the word list
  - Issues warnings when letters are missing, showing which letters are absent
  - Displays coverage percentage (e.g., "17/21 unique letters (81.0%)")
  - Suggests adding words containing missing letters when coverage is below 70% or more than 3 letters are missing
  - This helps ensure puzzles have good letter distribution for solvability

## Suggested Improvements

### Alternative Secret Message Methods

The current implementation uses a word-sorting method where players must find all words and then sort the remaining words to reveal the message. Here are other methods that could be implemented, all of which require actually solving the puzzle (not just looking at the word list):

1. **Grid Position Order Method**: After finding all words, order them by their position in the grid (top-to-bottom, left-to-right), then extract first/last letters or sort them to reveal the message.

2. **Direction-Based Encoding**: Encode letters based on the direction each word was found:
   - Horizontal words → certain letters
   - Vertical words → other letters
   - Diagonal words → other letters
   - Or use the sequence of directions (H, V, D, etc.) as a cipher

3. **Coordinate Cipher**: Use the starting coordinates (row, column) of each found word to:
   - Point to specific cells in the grid that contain message letters
   - Encode numbers that map to letters

4. **Intersection Sequence**: Collect letters where words intersect, in the order words are found, to form the message.

5. **Find Order Method**: Order words by the order they were found (requires tracking), then extract letters or sort to reveal the message.

6. **Unused Letters Pattern**: After finding all words, the remaining letters in the grid (not part of any word) form a pattern or message when read in a specific order (left-to-right, top-to-bottom, or a path).

These methods all require completing the puzzle first, making them suitable for scavenger hunts and educational activities.

## License

See LICENSE file for details.
