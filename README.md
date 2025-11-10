# Sopa de Letras

A Python script to generate word search puzzles ("sopa de letras" in Portuguese) from a list of words. The script creates both a puzzle worksheet and a solution sheet in PNG and PDF formats.

## Features

- Generate word search puzzles from a custom word list
- Support for Portuguese characters (accents are normalized)
- Configurable grid size
- Optional diagonal and backward word placements
- Deterministic output via random seed
- Automatic word deduplication
- Words can cross each other when letters match
- Generates both puzzle and solution files

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

### Command-Line Options

- `--words-file`, `-f`: Path to file with words (one per line). Lines starting with `#` are ignored.
- `--rows`: Number of rows in the grid (default: 12)
- `--cols`: Number of columns in the grid (default: 12)
- `--allow-diagonals`: Allow diagonal word placements
- `--allow-backwards`: Allow backward word placements (left and up)
- `--seed`: Random seed for reproducible puzzles (default: 42)
- `--output-prefix`: Output files prefix without extension (default: `sopa_de_letras`)

## Output Files

The script generates four files:

- `{prefix}.png` - Puzzle worksheet (PNG format)
- `{prefix}.pdf` - Puzzle worksheet (PDF format)
- `{prefix}_solucao.png` - Solution sheet (PNG format)
- `{prefix}_solucao.pdf` - Solution sheet (PDF format)

The solution sheet shows only the letters that are part of words (other cells are shown as dots).

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

## How It Works

1. Words are normalized: accents removed, converted to uppercase, spaces/hyphens removed
2. Duplicate words (after normalization) are removed
3. Words are placed in the grid starting with the longest words first
4. Words can cross each other when letters match
5. Empty cells are filled with random letters
6. The puzzle and solution are rendered as A4-sized images

## Notes

- If a word is too long for the grid, the script will raise an error
- If a word cannot be placed after many attempts, the script will suggest using a larger grid
- The script preserves the original word forms (with accents) for display in the puzzle, but uses normalized forms for placement

## License

See LICENSE file for details.
