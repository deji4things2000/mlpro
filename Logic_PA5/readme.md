**Course:** Artificial Intelligence  
**Author:** Adedeji Sunday Adediran  
**Institution:** Dartmouth College  
**Date:** November 2025


# PA5 - Logic: SAT Solvers for Sudoku

## Overview
This project implements GSAT and WalkSAT algorithms for solving propositional logic satisfiability problems, specifically applied to Sudoku puzzles converted to CNF format.

## Files
- `SAT.py` - Main implementation of GSAT and WalkSAT algorithms
- `solve_sudoku.py` - Main script to solve Sudoku CNF files
- `README.md` - This file
- `report.pdf` - Project report

## Requirements
- Python 3.6+
- No external dependencies required

## Usage

### Basic Usage
```bash
# Solve using WalkSAT (default)
python solve_sudoku.py puzzle1.cnf

# Solve using GSAT
python solve_sudoku.py puzzle1.cnf gsat

# Set custom iteration limit
python solve_sudoku.py puzzle1.cnf walksat 50000
