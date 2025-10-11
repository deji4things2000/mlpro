# CSP Solver for Map Coloring and Circuit Board Layout

## Overview
This project implements a general-purpose Constraint Satisfaction Problem (CSP) solver with backtracking search, heuristics (MRV, degree, LCV), and inference (AC-3). It solves two types of problems: map coloring and circuit board layout.

## Requirements
- Python 3.6+

## How to Run
1. python test_csp_py
2. Extension: python sudoku.py


```bash
python test_csp.py


## Key Features

1. **Generic CSP Solver**: Handles any CSP with binary constraints
2. **Complete Heuristic Support**: MRV, degree heuristic, and LCV
3. **AC-3 Inference**: Constraint propagation for early detection of failures
4. **Modular Design**: Easy to add new problem types
5. **Performance Testing**: Compare different heuristic combinations

## Answers to Discussion Questions

### Map Coloring Results
The solver efficiently solves map coloring problems. Heuristics significantly reduce search time:
- MRV reduces variable selection time
- LCV helps maintain flexibility
- AC-3 prunes domains early
- Combined heuristics provide best performance

### Circuit Board Domain
For a component of width `w` and height `h` on an `n × m` board, the domain consists of all positions `(x, y)` where:
- `0 ≤ x ≤ n - w`
- `0 ≤ y ≤ m - h`

### Non-Overlap Constraint
For components a (3×2) and b (5×2) on 10×3 board, the constraint ensures that for any positions (x_a, y_a) and (x_b, y_b), the rectangles are not overlapping.


### Constraint Representation
Constraints are represented as Python functions that take assigned values and return boolean. The generic CSP solver handles these through the Constraint interface, making it problem-agnostic.

This implementation provides a complete, efficient CSP solver that demonstrates the power of heuristics and inference in constraint satisfaction problems.