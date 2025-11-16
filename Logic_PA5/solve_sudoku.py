#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran
# Date: 11/15/2025 

#!/usr/bin/env python3
"""
solve_sudoku.py - Solve Sudoku problems using SAT solvers
"""

import sys
import time
from SAT import solve_sat, save_solution

def main():
    if len(sys.argv) < 2:
        print("Usage: python solve_sudoku.py <cnf_file> [algorithm] [max_iterations]")
        print("  algorithm: gsat or walksat (default: walksat)")
        print("  max_iterations: maximum iterations (default: 100000)")
        sys.exit(1)
    
    cnf_file = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 else 'walksat'
    max_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    
    print(f"Solving {cnf_file} using {algorithm.upper()}...")
    print(f"Max iterations: {max_iterations}")
    
    start_time = time.time()
    solution = solve_sat(cnf_file, algorithm=algorithm, max_iterations=max_iterations)
    end_time = time.time()
    
    if solution:
        print(f"Solution found in {end_time - start_time:.2f} seconds!")
        
        # Save solution
        sol_file = cnf_file.replace('.cnf', '.sol')
        save_solution(solution, sol_file)
        print(f"Solution saved to {sol_file}")
        
        # Display basic stats
        true_vars = sum(1 for value in solution.values() if value)
        print(f"True variables: {true_vars}")
        
    else:
        print("No solution found within iteration limit.")
        sys.exit(1)

if __name__ == "__main__":
    main()