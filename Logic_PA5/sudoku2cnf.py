#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran
# Date: 11/15/2025 

from Sudoku import Sudoku
import sys

if __name__ == "__main__":
    test_sudoku = Sudoku()

    test_sudoku.load(sys.argv[1])
    print(test_sudoku)

    puzzle_name = sys.argv[1][:-4]
    cnf_filename = puzzle_name + ".cnf"

    test_sudoku.generate_cnf(cnf_filename)
    print("Output file: " + cnf_filename)

