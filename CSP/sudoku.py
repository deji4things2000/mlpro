import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional
import random
from csp import CSP, BinaryConstraint

class SudokuProblem:
    """Sudoku CSP Problem that wraps integer-based CSP"""
    
    def __init__(self, initial_board: List[List[int]]):
        self.size = 9
        self.initial_board = initial_board
        
        # Create variable mapping: each cell is a variable (0-80)
        self.cell_to_index = {}
        self.index_to_cell = {}
        idx = 0
        for row in range(self.size):
            for col in range(self.size):
                self.cell_to_index[(row, col)] = idx
                self.index_to_cell[idx] = (row, col)
                idx += 1
    
    def create_csp(self) -> CSP:
        num_variables = self.size * self.size
        domains = []
        
        # Build domains based on initial board
        for row in range(self.size):
            for col in range(self.size):
                if self.initial_board[row][col] != 0:
                    # Fixed value - domain has only one option
                    domains.append([self.initial_board[row][col] - 1])  # Convert to 0-8
                else:
                    # Empty cell - domain is all possible values (0-8)
                    domains.append(list(range(self.size)))
        
        csp = CSP(num_variables, domains)
        
        # Add row constraints: all different in each row
        for row in range(self.size):
            for col1 in range(self.size):
                for col2 in range(col1 + 1, self.size):
                    var1 = self.cell_to_index[(row, col1)]
                    var2 = self.cell_to_index[(row, col2)]
                    
                    def not_equal_constraint(val1, val2):
                        return val1 != val2
                    
                    csp.add_constraint(BinaryConstraint(var1, var2, not_equal_constraint))
        
        # Add column constraints: all different in each column
        for col in range(self.size):
            for row1 in range(self.size):
                for row2 in range(row1 + 1, self.size):
                    var1 = self.cell_to_index[(row1, col)]
                    var2 = self.cell_to_index[(row2, col)]
                    
                    def not_equal_constraint(val1, val2):
                        return val1 != val2
                    
                    csp.add_constraint(BinaryConstraint(var1, var2, not_equal_constraint))
        
        # Add box constraints: all different in each 3x3 box
        for box_row in range(0, self.size, 3):
            for box_col in range(0, self.size, 3):
                cells_in_box = []
                for r in range(3):
                    for c in range(3):
                        cells_in_box.append(self.cell_to_index[(box_row + r, box_col + c)])
                
                # Add constraints for all pairs in the box
                for i in range(len(cells_in_box)):
                    for j in range(i + 1, len(cells_in_box)):
                        
                        def not_equal_constraint(val1, val2):
                            return val1 != val2
                        
                        csp.add_constraint(BinaryConstraint(cells_in_box[i], cells_in_box[j], not_equal_constraint))
        
        return csp
    
    def translate_solution(self, int_solution: Optional[List[int]]) -> Optional[List[List[int]]]:
        """Translate integer-based solution back to 9x9 board format"""
        if int_solution is None:
            return None
        
        solution = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for var_idx, value in enumerate(int_solution):
            if value is not None:
                row, col = self.index_to_cell[var_idx]
                solution[row][col] = value + 1  # Convert back to 1-9
        
        return solution

class SudokuGenerator:
    """Generates random Sudoku puzzles of varying difficulty"""
    
    @staticmethod
    def generate_solved_sudoku():
        """Generate a complete solved Sudoku board"""
        # Start with a valid base pattern
        base = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7, 8, 9, 1],
            [5, 6, 7, 8, 9, 1, 2, 3, 4],
            [8, 9, 1, 2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8, 9, 1, 2],
            [6, 7, 8, 9, 1, 2, 3, 4, 5],
            [9, 1, 2, 3, 4, 5, 6, 7, 8]
        ]
        
        # Randomize by swapping rows/columns and numbers
        for _ in range(20):
            # Swap two random rows within the same band
            band = random.randint(0, 2)
            row1 = band * 3 + random.randint(0, 2)
            row2 = band * 3 + random.randint(0, 2)
            base[row1], base[row2] = base[row2], base[row1]
            
            # Swap two random columns within the same stack
            stack = random.randint(0, 2)
            col1 = stack * 3 + random.randint(0, 2)
            col2 = stack * 3 + random.randint(0, 2)
            for row in base:
                row[col1], row[col2] = row[col2], row[col1]
        
        return base
    
    @staticmethod
    def generate_puzzle(difficulty="medium"):
        """Generate a Sudoku puzzle by removing numbers from a solved board"""
        solved = SudokuGenerator.generate_solved_sudoku()
        puzzle = [row[:] for row in solved]  # Make a copy
        
        # Determine how many cells to clear based on difficulty
        difficulty_levels = {
            "easy": (35, 40),      # 35-40 given numbers
            "medium": (25, 30),    # 25-30 given numbers  
            "hard": (20, 25),      # 20-25 given numbers
            "expert": (17, 20)     # 17-20 given numbers
        }
        
        min_given, max_given = difficulty_levels.get(difficulty, (25, 30))
        cells_to_remove = 81 - random.randint(min_given, max_given)
        
        # Remove cells randomly while ensuring the puzzle has a unique solution
        removed = 0
        positions = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(positions)
        
        for row, col in positions:
            if removed >= cells_to_remove:
                break
                
            # Temporarily remove the number
            backup = puzzle[row][col]
            puzzle[row][col] = 0
            
            # Check if the puzzle still has a unique solution
            problem = SudokuProblem(puzzle)
            csp = problem.create_csp()
            solution = csp.backtracking_search(use_mrv=True, use_lcv=True, use_inference=True)
            
            if solution is not None:
                # Convert back to check uniqueness (simplified check)
                translated = problem.translate_solution(solution)
                if translated:
                    removed += 1
                    continue
            
            # If removing causes issues, put the number back
            puzzle[row][col] = backup
        
        return puzzle, solved

class SudokuGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sudoku Solver")
        self.root.geometry("550x650")
        
        self.cells = []
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Sudoku Solver", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Sudoku grid frame
        grid_frame = ttk.Frame(main_frame)
        grid_frame.pack(pady=10)
        
        # Create 9x9 grid
        self.cells = []
        for row in range(9):
            row_cells = []
            for col in range(9):
                # Create entry with validation
                entry = tk.Entry(
                    grid_frame, 
                    width=2, 
                    font=('Arial', 18), 
                    justify='center',
                    validate='key'
                )
                entry.config(validatecommand=(entry.register(self.validate_input), '%P'))
                entry.grid(row=row, column=col, padx=1, pady=1, ipady=5)
                
                # Add thicker borders for 3x3 boxes
                padx = (3, 1) if col % 3 == 0 and col != 0 else (1, 1)
                pady = (3, 1) if row % 3 == 0 and row != 0 else (1, 1)
                entry.grid(padx=padx, pady=pady)
                
                row_cells.append(entry)
            self.cells.append(row_cells)
        
        # Difficulty selection
        difficulty_frame = ttk.Frame(main_frame)
        difficulty_frame.pack(pady=5)
        
        ttk.Label(difficulty_frame, text="Difficulty:").pack(side=tk.LEFT, padx=5)
        self.difficulty_var = tk.StringVar(value="medium")
        difficulties = ttk.Combobox(difficulty_frame, textvariable=self.difficulty_var, 
                                   values=["easy", "medium", "hard", "expert"], state="readonly")
        difficulties.pack(side=tk.LEFT, padx=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Buttons
        ttk.Button(button_frame, text="Solve", command=self.solve).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Example", command=self.load_example).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generate Random", command=self.generate_random).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Enter a Sudoku puzzle and click Solve")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=10)
    
    def validate_input(self, value):
        """Validate that input is a digit between 1-9 or empty"""
        if value == "":
            return True
        if len(value) > 1:
            return False
        return value.isdigit() and 1 <= int(value) <= 9
    
    def get_board_from_gui(self):
        """Get current board state from GUI"""
        board = []
        for row in range(9):
            board_row = []
            for col in range(9):
                value = self.cells[row][col].get()
                if value == "":
                    board_row.append(0)
                else:
                    board_row.append(int(value))
            board.append(board_row)
        return board
    
    def update_gui_from_board(self, board):
        """Update GUI with board values"""
        for row in range(9):
            for col in range(9):
                self.cells[row][col].delete(0, tk.END)
                if board[row][col] != 0:
                    self.cells[row][col].insert(0, str(board[row][col]))
    
    def solve(self):
        """Solve the current Sudoku puzzle"""
        try:
            self.status_var.set("Solving...")
            self.root.update_idletasks()
            
            # Get current board state
            board = self.get_board_from_gui()
            
            # Create and solve Sudoku problem
            problem = SudokuProblem(board)
            csp = problem.create_csp()
            
            # Solve with heuristics
            int_solution = csp.backtracking_search(
                use_mrv=True, 
                use_lcv=True, 
                use_inference=True
            )
            
            solution = problem.translate_solution(int_solution)
            
            if solution:
                self.update_gui_from_board(solution)
                self.status_var.set("Solved!")
            else:
                messagebox.showerror("No Solution", "No solution found for this Sudoku puzzle.")
                self.status_var.set("No solution found")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")
    
    def clear(self):
        """Clear the Sudoku grid"""
        for row in range(9):
            for col in range(9):
                self.cells[row][col].delete(0, tk.END)
        self.status_var.set("Grid cleared")
    
    def load_example(self):
        """Load an example Sudoku puzzle"""
        example = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        
        self.update_gui_from_board(example)
        self.status_var.set("Example loaded")
    
    def generate_random(self):
        """Generate a random Sudoku puzzle"""
        try:
            self.status_var.set("Generating random puzzle...")
            self.root.update_idletasks()
            
            difficulty = self.difficulty_var.get()
            puzzle, solved = SudokuGenerator.generate_puzzle(difficulty)
            
            self.update_gui_from_board(puzzle)
            self.status_var.set(f"Random {difficulty} puzzle generated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate puzzle: {str(e)}")
            self.status_var.set("Error generating puzzle")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    gui = SudokuGUI()
    gui.run()

if __name__ == "__main__":
    main()