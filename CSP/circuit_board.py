from csp import CSP, BinaryConstraint
from typing import List, Tuple, Dict, Optional

class Component:
    """Represents a circuit board component"""
    
    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self.width = width
        self.height = height
    
    def __repr__(self):
        return f"{self.name}({self.width}x{self.height})"

class CircuitBoardProblem:
    """Circuit Board Layout CSP Problem that wraps integer-based CSP with human-readable interface"""
    
    def __init__(self, board_width: int, board_height: int, components: List[Component]):
        self.board_width = board_width
        self.board_height = board_height
        self.components = components
        
        # Create mappings between human-readable and integer representations
        self.comp_to_index = {comp.name: i for i, comp in enumerate(components)}
        self.index_to_comp = {i: comp for i, comp in enumerate(components)}
        
        # Precompute all possible positions and create mapping
        self.positions = []
        self.pos_to_coords = {}
        self.coords_to_pos = {}
        
        pos_idx = 0
        for comp in components:
            for x in range(board_width - comp.width + 1):
                for y in range(board_height - comp.height + 1):
                    self.positions.append((comp.name, x, y))
                    self.pos_to_coords[pos_idx] = (comp.name, x, y)
                    self.coords_to_pos[(comp.name, x, y)] = pos_idx
                    pos_idx += 1
    
    def create_csp(self) -> CSP:
        num_variables = len(self.components)
        
        # Build domains: for each component, list of position indices where it fits
        domains = []
        for comp in self.components:
            comp_positions = []
            for pos_idx, (comp_name, x, y) in enumerate(self.positions):
                if comp_name == comp.name:
                    comp_positions.append(pos_idx)
            domains.append(comp_positions)
        
        csp = CSP(num_variables, domains)
        
        # Add non-overlap constraints between all pairs of components
        for i in range(len(self.components)):
            for j in range(i + 1, len(self.components)):
                comp1 = self.components[i]
                comp2 = self.components[j]
                
                def non_overlap_constraint(pos_idx1, pos_idx2, comp1=comp1, comp2=comp2):
                    # Get coordinates from position indices
                    _, x1, y1 = self.pos_to_coords[pos_idx1]
                    _, x2, y2 = self.pos_to_coords[pos_idx2]
                    
                    # Check if rectangles overlap
                    return not (x1 < x2 + comp2.width and 
                               x1 + comp1.width > x2 and 
                               y1 < y2 + comp2.height and 
                               y1 + comp1.height > y2)
                
                csp.add_constraint(BinaryConstraint(i, j, non_overlap_constraint))
        
        return csp
    
    def translate_solution(self, int_solution: Optional[List[int]]) -> Optional[Dict[str, Tuple[int, int]]]:
        """Translate integer-based solution back to human-readable format"""
        if int_solution is None:
            return None
        
        solution = {}
        for comp_idx, pos_idx in enumerate(int_solution):
            if pos_idx is not None:
                comp_name, x, y = self.pos_to_coords[pos_idx]
                solution[comp_name] = (x, y)
        
        return solution
    
    def display_solution(self, int_solution: Optional[List[int]]):
        """Display the circuit board layout as ASCII art"""
        solution = self.translate_solution(int_solution)
        if solution is None:
            print("No solution found")
            return
        
        # Create empty board
        board = [['.' for _ in range(self.board_width)] for _ in range(self.board_height)]
        
        # Place components on board
        for comp in self.components:
            if comp.name in solution:
                x, y = solution[comp.name]
                # Use first character of component name
                char = comp.name[0].upper()
                for i in range(x, x + comp.width):
                    for j in range(y, y + comp.height):
                        board[j][i] = char
        
        # Print board (flip y-axis for proper display)
        for row in reversed(board):
            print(''.join(row))

# Example circuit board problem from the assignment
def example_circuit_board():
    board_width = 10
    board_height = 3
    components = [
        Component('a', 3, 2),  # component a: 3x2
        Component('b', 5, 2),  # component b: 5x2  
        Component('c', 2, 3),  # component c: 2x3
        Component('e', 7, 1),  # component e: 7x1
    ]
    
    return CircuitBoardProblem(board_width, board_height, components)
