#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran
# Date: 11/15/2025 


import random
import time
from typing import List, Set, Tuple, Optional

class SATSolver:
    def __init__(self):
        self.variables = set()
        self.clauses = []
        self.assignment = {}
        self.num_vars = 0
        self.num_clauses = 0
        
    def load_cnf(self, filename: str):
        """Load CNF file in DIMACS format"""
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.startswith('c') or not line:
                continue
            elif line.startswith('p'):
                # Parse problem line: p cnf variables clauses
                parts = line.split()
                self.num_vars = int(parts[2])
                self.num_clauses = int(parts[3])
                self.variables = set(range(1, self.num_vars + 1))
            else:
                # Parse clause
                literals = list(map(int, line.split()))
                if literals[-1] == 0:
                    literals = literals[:-1]
                self.clauses.append(literals)
    
    def random_assignment(self) -> dict:
        """Generate random truth assignment"""
        return {var: random.choice([True, False]) for var in self.variables}
    
    def evaluate_clause(self, clause: List[int], assignment: dict) -> bool:
        """Evaluate a single clause under given assignment"""
        for literal in clause:
            var = abs(literal)
            value = assignment.get(var, False)
            if literal > 0 and value:
                return True
            elif literal < 0 and not value:
                return True
        return False
    
    def satisfied_clauses(self, assignment: dict) -> int:
        """Count number of satisfied clauses"""
        return sum(1 for clause in self.clauses if self.evaluate_clause(clause, assignment))
    
    def is_solution(self, assignment: dict) -> bool:
        """Check if assignment satisfies all clauses"""
        return all(self.evaluate_clause(clause, assignment) for clause in self.clauses)
    
    def get_unsatisfied_clauses(self, assignment: dict) -> List[List[int]]:
        """Get all unsatisfied clauses"""
        return [clause for clause in self.clauses if not self.evaluate_clause(clause, assignment)]

class GSAT(SATSolver):
    def solve(self, max_iterations: int = 10000, threshold: float = 0.3) -> Optional[dict]:
        """
        GSAT algorithm implementation
        
        Args:
            max_iterations: Maximum number of iterations to try
            threshold: Probability threshold for random flip
            
        Returns:
            Solution assignment or None if no solution found
        """
        self.assignment = self.random_assignment()
        
        for iteration in range(max_iterations):
            if self.is_solution(self.assignment):
                return self.assignment
            
            # Random flip with probability threshold
            if random.random() > threshold:
                var_to_flip = random.choice(list(self.variables))
                self.assignment[var_to_flip] = not self.assignment[var_to_flip]
                continue
            
            # Score each variable
            best_score = -1
            best_vars = []
            
            for var in self.variables:
                # Flip variable temporarily and count satisfied clauses
                self.assignment[var] = not self.assignment[var]
                score = self.satisfied_clauses(self.assignment)
                self.assignment[var] = not self.assignment[var]  # Flip back
                
                if score > best_score:
                    best_score = score
                    best_vars = [var]
                elif score == best_score:
                    best_vars.append(var)
            
            # Randomly choose one of the best variables
            if best_vars:
                var_to_flip = random.choice(best_vars)
                self.assignment[var_to_flip] = not self.assignment[var_to_flip]
        
        return None

class WalkSAT(SATSolver):
    def solve(self, max_iterations: int = 100000, threshold: float = 0.7) -> Optional[dict]:
        """
        WalkSAT algorithm implementation
        
        Args:
            max_iterations: Maximum number of iterations to try
            threshold: Probability threshold for random flip
            
        Returns:
            Solution assignment or None if no solution found
        """
        self.assignment = self.random_assignment()
        
        for iteration in range(max_iterations):
            if self.is_solution(self.assignment):
                return self.assignment
            
            # Get unsatisfied clauses
            unsatisfied = self.get_unsatisfied_clauses(self.assignment)
            if not unsatisfied:
                return self.assignment
            
            # Randomly choose an unsatisfied clause
            clause = random.choice(unsatisfied)
            
            # Random flip with probability threshold
            if random.random() > threshold:
                # Flip a random variable from the unsatisfied clause
                literal = random.choice(clause)
                var_to_flip = abs(literal)
                self.assignment[var_to_flip] = not self.assignment[var_to_flip]
                continue
            
            # Score only variables in the unsatisfied clause
            best_score = -1
            best_vars = []
            
            for literal in clause:
                var = abs(literal)
                # Flip variable temporarily and count satisfied clauses
                self.assignment[var] = not self.assignment[var]
                score = self.satisfied_clauses(self.assignment)
                self.assignment[var] = not self.assignment[var]  # Flip back
                
                if score > best_score:
                    best_score = score
                    best_vars = [var]
                elif score == best_score:
                    best_vars.append(var)
            
            # Randomly choose one of the best variables
            if best_vars:
                var_to_flip = random.choice(best_vars)
                self.assignment[var_to_flip] = not self.assignment[var_to_flip]
        
        return None

def solve_sat(filename: str, algorithm: str = 'walksat', **kwargs) -> Optional[dict]:
    """
    Convenience function to solve SAT problems
    
    Args:
        filename: Path to CNF file
        algorithm: 'gsat' or 'walksat'
        **kwargs: Additional parameters for the solver
        
    Returns:
        Solution assignment or None
    """
    if algorithm.lower() == 'gsat':
        solver = GSAT()
    else:
        solver = WalkSAT()
    
    solver.load_cnf(filename)
    return solver.solve(**kwargs)

def save_solution(assignment: dict, filename: str):
    """Save solution to file in .sol format"""
    with open(filename, 'w') as f:
        for var, value in sorted(assignment.items()):
            f.write(f"{var} {'T' if value else 'F'}\n")

def load_solution(filename: str) -> dict:
    """Load solution from .sol file"""
    assignment = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                var, value = line.split()
                assignment[int(var)] = (value.upper() == 'T')
    return assignment