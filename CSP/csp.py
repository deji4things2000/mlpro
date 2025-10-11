import heapq
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod

class Constraint(ABC):
    """Abstract base class for constraints using integer variables and values"""
    
    @abstractmethod
    def satisfied(self, assignment: List[Optional[int]]) -> bool:
        pass
    
    @abstractmethod
    def get_variables(self) -> List[int]:
        pass

class BinaryConstraint(Constraint):
    """Binary constraint between two integer variables"""
    
    def __init__(self, var1: int, var2: int, constraint_func: Callable):
        self.var1 = var1
        self.var2 = var2
        self.constraint_func = constraint_func
    
    def satisfied(self, assignment: List[Optional[int]]) -> bool:
        if assignment[self.var1] is None or assignment[self.var2] is None:
            return True
        return self.constraint_func(assignment[self.var1], assignment[self.var2])
    
    def get_variables(self) -> List[int]:
        return [self.var1, self.var2]

class CSP:
    """Constraint Satisfaction Problem Solver using integer variables and values"""
    
    def __init__(self, num_variables: int, domains: List[List[int]]):
        self.num_variables = num_variables
        self.domains = domains  # List of lists, where domains[i] are the values for variable i
        self.constraints: List[List[Constraint]] = [[] for _ in range(num_variables)]
    
    def add_constraint(self, constraint: Constraint):
        for variable in constraint.get_variables():
            if variable < self.num_variables:
                self.constraints[variable].append(constraint)
    
    def consistent(self, variable: int, assignment: List[Optional[int]]) -> bool:
        """Check if current assignment is consistent for given variable"""
        for constraint in self.constraints[variable]:
            if not constraint.satisfied(assignment):
                return False
        return True
    
    def backtracking_search(self, assignment: List[Optional[int]] = None, 
                          use_mrv: bool = False,
                          use_degree: bool = False,
                          use_lcv: bool = False,
                          use_inference: bool = False) -> Optional[List[int]]:
        """Main backtracking search algorithm"""
        
        if assignment is None:
            assignment = [None] * self.num_variables
        
        # If assignment is complete, return it
        if all(value is not None for value in assignment):
            return assignment
        
        # Select unassigned variable
        unassigned = [v for v in range(self.num_variables) if assignment[v] is None]
        
        if use_mrv:
            variable = self._mrv_heuristic(unassigned, assignment)
        elif use_degree:
            variable = self._degree_heuristic(unassigned)
        else:
            variable = unassigned[0]
        
        # Order domain values
        if use_lcv:
            domain_values = self._lcv_heuristic(variable, assignment)
        else:
            domain_values = self.domains[variable].copy()
        
        for value in domain_values:
            local_assignment = assignment.copy()
            local_assignment[variable] = value
            
            if self.consistent(variable, local_assignment):
                # Make inference if enabled
                if use_inference:
                    domains_before = [domain.copy() for domain in self.domains]
                    
                    if self._ac3():
                        result = self.backtracking_search(local_assignment, use_mrv, use_degree, use_lcv, use_inference)
                        if result is not None:
                            return result
                    
                    # Restore domains if inference didn't lead to solution
                    self.domains = domains_before
                else:
                    result = self.backtracking_search(local_assignment, use_mrv, use_degree, use_lcv, use_inference)
                    if result is not None:
                        return result
        
        return None
    
    def _mrv_heuristic(self, unassigned: List[int], assignment: List[Optional[int]]) -> int:
        """Minimum Remaining Values heuristic"""
        min_remaining = float('inf')
        best_variable = None
        
        for variable in unassigned:
            remaining = len(self.domains[variable])
            if remaining < min_remaining:
                min_remaining = remaining
                best_variable = variable
        
        return best_variable
    
    def _degree_heuristic(self, unassigned: List[int]) -> int:
        """Degree heuristic - select variable with most constraints"""
        max_degree = -1
        best_variable = None
        
        for variable in unassigned:
            degree = len(self.constraints[variable])
            if degree > max_degree:
                max_degree = degree
                best_variable = variable
        
        return best_variable
    
    def _lcv_heuristic(self, variable: int, assignment: List[Optional[int]]) -> List[int]:
        """Least Constraining Value heuristic"""
        value_scores = []
        
        for value in self.domains[variable]:
            score = 0
            test_assignment = assignment.copy()
            test_assignment[variable] = value
            
            # Count how many choices this leaves for neighbors
            for constraint in self.constraints[variable]:
                for neighbor in constraint.get_variables():
                    if neighbor != variable and test_assignment[neighbor] is None:
                        for neighbor_value in self.domains[neighbor]:
                            test_assignment[neighbor] = neighbor_value
                            if constraint.satisfied(test_assignment):
                                score += 1
                            test_assignment[neighbor] = None
            
            value_scores.append((value, score))
        
        # Sort by score descending (higher score = less constraining)
        value_scores.sort(key=lambda x: x[1], reverse=True)
        return [value for value, score in value_scores]
    
    def _ac3(self) -> bool:
        """AC-3 algorithm for constraint propagation"""
        queue = []
        
        # Initialize queue with all arcs
        for variable in range(self.num_variables):
            for constraint in self.constraints[variable]:
                for other_var in constraint.get_variables():
                    if other_var != variable:
                        queue.append((variable, other_var))
        
        while queue:
            xi, xj = queue.pop(0)
            
            if self._revise(xi, xj):
                if len(self.domains[xi]) == 0:
                    return False
                
                # Add neighbors to queue
                for constraint in self.constraints[xi]:
                    for xk in constraint.get_variables():
                        if xk != xi and xk != xj:
                            queue.append((xk, xi))
        
        return True
    
    def _revise(self, xi: int, xj: int) -> bool:
        """Revise domain of xi based on constraint with xj"""
        revised = False
        
        for x in self.domains[xi][:]:
            # Check if there exists a value in xj's domain that satisfies all constraints
            found = False
            for y in self.domains[xj]:
                test_assignment = [None] * self.num_variables
                test_assignment[xi] = x
                test_assignment[xj] = y
                consistent = True
                
                for constraint in self.constraints[xi]:
                    if xj in constraint.get_variables():
                        if not constraint.satisfied(test_assignment):
                            consistent = False
                            break
                
                if consistent:
                    found = True
                    break
            
            if not found:
                self.domains[xi].remove(x)
                revised = True
        
        return revised