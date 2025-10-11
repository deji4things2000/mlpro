import time
from csp import CSP
from map_coloring import australia_map_coloring, usa_map_coloring
from circuit_board import example_circuit_board

def test_map_coloring():
    """Test map coloring with different configurations"""
    print("=== Map Coloring Tests ===")
    
    problem = australia_map_coloring()
    csp = problem.create_csp()
    
    configurations = [
        ("No heuristics", False, False, False, False),
        ("MRV only", True, False, False, False),
        ("MRV + LCV", True, False, True, False),
        ("Degree heuristic", False, True, False, False),
        ("All heuristics", True, True, True, False),
        ("All + AC-3", True, True, True, True),
    ]
    
    for name, mrv, degree, lcv, inference in configurations:
        start_time = time.time()
        int_solution = csp.backtracking_search(use_mrv=mrv, use_degree=degree, 
                                             use_lcv=lcv, use_inference=inference)
        solution = problem.translate_solution(int_solution)
        end_time = time.time()
        
        print(f"\n{name}:")
        print(f"Time: {end_time - start_time:.4f}s")
        print(f"Solution: {solution}")

def test_circuit_board():
    """Test circuit board layout"""
    print("\n=== Circuit Board Layout Tests ===")
    
    problem = example_circuit_board()
    csp = problem.create_csp()
    
    print("Circuit board components:")
    for comp in problem.components:
        print(f"  {comp}")
    
    print(f"\nBoard size: {problem.board_width}x{problem.board_height}")
    
    int_solution = csp.backtracking_search(use_mrv=True, use_lcv=True, use_inference=True)
    
    if int_solution is not None:
        print("\nSolution found!")
        problem.display_solution(int_solution)
    else:
        print("\nNo solution found!")

def performance_comparison():
    """Compare performance of different heuristic combinations"""
    print("\n=== Performance Comparison ===")
    
    # Test with larger map coloring problem
    problem = usa_map_coloring()
    csp = problem.create_csp()
    
    heuristic_combinations = [
        ("Baseline", {}),
        ("MRV", {"use_mrv": True}),
        ("MRV+LCV", {"use_mrv": True, "use_lcv": True}),
        ("All heuristics", {"use_mrv": True, "use_degree": True, "use_lcv": True}),
        ("All + AC-3", {"use_mrv": True, "use_degree": True, "use_lcv": True, "use_inference": True}),
    ]
    
    for name, kwargs in heuristic_combinations:
        start_time = time.time()
        int_solution = csp.backtracking_search(**kwargs)
        end_time = time.time()
        
        print(f"{name}: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    test_map_coloring()
    test_circuit_board()
    performance_comparison()