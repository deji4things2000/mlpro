# You write this:
from SensorlessProblem import SensorlessProblem
from Maze import Maze
from astar_search import astar_search

def test_sensorless():
    # Test with maze3
    test_maze3 = Maze("/Users/user_1/mlpro/mazeworld/maze3.maz")
    test_problem = SensorlessProblem(test_maze3)
    
    print("Testing sensorless problem on maze3:")
    print(test_problem)
    print(f"Initial belief state size: {len(test_problem.start_state)}")
    
    # Test with null heuristic first
    print("\n--- Testing with null heuristic ---")
    result = astar_search(test_problem, test_problem.manhattan_heuristic)
    print(result)
    
    if result.path:
        print(f"Path length: {len(result.path)}")
        print("Final belief state:", result.path[-1])

if __name__ == "__main__":
    test_sensorless()

    # Additional comprehensive tests
    print("\n" + "="*60)
    print("ADDITIONAL COMPREHENSIVE SENSORLESS TESTS")
    print("="*60)
    
    # Test 1: Heuristic Comparison
    print("\n🧪 HEURISTIC COMPARISON ON MAZE3")
    test_maze3 = Maze("/Users/user_1/mlpro/mazeworld/maze3.maz")
    problem = SensorlessProblem(test_maze3)
    
    heuristics = [
        ("Basic", problem.basic_heuristic),
        ("Manhattan", problem.manhattan_heuristic),
        ("Advanced", problem.advanced_heuristic)
    ]
    
    for name, heuristic in heuristics:
        result = astar_search(problem, heuristic)
        print(f"{name:10} - Nodes: {result.nodes_visited:6d}, Path: {len(result.path):3d} steps")
    
    # Test 2: Small maze demonstration
    print("\n🧪 SMALL MAZE DEMONSTRATION")
    with open("blind_demo.maz", 'w') as f:
        f.write("#####\n")
        f.write("#...#\n")
        f.write("#.#.#\n")
        f.write("#...#\n")
        f.write("#####\n")
    
    small_maze = Maze("blind_demo.maz")
    small_problem = SensorlessProblem(small_maze)
    print(f"Initial belief: {len(small_problem.start_state)} positions")
    result = astar_search(small_problem, small_problem.manhattan_heuristic)
    if result.path:
        print(f"Solution: {len(result.path)} steps to reduce to 1 position")
        print("Animating solution...")
        small_problem.animate_path(result.path)
    
    # Test 3: Performance on different mazes
    print("\n🧪 PERFORMANCE ACROSS MAZES")
    mazes_to_test = ["maze1.maz", "maze2.maz", "maze3.maz"]
    for maze_file in mazes_to_test:
        try:
            maze = Maze(f"/Users/user_1/mlpro/mazeworld/{maze_file}")
            problem = SensorlessProblem(maze)
            result = astar_search(problem, problem.manhattan_heuristic)
            if result.path:
                reduction = len(problem.start_state)
                print(f"{maze_file:15} | {reduction:3d} → 1 pos | {len(result.path):3d} steps | {result.nodes_visited:6d} nodes")
        except:
            print(f"{maze_file:15} | File not found or error")

