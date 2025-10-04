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