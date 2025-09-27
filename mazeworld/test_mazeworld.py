from MazeworldProblem import MazeworldProblem
from Maze import Maze

from uninformed_search import bfs_search
from astar_search import astar_search

# null heuristic, useful for testing astar search without heuristic (uniform cost search).
def null_heuristic(state):
    return 0

# Test problems

test_maze3 = Maze("maze3.maz")
test_mp = MazeworldProblem(test_maze3, (1, 4, 1, 3, 1, 2))

print(test_mp.get_successors(test_mp.start_state))

# this should explore a lot of nodes; it's just uniform-cost search
result = astar_search(test_mp, null_heuristic)
print(result)

# this should do a bit better:
result = astar_search(test_mp, test_mp.manhattan_heuristic)
print(result)
test_mp.animate_path(result.path)

# Your additional tests here:
print("\n" + "="*50)
print("ADDITIONAL TESTS")
print("="*50)

# Test with maze1 (simpler maze)
print("\n--- Testing with maze1 ---")
test_maze1 = Maze("maze1.maz")
print("maze1 robot locations:", test_maze1.robotloc)

# Create a simple goal for maze1 (just move robots slightly)
if len(test_maze1.robotloc) >= 2:  # At least one robot
    goal_x = test_maze1.robotloc[0] + 1
    goal_y = test_maze1.robotloc[1]
    # Make sure goal is valid
    if not test_maze1.is_floor(goal_x, goal_y):
        goal_x = test_maze1.robotloc[0] - 1  # Try other direction
    
    if test_maze1.is_floor(goal_x, goal_y):
        test_mp1 = MazeworldProblem(test_maze1, (goal_x, goal_y))
        print(f"Testing single robot move from {test_maze1.robotloc[0:2]} to ({goal_x}, {goal_y})")
        result1 = astar_search(test_mp1, test_mp1.manhattan_heuristic)
        print(result1)

# Test with maze2
print("\n--- Testing with maze2 ---")
try:
    test_maze2 = Maze("maze2.maz")
    print("maze2 robot locations:", test_maze2.robotloc)
    
    # Create a reasonable goal
    if len(test_maze2.robotloc) >= 4:  # At least two robots
        goal_locations = (
            test_maze2.robotloc[0] + 2, test_maze2.robotloc[1],  # Robot 0 moves right 2
            test_maze2.robotloc[2], test_maze2.robotloc[3] - 1   # Robot 1 moves down 1
        )
        test_mp2 = MazeworldProblem(test_maze2, goal_locations)
        result2 = astar_search(test_mp2, test_mp2.manhattan_heuristic)
        print(result2)
except Exception as e:
    print(f"Error testing maze2: {e}")

# Edge case tests
print("\n--- Edge Case Tests ---")

# Test collision avoidance
print("Testing collision avoidance...")
# Create a state where robots are adjacent and test valid moves
collision_test_state = (0, 1, 1, 1, 2, 1, 3)  # Three robots in a vertical line
successors = test_mp.get_successors(collision_test_state)
print(f"Successors for collision-prone state {collision_test_state}:")
for i, (state, cost, action) in enumerate(successors[:3]):  # Show first 3
    print(f"  {i+1}. {action} -> State: {state}")

# Test boundary conditions
print("\nTesting boundary/wall collisions...")
edge_state = (0, 0, 0, 0, 1, 0, 2)  # Robots near edges
successors = test_mp.get_successors(edge_state)
print(f"Successors for edge state {edge_state}:")
for i, (state, cost, action) in enumerate(successors[:3]):
    print(f"  {i+1}. {action} -> State: {state}")

print("\nAll tests completed!")