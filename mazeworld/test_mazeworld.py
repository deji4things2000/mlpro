from MazeworldProblem import MazeworldProblem
from Maze import Maze

from uninformed_search import bfs_search
from astar_search import astar_search

import random

# null heuristic, useful for testing astar search without heuristic (uniform cost search).
def null_heuristic(state):
    return 0

# Test problems

test_maze3 = Maze("/Users/user_1/mlpro/mazeworld/maze3.maz")
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
test_maze1 = Maze("/Users/user_1/mlpro/mazeworld/maze1.maz")
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
    test_maze2 = Maze("/Users/user_1/mlpro/mazeworld/maze2.maz")
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

# NEW CUSTOM MAZE TESTS (maze4 - maze8)
print("\n--- Testing maze4: Corridor Order Reversal ---")
try:
    test_maze4 = Maze("/Users/user_1/mlpro/mazeworld/maze4.maz")
    print("maze4 (Corridor):")
    print(test_maze4)
    test_mp4 = MazeworldProblem(test_maze4, (5, 1, 3, 1, 1, 1))
    result4 = astar_search(test_mp4, test_mp4.manhattan_heuristic)
    print(result4)
except Exception as e:
    print(f"Error testing maze4: {e}")

print("\n--- Testing maze5: Bottleneck Passing ---")
try:
    test_maze5 = Maze("/Users/user_1/mlpro/mazeworld/maze5.maz")
    print("maze5 (Bottleneck):")
    print(test_maze5)
    test_mp5 = MazeworldProblem(test_maze5, (5, 2, 1, 2))
    result5 = astar_search(test_mp5, test_mp5.manhattan_heuristic)
    print(result5)
except Exception as e:
    print(f"Error testing maze5: {e}")

print("\n--- Testing maze6: Dead End Coordination ---")
try:
    test_maze6 = Maze("/Users/user_1/mlpro/mazeworld/maze6.maz")
    print("maze6 (Dead Ends):")
    print(test_maze6)
    test_mp6 = MazeworldProblem(test_maze6, (5, 1, 1, 1))
    result6 = astar_search(test_mp6, test_mp6.manhattan_heuristic)
    print(result6)
except Exception as e:
    print(f"Error testing maze6: {e}")

print("\n--- Testing maze7: Crossroads Challenge ---")
try:
    test_maze7 = Maze("/Users/user_1/mlpro/mazeworld/maze7.maz")
    print("maze7 (Crossroads):")
    print(test_maze7)
    test_mp7 = MazeworldProblem(test_maze7, (5, 1, 1, 1, 3, 5))
    result7 = astar_search(test_mp7, test_mp7.manhattan_heuristic)
    print(result7)
except Exception as e:
    print(f"Error testing maze7: {e}")

print("\n--- Testing maze8: Spiral Maze ---")
try:
    test_maze8 = Maze("/Users/user_1/mlpro/mazeworld/maze8.maz")
    print("maze8 (Spiral):")
    print(test_maze8)
    test_mp8 = MazeworldProblem(test_maze8, (5, 1, 1, 5, 3, 1))
    result8 = astar_search(test_mp8, test_mp8.manhattan_heuristic)
    print(result8)
except Exception as e:
    print(f"Error testing maze8: {e}")


print("\n" + "="*50)
print("ADDITIONAL TEST: 40x40 RANDOM MAZE")
print("="*50)

# Generate 40x40 maze with 3 robots
print("Generating 40x40 random maze with 3 robots...")

# Create random maze with guaranteed solvability
maze_grid = [['.' for _ in range(40)] for _ in range(40)]
for y in range(40):
    for x in range(40):
        if random.random() < 0.2 and (x > 1 or y > 1):
            maze_grid[y][x] = '#'

# Ensure a clear horizontal path along the bottom
for x in range(40):
    maze_grid[0][x] = '.'  # Clear bottom row
    maze_grid[1][x] = '.'  # Clear second row for flexibility

# Add 3 robots along bottom left
robot_positions = [
    (2, 1),    # Robot A 
    (5, 1),    # Robot B
    (8, 1)     # Robot C
]

# Set goals along bottom right
goal_locations = (35, 1, 32, 1, 29, 1)  # All move to bottom right area

# Write maze file
with open("random_40x40.maz", 'w') as f:
    for y in range(39, -1, -1):
        f.write(''.join(maze_grid[y]) + '\n')
    for x, y in robot_positions:
        f.write(f"\\robot {x} {y}\n")

# Test the generated maze
maze = Maze("random_40x40.maz")

print("Robot start positions (bottom left):")
for i, (x, y) in enumerate(robot_positions):
    print(f"  Robot {i}: ({x}, {y})")

print("Robot goal positions (bottom right):")
goals_list = [goal_locations[i:i+2] for i in range(0, len(goal_locations), 2)]
for i, (x, y) in enumerate(goals_list):
    print(f"  Robot {i} goal: ({x}, {y})")

problem = MazeworldProblem(maze, goal_locations)
result = astar_search(problem, problem.manhattan_heuristic)

print(f"\nA* Search Results:")
print(f"Nodes visited: {result.nodes_visited}")
print(f"Solution found: {len(result.path) > 0}")
print(f"Path length: {len(result.path) if result.path else 0} steps")
print(f"Total cost: {result.cost}")

# Show the animation
if result.path:
    print("\nAnimating 40x40 maze solution...")
    print("Robots moving from bottom-left to bottom-right:")
    for i in range(3):
        start = robot_positions[i]
        goal = goals_list[i]
        print(f"  Robot {i}: {start} → {goal}")
    problem.animate_path(result.path)
else:
    print("No solution found - trying closer goals...")
    # Fallback: try moving robots to middle bottom
    simple_goals = (20, 1, 23, 1, 26, 1)
    print(f"Trying closer goals: {[simple_goals[i:i+2] for i in range(0, len(simple_goals), 2)]}")
    problem2 = MazeworldProblem(maze, simple_goals)
    result2 = astar_search(problem2, problem2.manhattan_heuristic)
    if result2.path:
        print("Found solution with closer goals!")
        problem2.animate_path(result2.path)


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