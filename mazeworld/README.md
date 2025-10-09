markdown

# Maze World Robot Navigation

## How to Run:
1. Ensure all Python and maze files are in same directory
2. Run: `python test_mazeworld.py` for multi-robot tests
3. Run: `python test_sensorless.py` for blind robot tests

NB: Make sure to update the file path as required on the two test scripts.

## Files:
- `MazeworldProblem.py` - Multi-robot coordination
- `SensorlessProblem.py` - Blind robot problem  
- `astar_search.py` - A* algorithm
- `uninformed_search.py` - BFS/DFS/IDS
- `SearchSolution.py` - Solution data structure
- `Maze.py` - Maze loading/display