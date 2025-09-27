from Maze import Maze
from time import sleep

class SensorlessProblem:

    ## You write the good stuff here:
    def __init__(self, maze):
        self.maze = maze
        # Start with all possible positions
        all_positions = []
        for x in range(maze.width):
            for y in range(maze.height):
                if maze.is_floor(x, y):
                    all_positions.append((x, y))
        
        self.start_state = tuple(all_positions)
        self.width = maze.width
        self.height = maze.height

    def get_successors(self, state):
        successors = []
        moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dx, dy in moves:
            new_belief_state = set()
            
            for x, y in state:
                new_x, new_y = x + dx, y + dy
                
                if self.maze.is_floor(new_x, new_y):
                    new_belief_state.add((new_x, new_y))
                else:
                    # If move is invalid, robot stays in current position
                    new_belief_state.add((x, y))
            
            new_state = tuple(sorted(new_belief_state))
            cost = 1
            successors.append((new_state, cost, f"Move ({dx}, {dy})"))
        
        return successors

    def is_goal_state(self, state):
        # Goal is when we know exactly where the robot is (only one possible position)
        return len(state) == 1

    def manhattan_heuristic(self, state):
        # Heuristic: minimum number of moves needed to reduce to one position
        # A simple heuristic is the number of positions minus 1
        return len(state) - 1

    def __str__(self):
        string =  "Blind robot problem: "
        return string


        # given a sequence of states (including robot turn), modify the maze and print it out.
        #  (Be careful, this does modify the maze!)

    def animate_path(self, path):
        # reset the robot locations in the maze
        self.maze.robotloc = tuple(self.start_state)

        for state in path:
            print(str(self))
            self.maze.robotloc = tuple(state)
            sleep(1)

            print(str(self.maze))


## A bit of test code

if __name__ == "__main__":
    test_maze3 = Maze("maze3.maz")
    test_problem = SensorlessProblem(test_maze3)
