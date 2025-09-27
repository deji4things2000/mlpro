from Maze import Maze
from time import sleep

class MazeworldProblem:

    ## you write the constructor, and whatever methods your astar function needs

    def __init__(self, maze, goal_locations):
        self.maze = maze
        self.goal_locations = goal_locations
        self.start_state = (0,) + tuple(maze.robotloc)  # (current_robot_turn, x1, y1, x2, y2, ...)
        self.num_robots = len(maze.robotloc) // 2

    def get_successors(self, state):
        successors = []
        current_robot = state[0] % self.num_robots
        robot_locations = state[1:]
        
        # Possible moves: wait (stay in place) or move in four directions
        moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dx, dy in moves:
            new_locations = list(robot_locations)
            
            # Calculate new position for the current robot
            current_idx = current_robot * 2
            new_x = robot_locations[current_idx] + dx
            new_y = robot_locations[current_idx + 1] + dy
            
            # Check if the move is valid
            if self.is_valid_move(new_x, new_y, robot_locations, current_robot):
                new_locations[current_idx] = new_x
                new_locations[current_idx + 1] = new_y
                
                # Next robot's turn
                next_robot = (current_robot + 1) % self.num_robots
                new_state = (next_robot,) + tuple(new_locations)
                cost = 1  # Each move costs 1
                successors.append((new_state, cost, f"Robot {current_robot} moves to ({new_x}, {new_y})"))
        
        return successors

    def is_valid_move(self, x, y, robot_locations, moving_robot):
        # Check if the position is within bounds and is a floor
        if not self.maze.is_floor(x, y):
            return False
        
        # Check for collisions with other robots
        for i in range(self.num_robots):
            if i != moving_robot:
                robot_x = robot_locations[i * 2]
                robot_y = robot_locations[i * 2 + 1]
                if x == robot_x and y == robot_y:
                    return False
        
        return True

    def is_goal_state(self, state):
        # Ignore the turn indicator when checking goal state
        robot_locations = state[1:]
        return robot_locations == self.goal_locations

    def manhattan_heuristic(self, state):
        robot_locations = state[1:]
        total_distance = 0
        
        for i in range(self.num_robots):
            current_x = robot_locations[i * 2]
            current_y = robot_locations[i * 2 + 1]
            goal_x = self.goal_locations[i * 2]
            goal_y = self.goal_locations[i * 2 + 1]
            
            total_distance += abs(current_x - goal_x) + abs(current_y - goal_y)
        
        return total_distance

    def __str__(self):
        string =  "Mazeworld problem: "
        string += f"{self.num_robots} robots, goal locations {self.goal_locations}"
        return string


        # given a sequence of states (including robot turn), modify the maze and print it out.
        #  (Be careful, this does modify the maze!)

    def animate_path(self, path):
        # reset the robot locations in the maze
        self.maze.robotloc = tuple(self.start_state[1:])

        for state in path:
            print(str(self))
            self.maze.robotloc = tuple(state[1:])
            sleep(1)

            print(str(self.maze))


## A bit of test code. You might want to add to it to verify that things
#  work as expected.

if __name__ == "__main__":
    test_maze3 = Maze("maze3.maz")
    test_mp = MazeworldProblem(test_maze3, (1, 4, 1, 3, 1, 2))

    print(test_mp.get_successors((0, 1, 0, 1, 2, 2, 1)))
