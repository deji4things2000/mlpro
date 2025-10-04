from Maze import Maze
from time import sleep

class MazeworldProblem:

    ## you write the constructor, and whatever methods your astar function needs

    def __init__(self, maze, goal_locations):
        self.maze = maze
        self.goal_locations = goal_locations
        self.start_state = (0,) + tuple(maze.robotloc)  # (current_robot_turn, x1, y1, x2, y2, ...)
        self.num_robots = len(maze.robotloc) // 2
        
        # Validate that goal locations match number of robots
        if len(goal_locations) != self.num_robots * 2:
            raise ValueError(f"Goal locations {goal_locations} don't match {self.num_robots} robots")

    def get_successors(self, state):
        successors = []
        current_robot = state[0]  # Which robot's turn it is
        robot_locations = state[1:]  # All robot positions
        
        # Five possible actions: wait, move N, E, S, W
        moves = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
        move_names = ["wait", "north", "east", "south", "west"]
        
        for move_idx, (dx, dy) in enumerate(moves):
            new_locations = list(robot_locations)
            
            # For wait action, positions don't change
            if move_idx == 0:  # Wait
                # Just proceed to next robot's turn
                next_robot = (current_robot + 1) % self.num_robots
                new_state = (next_robot,) + tuple(new_locations)
                cost = 0  # Waiting costs no fuel
                action = f"Robot {current_robot} waits"
                successors.append((new_state, cost, action))
                continue
            
            # For movement actions, calculate new position
            current_idx = current_robot * 2
            current_x = robot_locations[current_idx]
            current_y = robot_locations[current_idx + 1]
            new_x = current_x + dx
            new_y = current_y + dy
            
            # Check if this move is valid
            if self.is_valid_move(new_x, new_y, robot_locations, current_robot):
                new_locations[current_idx] = new_x
                new_locations[current_idx + 1] = new_y
                
                # Next robot's turn (round-robin)
                next_robot = (current_robot + 1) % self.num_robots
                new_state = (next_robot,) + tuple(new_locations)
                cost = 1  # Movement costs 1 fuel unit
                action = f"Robot {current_robot} moves {move_names[move_idx]} to ({new_x}, {new_y})"
                successors.append((new_state, cost, action))
        
        return successors

    def is_valid_move(self, x, y, robot_locations, moving_robot):
        # Check if within maze bounds
        if x < 0 or x >= self.maze.width or y < 0 or y >= self.maze.height:
            return False
        
        # Check if target position is a floor (not a wall)
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
    
    def goal_test(self, state):
        return self.is_goal_state(state)

    def manhattan_heuristic(self, state):
        """
        Manhattan distance heuristic: sum of Manhattan distances 
        from each robot to its goal position
        """
        robot_locations = state[1:]
        total_distance = 0
        
        for i in range(self.num_robots):
            current_x = robot_locations[i * 2]
            current_y = robot_locations[i * 2 + 1]
            goal_x = self.goal_locations[i * 2]
            goal_y = self.goal_locations[i * 2 + 1]
            
            total_distance += abs(current_x - goal_x) + abs(current_y - goal_y)
        
        return total_distance
    
    def null_heuristic(self, state):
        """Zero heuristic for uniform cost search"""
        return 0

    def __str__(self):
        string =  "Mazeworld problem: "
        string += f"{self.num_robots} robots, goal locations {self.goal_locations}"
        return string


        # given a sequence of states (including robot turn), modify the maze and print it out.
        #  (Be careful, this does modify the maze!)

    def animate_path(self, path):
        """Animate the path by updating robot positions"""
        if not path:
            print("No path to animate")
            return
            
        # Reset to initial robot locations
        initial_locations = self.start_state[1:]
        self.maze.robotloc = list(initial_locations)
        
        print("Initial state:")
        print(self.maze)
        sleep(1)
        
        for step, state in enumerate(path):
            print(f"Step {step}: Robot {state[0] % self.num_robots}'s turn just ended")
            current_locations = state[1:]
            self.maze.robotloc = list(current_locations)
            
            print(self.maze)
            sleep(1)
            
            if step < len(path) - 1:
                next_state = path[step + 1]
                # Determine what action was taken
                current_robot = state[0]
                next_robot = next_state[0]
                
                if next_robot == (current_robot + 1) % self.num_robots:
                    # A move was made
                    current_idx = current_robot * 2
                    current_pos = (state[1:][current_idx], state[1:][current_idx + 1])
                    next_pos = (next_state[1:][current_idx], next_state[1:][current_idx + 1])
                    
                    if current_pos == next_pos:
                        action = f"Robot {current_robot} waited"
                    else:
                        action = f"Robot {current_robot} moved from {current_pos} to {next_pos}"
                    print(f"Next: {action}")
                sleep(1)


## A bit of test code. You might want to add to it to verify that things
#  work as expected.

if __name__ == "__main__":
    test_maze3 = Maze("maze3.maz")
    test_mp = MazeworldProblem(test_maze3, (1, 4, 1, 3, 1, 2))

    print(test_mp.get_successors((0, 1, 0, 1, 2, 2, 1)))
