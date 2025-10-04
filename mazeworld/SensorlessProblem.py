from Maze import Maze
from time import sleep

class SensorlessProblem:

    ## You write the good stuff here:
    def __init__(self, maze):
        self.maze = maze
        # Start with all possible floor positions
        all_positions = []
        for x in range(maze.width):
            for y in range(maze.height):
                if maze.is_floor(x, y):
                    all_positions.append((x, y))
        
        # Start state is a tuple of all possible positions (belief state)
        self.start_state = tuple(sorted(all_positions))
        self.width = maze.width
        self.height = maze.height

    def get_successors(self, state):
        successors = []
        # Four possible actions: north, south, east, west
        moves = [(0, 1, "north"), (0, -1, "south"), (1, 0, "east"), (-1, 0, "west")]
        
        for dx, dy, direction in moves:
            new_belief_state = set()
            
            for x, y in state:
                # Calculate intended new position
                new_x, new_y = x + dx, y + dy
                
                # Check if movement is possible (Pacman physics)
                if self.maze.is_floor(new_x, new_y):
                    # Movement successful - robot moves to new position
                    new_belief_state.add((new_x, new_y))
                else:
                    # Movement blocked - robot stays in current position
                    new_belief_state.add((x, y))
            
            # Convert to sorted tuple for hashability and consistency
            new_state = tuple(sorted(new_belief_state))
            cost = 1  # Each action costs 1
            successors.append((new_state, cost, direction))
        
        return successors

    def is_goal_state(self, state):
        # Goal is when we know exactly where the robot is (only one possible position)
        return len(state) == 1
    
    def goal_test(self, state):
        return self.is_goal_state(state)
    
    def basic_heuristic(self, state):
        """Basic heuristic: number of positions minus 1"""
        return len(state) - 1

    def manhattan_heuristic(self, state):
        """Manhattan distance based heuristic: find the bounding box diameter"""
        if len(state) <= 1:
            return 0
        
        min_x = min(pos[0] for pos in state)
        max_x = max(pos[0] for pos in state)
        min_y = min(pos[1] for pos in state)
        max_y = max(pos[1] for pos in state)
        
        # The minimum moves needed is at least the diameter of the belief state
        return (max_x - min_x) + (max_y - min_y)
    
    def advanced_heuristic(self, state):
        """More sophisticated heuristic considering position clustering"""
        if len(state) <= 1:
            return 0
        
        # Count distinct x and y coordinates
        x_coords = set(pos[0] for pos in state)
        y_coords = set(pos[1] for pos in state)
        
        # The robot needs to disambiguate both x and y coordinates
        return (len(x_coords) - 1) + (len(y_coords) - 1)

    def __str__(self):
        return f"Blind robot problem: {len(self.start_state)} possible starting positions"


        # given a sequence of states (including robot turn), modify the maze and print it out.
        #  (Be careful, this does modify the maze!)

    def animate_path(self, path):
        """Animate the path by showing how belief state changes"""
        if not path:
            print("No path to animate")
            return
            
        print("Initial belief state:")
        self.print_belief_state(path[0])
        sleep(1)
        
        for step, state in enumerate(path):
            print(f"\nStep {step}: {len(state)} possible positions")
            self.print_belief_state(state)
            
            if step < len(path) - 1:
                # Show the action taken
                print(f"Action: {self.get_action_between_states(state, path[step + 1])}")
            sleep(1)
        
    def print_belief_state(self, state):
        """Print the maze with all possible positions marked"""
        # Create a representation of the maze
        grid = []
        for y in range(self.height - 1, -1, -1):
            row = []
            for x in range(self.width):
                if (x, y) in state:
                    if self.maze.is_floor(x, y):
                        row.append('R')  # Robot possible position
                    else:
                        row.append('?')  # Shouldn't happen in valid states
                else:
                    if self.maze.is_floor(x, y):
                        row.append('.')
                    else:
                        row.append('#')
            grid.append("".join(row))
        
        for row in grid:
            print(row)

    def get_action_between_states(self, current_state, next_state):
        """Determine what action was taken between two belief states"""
        # This is a simplified version - in practice, we'd track actions
        moves = [(0, 1, "north"), (0, -1, "south"), (1, 0, "east"), (-1, 0, "west")]
        
        for dx, dy, direction in moves:
            simulated_next = set()
            for x, y in current_state:
                new_x, new_y = x + dx, y + dy
                if self.maze.is_floor(new_x, new_y):
                    simulated_next.add((new_x, new_y))
                else:
                    simulated_next.add((x, y))
            
            if set(next_state) == simulated_next:
                return direction
        
        return "unknown"

    

## A bit of test code

if __name__ == "__main__":
    test_maze3 = Maze("/Users/user_1/mlpro/mazeworld/maze3.maz")
    test_problem = SensorlessProblem(test_maze3)

    print("Testing blind robot problem:")
    print(f"Initial belief state size: {len(test_problem.start_state)}")
    print("Initial positions:", test_problem.start_state)
    
    # Test successor function
    print("\nTesting successor function:")
    successors = test_problem.get_successors(test_problem.start_state)
    for state, cost, action in successors:
        print(f"Action '{action}': new belief state size = {len(state)}")