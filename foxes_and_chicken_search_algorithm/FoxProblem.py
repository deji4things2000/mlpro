class FoxProblem:
    def __init__(self, start_state=(3, 3, 1)):
        self.start_state = start_state
        self.goal_state = (0, 0, 0) #the goal is to have zero chicken, foxes, and boats at the start state.
        self.total_chickens = start_state[0]
        self.total_foxes = start_state[1]
        self.boat_capacity = 2

    def get_successors(self, state):
        successors = []
        c, f, b = state # where c represents chicken, f represents foxes, and b represents boat
        if b == 1: # Boat is on starting side (i.e. b = 1)
            for move_c in range(self.boat_capacity + 1):
                for move_f in range(self.boat_capacity + 1):
                    total_move = move_c + move_f
                    if total_move > 0 and total_move <=self.boat_capacity:
                        new_c = c - move_c
                        new_f = f - move_f
                        new_b = 0

                        if self.is_safe((new_c, new_f, new_b)):
                            successors.append((new_c, new_f, new_b))
        else: # Boat is on starting side (i.e. b = 0)
            for move_c in range(self.boat_capacity + 1):
                for move_f in range(self.boat_capacity + 1):
                    total_move = move_c + move_f
                    if total_move > 0 and total_move <=self.boat_capacity:
                        new_c = c + move_c
                        new_f = f + move_f
                        new_b = 1

                        if self.is_safe((new_c, new_f, new_b)):
                            successors.append((new_c, new_f, new_b))
        return successors
    def is_safe(self, state):
        c, f, b = state #where c represents chicken, f represents foxes, and b represents boat
        goal_c = self.total_chickens - c
        goal_f = self.total_foxes - f

        #Check bounds

        if c < 0 or c > self.total_chickens or f < 0 or f > self.total_foxes:
            return False
        
        # Check starting side safety
        if c > 0 and f > c:
            return False
        
        # Check goal side safety
        if goal_c > 0 and goal_f > goal_c:
            return False
        
        return True
    
    def goal_test(self, state):
        return state == self.goal_state


    def __str__(self):
        string =  "Chickens and foxes problem: " + str(self.start_state)
        return string


## A bit of test code

if __name__ == "__main__":
    test_cp = FoxProblem((5, 5, 1))
    print(test_cp.get_successors((5, 5, 1)))
    print(test_cp)
