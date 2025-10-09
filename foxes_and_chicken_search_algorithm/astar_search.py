from SearchSolution import SearchSolution
from heapq import heappush, heappop

class AstarNode:
    # each search node except the root has a parent node
    # and all search nodes wrap a state object

    def __init__(self, state, heuristic, parent=None, transition_cost=0):
        # you write this part
        self.state = state
        self.heuristic = heuristic
        self.parent = parent
        self.transition_cost = transition_cost

    def priority(self):
        # you write this part
        return self.transition_cost + self.heuristic

    # comparison operator,
    # needed for heappush and heappop to work with AstarNodes:
    def __lt__(self, other):
        return self.priority() < other.priority()


# take the current node, and follow its parents back
#  as far as possible. Grab the states from the nodes,
#  and reverse the resulting list of states.
def backchain(node):
    result = []
    current = node
    while current:
        result.append(current.state)
        current = current.parent

    result.reverse()
    return result


def astar_search(search_problem, heuristic_fn):
    # I'll get you started:
    start_node = AstarNode(search_problem.start_state, heuristic_fn(search_problem.start_state))
    pqueue = []
    heappush(pqueue, start_node)

    solution = SearchSolution(search_problem, "Astar with heuristic " + heuristic_fn.__name__)

    # Dictionary to track best known cost to reach each state
    best_cost = {}
    best_cost[start_node.state] = 0

    while pqueue:
        current_node = heappop(pqueue)
        current_state = current_node.state
        
        # Lazy deletion: if we've found a better path to this state since this node was added, skip it
        if current_node.transition_cost > best_cost.get(current_state, float('inf')):
            continue
            
        solution.nodes_visited += 1

        if search_problem.goal_test(current_state):
            solution.path = backchain(current_node)
            solution.cost = current_node.transition_cost
            return solution

        for child_state, cost, action in search_problem.get_successors(current_state):
            new_cost = current_node.transition_cost + cost
            
            # Only consider this child if it's better than any path we've found before
            if child_state not in best_cost or new_cost < best_cost[child_state]:
                best_cost[child_state] = new_cost
                heuristic_val = heuristic_fn(child_state)
                child_node = AstarNode(child_state, heuristic_val, current_node, new_cost)
                heappush(pqueue, child_node)

    return solution

