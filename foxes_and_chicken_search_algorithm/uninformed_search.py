
from collections import deque
from SearchSolution import SearchSolution

class SearchNode:

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent


def bfs_search(search_problem):
    solution = SearchSolution(search_problem, "BFS")

    start_node = SearchNode(search_problem.start_state)
    frontier = deque([start_node])
    explored = set()

    while frontier:
        node = frontier.popleft()
        state = node.state

        if state in explored:
            continue

        explored.add(state)
        solution.nodes_visited += 1

        if search_problem.goal_test(state):
            # Backchan]in to get path
            path = []
            current = node
            while current:
                path.append(current.state)
                current = current.parent
            path.reverse()
            solution.path = path
            return solution
        
        for successor in search_problem.get_successors(state):
            if successor not in explored:
                child_node = SearchNode(successor, node)
                frontier.append(child_node)
    return solution

def dfs_search(search_problem):
    return dfs_search_recursive(search_problem)


def dfs_search_recursive(search_problem, depth_limit=100, node=None, solution=None):
    # if no node object given, create a new search from starting state
    if node == None:
        node = SearchNode(search_problem.start_state)
        solution = SearchSolution(search_problem, "DFS")

    state = node.state
    solution.nodes_visited +=1

    if search_problem.goal_test(state):
        path = []
        current = node
        while current:
            path.append(current.state)
            current = current.parent
        path.reverse()
        solution.path = path
        return solution
    
    if depth_limit <= 0:
        return solution
    
    # Path Checking - only checking current path
    current_path = set()
    temp = node
    while temp:
        current_path.add(temp.state)
        temp = temp.parent

    for successor in search_problem.get_successors(state):
        if successor not in current_path:
            child_node = SearchNode(successor, node)
            result = dfs_search_recursive(search_problem, depth_limit-1, child_node, solution)
            if result.path:
                return result
            
    return solution

def ids_search(search_problem):
    solution = SearchSolution(search_problem, "IDS")
    depth_limit = 100

    for depth in range(depth_limit + 1):
        temp_solution = SearchSolution(search_problem, "DFS")
        result = dfs_search_recursive(search_problem, depth)
        solution.nodes_visited += result.nodes_visited

        if result.path:
            solution.path = result.path
            return solution
    
    return solution