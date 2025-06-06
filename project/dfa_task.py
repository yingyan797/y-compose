import torch, sympy
import numpy as np
import heapq
from dataclasses import dataclass
from collections import deque
from atomic_task import AtomicTask, TraceStep, Room
from ltl_util import formula_to_dfa
from boolean_task import GoalOrientedQLearning

def dfa_and(atomic_qs):
    return torch.min(torch.stack(atomic_qs), dim=0).values

class DFA_Edge:
    def __init__(self, formula=None):
        self.formula = formula      # Single sympy formula, not choices
        self.policy = None
        self.condition_valid = None
        self.goal_valid = None
        self.cost_matrix = None
        self.const_cost = None
        if (isinstance(formula, str) and formula == "") or formula == sympy.false:
            self.const_cost = np.inf    # Edge has no viable routes
        elif formula == sympy.true:
            self.const_cost = 0    # Edge is always viable

    def key(self):
        return (self.vertices, self.locs)

    def policy_composition(self, qmodel: GoalOrientedQLearning, atomic_tasks: dict[str, AtomicTask], room: Room):
        if self.policy is not None or self.const_cost is not None:
            return      # Only calculate the policy once, and if the edge is not a dead end
            
        def sub_policy(formula):
            if isinstance(formula, sympy.Not):
                inner = formula.args[0]
                assert isinstance(inner, sympy.Symbol)
                task = atomic_tasks[inner.name]
                if task.negated_policy is None:     # Only calculate the negated policy once
                    task.policy_composition(qmodel, negation=True)
                return task.negated_policy
            elif isinstance(formula, sympy.And):
                return dfa_and([sub_policy(arg) for arg in formula.args])

            elif isinstance(formula, sympy.Symbol):
                task = atomic_tasks[formula.name]
                if task.policy is None:  # Only calculate the policy once
                    task.policy_composition(qmodel, negation=False)
                return task.policy
            
        def condition_valid(formula):
            if isinstance(formula, sympy.Not):
                return torch.logical_not(atomic_tasks[formula.args[0].name].goal_valid)
            elif isinstance(formula, sympy.And):
                return torch.all(condition_valid(arg) for arg in formula.args)
            elif isinstance(formula, sympy.Or):
                return torch.any(condition_valid(arg) for arg in formula.args)
            elif isinstance(formula, sympy.Symbol):
                return atomic_tasks[formula.name].condition_valid
            
        def goal_valid(formula):
            if isinstance(formula, sympy.Not):
                return torch.logical_not(atomic_tasks[formula.args[0].name].goal_valid)
            elif isinstance(formula, sympy.And):
                return torch.all(goal_valid(arg) for arg in formula.args)
            elif isinstance(formula, sympy.Or):
                return torch.any(goal_valid(arg) for arg in formula.args)
            elif isinstance(formula, sympy.Symbol):
                return atomic_tasks[formula.name].goal_valid

        self.policy = sub_policy(self.formula)
        self.condition_valid = condition_valid(self.formula)
        self.goal_valid = goal_valid(self.formula)

        if not torch.logical_and(self.condition_valid, self.goal_valid).any():
            self.const_cost = np.inf    # Dead edge for the task
        else:
            self.estimate_cost(room)

    def complete(self, loc):
        if self.goal_valid[loc[0], loc[1]] > 0:
            return 1    # Goal is reached
        elif self.condition_valid[loc[0], loc[1]] == 0:
            return -1   # Condition is not met
        else:
            return 0    # Task is not completed

    def estimate_cost(self, room: Room):
        avail_locs = torch.nonzero(self.condition_valid).numpy()
        cost_matrix = torch.zeros(room.shape, dtype=torch.int) - 3
        
        for loc in avail_locs:
            if cost_matrix[loc[0], loc[1]] > -3:
                continue
                
            step = TraceStep(loc[0], loc[1])
            cost_matrix[step.r, step.c] = -2
            trace = [step]
            
            def terminate():
                termination = 0
                if self.goal_valid[step.r, step.c]:
                    termination = 1
                elif step.out_of_range(room) or not self.condition_valid[step.r, step.c]:
                    termination = -1
                
                if termination != 0:
                    step.termination = termination
                return termination
            
            while terminate() == 0:
                action = step.best_action(self.policy)
                _, (next_r, next_c) = step.get_next_state(room, check_range=False)

                if cost_matrix[next_r, next_c] == -2 or cost_matrix[next_r, next_c] == -1:
                    step.termination = -1
                    break
                else:
                    step = TraceStep(next_r, next_c)
                    trace.append(step)
                    if cost_matrix[next_r, next_c] > -1:
                        step.termination = 1
                        break
                    cost_matrix[next_r, next_c] = -2
            
            n_steps = torch.IntTensor(list(range(len(trace)-1, -1, -1)))
            trace_coords = torch.IntTensor([[step.r, step.c] for step in trace])
            
            if step.termination == 1:
                if cost_matrix[step.r, step.c] > -1:
                    n_steps += cost_matrix[step.r, step.c]
                cost_matrix[trace_coords[:,0], trace_coords[:,1]] = n_steps
            else:
                cost_matrix[trace_coords[:,0], trace_coords[:,1]] = -1

        self.cost_matrix = cost_matrix

    def contextual_cost(self, prev_edge=None, start_loc=None, room: Room=None):
        def transport():
            step = TraceStep(start_loc[0], start_loc[1])
            while not step.out_of_range(room):
                if not self.condition_valid[step.r, step.c] or self.goal_valid[step.r, step.c]:
                    break
                action = step.best_action(self.policy)
                _, (next_r, next_c) = step.get_next_state(room, check_range=False)
                step = TraceStep(next_r, next_c)
            return (step.r, step.c)

        if start_loc is not None:
            cost = self.cost_matrix[start_loc[0], start_loc[1]].item()
            final_loc = transport()
            return cost if cost > -1 else np.inf, final_loc

        if prev_edge is not None:
            avail_locs = torch.nonzero(torch.logical_and(prev_edge.goal_valid, self.condition_valid))
            if len(avail_locs) == 0:
                return np.inf, None
        else:
            avail_locs = torch.nonzero(self.condition_valid)
        receptive_cost_values = self.cost_matrix[avail_locs[:,0], avail_locs[:,1]]
        reachable_rate = torch.sum(receptive_cost_values > -1).item() / len(receptive_cost_values)
        avg_cost = torch.sum(torch.maximum(torch.zeros_like(receptive_cost_values), 
                                        receptive_cost_values)).item() /len(receptive_cost_values)
        cost = (avg_cost/reachable_rate) if reachable_rate > 0 else np.inf
        return cost, None

@dataclass
class StateNode:
    """Represents a state with current position in search"""
    state: int
    coord: tuple[float, ...]
    total_cost: float
    path: list[tuple[int, int, tuple[float, ...]]]  # [(state, edge_index, coord), ...]
    
    def __lt__(self, other):
        return self.total_cost < other.total_cost
    
    def get_path(self):
        # state: (edge_index, next_state)
        return {self.path[i][0]: (self.path[i+1][0], self.path[i+1][1]) for i in range(len(self.path)-1)}

class DFA_Task:
    """
    Dynamic DFA Dijkstra planning algorithm that finds shortest path to accepting states
    considering location-dependent edge costs.
    """
    
    def __init__(self, formula: str, atomic_tasks: dict[str, AtomicTask], name="code_input_task"):
        self.name = name
        self.formula = formula
        self.atomic_tasks = atomic_tasks
        
        dfa, mona = formula_to_dfa(formula, name)
        self.dfa_matrix = dfa[1]
        self.n_states = len(self.dfa_matrix)
        self.accepting_states = [s-1 for s in dfa[0]['accepting_states']]
        self.rejecting_states = set()
        self.policy = self.policy_matrix()
        self.dfa_state = 0

    def __repr__(self):
        dfa_map = {}
        for i, row in enumerate(self.dfa_matrix):
            for j, choices in enumerate(row):
                if choices == (sympy.false,):
                    continue
                dfa_map[(i, j)] = f"Nondeterministic {choices}" if len(choices) > 1 else choices[0]
        return str(dfa_map)
    
    def policy_matrix(self):
        policy = []
        for i, row in enumerate(self.dfa_matrix):
            p_row = []
            for j, choices in enumerate(row):
                if sympy.false in choices:
                    edges = [DFA_Edge(formula) for formula in choices if formula != sympy.false]
                elif sympy.true in choices:
                    edges = [DFA_Edge(sympy.true)]
                else:
                    edges = [DFA_Edge(formula) for formula in choices]
                p_row.append(edges)
            policy.append(p_row)
        return policy 
        
    def find_shortest_path(self, start_state: int, start_loc: torch.Tensor, 
                           qmodel: GoalOrientedQLearning, room: Room, coordinate_tolerance: float = 1e-6):
        """
        Find shortest path from start state to any accepting state using Dijkstra's algorithm.
        
        Args:
            start_state: Starting state number
            start_coord: Starting physical coordinate
            coordinate_tolerance: Tolerance for coordinate comparison to avoid infinite loops
            
        Returns:
            Dictionary containing:
            - 'path': List of (state, coordinate) tuples representing the path
            - 'total_cost': Total cost of the path
            - 'final_state': The accepting state reached
            - 'final_coord': Final coordinate reached
            Returns None if no path exists
        """
        # Priority queue: (cost, StateNode)
        start_coord = tuple(start_loc.numpy().tolist())
        pq = [StateNode(start_state, start_coord, 0.0, [(start_state, -1, start_coord)])]
        heapq.heapify(pq)
        
        # Visited states with coordinates to avoid cycles
        # Key: (state, rounded_coord_tuple), Value: minimum_cost_reached
        visited = {}
        
        def coord_key(coord: tuple[float, ...]) -> tuple[int, ...]:
            """Create a discrete key from coordinates for cycle detection"""
            return tuple(round(c / coordinate_tolerance) for c in coord)
        
        while pq:
            current = heapq.heappop(pq)
            
            # Check if we reached an accepting state
            if current.state in self.accepting_states:
                return {
                    'path': current.get_path(),
                    'total_cost': current.total_cost,
                    'final_state': current.state,
                    'final_coord': current.coord
                }
            
            # Create key for visited tracking
            state_coord_key = (current.state, coord_key(current.coord))
            
            # Skip if we've visited this state-coordinate with lower cost
            if state_coord_key in visited and visited[state_coord_key] <= current.total_cost:
                continue
            
            visited[state_coord_key] = current.total_cost
            
            # Explore all outgoing edges from current state
            for next_state in range(self.n_states):
                if next_state == current.state:
                    continue    # Skip self-loops

                edges: list[DFA_Edge] = self.policy[current.state][next_state]
                if len(edges) == 0:
                    continue    # Skip edges with no viable routes
                
                for ei, edge in enumerate(edges):
                    # Get cost and next coordinate from edge's estimate function
                    edge.policy_composition(qmodel, self.atomic_tasks, room)
                    edge_cost, next_coord = edge.contextual_cost(None, current.coord, room)
                    
                    # Skip if cost is invalid
                    if edge_cost < 0 or np.isinf(edge_cost) or np.isnan(edge_cost):
                        continue
                    
                    new_total_cost = current.total_cost + edge_cost
                    new_path = current.path + [(next_state, ei, next_coord)]
                    
                    # Create new state node
                    next_node = StateNode(
                        state=next_state,
                        coord=next_coord,
                        total_cost=new_total_cost,
                        path=new_path
                    )
                    
                    # Check if this state-coordinate combination is worth exploring
                    next_key = (next_state, coord_key(next_coord))
                    if next_key not in visited or visited[next_key] > new_total_cost:
                        heapq.heappush(pq, next_node)
        
        # No path found
        return None
        
if __name__ == "__main__":
    room = Room(10, 10)
    task = DFA_Task("G(p1 U p2)", room, None, {"p1": "p1", "p2": "p2"})
    print("Shortest paths from each state to accepting states:")
    print(task.shortest_paths)
    print("\nPaths from state 0 to accepting states:")
    print(task.policy_composition(0))
    print("\nUnreachable states:")
    print(task.get_unreachable_states())