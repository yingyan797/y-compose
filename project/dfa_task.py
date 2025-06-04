import torch, sympy
import numpy as np
from collections import deque
from atomic_task import AtomicTask, TraceStep, Room
from ltl_util import formula_to_dfa

def dfa_and(atomic_qs):
    return torch.min(torch.stack(atomic_qs), dim=0).values
def dfa_or(atomic_qs):
    return torch.max(torch.stack(atomic_qs), dim=0).values

def distribute_not(formula):    # This function should not be used
    """Distributes NOT through a formula using De Morgan's Laws"""
    if isinstance(formula, sympy.Not):
        inner = formula.args[0]
        if isinstance(inner, sympy.And):
            # NOT(A AND B) = NOT(A) OR NOT(B)
            return sympy.Or(*[distribute_not(sympy.Not(arg)) for arg in inner.args])
        elif isinstance(inner, sympy.Or):
            # NOT(A OR B) = NOT(A) AND NOT(B) 
            return sympy.And(*[distribute_not(sympy.Not(arg)) for arg in inner.args])
        elif isinstance(inner, sympy.Not):
            # NOT(NOT(A)) = A
            return distribute_not(inner.args[0])
        else:
            # Base case - atomic proposition or True/False
            return formula
    elif isinstance(formula, sympy.And):
        return sympy.And(*[distribute_not(arg) for arg in formula.args])
    elif isinstance(formula, sympy.Or):
        return sympy.Or(*[distribute_not(arg) for arg in formula.args])
    else:
        # Base case - atomic proposition or True/False
        return formula

class DFA_Edge:
    def __init__(self, formula):
        self.formula = formula
        self.cost = -1   # Edge has a policy
        if (isinstance(formula, str) and formula == "") or isinstance(formula, sympy.false):
            self.cost = np.inf    # Edge has no viable routes
        elif isinstance(formula, sympy.true):
            self.cost = 0    # Edge is always viable

    def policy_composition(self, atomic_tasks:dict[str, AtomicTask], room:Room):
        def sub_policy(formula):
            if isinstance(formula, sympy.Not):
                inner = formula.args[0]
                assert isinstance(inner, sympy.Symbol)  # Upstream has made sure inner term is a symbol
                task = atomic_tasks[inner.name]
                if task.negated_policy is None:     # Only calculate the negated policy once
                    task.policy_composition(self.qmodel, negation=True)
                return task.negated_policy
            elif isinstance(formula, sympy.And):
                return dfa_and([sub_policy(arg) for arg in formula.args])
            elif isinstance(formula, sympy.Or):
                return dfa_or([sub_policy(arg) for arg in formula.args])
            elif isinstance(formula, sympy.Symbol):
                task = self.atomic_tasks[formula.name]
                if task.policy is None:  # Only calculate the policy once
                    task.policy_composition(self.qmodel, negation=False)
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
            self.cost = np.inf    # Dead edge for the task
        else:
            self.estimate_cost(room)

    def complete(self, loc):
        if self.goal_valid[loc[0], loc[1]] > 0:
            return 1    # Goal is reached
        elif not self.condition_valid[loc[0], loc[1]].item():
            return -1   # Condition is not met
        else:
            return 0    # Task is not completed

    def estimate_cost(self, room:Room): # Always viable edge
        avail_locs = torch.nonzero(self.condition_valid).numpy()
        cost_matrix = -3 * torch.ones(room.shape, dtype=int)
        for loc in avail_locs:
            if cost_matrix[loc[0], loc[1]] > -3:  # This location has already been processed
                continue
            step = TraceStep(loc[0], loc[1])
            cost_matrix[step.r, step.c] = -2    # Mark as currently processing
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
            
            while terminate() != 0:
                action = step.best_action(self.policy)
                _, (next_r, next_c) = step.get_next_state(room, check_range=False)
                if cost_matrix[next_r, next_c] == -2 or cost_matrix[next_r, next_c] == -1:
                    # This trace contains a loop or is unreachable
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
                    n_steps += cost_matrix[step.r, step.c]      # Offset by the number of steps already taken
                cost_matrix[trace_coords[:,0], trace_coords[:,1]] = n_steps  # Mark as reachable number of steps
            else:
                cost_matrix[trace_coords[:,0], trace_coords[:,1]] = -1  # Mark as unreachable

        n_reachable = torch.nonzero(cost_matrix > -1).shape[0]
        reachable_rate = n_reachable / torch.numel(cost_matrix)
        avgcost = torch.sum(torch.maximum(0, cost_matrix)).item() / n_reachable
        self.cost = (avgcost/reachable_rate) if reachable_rate > 0 else np.inf

class DFA_Task:
    def __init__(self, formula:str, room, qmodel, atask_formula:dict[str, str], name="code_input_task"):
        self.name = name
        self.formula = formula
        self.qmodel = qmodel
        self.atomic_tasks = {name: AtomicTask(formula, room, name) for name, formula in atask_formula.items()}
        dfa, mona = formula_to_dfa(formula, name)
        self.dfa_matrix = dfa[1]
        self.n_states = len(self.dfa_matrix)
        self.accepting_states = [s-1 for s in dfa[0]['accepting_states']]
        self.rejecting_states = set()
        self.policy = self.policy_matrix()
        self.shortest_paths = self._distance_to_accepting()
        self.dfa_state = 0

    def __repr__(self):
        return str(self.dfa_matrix)+"\n"+str(self.shortest_paths)
    
    def policy_matrix(self):
        policy = []
        for i, row in enumerate(self.dfa_matrix):
            p_row = []
            for formula in row:
                edge = DFA_Edge(formula)
                # Do not compose the policy here, because dijkstra's algorithm will do it if needed
                p_row.append(edge)
            policy.append(p_row)
        return policy
    
    def _distance_to_accepting(self):
        """
        Find shortest path length from each state to any accepting state using BFS.
        
        Returns:
            Dictionary mapping state_id -> shortest_distance (None if unreachable)
        """

        distances = {}
        for start_state in range(self.n_states):
            if start_state in self.accepting_states:
                distances[start_state] = 0
                continue
            
            # BFS from start_state
            queue = deque([(start_state, 0)])
            visited = {start_state}
            found = False
            
            while queue and not found:
                current_state, dist = queue.popleft()
                
                # Check all neighbors
                for next_state in range(self.n_states):
                    if current_state == next_state:
                        continue
                    if self.dfa_matrix[current_state][next_state] != "":  # Edge exists
                        if next_state in self.accepting_states:
                            distances[start_state] = dist + 1
                            found = True
                            break
                        
                        if next_state not in visited:
                            visited.add(next_state)
                            queue.append((next_state, dist + 1))
            
            if not found:
                self.rejecting_states.add(start_state)
                distances[start_state] = None
        
        return distances
    
    def policy_composition(self):
        '''Using Dijkstra's algorithm to find the shortest path to any accepting state'''
        # Don't use the distance function, create an independent solution
    
        # Initialize distances and paths
        distances = {(0, accept): float('inf') for accept in self.accepting_states}
        paths = {(0, accept): [] for accept in self.accepting_states}
        unvisited = set(range(self.n_states))
        current_paths = {state: [] for state in range(self.n_states)}
        
        # Distance to start is 0
        current_distances = {state: float('inf') for state in range(self.n_states)}
        current_distances[0] = 0
        
        while unvisited:
            # Find unvisited node with minimum distance
            current = min(unvisited, key=lambda x: current_distances[x])
            
            if current_distances[current] == float('inf'):
                break
                
            # If we reached an accepting state, record the path
            if current in self.accepting_states:
                distances[(0, current)] = current_distances[current]
                paths[(0, current)] = current_paths[current] + [current]
            
            # Check all neighbors
            for next_state in range(self.n_states):
                if current == next_state:
                    continue
                    
                edge = self.dfa_matrix[current][next_state]
                if edge == "":  # No edge exists
                    continue
                    
                # Only compute edge cost if this path might be better
                if current_distances[next_state] > current_distances[current]:
                    # Get edge cost using policy composition
                    edge_cost = edge.policy_composition()
                    if edge_cost == -1:  # Unknown cost
                        continue
                        
                    new_dist = current_distances[current] + edge_cost
                    
                    if new_dist < current_distances[next_state]:
                        current_distances[next_state] = new_dist
                        current_paths[next_state] = current_paths[current] + [current]
            
            unvisited.remove(current)
            
        return paths
        


