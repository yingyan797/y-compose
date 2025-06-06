import torch, sympy
import numpy as np
import heapq
from collections import deque
from atomic_task import AtomicTask, TraceStep, Room
from ltl_util import formula_to_dfa
from boolean_task import GoalOrientedQLearning

def dfa_and(atomic_qs):
    return torch.min(torch.stack(atomic_qs), dim=0).values

def dfa_or(atomic_qs):
    return torch.max(torch.stack(atomic_qs), dim=0).values

class DFA_Edge:
    def __init__(self, formula):
        self.formula = formula
        self.policy = None
        self.condition_valid = None
        self.goal_valid = None
        self.cost_matrix = None
        self.const_cost = None
        if (isinstance(formula, str) and formula == "") or formula == sympy.false:
            self.const_cost = np.inf    # Edge has no viable routes
        elif formula == sympy.true:
            self.const_cost = 0    # Edge is always viable

    def policy_composition(self, qmodel: GoalOrientedQLearning, atomic_tasks: dict[str, AtomicTask], room: Room):
        if self.policy is not None:
            return
            
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
            elif isinstance(formula, sympy.Or):
                return dfa_or([sub_policy(arg) for arg in formula.args])
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
        avail_locs = torch.nonzero(self.condition_valid)
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

    def contextual_cost(self, prev_edge=None, start_loc=None):
        if start_loc is not None:
            avail_locs = torch.tensor(start_loc.unsqueeze(0))
        elif prev_edge is not None:
            avail_locs = torch.nonzero(torch.logical_and(prev_edge.goal_valid, self.condition_valid))
            if len(avail_locs) == 0:
                return np.inf
        else:
            avail_locs = torch.nonzero(self.condition_valid)

        receptive_cost_values = self.cost_matrix[avail_locs[:,0], avail_locs[:,1]]
        reachable_rate = torch.sum(receptive_cost_values > -1).item() / len(receptive_cost_values)
        avg_cost = torch.sum(torch.maximum(torch.zeros_like(receptive_cost_values), 
                                           receptive_cost_values)).item() /len(receptive_cost_values)
        cost = (avg_cost/reachable_rate) if reachable_rate > 0 else np.inf
        return cost

    
class DFA_Task:
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
        return str(self.dfa_matrix)
    
    def policy_matrix(self):
        policy = []
        for i, row in enumerate(self.dfa_matrix):
            p_row = []
            for formula in row:
                edge = DFA_Edge(formula)
                p_row.append(edge)
            policy.append(p_row)
        return policy 
    
    def policy_composition(self, room: Room, qmodel: GoalOrientedQLearning, start_state=0, start_loc=None):
        """Get the shortest paths from start state to all accepting states with context-aware edge costs"""
        # State representation: (previous_state, current_state)
        dist = {}
        dist[(None, start_state)] = 0
        prev = {}
        visited = set()
        pq = [(0, None, start_state)]  # (distance, prev_state, current_state)
        
        while pq:
            current_dist, prev_state, current_state = heapq.heappop(pq)
            
            state_key = (prev_state, current_state)
            if state_key in visited:
                continue
                
            visited.add(state_key)
            
            # If we reached the target accepting state
            if current_state in self.accepting_states:
                # Reconstruct path
                curr_key = state_key
                path = {}
                while curr_key is not None:
                    path[curr_key[1]] = curr_key[0]
                    curr_key = prev.get(curr_key)
                
                return {
                    'path': path,
                    'accepting_state': current_state,
                    'distance': current_dist,
                }
            
            # Explore neighbors
            prev_edge = self.policy[prev_state][current_state] if prev_state is not None else None
            for next_state in range(self.n_states):
                next_key = (current_state, next_state)
                if next_key not in visited:
                    edge: DFA_Edge = self.policy[current_state][next_state]
                    if edge.const_cost is None:
                        edge.policy_composition(qmodel, self.atomic_tasks, room)
                        # Get contextual edge cost
                        edge_cost = edge.contextual_cost(prev_edge, start_loc)
                    else:
                        edge_cost = edge.const_cost
                        
                    if edge_cost == np.inf:
                        continue
                    new_dist = current_dist + edge_cost
                    if next_key not in dist or new_dist < dist[next_key]:
                        dist[next_key] = new_dist
                        prev[next_key] = state_key
                        heapq.heappush(pq, (new_dist, current_state, next_state))

            if start_loc is not None:
                start_loc = None    # Starting location is used only once
        
        return {}
        
if __name__ == "__main__":
    room = Room(10, 10)
    task = DFA_Task("G(p1 U p2)", room, None, {"p1": "p1", "p2": "p2"})
    print("Shortest paths from each state to accepting states:")
    print(task.shortest_paths)
    print("\nPaths from state 0 to accepting states:")
    print(task.policy_composition(0))
    print("\nUnreachable states:")
    print(task.get_unreachable_states())