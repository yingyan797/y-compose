import torch, itertools
from boolean_task import GoalOrientedBase, GoalOrientedQLearning
from atomic_task import AtomicTask, get_composed_policy
from reach_avoid_tabular import Room, load_room
import numpy as np
import sympy
from atomic_task import TraceStep
from dataclasses import dataclass

def conjunction_find_shortest_goal_sequence_optimized(
    shared_condition_zone,
    goals: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
    start_location,
    qmodel: GoalOrientedQLearning):
    """
    Optimized version using dynamic programming with memoization.
    Better for larger numbers of goals (> 10).
    
    Uses bitmask DP to reduce complexity from O(n!) to O(n^2 * 2^n).
    """
    n = len(goals)
    if n == 0:
        return [], 0.0, [start_location]
    
    # Cache for get_composed_policy results
    policy_cache = {}
    
    def get_policy_cached(gr:int, location):
        key = (gr, location)
        if key not in policy_cache:
            direct_policy, safe_policy = goals[gr][1]
            policy_cache[key] = get_composed_policy(qmodel, goals[gr][0], shared_condition_zone, direct_policy, safe_policy, location)
        return policy_cache[key]
    
    # DP state: dp[mask][last_goal] = (min_cost, path_to_reconstruct)
    # mask represents which goals have been visited
    dp = {}
    parent = {}
    
    # Initialize: start from each goal directly
    for i, goal in enumerate(goals):
        path, complete = get_policy_cached(i, start_location)
        mask = 1 << i
        dp[(mask, i)] = len(path)
        parent[(mask, i)] = (-1, start_location, path)
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if (mask, last) not in dp:
                continue
                
            current_cost = dp[(mask, last)]
            # Get the location after completing the last goal
            _, _, last_path = parent[(mask, last)]
            current_location = last_path[-1].loc if last_path else start_location
            
            # Try adding each unvisited goal
            for next_goal in range(n):
                if mask & (1 << next_goal):
                    continue
                
                path, cost = get_policy_cached(next_goal, current_location)
                new_mask = mask | (1 << next_goal)
                new_cost = current_cost + cost
                
                if (new_mask, next_goal) not in dp or new_cost < dp[(new_mask, next_goal)]:
                    dp[(new_mask, next_goal)] = new_cost
                    parent[(new_mask, next_goal)] = (last, current_location, path)
    
    # Find the best final state (all goals visited)
    final_mask = (1 << n) - 1
    best_cost = float('inf')
    best_last = -1
    
    for last_goal in range(n):
        if (final_mask, last_goal) in dp and dp[(final_mask, last_goal)] < best_cost:
            best_cost = dp[(final_mask, last_goal)]
            best_last = last_goal
    
    if best_last == -1:
        raise ValueError("No valid path found to complete all goals")
    
    # Reconstruct the path
    sequence = []
    complete_path = [start_location]
    current_mask = final_mask
    current_last = best_last
    
    while current_last != -1:
        sequence.append(current_last)
        prev_last, prev_location, path = parent[(current_mask, current_last)]
        
        if path and len(path) > 1:
            complete_path.extend(path[1:])
        
        if prev_last == -1:
            break
            
        current_mask ^= (1 << current_last)
        current_last = prev_last
    
    sequence.reverse()
    return sequence, best_cost, complete_path

class DFA_Edge:
    def __init__(self, formula=None):
        self.formula = formula      # Single sympy formula, not choices
        self.policy = None
        self.cost_matrix = None
        self.const_cost = None
        if (isinstance(formula, str) and formula == "") or formula == sympy.false:
            self.const_cost = np.inf    # Edge has no viable routes
        elif formula == sympy.true:
            self.const_cost = 0    # Edge is always viable

    def key(self):
        return (self.vertices, self.locs)

    def policy_composition(self, qmodel: GoalOrientedQLearning, atomic_tasks: dict[str, AtomicTask], start_loc):
        """
        Calculate the policy for the edge.
        """
        if self.policy is not None or self.const_cost is not None:
            return      # Only calculate the policy once, and if the edge is not a dead end
        
        shared_condition_zone = torch.ones(qmodel.env.shape, dtype=torch.bool)
        goals = dict[str, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]()
        subformulas = self.formula.args if isinstance(self.formula, sympy.And) else [self.formula]
        for subformula in subformulas:
            if isinstance(subformula, sympy.Not):
                term = subformula.args[0]
                assert isinstance(term, sympy.Symbol), "No nested not"
                task = atomic_tasks[term.name]
                condition, goal = task.find_negation()
                shared_condition_zone = torch.logical_and(shared_condition_zone, condition)
                goals[str(subformula)] = (goal, task.get_dual_policy(qmodel, negation=True))
            elif isinstance(subformula, sympy.Symbol):
                task = atomic_tasks[subformula.name]
                shared_condition_zone = torch.logical_and(shared_condition_zone, task.condition_valid)
                goals[subformula.name] = (task.goal_valid, task.get_dual_policy(qmodel))
            else:
                raise TypeError("No nested formula due to DNF")
        
        # Get all permutations of goals to try different orderings
        print(goals.keys())
        conjunction_goals = list(goals.values())
        sequence, cost, path = conjunction_find_shortest_goal_sequence_optimized(
            shared_condition_zone, conjunction_goals, tuple(start_loc), qmodel)

        print(sequence, cost, path)
            
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
    
if __name__ == "__main__":
    elk_name = "overlap"
    pretrained = True
    room = load_room("saved_disc", f"{elk_name}.pt", 4)
    room.start()
    starting_region = None
    if 'starting' in room.goals:
        starting_region = room.goals.pop('starting')
    print(room.goals.keys())
    atasks = {"t1": AtomicTask("F (goal_1)", room), "t2": AtomicTask("F (goal_2)", room)}
    qmodel = GoalOrientedQLearning(room)
    if pretrained:
        policy = torch.load(f"project/static/policy/{elk_name}.pt")
        qmodel.Q_joint = policy["joint"]
        qmodel.Q_subgoal = policy["subgoal"]
    else:
        qmodel.train_episodes(num_episodes=85, num_iterations=4, max_steps_per_episode=150)
        torch.save({"joint": qmodel.Q_joint, "subgoal": qmodel.Q_subgoal}, f"project/static/policy/{elk_name}.pt")
    
    edge = DFA_Edge(sympy.And(sympy.Symbol("t2"), sympy.Not(sympy.Symbol("t1"))))
    edge.policy_composition(qmodel, atasks, np.array([12,3]))