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
    path_segments = []
    current_mask = final_mask
    current_last = best_last
    
    while current_last != -1:
        sequence.append(current_last)
        prev_last, prev_location, path = parent[(current_mask, current_last)]
        
        path_segments = [path] + path_segments
        
        if prev_last == -1:
            break
            
        current_mask ^= (1 << current_last)
        current_last = prev_last
    
    sequence.reverse()
    return sequence, path_segments

class DFA_Edge:
    def __init__(self, formula=None):
        self.formula = formula      # Single sympy formula, not choices
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
        if self.const_cost is not None:
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
        conjunction_goals = list(goals.values())
        sequence, trace = conjunction_find_shortest_goal_sequence_optimized(
            shared_condition_zone, conjunction_goals, tuple(start_loc), qmodel)
        
        # print(self.formula, start_loc, len(trace))
        return [list(goals.keys())[s] for s in sequence], trace

            
    def complete(self, loc):
        if self.goal_valid[loc[0], loc[1]] > 0:
            return 1    # Goal is reached
        elif self.condition_valid[loc[0], loc[1]] == 0:
            return -1   # Condition is not met
        else:
            return 0    # Task is not completed

    
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