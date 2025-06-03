from ltl_util import formula_to_dfa, LTLfParser
import ltlf_tools.ltlf as ltlf
from reach_avoid_tabular import Room, load_room
from boolean_task import GoalOrientedBase, GoalOrientedNAF, GoalOrientedQLearning
from collections import deque

import torch, sympy
import numpy as np

formula_parser = LTLfParser()

def atomic_and(masks):
    masks = torch.stack(masks)
    return torch.min(masks, dim=0).values
def atomic_or(masks):
    masks = torch.stack(masks)
    return torch.max(masks, dim=0).values
def atomic_not(mask, full_goals):
    neg = torch.logical_not(mask)
    return torch.minimum(neg, full_goals)

class AtomicTask:
    def __init__(self, formula, room:Room, name="code_input_atomic_task"):
        self.name = name
        self.ifml = formula
        self.formula = formula_parser(formula)    
        self.goal_regions = room.goals
        self.full_goals = torch.where(room.terrain>=2, 1, 0).to(dtype=torch.bool)
        self.non_goals = torch.logical_not(self.full_goals)

        if isinstance(self.formula, ltlf.LTLfUntil):
            self.condition = self.formula.formulas[0]
            self.condition_valid = self._valid_region(self.condition)
            self.goal = self.formula.formulas[1]
        elif isinstance(self.formula, ltlf.LTLfEventually):
            self.condition = None
            self.condition_valid = torch.ones_like(self.full_goals)
            self.goal = self.formula.f
        else:
            raise TypeError(f"Unsupported formula: {formula}")
        
        self.goal_valid = self._valid_region(self.goal)

    def __repr__(self):
        return str(self.formula)
        
    def _valid_region(self, formula:ltlf.LTLfFormula):
        if isinstance(formula, ltlf.LTLfNot):
            return torch.logical_not(self._valid_region(formula.f))
        elif isinstance(formula, ltlf.LTLfAnd):
            return atomic_and([self._valid_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfOr):
            return atomic_or([self._valid_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfAtomic):
            return self.goal_regions[formula.s]
        
    def task_complete(self, loc):
        '''Evaluates if the atomic task is completed at the given location.'''
        return self.goal_valid[loc[0], loc[1]] > 0 and self.condition_valid[loc[0], loc[1]].item() > 0
    
    def policy_composition(self, qmodel:GoalOrientedBase):
        """
        Calculates the safe and efficient policy for the atomic task.
        """
        # This section is for retrieving both subgoal and joint policies for goal region
        goal_region = self.goal_valid
        goal_coords = torch.nonzero(goal_region)
        intersect_goals = []
        for gr, (gname, mask) in enumerate(self.goal_regions.items()):
            if torch.equal(mask, goal_region):
                intersect_goals = [gr]
                break
            if any(mask[goal_coords[:,0], goal_coords[:,1]]):
                # Atomic task goal has intersection with the goal region
                intersect_goals.append(gr)
        policy = qmodel.q_compose(qmodel.Q_subgoal, intersect_goals)    # initialize with subgoal policy for non-goal region
        safe_policy = qmodel.q_compose(qmodel.Q_joint, intersect_goals)
        # Check if the goal region is blank
        intersect_blank = any(self.non_goals[goal_coords[:,0], goal_coords[:,1]])
        if intersect_blank:
            # Atomic task goal has intersection with non-goal region
            interior_policy = qmodel.q_compose(qmodel.Q_subgoal, list(range(len(qmodel.goal_regions))))
            interior_safe = qmodel.q_compose(qmodel.Q_joint, list(range(len(qmodel.goal_regions))))
            nongoal_coords = torch.nonzero(self.non_goals)
            policy[nongoal_coords[:,0], nongoal_coords[:,1]] = interior_policy[nongoal_coords[:,0], nongoal_coords[:,1]]
            safe_policy[nongoal_coords[:,0], nongoal_coords[:,1]] = interior_safe[nongoal_coords[:,0], nongoal_coords[:,1]]
        else:
            for gr, (gname, mask) in enumerate(self.goal_regions.items()):
                other_gr = [g for g in intersect_goals if g != gr]
                if not other_gr:
                    continue
                goal_coords = torch.nonzero(mask)
                policy[goal_coords[:,0], goal_coords[:,1]] = qmodel.q_compose(qmodel.Q_subgoal, other_gr)[goal_coords[:,0], goal_coords[:,1]]
                safe_policy[goal_coords[:,0], goal_coords[:,1]] = qmodel.q_compose(qmodel.Q_joint, other_gr)[goal_coords[:,0], goal_coords[:,1]]

        # This section is for iterative safe policy replacement
        avoid_region = torch.logical_not(self.condition_valid)
        terrain_scan = torch.zeros_like(avoid_region)
        class TraceStep:
            def __init__(self, r, c, a=-1):
                self.r = r
                self.c = c
                self.action = a

            def best_action(self, policy):
                self.action = policy[self.r, self.c].argmax().item()
                return self.action
            def get_next_state(self):
                dx, dy = qmodel.env.action_map[self.action]
                next_x, next_y = self.r+dx, self.c+dy
                # Check if next state is in avoid region
                if (0 <= next_x < avoid_region.shape[0] and 
                    0 <= next_y < avoid_region.shape[1]):
                    return False, (next_x, next_y)
                else:
                    return True, (self.r, self.c)
                
            def __repr__(self):
                return f"Step(({self.r}, {self.c}), a={self.action})"

        composed_policy = policy.clone()
        for row in range(avoid_region.shape[0]):
            for col in range(avoid_region.shape[1]):
                if not goal_region[row,col] and not avoid_region[row,col] and not terrain_scan[row,col]:
                    r, c = row, col
                    terrain_scan[r,c] = 1
                    trace = [TraceStep(r, c)]
                    while True:
                        step = trace[-1]
                        a = step.best_action(composed_policy)
                        out_of_range, (next_r, next_c) = step.get_next_state()
                        if goal_region[next_r, next_c] or terrain_scan[next_r, next_c]:
                            break   # Keep original policy

                        elif out_of_range or avoid_region[next_r, next_c] or (
                                composed_policy[next_r, next_c].argmax().item() == qmodel.env.opposite_action(a)):
                            trace_tensor = torch.IntTensor([[step.r, step.c] for step in trace]+[[next_r, next_c]])
                            composed_policy[trace_tensor[:,0], trace_tensor[:,1]] = safe_policy[trace_tensor[:,0], trace_tensor[:,1]]
                            trace = trace[-1:]
                            if out_of_range:
                                break
                        else:
                            trace.append(TraceStep(next_r, next_c))
                            r, c = next_r, next_c
                            terrain_scan[r,c] = 1      

        return composed_policy, policy, safe_policy
        
def dfa_and(atomic_qs):
    return torch.min(torch.stack(atomic_qs), dim=0).values
def dfa_or(atomic_qs):
    return torch.max(torch.stack(atomic_qs), dim=0).values
def dfa_not(atomic_q):
    low_q = torch.min(atomic_q)
    high_q = torch.max(atomic_q)
    return low_q + high_q - atomic_q

class DFA_Task:
    def __init__(self, formula:str, atomic_tasks:dict[str, AtomicTask], name="code_input_task"):
        self.name = name
        self.formula = formula
        self.atomic_tasks = atomic_tasks
        dfa, mona = formula_to_dfa(formula, name)
        self.dfa_matrix = dfa[1]
        self.n_states = len(self.dfa_matrix)
        self.accepting_states = [s-1 for s in dfa[0]['accepting_states']]
        self.rejecting_states = set()
        self.policy = self._dfa_policy()
        self.shortest_paths = self._distance_to_accepting()
        self.dfa_state = 0

    def __repr__(self):
        return str(self.dfa_matrix)+"\n"+str(self.shortest_paths)
    
    def _dfa_policy(self):
        def edge_policy(formula):
            if formula == sympy.true:
                return 1
            elif isinstance(formula, sympy.Not):
                return dfa_not(edge_policy(formula.args[0]))
            elif isinstance(formula, sympy.And):
                return dfa_and([edge_policy(arg) for arg in formula.args])
            elif isinstance(formula, sympy.Or):
                return dfa_or([edge_policy(arg) for arg in formula.args])
            elif isinstance(formula, sympy.Symbol):
                return self.atomic_tasks[formula.name].get_policy("")
            else:
                raise TypeError(f"Unsupported edge formula: {formula}")
        
        policy = []
        for i, row in enumerate(self.dfa_matrix):
            p_row = []
            for formula in row:
                if formula == "":
                    p_row.append(0)
                else:
                    p_row.append(edge_policy(formula))
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
    
    def get_composed_policy(self):
        pass

class DFA_dijkstra(DFA_Task):
    def __init__(self, dfa):
        super().__init__(dfa)
        self.adjacency_matrix = np.zeros((self.n_states, self.n_states))

    def get_composed_policy(self):
        pass

if __name__ == "__main__":
    elk_name = "overlap"
    room = load_room("saved_disc", f"{elk_name}.pt", 4)
    if 'starting' in room.goals:
        starting = room.goals.pop('starting')
    print(room.goals.keys())
    room.start()
    pretrained = True           # Use the elk's existing knowledge
    goal_learner = GoalOrientedQLearning(room)
    if not pretrained:
        goal_learner.train_episodes(num_episodes=50, num_iterations=5, max_steps_per_episode=100)
        torch.save(goal_learner.Q_joint, f"project/static/policy/{elk_name}-jq.pt")
        torch.save(goal_learner.Q_subgoal, f"project/static/policy/{elk_name}-sq.pt")
    else:
        q_matrix = torch.load(f"project/static/policy/{elk_name}-jq.pt")
        goal_learner.Q_joint = q_matrix
        q_matrix = torch.load(f"project/static/policy/{elk_name}-sq.pt")
        goal_learner.Q_subgoal = q_matrix
    at = AtomicTask("!(goal_1 & goal_3) U goal_2", room)
    # at = AtomicTask("F goal_2", room)
    print(at)
    composed_policy, policy, safe_policy = at.policy_composition(goal_learner)
    # at = AtomicTask("!(goal_1) U goal_3", room)
    # policy = at.get_policy(goal_learner)
    room.draw_policy(composed_policy, fn=f"{elk_name}_at")
    room.draw_policy(safe_policy, fn=f"{elk_name}_safe")
    room.draw_policy(policy, fn=f"{elk_name}_policy")
    # policy = goal_learner.q_compose(goal_learner.Q_joint, [0,2])
    # room.draw_policy(policy, fn=f"{elk_name}_joint")
    # for i in range(len(room.goals)):
    #     subgoal_policy = goal_learner.q_compose(goal_learner.Q_subgoal, [i])
    #     # policy = policy.max()+policy.min()-policy
    #     # This policy negation is not correct, never use it for elk
    #     room.draw_policy(subgoal_policy, fn=f"{elk_name}_{i}_subgoal")
    #     joint_policy = goal_learner.q_compose(goal_learner.Q_joint, [i])
    #     room.draw_policy(joint_policy, fn=f"{elk_name}_{i}_joint")
    # print(at.formula)
    # dfa_task = DFA_Task("(G(t1) & t2)", {"t1": AtomicTask("F(goal_2)", room), "t2": AtomicTask("F(!goal_1)", room)})
