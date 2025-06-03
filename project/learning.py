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
        return self.goal_valid[loc[0], loc[1]] > 0 and self.condition_valid[loc[0], loc[1]].item() > 0
    
    def get_policy(self, qmodel:GoalOrientedBase):
        avoid_region = torch.logical_not(self.condition_valid)
        # Create a dilated avoid region by including adjacent cells

        def dilate_region(mask, reflex=False):
            h, w = mask.shape 
            adj_coords = [(-1,0), (1,0), (0,-1), (0,1)]
            if qmodel.env.n_actions == 8:
                adj_coords.extend([(-1,-1), (-1,1), (1,-1), (1,1)])
            dilation = torch.zeros_like(mask)
            for i, j in torch.nonzero(mask):
                for di, dj in adj_coords:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < h and 0 <= nj < w:
                        dilation[ni,nj] = 1
            if reflex:
                dilation[i,j] = 0     # Orignal avoid region is safe
            return dilation

        dilated_avoid = dilate_region(avoid_region, True)
        unsafe_coords = torch.nonzero(dilated_avoid)
        print(f"The unsafe region for elk has {unsafe_coords.shape[0]} cells.")

        for gname, mask in self.goal_regions.items():
            if any(mask[unsafe_coords[:,0], unsafe_coords[:,1]]):
                print(f"The goal {gname} has intersection with the unsafe region.")

        safe_policy = qmodel.q_compose(qmodel.Q_joint, [2])
        policy = qmodel.q_compose(qmodel.Q_subgoal, [2])
        # policy = torch.maximum(policy, safe_policy)
        # ux, uy = unsafe_coords[:,0], unsafe_coords[:,1]
        # policy[ux, uy] = safe_policy[ux, uy]
        return policy
        
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
    elk_name = "9room"
    room = load_room("saved_disc", f"{elk_name}.pt", 4)
    if 'starting' in room.goals:
        starting = room.goals.pop('starting')
    print(room.goals.keys())
    room.start()
    pretrained = False           # Use the elk's existing knowledge
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
    # at = AtomicTask("F(goal_1)", room)
    # # at = AtomicTask("F goal_2", room)
    # print(at)
    # policy = at.get_policy(goal_learner)
    # at = AtomicTask("!(goal_1) U goal_3", room)
    # policy = at.get_policy(goal_learner)
    # room.draw_policy(policy, fn=f"{elk_name}_at")
    # policy = goal_learner.q_compose(goal_learner.Q_joint, [0,2])
    # room.draw_policy(policy, fn=f"{elk_name}_joint")
    for i in range(len(room.goals)):
        subgoal_policy = goal_learner.q_compose(goal_learner.Q_subgoal, [i])
        # policy = policy.max()+policy.min()-policy
        # This policy negation is not correct, never use it for elk
        room.draw_policy(subgoal_policy, fn=f"{elk_name}_{i}_subgoal")
        joint_policy = goal_learner.q_compose(goal_learner.Q_joint, [i])
        room.draw_policy(joint_policy, fn=f"{elk_name}_{i}_joint")
    # print(at.formula)
    # dfa_task = DFA_Task("(G(t1) & t2)", {"t1": AtomicTask("F(goal_2)", room), "t2": AtomicTask("F(!goal_1)", room)})
