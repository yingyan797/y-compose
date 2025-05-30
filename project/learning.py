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
        self.goals_regions = room.goals
        self.full_goals = torch.where(room.terrain>=2, 1, 0).to(dtype=torch.bool)

        if isinstance(self.formula, ltlf.LTLfUntil):
            self.condition = self.formula.formulas[0]
            self.condition_region = self._goal_region(self.condition)
            self.condition_valid = self._valid_region(self.condition)
            self.goal = self.formula.formulas[1]
        elif isinstance(self.formula, ltlf.LTLfEventually):
            self.condition = None
            self.condition_region = None
            self.condition_valid = torch.ones_like(self.full_goals)
            self.goal = self.formula.f
        else:
            raise TypeError(f"Unsupported formula: {formula}")
        
        self.goal_region = self._goal_region(self.goal)
        self.goal_valid = self._valid_region(self.goal)

    def __repr__(self):
        return str(self.formula)
    
    def _goal_region(self, formula:ltlf.LTLfFormula):
        if isinstance(formula, ltlf.LTLfNot):
            return atomic_not(self._goal_region(formula.f), self.full_goals)
        elif isinstance(formula, ltlf.LTLfAnd):
            return atomic_and([self._goal_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfOr):
            return atomic_or([self._goal_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfAtomic):
            return self.goals_regions[formula.s]
        
    def _valid_region(self, formula:ltlf.LTLfFormula):
        if isinstance(formula, ltlf.LTLfNot):
            return torch.logical_not(self._valid_region(formula.f))
        elif isinstance(formula, ltlf.LTLfAnd):
            return atomic_and([self._valid_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfOr):
            return atomic_or([self._valid_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfAtomic):
            return self.goals_regions[formula.s]
        
    def task_complete(self, loc):
        return self.goal_valid[loc[0], loc[1]] > 0 and self.condition_valid[loc[0], loc[1]].item() > 0
    
    def get_policy(self, qmodel:GoalOrientedBase):
        # goal_policy = qmodel.q_compose(self.goal_region)
        valid_goal = self.goal_region & self.condition_valid
        if not torch.any(valid_goal):
            raise ValueError("No valid goal region for this task")
        joint_policy = qmodel.q_compose(valid_goal)
        condition_policy = qmodel.q_compose(self.condition_region) if self.condition_region is not None else torch.zeros_like(joint_policy)
        # return condition_policy + torch.where(goal_policy > 0, goal_policy, 0) # nonegative goal policy offsets condition policy
        return condition_policy + joint_policy
        
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
    room = load_room("saved_disc", "1room.pt", 4)
    if 'starting' in room.goals:
        starting = room.goals.pop('starting')
    print(room.goals.keys())
    room.start()
    goal_learner = GoalOrientedQLearning(room)
    goal_learner.train_episodes()

    at = AtomicTask("F(goal_2)", room)
    # at = AtomicTask("F goal_2", room)
    print(at)
    policy = at.get_policy(goal_learner)
    room.draw_policy(policy, fn="1room_goal_2")
    # print(at.formula)
    # dfa_task = DFA_Task("(G(t1) & t2)", {"t1": AtomicTask("F(goal_2)", room), "t2": AtomicTask("F(!goal_1)", room)})
