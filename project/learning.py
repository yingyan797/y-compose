from ltl_util import formula_to_dfa, LTLfParser
import ltlf_tools.ltlf as ltlf
from reach_avoid_tabular import Room, load_room
from boolean_task import GoalOrientedBase, GoalOrientedNAF, GoalOrientedQLearning
import torch

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
    def __init__(self, formula, room:Room):
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
            self.condition_region = self.full_goals
            self.condition_valid = torch.ones_like(self.full_goals)
            self.goal = self.formula.formula
        else:
            raise TypeError(f"Unsupported formula: {formula}")
        
        self.goal_region = self._goal_region(self.goal)
        self.goal_valid = self._valid_region(self.goal)

    def __repr__(self):
        return self.formula
    
    def _goal_region(self, formula:ltlf.LTLfFormula):
        if isinstance(formula, ltlf.LTLfNot):
            return atomic_not(self._goal_region(formula.formula), self.full_goals)
        elif isinstance(formula, ltlf.LTLfAnd):
            return atomic_and([self._goal_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfOr):
            return atomic_or([self._goal_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfAtomic):
            return self.goals_regions[formula.s]
        
    def _valid_region(self, formula:ltlf.LTLfFormula):
        if isinstance(formula, ltlf.LTLfNot):
            return torch.logical_not(self._valid_region(formula.formula))
        elif isinstance(formula, ltlf.LTLfAnd):
            return atomic_and([self._valid_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfOr):
            return atomic_or([self._valid_region(f) for f in formula.formulas])
        elif isinstance(formula, ltlf.LTLfAtomic):
            return self.goals_regions[formula.s]
        
    def task_complete(self, loc):
        return self.goal_valid[loc[0], loc[1]] > 0 and self.condition_valid[loc[0], loc[1]].item() > 0
    
    def get_policy(self, qmodel:GoalOrientedBase):
        goal_policy = qmodel.q_compose(self.goal_region)
        condition_policy = qmodel.q_compose(self.condition_region)
        return condition_policy + torch.where(goal_policy > 0, goal_policy, 0) # nonegative goal policy offsets condition policy
        
class DFA_Task:
    def __init__(self, formula:str):
        self.formula = formula
        self.dfa = formula_to_dfa(formula)

class DFA_dijkstra:
    def __init__(self, dfa):
        self.dfa = dfa

if __name__ == "__main__":
    room = load_room("saved_disc", "9room.pt")
    room.start()
    at = AtomicTask("(goal_1 U goal_2)", room)
    print(at.formula)
