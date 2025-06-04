from reach_avoid_tabular import Room
from atomic_task import AtomicTask
from dfa_task import DFA_Task, DFA_Edge
from boolean_task import GoalOrientedQLearning
import torch, random

class Y_Compose:
    '''Main class for learning and testing complex policies'''
    def __init__(self, room:Room, name:str, dfa_formula:str, atask_formula:dict[str, str], pretrained=True) -> None:
        self.room = room
        self.dfa_formula = dfa_formula
        atomic_tasks = {name: AtomicTask(formula, room, name) for name, formula in atask_formula.items()}
        self.dfa_task = DFA_Task(dfa_formula, atomic_tasks, name=name)

        qmodel = GoalOrientedQLearning(self.room)
        if pretrained:
            qmodel.load(f"project/static/policy/{name}.pt")
        else:
            qmodel.train_episodes(num_episodes=100, num_iterations=5, max_steps_per_episode=100)
            qmodel.save(f"project/static/policy/{name}.pt")

    def dynamic_planning(self, epsilon=0.01, start_loc=None):
        dfa_state = 0
        policy = self.dfa_task.policy_composition(dfa_state, start_loc)
        optimal_path = policy["path"]

        while dfa_state not in self.dfa_task.accepting_states:
            target_state = optimal_path[dfa_state]      # The next dfa state to move to
            target_edge: DFA_Edge = self.dfa_task.policy[dfa_state][target_state]
            target_policy = target_edge.policy
            x = self.room.loc

            if target_edge.complete(x) == 0:   # Task is not completed
                if random.random() < epsilon:
                    action = random.choice(list(range(self.room.n_actions)))
                else:
                    action = target_policy[x[0], x[1]].argmax()
                x, _, _ =self.room.step(action)
                continue
            elif target_edge.complete(x) == 1:   # Task is completed
                dfa_state = target_state
            else:   # No longer following the optimal path, need to re-plan
                for next_state in range(self.dfa_task.n_states):
                    if next_state in [dfa_state, target_state]:
                        continue
                    next_edge: DFA_Edge = self.dfa_task.policy[dfa_state][next_state]
                    if next_edge.complete(x) == 1:
                        dfa_state = next_state
                        break
                else:
                    raise ValueError("No valid next state found")
                
                optimal_path = self.dfa_task.policy_composition(dfa_state, x)["path"]
            

if __name__ == "__main__":
    print(torch.load("project/static/policy/overlap.pt")["joint"])
