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

    def dynamic_planning(self, epsilon=0.01):
        dfa_state = 0
        optimal_path = self.dfa_task.policy_composition(dfa_state)[1]
        while dfa_state not in self.dfa_task.accept_states:
            next_state = optimal_path[1]
            edge: DFA_Edge = self.dfa_task.policy[dfa_state][next_state]
            x = self.room.loc

            if edge.complete(x) == 1:
                if random.random() < epsilon:
                    action = random.choice(list(range(self.room.n_actions)))
                else:
                    action = edge.policy[x[0], x[1]].argmax()
                x, _, _ =self.room.step(action)
            
            

if __name__ == "__main__":
    print(torch.load("project/static/policy/overlap.pt")["joint"])
