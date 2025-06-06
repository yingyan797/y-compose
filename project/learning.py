from reach_avoid_tabular import Room, load_room
from atomic_task import AtomicTask, animate_trace
from dfa_task import DFA_Task, DFA_Edge
from boolean_task import GoalOrientedQLearning
import torch, random

class Y_Compose:
    '''Main class for learning and testing complex policies'''
    def __init__(self, room:Room, name:str, dfa_formula:str, atask_formula:dict[str, str], pretrained=True) -> None:
        self.room = room
        self.dfa_formula = dfa_formula
        room.start()
        self.starting_region = None
        if "starting" in room.goals:
            self.starting_region = room.goals.pop("starting")
        print(room.goals.keys())
        atomic_tasks = {name: AtomicTask(formula, room, name) for name, formula in atask_formula.items()}
        self.dfa_task = DFA_Task(dfa_formula, atomic_tasks, name=name)

        self.qmodel = GoalOrientedQLearning(self.room)
        if pretrained:
            policy = torch.load(f"project/static/policy/{name}.pt", weights_only=True)
            self.qmodel.Q_joint = policy["joint"]
            self.qmodel.Q_subgoal = policy["subgoal"]
        else:
            why_done_subgoal, why_done_joint = self.qmodel.train_episodes(num_episodes=85, num_iterations=4, max_steps_per_episode=85)
            torch.save({"joint": self.qmodel.Q_joint, "subgoal": self.qmodel.Q_subgoal}, f"project/static/policy/{name}.pt")
            self.qmodel.plot_training_results(why_done_subgoal, why_done_joint, f"project/static/training/{name}_training")

    def dynamic_planning(self, epsilon=0.01, start_loc=None):
        print(f"Dynamic planning started at {start_loc}")
        dfa_state = 0
        policy = self.dfa_task.find_shortest_path(dfa_state, start_loc, self.qmodel, self.room)
        optimal_path = policy["path"]

        print(f"Optimal path is available: {optimal_path}")
        while dfa_state not in self.dfa_task.accepting_states:
            target_state, edge_index = optimal_path[dfa_state]      # The next dfa state to move to
            target_edge: DFA_Edge = self.dfa_task.policy[dfa_state][target_state][edge_index]
            target_policy = target_edge.policy
            x = self.room.loc

            completion = target_edge.complete(x)
            if completion == 0:   # Task is not completed
                if random.random() < epsilon:
                    action = random.choice(list(range(self.room.n_actions)))
                else:
                    action = target_policy[x[0], x[1]].argmax().item()
                x, _, _ = self.room.step(action, trace=True)
                continue
            elif completion == 1:   # Task is completed
                dfa_state = target_state
            else:   # No longer following the optimal path, need to re-plan
                for next_state in range(self.dfa_task.n_states):
                    for next_edge in self.dfa_task.policy[dfa_state][next_state]:
                        if next_edge.complete(x) == 1:
                            dfa_state = next_state
                            break
                    break
                else:
                    next_state = dfa_state
 
                print(f"Re-planning to {next_state}")    
                
                policy = self.dfa_task.find_shortest_path(dfa_state, x, self.qmodel, self.room)
                optimal_path = policy["path"]
        
        animate_trace(torch.zeros_like(self.room.terrain), self.room.terrain, self.room.get_trace())


if __name__ == "__main__":
    room = load_room("saved_disc", "9room.pt")
    planning = Y_Compose(room, "9room", "t1 T t2", 
        {"t1": "! goal_1 U (goal_2 | goal_3)", "t2": "F(goal_4)"}, pretrained=True)
    print(planning.dfa_task)
    loc = room.start(restriction=planning.starting_region)
    planning.dynamic_planning(epsilon=0.01, start_loc=loc)
