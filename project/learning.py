from reach_avoid_tabular import Room, load_room
from atomic_task import AtomicTask, animate_trace
from dfa_task import DFA_Task, DFA_Edge
from boolean_task import GoalOrientedQLearning
import torch, random
import numpy as np

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
            # self.qmodel.plot_training_results(why_done_subgoal, why_done_joint, f"project/static/training/{name}_training")

    def policy_rollout(self, start_loc=None):
        print(f"Dynamic planning started at {start_loc}")
        policy = self.dfa_task.find_shortest_path(0, tuple(start_loc), self.qmodel)
        optimal_path = policy["path"]
        print(f"Optimal path is available!")

        for start_state, next_state, edge_policy in optimal_path:
            # print(start_state, next_state, edge.formula, edge.policy)
            for segment in edge_policy:
                for step in segment:
                    # print(step)
                    if step.action >= 0:                        # State Transition Point
                        self.room.step(step.action, True)
                
        animate_trace(torch.zeros_like(self.room.terrain), self.room.terrain, self.room.get_trace())
        print("Please check animation.")


if __name__ == "__main__":
    elk_name = "office"
    room = load_room("saved_disc", f"{elk_name}.pt", 4)
    planning = Y_Compose(room, elk_name, "G(!t3) & (((t4 | t5) T t1) & (t2 T t1))", 
        {"t1": "F(goal_1)", "t2": "F(goal_2)", "t5": "F(goal_5)", "t3": "F(goal_3)", "t4": "F(goal_4)"}, pretrained=True)
    print(planning.dfa_task)

    constant_start = np.array([10, 3])
    room.start(start_state=constant_start)
    planning.policy_rollout(constant_start)
