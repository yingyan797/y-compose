import torch, random
from boolean_task import GoalOrientedBase, GoalOrientedQLearning
from reach_avoid_tabular import Room, load_room
from ltl_util import formula_parser
import ltlf_tools.ltlf as ltlf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def atomic_and(masks):
    masks = torch.stack(masks)
    return torch.min(masks, dim=0).values
def atomic_or(masks):
    masks = torch.stack(masks)
    return torch.max(masks, dim=0).values
def atomic_not(mask, full_goals):
    neg = torch.logical_not(mask)
    return torch.minimum(neg, full_goals)

def animate_trace(avoid_region, goal_valid, trace_points):
    """Animate the policy testing trace showing the elk's movement through the environment"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    rows, cols = avoid_region.shape
    # Set up the plot limits (swap dimensions for x,y)
    ax.set_xlim(-0.5, cols-0.5)  # x corresponds to columns
    ax.set_ylim(-0.5, rows-0.5)  # y corresponds to rows
    ax.invert_yaxis()
    # Plot the environment - need to swap axes and flip y-axis for proper orientation
    condition_region = plt.imshow(avoid_region.numpy(),
                                cmap='Reds', alpha=0.3,
                                extent=(-0.5, cols-0.5, 
                                        rows-0.5, -0.5),  # Note: y-extent flipped
                                origin='upper')  # This ensures proper orientation

    goal_region = plt.imshow(goal_valid.numpy(),
                            cmap='Greens', alpha=0.3,
                            extent=(-0.5, cols-0.5, 
                                    rows-0.5, -0.5),  # Note: y-extent flipped
                            origin='upper')  # This ensures proper orientation

    # Initialize empty line for the trace
    line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)
    # Initialize marker for current position
    point, = ax.plot([], [], 'ro', markersize=10)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def animate(frame):
        # Convert (row, col) to (x, y) by swapping and inverting y
        x_coords = trace_points[:frame+1, 1]  # col -> x
        y_coords = trace_points[:frame+1, 0]  # row -> y
        
        # Plot trace up to current frame
        line.set_data(x_coords, y_coords)
        
        # Plot current position
        point.set_data([trace_points[frame, 1]], [trace_points[frame, 0]])
        
        return line, point
    

        # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(trace_points), interval=200,
                                blit=True, repeat=True)
    
    plt.grid(True, alpha=0.3)
    plt.title(f"Policy Testing Trace Animation")
    anim.save(f"project/static/training/trace.gif", 
                writer='pillow')
    plt.close()

class TraceStep:
    def __init__(self, r, c, a=-1):
        self.r = r
        self.c = c
        self.action = a

    def best_action(self, policy):
        self.action = policy[self.r, self.c].argmax().item()
        return self.action
    
    def out_of_range(self, room:Room):
        return self.r < 0 or self.r >= room.shape[0] or self.c < 0 or self.c >= room.shape[1] or room.terrain[self.r, self.c] == 0
    
    def get_next_state(self, room:Room, check_range=True):
        dx, dy = room.action_map[self.action]
        next_x, next_y = self.r+dx, self.c+dy
        # Check if next state is in avoid region
        if check_range and not self.out_of_range(room):
            return False, (next_x, next_y)
        else:
            return True, (self.r, self.c)
        
    def __repr__(self):
        return f"Step(({self.r}, {self.c}), a={self.action})"

class AtomicTask:
    def __init__(self, formula, room:Room, name="code_input_atomic_task"):
        self.name = name
        self.room = room
        self.ifml = formula
        self.formula = formula_parser(formula)    
        self.goal_regions = room.goals
        self.full_goals = torch.where(room.terrain>=2, 1, 0).to(dtype=torch.bool)
        self.non_goals = torch.logical_not(self.full_goals)
        self.negated_policy = None
        self.policy = None
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
        if self.goal_valid[loc[0], loc[1]] > 0:
            return 1    # Goal is reached
        elif not self.condition_valid[loc[0], loc[1]].item():
            return -1   # Condition is not met
        else:
            return 0    # Task is not completed
    
    def policy_composition(self, qmodel:GoalOrientedBase, negation=False):
        """
        Calculates the safe and efficient policy for the atomic task.
        """
        # This section is for retrieving both subgoal and joint policies for goal region
        
        if not negation:
            goal_region = self.goal_valid
            avoid_region = torch.logical_not(self.condition_valid)
        else:
            print(f"Negation is called for the atomic task {self.formula}")
            goal_region = torch.logical_not(self.condition_valid)
            avoid_region = self.goal_valid

        goal_coords = torch.nonzero(goal_region)
        intersect_goals = []
        for gr, mask in enumerate(self.goal_regions.values()):
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
            for gr, mask in enumerate(self.goal_regions.values()):
                other_gr = [g for g in intersect_goals if g != gr]
                if not other_gr:
                    continue
                goal_coords = torch.nonzero(mask)
                policy[goal_coords[:,0], goal_coords[:,1]] = qmodel.q_compose(qmodel.Q_subgoal, other_gr)[goal_coords[:,0], goal_coords[:,1]]
                safe_policy[goal_coords[:,0], goal_coords[:,1]] = qmodel.q_compose(qmodel.Q_joint, other_gr)[goal_coords[:,0], goal_coords[:,1]]

        # This section is for iterative safe policy replacement
        terrain_scan = torch.zeros_like(avoid_region)

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
                        out_of_range, (next_r, next_c) = step.get_next_state(self.room)
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
        if negation:
            self.negated_policy = composed_policy
        else:
            self.policy = composed_policy
        return composed_policy, policy, safe_policy
    
    def test_policy(self, policy_name, start_state=None, epsilon=0.05, visualize=True, pretrained=True):
        self.room.start(start_state=start_state, restriction=self.condition_valid)
        qmodel = GoalOrientedQLearning(self.room)
        if pretrained:
            qmodel.Q_joint = torch.load(f"project/static/policy/{policy_name}-jq.pt")
            qmodel.Q_subgoal = torch.load(f"project/static/policy/{policy_name}-sq.pt")
        else:
            qmodel.train_episodes(num_episodes=50, num_iterations=5, max_steps_per_episode=100)
            torch.save(qmodel.Q_joint, f"project/static/policy/{policy_name}-jq.pt")
            torch.save(qmodel.Q_subgoal, f"project/static/policy/{policy_name}-sq.pt")

        policy = self.policy_composition(qmodel)[0]
        max_steps = 100
        steps = 0
        print(f"Testing with initial location: {self.room.loc}")
        while steps < max_steps:
            if epsilon > 0 and random.random() < epsilon:
                action = random.randint(0, self.room.n_actions-1)
            else:
                action = policy[self.room.loc[0], self.room.loc[1]].argmax().item()
            self.room.step(action, trace=True)
            steps += 1
            if self.task_complete(self.room.loc) > 0:
                print(f"Good, Task completed safely in {steps} steps. Final location: {self.room.loc}")
                break
            elif self.task_complete(self.room.loc) < 0:
                print(f"Safety constraint violated. Final location: {self.room.loc}")
                break
        else:
            print(f"Task not completed in {max_steps} steps. Final location: {self.room.loc}")
        if visualize:
            self.room.draw_policy(policy, fn=self.name)
            # print(torch.stack(self.room._trace).numpy().tolist())
            animate_trace(self.condition_valid.logical_not(), self.goal_valid, torch.stack(self.room._trace).numpy())

if __name__ == "__main__":
    elk_name = "overlap"
    room = load_room("saved_disc", f"{elk_name}.pt", 4)
    room.start()
    if 'starting' in room.goals:
        starting = room.goals.pop('starting')
    print(room.goals.keys())
    task = AtomicTask("!(goal_1 & goal_3) U goal_2", room)
    qmodel = GoalOrientedQLearning(room)
    qmodel.Q_joint = torch.load(f"project/static/policy/{elk_name}-jq.pt")
    qmodel.Q_subgoal = torch.load(f"project/static/policy/{elk_name}-sq.pt")
    
    negated_policy, policy, safe_policy = task.policy_composition(qmodel, negation=False)
    # task.test_policy(elk_name, start_state=(11,11), epsilon=0, visualize=True)
    # task.test_policy(elk_name, start_state=(11,11), epsilon=0, visualize=True)

    # negated_policy = policy.max(dim=2, keepdim=True).values+policy.min(dim=2, keepdim=True).values-policy
    room.draw_policy(policy, fn="experiment_negation")


