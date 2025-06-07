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
    # Not used
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
    def __init__(self, loc:tuple, a=-1):
        self.loc = loc
        self.action = a

    def best_action(self, policy):
        self.action = policy[self.loc].argmax().item()
        return self.action
    
    def out_of_range(self, room:Room):
        oor = tuple(self.loc[i] < 0 or self.loc[i] >= room.shape[i] for i in range(room.state_dim))
        return any(oor) or room.terrain[self.loc] == 0
    
    def get_next_state(self, room:Room, check_range=True):
        next_loc = tuple(self.loc[i] + room.action_map[self.action][i] for i in range(room.state_dim))
        # Check if next state is in avoid region
        if not check_range or not self.out_of_range(room):
            return False, next_loc
        else:
            return True, self.loc
        
    def __repr__(self):
        return f"Step({self.loc}, a={self.action})"

class AtomicTask:
    def __init__(self, formula, room:Room, name="code_input_atomic_task"):
        self.name = name
        self.room = room
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
        self.condition_negated = None
        self.goal_negated = None
        self.dual_policy = None
        self.dual_negated_policy = None

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
        ctuple = tuple(loc)
        if self.goal_valid[ctuple] > 0:
            return 1    # Goal is reached
        elif not self.condition_valid[ctuple].item():
            return -1   # Condition is not met
        else:
            return 0    # Task is not completed
        
    def find_negation(self):
        if self.condition_negated is None:
            self.condition_negated = torch.logical_not(self.goal_valid)
            self.goal_negated = self.condition_negated
        return self.condition_negated, self.goal_negated
    
    def get_dual_policy(self, qmodel:GoalOrientedBase, negation=False):
        if negation:
            if self.dual_negated_policy is not None:
                return self.dual_negated_policy
        else:
            if self.dual_policy is not None:
                return self.dual_policy
        goal_valid = self.goal_valid if not negation else self.goal_negated
        goal_coords = torch.nonzero(goal_valid)
        ctuple = tuple(goal_coords[:,i] for i in range(qmodel.env.state_dim))
        intersect_goals = []
        for gr, mask in enumerate(self.goal_regions.values()):
            if torch.equal(mask, goal_valid):
                intersect_goals = [gr]
                break
            if any(mask[ctuple]):
                # Atomic task goal has intersection with the goal region
                intersect_goals.append(gr)
        policy = qmodel.q_compose(qmodel.Q_subgoal, intersect_goals)    # initialize with subgoal policy for non-goal region
        safe_policy = qmodel.q_compose(qmodel.Q_joint, intersect_goals)
        # Check if the goal region is blank
        for gr, mask in enumerate(self.goal_regions.values()):
            other_gr = [g for g in intersect_goals if g != gr]
            if not other_gr:
                continue
            goal_coords = torch.nonzero(mask)
            ctuple = tuple(goal_coords[:,i] for i in range(qmodel.env.state_dim))
            policy[ctuple] = qmodel.q_compose(qmodel.Q_subgoal, other_gr)[ctuple]
            safe_policy[ctuple] = qmodel.q_compose(qmodel.Q_joint, other_gr)[ctuple]
        
        if negation:
            self.dual_negated_policy = policy, safe_policy
        else:
            self.dual_policy = policy, safe_policy
        return policy, safe_policy

    def policy_composition(self, qmodel:GoalOrientedBase):
        policy, safe_policy = self.get_dual_policy(qmodel)
        return get_composed_policy(qmodel, self.goal_valid, self.condition_valid, policy, safe_policy)
        
    def test_policy(self, qmodel, start_state=None, restriction=None, epsilon=0.05, visualize=True):
        self.room.start(start_state=start_state, restriction=self.condition_valid if restriction is None else 
                        torch.logical_and(self.condition_valid, restriction))
        policy = self.policy_composition(qmodel)[0]
        max_steps = 100
        steps = 0
        print(f"Testing with initial location: {self.room.loc}")
        while steps < max_steps:
            if epsilon > 0 and random.random() < epsilon:
                action = random.randint(0, self.room.n_actions-1)
            else:
                action = policy[tuple(self.room.loc)].argmax().item()
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
            animate_trace(self.condition_valid.logical_not(), self.goal_valid, self.room.get_trace())

def get_composed_policy(qmodel:GoalOrientedBase, goal, condition, policy, safe_policy, starting_loc=None):
    """
    Calculates the safe and efficient policy for the atomic task.
        """    
    # This section is for iterative safe policy replacement
    terrain_scan = torch.zeros(goal.shape, dtype=torch.int)
    composed_policy = policy.clone()
    
    def policy_replacement(ctuple):
        trace = [TraceStep(ctuple)]
        if goal[ctuple]:
            return trace, True
        out_of_range = False
        for _ in range(torch.numel(goal)):
            step = trace[-1]
            a = step.best_action(composed_policy)
            out_of_range, next_loc = step.get_next_state(qmodel.env)
            if goal[next_loc] or terrain_scan[next_loc] == 2:
                trace_tensor = torch.IntTensor([step.loc for step in trace])
                terrain_scan[tuple(trace_tensor[:,i] for i in range(qmodel.env.state_dim))] = 2
                return trace + [TraceStep(next_loc)], True      # Completed within limit
            terrain_scan[step.loc] = 1     
            if out_of_range or not condition[next_loc] or terrain_scan[next_loc] == 1:
                composed_policy[step.loc] = safe_policy[step.loc]
                a = step.best_action(composed_policy)
                out_of_range, next_loc = step.get_next_state(qmodel.env)
            trace.append(TraceStep(next_loc))
        return trace, False      # Not completed within limit

    if starting_loc is None:
        for loc in condition.nonzero().numpy():
            ctuple = tuple(loc)
            if condition[ctuple] and not terrain_scan[ctuple]:
                policy_replacement(ctuple)

        return composed_policy, policy, safe_policy
    
    return policy_replacement(starting_loc)

if __name__ == "__main__":
    elk_name = "overlap"
    pretrained = True
    room = load_room("saved_disc", f"{elk_name}.pt", 4)
    room.start()
    starting_region = None
    if 'starting' in room.goals:
        starting_region = room.goals.pop('starting')
    print(room.goals.keys())
    task = AtomicTask("F (!goal_3)", room)
    qmodel = GoalOrientedQLearning(room)
    if pretrained:
        policy = torch.load(f"project/static/policy/{elk_name}.pt")
        qmodel.Q_joint = policy["joint"]
        qmodel.Q_subgoal = policy["subgoal"]
    else:
        qmodel.train_episodes(num_episodes=85, num_iterations=4, max_steps_per_episode=150)
        torch.save({"joint": qmodel.Q_joint, "subgoal": qmodel.Q_subgoal}, f"project/static/policy/{elk_name}.pt")
    
    composed_policy, policy, safe_policy = task.policy_composition(qmodel)
    # task.test_policy(qmodel, restriction=starting_region, epsilon=0, visualize=True)
    # task.test_policy(elk_name, start_state=(11,11), epsilon=0, visualize=True)
    for i, p in enumerate([composed_policy]):
        room.draw_policy(p, fn=f"{task.name}_{i}")



