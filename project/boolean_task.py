import numpy as np
import matplotlib.pyplot as plt
import random
from reach_avoid_tabular import torch, Room, create_room, load_room
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle

class GoalOrientedQLearning:
    def __init__(self, room:Room, pretrained=False, alpha=0.1, gamma=0.98, epsilon=0.1, fn=""):
        """
        Initialize the Goal-Oriented Q-Learning algorithm for a 2D reach-avoid navigation task.
        
        Args:
            grid_size: Size of the square grid environment
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration constant
            r_min: Lower-bound extended reward
        """
        self.env = room
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.r_min = -1e8
        
        # Action space: 8 directions (N, NE, E, SE, S, SW, W, NW)
        self.actions = list(range(8))
        
        # Q-table: state, subgoal, action -> value
        # State is (x, y), subgoal is (x, y)
        self.Q = torch.zeros(room.shape+room.shape+(8,)) if not pretrained else torch.load(f"project/static/{fn}goal-q.pt")
        # Set of visited states that can serve as subgoals
        self.G = set() if not pretrained else set((goal[0], goal[1]) for goal in torch.load(f"project/static/{fn}subgoals.pt"))
        
    def _q_learning_update(self, state, next_state, action, reward, done):
        """Get Q-value for a state-subgoal-action triple."""

        subgoals = torch.tensor(list(self.G), dtype=int)
        sx, sy = tuple(state[i].expand(subgoals.shape[0]) for i in range(2))
        action = torch.tensor(action, dtype=int).expand(subgoals.shape[0])
        nx, ny = tuple(next_state[i].expand(subgoals.shape[0]) for i in range(2))

        goal_neq = []
        for i in range(subgoals.shape[0]):
            if not torch.equal(next_state, subgoals[i]):
                goal_neq.append(i)
        sub_q = self.Q[sx, sy, subgoals[:, 0], subgoals[:, 1], action]

        if done > 0:
            delta = torch.tensor(reward) - sub_q
            delta[goal_neq] = self.r_min
        else:
            max_next_q = torch.max(self.Q[nx, ny, subgoals[:, 0], subgoals[:, 1]])
            delta = reward + self.gamma * max_next_q - sub_q

        new_q = sub_q + self.alpha * delta
        self.Q[sx, sy, subgoals[:, 0], subgoals[:, 1], action] = new_q
    
    def select_action(self, state, mask=None):
        """Select an action using epsilon-greedy policy."""
        if not self.G or random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Choose the best action based on current subgoals
        if mask is None:
            subgoals = torch.IntTensor(list(self.G))
        sx, sy = tuple(state[i].expand(subgoals.shape[0]) for i in range(2))
        qs = self.Q[sx, sy, subgoals[:, 0], subgoals[:, 1]]
        max_av = torch.max(qs, 0)
        action_values = {}
        for a in self.actions:
            if max_av.values[a] not in action_values:
                action_values[max_av.values[a]] = [a]
            else:
                action_values[max_av.values[a]].append(a)

        best_actions = action_values[max(action_values.keys())]
        if len(best_actions) == 1:
            return best_actions[0]
        return random.choice(best_actions)
        
    def train(self, num_episodes=1000, max_steps_per_episode=100, fn=""):
        """Train the agent using Goal-Oriented Q-Learning."""
        rewards_per_episode = []
        
        for episode in range(num_episodes):
            # Initialize state
            state = self.env.start()
                
            episode_reward = 0
            steps = 0
            
            while steps < max_steps_per_episode:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                
                # Update Q-values for each subgoal
                if self.G:
                    self._q_learning_update(state, next_state, action, reward, done)
                
                state = next_state
                steps += 1
                
                if done > 0:
                    # Add the current state to the set of subgoals
                    self.G.add((state[0].item(), state[1].item()))

            rewards_per_episode.append(episode_reward)
            if episode % 100 == 0:
                print(f"Episode {episode}, num goals {len(self.G)}, Avg Reward: {np.mean(rewards_per_episode[-100:]):.2f}")
                torch.save(self.Q, f"project/static/{fn}goal-q.pt")
                torch.save(self.G, f"project/static/{fn}subgoals.pt")
        
        return rewards_per_episode
    
    def parse_compose(self, instr:str):
        full_goals = torch.where(self.env.terrain==2, 1, 0)
        def task_and(masks):
            masks = torch.stack(masks)
            return torch.min(masks, dim=0).values
        def task_or(masks):
            masks = torch.stack(masks)
            return torch.min(masks, dim=0).values
        def task_not(mask):
            neg = 1-mask
            return torch.minimum(neg, full_goals)
        comp_functions = {"and": task_and, "or": task_or, "not": task_not}
        def parse_expr(expression:str) -> torch.Tensor:
            if expression.startswith("'") and expression.endswith("'"):
                return self.env.goals[expression[1:-1]]
        
            # Check for not, and, or operations
            for op in ["not", "and", "or"]:
                if expression.startswith(op + "(") and expression.endswith(")"):
                    # Extract the content inside the parentheses
                    content = expression[len(op)+1:-1]
                    
                    # For "not", we expect only one argument
                    if op == "not":
                        return task_not(parse_expr(content))
                    
                    # For "and" and "or", we split the arguments by comma
                    # but we need to handle nested expressions correctly
                    args = []
                    current_arg = ""
                    paren_level = 0
                    
                    for char in content:
                        if char == '(' and (current_arg.endswith("not") or 
                                        current_arg.endswith("and") or 
                                        current_arg.endswith("or")):
                            current_arg += char
                            paren_level += 1
                        elif char == '(':
                            current_arg += char
                            paren_level += 1
                        elif char == ')':
                            current_arg += char
                            paren_level -= 1
                        elif char == ',' and paren_level == 0:
                            args.append(current_arg.strip())
                            current_arg = ""
                        else:
                            current_arg += char
                    
                    if current_arg.strip():
                        args.append(current_arg.strip())
                    
                    # Parse each argument recursively
                    return comp_functions[op]([parse_expr(arg) for arg in args])
            raise SyntaxError("Not recognizable compose instructions")
        
        return parse_expr(instr)
            
    def q_compose(self, mask):
        all_goals = torch.IntTensor([r*self.env.shape[1]+c 
                                     for r in range(self.env.shape[0]) 
                                     for c in range(self.env.shape[1]) 
                                     if mask[r,c] > 0])
        q_subgoal = self.Q.permute(0,1,4,2,3).reshape(self.env.shape+(8,-1)).index_select(3, all_goals)
        return q_subgoal.max(dim=3).values

    def visualize_policy_with_arrows(self, mask, fn="policy"):
        """
        Visualize a policy grid using directional arrows.
        
        Args:
            
        """
        policy = self.q_compose(mask).max(2)
        policy_grid = policy.indices.numpy()
        value = policy.values.numpy()
        vl, vu = value.min(), value.max()
        if vl != vu:
            value = (value-vl) / (vu-vl)

        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set background color and limits
        ax.set_facecolor('white')
        ax.set_xlim(-0.5, policy_grid.shape[1]-0.5)
        ax.set_ylim(-0.5, policy_grid.shape[0]-0.5)
        
        # Draw grid lines
        for i in range(policy_grid.shape[0]+1):
            ax.axhline(i-0.5, color='gray', linestyle='-', alpha=0.3)
        for i in range(policy_grid.shape[1]+1):
            ax.axvline(i-0.5, color='gray', linestyle='-', alpha=0.3)

        # Define arrow directions (dx, dy) for each action
        # Action directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        directions = {
            0: (0, 0.4),    # North (up)
            1: (0.3, 0.3),  # Northeast
            2: (0.4, 0),    # East (right)
            3: (0.3, -0.3), # Southeast
            4: (0, -0.4),   # South (down)
            5: (-0.3, -0.3),# Southwest
            6: (-0.4, 0),   # West (left)
            7: (-0.3, 0.3)  # Northwest
        }
        
        # Define colors for different elements
        goal_color = 'lightgreen'
        terminal_color = 'yellow'
        obstacle_color = 'black'
        for x in range(policy_grid.shape[1]):
            for y in range(policy_grid.shape[0]):
                if self.env.terrain[y,x] == 0:
                    ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor=obstacle_color, alpha=0.7))
                    continue
                elif mask[y][x] > 0:
                    ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor=goal_color, alpha=0.7))
                elif self.env.terrain[y,x] == 2:
                    ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor=terminal_color, alpha=0.7))
                
                # Draw arrows for each cell in the grid
                action = policy_grid[y, x]
                    
                # Check if this is a valid action
                if action in directions:
                    dx, dy = directions[action]
                    v = value[y,x]
                    arrow_color = (v, 0, 1-v)
                    # Create arrow
                    arrow = Arrow(x, y, dx, -dy, width=0.3, color=arrow_color, alpha=(v+0.2)/1.2)
                    ax.add_patch(arrow)
        
        # Add a legend for directions
        legend_elements = []
        direction_names = ['North', 'Northeast', 'East', 'Southeast', 
                        'South', 'Southwest', 'West', 'Northwest']
        
        # Set axis labels and title
        ax.set_xlabel('Column (x)')
        ax.set_ylabel('Row (y)')
        ax.invert_yaxis()
        ax.set_title('Policy Visualization with Direction Arrows')
        
        plt.tight_layout()
        plt.savefig(f"project/static/policy/{fn}.png")
    
    def test_policy(self, mask, start_state=None, max_steps=200):
        """Test the learned policy from a given start state."""
        self.env.start(start_state, None)
        total_reward = 0
        steps = 0
        policy = self.q_compose(mask).max(2).indices
        while steps < max_steps:
            if random.random() < self.epsilon:
                action = random.choice(self.actions)
            else:
                action = policy[self.env.loc[0], self.env.loc[1]]
            next_state, reward, done = self.env.step(action, True)
            total_reward += reward
            steps += 1
            
            if mask[self.env.loc[0], self.env.loc[1]] > 0:
                print(f"Reached goal position {self.env.loc}")
                break
        
        print(f"Test completed: {steps} steps, Total reward: {total_reward:.2f}")
        self.env.visual()

# Example usage
if __name__ == "__main__":
    # Initialize the environment and agent
    agent = GoalOrientedQLearning(
        room=load_room('20250425142229.json'),
        pretrained=True,
    )
    
    # Train the agent
    agent.env.start()
    # rewards = agent.train(num_episodes=401, max_steps_per_episode=100)
    # mask = agent.parse_compose("and(not('goal-2'), 'goal-1')")
    mask = agent.parse_compose("not(and('goal-1', 'goal-2'))")
    agent.visualize_policy_with_arrows(mask, "not-(g1 and g2")
    agent.test_policy(mask, [8,1])
    # # Plot learning curve
    # plt.figure(figsize=(10, 5))
    # plt.plot(rewards)
    # plt.title('Learning Curve')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.grid(True)
    # plt.savefig('learning_curve.png')
    # plt.show()
    
    # # Visualize the learned policy
    # agent.visualize_policy()
    
    # # Test the policy from different starting points
    # test_starts = [(0, 0), (15, 0), (29, 0)]
    # for start in test_starts:
    #     print(f"\nTesting from start position: {start}")
    #     agent.test_policy(start)