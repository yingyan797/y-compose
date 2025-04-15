import numpy as np
import matplotlib.pyplot as plt
import random
from reach_avoid_tabular import torch, Room, create_room
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle

class GoalOrientedQLearning:
    def __init__(self, room:Room, pretrained=False, alpha=0.1, gamma=0.99, epsilon=0.1):
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
        self.Q = torch.zeros(room.shape+room.shape+(8,)) if not pretrained else torch.load("project/static/goal-q.pt")
        # Set of visited states that can serve as subgoals
        self.G = set()
        
    def get_q_value(self, state, subgoal, action):
        """Get Q-value for a state-subgoal-action triple."""
        return self.Q[state[0], state[1], subgoal[0], subgoal[1], action].item()
    
    def update_q_value(self, state, subgoal, action, value):
        """Update Q-value for a state-subgoal-action triple."""
        self.Q[state[0], state[1], subgoal[0], subgoal[1], action] = value
    
    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if not self.G or random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Choose the best action based on current subgoals
        action_values = {}
        for action in self.actions:
            max_q_value = float('-inf')
            for subgoal in self.G:
                q_value = self.get_q_value(state, subgoal, action)
                if q_value > max_q_value:
                    max_q_value = q_value
            
            if max_q_value not in action_values:
                action_values[max_q_value] = [action]
            else:
                action_values[max_q_value].append(action)
        best_actions = action_values[max(action_values.keys())]
        return best_actions[0] if len(best_actions) == 1 else random.choice(best_actions)
    
    def train(self, num_episodes=1000, max_steps_per_episode=1000):
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
                for subgoal in self.G:
                    if done > 0:
                        if not torch.equal(next_state, subgoal):
                            delta = self.r_min
                        else:
                            delta = reward - self.get_q_value(state, subgoal, action)
                    else:
                        # Calculate maximum Q-value for the next state and this subgoal
                        max_next_q = max([self.get_q_value(next_state, subgoal, a) for a in self.actions], default=0)
                        delta = reward + self.gamma * max_next_q - self.get_q_value(state, subgoal, action)
                    
                    # Update Q-value
                    new_q = self.get_q_value(state, subgoal, action) + self.alpha * delta
                    self.update_q_value(state, subgoal, action, new_q)
                
                state = next_state
                steps += 1
                
                if done > 0:
                    # Add the current state to the set of subgoals
                    self.G.add(state)
                    break

            rewards_per_episode.append(episode_reward)
            if episode % 20 == 0:
                print(f"Episode {episode}, status {done}|{steps}|{state}, Avg Reward: {np.mean(rewards_per_episode[-100:]):.2f}")
                torch.save(self.Q, "project/static/goal-q.pt")
        
        return rewards_per_episode
    
    def q_compose(self, mask):
        subgoals = [r*self.env.shape[1]+c for r in range(self.env.shape[0]) for c in range(self.env.shape[1]) if mask[r, c] > 0]
        return self.Q.permute(0,1,4,2,3).reshape(self.env.shape+(8,-1)).index_select(3, torch.tensor(subgoals, dtype=torch.int)).max(dim=3).values

    def visualize_policy_with_arrows(self, mask):
        """
        Visualize a policy grid using directional arrows.
        
        Args:
            policy_grid: An n x n numpy array where each cell contains an integer 0-7 representing a direction
            goal: Tuple (x, y) indicating the goal position
            obstacles: Set of (x, y) tuples representing obstacle positions
            danger_zones: Set of (x, y) tuples representing danger zone positions
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
        goal_color = 'green'
        void_colors = ['black', '', 'blue']
        for x in range(policy_grid.shape[1]):
            for y in range(policy_grid.shape[0]):
                if mask[y][x] > 0:
                    ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor=goal_color, alpha=0.7))
                    continue
                elif self.env.terrain[y,x] != 1:
                    ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor=void_colors[self.env.terrain[y,x]], alpha=0.7))
                    continue
                
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
        plt.savefig("project/static/policy.png")
    
    def test_policy(self, start_state=None, max_steps=200):
        """Test the learned policy from a given start state."""
        if start_state is None:
            start_state = (0, 0)  # Default to bottom-left corner
        
        state = start_state
        path = [state]
        total_reward = 0
        steps = 0
        
        while not self.is_terminal(state) and steps < max_steps:
            action = self.select_action(state)
            next_state, reward, done = self.step(state, action)
            total_reward += reward
            state = next_state
            path.append(state)
            steps += 1
            
            if done:
                break
        
        print(f"Test completed: {len(path)} steps, Total reward: {total_reward:.2f}")
        print(f"Reached goal: {state == self.goal_state}")
        
        # Visualize the path
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        
        plt.figure(figsize=(10, 10))
        plt.plot(y_coords, x_coords, 'b-', linewidth=2)
        plt.plot(y_coords, x_coords, 'bo', markersize=6)
        plt.plot(start_state[1], start_state[0], 'go', markersize=12)
        plt.plot(self.goal_state[1], self.goal_state[0], 'r*', markersize=15)
        
        # Mark obstacles and danger zones
        for x, y in self.obstacles:
            plt.plot(y, x, 'ks', markersize=10)
        
        for x, y in self.danger_zones:
            plt.plot(y, x, 'rx', markersize=6, alpha=0.3)
        
        plt.grid(True)
        plt.xlim(-1, self.grid_size)
        plt.ylim(-1, self.grid_size)
        plt.gca().invert_yaxis()  # To match grid coordinates
        plt.title('Test Path')
        plt.tight_layout()
        plt.savefig('test_path.png')
        plt.show()
        
        return path, total_reward

# Example usage
if __name__ == "__main__":
    # Initialize the environment and agent
    agent = GoalOrientedQLearning(
        room=create_room(),
        pretrained=True,
        alpha=0.1, 
        gamma=0.99, 
        epsilon=0.1, 
    )
    
    # Train the agent
    agent.env.start()
    # rewards = agent.train(num_episodes=100, max_steps_per_episode=500)\
    agent.visualize_policy_with_arrows(agent.env.goals["Danger 1"])
    
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