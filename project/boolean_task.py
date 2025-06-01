import numpy as np
import random
from reach_avoid_tabular import torch, Room, create_room, load_room, Image

class GoalOrientedBase:
    def __init__(self, room:Room, learning_rate=0.08, gamma=0.99, r_min=-1e9):
        """
        Initialize the Goal-Oriented Q-Learning algorithm for a 2D reach-avoid navigation task.
        
        Args:
            room: Room object
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration constant
            r_min: Lower-bound extended reward
        """
        self.env = room
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.r_min = r_min
        self.G = set()
        self.epsilon = 0.8
        self.decay_rate = 0.98
        self.mis_coverage_terrain = None

    def _random_condition(self):
        return not self.G or random.random() < self.epsilon
    
    def _train(self, *args):
        raise NotImplementedError()

    def _save_progress(self):
        pass
    
    def _add_goal(self, state):
        self.G.add(tuple(state.numpy()))

    def _remove_goal(self):
        goals = list(self.G)
        hidden_goals = []
        if self.mis_coverage_terrain is not None:
            i = 0
            while i < len(goals):
                if not self.mis_coverage_terrain[goals[i][0], goals[i][1]]:
                    hidden_goals.append(goals.pop(i))
                else:
                    i += 1
        return goals, hidden_goals
    
    def select_action(self, *args):
        raise NotImplementedError()

    def train_episodes(self, num_episodes=150, max_steps_per_episode=100):
        """Train the agent using Goal-Oriented Q-Learning."""
        epsilon = self.epsilon
        goal_regions = list(self.env.goals.values())
        for iteration in range(5):
            random.shuffle(goal_regions)
            for goal_region in goal_regions:
                done_rate = 0
                for episode in range(1,num_episodes+1):
                    # Initialize state
                    state = self.env.start()

                    steps = 0
                    while steps < max_steps_per_episode:
                        action = self.select_action(state, goal_region)
                        next_state, reward, done = self.env.step(action)

                        if done >= 2 and not goal_region[next_state[0], next_state[1]]:
                            done = 1
                            reward = 10

                        if done > 0:
                            # Add the current state to the set of subgoals
                            self._add_goal(next_state)
                        # Update Q-values for each subgoal
                        if self.G:
                            self._train(state, action, reward, next_state, done)
                        
                        if done >= 2: 
                            done_rate = (done_rate*episode+1) / (episode+1)
                            break
                        
                        state = next_state
                        steps += 1     
                    else:
                        done_rate = (done_rate*episode) / (episode+1)
                    
                    self.epsilon = max(self.epsilon * self.decay_rate, 0.05)
                
                print(f"Iteration {iteration} Goal switch why {done_rate}, epsilon {self.epsilon:.2f} completed with {len(self.G)} goals")
                    
        self.epsilon = epsilon
    
    def q_compose(self, mask):
        pass
    
class GoalOrientedQLearning(GoalOrientedBase):
    def __init__(self, room:Room, pretrained=False):
        super().__init__(room)
        # Action space: 8 directions (N, NE, E, SE, S, SW, W, NW)
        self.actions = list(range(room.n_actions))        
        # Q-table: state, subgoal, action -> value
        # State is (x, y), subgoal is (x, y)
        self.Q = torch.zeros(room.shape+room.shape+(room.n_actions,)) if not pretrained else torch.load(f"project/static/goal-q.pt")
    
    def _train(self, state, action, reward, next_state, done):
        """Get Q-value for a state-subgoal-action triple."""
        goal_tensor = torch.IntTensor(list(self.G))
        sx, sy = tuple(state[i].expand(goal_tensor.shape[0]) for i in range(2))
        action_tensor = torch.IntTensor([action]).expand(goal_tensor.shape[0])
        nx, ny = tuple(next_state[i].expand(goal_tensor.shape[0]) for i in range(2))
        current_q = self.Q[sx, sy, goal_tensor[:, 0], goal_tensor[:, 1], action_tensor]

        if done > 0:
            delta = torch.zeros_like(current_q)-1e-3
            goal_eq = -1
            for i in range(goal_tensor.shape[0]):
                if torch.equal(next_state, goal_tensor[i]):
                    goal_eq = i
                    break

            if goal_eq == -1:
                raise ValueError("Goal not found in subgoals, why training?")
            delta[goal_eq] = reward - current_q[goal_eq]
        else:
            next_q = self.Q[nx, ny, goal_tensor[:, 0], goal_tensor[:, 1]]
            max_next_q = next_q.max(1).values
            delta = reward + self.gamma * max_next_q - current_q
            
        new_q = current_q + self.learning_rate * delta
        self.Q[sx, sy, goal_tensor[:, 0], goal_tensor[:, 1], action_tensor] = new_q

    def _save_progress(self):
        torch.save(self.Q, f"project/static/goal-q.pt")
        torch.save(self.G, f"project/static/subgoals.pt")
    
    def select_action(self, state, goal_region):
        '''Select the best action based on currently explored goals'''
        if self._random_condition():
            return random.choice(self.actions)
         # Choose the best action based on current goals
        visible_goals = list((x,y) for x,y in self.G if goal_region[x,y])
        if not visible_goals:
            return random.choice(self.actions)
        goal_tensor = torch.IntTensor(visible_goals)
        sx, sy = tuple(state[i].expand(goal_tensor.shape[0]) for i in range(2))
        qs = self.Q[sx, sy, goal_tensor[:, 0], goal_tensor[:, 1]]
        max_av = torch.max(qs, 0).values
        action_values = {}
        for a in self.actions:
            a_val = max_av[a].item()
            if a_val not in action_values:
                action_values[a_val] = [a]
            else:
                action_values[a_val].append(a)
        best_actions = action_values[max(action_values.keys())]
        if len(best_actions) == 1:
            return best_actions[0]
        return random.choice(best_actions)
            
    def q_compose(self, mask):
        all_goals = torch.IntTensor([r*self.env.shape[1]+c 
                                     for r in range(self.env.shape[0]) 
                                     for c in range(self.env.shape[1]) 
                                     if mask[r,c] > 0])
        q_subgoal = self.Q.permute(0,1,4,2,3).reshape(self.env.shape+(self.env.n_actions,-1)).index_select(3, all_goals)
        # q_subgoal = q_subgoal.max()+q_subgoal.min()-q_subgoal   
        return q_subgoal.max(dim=3).values
    
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

import torch.optim as optim
from collections import deque
from neuralnets import NAFNetwork, F

class GoalOrientedNAF(GoalOrientedBase):
    def __init__(self, room:Room, goal_resolution):
        super().__init__(room)
        self.goal_resolution = goal_resolution
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks
        self.q_network = NAFNetwork(room.state_dim, room.state_dim, room.action_dim).to(self.device)
        self.target_network = NAFNetwork(room.state_dim, room.state_dim, room.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network doesn't need gradients
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=65536)
        self.batch_size = 64
        
        # Set of discovered goals
        self.G = list[torch.Tensor]()
        
    def _update_target_network(self, tau=0.005):
        """Soft update target network: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def select_action(self, state:torch.Tensor):
        """Select action using epsilon-greedy policy with analytical maximum"""
        if self._random_condition():
            # Random action for exploration
            return 2*torch.rand(self.env.action_dim)-1
        
        with torch.no_grad():
            states = state.unsqueeze(0).expand(len(self.G),2).to(self.device)
            goals = torch.stack(self.G).to(self.device)
            _, mu, _, V = self.q_network(states, goals)
            best_q = torch.max(V, 0)

            return mu[best_q.indices.item()].cpu()
    
    def _add_goal(self, state:torch.Tensor):
        if self.G:
            all_goals = torch.stack(self.G)
            diff = all_goals - state
            dist = torch.sqrt(torch.square(diff[:, 0]) + torch.square(diff[:, 1]))
            if torch.any(dist < self.goal_resolution):
                return  # skip the current goal state as close enough to some other goals
        self.G.append(state)
    
    def _train(self, state, action, reward, next_state, done):
        """Update Q-network from experiences in memory"""
        for goal in self.G:
            self.memory.append((state, goal, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        states = []
        goals = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        
        for state, goal, action, reward, next_state, is_terminal in batch:
            states.append(state)
            goals.append(goal)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(is_terminal)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        goals = torch.stack(goals).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.Tensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        terminals = torch.Tensor(terminals).unsqueeze(1).to(self.device)
        
        # Get current Q-values
        current_q, _, _, _ = self.q_network(states, goals, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            # For each state-goal pair, compute V(s',g) for each goal in G
            target_q = rewards.clone()
            
            for i in range(len(batch)):
                if not terminals[i].item():  # Non-terminal state
                    # Convert goals set to tensor batch
                    if self.G:
                        goal_list = list(self.G)
                        goal_batch = torch.FloatTensor([g for g in goal_list]).to(self.device)
                        next_state_batch = next_states[i].unsqueeze(0).repeat(len(goal_list), 1)
                        
                        # Get value estimates for all goals
                        next_values, _, _, _ = self.target_network(next_state_batch, goal_batch)
                        max_next_v = torch.max(next_values)
                        
                        # Update target with discounted max value
                        target_q[i] += self.gamma * max_next_v
                    else:
                        # No goals discovered yet
                        pass
                else:  # Terminal state
                    # Check if terminal state equals goal
                    goal_tensor = goals[i]
                    next_state_tensor = next_states[i]
                    
                    # If this isn't the goal state, apply r_min penalty
                    if not torch.allclose(next_state_tensor, goal_tensor, atol=1e-3):
                        target_q[i] = torch.tensor([self.r_min], device=self.device)
        
        # Compute loss and update
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q, target_q)
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self._update_target_network()
        
        return loss.item()
    
    def save(self, filename):
        """Save model parameters"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        """Load model parameters"""
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def q_compose(self, mask):
        dgoals = torch.IntTensor(torch.stack(self.G))
        indices = torch.nonzero(mask[dgoals[:,0], dgoals[:,1]]).squeeze(1)
        all_goals = torch.stack(self.G).index_select(0, indices)
        return all_goals
    
    def view_goals(self):
        dgoals = torch.IntTensor(torch.stack(self.G))
        imarr = np.ones(self.env.shape, dtype=np.uint8)
        for loc in dgoals.numpy().tolist():
            imarr[loc[0], loc[1]] = 0
        
        Image.fromarray(imarr, "L").save("project/static/cont-goals.png")

# Example usage
if __name__ == "__main__":
    # Initialize the environment and agent
    agent = GoalOrientedQLearning(
        room=load_room("saved_disc", "9room.pt", 4),
        pretrained=False,
    )
    # agent = GoalOrientedNAF(
    #     room=load_room('saved_cont', 'road.pt', 0),
    #     goal_resolution=5
    # )
    
    # Train the agent
    agent.env.start()
    rewards = agent.train_episodes(num_episodes=501, max_steps_per_episode=40)
    mask = agent.env.goals["goal_2"]
    agent.visualize_policy_with_arrows(mask, "9room_goal_2")
    # agent.test_policy(mask, [8,1])
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