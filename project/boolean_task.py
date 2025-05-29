import numpy as np
import random
from reach_avoid_tabular import torch, Room, create_room, load_room, Image

class GoalOrientedBase:
    def __init__(self, room:Room, learning_rate=0.1, gamma=0.99, r_min=-1e9):
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
        self.epsilon = 0.5
        self.r_min = r_min
        self.G = [set() for _ in range(len(room.goals))] 
        self.goal_regions = []
        self.goal_ids = []
        for goal_id, goal_region in room.goals.items():
            if goal_region.any():
                self.goal_ids.append(goal_id)
                self.goal_regions.append(goal_region)
        print(self.goal_ids)


    def _random_condition(self, goal_id):
        return not self.G[goal_id] or random.random() < self.epsilon
    
    def _train(self, *args):
        raise NotImplementedError()

    def _save_progress(self):
        pass
    
    def _get_all_goals(self):
        all_goals = set()
        for g in self.G:
            all_goals.update(g)
        return all_goals
    
    def _add_goal(self, state):
        state = tuple(state.numpy())
        if state not in self._get_all_goals():
            for i, goal_region in enumerate(self.goal_regions):
                if goal_region[state[0], state[1]]:
                    self.G[i].add((state[0], state[1]))

    def select_action(self, *args):
        raise NotImplementedError()

    def train_episodes(self, num_iterations=10, num_episodes=1000, max_steps_per_episode=100, fn=""):
        """Train the agent using Goal-Oriented Q-Learning."""
        self.epsilon = 0.1
        for iteration in range(num_iterations):
            for goal_id in random.sample(range(len(self.goal_ids)), len(self.goal_ids)):
                gmask = self.goal_regions[goal_id]
                for episode in range(num_episodes):
                    # Initialize state
                    state = self.env.start()
                    steps = 0
                    
                    while steps < max_steps_per_episode:
                        action = self.select_action(state, goal_id)
                        next_state, reward, done = self.env.step(action)
                        # Update Q-values for each subgoal
                        if done > 0:
                            # Add the current state to the set of subgoals
                            self._add_goal(next_state)
                        if self.G[goal_id]:
                            self._train(state, action, reward, next_state, done, goal_id)
                        
                        state = next_state
                        steps += 1

                        if gmask[next_state[0], next_state[1]]:
                            break   # Reach goal eventually, no need to continue (mask goal reached)
                    # self.epsilon = max(0.05, self.epsilon*0.99) # decay epsilon
                    
            print(f"Iteration {iteration}, gnames: {goal_id}, num goals {[len(g) for g in self.G]}")
            # self._save_progress()
    
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
    
    def _train(self, state, action, reward, next_state, done, goal_id):
        """Get Q-value for a state-subgoal-action triple."""
        goals = torch.IntTensor(list(self._get_all_goals()))
        sx, sy = tuple(state[i].expand(goals.shape[0]) for i in range(2))
        action = torch.IntTensor([action]).expand(goals.shape[0])
        nx, ny = tuple(next_state[i].expand(goals.shape[0]) for i in range(2))

        sub_q = self.Q[sx, sy, goals[:, 0], goals[:, 1], action]

        def nongoal_update(r):
            mask_goals = torch.IntTensor(list(self.G[goal_id]))
            nx, ny = tuple(next_state[i].expand(mask_goals.shape[0]) for i in range(2))
            max_next_q = torch.max(self.Q[nx, ny, mask_goals[:, 0], mask_goals[:, 1]])
            delta = r + self.gamma * max_next_q - sub_q
            return delta
        
        if done > 0:
            goal_eq = -1
            for i in range(goals.shape[0]):
                if torch.equal(next_state, goals[i]):
                    goal_eq = i
                    break
            if goal_eq < 0:
                raise ValueError("No goal reached during training, error in goal identification")
            if tuple(next_state.numpy()) in self.G[goal_id]:
                delta = reward - sub_q
                goal_neq = list(range(goals.shape[0]))
                goal_neq.remove(goal_eq)
                delta[goal_neq] = self.r_min
            else:
                delta = torch.zeros_like(sub_q)
                delta[goal_eq] = reward - sub_q[goal_eq]
        else:
            delta = nongoal_update(reward)

        new_q = sub_q + self.learning_rate * delta
        self.Q[sx, sy, goals[:, 0], goals[:, 1], action] = new_q

    def _save_progress(self):
        torch.save(self.Q, f"project/static/goal-q.pt")
        torch.save(self.G, f"project/static/subgoals.pt")
    
    def select_action(self, state, goal_id):
        '''Select the best action based on currently explored goals'''
        if self._random_condition(goal_id):
            return random.choice(self.actions)
         # Choose the best action based on current goals
        goals = torch.IntTensor(list(self.G[goal_id]))
        sx, sy = tuple(state[i].expand(goals.shape[0]) for i in range(2))
        qs = self.Q[sx, sy, goals[:, 0], goals[:, 1]]
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