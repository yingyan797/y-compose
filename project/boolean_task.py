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
        self.epsilon = 0.8
        self.decay_rate = 0.98
        self.goal_regions = list(enumerate(self.env.goals.values()))

    def _random_condition(self):
        return random.random() < self.epsilon
    
    def _train(self, *args):
        raise NotImplementedError()

    def _save_progress(self):
        pass

    def _partition_goals(self):
        groups = []
        for gr, region in self.goal_regions:
            i = 0
            while i < len(groups):
                if torch.any(region & groups[i][0]):
                    mask, members = groups.pop(i)
                    groups.append((mask | region, members + [gr]))
                    break
                i += 1
            else:
                groups.append((region, [gr]))

        return groups

    def select_action(self, *args):
        raise NotImplementedError()

    def train_episodes(self, num_episodes=150, num_iterations=5, max_steps_per_episode=100):
        """Train the agent using Goal-Oriented Q-Learning."""
        epsilon = self.epsilon
        goal_regions = self.goal_regions
        # This section is for training elk to reach each goal iteratively.
        print(f"Beginning training non-goal starting points")
        for iteration in range(num_iterations):
            random.shuffle(goal_regions)
            for gr, goal_region in goal_regions:
                done_rate = 0
                for episode in range(1,num_episodes+1):
                    # Initialize state
                    state = self.env.start(restriction=torch.where(self.env.terrain==1, 1, 0))

                    steps = 0
                    # recent_goals = []
                    while steps < max_steps_per_episode:
                        action = self.select_action(state, gr)
                        next_state, reward, done = self.env.step(action)

                        all_reached_gr = []
                        training_finished = False
                        if done >= 2:
                            for i, mask in goal_regions:
                                if mask[next_state[0], next_state[1]]:
                                    all_reached_gr.append(i)
                            if not all_reached_gr:
                                raise ValueError("Goal not reached, why done >= 2? Should check the code.")
                            elif gr not in all_reached_gr:
                                # Although some goals are reached, none matches gr. Elk reward is reduced.
                                reward = -10
                            training_finished = True

                            # recent_goals = all_reached_gr
                        # elif recent_goals:
                        #     reward = -1
                        #     recent_goals = []

                        self._train(state, action, reward, next_state, all_reached_gr)
                        if training_finished: 
                            done_rate = (done_rate*episode+1) / (episode+1)
                            break
                        
                        state = next_state
                        steps += 1     
                    else:
                        done_rate = (done_rate*episode) / (episode+1)
                    
                    self.epsilon = max(self.epsilon * self.decay_rate, 0.05)
                print(f"Iteration {iteration} Goal switch why {done_rate}, epsilon {self.epsilon:.2f}")

        # This section is for training within goal starting points, reaching different goals or no goals are encouraged.
        print(f"Beginning training within goal starting points")
        # Create groups of overlapping goal regions
        goal_groups = self._partition_goals()
        for gmask, members in goal_groups:
            self.epsilon = epsilon  # reset epsilon
            self.env._first_restriction = True
            for iteration in range(num_iterations):
                random.shuffle(members)
                for m in members:
                    done_rate = 0
                    for episode in range(1,num_episodes+1):
                        state = self.env.start(restriction=gmask)
                        steps = 0
                        while steps < max_steps_per_episode:
                            action = self.select_action(state, m)
                            next_state, reward, done = self.env.step(action)

                            all_reached_gr = []
                            training_finished = True
                            if done >= 2:
                                for i, mask in goal_regions:
                                    if mask[next_state[0], next_state[1]]:
                                        all_reached_gr.append(i)
                                if not all_reached_gr:
                                    raise ValueError("Goal not reached, why done >= 2? Should check the code.")
                                elif m not in all_reached_gr:
                                    reward = 0
                                    training_finished = False
                                    all_reached_gr = []
                            else:
                                all_reached_gr = list(range(len(self.goal_regions)))
                                reward = -10

                            self._train(state, action, reward, next_state, all_reached_gr)
                            if training_finished: 
                                success = 1 if done >= 2 else 0
                                done_rate = (done_rate*episode+success) / (episode+1)
                                break

                            state = next_state
                            steps += 1     
                        else:
                            done_rate = (done_rate*episode) / (episode+1)
                    
                        self.epsilon = max(self.epsilon * self.decay_rate, 0.05)
                    print(f"Interior {iteration} Goal {m} why {done_rate}, epsilon {self.epsilon:.2f}")

        self.epsilon = epsilon  # reset epsilon eventually, training is done for elk

    def q_compose(self, mask):
        pass
    
class GoalOrientedQLearning(GoalOrientedBase):
    def __init__(self, room:Room, pretrained=False):
        super().__init__(room)
        # Action space: 8 directions (N, NE, E, SE, S, SW, W, NW)
        self.actions = list(range(room.n_actions))        
        # Q-table: state, subgoal, action -> value
        # State is (x, y), subgoal is goal region index
        self.Q = torch.zeros(room.shape+(len(self.goal_regions), room.n_actions,)) if not pretrained else torch.load(f"project/static/goal-q.pt")
    
    def _train(self, state, action, reward, next_state, done):
        """Get Q-value for a state-subgoal-action triple."""
        sx, sy = tuple(state[i].expand(len(self.goal_regions)) for i in range(2))
        action_tensor = torch.IntTensor([action]).expand(len(self.goal_regions))
        nx, ny = tuple(next_state[i].expand(len(self.goal_regions)) for i in range(2))
        current_q = self.Q[sx, sy, list(range(len(self.goal_regions))), action_tensor]

        if done:
            delta = torch.zeros_like(current_q)-1e-3
            delta[done] = reward - current_q[done]
        else:
            next_q = self.Q[nx, ny, list(range(len(self.goal_regions)))]
            max_next_q = next_q.max(1).values
            delta = reward + self.gamma * max_next_q - current_q
            
        new_q = current_q + self.learning_rate * delta
        self.Q[sx, sy, list(range(len(self.goal_regions))), action_tensor] = new_q

    def _save_progress(self):
        torch.save(self.Q, f"project/static/goal-q.pt")
        torch.save(self.G, f"project/static/subgoals.pt")
    
    def select_action(self, state, gr):
        '''Select the best action based on currently explored goals'''
        if self._random_condition():
            return random.choice(self.actions)
        if isinstance(gr, list):  # no subgoal, choose the best action for all subgoals
            if not gr:
                gr = list(range(len(self.goal_regions)))
            sx, sy = tuple(state[i].expand(len(gr)) for i in range(2))
            q_sg = torch.max(self.Q[sx, sy, gr], dim=0).values
        else:
            q_sg = self.Q[state[0], state[1], gr]
        action_values = {}
        for a in self.actions:
            a_val = q_sg[a].item()
            if a_val not in action_values:
                action_values[a_val] = [a]
            else:
                action_values[a_val].append(a)
        best_actions = action_values[max(action_values.keys())]
        if len(best_actions) == 1:
            return best_actions[0]
        return random.choice(best_actions)
            
    def q_compose(self, grs):
        q_subgoal = self.Q.permute(0,1,3,2)[:,:,:,grs]
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
        room=load_room("saved_disc", "overlap.pt", 4),
        pretrained=False,
    )
    print([group[1] for group in agent._partition_goals()])