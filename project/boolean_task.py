import numpy as np
import random
from reach_avoid_tabular import torch, Room, create_room, load_room, Image
import matplotlib.pyplot as plt

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
        self.Q_joint = None
        self.Q_subgoal = None

    def _random_condition(self):
        return random.random() < self.epsilon
    
    def _train_jointq(self, *args):
        raise NotImplementedError()
    def _train_subgoalq(self, *args):
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

        def find_all_reached_gr(next_state):
            all_reached_gr = []
            for i, mask in self.goal_regions:
                if mask[next_state[0], next_state[1]]:
                    all_reached_gr.append(i)
            if not all_reached_gr:
                print(mask, next_state)
                raise ValueError("Goal not reached, why done >= 2? Function shouldn't be called.")
            return all_reached_gr   

        def training_group(next_state, done, gr):
            all_reached_gr = list(range(len(self.goal_regions)))
            training_finished = True
            if done >= 2:
                all_reached_gr = find_all_reached_gr(next_state)
                if gr not in all_reached_gr:
                    reward = 0
                    training_finished = False
                    all_reached_gr = []
                else:
                    reward = 100
            else:
                reward = -10
            return reward, all_reached_gr, training_finished
        
        def training_nongoal(next_state, done, gr):
            all_reached_gr = []
            training_finished = False
            if done >= 2:
                all_reached_gr = find_all_reached_gr(next_state)
                if gr not in all_reached_gr:
                    reward = -1
                else:
                    reward = 100
                training_finished = True
            else:
                reward = 0 if done == 1 else -10

            return reward, all_reached_gr, training_finished
        
        self.partition = groups
        return [g + (training_group, ) for g in groups if len(g[1]) > 1] + [
            (torch.where(self.env.terrain==1, 1, 0), list(range(len(self.goal_regions))), training_nongoal)]

    def select_action(self, *args):
        raise NotImplementedError()

    def train_episodes(self, num_episodes=150, num_iterations=5, max_steps_per_episode=100):
        """Train the agent using Goal-Oriented Q-Learning."""
        import time
        epsilon = self.epsilon
        goal_regions = self.goal_regions
        # This section is for training elk to reach each goal iteratively.
        starting_time = time.time()
        print(f"Beginning training non-goal starting points at {starting_time}")
        subgoal_episodes = num_episodes*3
        why_done_subgoal = np.zeros((num_iterations, len(goal_regions), subgoal_episodes), dtype=np.bool_)
        for iteration in range(1, num_iterations+1):
            random.shuffle(goal_regions)
            for gr, goal_region in goal_regions:
                done_rate = 0
                for episode in range(1,subgoal_episodes+1):
                    state = self.env.start()
                    steps = 0
                    while steps < max_steps_per_episode:
                        action = self.select_action(self.Q_subgoal, state, gr)
                        next_state, done = self.env.step(action)
                        reward = 0
                        training_finished = False
                        if done >= 2:
                            if goal_region[tuple(next_state)]:
                                reward = 100
                                training_finished = True
                            else:
                                done = 0
                        elif done == 0:  # Reaching an obstacle
                            reward = -10

                        self._train_subgoalq(state, action, reward, next_state, gr)
                        if training_finished: 
                            done_rate = (done_rate*episode+1) / (episode+1)
                            why_done_subgoal[iteration-1, gr, episode-1] = 1
                            break
                        state = next_state
                        steps += 1     
                    else:
                        done_rate = (done_rate*episode) / (episode+1)
                    
                    self.epsilon = max(self.epsilon * self.decay_rate, 0.05)
                print(f"Iteration {iteration} Goal switch why {done_rate}, epsilon {self.epsilon:.2f}")
        
        
        # This section is for training within goal starting points, reaching different goals or no goals are encouraged.
        print(f"Beginning training within goal starting points in {time.time()-starting_time}")
        # Create groups of overlapping goal regions
        goal_groups = self._partition_goals()
        why_done_joint = np.zeros((len(goal_groups), num_iterations*2))
        for g, (gmask, members, training_function) in enumerate(goal_groups):
            if len(members) == 1:
                continue
            self.epsilon = epsilon  # reset epsilon
            self.env._first_restriction = True
            for iteration in range(1, num_iterations*2+1):
                random.shuffle(members)
                n_success = 0
                for m in members:
                    done_rate = 0
                    for episode in range(1,num_episodes+1):
                        state = self.env.start(restriction=gmask)
                        steps = 0
                        while steps < max_steps_per_episode:
                            action = self.select_action(self.Q_joint, state, m)
                            next_state, done = self.env.step(action)

                            reward, all_reached_gr, training_finished = training_function(next_state, done, m)
                            self._train_jointq(state, action, reward, next_state, all_reached_gr)
                            if training_finished: 
                                success = 1 if reward > 0 else 0
                                n_success += success
                                done_rate = (done_rate*episode+success) / (episode+1)
                                break

                            state = next_state
                            steps += 1     
                        else:
                            done_rate = (done_rate*episode) / (episode+1)
                    
                        self.epsilon = max(self.epsilon * self.decay_rate, 0.05)
                    print(f"Interior {iteration} Goal {m} why {done_rate}, epsilon {self.epsilon:.2f}")
                why_done_joint[g, iteration-1] = n_success/(len(members)*num_episodes)

        self.epsilon = epsilon  # reset epsilon eventually, training is done for elk
        print(f"Training completed in {time.time()-starting_time}")
        return why_done_subgoal, why_done_joint

    def plot_training_results(self, why_done_subgoal, why_done_joint, fn="training"):
        # subgoal_success = np.sum(why_done_subgoal, axis=2)/why_done_subgoal.shape[2]    # 2D heat map
        subgoal_success = why_done_subgoal.transpose(0, 2, 1).reshape(-1, why_done_subgoal.shape[1])
        joint_success = why_done_joint  # 2D line plots with each row representing one line

        # Create figure with two subplots stacked horizontally (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot subgoal training results on left subplot
        # for gr, goal in enumerate(self.env.goals.keys()):
        #     ax1.plot(range(1, subgoal_success.shape[0] + 1),
        #             subgoal_success[:, gr],
        #             marker='o',
        #             label=f'{goal}')
        ax1.set_xlabel('Iteration * episodes')
        ax1.set_ylabel('Goal Number')
        # ax1.set_xticks(range(0, subgoal_success.shape[0]))["O", "S1", "S2", "S3"]
        colors = ["beige", "blue", "purple"]
        shapes = ["square", "circle"]
        ax1.set_yticks(range(0, subgoal_success.shape[1]), labels=colors+shapes)
        ax1.set_title('Directed Policy training success rate')
        # ax1.legend()
        ax1.imshow(subgoal_success.T, aspect='auto', cmap='viridis')
        # ax1.grid(True)

        # Plot joint training results on right subplot
        for group in range(joint_success.shape[0]):
            ax2.plot(range(1, joint_success.shape[1] + 1),
                    joint_success[group, :],
                    marker='o',
                    label=f'Group {group}')
        ax2.set_xlabel('Iteration')
        ax2.set_xticks(range(1, joint_success.shape[1]+1))
        ax2.set_ylabel('Success rate')
        ax2.set_title('Safe Policy Training success rate')
        ax2.grid(axis='y')
        ax2.legend()

        # Adjust spacing between subplots
        plt.tight_layout()
        
        # Save the figure BEFORE closing it
        plt.savefig(f"project/static/training/{fn}.png")
        plt.close()

    def q_compose(self, mask):
        pass
    
class GoalOrientedQLearning(GoalOrientedBase):
    def __init__(self, room:Room):
        super().__init__(room)
        # Action space: 8 directions (N, NE, E, SE, S, SW, W, NW)
        self.actions = list(range(room.n_actions))        
        # Q-table: state, subgoal, action -> value
        # State is (x, y), subgoal is goal region index
        self.Q_joint = torch.zeros(room.shape+(len(self.goal_regions), room.n_actions,))
        self.Q_subgoal = torch.zeros(room.shape+(len(self.goal_regions), room.n_actions,))
    
    def _train_jointq(self, state, action, reward, next_state, done):
        """Get Q-value for a state-subgoal-action triple."""
        num_goals = len(self.goal_regions)
        states = tuple([state[i]]*num_goals for i in range(self.env.state_dim))
        actions = [action] * num_goals
        next_states = tuple([next_state[i]]*num_goals for i in range(self.env.state_dim))
        current_q = self.Q_joint[states + (list(range(num_goals)), actions)]

        if done:
            delta = torch.zeros_like(current_q)-1e-3
            delta[done] = reward - current_q[done]
        else:
            next_q = self.Q_joint[next_states + (list(range(num_goals)),)]
            max_next_q = next_q.max(1).values
            delta = reward + self.gamma * max_next_q - current_q
            
        new_q = current_q + self.learning_rate * delta
        self.Q_joint[states + (list(range(num_goals)), actions)] = new_q

    def _train_subgoalq(self, state, action, reward, next_state, done:int):
        num_goals = len(self.goal_regions)
        states = tuple([state[i]]*num_goals for i in range(self.env.state_dim))
        actions = [action] * num_goals
        current_q = self.Q_subgoal[states + (list(range(num_goals)), actions)]
        delta = torch.zeros_like(current_q)
        if reward > 0:
            delta[done] = reward - current_q[done]
        else:
            next_q = self.Q_subgoal[tuple(next_state) + (done,)]
            max_next_q = next_q.max()
            delta[done] = reward + self.gamma * max_next_q - current_q[done]
        new_q = current_q + self.learning_rate * delta
        self.Q_subgoal[states + (list(range(num_goals)), actions)] = new_q
    
    def select_action(self, q, state, gr):
        '''Select the best action based on q, q can be joint or subgoal'''
        if self._random_condition():
            return random.choice(self.actions)

        q_sg = q[tuple(state) + (gr,)]
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
            
    def q_compose(self, q, grs):
        # q can be joint or subgoal
        q_subgoal = q.permute(0,1,3,2)[:,:,:,grs]
        return q_subgoal.max(dim=3).values

import torch.optim as optim
from collections import deque
from neuralnets import NAFNetwork, F

class GoalOrientedNAF(GoalOrientedBase):
    # Not used extension, maybe useful for continuous tasks
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
    d = Image.open("project/static/training/exmp2.png")
    s = Image.open("project/static/training/mat.png")
    # i = Image.open("project/static/training/od3.png")
    images = [d,s]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save('project/static/training/dfaexp.png')
