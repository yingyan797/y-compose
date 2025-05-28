import torch, random, os, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle

class Room:
    def __init__(self, height=30, width=30, n_actions=8):
        self.shape = (height, width)
        self.state_dim = 2
        self.action_dim = 1
        self.n_actions = n_actions
        self.base = torch.ones((height, width), dtype=torch.bool)
        self._reward_lev = 100
        self.goals = dict[str, torch.BoolTensor]()
        self.always = None
        self.terrain = None
        self.loc = torch.zeros(2, dtype=torch.int)

    def visual(self, animate=True):
        canvas = self.terrain.numpy().astype(np.uint8)*80
        magnify = 1+int(1000/canvas.shape[0])
        imarr = np.repeat(np.repeat(canvas, magnify, axis=1), magnify, axis=0)
        Image.fromarray(imarr, mode="L").save("project/static/room-terrain.png")            
        if animate and len(self._trace) > 1:
            frames = []
            for loc in self._trace:
                frame = np.copy(canvas)
                frame[int(loc[0].item()), int(loc[1].item())] = [0, 200, 200]
                frames.append(Image.fromarray(frame))
            frames[0].save("project/static/trace-animate.gif", save_all=True, append_images=frames[1:])
            
    def start(self, start_state=None, restriction=None):
        def to_tensor(loc):
            return torch.IntTensor(loc) if self.n_actions else torch.FloatTensor(loc)

        if self.terrain is None:
            base = self.base.to(torch.uint8)*2
            masks = [goal.to(torch.uint8)+1 for goal in self.goals.values()]
            self.terrain = torch.minimum(base, torch.max(torch.stack(masks), dim=0).values) + self.always.to(torch.uint8)
            self._avail_locs = torch.nonzero(self.terrain).numpy().tolist()
        if start_state is not None:
            loc = to_tensor(start_state)
        elif restriction is not None:
            region = [(r,c) for r,c in self._avail_locs if restriction[r,c] > 0]
            loc = to_tensor(random.choice(region))
        else:
            loc = to_tensor(random.choice(self._avail_locs))

        self.loc = loc
        self._trace = [self.loc]
        return self.loc

    def step(self, action, trace=False):
        new_loc = [0,0]
        label = 0
        if self.n_actions:
            match action:   # 0..7
                case 0:
                    drn = [-1, 0]
                case 4: 
                    drn = [-1, 1]
                case 1:
                    drn = [0, 1]
                case 5:
                    drn = [1, 1]
                case 2:
                    drn = [1, 0]
                case 6:
                    drn = [1, -1]
                case 3:
                    drn = [0, -1]
                case 7:
                    drn = [-1, -1]
                case _:
                    raise ValueError("Not recognized action")
            new_loc = self.loc + torch.IntTensor(drn)
            row_in = new_loc[0] in range(self.shape[0])
            col_in = new_loc[1] in range(self.shape[1])
            if not (row_in and col_in):
                return self.loc, -1, 0
            label = self.terrain[new_loc[0], new_loc[1]]
            if not label:
                return self.loc, -1, 0
        else:
            # -1, 1
            ang = torch.squeeze(action * torch.pi)
            new_loc = self.loc + 10*torch.stack([-torch.cos(ang), torch.sin(ang)])
            row_in = new_loc[0] >= 0 and new_loc[0] <= self.shape[0]
            col_in = new_loc[1] >= 0 and new_loc[1] <= self.shape[1]
            if not (row_in and col_in):
                return self.loc, -1, 0
            label = self.terrain[int(new_loc[0].item()), int(new_loc[1].item())]
            if not label:
                return self.loc, -1, 0

        self.loc = new_loc
        if trace:
            self._trace.append(new_loc)

        if label >= 2:
            return new_loc, self._reward_lev, label
        return new_loc, -0.01, 0     # -0.01 for no goal reached
    
    def draw_policy(self, q_values, mask=None, fn="policy"):
        """
        Visualize a policy grid using directional arrows.
        
        Args:
            q_values: A tensor of shape (height, width, n_actions) containing the Q-values for each action.

        """
        policy = q_values.max(2)
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
            4: (0.3, 0.3),  # Northeast
            1: (0.4, 0),    # East (right)
            5: (0.3, -0.3), # Southeast
            2: (0, -0.4),   # South (down)
            6: (-0.3, -0.3),# Southwest
            3: (-0.4, 0),   # West (left)
            7: (-0.3, 0.3)  # Northwest
        }
        
        # Define colors for different elements
        goal_color = 'lightgreen'
        terminal_color = 'yellow'
        obstacle_color = 'black'
        for x in range(policy_grid.shape[1]):
            for y in range(policy_grid.shape[0]):
                if self.terrain[y,x] == 0:
                    ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor=obstacle_color, alpha=0.7))
                    continue
                elif mask is not None and mask[y][x] > 0:
                    ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor=goal_color, alpha=0.7))
                elif self.terrain[y,x] >= 2:
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

def create_room(name):
    match name:
        case "custom default":
            h, w = 16, 16
            room = Room(h, w)
            room.base[2:7, 8:9] = 0
            room.base[9:14, 8:9] = 0
            o0 = torch.zeros(h, w, dtype=torch.bool)
            o0[7:10, 3:4] = 1
            o1 = torch.zeros(h, w, dtype=torch.bool)
            o1[9:10, 3:7] = 1
            g0 = torch.zeros(h, w, dtype=torch.bool)
            g0[13:16, 12:13] = 1
            g1 = torch.zeros(h, w, dtype=torch.bool)
            g1[15:16, 10:14] = 1
            
            room.goals = {"Danger 0":o0, "Danger 1":o1, "Target 0":g0, "Target 1":g1}
        case "color shape experiment":
            room = Room(8,8)
            room.base[4,2] = 0
            room.base[3:6,3] = 0
            room.base[2:4,4] = 0
            locs = [[(0,0), (0,6)], [(0,7), (7,0)], [(5,2), (6,6)], [(0,0),(7,0),(5,2)], [(0,6),(0,7),(6,6)]]
            names = ["beige", "blue", "purple", "square", "circle"]
            for i in range(len(names)):
                goal = torch.zeros(8,8, dtype=torch.bool)
                for loc in locs[i]:
                    goal[loc] = 1
                room.goals[names[i]] = goal

    return room

def load_room(mode, name, n_actions=8):
    fn = f"project/static/{mode}/{name}"
    project = torch.load(fn)
    h, w = tuple(project['dim'])
    room = Room(h, w, n_actions)
    layers = {}
    always_mask = torch.zeros(h, w, dtype=bool)
    for layer in project["terrain"]:
        goal = layer["name"]
        mask = layer["mask"]
        if goal == 'obstacle':
            room.base = torch.logical_not(mask)
        else:
            if torch.any(mask):
                layers[goal] = mask
                if layer["always"]:
                    always_mask = torch.maximum(always_mask, mask)
    room.always = always_mask
    room.goals = layers
    return room


if __name__ == "__main__":
    room = load_room("saved_cont", "road.pt")
    room.start()
    room.visual()

    # a = torch.ones((16,16,10,10,8))
    # pa = a.permute(0,1,4,2,3).reshape(16,16,8,-1)

    # indices = torch.tensor([12, 45, 79, 88], dtype=torch.int)
    # print(pa.index_select(3, indices).shape)
    
    # t = torch.IntTensor([1,2,4,4,0])
    # print(torch.max(t, 0))

