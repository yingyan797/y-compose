import torch, random, os, json
import numpy as np
from PIL import Image

class Room:
    def __init__(self, height=30, width=30, mode="discrete"):
        self.shape = (height, width)
        self.state_dim = 2
        self.action_dim = 1
        self.is_discrete = mode == "discrete"
        self.base = torch.ones((height, width), dtype=torch.bool)
        self._vis_size = (min(width*40, 2000), min(height*40, 2000))
        self._reward_lev = 100
        self.goals = dict[str, torch.BoolTensor]()
        self.terrain = None
        self.loc = torch.zeros(2, dtype=torch.int)

    def visual(self, animate=True):
        canvas = self.terrain.numpy().astype(np.uint8)*100
        imarr = np.repeat(np.repeat(canvas, 10, axis=1), 10, axis=0)
        Image.fromarray(imarr, mode="L").resize(self._vis_size).save("project/static/room-terrain.png")
        if animate and len(self._trace) > 1:
            frames = []
            for loc in self._trace:
                frame = np.copy(canvas)
                frame[int(loc[0].item()), int(loc[1].item())] = [0, 200, 200]
                frames.append(Image.fromarray(frame))
            frames[0].save("project/static/trace-animate.gif", save_all=True, append_images=frames[1:])
            
    def start(self, start_state=None, restriction=None):
        def to_tensor(loc):
            return torch.IntTensor(loc) if self.is_discrete else torch.FloatTensor(loc)

        if self.terrain is None:
            base = self.base.to(torch.uint8)*2
            masks = [goal.to(torch.uint8)+1 for goal in self.goals.values()]
            self.terrain = torch.minimum(base, torch.max(torch.stack(masks), dim=0).values)
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
        if self.is_discrete:
            match action:   # 0..7
                case 0:
                    drn = [-1, 0]
                case 1: 
                    drn = [-1, 1]
                case 2:
                    drn = [0, 1]
                case 3:
                    drn = [1, 1]
                case 4:
                    drn = [1, 0]
                case 5:
                    drn = [1, -1]
                case 6:
                    drn = [0, -1]
                case 7:
                    drn = [-1, -1]
                case _:
                    raise ValueError("Not recognized action")
            new_loc = self.loc + torch.IntTensor(drn)
            row_in = new_loc[0] in range(self.shape[0])
            col_in = new_loc[1] in range(self.shape[1])
            if not (row_in and col_in):
                return self.loc, 0, 0
            label = self.terrain[new_loc[0], new_loc[1]]
        else:
            # -1, 1
            ang = action * torch.pi
            new_loc = self.loc + 10*torch.stack([-torch.cos(ang), torch.sin(ang)])
            row_in = new_loc[0] >= 0 and new_loc[0] <= self.shape[0]
            col_in = new_loc[1] >= 0 and new_loc[1] <= self.shape[1]
            if not (row_in and col_in):
                return self.loc, 0, 0
            label = self.terrain[int(new_loc[0].item()), int(new_loc[1].item())]

        self.loc = new_loc
        if trace:
            self._trace.append(new_loc)
        if label == 2:
            return new_loc, self._reward_lev, 1
        return new_loc, 0, 0
    
    def parse_compose(self, instr:str):
        full_goals = torch.where(self.terrain==2, 1, 0).to(dtype=torch.bool)
        def task_and(masks):
            masks = torch.stack(masks)
            return torch.min(masks, dim=0).values
        def task_or(masks):
            masks = torch.stack(masks)
            return torch.min(masks, dim=0).values
        def task_not(mask):
            neg = torch.logical_not(mask)
            return torch.minimum(neg, full_goals)
        comp_functions = {"and": task_and, "or": task_or, "not": task_not}
        def parse_expr(expression:str) -> torch.Tensor:
            if expression.startswith("'") and expression.endswith("'"):
                return self.goals[expression[1:-1]]
        
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

def load_room(name):
    fn = f"project/static/{name}"
    project = torch.load(fn)
    h, w = tuple(project['dim'])
    room = Room(h, w)
    layers = {}
    for layer in project["terrain"]:
        goal = layer["name"]
        if goal == 'obstacle':
            room.base = torch.logical_not(layer["mask"])
        else:
            layers[goal] = layer["mask"]

    room.goals = layers
    return room


if __name__ == "__main__":
    # room=load_room('saved_disc/color shape.pt')
    # room=load_room('saved_cont/road2goals.pt')
    # room.start()
    # room.visual()

    # a = torch.ones((16,16,10,10,8))
    # pa = a.permute(0,1,4,2,3).reshape(16,16,8,-1)

    # indices = torch.tensor([12, 45, 79, 88], dtype=torch.int)
    # print(pa.index_select(3, indices).shape)
    
    t = torch.IntTensor([1,2,4,4,0])
    print(torch.max(t, 0))

