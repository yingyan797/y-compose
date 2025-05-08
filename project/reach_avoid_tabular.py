import torch, random, os, json
import numpy as np
from PIL import Image

class Room:
    def __init__(self, height=30, width=30):
        self.shape = (height, width)
        self.base = torch.ones((height, width), dtype=torch.bool)
        self._vis_size = (min(width*40, 2000), min(height*40, 2000))
        self._reward_lev = 100
        self.goals = dict[str, torch.BoolTensor]()
        self.terrain = None
        self.loc = torch.zeros(2, dtype=torch.int)

    def visual(self, animate=True):
        # c_step = int(255 / (len(self.goals)-1)) if len(self.goals) > 1 else 0
        # canvas = torch.unsqueeze(self.base, 2).repeat(1,1,3).numpy().astype(int)*75
        # canvas[:, :, :2] = 0
        # r = 0
        # for goal in self.goals.values():
        #     canvas[:, :, 0] += goal.numpy() * r
        #     canvas[:, :, 1] += goal.numpy() * 80
        #     r += c_step
        # canvas = np.minimum(canvas, 255).astype(np.uint8)
        # Image.fromarray(np.repeat(np.repeat(canvas, 10, axis=1), 10, axis=0), mode="RGB").resize(self._vis_size).save("project/static/room-goals.png")
        canvas = self.terrain.numpy().astype(np.uint8)*100
        imarr = np.repeat(np.repeat(canvas, 10, axis=1), 10, axis=0)
        Image.fromarray(imarr, mode="L").resize(self._vis_size).save("project/static/room-terrain.png")
        if animate and len(self._trace) > 1:
            frames = []
            for loc in self._trace:
                frame = np.copy(canvas)
                frame[loc[0], loc[1]] = [0, 200, 200]
                frames.append(Image.fromarray(np.repeat(np.repeat(frame, 10, axis=1), 10, axis=0)))
            frames[0].save("project/static/trace-animate.gif", save_all=True, append_images=frames[1:])
            

    def start(self, start_state=None, restriction=None):
        if self.terrain is None:
            base = self.base.to(torch.uint8)*2
            masks = [goal.to(torch.uint8)+1 for goal in self.goals.values()]
            self.terrain = torch.minimum(base, torch.max(torch.stack(masks), dim=0).values)
            self._avail_locs = torch.nonzero(self.terrain).numpy().tolist()
        if start_state is not None:
            loc = torch.IntTensor(start_state)
        elif restriction is not None:
            region = [(r,c) for r,c in self._avail_locs if restriction[r,c] > 0]
            loc = torch.IntTensor(random.choice(region))
        else:
            loc = torch.IntTensor(random.choice(self._avail_locs))

        self.loc = loc
        self._trace = [self.loc]
        return self.loc

    def step(self, action:int, trace=False):
        drn = [0,0]
        match action:
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
        if new_loc[0] not in range(self.shape[0]) or new_loc[1] not in range(self.shape[1]) or self.terrain[new_loc[0], new_loc[1]] == 0:
            return self.loc, 0, 0
        self.loc = new_loc
        if trace:
            self._trace.append(new_loc)
        if self.terrain[new_loc[0], new_loc[1]] == 2:
            return new_loc, self._reward_lev, 1
        return new_loc, 0, 0

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
    room=load_room('saved_cont/road2goals.pt')
    room.start()
    room.visual()

    # a = torch.ones((16,16,10,10,8))
    # pa = a.permute(0,1,4,2,3).reshape(16,16,8,-1)

    # indices = torch.tensor([12, 45, 79, 88], dtype=torch.int)
    # print(pa.index_select(3, indices).shape)
    


