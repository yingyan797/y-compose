import torch, random
import numpy as np
from PIL import Image

class Room:
    def __init__(self, height=30, width=30):
        self.shape = (height, width)
        self._vis_size = (width*40, height*40)
        self._reward_lev = 1000
        self.masks = torch.tensor([], dtype=torch.bool)
        self.labels = []
        self.loc = torch.zeros(2, dtype=torch.uint8)
        self.safety_count = {}
        self.subgoal_training = False

    def add_masks(self, masks:list[tuple[str, bool, torch.Tensor]]):
        self.masks = torch.cat((self.masks, torch.stack([mask[2] for mask in masks])))
        self.labels += [(mask[0], mask[1]) for mask in masks]
        self.terrain = torch.sum(self.masks, 0)
        self._avail_locs = [loc for loc in filter(lambda loc: self.terrain[loc[0], loc[1]] == 0,
            [(r,c) for r in range(self.terrain.shape[0]) for c in range(self.terrain.shape[1])])]

    def visual(self):
        c_step = int(255 / (self.masks.shape[0]-1)) if self.masks.shape[0] else 0
        canvas = np.zeros(self.shape+(3,))
        canvas[:, :, 2] = 50
        for loc in self.trace:
            canvas[loc[0], loc[1]] = torch.tensor([0, 0, 255])
        r = 0
        for goal in self.masks.numpy():
            canvas[:, :, 0] += goal * r
            canvas[:, :, 1] += goal * 80
            r += c_step
        canvas = np.minimum(canvas, 255).astype(np.uint8)
        Image.fromarray(np.repeat(np.repeat(canvas, 10, axis=1), 10, axis=0), mode="RGB").resize(self._vis_size).save("project/static/room.png")

    def start(self, restricted=False):
        if restricted:
            full_range = torch.tensor(self.shape)/3
            loc = torch.rand(2,) * full_range
        else:
            loc = torch.Tensor(random.choice(self._avail_locs))

        self.loc = loc.to(dtype=torch.int)
        self.trace = [self.loc]
        return self.loc

    def step(self, action:int):
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
        
        new_loc = self.loc + torch.tensor(drn, dtype=torch.int)
        if new_loc[0] not in range(self.shape[0]) or new_loc[1] not in range(self.shape[1]):
            return self.loc, 0, 0
        self.loc = new_loc
        if self.subgoal_training:
            self.trace.append(self.loc)
        for i in range(len(self.labels)):
            if self.masks[i, new_loc[0], new_loc[1]]:
                if self.subgoal_training and self.labels[i][1] == 2:
                    if self.labels[i][0] not in self.safety_count:
                        self.safety_count[self.labels[i][0]] = 1
                    else:
                        self.safety_count[self.labels[i][0]] += 1
                    return self.loc, -self._reward_lev, 2
                return self.loc, self._reward_lev, 1
        return self.loc, 0, 0

def create_room():
    h, w = 16, 16
    room = Room(h, w)
    o0 = torch.zeros(h, w, dtype=torch.bool)
    o0[8:12, 8:9] = 1
    o1 = torch.zeros(h, w, dtype=torch.bool)
    o1[11:12, 8:12] = 1
    g0 = torch.zeros(h, w, dtype=torch.bool)
    g0[14:16, 12:13] = 1
    g1 = torch.zeros(h, w, dtype=torch.bool)
    g1[15:16, 11:14] = 1
    
    room.add_masks([("Danger 0", 2, o0), ("Danger 1", 2, o1), ("Goal 0", 1, g0), ("Goal 1", 1, g1)])
    return room

if __name__ == "__main__":
    room = create_room()
    for i in range(5):
        room.step(2)
    for i in range(4):
        room.step(3)
    # print(room.trace)
    room.visual()

