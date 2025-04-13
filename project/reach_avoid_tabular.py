import torch
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

    def add_masks(self, masks:list[tuple[str, bool, torch.Tensor]]):
        self.masks = torch.cat((self.masks, torch.stack([mask[2] for mask in masks])))
        self.labels += [(mask[0], mask[1]) for mask in masks]

    def visual(self):
        c_step = int(255 / (self.masks.shape[0]-1)) if self.masks.shape[0] else 0
        canvas = np.zeros(self.shape+(3,))
        canvas[:, :, 2] = 50
        canvas[self.loc[0], self.loc[1]] = torch.tensor([0, 0, 255])
        r = 0
        for goal in self.masks.numpy():
            canvas[:, :, 0] += goal * r
            canvas[:, :, 1] += goal * 80
            r += c_step
        print(self.labels)
        Image.fromarray(canvas.astype(np.uint8), mode="RGB").resize(self._vis_size).save("project/static/room.png")

    def start(self):
        full_range = torch.tensor(self.shape)/3
        loc = torch.rand(2,) * full_range
        self.loc = loc.to(dtype=torch.uint8)

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
        
        new_loc = self.loc + torch.Tensor(drn)
        if new_loc[0] not in range(self.shape[0]) or new_loc[1] not in range(self.shape[1]):
            return self.loc, 0, False

        self.loc = new_loc
        for i in range(len(self.labels)):
            if self.masks[new_loc[0], new_loc[1]]:
                if not self.labels[i][1]:
                    if self.labels[i][0] not in self.safety_count:
                        self.safety_count[self.labels[i][0]] = 1
                    else:
                        self.safety_count[self.labels[i][0]] += 1
                    return self.loc, -self._reward_lev, False
                return self.loc, self._reward_lev, True
        return self.loc, 0, False

if __name__ == "__main__":
    h, w = 30, 30
    room = Room(h, w)
    o0 = torch.zeros(h, w, dtype=torch.bool)
    o0[16:21, 14:20] = 1
    o1 = torch.zeros(h, w, dtype=torch.bool)
    o1[19:23, 11:21] = 1
    g0 = torch.zeros(h, w, dtype=torch.bool)
    g0[27:28, 20:25] = 1
    g1 = torch.zeros(h, w, dtype=torch.bool)
    g1[25:29, 22:23] = 1
    
    room.add_masks([("Danger 0", False, o0), ("Danger 1", False, o1), ("Goal 0", True, g0), ("Goal 1", True, g1)])
    room.start()
    room.visual()

