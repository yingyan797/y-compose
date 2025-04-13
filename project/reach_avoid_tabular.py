import torch
import numpy as np
from PIL import Image

class Room:
    def __init__(self, height=30, width=30):
        self.shape = (height, width)
        self._vis_size = (width*40, height*40)
        self.masks = torch.tensor([], dtype=torch.bool)
        self.names = []
        self.loc = torch.zeros(2, dtype=torch.uint8)

    def add_masks(self, masks:dict[str, torch.Tensor]):
        self.masks = torch.cat((self.masks, torch.stack(tuple(masks.values()))))
        self.names += [n for n in masks.keys()]

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
        
        Image.fromarray(canvas.astype(np.uint8), mode="RGB").resize(self._vis_size).show()

    def start(self):
        full_range = torch.tensor(self.shape)/3
        loc = torch.rand(2,) * full_range
        self.loc = loc.to(dtype=torch.uint8)


if __name__ == "__main__":
    h, w = 30, 30
    room = Room(h, w)
    o0 = torch.zeros(h, w, dtype=torch.bool)
    o0[16:21, 16:21] = 1
    o1 = torch.zeros(h, w, dtype=torch.bool)
    o1[19:22, 15:23] = 1
    g0 = torch.zeros(h, w, dtype=torch.bool)
    g0[27:28, 20:25] = 1
    g1 = torch.zeros(h, w, dtype=torch.bool)
    g1[25:29, 22:23] = 1
    
    room.add_masks({"Obstacle 0": o0, "Obstacle 1": o1, "Goal 0": g0, "Goal 1": g1})
    room.start()
    room.visual()

