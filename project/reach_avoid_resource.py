from reach_avoid_tabular import Room, torch, random
import numpy as np

class ResourceRoom(Room):
    '''An extension of Room class for resource-constrained reach-avoid problems'''
    def __init__(self, height=30, width=30, n_actions=8, max_resource=100):
        super().__init__(height, width, n_actions)
        self.shape = (height, width, max_resource)
        self.state_dim = 3
        for vec in self.action_map.values():
            vec.append(-1) # -1 is the resource cost
        if n_actions:
            self.action_map[8] = [0, 0, 1]
        self.base = torch.ones(self.shape, dtype=torch.bool)
        self.loc = np.zeros(3, dtype=np.int)

    # start and step are the same as parent class Room for elk, just have an additional resource dimension 
