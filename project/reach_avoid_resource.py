from reach_avoid_tabular import Room, torch, random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


# Alternatively, create a more structured pattern
# Uncomment below for a gradient pattern instead of random colors
# data = np.zeros((10, 10, 10, 3))
# for i in range(10):
#     for j in range(10):
#         for k in range(10):
#             data[i, j, k, 0] = int(255 * i / 9)  # Red gradient along x
#             data[i, j, k, 1] = int(255 * j / 9)  # Green gradient along y  
#             data[i, j, k, 2] = int(255 * k / 9)  # Blue gradient along z

def create_brick_mesh(x, y, z, size=0.8):
    """Create vertices and faces for a cube/brick at position (x, y, z)"""
    # Define the 8 vertices of a cube
    vertices = np.array([
        [x-size/2, y-size/2, z-size/2],
        [x+size/2, y-size/2, z-size/2],
        [x+size/2, y+size/2, z-size/2],
        [x-size/2, y+size/2, z-size/2],
        [x-size/2, y-size/2, z+size/2],
        [x+size/2, y-size/2, z+size/2],
        [x+size/2, y+size/2, z+size/2],
        [x-size/2, y+size/2, z+size/2]
    ])
    
    # Define the 12 triangular faces (2 triangles per cube face)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 7, 6], [4, 6, 5],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2]   # right
    ])
    
    return vertices, faces


def visual3d(data):
    # Create the 3D visualization
    fig = go.Figure()

    # Add each brick as a separate mesh
    for i in range(10):
        for j in range(10):
            for k in range(10):
                # Get RGB color for this brick
                r, g, b, a = data[i, j, k]
                color = f'rgba({r},{g},{b},{a})'
                
                # Create brick mesh
                vertices, faces = create_brick_mesh(i, j, k)
                
                # Add mesh to figure
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.8,
                    showscale=False,
                    hovertemplate=f'Position: ({i},{j},{k})<br>RGB: ({r},{g},{b})<extra></extra>'
                ))

    # Update layout for better visualization
    fig.update_layout(
        title='3D Brick Visualization (10×10×10 Array with RGB Colors)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        width=800,
        height=600,
        showlegend=False
    )

    # Show the plot
    fig.show()

    # Optional: Save as HTML file
    # fig.write_html("brick_visualization.html")

    print("3D brick visualization created!")
    print(f"Array shape: {data.shape}")
    print(f"Total bricks: {data.shape[0] * data.shape[1] * data.shape[2]}")
    print(f"RGB value range: {data.min()} to {data.max()}")

if __name__ == "__main__":
    space = np.zeros((10, 10, 10, 4), dtype=np.uint8)
    space[4:7, 2:4, :] = [180, 0, 0, 150]
    space[7:9, 7:9, :5] = [100, 0, 0, 150]
    space[2:4, 7:9, 3:] = [0, 100, 0, 150]
    visual3d(space)