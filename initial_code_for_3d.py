import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def draw_cube_with_data_points(corner_points):
    """
    Draws a filled cube on a 3D plot using the provided corner points

    Parameters:
    -----------
    corner_points: list of lists (float)
        A list of corner coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        provided in a cyclic order.

    Returns:
    --------
    None
    """
    x_coords = [point[0] for point in corner_points]
    y_coords = [point[1] for point in corner_points]
    plt.figure()
    plt.fill(x_coords, y_coords, color='skyblue')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title("Rectangle with Data Points")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define cube vertices
x = [0, 0, 0, 0, 1, 1, 1, 1]
y = [0, 0, 1, 1, 0, 0, 1, 1]
z = [0, 1, 0, 1, 0, 1, 0, 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define cube edges
edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7),
         (4,5), (4,6), (5,7), (6,7)]

# Plot edges
for edge in edges:
    ax.plot([x[edge[0]], x[edge[1]]], 
            [y[edge[0]], y[edge[1]]], 
            [z[edge[0]], z[edge[1]]], 'k')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
