import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# Step 1: Generate Random Scatter Data
np.random.seed(42)
num_points = 200  # Number of scattered points

x = np.random.uniform(0, 10, num_points)  # X-coordinates
y = np.random.uniform(0, 10, num_points)  # Y-coordinates
colors = np.sin(x) + np.cos(y)  # Function to generate color values

# Step 2: Create a Regular Grid for Interpolation
grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))

# Step 3: Interpolate Using Radial Basis Function (RBF)
rbf = scipy.interpolate.Rbf(x, y, colors, function='cubic')  # 'cubic' interpolation
grid_colors = rbf(grid_x, grid_y)

# Step 4: Plot the Smooth Continuous Color Map
fig, ax = plt.subplots(figsize=(8, 6))
c = ax.imshow(grid_colors, extent=[0, 10, 0, 10], origin='lower', cmap='jet', alpha=0.8)
plt.colorbar(c, label="Interpolated Color Values")

# Step 5: Overlay Scatter Plot (for reference)
ax.scatter(x, y, c=colors, cmap='jet', edgecolors='black', marker='o', s=50)
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_title("Smooth Continuous Color Plot from Scatter Data")
plt.show()
