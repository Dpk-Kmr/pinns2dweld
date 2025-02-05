import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from matplotlib import animation

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

###############################################################################
# Geometry & Data Generation Functions
###############################################################################

def draw_rectangle_with_data_points(corner_points):
    x_coords = [pt[0] for pt in corner_points]
    y_coords = [pt[1] for pt in corner_points]
    plt.figure()
    plt.fill(x_coords, y_coords, color='skyblue')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title("Rectangle with Data Points")
    plt.show()


def generate_1d_array(size, min_val=0, max_val=10):
    return np.random.uniform(min_val, max_val, size)


def solid_blocks(x_corner, y_corner, direction, x_len, block_height, block_length, new_block):
    x_low = 0.0
    x_high = x_len
    x_mid = x_corner
    if direction == 1:
        y_left_high = y_corner
        y_right_high = y_corner - block_height
        if new_block:
            x_new_block_low = x_mid - block_length
            x_new_block_high = x_mid
            y_new_block_low = y_corner - block_height
            y_new_block_high = y_corner
    else:
        y_left_high = y_corner - block_height
        y_right_high = y_corner
        if new_block:
            x_new_block_high = x_mid + block_length
            x_new_block_low = x_mid
            y_new_block_low = y_corner - block_height
            y_new_block_high = y_corner

    corner_points_left = [
        [x_low, 0.0],
        [x_low, y_left_high],
        [x_mid, y_left_high],
        [x_mid, 0.0]
    ]
    corner_points_right = [
        [x_mid, 0.0],
        [x_mid, y_right_high],
        [x_high, y_right_high],
        [x_high, 0.0]
    ]
    if new_block:
        corner_points_new = [
            [x_new_block_low, y_new_block_low],
            [x_new_block_low, y_new_block_high],
            [x_new_block_high, y_new_block_high],
            [x_new_block_high, y_new_block_low]
        ]
    else:
        corner_points_new = [[x_corner, y_corner]] * 4  # dummy block

    return [corner_points_left, corner_points_right, corner_points_new]


def generate_points(three_block_corner_point, time, mode='uniform', dx=0.01, dy=0.01, density=100):
    grid = []
    for corner_points in three_block_corner_point[:2]:  # use only left and right blocks
        x_coords = [pt[0] for pt in corner_points]
        y_coords = [pt[1] for pt in corner_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        if (x_max - x_min) == 0.0 or (y_max - y_min) == 0.0:
            continue
        if mode == 'uniform':
            x_vals = np.arange(x_min, x_max, dx)
            y_vals = np.arange(y_min, y_max, dy)
            for xi in x_vals:
                for yi in y_vals:
                    grid.append([xi, yi, time])
        elif mode == 'random':
            area = (x_max - x_min) * (y_max - y_min)
            n_points = int(area * density)
            random_x = np.random.uniform(x_min, x_max, n_points)
            random_y = np.random.uniform(y_min, y_max, n_points)
            for xi, yi in zip(random_x, random_y):
                grid.append([xi, yi, time])
        else:
            raise ValueError("Invalid mode. Use 'uniform' or 'random'.")
    return grid


def generate_points_using_line_extremes(x_extreme, y_extreme, time, mode='uniform', dx=0.01, dy=0.01, density=100):
    grid = []
    x_min, x_max = min(x_extreme), max(x_extreme)
    y_min, y_max = min(y_extreme), max(y_extreme)
    if (y_max - y_min) == 0.0 and (x_max - x_min) != 0.0:
        if mode == 'uniform':
            x_vals = np.arange(x_min, x_max, dx)
            for xi in x_vals:
                grid.append([xi, y_max, time])
        elif mode == 'random':
            line_length = x_max - x_min
            n_points = int(line_length * density)
            random_x = np.random.uniform(x_min, x_max, n_points)
            for xi in random_x:
                grid.append([xi, y_max, time])
        else:
            raise ValueError("Invalid mode.")
    elif (x_max - x_min) == 0.0 and (y_max - y_min) != 0.0:
        if mode == 'uniform':
            y_vals = np.arange(y_min, y_max, dy)
            for yi in y_vals:
                grid.append([x_max, yi, time])
        elif mode == 'random':
            line_length = y_max - y_min
            n_points = int(line_length * density)
            random_y = np.random.uniform(y_min, y_max, n_points)
            for yi in random_y:
                grid.append([x_max, yi, time])
        else:
            raise ValueError("Invalid mode.")
    else:
        if (x_max - x_min) == 0.0 and (y_max - y_min) == 0.0:
            print("Warning: single point provided.")
        else:
            raise ValueError("Not a single point or straight line.")
    return grid


def get_boundary_extremes(three_block_corner_point, consider_bottom_as_boundary=False):
    left_xs = [pt[0] for pt in three_block_corner_point[0]]
    left_ys = [pt[1] for pt in three_block_corner_point[0]]
    right_xs = [pt[0] for pt in three_block_corner_point[1]]
    right_ys = [pt[1] for pt in three_block_corner_point[1]]
    left_x = min(left_xs)
    right_x = max(right_xs)
    mid_x = max(left_xs)
    if max(left_xs) != min(right_xs):
        raise ValueError("Left and right rectangles not touching properly")
    left_y_low, left_y_high = min(left_ys), max(left_ys)
    right_y_low, right_y_high = min(right_ys), max(right_ys)
    mid_y_low = min(max(left_ys), max(right_ys))
    mid_y_high = max(max(left_ys), max(right_ys))
    left_boundary_extreme = [[left_x, left_x], [left_y_low, left_y_high]]
    right_boundary_extreme = [[right_x, right_x], [right_y_low, right_y_high]]
    shared_boundary_extreme = [[mid_x, mid_x], [mid_y_low, mid_y_high]]
    top_left_extreme = [[left_x, mid_x], [left_y_high, left_y_high]]
    top_right_extreme = [[mid_x, right_x], [right_y_high, right_y_high]]
    return_data = [
        left_boundary_extreme,
        right_boundary_extreme,
        shared_boundary_extreme,
        top_left_extreme,
        top_right_extreme
    ]
    if consider_bottom_as_boundary:
        if right_y_low == right_y_high:
            bottom_extreme = [[left_x, mid_x], [right_y_low, right_y_low]]
        else:
            bottom_extreme = [[left_x, right_x], [right_y_low, right_y_low]]
        return_data.append(bottom_extreme)
    return return_data


def generate_boundary_point_using_three_blocks(three_block_corner_point, time, mode='uniform', dx=0.01, dy=0.01, density=100, consider_bottom_as_boundary=False):
    grid = []
    extremes = get_boundary_extremes(three_block_corner_point, consider_bottom_as_boundary)
    for e in extremes:
        grid += generate_points_using_line_extremes(e[0], e[1], time, mode, dx, dy, density)
    return grid


def generate_individual_boundary_data(three_block_corner_point, time, mode='uniform', dx=0.01, dy=0.01, density=100, consider_bottom_as_boundary=False):
    grid = {"left": [], "right": [], "shared": [], "top_left": [], "top_right": []}
    if consider_bottom_as_boundary:
        grid["bottom"] = []
    extremes = get_boundary_extremes(three_block_corner_point, consider_bottom_as_boundary)
    grid["left"] += generate_points_using_line_extremes(extremes[0][0], extremes[0][1], time, mode, dx, dy, density)
    grid["right"] += generate_points_using_line_extremes(extremes[1][0], extremes[1][1], time, mode, dx, dy, density)
    grid["shared"] += generate_points_using_line_extremes(extremes[2][0], extremes[2][1], time, mode, dx, dy, density)
    grid["top_left"] += generate_points_using_line_extremes(extremes[3][0], extremes[3][1], time, mode, dx, dy, density)
    grid["top_right"] += generate_points_using_line_extremes(extremes[4][0], extremes[4][1], time, mode, dx, dy, density)
    if consider_bottom_as_boundary:
        grid["bottom"] += generate_points_using_line_extremes(extremes[5][0], extremes[5][1], time, mode, dx, dy, density)
    return grid


def generate_wall_internal_data(block_corner_data, ctd_data, total_mid_times=10, t_mode="uniform", mode="random", dx=0.01, dy=0.01, density=50):
    wall_internal_data = []
    if t_mode == "uniform":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for t_mid in torch.linspace(ctdip[2], ctdin[2], total_mid_times)[:-1]:
                pts = generate_points(bcdi, t_mid.item(), mode, dx, dy, density)
                wall_internal_data += pts
    elif t_mode == "random":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for _ in range(total_mid_times):
                t_mid = random.uniform(ctdip[2], ctdin[2])
                pts = generate_points(bcdi, t_mid, mode, dx, dy, density)
                wall_internal_data += pts
    else:
        raise ValueError("Invalid t_mode.")
    return np.array(wall_internal_data)


def generate_new_block_internal_data(block_corner_data, ctd_data, mode='random', dx=0.01, dy=0.01, density=100):
    new_block_internal_data = []
    for bcdi, ctdi in zip(block_corner_data, ctd_data):
        new_block = [bcdi[2]]  # new block corners
        pts = generate_points(new_block, ctdi[2], mode, dx, dy, density)
        new_block_internal_data += pts
    return np.array(new_block_internal_data)


def generate_boundary_data(block_corner_data, ctd_data, total_mid_times=10, t_mode="uniform", mode="random", dx=0.01, dy=0.01, density=50, consider_bottom_as_boundary=False):
    boundary_data = []
    if t_mode == "uniform":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for t_mid in torch.linspace(ctdip[2], ctdin[2], total_mid_times)[:-1]:
                pts = generate_boundary_point_using_three_blocks(bcdi, t_mid.item(), mode, dx, dy, density, consider_bottom_as_boundary)
                boundary_data += pts
    elif t_mode == "random":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for _ in range(total_mid_times):
                t_mid = random.uniform(ctdip[2], ctdin[2])
                pts = generate_boundary_point_using_three_blocks(bcdi, t_mid, mode, dx, dy, density, consider_bottom_as_boundary)
                boundary_data += pts
    else:
        raise ValueError("Invalid t_mode.")
    return np.array(boundary_data)


def generate_all_individual_boundary_data(block_corner_data, ctd_data, total_mid_times=10, t_mode="uniform", mode="random", dx=0.01, dy=0.01, density=50, consider_bottom_as_boundary=False):
    boundary_data = {"left": [], "right": [], "shared": [], "top_left": [], "top_right": []}
    if consider_bottom_as_boundary:
        boundary_data["bottom"] = []
    if t_mode == "uniform":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for t_mid in torch.linspace(ctdip[2], ctdin[2], total_mid_times)[:-1]:
                pts = generate_individual_boundary_data(bcdi, t_mid.item(), mode, dx, dy, density, consider_bottom_as_boundary)
                for key in boundary_data:
                    boundary_data[key] += pts[key]
    elif t_mode == "random":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for _ in range(total_mid_times):
                t_mid = random.uniform(ctdip[2], ctdin[2])
                pts = generate_individual_boundary_data(bcdi, t_mid, mode, dx, dy, density, consider_bottom_as_boundary)
                for key in boundary_data:
                    boundary_data[key] += pts[key]
    else:
        raise ValueError("Invalid t_mode.")
    for key in boundary_data:
        boundary_data[key] = np.array(boundary_data[key])
    return boundary_data


def random_sample(tensor_, num_samples):
    indices = random.sample(range(len(tensor_)), min(num_samples, len(tensor_)))
    return tensor_[indices]


def heat_source_origin(current_time, travel_speed, jump, x_max, x_min):
    one_lap_time = (x_max - x_min) / travel_speed
    current_lap = (current_time / one_lap_time).to(torch.int32)
    current_y = (current_lap + 1) * jump
    direction = current_lap % 2
    current_x = (travel_speed * current_time) % (x_max - x_min)
    mask = (direction == 1)
    current_x[mask] = x_max - current_x[mask]
    return [current_x, current_y]


def heat_source_equation(x, y, current_time, travel_speed, jump, x_max, x_min):
    origin_x, origin_y = heat_source_origin(current_time, travel_speed, jump, x_max, x_min)
    ret = torch.exp(-((x - origin_x)**2 + (y - origin_y)**2))
    ret_modified = ret.clone()
    ret_modified[y > origin_y] = 0
    return ret_modified

###############################################################################
# PINN Model & PDE Residual Definitions
###############################################################################

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
    
    def forward(self, x):
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        return self.linears[-1](x)
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


def equation(u, X, f, k=11.4, rho=4.5e-9, cp=7.14e8, tc=0.2, lc=2):
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_d[:,0], X, grad_outputs=torch.ones_like(du_d[:,0]), create_graph=True)[0][:,0:1]
    d2u_dy2 = torch.autograd.grad(du_d[:,1], X, grad_outputs=torch.ones_like(du_d[:,1]), create_graph=True)[0][:,1:2]
    d_u_dt = du_d[:,2:3]
    return d_u_dt - (k/(rho*cp))*(tc/(lc**2))*(d2u_dx2 + d2u_dy2) - f


def bc_left(u, X, h=0.02, u_amb=298/3000, sigma=5.67e-11, epsilon=0.3, k=11.4, lc=2.0, tc=3000):
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx = du_d[:,0:1]
    conv = (h*lc/k)*(u - u_amb)
    rad = (sigma*epsilon*(tc**3)*lc/k)*(u**4 - u_amb**4)
    return du_dx - conv - rad


def bc_right(u, X, **kwargs):
    return bc_left(u, X, **kwargs)


def bc_shared(u, X, **kwargs):
    return bc_left(u, X, **kwargs)


def bc_top_left(u, X, h=0.02, u_amb=298/3000, sigma=5.67e-11, epsilon=0.3, k=11.4, lc=2.0, tc=3000):
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dy = du_d[:,1:2]
    conv = h*(u - u_amb)
    rad = sigma*epsilon*(u**4 - u_amb**4)
    return k*du_dy - conv - rad


def bc_top_right(u, X, **kwargs):
    return bc_top_left(u, X, **kwargs)


def bc_bottom(u, X, **kwargs):
    return bc_top_left(u, X, **kwargs)

###############################################################################
# Main Code: Data Generation, Normalization, Training
###############################################################################

# Domain parameters (mm and sec)
x_range = [0.0, 15.0]
y_range = [0.0, 4.0]
t_range = [0.0, 6.0]
time_gap = 0.1
t_data = np.arange(t_range[0], t_range[1], time_gap)

# Block geometry parameters
x_len = x_range[-1]
x_gap = 1.0
y_gap = 1.0
direction = 1
y_level = 1.0

# Generate ctd_data (tracking joint positions)
ctd_data = [[x_gap, y_gap, t_data[0], direction]]
for ti in t_data[1:]:
    if direction == 1:
        x_value = ctd_data[-1][0] + 1.0
    else:
        x_value = ctd_data[-1][0] - 1.0
    ctd_data.append([x_value, y_level, ti, direction])
    if x_value == x_range[0] or x_value == x_range[1]:
        direction = -direction
        y_level += y_gap

# Generate corner data for each time step
block_corner_data = []
for x_corner, y_corner, ti, direction in ctd_data:
    corners = solid_blocks(x_corner, y_corner, direction, x_len, y_gap, x_gap, True)
    block_corner_data.append(np.array(corners))
block_corner_data = np.array(block_corner_data)

# Generate interior and boundary datasets
new_block_internal_data = generate_new_block_internal_data(block_corner_data, ctd_data, mode='random', dx=0.01, dy=0.01, density=2)
total_mid_times_internal = 10
wall_internal_data = generate_wall_internal_data(block_corner_data, ctd_data, total_mid_times=total_mid_times_internal, 
                                                 t_mode="random", mode="random", dx=0.01, dy=0.01, density=2)
total_mid_times_boundary = 10
consider_bottom_as_boundary = False
indi_boundary_data = generate_all_individual_boundary_data(block_corner_data, ctd_data, total_mid_times=total_mid_times_boundary, 
                                                           t_mode="random", mode="random", dx=0.01, dy=0.01, density=2, 
                                                           consider_bottom_as_boundary=consider_bottom_as_boundary)

###############################################################################
# Non-dimensionalization & Normalization
###############################################################################
phi_laser = 2   # mm
v_laser = 10    # mm/s
lc = phi_laser
tc = phi_laser / v_laser

nd_new_block_internal_data = new_block_internal_data / np.array([lc, lc, tc])
nd_wall_internal_data = wall_internal_data / np.array([lc, lc, tc])
nd_indi_boundary_data = {}
for key in indi_boundary_data:
    nd_indi_boundary_data[key] = indi_boundary_data[key] / np.array([lc, lc, tc])

# Normalize to [-1,1]
ndx_min, ndx_max = 0, 7.5
ndy_min, ndy_max = 0, 2.0
ndt_min, ndt_max = 0, 30

def normalize_to_nn(data, lower, upper):
    return 2 * ((data - lower) / (upper - lower)) - 1.0

nnd_new_block_internal_data = normalize_to_nn(nd_new_block_internal_data,
                                              np.array([ndx_min, ndy_min, ndt_min]),
                                              np.array([ndx_max, ndy_max, ndt_max]))
nnd_wall_internal_data = normalize_to_nn(nd_wall_internal_data,
                                         np.array([ndx_min, ndy_min, ndt_min]),
                                         np.array([ndx_max, ndy_max, ndt_max]))
nnd_indi_boundary_data = {}
for key in nd_indi_boundary_data:
    nnd_indi_boundary_data[key] = normalize_to_nn(nd_indi_boundary_data[key],
                                                  np.array([ndx_min, ndy_min, ndt_min]),
                                                  np.array([ndx_max, ndy_max, ndt_max]))

###############################################################################
# Convert Data to Torch Tensors and Set requires_grad
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

nnd_new_block_internal_data = torch.tensor(nnd_new_block_internal_data, dtype=torch.float32).to(device)
nnd_wall_internal_data = torch.tensor(nnd_wall_internal_data, dtype=torch.float32).to(device)
nnd_boundary_data_left = torch.tensor(nnd_indi_boundary_data["left"], dtype=torch.float32).to(device)
nnd_boundary_data_right = torch.tensor(nnd_indi_boundary_data["right"], dtype=torch.float32).to(device)
nnd_boundary_data_shared = torch.tensor(nnd_indi_boundary_data["shared"], dtype=torch.float32).to(device)
nnd_boundary_data_top_left = torch.tensor(nnd_indi_boundary_data["top_left"], dtype=torch.float32).to(device)
nnd_boundary_data_top_right = torch.tensor(nnd_indi_boundary_data["top_right"], dtype=torch.float32).to(device)
if consider_bottom_as_boundary:
    nnd_boundary_data_bottom = torch.tensor(nnd_indi_boundary_data["bottom"], dtype=torch.float32).to(device)

nnd_new_block_internal_data.requires_grad_(True)
nnd_wall_internal_data.requires_grad_(True)
nnd_boundary_data_left.requires_grad_(True)
nnd_boundary_data_right.requires_grad_(True)
nnd_boundary_data_shared.requires_grad_(True)
nnd_boundary_data_top_left.requires_grad_(True)
nnd_boundary_data_top_right.requires_grad_(True)
if consider_bottom_as_boundary:
    nnd_boundary_data_bottom.requires_grad_(True)

###############################################################################
# Initialize and Train PINN
###############################################################################
layers = [3, 20, 40, 40, 20, 1]
model = PINN(layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)

# For the new block region, enforce fixed temperature
constant_u = torch.tensor([2900.0 / 3000.0], dtype=torch.float32).to(device)

epochs = 5000           # Increased number of epochs
num_samples = 500       # Increased sampling per epoch
loss_history = []
loss_components = {"interior": [], "new_block": [], "boundary": []}

for epoch in range(epochs):
    optimizer.zero_grad()

    # Sample points randomly from datasets
    sampled_wall = random_sample(nnd_wall_internal_data, num_samples)
    sampled_new_block = random_sample(nnd_new_block_internal_data, num_samples)
    sampled_b_left = random_sample(nnd_boundary_data_left, num_samples)
    sampled_b_right = random_sample(nnd_boundary_data_right, num_samples)
    sampled_b_shared = random_sample(nnd_boundary_data_shared, num_samples)
    sampled_b_top_left = random_sample(nnd_boundary_data_top_left, num_samples)
    sampled_b_top_right = random_sample(nnd_boundary_data_top_right, num_samples)
    if consider_bottom_as_boundary:
        sampled_b_bottom = random_sample(nnd_boundary_data_bottom, num_samples)

    # Interior loss
    u_interior = model(sampled_wall)
    x = sampled_wall[:,0:1]
    y = sampled_wall[:,1:2]
    tvar = sampled_wall[:,2:3]
    f = heat_source_equation(x, y, tvar, travel_speed=1.0, jump=0.5, x_max=10.0, x_min=0.0) - 1.5
    res = equation(u_interior, sampled_wall, f)
    loss_interior = torch.mean(res**2)

    # New block loss
    u_new = model(sampled_new_block)
    loss_new = torch.mean((u_new - constant_u)**2)

    # Boundary loss
    u_b_left = model(sampled_b_left)
    u_b_right = model(sampled_b_right)
    u_b_shared = model(sampled_b_shared)
    u_b_top_left = model(sampled_b_top_left)
    u_b_top_right = model(sampled_b_top_right)
    if consider_bottom_as_boundary:
        u_b_bottom = model(sampled_b_bottom)
    loss_b_left = torch.mean(bc_left(u_b_left, sampled_b_left)**2)
    loss_b_right = torch.mean(bc_right(u_b_right, sampled_b_right)**2)
    loss_b_shared = torch.mean(bc_shared(u_b_shared, sampled_b_shared)**2)
    loss_b_top_left = torch.mean(bc_top_left(u_b_top_left, sampled_b_top_left)**2)
    loss_b_top_right = torch.mean(bc_top_right(u_b_top_right, sampled_b_top_right)**2)
    loss_b_bottom = torch.mean(bc_bottom(u_b_bottom, sampled_b_bottom)**2) if consider_bottom_as_boundary else 0.0
    loss_boundary = loss_b_left + loss_b_right + loss_b_shared + loss_b_top_left + loss_b_top_right + loss_b_bottom

    total_loss = loss_interior + loss_new + loss_boundary
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss)

    loss_history.append(total_loss.item())
    loss_components["interior"].append(loss_interior.item())
    loss_components["new_block"].append(loss_new.item())
    loss_components["boundary"].append(loss_boundary.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss.item():.5f}")

# Optionally, perform a second-phase optimization using L-BFGS
def closure():
    optimizer_lbfgs.zero_grad()
    u_interior = model(nnd_wall_internal_data)
    x = nnd_wall_internal_data[:,0:1]
    y = nnd_wall_internal_data[:,1:2]
    tvar = nnd_wall_internal_data[:,2:3]
    f = heat_source_equation(x, y, tvar, travel_speed=1.0, jump=0.5, x_max=10.0, x_min=0.0) - 1.5
    res = equation(u_interior, nnd_wall_internal_data, f)
    loss_interior = torch.mean(res**2)
    u_new = model(nnd_new_block_internal_data)
    loss_new = torch.mean((u_new - constant_u)**2)
    # For simplicity, use only one representative boundary
    u_b_left = model(nnd_boundary_data_left)
    loss_b_left = torch.mean(bc_left(u_b_left, nnd_boundary_data_left)**2)
    loss = loss_interior + loss_new + loss_b_left
    loss.backward()
    return loss

optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-9, history_size=50)
optimizer_lbfgs.step(closure)

###############################################################################
# Post-Training Analysis: Tables, Plots, and Animations
###############################################################################

# Create a pandas DataFrame summarizing final losses
final_losses = {
    "Interior Loss": loss_components["interior"][-1],
    "New Block Loss": loss_components["new_block"][-1],
    "Boundary Loss": loss_components["boundary"][-1],
    "Total Loss": loss_history[-1]
}
df_losses = pd.DataFrame([final_losses])
print("\nFinal Loss Components:")
print(df_losses)

# Plot training loss history
plt.figure(figsize=(8,5))
plt.plot(loss_history, label="Total Loss")
plt.plot(loss_components["interior"], label="Interior Loss")
plt.plot(loss_components["new_block"], label="New Block Loss")
plt.plot(loss_components["boundary"], label="Boundary Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss History")
plt.grid(True)
plt.show()

# Generate prediction snapshots over a grid for selected times.
# (We choose 5 time slices uniformly in the physical time domain.)
npts = 50
x_lin = np.linspace(ndx_min, ndx_max, npts)
y_lin = np.linspace(ndy_min, ndy_max, npts)
X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
time_slices = np.linspace(ndt_min, ndt_max, 5)

predictions = []
for t in time_slices:
    # Create a grid of (x,y,t)
    tt = np.full_like(X_grid, t)
    pts = np.stack([X_grid.flatten(), Y_grid.flatten(), tt.flatten()], axis=1)
    pts_torch = torch.tensor(pts, dtype=torch.float32).to(device)
    with torch.no_grad():
        u_pred = model(pts_torch).cpu().numpy()
    U_pred = u_pred.reshape(npts, npts)
    predictions.append(U_pred)

# Plot the snapshots
fig, axes = plt.subplots(1, len(time_slices), figsize=(15,3))
for i, ax in enumerate(axes):
    cs = ax.contourf(X_grid, Y_grid, predictions[i], levels=50, cmap="viridis")
    ax.set_title(f"t = {time_slices[i]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.colorbar(cs, ax=axes.ravel().tolist(), shrink=0.8)
plt.suptitle("Predicted Field Snapshots")
plt.show()

# Create an animation of the predicted field over time.
fig_anim, ax_anim = plt.subplots(figsize=(6,5))
contour = None

def animate(i):
    global contour
    t = np.linspace(ndt_min, ndt_max, 50)[i]
    tt = np.full_like(X_grid, t)
    pts = np.stack([X_grid.flatten(), Y_grid.flatten(), tt.flatten()], axis=1)
    pts_torch = torch.tensor(pts, dtype=torch.float32).to(device)
    with torch.no_grad():
        u_pred = model(pts_torch).cpu().numpy()
    U_pred = u_pred.reshape(npts, npts)
    ax_anim.clear()
    contour = ax_anim.contourf(X_grid, Y_grid, U_pred, levels=50, cmap="viridis")
    ax_anim.set_title(f"t = {t:.2f}")
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    return contour.collections

anim = animation.FuncAnimation(fig_anim, animate, frames=50, interval=200, blit=False)
# Uncomment one of the lines below to save the animation:
# anim.save("solution_animation.mp4", writer="ffmpeg")
# anim.save("solution_animation.gif", writer="imagemagick")
plt.show()
