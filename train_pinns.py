from modules import *
from pinn_modules import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
torch.autograd.set_detect_anomaly(True)



x_range = [0.0, 15.0] # mm
y_range = [0.0, 4.0] # mm
t_range = [0.0, 6.0] # sec

# Time step for generating major changes (new block install)
time_gap = 0.1 #
t_data = np.arange(t_range[0], t_range[1], time_gap)

# Block geometry parameters
x_len = x_range[-1]  # total length of wall in mm
x_gap = 1.0          # mm (length of each new block)
y_gap = 1.0          # mm (height of each new block)
direction = 1        # start moving from left to right
y_level = 1.0        # mm (initial block top y)

# ctd_data will track the 'center' or 'joint' data of the new block
ctd_data = [[x_gap, y_gap, t_data[0], direction]]

# Create ctd_data for each time step
for ti in t_data[1:]:
    if direction == 1:
        x_value = ctd_data[-1][0] + 1.0
    else:
        x_value = ctd_data[-1][0] - 1.0

    ctd_data.append([x_value, y_level, ti, direction])

    # If we reached the boundary, flip direction and move up in y
    if x_value == x_range[0] or x_value == x_range[1]:
        direction = -direction
        y_level = y_level + y_gap

# Generate corner data for each time step
block_corner_data = []
for x_corner, y_corner, ti, direction in ctd_data:
    new_data = np.array(
        solid_blocks(x_corner, y_corner, direction, x_len, y_gap, x_gap, True)
    )
    block_corner_data.append(new_data)
block_corner_data = np.array(block_corner_data)


new_block_internal_data = generate_new_block_internal_data(block_corner_data, ctd_data, mode='random', dx=0.01, dy=0.01, density=2)
total_mid_times_internal = 10
wall_internal_data = generate_wall_internal_data(block_corner_data, ctd_data, total_mid_times = total_mid_times_internal,
                                                 t_mode = "random", mode = "random", dx=0.01, dy=0.01, density=2)


total_mid_times_boundary = 10
consider_bottom_as_boundary = False
boundary_data = generate_boundary_data(block_corner_data, ctd_data, total_mid_times = total_mid_times_boundary,
                                       t_mode = "random", mode = "random", dx=0.01, dy=0.01,
                                       density=2, consider_bottom_as_boundary = consider_bottom_as_boundary)



total_mid_times_boundary = 10
consider_bottom_as_boundary = False
indi_boundary_data = generate_all_individual_boundary_data(block_corner_data, ctd_data, total_mid_times = total_mid_times_boundary,
                                                           t_mode = "random", mode = "random", dx=0.01, dy=0.01,
                                                           density=2, consider_bottom_as_boundary = consider_bottom_as_boundary)


#############################################################################
# convert to non-dimensionalized form
phi_laser = 2 #mm
v_laser = 10 # mm/s
lc = phi_laser # mm
tc = phi_laser/ v_laser

nd_new_block_internal_data = new_block_internal_data/ np.array([lc, lc, tc])

nd_wall_internal_data = wall_internal_data/ np.array([lc, lc, tc])

nd_indi_boundary_data = {}
for i in indi_boundary_data:
    nd_indi_boundary_data[i] = indi_boundary_data[i] / np.array([lc, lc, tc])


#############################################################################
# convert to ormalized form

ndx_max = 7.5
ndy_max = 2.0
ndt_max = 30
ndx_min = 0
ndy_min = 0
ndt_min = 0


nnd_new_block_internal_data = 2*(nd_new_block_internal_data - np.array([ndx_max, ndy_max, ndt_max])/ \
                              np.array([ndx_max-ndx_min, ndy_max-ndy_min, ndt_max-ndt_min])) - 1.0

nnd_wall_internal_data = 2*(nd_wall_internal_data - np.array([ndx_max, ndy_max, ndt_max])/ \
                              np.array([ndx_max-ndx_min, ndy_max-ndy_min, ndt_max-ndt_min])) - 1.0

nnd_indi_boundary_data = {}
for i in nd_indi_boundary_data:
    nnd_indi_boundary_data[i] = 2*(nd_indi_boundary_data[i] - np.array([ndx_max, ndy_max, ndt_max])/ \
                              np.array([ndx_max-ndx_min, ndy_max-ndy_min, ndt_max-ndt_min])) - 1.0



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

nnd_new_block_internal_data = torch.tensor(nnd_new_block_internal_data, dtype=torch.float32).to(device)
nnd_wall_internal_data = torch.tensor(nnd_wall_internal_data, dtype=torch.float32).to(device)
# boundary_data = torch.tensor(boundary_data, dtype=torch.float32).to(device)

nnd_boundary_data_left = torch.tensor(nnd_indi_boundary_data["left"], dtype=torch.float32).to(device)
nnd_boundary_data_right = torch.tensor(nnd_indi_boundary_data["right"], dtype=torch.float32).to(device)
nnd_boundary_data_shared = torch.tensor(nnd_indi_boundary_data["shared"], dtype=torch.float32).to(device)
nnd_boundary_data_top_left = torch.tensor(nnd_indi_boundary_data["top_left"], dtype=torch.float32).to(device)
nnd_boundary_data_top_right = torch.tensor(nnd_indi_boundary_data["top_right"], dtype=torch.float32).to(device)
if consider_bottom_as_boundary:
    nnd_boundary_data_bottom = torch.tensor(nnd_indi_boundary_data["bottom"], dtype=torch.float32).to(device)

# Ensure x requires gradient
nnd_new_block_internal_data.requires_grad_(True)
nnd_wall_internal_data.requires_grad_(True)
# boundary_data.requires_grad_(True)
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
layers = [3, 20, 40, 40, 20, 1]  # [input_dim=3, hidden_layers..., output_dim=1]
model = PINN(layers).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)



# Example: reducing epochs and sampling to speed up
epochs = 1000    # was 10000
num_samples = 100 # number of random points from each data set per epoch
constant_u = torch.tensor([2900.0/3000.0], dtype=torch.float32).to(device) # initial fix temperature of newly added block


for epoch in range(epochs):
    optimizer.zero_grad()

    # Randomly sample from each dataset
    sampled_wall_internal_data = random_sample(nnd_wall_internal_data, num_samples)
    sampled_new_block_internal_data = random_sample(nnd_new_block_internal_data, num_samples)
    # sampled_boundary_data = random_sample(boundary_data, num_samples)
    sampled_boundary_data_left = random_sample(nnd_boundary_data_left, num_samples)
    sampled_boundary_data_right = random_sample(nnd_boundary_data_right, num_samples)
    sampled_boundary_data_shared = random_sample(nnd_boundary_data_shared, num_samples)
    sampled_boundary_data_top_left = random_sample(nnd_boundary_data_top_left, num_samples)
    sampled_boundary_data_top_right = random_sample(nnd_boundary_data_top_right, num_samples)
    if consider_bottom_as_boundary:
        sampled_boundary_data_bottom = random_sample(nnd_boundary_data_bottom, num_samples)

    # Model predictions at interior (wall) points
    u_wall_internal_data = model(sampled_wall_internal_data)
    x = sampled_wall_internal_data[:, 0:1]
    y = sampled_wall_internal_data[:, 1:2]
    tvar = sampled_wall_internal_data[:, 2:3]

    # Heat source term: For PDE dU/dt - Laplacian(U) = f, define f below
    #   e.g., f = - (Gaussian heat source) - 1.5
    #   The minus sign depends on PDE form; adapt as needed

    f = heat_source_equation(x, y, tvar, 1.0, 0.5, 10.0, 0.0) - 1.5

    # PDE residual for interior
    residual_interior = equation(u_wall_internal_data, sampled_wall_internal_data, f)
    loss_interior = torch.mean(residual_interior**2)
    # 'New block' boundary condition: e.g., T(new block region) = 1.0
    u_new_block_internal_data = model(sampled_new_block_internal_data)
    loss_new_block = torch.mean((u_new_block_internal_data - constant_u)**2)

    # External boundary condition: e.g., T = 0.0 on domain boundary
    # If you want to enforce Dirichlet = 0, uncomment next line:
    # loss_boundary = torch.mean((model(sampled_boundary_data) - 0.0)**2)

    # Currently we set boundary condition as T = 0 as an example:
    # u_boundary_data = model(sampled_boundary_data)
    u_boundary_data_left = model(sampled_boundary_data_left)
    u_boundary_data_right = model(sampled_boundary_data_right)
    u_boundary_data_shared = model(sampled_boundary_data_shared)
    u_boundary_data_top_left = model(sampled_boundary_data_top_left)
    u_boundary_data_top_right = model(sampled_boundary_data_top_right)
    if consider_bottom_as_boundary:
        u_boundary_data_bottom = model(sampled_boundary_data_bottom)
    loss_boundary_left = torch.mean(bc_left(u_boundary_data_left, sampled_boundary_data_left,
                                            h = 0.02, u_amb = 298/3000,
                                            sigma= 5.67*(10**(-11)), epsilon = 0.3,
                                            k = 11.4, lc = 2.0, tc = 3000)**2)

    loss_boundary_right = torch.mean(bc_left(u_boundary_data_right, sampled_boundary_data_right,
                                             h = 0.02, u_amb = 298/3000,
                                             sigma= 5.67*(10**(-11)), epsilon = 0.3,
                                             k = 11.4, lc = 2.0, tc = 3000)**2)

    loss_boundary_shared = torch.mean(bc_left(u_boundary_data_shared, sampled_boundary_data_shared,
                                              h = 0.02, u_amb = 298/3000,
                                              sigma= 5.67*(10**(-11)), epsilon = 0.3,
                                              k = 11.4, lc = 2.0, tc = 3000)**2)

    loss_boundary_top_left = torch.mean(bc_left(u_boundary_data_top_left, sampled_boundary_data_top_left,
                                                h = 0.02, u_amb = 298/3000,
                                                sigma= 5.67*(10**(-11)), epsilon = 0.3,
                                                k = 11.4, lc = 2.0, tc = 3000)**2)

    loss_boundary_top_right = torch.mean(bc_left(u_boundary_data_top_right, sampled_boundary_data_top_right,
                                                 h = 0.02, u_amb = 298/3000,
                                                 sigma= 5.67*(10**(-11)), epsilon = 0.3,
                                                 k = 11.4, lc = 2.0, tc = 3000)**2)

    if consider_bottom_as_boundary:
        loss_boundary_bottom = torch.mean(bc_left(u_boundary_data_bottom, sampled_boundary_data_bottom,
                                                  h = 0.02, u_amb = 298/3000,
                                                  sigma= 5.67*(10**(-11)), epsilon = 0.3,
                                                  k = 11.4, lc = 2.0, tc = 3000)**2)

    # loss_boundary = torch.mean(u_boundary_data**2)  # "0" boundary condition

    loss_boundary = loss_boundary_left + loss_boundary_right + \
                    loss_boundary_shared + loss_boundary_top_left + \
                    loss_boundary_top_right
    if  consider_bottom_as_boundary:
        loss_boundary = loss_boundary + loss_boundary_bottom

    # Total loss
    loss = loss_interior + loss_new_block + loss_boundary

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 100 == 0:  # print every 100 epochs
        print(f"Epoch {epoch}, Loss: {loss.item()}")