from modules import *
from pinn_modules import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
torch.autograd.set_detect_anomaly(True)



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final 3D PINN implementation for dynamic welding heat-transfer.

This file implements:
    - 3D geometry generation (interior and boundary points) for a welding process.
    - Non-dimensionalization and normalization of 4D data: (x,y,z,t).
    - The 3D heat equation residual with a moving Gaussian heat source.
    - Boundary conditions on all six faces (convective–radiative).
    - A PINN network with input dimension 4.
    - A warm-up training phase with AdamW and learning rate scheduling.
    - A refinement phase using a Self-Scaled Broyden (SSBroyden) optimizer.
    - Post-training visualization: loss plots, 2D contour snapshots, 3D scatter, and animation.
    
References:
    - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks...
    - Advanced quasi-Newton methods (e.g. SSBroyden) are reported to enhance PINN training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from matplotlib import animation

# Enable autograd anomaly detection for debugging.
torch.autograd.set_detect_anomaly(True)

###############################################################################
# Advanced Optimizer: Self-Scaled Broyden (SSBroyden)
###############################################################################
class SSBroyden(optim.Optimizer):
    r"""Self-Scaled Broyden quasi-Newton optimizer.
    
    For a flattened parameter vector p, the update is:
      p_{k+1} = p_k - lr * B_k * g_k,
    where g_k is the gradient and B_k is the approximate inverse Hessian updated as:
      B_{k+1} = α * (B_k + ((s_k - B_k y_k) s_k^T) / (s_k^T y_k)),
    with s_k = p_{k+1} - p_k, y_k = g_{k+1} - g_k, and scaling factor α = (s_k^T y_k)/(y_k^T y_k + eps).
    """
    def __init__(self, params, lr=1.0, damping=1e-4):
        defaults = dict(lr=lr, damping=damping)
        super(SSBroyden, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            damping = group['damping']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                n = p.numel()
                p_flat = p.data.view(n)
                grad_flat = grad.view(n)
                if len(state) == 0:
                    state['step'] = 0
                    state['B'] = torch.eye(n, device=p.device, dtype=p.dtype)
                    state['prev_p'] = p_flat.clone()
                    state['prev_grad'] = grad_flat.clone()
                else:
                    prev_p = state['prev_p']
                    prev_grad = state['prev_grad']
                    s = p_flat - prev_p
                    y = grad_flat - prev_grad
                    sy = torch.dot(s, y)
                    if sy.abs() < 1e-10:
                        sy = torch.tensor(1e-10, device=p.device, dtype=p.dtype)
                    s_col = s.unsqueeze(1)
                    y_col = y.unsqueeze(1)
                    B = state['B']
                    By = B @ y_col
                    update = (s_col - By) @ s_col.t() / sy
                    B_new = B + update
                    yt_y = (y_col.t() @ y_col).item()
                    if abs(yt_y) < 1e-10:
                        yt_y = 1e-10
                    alpha = sy.item() / yt_y
                    B_new = alpha * B_new
                    state['B'] = B_new
                    state['prev_p'] = p_flat.clone()
                    state['prev_grad'] = grad_flat.clone()
                update_step = state['B'].mv(grad_flat)
                p_new = p_flat - lr * update_step
                p.data.copy_(p_new.view(p.data.size()))
                state['step'] += 1
        return loss

###############################################################################
# 3D Geometry Generation Functions
###############################################################################
def generate_interior_points(x_min, x_max, y_min, y_max, z_min, z_max, t, 
                               mode='uniform', dx=0.01, dy=0.01, dz=0.01, density=100):
    """Generate interior points in a 3D cuboid at time t."""
    pts = []
    if mode == 'uniform':
        x_vals = np.arange(x_min, x_max, dx)
        y_vals = np.arange(y_min, y_max, dy)
        z_vals = np.arange(z_min, z_max, dz)
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    pts.append([x, y, z, t])
    elif mode == 'random':
        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        n_points = int(volume * density)
        rand_x = np.random.uniform(x_min, x_max, n_points)
        rand_y = np.random.uniform(y_min, y_max, n_points)
        rand_z = np.random.uniform(z_min, z_max, n_points)
        for x, y, z in zip(rand_x, rand_y, rand_z):
            pts.append([x, y, z, t])
    return pts

def generate_boundary_points(x_range, y_range, z_range, t, face, mode='uniform', d=0.01, density=100):
    """Generate boundary points on one face of a cuboid at time t.
       face: one of 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'.
    """
    pts = []
    if face == 'x_min':
        x = x_range[0]
        if mode == 'uniform':
            y_vals = np.arange(y_range[0], y_range[1], d)
            z_vals = np.arange(z_range[0], z_range[1], d)
            for y in y_vals:
                for z in z_vals:
                    pts.append([x, y, z, t])
        elif mode == 'random':
            area = (y_range[1]-y_range[0]) * (z_range[1]-z_range[0])
            n_points = int(area * density)
            rand_y = np.random.uniform(y_range[0], y_range[1], n_points)
            rand_z = np.random.uniform(z_range[0], z_range[1], n_points)
            for y, z in zip(rand_y, rand_z):
                pts.append([x, y, z, t])
    elif face == 'x_max':
        x = x_range[1]
        if mode == 'uniform':
            y_vals = np.arange(y_range[0], y_range[1], d)
            z_vals = np.arange(z_range[0], z_range[1], d)
            for y in y_vals:
                for z in z_vals:
                    pts.append([x, y, z, t])
        elif mode == 'random':
            area = (y_range[1]-y_range[0]) * (z_range[1]-z_range[0])
            n_points = int(area * density)
            rand_y = np.random.uniform(y_range[0], y_range[1], n_points)
            rand_z = np.random.uniform(z_range[0], z_range[1], n_points)
            for y, z in zip(rand_y, rand_z):
                pts.append([x, y, z, t])
    elif face == 'y_min':
        y = y_range[0]
        if mode == 'uniform':
            x_vals = np.arange(x_range[0], x_range[1], d)
            z_vals = np.arange(z_range[0], z_range[1], d)
            for x in x_vals:
                for z in z_vals:
                    pts.append([x, y, z, t])
        elif mode == 'random':
            area = (x_range[1]-x_range[0]) * (z_range[1]-z_range[0])
            n_points = int(area * density)
            rand_x = np.random.uniform(x_range[0], x_range[1], n_points)
            rand_z = np.random.uniform(z_range[0], z_range[1], n_points)
            for x, z in zip(rand_x, rand_z):
                pts.append([x, y, z, t])
    elif face == 'y_max':
        y = y_range[1]
        if mode == 'uniform':
            x_vals = np.arange(x_range[0], x_range[1], d)
            z_vals = np.arange(z_range[0], z_range[1], d)
            for x in x_vals:
                for z in z_vals:
                    pts.append([x, y, z, t])
        elif mode == 'random':
            area = (x_range[1]-x_range[0]) * (z_range[1]-z_range[0])
            n_points = int(area * density)
            rand_x = np.random.uniform(x_range[0], x_range[1], n_points)
            rand_z = np.random.uniform(z_range[0], z_range[1], n_points)
            for x, z in zip(rand_x, rand_z):
                pts.append([x, y, z, t])
    elif face == 'z_min':
        z = z_range[0]
        if mode == 'uniform':
            x_vals = np.arange(x_range[0], x_range[1], d)
            y_vals = np.arange(y_range[0], y_range[1], d)
            for x in x_vals:
                for y in y_vals:
                    pts.append([x, y, z, t])
        elif mode == 'random':
            area = (x_range[1]-x_range[0]) * (y_range[1]-y_range[0])
            n_points = int(area * density)
            rand_x = np.random.uniform(x_range[0], x_range[1], n_points)
            rand_y = np.random.uniform(y_range[0], y_range[1], n_points)
            for x, y in zip(rand_x, rand_y):
                pts.append([x, y, z, t])
    elif face == 'z_max':
        z = z_range[1]
        if mode == 'uniform':
            x_vals = np.arange(x_range[0], x_range[1], d)
            y_vals = np.arange(y_range[0], y_range[1], d)
            for x in x_vals:
                for y in y_vals:
                    pts.append([x, y, z, t])
        elif mode == 'random':
            area = (x_range[1]-x_range[0]) * (y_range[1]-y_range[0])
            n_points = int(area * density)
            rand_x = np.random.uniform(x_range[0], x_range[1], n_points)
            rand_y = np.random.uniform(y_range[0], y_range[1], n_points)
            for x, y in zip(rand_x, rand_y):
                pts.append([x, y, z, t])
    return pts

###############################################################################
# Governing Equation and Boundary Conditions (3D version)
###############################################################################
def equation(u, X, f, k=11.4, rho=4.5e-9, cp=7.14e8, tc=0.2, lc=2):
    """
    3D PDE Residual: 
      ∂u/∂t - (k/(ρ cₚ))*(tc/lc²)*(∂²u/∂x²+∂²u/∂y²+∂²u/∂z²) - f = 0
    X: (N,4) tensor with columns (x, y, z, t)
    """
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_d[:,0], X, grad_outputs=torch.ones_like(du_d[:,0]), create_graph=True)[0][:,0:1]
    d2u_dy2 = torch.autograd.grad(du_d[:,1], X, grad_outputs=torch.ones_like(du_d[:,1]), create_graph=True)[0][:,1:2]
    d2u_dz2 = torch.autograd.grad(du_d[:,2], X, grad_outputs=torch.ones_like(du_d[:,2]), create_graph=True)[0][:,2:3]
    d_u_dt  = du_d[:,3:4]
    return d_u_dt - (k/(rho*cp))*(tc/(lc**2))*(d2u_dx2 + d2u_dy2 + d2u_dz2) - f

# Boundary conditions on the six faces.
def bc_xmin(u, X, h=0.02, u_amb=298/3000, sigma=5.67e-11, epsilon=0.3,
            k=11.4, lc=2.0, tc=3000):
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx = du_d[:,0:1]
    conv = (h*lc/k)*(u - u_amb)
    rad = (sigma*epsilon*(tc**3)*lc/k)*(u**4 - u_amb**4)
    return du_dx - conv - rad

def bc_xmax(u, X, h=0.02, u_amb=298/3000, sigma=5.67e-11, epsilon=0.3,
            k=11.4, lc=2.0, tc=3000):
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx = du_d[:,0:1]
    conv = (h*lc/k)*(u - u_amb)
    rad = (sigma*epsilon*(tc**3)*lc/k)*(u**4 - u_amb**4)
    return -du_dx - conv - rad

def bc_ymin(u, X, h=0.02, u_amb=298/3000, sigma=5.67e-11, epsilon=0.3,
            k=11.4, lc=2.0, tc=3000):
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dy = du_d[:,1:2]
    conv = (h*lc/k)*(u - u_amb)
    rad = (sigma*epsilon*(tc**3)*lc/k)*(u**4 - u_amb**4)
    return -du_dy - conv - rad

def bc_ymax(u, X, h=0.02, u_amb=298/3000, sigma=5.67e-11, epsilon=0.3,
            k=11.4, lc=2.0, tc=3000):
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dy = du_d[:,1:2]
    conv = (h*lc/k)*(u - u_amb)
    rad = (sigma*epsilon*(tc**3)*lc/k)*(u**4 - u_amb**4)
    return du_dy - conv - rad

def bc_zmin(u, X, h=0.02, u_amb=298/3000, sigma=5.67e-11, epsilon=0.3,
            k=11.4, lc=2.0, tc=3000):
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dz = du_d[:,2:3]
    conv = (h*lc/k)*(u - u_amb)
    rad = (sigma*epsilon*(tc**3)*lc/k)*(u**4 - u_amb**4)
    return -du_dz - conv - rad

def bc_zmax(u, X, h=0.02, u_amb=298/3000, sigma=5.67e-11, epsilon=0.3,
            k=11.4, lc=2.0, tc=3000):
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dz = du_d[:,2:3]
    conv = (h*lc/k)*(u - u_amb)
    rad = (sigma*epsilon*(tc**3)*lc/k)*(u**4 - u_amb**4)
    return du_dz - conv - rad

###############################################################################
# Data Generation and Normalization
###############################################################################
# Domain parameters (units: mm for space, sec for time)
x_range = [0.0, 15.0]   # x-direction
y_range = [0.0, 4.0]    # y-direction
z_range = [0.0, 1.0]    # z-direction (thickness)
t_range = [0.0, 6.0]    # time

# For dynamic welding geometry, one may use a moving weld bead. Here we follow the
# original 2D dynamic geometry logic and then extrude it in z.
time_gap = 0.1
t_data = np.arange(t_range[0], t_range[1], time_gap)
x_len = x_range[1]
x_gap = 1.0
y_gap = 1.0
direction = 1
y_level = 1.0
# ctd_data tracks the 2D weld bead joint (in x-y) over time
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

# Generate 2D block corners as before and extrude them in z (here we use the full z_range).
block_corner_data_2D = []
for x_corner, y_corner, ti, direction in ctd_data:
    corners = solid_blocks(x_corner, y_corner, direction, x_len, y_gap, x_gap, True)
    # Each block is represented by its "new block" 2D region (using the third element).
    block_corner_data_2D.append(np.array(corners[2]))  # shape (4,2)
block_corner_data_2D = np.array(block_corner_data_2D)

# For interior data, we generate points in the weld volume over time.
# (For simplicity, here we generate points over the entire domain.)
total_mid_times_internal = 10
# Generate interior points for a set of time samples.
wall_interior_points = []
for t in t_data:
    pts = generate_interior_points(x_range[0], x_range[1],
                                   y_range[0], y_range[1],
                                   z_range[0], z_range[1],
                                   t, mode='random', density=2)
    wall_interior_points += pts
wall_interior_data = np.array(wall_interior_points)

# For the new weld bead (new block), we generate points in the 2D region extruded in z.
new_block_points = []
for poly in block_corner_data_2D:
    # Determine bounding box of the polygon:
    x_vals = poly[:,0]
    y_vals = poly[:,1]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    # Use a fixed time for each new block (from ctd_data)
    # Here we simply loop over the 2D blocks and extrude:
    idx = len(new_block_points) % len(ctd_data)  # for demonstration
    t_val = ctd_data[idx][2]
    pts = generate_interior_points(x_min, x_max, y_min, y_max, z_range[0], z_range[1],
                                   t_val, mode='random', density=2)
    new_block_points += pts
new_block_data = np.array(new_block_points)

# For boundary data, generate points on each of the six faces for a few time instances.
total_mid_times_boundary = 10
boundary_points = {"x_min": [], "x_max": [], "y_min": [], "y_max": [], "z_min": [], "z_max": []}
for t in np.linspace(t_range[0], t_range[1], total_mid_times_boundary):
    boundary_points["x_min"] += generate_boundary_points(x_range, y_range, z_range, t, 'x_min', mode='random', density=2)
    boundary_points["x_max"] += generate_boundary_points(x_range, y_range, z_range, t, 'x_max', mode='random', density=2)
    boundary_points["y_min"] += generate_boundary_points(x_range, y_range, z_range, t, 'y_min', mode='random', density=2)
    boundary_points["y_max"] += generate_boundary_points(x_range, y_range, z_range, t, 'y_max', mode='random', density=2)
    boundary_points["z_min"] += generate_boundary_points(x_range, y_range, z_range, t, 'z_min', mode='random', density=2)
    boundary_points["z_max"] += generate_boundary_points(x_range, y_range, z_range, t, 'z_max', mode='random', density=2)

# Non-dimensionalization using characteristic scales
phi_laser = 2   # mm
v_laser = 10    # mm/s
lc_val = phi_laser
tc_val = phi_laser / v_laser

nd_wall_interior = wall_interior_data / np.array([lc_val, lc_val, lc_val, tc_val])
nd_new_block = new_block_data / np.array([lc_val, lc_val, lc_val, tc_val])
nd_boundary = {}
for key in boundary_points:
    nd_boundary[key] = np.array(boundary_points[key]) / np.array([lc_val, lc_val, lc_val, tc_val])

# Normalization to [-1, 1] for each coordinate.
ndx_min, ndx_max = 0, 7.5
ndy_min, ndy_max = 0, 2.0
ndz_min, ndz_max = 0, 1.0
ndt_min, ndt_max = 0, 30
def normalize_to_nn(data, lower, upper):
    return 2 * ((data - lower) / (upper - lower)) - 1.0

lower_bound = np.array([ndx_min, ndy_min, ndz_min, ndt_min])
upper_bound = np.array([ndx_max, ndy_max, ndz_max, ndt_max])

nnd_wall_interior = normalize_to_nn(nd_wall_interior, lower_bound, upper_bound)
nnd_new_block = normalize_to_nn(nd_new_block, lower_bound, upper_bound)
nnd_boundary = {}
for key in nd_boundary:
    nnd_boundary[key] = normalize_to_nn(nd_boundary[key], lower_bound, upper_bound)

# Convert to Torch tensors and enable gradient for interior and new block data.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
nnd_wall_interior = torch.tensor(nnd_wall_interior, dtype=torch.float32).to(device)
nnd_new_block = torch.tensor(nnd_new_block, dtype=torch.float32).to(device)
for key in nnd_boundary:
    nnd_boundary[key] = torch.tensor(nnd_boundary[key], dtype=torch.float32).to(device)

nnd_wall_interior.requires_grad_(True)
nnd_new_block.requires_grad_(True)
for key in nnd_boundary:
    nnd_boundary[key].requires_grad_(True)

###############################################################################
# PINN Model Definition (Input dimension is now 4: x, y, z, t)
###############################################################################
layers = [4, 20, 40, 40, 20, 1]
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                      for i in range(len(layers)-1)])
    def forward(self, x):
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        return self.linears[-1](x)
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

model = PINN(layers).to(device)

###############################################################################
# Training: AdamW Warm-up followed by SSBroyden Refinement
###############################################################################
# Warm-up using AdamW
adamw_optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(adamw_optimizer, mode='min', factor=0.5, patience=500, verbose=True)
constant_u = torch.tensor([2900.0/3000.0], dtype=torch.float32).to(device)
epochs = 5000
num_samples = 500
loss_history = []
loss_components = {"interior": [], "new_block": [], "boundary": []}

def heat_source_origin(current_time, travel_speed, jump, x_max, x_min):
    one_lap = (x_max - x_min) / travel_speed
    current_lap = (current_time / one_lap).to(torch.int32)
    current_y = (current_lap + 1) * jump
    direction = current_lap % 2
    current_x = (travel_speed * current_time) % (x_max - x_min)
    mask = (direction == 1)
    current_x[mask] = x_max - current_x[mask]
    return [current_x, current_y]

def heat_source_equation(x, y, z, current_time, travel_speed, jump, x_max, x_min):
    # For 3D, we assume the heat source acts on the top surface (for example, z near z_max)
    origin_x, origin_y = heat_source_origin(current_time, travel_speed, jump, x_max, x_min)
    # For simplicity, the Gaussian depends on (x - origin_x)^2 + (y - origin_y)^2; 
    # one may extend it to z if needed.
    f_val = torch.exp(-((x - origin_x)**2 + (y - origin_y)**2))
    # Optionally, set heat source to zero above a certain z (or below the weld bead)
    f_val[z > 0.9] = 0  # example condition
    return f_val

for epoch in range(epochs):
    adamw_optimizer.zero_grad()
    sampled_wall = nnd_wall_interior[random.sample(range(nnd_wall_interior.shape[0]), num_samples)]
    sampled_new_block = nnd_new_block[random.sample(range(nnd_new_block.shape[0]), num_samples)]
    # For boundary, here we only use one representative face (e.g., x_min) for simplicity.
    sampled_b_xmin = nnd_boundary["x_min"][random.sample(range(nnd_boundary["x_min"].shape[0]), num_samples)]
    
    # Interior loss
    u_interior = model(sampled_wall)
    x = sampled_wall[:, 0:1]
    y = sampled_wall[:, 1:2]
    z = sampled_wall[:, 2:3]
    tvar = sampled_wall[:, 3:4]
    f = heat_source_equation(x, y, z, tvar, travel_speed=1.0, jump=0.5, x_max=10.0, x_min=0.0) - 1.5
    res = equation(u_interior, sampled_wall, f)
    loss_interior = torch.mean(res**2)
    
    # New block loss: enforcing a fixed temperature in the weld bead region.
    u_new = model(sampled_new_block)
    loss_new = torch.mean((u_new - constant_u)**2)
    
    # Boundary loss (using the x_min face)
    u_b_xmin = model(sampled_b_xmin)
    loss_b_xmin = torch.mean(bc_xmin(u_b_xmin, sampled_b_xmin)**2)
    
    total_loss = loss_interior + loss_new + loss_b_xmin
    total_loss.backward()
    adamw_optimizer.step()
    scheduler.step(total_loss)
    
    loss_history.append(total_loss.item())
    loss_components["interior"].append(loss_interior.item())
    loss_components["new_block"].append(loss_new.item())
    loss_components["boundary"].append(loss_b_xmin.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item():.5f}")

# Refinement with SSBroyden optimizer.
def closure():
    ssbroyden_optimizer.zero_grad()
    u_interior = model(nnd_wall_interior)
    x = nnd_wall_interior[:, 0:1]
    y = nnd_wall_interior[:, 1:2]
    z = nnd_wall_interior[:, 2:3]
    tvar = nnd_wall_interior[:, 3:4]
    f = heat_source_equation(x, y, z, tvar, travel_speed=1.0, jump=0.5, x_max=10.0, x_min=0.0) - 1.5
    loss_interior = torch.mean(equation(u_interior, nnd_wall_interior, f)**2)
    
    u_new = model(nnd_new_block)
    loss_new = torch.mean((u_new - constant_u)**2)
    
    u_b_xmin = model(nnd_boundary["x_min"])
    loss_b_xmin = torch.mean(bc_xmin(u_b_xmin, nnd_boundary["x_min"])**2)
    
    loss = loss_interior + loss_new + loss_b_xmin
    loss.backward()
    return loss

ssbroyden_optimizer = SSBroyden(model.parameters(), lr=1.0, damping=1e-4)
ssbroyden_optimizer.step(closure)

###############################################################################
# Post-Training Visualization and Analysis
###############################################################################
df_losses = pd.DataFrame([{"Interior Loss": loss_components["interior"][-1],
                            "New Block Loss": loss_components["new_block"][-1],
                            "Boundary Loss": loss_components["boundary"][-1],
                            "Total Loss": loss_history[-1]}])
print("Final Loss Components:")
print(df_losses)

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

# 2D Contour Visualization at a fixed z-slice and selected times.
npts = 50
x_lin = np.linspace(ndx_min, ndx_max, npts)
y_lin = np.linspace(ndy_min, ndy_max, npts)
X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
# Choose a fixed z-slice (e.g., z = (ndz_min+ndz_max)/2)
fixed_z = (ndz_min + ndz_max) / 2
time_slices = np.linspace(ndt_min, ndt_max, 5)
predictions = []
for t in time_slices:
    tt = np.full_like(X_grid, t)
    zz = np.full_like(X_grid, fixed_z)
    pts = np.stack([X_grid.flatten(), Y_grid.flatten(), zz.flatten(), tt.flatten()], axis=1)
    pts_torch = torch.tensor(pts, dtype=torch.float32).to(device)
    with torch.no_grad():
        u_pred = model(pts_torch).cpu().numpy()
    U_pred = u_pred.reshape(npts, npts)
    predictions.append(U_pred)
fig, axes = plt.subplots(1, len(time_slices), figsize=(15, 3))
for i, ax in enumerate(axes):
    cs = ax.contourf(X_grid, Y_grid, predictions[i], levels=50, cmap="viridis")
    ax.set_title(f"t = {time_slices[i]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.colorbar(cs, ax=axes.ravel().tolist(), shrink=0.8)
plt.suptitle("Predicted Temperature (Fixed z-slice)")
plt.show()

# 3D Visualization: Example scatter plot on the top surface (z=z_max)
def plot_top_surface(model, x_range, y_range, z_value, t_val, d=0.05):
    x_vals = np.arange(x_range[0], x_range[1], d)
    y_vals = np.arange(y_range[0], y_range[1], d)
    pts = []
    for x in x_vals:
        for y in y_vals:
            pts.append([x, y, z_value, t_val])
    pts = np.array(pts)
    pts_torch = torch.tensor(pts, dtype=torch.float32).to(device)
    with torch.no_grad():
        u_pred = model(pts_torch).cpu().numpy().flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=u_pred, cmap="jet")
    ax.set_title(f"Top Surface Temperature at t={t_val:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.colorbar(sc, label="Temperature")
    plt.show()

plot_top_surface(model, [ndx_min, ndx_max], [ndy_min, ndy_max], ndz_max, t_val=ndt_max)

# Animation: Evolve the predicted field on a fixed z-slice over time.
fig_anim, ax_anim = plt.subplots(figsize=(6,5))
def animate(i):
    t = np.linspace(ndt_min, ndt_max, 50)[i]
    tt = np.full_like(X_grid, t)
    zz = np.full_like(X_grid, fixed_z)
    pts = np.stack([X_grid.flatten(), Y_grid.flatten(), zz.flatten(), tt.flatten()], axis=1)
    pts_torch = torch.tensor(pts, dtype=torch.float32).to(device)
    with torch.no_grad():
        u_pred = model(pts_torch).cpu().numpy()
    U_pred = u_pred.reshape(npts, npts)
    ax_anim.clear()
    cont = ax_anim.contourf(X_grid, Y_grid, U_pred, levels=50, cmap="viridis")
    ax_anim.set_title(f"t = {t:.2f}")
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    return cont.collections

anim = animation.FuncAnimation(fig_anim, animate, frames=50, interval=200, blit=False)
# To save the animation, uncomment one of the following:
# anim.save("solution_animation.mp4", writer="ffmpeg")
# anim.save("solution_animation.gif", writer="imagemagick")
plt.show()

print("3D Inference and visualization complete.")
