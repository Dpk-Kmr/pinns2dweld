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
print(nd_indi_boundary_data)
for i in indi_boundary_data:
    nnd_indi_boundary_data[i] = 2*(nd_indi_boundary_data - np.array([ndx_max, ndy_max, ndt_max])/ \
                              np.array([ndx_max-ndx_min, ndy_max-ndy_min, ndt_max-ndt_min])) - 1.0
