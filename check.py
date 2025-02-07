from formatted_2d_datagen import *
import matplotlib.pyplot as plt

# User-specified simulation parameters:
x_range = [0.0, 15.0]   # mm
y_range = [0.0, 4.0]    # mm
t_range = [0.0, 6.0]    # sec
time_gap = 0.1          # sec
x_gap = 1.0             # mm (length of new block)
y_gap = 1.0             # mm (height of new block)
initial_direction = 1   # 1 for left-to-right
initial_y_level = 1.0   # mm
dx = 0.01
dy = 0.01
density = 2           # lower density for faster (random) generation
total_mid_times_internal = 10
total_mid_times_boundary = 10
t_mode = "random"     # can be "uniform" or "random"
mode = "random"       # for point generation mode
consider_bottom_as_boundary = False
phi_laser = 2         # mm (length scale)
v_laser = 10          # mm/s (speed)
nd_bounds = (7.5, 2.0, 30, 0, 0, 0)

# Create an instance of the simulation with the given parameters.
sim = BlockSimulation(
    x_range, y_range, t_range, time_gap, x_gap, y_gap,
    initial_direction, initial_y_level,
    dx=dx, dy=dy, density=density,
    total_mid_times_internal=total_mid_times_internal,
    total_mid_times_boundary=total_mid_times_boundary,
    t_mode=t_mode, mode=mode,
    consider_bottom_as_boundary=consider_bottom_as_boundary,
    phi_laser=phi_laser, v_laser=v_laser,
    nd_bounds=nd_bounds
)

# Process the data (this returns a dictionary with torch tensors).
data = sim.process_data()
print(data["new_block_internal"])

_ = sim.block_corner_data
print(_)
_ = sim.ctd_data
print(_)
# # For example, print the shapes of the generated data:
# print("New Block Internal Data:", data["new_block_internal"].shape)
# print("Wall Internal Data:", data["wall_internal"].shape)
# print("Boundary Left Data:", data["boundary_left"].shape)
# print("Boundary Right Data:", data["boundary_right"].shape)
# print("Boundary Shared Data:", data["boundary_shared"].shape)
# print("Boundary Top Left Data:", data["boundary_top_left"].shape)
# print("Boundary Top Right Data:", data["boundary_top_right"].shape)
# if consider_bottom_as_boundary:
#     print("Boundary Bottom Data:", data["boundary_bottom"].shape)
