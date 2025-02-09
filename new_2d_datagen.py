def get_pointer_cords(
        time, x_speed, y_speed, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 10, 
        y_max = 10, 
        block_dx = 0.5, 
        block_dy = 0.5,
        start_t = 0.0
        ):
    """
    pointer start from mid of new drop and ends at mid of last drop of each bead.
    one_bead_time is the total time taken by the pointer to cover the distance between 
    first drop to last drop of one bead plus the time to reach the mid point of start drop od next bead.
    """
    x_travel_time = (x_max-block_dx)/x_speed
    y_travel_time = block_dy/y_speed
    one_bead_time = x_travel_time + y_travel_time
    total_full_beads = int((time-start_t)/one_bead_time)
    """
    positive x direction means 0
    negative x direction means 1
    """
    if movement_type == "unidirectional":
        current_direction = 0
    elif movement_type == "bidirectional":
        current_direction = total_full_beads%2
    else:
        raise ValueError("Define correct movement_type")

    ongoing_bead_time = (time-start_t) - total_full_beads*one_bead_time

    x_dis = block_dx/2 + min([ongoing_bead_time, x_travel_time])*x_speed
    if current_direction == 1:
        x_dis = x_max - x_dis
    y_dis = (total_full_beads+1)*block_dy + max([0, ongoing_bead_time-x_travel_time])*y_speed
    if y_dis > y_max:
        raise ValueError("maximum y is achieved. Decrease time or increase maximum y")

    if (x_dis- block_dx/2) % block_dx < 1e-5 : # to get rid of float precision
        new_block = True
    else:
        new_block = False 
    if if_continuous:
        return [x_dis, y_dis], current_direction, new_block
    elif if_discrete:
        print(x_dis, y_dis)
        x_dis = (x_dis//block_dx)*block_dx + block_dx/2 
        y_dis = (y_dis//block_dy)*block_dy
        return [x_dis, y_dis], current_direction, new_block
    else:
        raise ValueError("Both if_continuous and if_discrete can not be False")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate Data from an External Function
def get_data(frame):
    """Returns x and y data for each frame."""
    x = np.linspace(0, 10, 100)
    if frame % 2 == 0:
        y = np.sin(x + frame / 10)  # Line plot data
        plot_type = "line"
    else:
        y = np.random.rand(len(x))  # Scatter plot data
        plot_type = "scatter"
    return x, y, plot_type

# Create Figure and Axis
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# Initialize line and scatter plot
line, = ax.plot([], [], lw=2, label="Sine Wave")  
scatter = ax.scatter([], [], color='red', label="Random Points")

# Initialization Function
def init():
    line.set_data([], [])
    scatter.set_offsets(np.empty((0, 2)))
    return line, scatter

# Update Function
def update(frame):
    x, y, plot_type = get_data(frame)

    if plot_type == "line":
        line.set_data(x, y)
        scatter.set_offsets(np.empty((0, 2)))
    else:
        scatter.set_offsets(np.c_[x, y])  # Set scatter points
        line.set_data([], [])  # Hide line
    
    return line, scatter

# Create Animation
ani = animation.FuncAnimation(fig, update, frames=100, init_func=init, blit=True)

# Show Animation
plt.legend()
plt.show()
