import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

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
        x_dis = (x_dis//block_dx)*block_dx + block_dx/2 
        y_dis = (y_dis//block_dy)*block_dy
        return [x_dis, y_dis], current_direction, new_block
    else:
        raise ValueError("Both if_continuous and if_discrete can not be False")


def data_gen_line_rectangle(mode, density, dx, dy, xmin, xmax, ymin, ymax):
    if mode == "random":
        # Total points is density times area
        total_points = int(density * max(dx, dy))
        xs = np.random.uniform(xmin, xmax, total_points)
        ys = np.random.uniform(ymin, ymax, total_points)
        points = np.column_stack((xs, ys))
    elif mode == "uniform":
        # To obtain roughly the desired density, we set grid spacing h ~ 1/sqrt(density)
        spacing = 1 / np.sqrt(density)
        # Compute number of points along each axis. Adding 1 ensures the boundaries are included.
        n_x = int(np.floor(dx / spacing)) + 1
        n_y = int(np.floor(dy / spacing)) + 1
        xs = np.linspace(xmin, xmax, n_x)
        ys = np.linspace(ymin, ymax, n_y)
        xv, yv = np.meshgrid(xs, ys)
        points = np.column_stack((xv.ravel(), yv.ravel()))
    else:
        raise ValueError("Mode must be either 'random' or 'uniform'.")

    return points

def gen_data_in_rectangle(x_range, y_range, mode="random", density=100):
    """
    Generate data points within a given rectangle.
    
    Parameters
    ----------
    x_range : tuple of floats
        (xmin, xmax) boundaries of the rectangle.
    y_range : tuple of floats
        (ymin, ymax) boundaries of the rectangle.
    mode : str, optional
        "random" or "uniform". 
        - "random": points are randomly distributed over the rectangle.
        - "uniform": points are arranged on a uniform grid.
    density : float, optional
        Number of data points per unit area.
        
    Returns
    -------
    points : numpy.ndarray
        Array of shape (N, 2) containing the generated (x, y) data points.
    """
    xmin, xmax = min(x_range), max(x_range)
    ymin, ymax = min(y_range), max(y_range)

    dx = xmax - xmin
    dy = ymax - ymin
    area = dx * dy
    if area == 0.0:
        print("expected a rectangle, but got line")
        return np.empty((0, 2))
    else:
        return data_gen_line_rectangle(mode, density, dx, dy, xmin, xmax, ymin, ymax)
    

def gen_data_in_line(x_range, y_range, mode="random", density=100):
    """
    Generate data points within a given rectangle.
    
    Parameters
    ----------
    x_range : tuple of floats
        (xmin, xmax) boundaries of the rectangle.
    y_range : tuple of floats
        (ymin, ymax) boundaries of the rectangle.
    mode : str, optional
        "random" or "uniform". 
        - "random": points are randomly distributed over the rectangle.
        - "uniform": points are arranged on a uniform grid.
    density : float, optional
        Number of data points per unit area.
        
    Returns
    -------
    points : numpy.ndarray
        Array of shape (N, 2) containing the generated (x, y) data points.
    """
    xmin, xmax = min(x_range), max(x_range)
    ymin, ymax = min(y_range), max(y_range)

    dx = xmax - xmin
    dy = ymax - ymin
    area = dx * dy
    if area != 0.0:
        print("expected a line, but got rectangle")
        return np.empty((0, 2))
    elif dy == 0.0 and dx == 0.0:
        print("expected a line, but got a point")
        return np.empty((0, 2))
    else:
        area = max(dx, dy)
        return data_gen_line_rectangle(mode, density, dx, dy, xmin, xmax, ymin, ymax)

def get_t_wall_data(
        time, x_speed, y_speed, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 10, 
        y_max = 10, 
        block_dx = 0.5, 
        block_dy = 0.5,
        start_t = 0.0, 
        mode = "random", 
        lower_density = 100, 
        upper_density = 100,
        new_density = 100,
        increase_latest_block_data = False,
):
    mid_xy, current_direction, new_block = get_pointer_cords(
        time, x_speed, y_speed, 
        movement_type = movement_type, 
        if_continuous = if_continuous, 
        if_discrete = if_discrete, 
        x_max = x_max, 
        y_max = y_max, 
        block_dx = block_dx, 
        block_dy = block_dy,
        start_t = start_t
        )
    wall_data = np.empty((0, 2))
    # get data in lower reactangle
    if  mid_xy[1] - block_dy != 0:
        wall_data = np.vstack((wall_data, gen_data_in_rectangle([0, x_max], 
                                                                [0, mid_xy[1] - block_dy], 
                                                                mode=mode, 
                                                                density=lower_density)))
    # get data if moving forward
    if current_direction == 0:
        wall_data = np.vstack((wall_data, gen_data_in_rectangle([0, mid_xy[0] + block_dx], 
                                                                [mid_xy[1] - block_dy, mid_xy[1]], 
                                                                mode=mode, 
                                                                density=upper_density)))
    # get data if moving backward
    if current_direction == 1:
        wall_data = np.vstack((wall_data, gen_data_in_rectangle([mid_xy[0] - block_dx, x_max], 
                                                                [mid_xy[1] - block_dy, mid_xy[1]], 
                                                                mode=mode, 
                                                                density=upper_density)))
    if increase_latest_block_data:
        wall_data = np.vstack((wall_data, gen_data_in_rectangle([mid_xy[0] - block_dx, mid_xy[0] + block_dx], 
                                                                [mid_xy[1] - block_dy, mid_xy[1]], 
                                                                mode=mode, 
                                                                density=new_density)))

    time_col = np.full((wall_data.shape[0], 1), time)
    return np.hstack((time_col, wall_data))

def get_t_boundary_data(
        time, x_speed, y_speed, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 10, 
        y_max = 10, 
        block_dx = 0.5, 
        block_dy = 0.5,
        start_t = 0.0, 
        mode = "random", 
        density = 1000, 
        bottom_data = True,
):
    mid_xy, current_direction, new_block = get_pointer_cords(
        time, x_speed, y_speed, 
        movement_type = movement_type, 
        if_continuous = if_continuous, 
        if_discrete = if_discrete, 
        x_max = x_max, 
        y_max = y_max, 
        block_dx = block_dx, 
        block_dy = block_dy,
        start_t = start_t
        )
    
    left_y = mid_xy[1] if current_direction == 0 else mid_xy[1] - block_dy
    right_y = mid_xy[1] if current_direction == 1 else mid_xy[1] - block_dy
    left_boundary_data = gen_data_in_line(
        [0, 0], 
        [0, left_y], 
        mode=mode, 
        density=density)
    right_boundary_data = gen_data_in_line(
        [0, 0], 
        [0, right_y], 
        mode=mode, 
        density=density)
    if current_direction == 0:
        top_left_boundary_data = gen_data_in_line(
            [0, mid_xy[0] + block_dx/2], 
            [left_y, left_y], 
            mode=mode, 
            density=density)
        top_right_boundary_data = gen_data_in_line(
            [mid_xy[0] + block_dx/2, x_max], 
            [left_y, left_y], 
            mode=mode, 
            density=density)
        mid_boundary_data = gen_data_in_line(
            [mid_xy[0] + block_dx/2, mid_xy[0] + block_dx/2], 
            [mid_xy[1] - block_dy, mid_xy[1]], 
            mode=mode, 
            density=density)
    else:
        top_left_boundary_data = gen_data_in_line(
            [0, mid_xy[0] - block_dx/2], 
            [left_y, left_y], 
            mode=mode, 
            density=density)
        top_right_boundary_data = gen_data_in_line(
            [mid_xy[0] - block_dx/2, x_max], 
            [left_y, left_y], 
            mode=mode, 
            density=density)
        mid_boundary_data = gen_data_in_line(
            [mid_xy[0] - block_dx/2, mid_xy[0] - block_dx/2], 
            [mid_xy[1] - block_dy, mid_xy[1]], 
            mode=mode, 
            density=density)
    if bottom_data:
        if mid_xy[1] != block_dy:
            bottom_boundary_data = gen_data_in_line(
                [0, x_max], 
                [0, 0], 
                mode=mode, 
                density=density)
        else:
            bottom_boundary_data = gen_data_in_line(
                [0, mid_xy[0]+block_dx/2], 
                [0, 0], 
                mode=mode, 
                density=density)
        bottom_boundary_data = np.hstack((np.full((bottom_boundary_data.shape[0], 1), time), bottom_boundary_data))
    else:
        bottom_boundary_data = np.empty((0, 3))
    left_boundary_data = np.hstack((np.full((left_boundary_data.shape[0], 1), time), left_boundary_data))
    right_boundary_data = np.hstack((np.full((right_boundary_data.shape[0], 1), time), right_boundary_data))
    mid_boundary_data = np.hstack((np.full((mid_boundary_data.shape[0], 1), time), mid_boundary_data))
    top_left_boundary_data = np.hstack((np.full((top_left_boundary_data.shape[0], 1), time), top_left_boundary_data))
    top_right_boundary_data = np.hstack((np.full((top_right_boundary_data.shape[0], 1), time), top_right_boundary_data))
    left_boundary_data = np.hstack((np.full((left_boundary_data.shape[0], 1), time), left_boundary_data))
    
    return (left_boundary_data,
            right_boundary_data,
            mid_boundary_data,
            top_left_boundary_data, 
            top_right_boundary_data, 
            bottom_boundary_data)




# Define figure dimensions
x_max, y_max = 10, 10

# Prepare the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_title("Welding Wall")
ax.grid(False)

# Initialize scatter plots
scatter_wall = ax.scatter([], [], s=6, c="black")
scatter_b0 = ax.scatter([], [], s=6, c="pink")
scatter_b1 = ax.scatter([], [], s=6, c="red")
scatter_b2 = ax.scatter([], [], s=6, c="blue")
scatter_b3 = ax.scatter([], [], s=6, c="green")
scatter_b4 = ax.scatter([], [], s=6, c="yellow")
scatter_b5 = ax.scatter([], [], s=6, c="orange")


def get_data(frame):
    t = frame
    all_boundary_data = get_t_boundary_data(t, 1, 1)
    wall_data = get_t_wall_data(t, 1, 1)
    return (
        wall_data[:,1:],
        all_boundary_data[0][:,1:],
        all_boundary_data[1][:,1:],
        all_boundary_data[2][:,1:],
        all_boundary_data[3][:,1:],
        all_boundary_data[4][:,1:],
        all_boundary_data[5][:,1:]
    )

def init():
    scatter_wall.set_offsets([], [])
    scatter_b0.set_offsets([], [])
    scatter_b1.set_offsets([], [])
    scatter_b2.set_offsets([], [])
    scatter_b3.set_offsets([], [])
    scatter_b4.set_offsets([], [])
    scatter_b5.set_offsets([], [])
    return scatter_wall, scatter_b0, scatter_b1, scatter_b2, scatter_b3, scatter_b4, scatter_b5

def update(frame):
    data = get_data(frame)

    # Update scatter plots
    scatter_wall.set_offsets(data[0][:,0], data[0][:,1])
    scatter_b0.set_offsets(data[1][:,0], data[1][:,1])
    scatter_b1.set_offsets(data[2][:,0], data[2][:,1])
    scatter_b2.set_offsets(data[3][:,0], data[3][:,1])
    scatter_b3.set_offsets(data[4][:,0], data[4][:,1])
    scatter_b4.set_offsets(data[5][:,0], data[5][:,1])
    scatter_b5.set_offsets(data[6][:,0], data[6][:,1])

    return scatter_wall, scatter_b0, scatter_b1, scatter_b2, scatter_b3, scatter_b4, scatter_b5

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=25, init_func=init,
                               interval=1000, blit=True, repeat=False)

plt.show()

