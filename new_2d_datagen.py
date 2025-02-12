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
        print(x_dis, y_dis)
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
    else:
        area = max(dx, dy)
        data_gen_line_rectangle(mode, density, dx, dy, xmin, xmax, ymin, ymax)

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
    boundary_data = np.empty((0, 2))
    left_y = mid_xy[1] if current_direction == 0 else mid_xy[1] - block_dy
    right_y = mid_xy[1] if current_direction == 1 else mid_xy[1] - block_dy
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    left_boundary_data = gen_data_in_line(
        [0, 0], 
        [0, mid_xy[1]], 
        mode=mode, 
        density=density)
        np.vstack((wall_data, gen_data_in_rectangle([0, x_max], 
                                                                [0, mid_xy[1] - block_dy], 
                                                                mode=mode, 
                                                                density=lower_density)))
    if current_direction == 0:
        wall_data = np.vstack((wall_data, gen_data_in_rectangle([0, mid_xy[0] + block_dx], 
                                                                [mid_xy[1] - block_dy, mid_xy[1]], 
                                                                mode=mode, 
                                                                density=upper_density)))
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


    
    
    
print(get_t_wall_data(0, 1, 1))



# _ = gen_data_in_rectangle([0, 1], [0, 1], mode="uniform", density=100)
# plt.figure()
# plt.scatter(_[:,0], _[:,1])
# plt.show()