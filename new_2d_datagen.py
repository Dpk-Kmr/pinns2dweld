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
        _x_dis = (x_dis//block_dx)*block_dx + block_dx/2 
        _y_dis = (y_dis//block_dy)*block_dy
        return [_x_dis, _y_dis], current_direction, new_block, [x_dis, y_dis]
    else:
        raise ValueError("Both if_continuous and if_discrete can not be False")


def data_gen_line_rectangle(mode, density, dx, dy, xmin, xmax, ymin, ymax, area):
    if mode == "random":
        # Total points is density times area
        total_points = int(density * area)
        xs = np.random.uniform(xmin, xmax, total_points)
        ys = np.random.uniform(ymin, ymax, total_points)
        points = np.column_stack((xs, ys))
    elif mode == "uniform":
        # To obtain roughly the desired density, we set grid spacing h ~ 1/sqrt(density)
        spacing = 1 / np.sqrt(density) if dx*dy != 0 else 1/density
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
        return data_gen_line_rectangle(mode, density, dx, dy, xmin, xmax, ymin, ymax, area)
    

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
        return data_gen_line_rectangle(mode, density, dx, dy, xmin, xmax, ymin, ymax, area)

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
        wall_density = 10, 
        new_density = 100,
        boundary_density = 50,
        increase_latest_block_data = True,
        increased_boundary_data = True,
        boundary_width = None, 
        top_boundary_layers = 2
):
    if boundary_width == None:
        boundary_width = block_dy
    top_boundary_width = top_boundary_layers*block_dy
    mid_xy, current_direction, new_block, heat_xy = get_pointer_cords(
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
                                                                density=wall_density)))
    # get data if moving forward
    if current_direction == 0:
        # get data in upper rectangle
        wall_data = np.vstack((wall_data, gen_data_in_rectangle([0, mid_xy[0] + block_dx/2], 
                                                                [mid_xy[1] - block_dy, mid_xy[1]], 
                                                                mode=mode, 
                                                                density=wall_density)))
        if increased_boundary_data:
            # top left boundary
            wall_data = np.vstack((wall_data, gen_data_in_rectangle([0, mid_xy[0] + block_dx/2], 
                                                                    [max([0, mid_xy[1] - block_dy - top_boundary_width]), mid_xy[1]], 
                                                                    mode=mode, 
                                                                    density=boundary_density)))
            # if mid_xy[1] > block_dy:
            # top right boundary
            wall_data = np.vstack((wall_data, gen_data_in_rectangle([mid_xy[0] + block_dx/2, x_max], 
                                                                    [max([0, mid_xy[1] - block_dy - top_boundary_width]), mid_xy[1] - block_dy], 
                                                                    mode=mode, 
                                                                    density=boundary_density)))
        
    # get data if moving backward
    if current_direction == 1:
        # get data in upper rectangle
        wall_data = np.vstack((wall_data, gen_data_in_rectangle([mid_xy[0] - block_dx/2, x_max], 
                                                                [mid_xy[1] - block_dy, mid_xy[1]], 
                                                                mode=mode, 
                                                                density=wall_density)))
        if increased_boundary_data:
            # top left boundary
            wall_data = np.vstack((wall_data, gen_data_in_rectangle([0, mid_xy[0] - block_dx/2], 
                                                                    [max([0,mid_xy[1] - block_dy - top_boundary_width]), max([0,mid_xy[1] - block_dy])], 
                                                                    mode=mode, 
                                                                    density=boundary_density)))
            # if mid_xy[1] > block_dy:
            # top right boundary
            wall_data = np.vstack((wall_data, gen_data_in_rectangle([mid_xy[0] - block_dx/2, x_max], 
                                                                    [max([0,mid_xy[1] - block_dy - top_boundary_width]), mid_xy[1]], 
                                                                    mode=mode, 
                                                                    density=boundary_density)))
               
        
    if increase_latest_block_data:
        wall_data = np.vstack((wall_data, gen_data_in_rectangle([mid_xy[0] - block_dx/2, mid_xy[0] + block_dx/2], 
                                                                [mid_xy[1] - block_dy, mid_xy[1]], 
                                                                mode=mode, 
                                                                density=new_density)))
    if increased_boundary_data:
        # left boundary
        wall_data = np.vstack(
            (wall_data, gen_data_in_rectangle(
                [0, boundary_width], 
                [boundary_width, max([boundary_width, mid_xy[1] - block_dy - top_boundary_width])],
                mode=mode, 
                density=boundary_density)))
        # right boundary
        wall_data = np.vstack(
            (wall_data, gen_data_in_rectangle(
                [x_max - boundary_width, x_max], 
                [boundary_width, max([boundary_width, mid_xy[1] - block_dy - top_boundary_width])], 
                mode=mode, 
                density=boundary_density)))
        # bottom boundary
        wall_data = np.vstack(
            (wall_data, gen_data_in_rectangle(
                [0, x_max], 
                [0, min([boundary_width, max([0, mid_xy[1] - block_dy - top_boundary_width])])], 
                mode=mode, 
                density=boundary_density)))
        
    time_col = np.full((wall_data.shape[0], 1), time)
    x0col = np.full((wall_data.shape[0], 1), heat_xy[0])
    y0col = np.full((wall_data.shape[0], 1), heat_xy[1])

    return np.hstack((time_col, wall_data, x0col, y0col))

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
        density = 50, 
        bottom_data = True,
):
    mid_xy, current_direction, new_block, heat_xy = get_pointer_cords(
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
        [x_max, x_max], 
        [0, right_y], 
        mode=mode, 
        density=density)
    if current_direction == 0:
        top_left_boundary_data = gen_data_in_line(
            [0, mid_xy[0] + block_dx/2], 
            [left_y, left_y], 
            mode=mode, 
            density=density)
        if right_y > 0.0:
            top_right_boundary_data = gen_data_in_line(
                [mid_xy[0] + block_dx/2, x_max], 
                [right_y, right_y], 
                mode=mode, 
                density=density)
        else:
            top_right_boundary_data = np.empty((0, 2))
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
        if right_y > 0.0:
            top_right_boundary_data = gen_data_in_line(
                [mid_xy[0] - block_dx/2, x_max], 
                [right_y, right_y], 
                mode=mode, 
                density=density)
        else:
            top_right_boundary_data = np.empty((0, 2))
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
        bbr = bottom_boundary_data.shape[0]# bottom boundary rows
        bottom_boundary_data = np.hstack((
            np.full((bbr, 1), time), 
            bottom_boundary_data, 
            np.full((bbr, 1), heat_xy[0]), 
            np.full((bbr, 1), heat_xy[1])))
    else:
        bottom_boundary_data = np.empty((0, 5))
    lbr = left_boundary_data.shape[0]# left boundary rows
    rbr = right_boundary_data.shape[0]# right boundary rows
    mbr = mid_boundary_data.shape[0]# mid boundary rows
    tlbr = top_left_boundary_data.shape[0]# top left boundary rows
    trbr = top_right_boundary_data.shape[0]# top right boundary rows

    left_boundary_data = np.hstack((
        np.full((lbr, 1), time), 
        left_boundary_data, 
        np.full((lbr, 1), heat_xy[0]), 
        np.full((lbr, 1), heat_xy[1])))
    right_boundary_data = np.hstack((
        np.full((rbr, 1), time), 
        right_boundary_data, 
        np.full((rbr, 1), heat_xy[0]), 
        np.full((rbr, 1), heat_xy[1])))
    mid_boundary_data = np.hstack((
        np.full((mbr, 1), time), 
        mid_boundary_data, 
        np.full((mbr, 1), heat_xy[0]), 
        np.full((mbr, 1), heat_xy[1])))
    top_left_boundary_data = np.hstack((
        np.full((tlbr, 1), time), 
        top_left_boundary_data, 
        np.full((tlbr, 1), heat_xy[0]), 
        np.full((tlbr, 1), heat_xy[1])))
    top_right_boundary_data = np.hstack((
        np.full((trbr, 1), time), 
        top_right_boundary_data, 
        np.full((trbr, 1), heat_xy[0]), 
        np.full((trbr, 1), heat_xy[1])))
    
    return (left_boundary_data,
            right_boundary_data,
            mid_boundary_data,
            top_left_boundary_data, 
            top_right_boundary_data, 
            bottom_boundary_data)

def new_block_times(
        x_max, block_dx, block_dy, x_speed, y_speed, start_t, tot_time
):
    total_blocks_per_bead = int(x_max/block_dx)
    time_per_block = block_dx/x_speed
    interval_between_bead = block_dy/y_speed
    current_bead_blocks = 0
    _new_block_times = [start_t, ]
    while _new_block_times[-1] <= tot_time:
        current_bead_blocks += 1
        
        next_block_time = _new_block_times[-1]+time_per_block if current_bead_blocks < total_blocks_per_bead else _new_block_times[-1]+interval_between_bead
        _new_block_times.append(next_block_time)
        if current_bead_blocks == total_blocks_per_bead:
            current_bead_blocks = 0
    return  _new_block_times[:-1]

def get_newblock_data(
        tot_time, x_speed, y_speed, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 10, 
        y_max = 10, 
        block_dx = 0.5, 
        block_dy = 0.5,
        start_t = 0.0, 
        mode = "random", 
        density = 500
):
    final_new_block_data = np.empty((0, 5))
    nbt = new_block_times(x_max, block_dx, block_dy, x_speed, y_speed, start_t, tot_time)
    print(len(nbt), len(nbt)*density*block_dx*block_dy)
    for time in nbt:
        mid_xy, current_direction, new_block, heat_xy = get_pointer_cords(
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
        if new_block:
            new_data = gen_data_in_rectangle(
                [mid_xy[0]-block_dx/2, mid_xy[0]+block_dx/2], 
                [mid_xy[1]-block_dy, mid_xy[1]], 
                mode=mode, 
                density=density)
            new_data = np.hstack(
                (
                    np.full((new_data.shape[0], 1), time), 
                    new_data, 
                    np.full((new_data.shape[0], 1), heat_xy[0]), 
                    np.full((new_data.shape[0], 1), heat_xy[1])))
            final_new_block_data = np.vstack((final_new_block_data, new_data))
        else:
            raise ValueError("Calculation is wrong for new block")
    return final_new_block_data

def get_wall_data(
        tot_time, x_speed, y_speed, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 10, 
        y_max = 10, 
        block_dx = 0.5, 
        block_dy = 0.5,
        start_t = 0.0, 
        mode = "random", 
        wall_density = 10, 
        new_density = 100,
        boundary_density = 50,
        increase_latest_block_data = True,
        increased_boundary_data = True,
        boundary_width = None, 
        top_boundary_layers = 2,
        t_grid = "random",
        t_density = 10     
):
    if t_grid == "random":
        all_times = np.random.uniform(start_t, tot_time, int(t_density*(tot_time-start_t)))
    elif t_grid == "uniform":
        all_times = np.linspace(start_t, tot_time, int(t_density*(tot_time-start_t)))
    else:
        raise ValueError("t_grid can only be uniform or random")
    final_wall_data = np.empty((0, 5))
    for time in all_times:
        final_wall_data = np.vstack(
            (
                final_wall_data, 
                get_t_wall_data(
                    time, x_speed, y_speed, 
                    movement_type = movement_type, 
                    if_continuous = if_continuous, 
                    if_discrete = if_discrete, 
                    x_max = x_max, 
                    y_max = y_max, 
                    block_dx = block_dx, 
                    block_dy = block_dy,
                    start_t = start_t, 
                    mode = mode, 
                    wall_density = wall_density, 
                    new_density = new_density,
                    boundary_density = boundary_density,
                    increase_latest_block_data = increase_latest_block_data,
                    increased_boundary_data = increased_boundary_data,
                    boundary_width = boundary_width, 
                    top_boundary_layers = top_boundary_layers)))
    return final_wall_data


def get_boundary_data(
        tot_time, x_speed, y_speed, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 10, 
        y_max = 10, 
        block_dx = 0.5, 
        block_dy = 0.5,
        start_t = 0.0, 
        mode = "random", 
        density = 50, 
        bottom_data = True,
        groups = [[0, 1, 2, 3, 4], [5,]], 
        t_grid = "random",
        t_density = 10     
):
    if t_grid == "random":
        all_times = np.random.uniform(start_t, tot_time, int(t_density*(tot_time-start_t)))
    elif t_grid == "uniform":
        all_times = np.linspace(start_t, tot_time, int(t_density*(tot_time-start_t)))
    else:
        raise ValueError("t_grid can only be uniform or random")
    final_boundary_data = [np.empty((0, 5)),]*len(groups)
    for time in all_times:
        tbd = get_t_boundary_data(
            time, x_speed, y_speed, 
            movement_type = movement_type, 
            if_continuous = if_continuous, 
            if_discrete = if_discrete, 
            x_max = x_max, 
            y_max = y_max, 
            block_dx = block_dx, 
            block_dy = block_dy,
            start_t = start_t, 
            mode = mode, 
            density = density, 
            bottom_data = bottom_data)
        for i, group in enumerate(groups):
            for bi in group:
                final_boundary_data[i] = np.vstack((final_boundary_data[i], tbd[bi]))
    return final_boundary_data
        
    
    
gnbd = get_newblock_data(
        10, 1, 1, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 10, 
        y_max = 10, 
        block_dx = 0.5, 
        block_dy = 0.5,
        start_t = 0.0, 
        mode = "random", 
        density = 500
)


gwd = get_wall_data(
        10, 1, 1, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 10, 
        y_max = 10, 
        block_dx = 0.5, 
        block_dy = 0.5,
        start_t = 0.0, 
        mode = "random", 
        wall_density = 10, 
        new_density = 100,
        boundary_density = 50,
        increase_latest_block_data = True,
        increased_boundary_data = True,
        boundary_width = None, 
        top_boundary_layers = 2,
        t_grid = "random",
        t_density = 10     
)
    

gbd = get_boundary_data(
        10, 1, 1, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 10, 
        y_max = 10, 
        block_dx = 0.5, 
        block_dy = 0.5,
        start_t = 0.0, 
        mode = "random", 
        density = 50, 
        bottom_data = True,
        groups = [[0, 1, 2], [3, 4], [5,]], 
        t_grid = "random",
        t_density = 10     
)
print(gnbd.shape)
print(gwd.shape)
print([gbdi.shape for gbdi in gbd])







    
    
    

# def get_data(frame):
#     t = frame/2
#     all_boundary_data = get_t_boundary_data(
#         t, 1, 1, 
#         movement_type = "bidirectional",
#         mode = "random",
#         block_dx = 0.5)
#     wall_data = get_t_wall_data(
#         t, 1, 1, 
#         movement_type = "bidirectional", 
#         mode = "random",
#         block_dx = 0.5)
#     # Remove the time column (first column) before plotting.
#     return (
#         wall_data[:, 1:],
#         all_boundary_data[0][:, 1:3],
#         all_boundary_data[1][:, 1:3],
#         all_boundary_data[2][:, 1:3],
#         all_boundary_data[3][:, 1:3],
#         all_boundary_data[4][:, 1:3],
#         all_boundary_data[5][:, 1:3]
#     )

# # Define figure dimensions
# x_max, y_max = 10, 10

# # Prepare the figure and axis
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.set_xlim(-1, x_max+1)
# ax.set_ylim(-1, y_max+1)
# ax.set_xlabel("X Coordinate")
# ax.set_ylabel("Y Coordinate")
# ax.set_title("Welding Wall")
# ax.grid(False)

# # Initialize scatter plots (each with its own color)
# scatter_wall = ax.scatter([], [], s=6, c="black")
# scatter_b0   = ax.scatter([], [], s=6, c="pink")
# scatter_b1   = ax.scatter([], [], s=6, c="red")
# scatter_b2   = ax.scatter([], [], s=6, c="blue")
# scatter_b3   = ax.scatter([], [], s=6, c="green")
# scatter_b4   = ax.scatter([], [], s=6, c="yellow")
# scatter_b5   = ax.scatter([], [], s=6, c="orange")

# def init():
#     # Initialize all scatter plots with an empty (0,2) array.
#     empty_offsets = np.empty((0, 2))
#     scatter_wall.set_offsets(empty_offsets)
#     scatter_b0.set_offsets(empty_offsets)
#     scatter_b1.set_offsets(empty_offsets)
#     scatter_b2.set_offsets(empty_offsets)
#     scatter_b3.set_offsets(empty_offsets)
#     scatter_b4.set_offsets(empty_offsets)
#     scatter_b5.set_offsets(empty_offsets)
#     return scatter_wall, scatter_b0, scatter_b1, scatter_b2, scatter_b3, scatter_b4, scatter_b5

# def update(frame):
#     data = get_data(frame)
#     # Each element in data is an array of shape (n, 2)
#     scatter_wall.set_offsets(data[0])
#     scatter_b0.set_offsets(data[1])
#     scatter_b1.set_offsets(data[2])
#     scatter_b2.set_offsets(data[3])
#     scatter_b3.set_offsets(data[4])
#     scatter_b4.set_offsets(data[5])
#     scatter_b5.set_offsets(data[6])
#     return scatter_wall, scatter_b0, scatter_b1, scatter_b2, scatter_b3, scatter_b4, scatter_b5

# # Create the animation
# anim = animation.FuncAnimation(fig, update, frames=200, init_func=init,
#                                interval=1000, blit=True, repeat=True)

# plt.show()
