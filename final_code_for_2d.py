import torch
import numpy as np
import matplotlib.pyplot as plt
import random


# def draw_rectangle_with_data_points(corner_points):
#     """
#     Draws a filled rectangle on a 2D plot using the provided corner points
#     (in cyclic order).

#     Parameters:
#     -----------
#     corner_points: list of lists (float)
#         A list of corner coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#         provided in a cyclic order.

#     Returns:
#     --------
#     None
#     """
#     x_coords = [point[0] for point in corner_points]
#     y_coords = [point[1] for point in corner_points]
#     plt.figure()
#     plt.fill(x_coords, y_coords, color='skyblue')
#     plt.xlim(0, 10)
#     plt.ylim(0, 10)
#     plt.title("Rectangle with Data Points")
#     plt.show()


# def generate_1d_array(size, min_val=0, max_val=10):
#     """
#     Generates a 1D NumPy array of uniformly distributed random numbers.

#     Parameters:
#     -----------
#     size: int
#         Number of random samples to generate.
#     min_val: float
#         Lower bound of the uniform distribution.
#     max_val: float
#         Upper bound of the uniform distribution.

#     Returns:
#     --------
#     np.ndarray
#         1D array of random numbers.
#     """
#     return np.random.uniform(min_val, max_val, size)


def solid_blocks(x_corner, y_corner, direction, x_len, block_height, block_length, new_block):
    """
    Constructs corner points for a left rectangle, a right rectangle, and an
    optional 'new block' rectangle to simulate the dynamic geometry in the domain.

    Parameters:
    -----------
    x_corner : float
        The x-coordinate where the 'moving' part meets or is inserted.
    y_corner : float
        The y-coordinate of the top boundary of the new block.
    direction : int
        Direction of movement or insertion (1 for left-to-right, -1 for right-to-left).
    x_len : float
        Total length in x direction.
    block_height : float
        The height of the new block.
    block_length : float
        The length in the x direction of the new block.
    new_block : bool
        Whether to create a new block or not.

    Returns:
    --------
    list of length 3
        Each item is a list of corner points for a sub-rectangle. The third one is
        either the newly created block or a dummy list if new_block=False.
    """
    # Fixed boundaries in x
    x_low = 0.0
    x_high = x_len

    # The 'mid' x is where we add new block
    x_mid = x_corner

    if direction == 1:
        # If direction is 1, new block is from right to left at x_mid
        y_left_high = y_corner
        y_right_high = y_corner - block_height
        if new_block:
            x_new_block_low = x_mid - block_length
            x_new_block_high = x_mid
            y_new_block_low = y_corner - block_height
            y_new_block_high = y_corner
    else:
        # If direction is -1, new block is from left to right at x_mid
        y_left_high = y_corner - block_height
        y_right_high = y_corner
        if new_block:
            x_new_block_high = x_mid + block_length
            x_new_block_low = x_mid
            y_new_block_low = y_corner - block_height
            y_new_block_high = y_corner

    # Build the corner points for two main blocks:
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
        corner_points_new = [[x_corner, y_corner]] * 4  # Dummy block (degenerate)

    return [corner_points_left, corner_points_right, corner_points_new]


def generate_points(three_block_corner_point, time, mode='uniform',
                    dx=0.01, dy=0.01, density=100):
    """
    Generate 2D points within specified rectangular blocks, appending a time value.

    Parameters
    ----------
    three_block_corner_point : list
        A list whose elements are lists of corner points defining one or more
        rectangular regions. Each element is expected to be something like:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    time : float
        A time value to attach to each generated (x,y) point.
    mode : str, optional
        'uniform' to generate evenly spaced points, 'random' to generate
        random points from a uniform distribution.
    dx : float, optional
        Spacing in the x direction (used only if mode='uniform').
    dy : float, optional
        Spacing in the y direction (used only if mode='uniform').
    density : float, optional
        Number of random points to generate per unit area (used only if mode='random').

    Returns
    -------
    list
        A list of [x, y, time] for all generated points.
    """
    grid = []

    for corner_points in three_block_corner_point[:2]: # use only left and right reactangle because these span the complete geometry
        # Extract x, y coordinates from corner points
        x_coords = [pt[0] for pt in corner_points]
        y_coords = [pt[1] for pt in corner_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Skip degenerate blocks
        if (x_max - x_min) == 0.0 or (y_max - y_min) == 0.0:
            continue

        if mode == 'uniform':
            # Generate uniformly spaced grid
            x_vals = np.arange(x_min, x_max, dx)
            y_vals = np.arange(y_min, y_max, dy)
            for xi in x_vals:
                for yi in y_vals:
                    grid.append([xi, yi, time])

        elif mode == 'random':
            # Compute the area of the rectangle
            area = (x_max - x_min) * (y_max - y_min)
            # Based on specified density, compute how many points to generate
            n_points = int(area * density)
            # Generate random x, y within [x_min, x_max] and [y_min, y_max]
            random_x = np.random.uniform(low=x_min, high=x_max, size=n_points)
            random_y = np.random.uniform(low=y_min, high=y_max, size=n_points)

            for xi, yi in zip(random_x, random_y):
                grid.append([xi, yi, time])

        else:
            raise ValueError("Invalid mode. Use 'uniform' or 'random'.")

    return grid


def generate_points_using_line_extremes(x_extreme, y_extreme, time, mode='uniform', dx=0.01, dy=0.01, density=100):
    """
    Generates points along a line boundary (either vertical or horizontal) using the specified mode.

    Parameters:
    -----------
    x_extreme: list (float)
        [x_min, x_max] if horizontal line or same repeated if vertical line
    y_extreme: list (float)
        [y_min, y_max] if vertical line or same repeated if horizontal line
    time: float
        The time to attach to each boundary point.
    mode: str, optional
        'uniform' for evenly spaced points, 'random' for random points.
    dx: float, optional
        Spacing if the line is horizontal (used in 'uniform' mode).
    dy: float, optional
        Spacing if the line is vertical (used in 'uniform' mode).
    density: float, optional
        Number of random points to generate per unit length (used in 'random' mode).

    Returns:
    --------
    list of [x, y, t]
        Points along the specified line, each appended with time.
    """
    grid = []
    x_min, x_max = min(x_extreme), max(x_extreme)
    y_min, y_max = min(y_extreme), max(y_extreme)

    # If it's a purely horizontal line
    if (y_max - y_min) == 0.0 and (x_max - x_min) != 0.0:
        if mode == 'uniform':
            x_vals = np.arange(x_min, x_max, dx)
            for xi in x_vals:
                grid.append([xi, y_max, time])

        elif mode == 'random':
            line_length = x_max - x_min
            n_points = int(line_length * density)
            random_x = np.random.uniform(low=x_min, high=x_max, size=n_points)
            for xi in random_x:
                grid.append([xi, y_max, time])

        else:
            raise ValueError("Invalid mode. Use 'uniform' or 'random'.")

    # If it's a purely vertical line
    elif (x_max - x_min) == 0.0 and (y_max - y_min) != 0.0:
        if mode == 'uniform':
            y_vals = np.arange(y_min, y_max, dy)
            for yi in y_vals:
                grid.append([x_max, yi, time])

        elif mode == 'random':
            line_length = y_max - y_min
            n_points = int(line_length * density)
            random_y = np.random.uniform(low=y_min, high=y_max, size=n_points)
            for yi in random_y:
                grid.append([x_max, yi, time])

        else:
            raise ValueError("Invalid mode. Use 'uniform' or 'random'.")

    # Otherwise, it's not a single line
    else:
        if (x_max - x_min) == 0.0 and (y_max - y_min) == 0.0:
            # Single point, no line
            print("Warning: It is a single point not line. Check input extremes.")

        else:
            raise ValueError("It is not a single point or a straight line. Check input extremes.")

    return grid


def get_boundary_extremes(three_block_corner_point, consider_bottom_as_boundary = False):
    """
    Extracts boundary lines (extremes) from the 3-block corner points.
    This helps define the boundary for each time step.

    Parameters:
    -----------
    three_block_corner_point: list of lists of corner points
        [ [corners_left], [corners_right], [corners_new] ]

    Returns:
    --------
    list of extremes:
        Each item is of form: [[x_min, x_max], [y_min, y_max]], corresponding
        to a line boundary. For instance, a vertical line boundary or a horizontal line boundary.
    """

    # corners for left block
    left_xs = [xi for xi, yi in three_block_corner_point[0]]
    left_ys = [yi for xi, yi in three_block_corner_point[0]]
    # corners for right block
    right_xs = [xi for xi, yi in three_block_corner_point[1]]
    right_ys = [yi for xi, yi in three_block_corner_point[1]]

    # Basic checks
    left_x = min(left_xs)    # left-most boundary
    right_x = max(right_xs)  # right-most boundary
    mid_x = max(left_xs)     # where left block ends / right block starts

    if max(left_xs) != min(right_xs):
        raise ValueError("Left and right rectangle not touching properly")

    left_y_low, left_y_high = min(left_ys), max(left_ys)
    right_y_low, right_y_high = min(right_ys), max(right_ys)

    # The top and bottom y for the 'shared' boundary (mid_x)
    mid_y_low = min(max(left_ys), max(right_ys))
    mid_y_high = max(max(left_ys), max(right_ys))

    # Build sets of extremes for vertical/horizontal lines
    # 1. left boundary
    left_boundary_extreme = [[left_x, left_x], [left_y_low, left_y_high]]
    # 2. right boundary
    right_boundary_extreme = [[right_x, right_x], [right_y_low, right_y_high]]
    # 3. shared vertical boundary
    shared_boundary_extreme = [[mid_x, mid_x], [mid_y_low, mid_y_high]]
    # 4. top boundary for left part
    top_left_extreme = [[left_x, mid_x], [left_y_high, left_y_high]]
    # 5. top boundary for right part
    top_right_extreme = [[mid_x, right_x], [right_y_high, right_y_high]]

    return_data = [
        left_boundary_extreme,
        right_boundary_extreme,
        shared_boundary_extreme,
        top_left_extreme,
        top_right_extreme
    ]

    if consider_bottom_as_boundary:
        # 6. bottom boundary
        if right_y_low == right_y_high:
            bottom_extreme = [[left_x, mid_x], [right_y_low, right_y_low]]
        else:
            bottom_extreme = [[left_x, right_x], [right_y_low, right_y_low]]
        return_data.append(bottom_extreme)

    return return_data


def generate_boundary_point_using_three_blocks(three_block_corner_point, time,
                                               mode = 'uniform', dx=0.01, dy=0.01,
                                               density = 100, consider_bottom_as_boundary = False):
    """
    Generates boundary points for the entire 3-block system at a given time.

    Parameters:
    -----------
    three_block_corner_point: list of lists of corner points
    time: float
        Time to attach to each boundary point.
    dx: float
        Spacing for horizontal boundary lines.
    dy: float
        Spacing for vertical boundary lines.

    Returns:
    --------
    grid: list of [x, y, t]
        The boundary points along all edges of the 3-block system at a given time.
    """
    grid = []
    all_extremes = get_boundary_extremes(three_block_corner_point, consider_bottom_as_boundary = consider_bottom_as_boundary)
    for extremes in all_extremes:
        grid += generate_points_using_line_extremes(
            extremes[0], extremes[1], time, mode = mode, dx=dx, dy=dy, density = density
        )
    return grid


def generate_individual_boundary_data(three_block_corner_point, time,
                                      mode = 'uniform', dx=0.01, dy=0.01,
                                      density = 100, consider_bottom_as_boundary = False):
    """
    Generates boundary points for the entire 3-block system at a given time.

    Parameters:
    -----------
    three_block_corner_point: list of lists of corner points
    time: float
        Time to attach to each boundary point.
    dx: float
        Spacing for horizontal boundary lines.
    dy: float
        Spacing for vertical boundary lines.

    Returns:
    --------
    grid: dictionary with keys corresponding to boundary type and values as data generated for that boundary.
    """
    grid = {}
    grid["left"] = []
    grid["right"] = []
    grid["shared"] = []
    grid["top_left"] = []
    grid["top_right"] = []
    if consider_bottom_as_boundary:
        grid["bottom"] = []


    all_extremes = get_boundary_extremes(three_block_corner_point, consider_bottom_as_boundary = consider_bottom_as_boundary)
    grid["left"] += generate_points_using_line_extremes(all_extremes[0][0], all_extremes[0][1],
                                                        time, mode = mode, dx=dx, dy=dy, density = density)
    grid["right"] += generate_points_using_line_extremes(all_extremes[1][0], all_extremes[1][1],
                                                         time, mode = mode, dx=dx, dy=dy, density = density)
    grid["shared"] += generate_points_using_line_extremes(all_extremes[2][0], all_extremes[2][1],
                                                           time, mode = mode, dx=dx, dy=dy, density = density)
    grid["top_left"] += generate_points_using_line_extremes(all_extremes[3][0], all_extremes[3][1],
                                                              time, mode = mode, dx=dx, dy=dy, density = density)
    grid["top_right"] += generate_points_using_line_extremes(all_extremes[4][0], all_extremes[4][1],
                                                               time, mode = mode, dx=dx, dy=dy, density = density)
    if consider_bottom_as_boundary:
        grid["bottom"] += generate_points_using_line_extremes(all_extremes[5][0], all_extremes[5][1],
                                                              time, mode = mode, dx=dx, dy=dy, density = density)
    return grid


def heat_source_origin(current_time, travel_speed, jump, x_max, x_min):
    """
    Computes the instantaneous position (x,y) of a moving heat source that travels
    repeatedly along [x_min, x_max]. Each time the heat source finishes one sweep
    in the x direction, it 'jumps' in y by 'jump'.

    Parameters:
    -----------
    current_time: torch.Tensor
        Current time (or times) at which to compute the position.
    travel_speed: float
        Speed of the heat source in the x direction.
    jump: float
        Distance by which the heat source moves in y after each full pass.
    x_max: float
        Right boundary in x.
    x_min: float
        Left boundary in x.

    Returns:
    --------
    [current_x, current_y]: [torch.Tensor, torch.Tensor]
        The (x, y) coordinates of the moving heat source origin.
    """

    one_lap_time = (x_max - x_min) / travel_speed  # time to go from x_min to x_max
    current_lap = (current_time / one_lap_time).to(torch.int32)  # integer # of laps

    # Each completed lap increments y
    current_y = (current_lap + 1) * jump

    # direction = 0 means left to right, 1 means right to left
    direction = current_lap % 2
    current_x = (travel_speed * current_time) % (x_max - x_min)

    # If direction == 1, invert position (like traveling back)
    mask = (direction == 1)
    current_x[mask] = x_max - current_x[mask]

    return [current_x, current_y]


def heat_source_equation(x, y, current_time, travel_speed, jump, x_max, x_min):
    """
    A Gaussian-like heat source that moves with time.

    The amplitude decays with distance from the moving heat source origin.

    Calculates heat value at (xi, yi) in zip(x, y) and at single current_time.
    Parameters:
    -----------
    x, y, current_time: torch.Tensor
        Batches of x, y, time coordinates.
    travel_speed, jump, x_max, x_min: float
        Parameters controlling the heat source motion in the domain.

    Returns:
    --------
    torch.Tensor
        A scalar field representing the strength of the heat source at each (x,y,t).
    """
    heat_source_origin_x, heat_source_origin_y = heat_source_origin(
        current_time, travel_speed, jump, x_max, x_min
    )
    return_data = torch.exp(-((x - heat_source_origin_x)**2 + (y - heat_source_origin_y)**2))
    modified_return_data = return_data.clone()  # Create a copy
    modified_return_data[y > heat_source_origin_y] = 0
    # return_data[y>heat_source_origin_y] = 0
    return modified_return_data


def generate_new_block_internal_data(block_corner_data, ctd_data, mode='random', dx=0.01, dy=0.01, density=100):
    new_block_internal_data = []
    for bcdi, ctdi in zip(block_corner_data, ctd_data):
        new_block = [bcdi[2]]  # bcdi[2] is the newly created block corners
        time_ = ctdi[2]
        new_datai_new_block = generate_points(new_block, time_, mode=mode, dx=dx, dy=dy, density=density)
        new_block_internal_data += new_datai_new_block
    new_block_internal_data = np.array(new_block_internal_data)
    return new_block_internal_data


def generate_wall_internal_data(block_corner_data, ctd_data, total_mid_times = 10, t_mode = "uniform", mode = "random", dx=0.01, dy=0.01, density=50):
    wall_internal_data = []
    if t_mode == "uniform":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for t_mid in torch.linspace(ctdip[2], ctdin[2], 10)[:-1]:
                mid_time_data = generate_points(bcdi, t_mid, mode=mode, dx=dx, dy=dy, density=density)
                wall_internal_data += mid_time_data
    elif t_mode == "random":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for _ in range(total_mid_times):
                # Random time between ctdip[2] and ctdin[2]
                t_mid = random.uniform(ctdip[2], ctdin[2])
                mid_time_data = generate_points(bcdi, t_mid, mode=mode, dx=dx, dy=dy, density=density)
                wall_internal_data += mid_time_data
    else:
        raise ValueError("Invalid mode. Use 'uniform' or 'random'.")
    wall_internal_data = np.array(wall_internal_data)
    return wall_internal_data


def generate_boundary_data(block_corner_data, ctd_data, total_mid_times = 10,
                           t_mode = "uniform", mode = "random", dx=0.01, dy=0.01,
                           density=50, consider_bottom_as_boundary = False):
    boundary_data = []
    if t_mode == "uniform":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for t_mid in torch.linspace(ctdip[2], ctdin[2], 10)[:-1]:
                mid_time_data = generate_boundary_point_using_three_blocks(bcdi, t_mid, mode=mode,
                                                                           dx=dx, dy=dy, density=density,
                                                                           consider_bottom_as_boundary = consider_bottom_as_boundary)
                boundary_data += mid_time_data
    elif t_mode == "random":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for _ in range(total_mid_times):
                # Random time between ctdip[2] and ctdin[2]
                t_mid = random.uniform(ctdip[2], ctdin[2])
                mid_time_data = generate_boundary_point_using_three_blocks(bcdi, t_mid, mode=mode,
                                                                           dx=dx, dy=dy, density=density,
                                                                           consider_bottom_as_boundary = consider_bottom_as_boundary)
                boundary_data += mid_time_data
    else:
        raise ValueError("Invalid mode. Use 'uniform' or 'random'.")
    boundary_data = np.array(boundary_data)
    return boundary_data




def generate_all_individual_boundary_data(block_corner_data, ctd_data, total_mid_times = 10,
                                          t_mode = "uniform", mode = "random", dx=0.01, dy=0.01,
                                          density=50, consider_bottom_as_boundary = False):
    boundary_data = {}
    boundary_data["left"] = []
    boundary_data["right"] = []
    boundary_data["shared"] = []
    boundary_data["top_left"] = []
    boundary_data["top_right"] = []
    if consider_bottom_as_boundary:
        boundary_data["bottom"] = []
    if t_mode == "uniform":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for t_mid in torch.linspace(ctdip[2], ctdin[2], 10)[:-1]:
                mid_time_data = generate_individual_boundary_data(bcdi, t_mid, mode=mode,
                                                                  dx=dx, dy=dy, density=density,
                                                                  consider_bottom_as_boundary = consider_bottom_as_boundary)

                boundary_data["left"] += mid_time_data["left"]
                boundary_data["right"] += mid_time_data["right"]
                boundary_data["shared"] += mid_time_data["shared"]
                boundary_data["top_left"] += mid_time_data["top_left"]
                boundary_data["top_right"] += mid_time_data["top_right"]
                if consider_bottom_as_boundary:
                    boundary_data["bottom"] += mid_time_data["bottom"]
    elif t_mode == "random":
        for bcdi, ctdip, ctdin in zip(block_corner_data[:-1], ctd_data[:-1], ctd_data[1:]):
            for _ in range(total_mid_times):
                t_mid = random.uniform(ctdip[2], ctdin[2])
                mid_time_data = generate_individual_boundary_data(bcdi, t_mid, mode=mode,
                                                                  dx=dx, dy=dy, density=density,
                                                                  consider_bottom_as_boundary = consider_bottom_as_boundary)

                boundary_data["left"] += mid_time_data["left"]
                boundary_data["right"] += mid_time_data["right"]
                boundary_data["shared"] += mid_time_data["shared"]
                boundary_data["top_left"] += mid_time_data["top_left"]
                boundary_data["top_right"] += mid_time_data["top_right"]
                if consider_bottom_as_boundary:
                    boundary_data["bottom"] += mid_time_data["bottom"]

    else:
        raise ValueError("Invalid mode. Use 'uniform' or 'random'.")
    boundary_data["left"] = np.array(boundary_data["left"])
    boundary_data["right"] = np.array(boundary_data["right"])
    boundary_data["shared"] = np.array(boundary_data["shared"])
    boundary_data["top_left"] = np.array(boundary_data["top_left"])
    boundary_data["top_right"] = np.array(boundary_data["top_right"])
    if consider_bottom_as_boundary:
        boundary_data["bottom"] = np.array(boundary_data["bottom"])
    return boundary_data



def random_sample(tensor_, num_samples):
    """
    Randomly samples 'num_samples' data points from 'tensor_'.
    """
    random_indices = random.sample(range(len(tensor_)), min(num_samples, len(tensor_)))
    return tensor_[random_indices]

if __name__ == "__main__":


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


    print(nnd_wall_internal_data)