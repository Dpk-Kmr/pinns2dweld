import torch
import numpy as np
import matplotlib.pyplot as plt
import random


class BlockSimulation:
    def __init__(
        self,
        x_range,           # e.g. [0.0, 15.0]
        y_range,           # e.g. [0.0, 4.0]
        t_range,           # e.g. [0.0, 6.0]
        time_gap,          # time step for new block (e.g., 0.1 sec)
        x_gap,             # new block length in x-direction (mm)
        y_gap,             # new block height (mm)
        initial_direction, # start direction (1 for left-to-right, -1 for right-to-left)
        initial_y_level,   # height of each bead
        dx=0.01,           # gap for uniform x
        dy=0.01,           # gap for uniform y
        density=100,       # data density for random mode
        total_mid_times_internal=10, # total times in one time step for internal data
        total_mid_times_boundary=10, # total times in one time step for boundary data
        t_mode="random",   # mode for time sampling: "uniform" or "random"
        mode="random",     # mode for point generation: "uniform" or "random"
        consider_bottom_as_boundary=False, # whether to generate data for bottom boundary
        phi_laser=2,       # mm; length scale for non-dimensionalization
        v_laser=10,        # mm/s; speed for time scale
        nd_bounds=(7.5, 2.0, 30, 0, 0, 0)  # (ndx_max, ndy_max, ndt_max, ndx_min, ndy_min, ndt_min)
    ):
        # Simulation and geometry parameters
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.time_gap = time_gap
        self.x_gap = x_gap
        self.y_gap = y_gap
        self.initial_direction = initial_direction
        self.initial_y_level = initial_y_level
        self.x_len = x_range[-1]  # total length of the wall in x-direction
        self.dx = dx
        self.dy = dy
        self.density = density
        self.total_mid_times_internal = total_mid_times_internal
        self.total_mid_times_boundary = total_mid_times_boundary
        self.t_mode = t_mode
        self.mode = mode
        self.consider_bottom_as_boundary = consider_bottom_as_boundary
        self.phi_laser = phi_laser
        self.v_laser = v_laser
        self.nd_bounds = nd_bounds

        # Create time data and then generate the center data (ctd_data) and block-corner data.
        self.t_data = np.arange(t_range[0], t_range[1], time_gap)
        self.ctd_data = self.generate_ctd_data()
        self.block_corner_data = self.generate_block_corner_data()

    @staticmethod
    def solid_blocks(x_corner, y_corner, direction, x_len, block_height, block_length, new_block):
        """
        Constructs three sets of corner points: left block, right block and (if True) a new block.
        """
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

        # Left block
        corner_points_left = [
            [x_low, 0.0],
            [x_low, y_left_high],
            [x_mid, y_left_high],
            [x_mid, 0.0]
        ]
        # Right block
        corner_points_right = [
            [x_mid, 0.0],
            [x_mid, y_right_high],
            [x_high, y_right_high],
            [x_high, 0.0]
        ]
        # New block (or a dummy degenerate block if new_block is False)
        if new_block:
            corner_points_new = [
                [x_new_block_low, y_new_block_low],
                [x_new_block_low, y_new_block_high],
                [x_new_block_high, y_new_block_high],
                [x_new_block_high, y_new_block_low]
            ]
        else:
            corner_points_new = [[x_corner, y_corner]] * 4

        return [corner_points_left, corner_points_right, corner_points_new]

    def generate_ctd_data(self):
        """
        Generates the center (joint) data for the new block installation.
        Each entry is a list: [x_value, y_level, time, direction]
        """
        ctd_data = []
        direction = self.initial_direction
        y_level = self.initial_y_level
        # Initialize with the first time step.
        ctd_data.append([self.x_gap, self.y_gap, self.t_data[0], direction])
        for ti in self.t_data[1:]:
            if direction == 1:
                x_value = ctd_data[-1][0] + 1.0
            else:
                x_value = ctd_data[-1][0] - 1.0
            ctd_data.append([x_value, y_level, ti, direction])
            # When a boundary is reached, flip direction and increment the y_level.
            if x_value == self.x_range[0] or x_value == self.x_range[1]:
                direction = -direction
                y_level = y_level + self.y_gap
        return ctd_data

    def generate_block_corner_data(self):
        """
        Generates block corner data for each time step using the solid_blocks function.
        """
        block_corner_data = []
        for entry in self.ctd_data:
            x_corner, y_corner, t, direction = entry
            corners = np.array(
                self.solid_blocks(x_corner, y_corner, direction, self.x_len, self.y_gap, self.x_gap, True)
            )
            block_corner_data.append(corners)
        return np.array(block_corner_data)

    @staticmethod
    def generate_points(corner_points_list, time, mode='uniform', dx=0.01, dy=0.01, density=100):
        """
        Generates 2D points (with attached time) within the first two blocks.
        """
        grid = []
        # Use only left and right blocks (first two entries)
        for corner_points in corner_points_list[:2]:
            x_coords = [pt[0] for pt in corner_points]
            y_coords = [pt[1] for pt in corner_points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Skip degenerate blocks
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
                random_x = np.random.uniform(low=x_min, high=x_max, size=n_points)
                random_y = np.random.uniform(low=y_min, high=y_max, size=n_points)
                for xi, yi in zip(random_x, random_y):
                    grid.append([xi, yi, time])
            else:
                raise ValueError("Invalid mode. Use 'uniform' or 'random'.")
        return grid

    @staticmethod
    def generate_points_using_line_extremes(x_extreme, y_extreme, time, mode='uniform', dx=0.01, dy=0.01, density=100):
        """
        Generates points along a line (vertical or horizontal) defined by the given extremes.
        """
        grid = []
        x_min, x_max = min(x_extreme), max(x_extreme)
        y_min, y_max = min(y_extreme), max(y_extreme)

        # Horizontal line
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
        # Vertical line
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
        else:
            if (x_max - x_min) == 0.0 and (y_max - y_min) == 0.0:
                print("Warning: It is a single point, not a line. Check input extremes.")
            else:
                raise ValueError("It is not a single point or a straight line. Check input extremes.")
        return grid

    @staticmethod
    def get_boundary_extremes(three_block_corner_point, consider_bottom_as_boundary=False):
        """
        From the 3-block corner points extract boundary extremes.
        Returns a list of line extremes (each extreme is given as [[x_min, x_max], [y_min, y_max]]).
        """
        # For left block
        left_xs = [xi for xi, yi in three_block_corner_point[0]]
        left_ys = [yi for xi, yi in three_block_corner_point[0]]
        # For right block
        right_xs = [xi for xi, yi in three_block_corner_point[1]]
        right_ys = [yi for xi, yi in three_block_corner_point[1]]

        left_x = min(left_xs)
        right_x = max(right_xs)
        mid_x = max(left_xs)

        if max(left_xs) != min(right_xs):
            raise ValueError("Left and right rectangle not touching properly")

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

    def generate_boundary_point_using_three_blocks(self, three_block_corner_point, time, mode='uniform'):
        """
        Generates boundary points for all edges of the 3-block system.
        """
        grid = []
        all_extremes = self.get_boundary_extremes(three_block_corner_point, consider_bottom_as_boundary=self.consider_bottom_as_boundary)
        for extremes in all_extremes:
            grid += self.generate_points_using_line_extremes(
                extremes[0], extremes[1], time, mode=mode, dx=self.dx, dy=self.dy, density=self.density
            )
        return grid

    def generate_individual_boundary_data(self, three_block_corner_point, time, mode='uniform'):
        """
        Generates boundary points for each boundary separately.
        """
        grid = {
            "left": [],
            "right": [],
            "shared": [],
            "top_left": [],
            "top_right": []
        }
        if self.consider_bottom_as_boundary:
            grid["bottom"] = []
        all_extremes = self.get_boundary_extremes(three_block_corner_point, consider_bottom_as_boundary=self.consider_bottom_as_boundary)
        grid["left"] += self.generate_points_using_line_extremes(all_extremes[0][0], all_extremes[0][1], time, mode=mode, dx=self.dx, dy=self.dy, density=self.density)
        grid["right"] += self.generate_points_using_line_extremes(all_extremes[1][0], all_extremes[1][1], time, mode=mode, dx=self.dx, dy=self.dy, density=self.density)
        grid["shared"] += self.generate_points_using_line_extremes(all_extremes[2][0], all_extremes[2][1], time, mode=mode, dx=self.dx, dy=self.dy, density=self.density)
        grid["top_left"] += self.generate_points_using_line_extremes(all_extremes[3][0], all_extremes[3][1], time, mode=mode, dx=self.dx, dy=self.dy, density=self.density)
        grid["top_right"] += self.generate_points_using_line_extremes(all_extremes[4][0], all_extremes[4][1], time, mode=mode, dx=self.dx, dy=self.dy, density=self.density)
        if self.consider_bottom_as_boundary:
            grid["bottom"] += self.generate_points_using_line_extremes(all_extremes[5][0], all_extremes[5][1], time, mode=mode, dx=self.dx, dy=self.dy, density=self.density)
        return grid

    def generate_new_block_internal_data(self):
        """
        Generates internal data for the new block across time steps.
        """
        new_block_internal_data = []
        for bcdi, ctdi in zip(self.block_corner_data, self.ctd_data):
            # bcdi[2] contains the new block's corners.
            new_block = [bcdi[2]]
            time_ = ctdi[2]
            new_data = self.generate_points(new_block, time_, mode=self.mode, dx=self.dx, dy=self.dy, density=self.density)
            new_block_internal_data += new_data
        return np.array(new_block_internal_data)

    def generate_wall_internal_data(self):
        """
        Generates internal wall data over intermediate times.
        """
        wall_internal_data = []
        if self.t_mode == "uniform":
            for bcdi, ctdip, ctdin in zip(self.block_corner_data[:-1], self.ctd_data[:-1], self.ctd_data[1:]):
                # Convert each t_mid to float.
                for t_mid in torch.linspace(ctdip[2], ctdin[2], self.total_mid_times_internal)[:-1]:
                    t_val = t_mid.item()
                    mid_time_data = self.generate_points(bcdi, t_val, mode=self.mode, dx=self.dx, dy=self.dy, density=self.density)
                    wall_internal_data += mid_time_data
        elif self.t_mode == "random":
            for bcdi, ctdip, ctdin in zip(self.block_corner_data[:-1], self.ctd_data[:-1], self.ctd_data[1:]):
                for _ in range(self.total_mid_times_internal):
                    t_val = random.uniform(ctdip[2], ctdin[2])
                    mid_time_data = self.generate_points(bcdi, t_val, mode=self.mode, dx=self.dx, dy=self.dy, density=self.density)
                    wall_internal_data += mid_time_data
        else:
            raise ValueError("Invalid t_mode. Use 'uniform' or 'random'.")
        return np.array(wall_internal_data)

    def generate_boundary_data(self):
        """
        Generates boundary data for the entire 3-block system at intermediate times.
        """
        boundary_data = []
        if self.t_mode == "uniform":
            for bcdi, ctdip, ctdin in zip(self.block_corner_data[:-1], self.ctd_data[:-1], self.ctd_data[1:]):
                for t_mid in torch.linspace(ctdip[2], ctdin[2], self.total_mid_times_boundary)[:-1]:
                    t_val = t_mid.item()
                    mid_time_data = self.generate_boundary_point_using_three_blocks(bcdi, t_val, mode=self.mode)
                    boundary_data += mid_time_data
        elif self.t_mode == "random":
            for bcdi, ctdip, ctdin in zip(self.block_corner_data[:-1], self.ctd_data[:-1], self.ctd_data[1:]):
                for _ in range(self.total_mid_times_boundary):
                    t_val = random.uniform(ctdip[2], ctdin[2])
                    mid_time_data = self.generate_boundary_point_using_three_blocks(bcdi, t_val, mode=self.mode)
                    boundary_data += mid_time_data
        else:
            raise ValueError("Invalid t_mode. Use 'uniform' or 'random'.")
        return np.array(boundary_data)

    def generate_all_individual_boundary_data(self):
        """
        Generates and groups boundary data for individual boundaries.
        """
        boundary_data = {
            "left": [],
            "right": [],
            "shared": [],
            "top_left": [],
            "top_right": []
        }
        if self.consider_bottom_as_boundary:
            boundary_data["bottom"] = []
        if self.t_mode == "uniform":
            for bcdi, ctdip, ctdin in zip(self.block_corner_data[:-1], self.ctd_data[:-1], self.ctd_data[1:]):
                for t_mid in torch.linspace(ctdip[2], ctdin[2], self.total_mid_times_boundary)[:-1]:
                    t_val = t_mid.item()
                    mid_time_data = self.generate_individual_boundary_data(bcdi, t_val, mode=self.mode)
                    boundary_data["left"] += mid_time_data["left"]
                    boundary_data["right"] += mid_time_data["right"]
                    boundary_data["shared"] += mid_time_data["shared"]
                    boundary_data["top_left"] += mid_time_data["top_left"]
                    boundary_data["top_right"] += mid_time_data["top_right"]
                    if self.consider_bottom_as_boundary:
                        boundary_data["bottom"] += mid_time_data["bottom"]
        elif self.t_mode == "random":
            for bcdi, ctdip, ctdin in zip(self.block_corner_data[:-1], self.ctd_data[:-1], self.ctd_data[1:]):
                for _ in range(self.total_mid_times_boundary):
                    t_val = random.uniform(ctdip[2], ctdin[2])
                    mid_time_data = self.generate_individual_boundary_data(bcdi, t_val, mode=self.mode)
                    boundary_data["left"] += mid_time_data["left"]
                    boundary_data["right"] += mid_time_data["right"]
                    boundary_data["shared"] += mid_time_data["shared"]
                    boundary_data["top_left"] += mid_time_data["top_left"]
                    boundary_data["top_right"] += mid_time_data["top_right"]
                    if self.consider_bottom_as_boundary:
                        boundary_data["bottom"] += mid_time_data["bottom"]
        else:
            raise ValueError("Invalid t_mode. Use 'uniform' or 'random'.")
        # Convert lists to arrays
        boundary_data["left"] = np.array(boundary_data["left"])
        boundary_data["right"] = np.array(boundary_data["right"])
        boundary_data["shared"] = np.array(boundary_data["shared"])
        boundary_data["top_left"] = np.array(boundary_data["top_left"])
        boundary_data["top_right"] = np.array(boundary_data["top_right"])
        if self.consider_bottom_as_boundary:
            boundary_data["bottom"] = np.array(boundary_data["bottom"])
        return boundary_data

    @staticmethod
    def random_sample(tensor_, num_samples):
        """Randomly samples 'num_samples' points from a tensor or array."""
        random_indices = random.sample(range(len(tensor_)), min(num_samples, len(tensor_)))
        return tensor_[random_indices]

    @staticmethod
    def heat_source_origin(current_time, travel_speed, jump, x_max, x_min):
        """
        Computes the moving heat source origin.
        """
        one_lap_time = (x_max - x_min) / travel_speed  # time for one sweep
        current_lap = (current_time / one_lap_time).to(torch.int32)
        current_y = (current_lap + 1) * jump
        direction = current_lap % 2
        current_x = (travel_speed * current_time) % (x_max - x_min)
        mask = (direction == 1)
        current_x[mask] = x_max - current_x[mask]
        return [current_x, current_y]

    @staticmethod
    def heat_source_equation(x, y, current_time, travel_speed, jump, x_max, x_min):
        """
        Computes a Gaussian-like heat source whose amplitude decays with distance.
        """
        origin_x, origin_y = BlockSimulation.heat_source_origin(current_time, travel_speed, jump, x_max, x_min)
        return_data = torch.exp(-((x - origin_x) ** 2 + (y - origin_y) ** 2))
        modified_return_data = return_data.clone()
        modified_return_data[y > origin_y] = 0
        return modified_return_data

    def non_dimensionalize(self, data):
        """
        Converts the data (assumed shape [N,3]) to non-dimensional form.
        """
        lc = self.phi_laser
        tc = self.phi_laser / self.v_laser
        return data / np.array([lc, lc, tc])

    def normalize(self, nd_data):
        """
        Normalizes the non-dimensional data to the range [-1, 1] using provided bounds.
        The formula below follows the original script.
        """
        ndx_max, ndy_max, ndt_max, ndx_min, ndy_min, ndt_min = self.nd_bounds
        factor = np.array([ndx_max - ndx_min, ndy_max - ndy_min, ndt_max - ndt_min])
        offset = np.array([ndx_max, ndy_max, ndt_max]) / factor
        return 2 * (nd_data - offset) - 1.0

    def process_data(self):
        """
        Generates all internal and boundary data, non-dimensionalizes and normalizes them,
        and then converts the results to torch tensors on the available device.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        new_block_internal_data = self.generate_new_block_internal_data()
        wall_internal_data = self.generate_wall_internal_data()
        indi_boundary_data = self.generate_all_individual_boundary_data()

        nd_new_block_internal_data = self.non_dimensionalize(new_block_internal_data)
        nd_wall_internal_data = self.non_dimensionalize(wall_internal_data)
        nd_indi_boundary_data = {}
        for key, value in indi_boundary_data.items():
            nd_indi_boundary_data[key] = self.non_dimensionalize(value)

        nnd_new_block_internal_data = self.normalize(nd_new_block_internal_data)
        nnd_wall_internal_data = self.normalize(nd_wall_internal_data)
        nnd_indi_boundary_data = {}
        for key, value in nd_indi_boundary_data.items():
            nnd_indi_boundary_data[key] = self.normalize(value)

        # Convert to torch tensors
        nnd_new_block_internal_data = torch.tensor(nnd_new_block_internal_data, dtype=torch.float32).to(device)
        nnd_wall_internal_data = torch.tensor(nnd_wall_internal_data, dtype=torch.float32).to(device)
        nnd_boundary_data_left = torch.tensor(nnd_indi_boundary_data["left"], dtype=torch.float32).to(device)
        nnd_boundary_data_right = torch.tensor(nnd_indi_boundary_data["right"], dtype=torch.float32).to(device)
        nnd_boundary_data_shared = torch.tensor(nnd_indi_boundary_data["shared"], dtype=torch.float32).to(device)
        nnd_boundary_data_top_left = torch.tensor(nnd_indi_boundary_data["top_left"], dtype=torch.float32).to(device)
        nnd_boundary_data_top_right = torch.tensor(nnd_indi_boundary_data["top_right"], dtype=torch.float32).to(device)
        if self.consider_bottom_as_boundary:
            nnd_boundary_data_bottom = torch.tensor(nnd_indi_boundary_data["bottom"], dtype=torch.float32).to(device)
        else:
            nnd_boundary_data_bottom = None

        return {
            "new_block_internal": nnd_new_block_internal_data,
            "wall_internal": nnd_wall_internal_data,
            "boundary_left": nnd_boundary_data_left,
            "boundary_right": nnd_boundary_data_right,
            "boundary_shared": nnd_boundary_data_shared,
            "boundary_top_left": nnd_boundary_data_top_left,
            "boundary_top_right": nnd_boundary_data_top_right,
            "boundary_bottom": nnd_boundary_data_bottom
        }



if __name__ == "__main__":
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

    # For example, print the shapes of the generated data:
    print("New Block Internal Data:", data["new_block_internal"].shape)
    print("Wall Internal Data:", data["wall_internal"].shape)
    print("Boundary Left Data:", data["boundary_left"].shape)
    print("Boundary Right Data:", data["boundary_right"].shape)
    print("Boundary Shared Data:", data["boundary_shared"].shape)
    print("Boundary Top Left Data:", data["boundary_top_left"].shape)
    print("Boundary Top Right Data:", data["boundary_top_right"].shape)
    if consider_bottom_as_boundary:
        print("Boundary Bottom Data:", data["boundary_bottom"].shape)
