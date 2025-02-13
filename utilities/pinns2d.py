from datagen import *
import torch

def gen_2ddata(
        tot_time, x_speed, y_speed, 
        movement_type = "bidirectional", 
        if_continuous = False, 
        if_discrete = True, 
        x_max = 6, 
        y_max = 6, 
        block_dx = 0.3, 
        block_dy = 0.3,
        start_t = 0.0, 
        mode = "random", 
        ic_density = 500,
        wall_density = 10, 
        new_density = 100,
        boundary_density = 50,
        bc_density = 50,
        increase_latest_block_data = True,
        increased_boundary_data = True,
        boundary_width = None, 
        top_boundary_layers = 2,
        t_grid = "random",
        t_density = 10, 
        bottom_data = True,
        bc_groups = [[0, 1, 2], [3, 4], [5,]]      
):
    gnbd = get_newblock_data(
            tot_time, x_speed, y_speed, 
            movement_type = movement_type, 
            if_continuous = if_continuous, 
            if_discrete = if_discrete, 
            x_max = x_max, 
            y_max = y_max, 
            block_dx = block_dx, 
            block_dy = block_dy,
            start_t = start_t, 
            mode = mode, 
            density = ic_density
    )


    gwd = get_wall_data(
            tot_time, x_speed, y_speed, 
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
            top_boundary_layers = top_boundary_layers,
            t_grid = t_grid,
            t_density = t_density     
    )
        

    gbd = get_boundary_data(
            tot_time, x_speed, y_speed, 
            movement_type = movement_type, 
            if_continuous = if_continuous, 
            if_discrete = if_discrete, 
            x_max = x_max, 
            y_max = y_max, 
            block_dx = block_dx, 
            block_dy = block_dy,
            start_t = start_t, 
            mode = mode, 
            density = bc_density, 
            bottom_data = bottom_data,
            groups = bc_groups, 
            t_grid = t_grid,
            t_density = t_density     
    )
    return gnbd, gwd, gbd


def main_ode(T, X, Q, ps):
    dT_dX = torch.autograd.grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    dT_dt  = dT_dX[:, 0:1]
    d2T_dX2 = torch.autograd.grad(dT_dX[:, 0], X, grad_outputs=torch.ones_like(dT_dX[:, 0]), create_graph=True)[0][:, 1:]
    return dT_dt\
          - (ps["k"]/(ps["rho"]*ps["cp"]))*(ps["tc"]/(ps["lc"]**2))*(torch.sum(d2T_dX2, dim = 1).reshape(-1, 1)) \
            - Q

def get_2dQ(X, QX, ps):
    multiplying_factor = (6*ps["tc"]*ps["eta"]*ps["p"])/(torch.pi*ps["rho"]*ps["cp"]*ps["Tc"]*ps["a"]*ps["b"])
    bracket_term = ((X-QX)**2)/(torch.tensor([ps["a"]**2, ps["b"]**2]))
    exp_factor = -3*(ps["lc"]**2)*(torch.sum(bracket_term, dim = 1).reshape(-1, 1))
    return multiplying_factor*torch.exp(exp_factor)

def vbc_eq(T, X, ps):
    
