from datagen import *
import torch
import random



########################  check ode code perfectly  ######################################
def main_ode(T, X, Q, ps):
    # create_graph keep track of previous derivatives
    dT_dX = torch.autograd.grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True, allow_unused = True)[0]
    dT_dt  = dT_dX[:, (0,)]
    sec_der1 = torch.autograd.grad(dT_dX[:, (1,)], X, grad_outputs=torch.ones_like(dT_dX[:, (1,)]), create_graph=True, allow_unused = True)[0][:, (1,)]
    sec_der2 = torch.autograd.grad(dT_dX[:, (2,)], X, grad_outputs=torch.ones_like(dT_dX[:, (2,)]), create_graph=True, allow_unused = True)[0][:, (2,)]
    d2T_dX2 = torch.hstack((sec_der1, sec_der2))
    return dT_dt\
          - (ps["k"]/(ps["rho"]*ps["cp"]))*(ps["tc"]/(ps["lc"]**2))*(torch.sum(d2T_dX2, dim = 1).reshape(-1, 1)) \
            - Q

def get_2dQ(X, QX, ps):
    multiplying_factor = (6*ps["tc"]*ps["eta"]*ps["p"])/(torch.pi*ps["rho"]*ps["cp"]*ps["Tc"]*ps["a"]*ps["b"])
    bracket_term = ((X-QX)**2)/(torch.tensor([ps["a"]**2, ps["b"]**2]))
    exp_factor = -3*(ps["lc"]**2)*(torch.sum(bracket_term, dim = 1).reshape(-1, 1))
    return multiplying_factor*torch.exp(exp_factor)

def bc_eq(T, X, nX, ps, if_bottom = False):
    dT_dX = torch.autograd.grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True, allow_unused = True)[0][:,1:]
    if not if_bottom:
        return torch.sum(-ps["k"]*(dT_dX*nX), dim = 1).reshape(-1, 1)\
              - ps["h"]*ps["lc"]*(T - (ps["T_amb"]/ps["Tc"]))\
                  - ps["sigma"]*ps["epsilon"]*(ps["Tc"]**3)*ps["lc"]*(T**4 - (ps["T_amb"]/ps["Tc"])**4)

    return torch.sum(-ps["k"]*(dT_dX*nX), dim = 1).reshape(-1, 1)\
            - ps["h_force"]*ps["lc"]*(T - (ps["T_amb"]/ps["Tc"]))\
                  
def ic_eq(T, ps):
    return T - ps["T_ref"]/ps["Tc"]



if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation


    # Define the neural network model
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(3, 64) 
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 64)
            self.fc4 = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))  # Tanh activation
            x = torch.tanh(self.fc2(x))  # Tanh activation
            x = torch.tanh(self.fc3(x))
            x = self.fc4(x)  # No activation in the output layer
            return x

    # Generate some synthetic training data (for example)
    np.random.seed(42)
    torch.manual_seed(42)
    ps = {}
    ps["k"] = 1.14e1*1e1
    ps["rho"] = 4.5e-9*1e3
    ps["cp"] = 7.14e8
    ps["tc"] = 2
    ps["lc"] = 0.2
    ps["eta"] = 4
    ps["p"] = 3.5e5
    ps["Tc"] = 3000
    ps["a"] = 0.1
    ps["b"] = 0.2
    ps["h"] = 2e-2*1e2
    ps["T_amb"] = 298
    ps["sigma"] = 5.67e-11*1e2
    ps["epsilon"] = 3e-1
    ps["h_force"] = 2e-2*1e2
    ps["T_ref"] = 3000

    gnbd, gwd, gbd = gen_2ddata(10, 2/3, 2/3) 

    gnbd, gwd, gbd = nd_data(gnbd, gwd, gbd, ps, ti = [0,], xs = [1, 2, 3, 4])

    gnbd, gwd, gbd = nnd_data(gnbd, gwd, gbd, ti = [0,], xs = [1, 2], apply_inds = [3, 4])


    # Convert to PyTorch tensors
    gnbd = torch.tensor(gnbd, dtype=torch.float32)
    gwd = torch.tensor(gwd, dtype=torch.float32)
    for i in range(len(gbd)):
        gbd[i] = torch.tensor(gbd[i], dtype=torch.float32)

    # Define model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.MSELoss()  # Mean Squared Error loss (for regression)
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer
    gnbd_input = gnbd[:, :3].clone().detach()
    gwd_input = gwd[:, :3].clone().detach()
    gbd_input1 = gbd[0][:, :3].clone().detach()
    gbd_input2 = gbd[1][:, :3].clone().detach()
    # Training loop
    num_epochs = 20
    print(gnbd_input.size())
    print(gwd_input.size())
    print(gbd_input1.size())
    print(gbd_input2.size())
    # rs for rows
    gnbd_rs = gnbd_input.size()[0]
    gwd_rs = gwd_input.size()[0]
    gbd1_rs = gbd_input1.size()[0]
    gbd2_rs = gbd_input2.size()[0]
    training_fraction = 0.01
    for epoch in range(num_epochs):
        gnbd_randoms = random.sample(range(gnbd_rs), gnbd_rs)
        gwd_randoms = random.sample(range(gwd_rs), gwd_rs)
        gbd1_randoms = random.sample(range(gbd1_rs), gbd1_rs)
        gbd2_randoms = random.sample(range(gbd2_rs), gbd2_rs)
        data_wins = np.arange(0, 1+training_fraction, training_fraction)
        epoch_losses = [0, 0, 0, 0] 
        batch_count = 0
        for s, e in zip(data_wins[:-1], data_wins[1:]):
            optimizer.zero_grad()
            T_gnbd_inputs = gnbd_input[gnbd_randoms[int(gnbd_rs*s):int(gnbd_rs*e)]].requires_grad_(True)
            T_gnbd = model(T_gnbd_inputs)
            T_gwd_inputs = gwd_input[gwd_randoms[int(gwd_rs*s):int(gwd_rs*e)]].requires_grad_(True)
            T_gwd = model(T_gwd_inputs)
            T_gbd1_inputs = gbd_input1[gbd1_randoms[int(gbd1_rs*s):int(gbd1_rs*e)]].requires_grad_(True)
            T_gbd1 = model(T_gbd1_inputs)  
            T_gbd2_inputs = gbd_input2[gbd2_randoms[int(gbd2_rs*s):int(gbd2_rs*e)]].requires_grad_(True)
            T_gbd2 = model(T_gbd2_inputs)   

            Q = get_2dQ(gwd_input[gwd_randoms[int(gwd_rs*s):int(gwd_rs*e)],1:3], gwd[gwd_randoms[int(gwd_rs*s):int(gwd_rs*e)],3:5], ps)
            ode_loss = torch.mean(main_ode(T_gwd, T_gwd_inputs, Q, ps)**2)
            ic_loss = torch.mean(ic_eq(T_gnbd, ps)**2)
            bc1_loss = bc_eq(T_gbd1, T_gbd1_inputs, gbd[0][int(gbd1_rs*s):int(gbd1_rs*e),5:], ps, if_bottom = False)
            bc2_loss = bc_eq(T_gbd2, T_gbd2_inputs, gbd[1][int(gbd2_rs*s):int(gbd2_rs*e),5:], ps, if_bottom = True)
            # final_bc_loss = torch.mean(torch.vstack((bc1_loss, bc2_loss))**2)
            final_bc_loss = (torch.mean(bc1_loss**2) + torch.mean(bc2_loss**2))/2
            final_loss = ode_loss + 100*ic_loss + 100*final_bc_loss

            
            final_loss.backward()
            optimizer.step()
            epoch_losses[0] += final_loss.item()
            epoch_losses[1] += ode_loss.item()
            epoch_losses[2] += ic_loss.item()
            epoch_losses[3] += final_bc_loss.item()
            batch_count += 1

        epoch_l = [i / batch_count if batch_count > 0 else float('inf') for i in epoch_losses]
        print(f"[{epoch+1}/{num_epochs}], {epoch_l[0]:.8f}, {epoch_l[1]:.8f}, {epoch_l[2]:.8f}, {epoch_l[3]:.8f}")
        # if epoch % 1 == 0:
        #     print(T_gwd[:5], T_gwd_inputs[:5])
            # # if epoch % 1 == 0:
            # print(f"Epoch [{epoch}/{num_epochs}], Loss: {ode_loss.item():.8f}, {ic_loss.item():.8f}, {final_bc_loss.item():.8f}, {final_loss.item():.8f}")

    # testing data
    def get_tdata(time):
        return get_t_wall_data(
            time, 2/3, 2/3, 
            x_max = 6, 
            y_max = 6, 
            block_dx = 0.3, 
            block_dy = 0.3,
            mode = "uniform",
            wall_density = 1000,
            increase_latest_block_data = False,
            increased_boundary_data = False)


    def get_data(frame):
        t = frame/100
        data = get_tdata(t)
        txy_data =torch.tensor(data[:,:3], dtype = torch.float32)
        T_data = model(txy_data)
        
        return torch.hstack((txy_data, T_data))

    # # Define figure dimensions
    x_max, y_max = 6, 6

    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1, x_max+1)
    ax.set_ylim(-1, y_max+1)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Welding Wall")
    ax.grid(False)
    global colors
    colors = []
    scatter_wall = ax.scatter([], [], s=0.5, c=colors)
    scatter_wall = ax.scatter([], [], s=0.5, c=[], cmap='plasma')
    scatter_wall.set_clim(vmin=0.0, vmax=1.0)  
    cbar = fig.colorbar(scatter_wall, ax=ax)

    def init():

        empty_offsets = np.empty((0, 2))
        colors =[]
        scatter_wall.set_offsets(empty_offsets)
        scatter_wall.set_array(colors)

        return (scatter_wall,)

    def update(frame):
        data = get_data(frame)
        xy_data = data[:,1:3].detach().clone().numpy()
        colors = data[:,3].detach().clone().numpy()
        print(colors)

        scatter_wall.set_offsets(xy_data)
        scatter_wall.set_array(colors)
        return (scatter_wall,)

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=1000, init_func=init,
                                interval=1000, blit=True, repeat=True)

    plt.show()

