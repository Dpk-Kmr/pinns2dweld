from datagen import *
import torch




def main_ode(T, X, Q, ps):
    dT_dX = torch.autograd.grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True, allow_unused = True)[0]
    dT_dt  = dT_dX[:, 0:1]
    d2T_dX2 = torch.autograd.grad(dT_dX[:, 0], X, grad_outputs=torch.ones_like(dT_dX[:, 0]), create_graph=True, allow_unused = True)[0][:, 1:]
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

    gnbd, gwd, gbd = gen_2ddata(100, 2/3, 2/3) 

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
    gnbd_input = gnbd[:, :3].clone().detach().requires_grad_(True)
    gwd_input = gwd[:, :3].clone().detach().requires_grad_(True)
    gbd_input1 = gbd[0][:, :3].clone().detach().requires_grad_(True)
    gbd_input2 = gbd[1][:, :3].clone().detach().requires_grad_(True)
    # Training loop
    num_epochs = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        T_gnbd = model(gnbd_input)
        T_gwd = model(gwd_input)
        T_gbd1 = model(gbd_input1)  
        T_gbd2 = model(gbd_input2)   

        Q = get_2dQ(gwd_input[:,1:3], gwd[:,3:5], ps)**2
        ode_loss = torch.mean(main_ode(T_gwd, gwd_input, Q, ps)**2)
        ic_loss = torch.mean(ic_eq(T_gnbd, ps)**2)
        bc1_loss = bc_eq(T_gbd1, gbd_input1, gbd[0][:,5:], ps, if_bottom = False)
        bc2_loss = bc_eq(T_gbd2, gbd_input2, gbd[1][:,5:], ps, if_bottom = True)
        final_bc_loss = torch.mean(torch.vstack((bc1_loss, bc2_loss))**2)
        final_loss = ode_loss + ic_loss + final_bc_loss
        # print(ode_loss, ic_loss, final_bc_loss)
        
        final_loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {ode_loss.item():.8f}, {ic_loss.item():.8f}, {final_bc_loss.item():.8f}, {final_loss.item():.8f}")


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
        t = 2*frame
        data = get_tdata(t)
        return torch.tensor(data[:,1:3], dtype = torch.float32)

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

    # # Initialize scatter plots (each with its own color)
    scatter_wall = ax.scatter([], [], s=6, c="black")
    # scatter_b0   = ax.scatter([], [], s=6, c="pink")
    # scatter_b1   = ax.scatter([], [], s=6, c="red")
    # scatter_b2   = ax.scatter([], [], s=6, c="blue")
    # scatter_b3   = ax.scatter([], [], s=6, c="green")
    # scatter_b4   = ax.scatter([], [], s=6, c="yellow")
    # scatter_b5   = ax.scatter([], [], s=6, c="orange")

    def init():
        # Initialize all scatter plots with an empty (0,2) array.
        empty_offsets = np.empty((0, 2))
        scatter_wall.set_offsets(empty_offsets)
    #     scatter_b0.set_offsets(empty_offsets)
    #     scatter_b1.set_offsets(empty_offsets)
    #     scatter_b2.set_offsets(empty_offsets)
    #     scatter_b3.set_offsets(empty_offsets)
    #     scatter_b4.set_offsets(empty_offsets)
    #     scatter_b5.set_offsets(empty_offsets)
        return (scatter_wall,)#, scatter_b0, scatter_b1, scatter_b2, scatter_b3, scatter_b4, scatter_b5

    def update(frame):
        data = get_data(frame)
        # Each element in data is an array of shape (n, 2)
        scatter_wall.set_offsets(np.array(data))
    #     scatter_b0.set_offsets(data[1])
    #     scatter_b1.set_offsets(data[2])
    #     scatter_b2.set_offsets(data[3])
    #     scatter_b3.set_offsets(data[4])
    #     scatter_b4.set_offsets(data[5])
    #     scatter_b5.set_offsets(data[6])
        return (scatter_wall,)# scatter_b0, scatter_b1, scatter_b2, scatter_b3, scatter_b4, scatter_b5

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=10, init_func=init,
                                interval=1000, blit=True, repeat=True)

    plt.show()

