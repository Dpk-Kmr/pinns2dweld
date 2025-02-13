from datagen import *
import torch




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

def bc_eq(T, X, nX, ps, if_bottom = False):
    dT_dX = torch.autograd.grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True)[0][1:]
    if not if_bottom:
        return torch.sum(-ps["k"]*(dT_dX*nX), dim = 1).reshape(-1, 1)\
              - ps["h"]*ps["lc"]*(T - ps["T_amb"])\
                  - ps["sigma"]*ps["epsilon"]*(ps["Tc"]**3)*ps["lc"]*(T**4 - ps["T_amb"]**4)
    if if_bottom:
        return torch.sum(-ps["k"]*(dT_dX*nX), dim = 1).reshape(-1, 1)\
              - ps["h_force"]*ps["lc"]*(T - ps["T_amb"])\
                  
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
            self.fc1 = nn.Linear(3, 64)  # First hidden layer (input: 3 neurons)
            self.fc2 = nn.Linear(64, 64)  # Second hidden layer
            self.fc3 = nn.Linear(64, 1)   # Output layer (1 neuron)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))  # Tanh activation
            x = torch.tanh(self.fc2(x))  # Tanh activation
            x = self.fc3(x)  # No activation in the output layer
            return x

    # Generate some synthetic training data (for example)
    np.random.seed(42)
    torch.manual_seed(42)
    ps = {}
    ps["k"] = 1.14e1
    ps["rho"] = 4.5e-9
    ps["cp"] = 7.14e8
    ps["tc"] = 
    ps["lc"]
    ps["eta"]
    ps["p"]
    ps["Tc"]
    ps["a"]
    ps["b"]
    ps["h"]
    ps["T_amb"]
    ps["sigma"]
    ps["epsilon"]
    ps["h_force"]
    ps["T_ref"]

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

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        T_gnbd = model(gnbd[:,:3])
        T_gwd = model(gwd[:,:3])
        T_gbd = []
        for i in range(len(gbd)):
            T_gbd[i] = model(gbd[i][:,:3])
        
        Q = get_2dQ(gwd[:,:3], gwd[:,3:5], ps)**2
        ode_loss = torch.mean(main_ode(T_gwd, gwd[:,:3], Q, ps))
        ic_loss = torch.mean(ic_eq(T_gnbd, ps))
        bc1_loss = bc_eq(T_gbd[0], gbd[0][:,:3], gbd[0][:,5:7], ps, if_bottom = False)
        bc2_loss = bc_eq(T_gbd[1], gbd[1][:,:3], gbd[1][:,5:7], ps, if_bottom = True)
        final_bc_loss = torch.mean(torch.vstack((bc1_loss, bc2_loss)))
        print(ode_loss, ic_loss, final_bc_loss)



        loss = criterion(outputs, y_train)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}")

    # Test on a new sample
    sample_input = torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32)
    prediction = model(sample_input)
    print("\nSample Input:", sample_input.numpy())
    print("Model Prediction:", prediction.item())





