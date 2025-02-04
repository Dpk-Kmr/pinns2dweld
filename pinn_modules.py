from modules import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
torch.autograd.set_detect_anomaly(True)



class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving PDE-based problems.
    In this example, a heat-type PDE is used with a moving heat source and
    dynamic geometry.
    """
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss()

        # Create fully-connected layers
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
        -----------
        x: torch.Tensor (N, D)
            N data samples, each with dimension D (here, D=3 for x, y, t).

        Returns:
        --------
        torch.Tensor (N, 1)
            The network's prediction for the field variable (e.g., temperature).
        """
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x

    def predict(self, x):
        """
        Helper function for inference in no_grad mode.
        """
        with torch.no_grad():
            return self.forward(x)




def equation(u, X, f, k = 11.4, rho = 4.5*(10**(-9)), cp = 7.14*(10**8), tc = 0.2, lc = 2):
    """
    PDE Residual function:
    Residual = dU/dt - Laplacian(U) - f(x,y,t)

    For illustration, the code includes:
        residual = dU/dt - (d2U/dx2 + d2U/dy2) - f

    Note:
    -----
    The code below simply lumps everything together as a demonstration
    of PDE derivatives with autograd. Modify for the actual PDE you're solving.

    Parameters:
    -----------
    u: torch.Tensor
        The predicted solution, shape (N, 1).
    X: torch.Tensor
        The input coordinates (x, y, t), shape (N, 3).
    f: torch.Tensor
        The source term, shape (N, 1).

    Returns:
    --------
    torch.Tensor
        PDE residual, shape (N, 1).
    """
    # Compute gradient of u w.r.t X = (x, y, t)
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    # du_d has shape (N, 3) => [dU/dx, dU/dy, dU/dt]

    # Compute second derivatives
    d2u_dx2 = torch.autograd.grad(du_d[:, 0], X, grad_outputs=torch.ones_like(du_d[:, 0]),
                                  create_graph=True)[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(du_d[:, 1], X, grad_outputs=torch.ones_like(du_d[:, 1]),
                                  create_graph=True)[0][:, 1:2]
    d_u_dt  = du_d[:, 2:3]  # first derivative in time

    # Example PDE: heat equation with source => dU/dt - Laplacian(U) - f = 0
    # Laplacian(U) = d2U/dx2 + d2U/dy2
    return d_u_dt - (k/rho*cp)*(tc/(lc**2))(d2u_dx2 + d2u_dy2) - f



def bc_left(u, X, h = 0.02, u_amb = 298/3000,
            sigma= 5.67*(10**(-11)), epsilon = 0.3,
            k = 11.4, lc = 2.0, tc = 3000):
    """

    """
    # Compute gradient of u w.r.t X = (x, y, t)
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    du_dx  = du_d[:, 0:1]  # first derivative in x

    conv = (h*lc/k)*(u-u_amb)

    rad = (sigma*epsilon*(tc**3)*lc/k)(u**4 - u_amb**4)


    return du_dx - conv - rad


def bc_right(u, X, h = 0.02, u_amb = 298/3000,
             sigma= 5.67*(10**(-11)), epsilon = 0.3,
             k = 11.4, lc = 2.0, tc = 3000):
    """

    """
    return bc_left(u, X, h = 0.02, u_amb = 298/3000,
                   sigma= 5.67*(10**(-11)), epsilon = 0.3,
                   k = 11.4, lc = 2.0, tc = 3000)

def bc_shared(u, X, h = 0.02, u_amb = 298/3000,
              sigma= 5.67*(10**(-11)), epsilon = 0.3,
              k = 11.4, lc = 2.0, tc = 3000):
    """

    """
    return bc_left(u, X, h = 0.02, u_amb = 298/3000,
                   sigma= 5.67*(10**(-11)), epsilon = 0.3,
                   k = 11.4, lc = 2.0, tc = 3000)


def bc_top_left(u, X, h = 0.02, u_amb = 298/3000,
                sigma= 5.67*(10**(-11)), epsilon = 0.3,
                k = 11.4, lc = 2.0, tc = 3000):
    """

    """
    # Compute gradient of u w.r.t X = (x, y, t)
    du_d = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    du_dy  = du_d[:, 1:0]  # first derivative in x
    conv = h*(u-u_amb)
    rad = sigma*epsilon*(u**4 - u_amb**4)

    return k*du_dy - conv - rad


def bc_top_right(u, X, h = 0.02, u_amb = 298/3000,
                 sigma= 5.67*(10**(-11)), epsilon = 0.3,
                 k = 11.4, lc = 2.0, tc = 3000):
    """

    """
    return bc_top_left(u, X, h = 0.02, u_amb = 298/3000,
                       sigma= 5.67*(10**(-11)), epsilon = 0.3,
                       k = 11.4, lc = 2.0, tc = 3000)

# The boundary may not be apprepriate for our case
def bc_bottom(u, X, h = 0.02, u_amb = 298/3000,
              sigma= 5.67*(10**(-11)), epsilon = 0.3,
              k = 11.4, lc = 2.0, tc = 3000):
    """

    """
    return bc_top_left(u, X, h = 0.02, u_amb = 298/3000,
                       sigma= 5.67*(10**(-11)), epsilon = 0.3,
                       k = 11.4, lc = 2.0, tc = 3000)
