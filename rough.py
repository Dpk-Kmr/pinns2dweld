from torch.autograd.functional import hessian, jacobian
import torch

# Define input tensor (matrix)
x = torch.tensor([[1.0, 2.0], [1.0, 2.0]], requires_grad=True)

# Define function y
def fun(x):
    return torch.hstack((x[:, (0,)]** 2, x[:, (1,)]** 3))
y = torch.hstack((torch.sum(x** 2, dim = 1).reshape(-1, 1), x[:, (1,)]** 3))  # Ensure correct shape

# Compute gradient of y[:, 0] (first column of y) w.r.t x
first_der = torch.autograd.grad(y[:, (0,)], x, create_graph= True, grad_outputs = torch.ones_like(y[:, (0,)]))
sec_der = torch.autograd.grad(first_der[0][:, (0,)], x, create_graph = True, grad_outputs = torch.ones_like(first_der[0][:, (0,)]))
thi_der = torch.autograd.grad(sec_der[0][:, (0,)], x, create_graph = True, grad_outputs = torch.ones_like(sec_der[0][:, (0,)]))
jac = jacobian(fun, x)
print("First Derivative:")
print(first_der[0])  # Since grad() returns a tuple
print(sec_der[0])
print(thi_der[0])
# print(fun(x))
# print(jac)
