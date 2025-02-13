import torch
_ = torch.rand((3, 2))
__ = torch.rand((3, 2))
main = _ - __
print(main)
print(main/torch.tensor([1, 100000]))