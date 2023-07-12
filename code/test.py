import numpy as np
import torch

# print(torch.unsqueeze(torch.Tensor([9, 8, 2, 1]), dim=0))
feat = torch.Tensor([0.1, 0.02, 0.18, 0.7])
ind_feat = torch.Tensor(
    [[0.1, 0.02, 0.18, 0.7], [0.1, 0.02, 0.18, 0.7], [0.3, 0.4, 0.1, 0.2]]
)
scalar = torch.Tensor([0.1, 0.9, 0.2])  # -> node mask
x = torch.Tensor([[1, 2, 0, 0], [0, 4, 0, 0], [1, 1, 1, 1]])

print(x * feat)
print(x * ind_feat)
# print(x * scalar)


a = np.array(0.9)

print(a.ndim)
print(a.reshape(-1).ndim)
