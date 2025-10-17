'''
GPU computation based on PyTorch (By default, tensors are created on the CPU)
'''

import torch
from torch import nn

print('0--------------------------------')
print(f'torch.device("cpu"): {torch.device("cpu")}')
print(f'torch.device("cuda"): {torch.device("cuda")}')
print(f'torch.device("cuda:1"): {torch.device("cuda:1")}')
print(torch.cuda.device_count())

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists. """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(f'try_gpu(): {try_gpu()}')
print(f'try_gpu(10): {try_gpu(10)}')
print(f'try_all_gpus(): {try_all_gpus()}')

x = torch.tensor([1, 2, 3])
print(f'x device: {x.device}')

X = torch.ones(2, 3, device=try_gpu())
print(f'X: {X}')

# Operations on the same device
Y = torch.rand(2, 3, device=try_gpu())
print(f'Y: {Y}')
print(f'X + Y: {X + Y}') # X + Y will be computed on GPU 0
print('1--------------------------------')

# Cross-device operations
X_1 = X.to(try_gpu(1)) # X_1 = X.cuda(1) if you have two GPUs
Y_1 = Y.to(try_gpu(1)) # Y_1 = Y.cuda(1) if you have two GPUs
print(f'X_1: {X_1}')
print(f'Y_1: {Y_1}')
print(f'X_1 + Y_1: {X_1 + Y_1}') # X_1 + Y_1 will be computed on GPU 1 (CPU)
print('2--------------------------------')

net = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)
