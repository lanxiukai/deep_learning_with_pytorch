'''
Activation Functions
'''

import torch
from d2l_importer import d2l_save

# ReLU activation function
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l_save.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
d2l_save.plt.show()

# Gradient of ReLU
y.backward(torch.ones_like(x), retain_graph=True)
d2l_save.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
d2l_save.plt.show()

# Sigmoid activation function
y = torch.sigmoid(x)
d2l_save.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
d2l_save.plt.show()

# Gradient of Sigmoid
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l_save.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
d2l_save.plt.show()

# Tanh activation function
y = torch.tanh(x)
d2l_save.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
d2l_save.plt.show()

# Gradient of Tanh
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l_save.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l_save.plt.show()
