"""
Attention Pooling (parametric)
"""
import torch
from torch import nn
from d2l_importer import d2l_save

n_train = 50  # Number of training samples
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # Sorted training samples from minimum to maximum

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # Outputs of training samples
x_test = torch.arange(0, 5, 0.1)  # Test samples
y_truth = f(x_test)  # Ground-truth outputs of test samples
n_test = len(x_test)  # Number of test samples

def plot_kernel_reg(y_hat):
    d2l_save.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l_save.plt.plot(x_train, y_train, 'o', alpha=0.5);

X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(torch.bmm(X, Y).shape)  # batch matrix multiplication

weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # Shape of queries and attention_weights: (num_queries, num_key_value_pairs)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of values: (num_queries, num_key_value_pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

# Shape of X_tile: (n_train, n_train), each row contains the same training input
X_tile = x_train.repeat((n_train, 1))
# Shape of Y_tile: (n_train, n_train), each row contains the same training output
Y_tile = y_train.repeat((n_train, 1))
# Shape of keys: (n_train, n_train - 1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# Shape of values: (n_train, n_train - 1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l_save.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    loss_value = l.sum().detach().item()
    print(f'epoch {epoch + 1}, loss {loss_value:.6f}')
    animator.add(epoch + 1, loss_value)

"""
# Shape of keys: (n_test, n_train), each row contains the same training input (e.g., the same key)
keys = x_train.repeat((n_test, 1))
# Shape of values: (n_test, n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

d2l_save.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
"""

d2l_save.plt.ioff()
d2l_save.plt.show()
