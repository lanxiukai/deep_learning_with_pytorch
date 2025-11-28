'''
Multilayer Perceptron
'''

import torch
from torch import nn
from d2l_importer import d2l_save

def relu(X):
    return torch.max(X, torch.zeros_like(X))

def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X @ W1 + b1)
    return H @ W2 + b2

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l_save.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    loss = nn.CrossEntropyLoss(reduction='none')
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)

    timer = d2l_save.Timer()
    d2l_save.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    d2l_save.predict_ch3(net, test_iter)
    print(f'{timer.stop():.3f} sec')
    # d2l_save.plt.ioff()
    # d2l_save.plt.show()
