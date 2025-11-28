'''
Dropout (Concise Implementation)
'''

import torch
from torch import nn
from d2l_importer import d2l_save

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5
    num_epochs, lr, batch_size = 10, 0.5, 256
    net = nn.Sequential(nn.Flatten(), 
                        nn.Linear(num_inputs, num_hiddens1), 
                        nn.ReLU(), 
                        nn.Dropout(dropout1), 
                        nn.Linear(num_hiddens1, num_hiddens2), 
                        nn.ReLU(), 
                        nn.Dropout(dropout2), 
                        nn.Linear(num_hiddens2, num_outputs))

    loss = nn.CrossEntropyLoss(reduction='none')
    net.apply(init_weights)
    train_iter, test_iter = d2l_save.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    timer = d2l_save.Timer()
    d2l_save.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    print(f'{timer.stop():.3f} sec')
    # d2l_save.plt.ioff()
    # d2l_save.plt.show()
