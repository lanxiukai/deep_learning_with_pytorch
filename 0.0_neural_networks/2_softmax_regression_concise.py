'''
Softmax Regression (Concise Implementation)
'''

import torch
from torch import nn
from d2l_importer import d2l_save

def init_weights(m):
    '''Initialize the weights of the linear layer.'''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l_save.load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10
    # A flatten layer is defined before the linear layer to reshape the network input
    net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))
    # Flatten layer (No parameters) : reshape the network input from (B, 1, 28, 28) to (B, 784)
    # Linear layer (Parameters): the output of the linear layer is (B, num_outputs)
    
    net.apply(init_weights)  # Initialize the weights of the linear layer
    loss = nn.CrossEntropyLoss(reduction='none') # Cross Entropy Loss function: LogSoftmax + NLLLoss
    # reduction='none' means the loss will not be reduced, and the loss will be a vector (B,)
    trainer = torch.optim.SGD(net.parameters(), lr=0.1) # SGD optimizer
    num_epochs = 10

    timer = d2l_save.Timer()
    d2l_save.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l_save.predict_ch3(net, test_iter)
    print(f'{timer.stop():.3f} sec')
    # d2l_save.plt.ioff()
    # d2l_save.plt.show()
