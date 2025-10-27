'''
AlexNet
'''

import torch
from torch import nn
from d2l_importer import d2l, d2l_save

def print_net(net):
    X = torch.randn(size=(1, 1, 224, 224), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), 
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10))
    
    # print_net(net)
    
    lr, num_epochs, batch_size = 0.01, 10, 128
    train_iter, test_iter = d2l_save.load_data_fashion_mnist(batch_size, resize=224)
    d2l_save.train_ch6(net, train_iter, test_iter, num_epochs, lr, 
                       device=d2l_save.try_gpu(), net_name='AlexNet')
    # d2l.plt.ioff()
    # d2l.plt.show()
