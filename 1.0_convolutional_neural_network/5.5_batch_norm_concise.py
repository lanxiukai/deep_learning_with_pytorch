'''
Batch Normalization (Concise Implementation)
'''

from torch import nn
from d2l_importer import d2l_save

if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
        nn.Linear(84, 10))

    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l_save.load_data_fashion_mnist(batch_size)
    d2l_save.train_ch6(net, train_iter, test_iter, num_epochs, lr, 
                       device=d2l_save.try_gpu(), net_name='LeNet-5 with BatchNorm')
    
    # d2l_save.plt.ioff()
    # d2l_save.plt.show()
