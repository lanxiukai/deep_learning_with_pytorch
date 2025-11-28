'''
LeNet-5 using GPU
'''

import torch
from torch import nn
from d2l_importer import d2l_save

def evaluate_accuracy_gpu(net, data_iter, device=None):
    '''Evaluate the accuracy of the model on the given dataset using GPU.'''
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            # Get the device of the first parameter of the net
            device = next(iter(net.parameters())).device
    metric = d2l_save.Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT fine-tuning (to be introduced later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l_save.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]  # Return the accuracy

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device, net_name=None):
    '''Train the model using GPU.'''
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print(f'{net_name} training on {device}')
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l_save.Animator(xlabel='epoch', xlim=[1, num_epochs], 
                                 legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l_save.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # (training) loss_sum, total number of correct predictions, total number of samples
        metric = d2l_save.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l_save.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    print(timer)

def print_net(net):
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))

    # print_net(net)
    lr, num_epochs, batch_size = 0.9, 10, 256
    train_iter, test_iter = d2l_save.load_data_fashion_mnist(batch_size)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, 
              device=d2l_save.try_gpu(), net_name='LeNet-5')
    
    # d2l_save.plt.ioff()
    # d2l_save.plt.show()
