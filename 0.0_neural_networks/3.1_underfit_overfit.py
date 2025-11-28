'''
Model Selection, Underfitting and Overfitting (Polynomial Regression)
'''

import math
import numpy as np
import torch
from torch import nn
from d2l_importer import d2l_save

def evaluate_loss(net, data_iter, loss):
    '''Evaluate the model’s loss on the given dataset.
    
    Args:
        net: the model
        data_iter: the data iterator
        loss: the loss function
    Returns:
        the average loss
    '''
    metric = d2l_save.Accumulator(2)  # loss_sum, num_samples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]  # the number of features
    # Set bias=False, because we have already considered the bias in the polynomial features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l_save.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l_save.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l_save.Animator(xlabel='epoch', ylabel='loss', yscale='log', 
                                 xlim=[1, num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l_save.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), 
                                     evaluate_loss(net, test_iter, loss)))
    print(f'weight: {net[0].weight.data.numpy()}')

if __name__ == '__main__':
    max_degree = 20  # maximum order of the polynomial
    n_train, n_test = 100, 100  # number of training and testing samples
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))  # Sample-wise power operation
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(n) = (n-1)!
    labels = np.matmul(poly_features, true_w)  # (n_train + n_test,)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    # convert to tensor
    true_w, features, poly_features, labels = [
        torch.tensor(x, dtype=torch.float32)
        for x in (true_w, features, poly_features, labels)
    ]

    train(poly_features[:n_train, :4], poly_features[n_train:, :4], 
          labels[:n_train], labels[n_train:])

    train(poly_features[:n_train, :2], poly_features[n_train:, :2], 
          labels[:n_train], labels[n_train:], num_epochs=1500)

    train(poly_features[:n_train, :20], poly_features[n_train:, :20], 
          labels[:n_train], labels[n_train:], num_epochs=1500)
