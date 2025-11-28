'''
Weight Decay (L2 Regularization)
'''

import torch
from torch import nn
from d2l_importer import d2l_save

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l_save.synthetic_data(true_w, true_b, n_train)
train_iter = d2l_save.load_array(train_data, batch_size)
test_data = d2l_save.synthetic_data(true_w, true_b, n_test)
test_iter = d2l_save.load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l_save.linreg(X, w, b), d2l_save.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l_save.Animator(xlabel='epochs', ylabel='loss', yscale='log', 
                                 xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # Added an L2 norm penalty term
            # The broadcasting mechanism makes l2_penalty(w) a vector of length batch_size
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.mean().backward()
            d2l_save.sgd([w, b], lr)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l_save.evaluate_loss(net, train_iter, loss), 
                                     d2l_save.evaluate_loss(net, test_iter, loss)))
    print(f'L2 norm of w: ', torch.norm(w).item())

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # The bias does not have a weight_decay parameter, so it is not penalized
    trainer = torch.optim.SGD([
        {"params": net[0].weight, "weight_decay": wd}, 
        {"params": net[0].bias}], lr=lr)
    animator = d2l_save.Animator(xlabel='epochs', ylabel='loss', yscale='log', 
                                 xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l_save.evaluate_loss(net, train_iter, loss), 
                                     d2l_save.evaluate_loss(net, test_iter, loss)))
    print(f'L2 norm of w: ', net[0].weight.norm().item())

if __name__ == '__main__':
    train(lambd=0)
    train(lambd=3)
    train_concise(0)
    train_concise(3)
    # d2l_save.plt.ioff()
    # d2l_save.plt.show()
