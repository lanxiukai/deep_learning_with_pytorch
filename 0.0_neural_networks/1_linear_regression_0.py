'''
Linear Regression
'''

import random
import torch

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    """Generate a small stochastic batch of data.
    
    Args:
        batch_size: the size of the batch
        features: the input features
        labels: the true values

    Returns:
        a small stochastic batch of data
    """
    num_examples = len(features)  # features.shape[0]
    indices = list(range(num_examples))
    # These examples are stochastically shuffled.
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    """The linear regression model.

    Args:
        X: the input features
        w: the weights
        b: the bias

    Returns:
        the predicted value
    """
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """Squared loss.
    
    Args:
        y_hat: the predicted value
        y: the true value

    Returns:
        the squared loss
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():  # no need to track gradient, just update the parameters
        for param in params:  # traverse the parameters
            param -= lr * param.grad / batch_size  # update the parameter
            param.grad.zero_()                     # reset the gradient

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10

    # Initialize the weights and bias and automatically tracks gradient
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    num_epochs = 3
    lr = 0.03
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # 'X': the features, 'y': the true values
            # Sum each element in 'l', then calculate the gradient of the parameters.
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # Update parameters using their gradient.
        with torch.no_grad():  # no need to track gradient, just judge the loss
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.sum()):f}')

    print(f'The estimated error in w: {true_w - w.reshape(true_w.shape)}')
    print(f'The estimated error in b: {true_b - b}')
