'''
Save d2l functions with PyTorch
'''

import hashlib
import tarfile
import zipfile
import requests
import os, sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import matplotlib
from pathlib import Path
import shutil
import atexit
from matplotlib_inline import backend_inline
try:
    matplotlib.use('TkAgg')
    import tkinter  # TkAgg depends on Tkinter
    tkinter.Tk().destroy()
except Exception as err:  # pragma: no cover
    print('TkAgg failed:', err)
    matplotlib.use('Agg')

from d2l import torch as d2l
# def _check_d2l_path():
#     print(d2l.__file__)  # check the path of d2l
# _check_d2l_path()

d2l_save = sys.modules[__name__]

def _infer_project_root() -> Path:
    if '__file__' in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()

def clean_pycache():
    '''Clean the __pycache__ folder.'''
    root = _infer_project_root()
    for folder in root.rglob('__pycache__'):
        shutil.rmtree(folder, ignore_errors=True)

atexit.register(clean_pycache)  # clean the __pycache__ folder when the program exits

bash_path = _infer_project_root()

def use_svg_display():
    '''Use SVG display in Jupyter.'''
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    '''Set the figure size in Matplotlib.'''
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    '''Set the axes in Matplotlib.'''
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    '''Plot the data in Matplotlib.'''
    if legend is None:
        legend = []
    
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()
    
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
        
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

class Timer:
    '''Record multiple running times.'''
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        '''Start the timer.'''
        self.start_time = time.time()

    def stop(self):
        '''Stop the timer and record the time in a list.'''
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        '''Return the average time.'''
        return sum(self.times) / len(self.times)

    def sum(self):
        '''Return the sum of time.'''
        return sum(self.times)

    def format_time(self, seconds=None, precision=1):
        '''Return a formatted string given seconds in h, min and sec.'''
        total = self.sum() if seconds is None else seconds
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        secs = total % 60
        if precision is None:
            return f'{hours} h {minutes} min {int(round(secs))} sec'
        return f'{hours} h {minutes} min {secs:.{precision}f} sec'

    def cumsum(self):
        '''Return the accumulated time.'''
        return np.array(self.times).cumsum().tolist()
    
    def __str__(self):
        '''Return the string representation of the timer.'''
        return f'Time taken: {self.format_time()}'

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise (gaussian noise, N(0, 0.01)).

    Args:
        w: the weights
        b: the bias
        num_examples: the number of examples
    Returns:
        the input features and the true values
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

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

def sgd(params, lr):
    """Minibatch stochastic gradient descent.
    
    Args:
        params: the parameters
        lr: the learning rate
    """
    with torch.no_grad():  # no need to track gradient, just update the parameters
        for param in params:  # traverse the parameters
            param -= lr * param.grad  # update the parameter
            param.grad.zero_()        # reset the gradient

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.
    
    Args:
        data_arrays: a tuple of tensors
        batch_size: the batch size
        is_train: whether to shuffle the data
    Returns:
        the data iterator
    """
    dataset = data.TensorDataset(*data_arrays)  # input: a tuple of tensors
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset.
    
    Args:
        labels: the labels
    Returns:
        the text labels
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.
    
    Args:
        imgs: the images
        num_rows: the number of rows
        num_cols: the number of columns
        titles: the titles
        scale: the scale
    Returns:
        the axes
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor image
            ax.imshow(img.numpy())
        else:
            # PIL image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_workers(process_count=16):
    """Use process_count processes to read the data."""
    return process_count

def load_data_fashion_mnist(batch_size, resize=None, process_count=16):
    """Download the Fashion-MNIST dataset and then load it into memory.
    
    Args:
        batch_size: the batch size
        resize: the resize (Default: None)
    Returns:
        the training data iterator and the test data iterator
    """
    trans = [transforms.ToTensor()]  # PIL/ndarray -> Tensor [C,H,W], range [0,1]
    if resize:
        trans.insert(0, transforms.Resize(resize))  # Insert at the beginning of the list
    # 'trans' is a list of transforms, so we need to use transforms.Compose to compose them
    # Constructs and returns a callable transform pipeline object. 
    # Anything "callable" (like a function) can be put into Compose
    trans = transforms.Compose(trans)
    # 'transform' takes PIL images and returns the processed sample (transforms.Compose([transforms.ToTensor()]))
    mnist_train = torchvision.datasets.FashionMNIST(
        root=f"{bash_path}/data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=f"{bash_path}/data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers(process_count)),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers(process_count)))

def accuracy(y_hat, y):
    '''Compute the number of correct predictions.

    Args:
        y_hat: the predicted value (B, num_outputs) or (B,)
        y: the true value (B,)
    Returns:
        the number of correct predictions
    '''
    # len(y_hat.shape) > 1 and y_hat.shape[1] > 1: means y_hat is a matrix
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat_idx = y_hat.argmax(axis=1)  # y_hat_idx: (B,), obtain the index of the maximum probability
        cmp = y_hat_idx.type(y.dtype) == y  # cmp: (B,), True or False
        return float(cmp.type(y.dtype).sum())  # sum the True values, convert to float
    return 0.0  # return 0.0 if y_hat is not a matrix

def evaluate_accuracy(net, data_iter):
    '''Compute the accuracy for a model on a dataset.

    Args:
        net: the network
        data_iter: the data iterator
    Returns:
        the accuracy of the model
    '''
    if isinstance(net, torch.nn.Module): # Determine whether net is an instance of torch.nn.Module
        net.eval()  # set the model to evaluation mode
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # metric.add(correct predictions, total predictions)
    return metric[0] / metric[1]  # return the accuracy

class Accumulator:
    '''Accumulate sum over n variables.
    
    Args:
        n: the number of variables, initialize the data with n zeros
    '''
    def __init__(self, n):
        self.data = [0.0] * n  # initialize the data with n zeros

    def add(self, *args):
        '''Add the arguments to the data.
        
        Args:
            args: the arguments to add (can be a tuple)
        '''
        vals = []
        for b in args:
            if torch.is_tensor(b):
                b = b.detach()
                if b.dim() == 0:
                    vals.append(b.item())
                else:
                    vals.append(b.float().sum().item())
            else:
                vals.append(float(b))
        self.data = [a + v for a, v in zip(self.data, vals)]  # add the arguments to the data

    def reset(self):
        '''Reset the data.'''
        self.data = [0.0] * len(self.data)  # reset the data to n zeros

    def __getitem__(self, idx):  # double underscores getitem: get the data at the index
        '''Get the data at the index.'''
        return self.data[idx]  # return the data at the index

def train_epoch_ch3(net, train_iter, loss, updater): 
    '''Train the model for one epoch.

    Args:
        net: the network
        train_iter: the training data iterator
        loss: the loss function
        updater: the optimizer
    Returns:
        the average loss and accuracy
    '''
    if isinstance(net, torch.nn.Module): # Determine whether net is an instance of torch.nn.Module
        net.train()  # set the model to training mode
    metric = Accumulator(3)  # total loss, total accuracy, total number of samples
    for X, y in train_iter:
        # compute the gradient and update the parameters
        y_hat = net(X)      # y_hat: the predicted value (B, num_outputs) or (B,)
        l = loss(y_hat, y)  # l: the loss (B,); y: the true value (B,)
        if isinstance(updater, torch.optim.Optimizer):  # Determine whether updater is an instance of torch.optim.Optimizer
            # use the built-in optimizer and loss function in PyTorch
            updater.zero_grad()  # reset the gradient
            l.mean().backward()  # compute the gradient
            updater.step()       # update the parameters
        else:
            # use the custom optimizer and loss function
            l.mean().backward()  # compute the gradient
            updater()  # update the parameters
        metric.add(l.detach().sum().item(), accuracy(y_hat, y), y.numel())  # total loss, total accuracy, total number of samples
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """Animate data curves using Matplotlib interactive mode."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l_save.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        d2l.plt.ion()
        self.fig.show()
        self._closed = False

    def add(self, x, y):
        '''Add the data to the animator.
        
        Args:
            x: the x-axis data
            y: the y-axis data
        '''
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        self.fig.canvas.draw_idle()
        if hasattr(self.fig.canvas, "flush_events"):
            self.fig.canvas.flush_events()
        d2l.plt.pause(0.001)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model with multiple epochs.
    
    Args:
        net: the network
        train_iter: the training data iterator
        test_iter: the test data iterator
        loss: the loss function
        num_epochs: the number of epochs
        updater: the optimizer
        is_hold: whether to hold the animator (Default: False)
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  # train_metrics: (train_loss, train_acc)
        test_acc = evaluate_accuracy(net, test_iter)                     # test_acc: the accuracy of the test data
        animator.add(epoch + 1, train_metrics + (test_acc,))             # add the train_metrics and test_acc to the animator
    train_loss, train_acc = train_metrics
    # Check whether the data is out of bounds
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):
    """Predict labels.
    
    Args:
        net: the network
        test_iter: the test data iterator
        n: the number of images to predict (Default: 6)
    """
    for X, y in test_iter:
        break
    trues = d2l_save.get_fashion_mnist_labels(y)
    preds = d2l_save.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l_save.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

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

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=None):
    '''Download a file inserted into DATA_HUB, and return the local filename.'''
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    if cache_dir is None:
        # Place data directory at the project root regardless of CWD
        repo_root = os.path.abspath(os.path.dirname(__file__))
        cache_dir = os.path.join(repo_root, 'data')
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # cache hit
    print(f"Downloading {fname} from {url}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    '''Download and extract a zip/tar file.'''
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'only zip/tar files can be extracted'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    '''Download all files in DATA_HUB'''
    for name in DATA_HUB:
        download(name)

DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists. """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def corr2d(X, K):
    """Calculate 2D cross-correlation.
    
    Args:
        X: the input tensor
        K: the kernel tensor
    Returns:
        the output tensor
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

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
