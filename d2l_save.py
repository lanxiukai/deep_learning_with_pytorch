"""
Save d2l modules with PyTorch
"""

import hashlib
import tarfile
import zipfile
import requests
import os, sys
import time
from pathlib import Path
import shutil
import atexit
import re
import collections
import random
import math
from typing import Callable, Tuple, Optional, Union, Sequence
from dataclasses import dataclass, replace, fields, is_dataclass

import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.nn import functional as F

from matplotlib import pyplot as plt
import matplotlib
from matplotlib_inline import backend_inline

from tqdm.auto import tqdm


try:
    matplotlib.use('TkAgg')
    import tkinter  # TkAgg depends on Tkinter
    tkinter.Tk().destroy()
except Exception as err:  # pragma: no cover
    print('TkAgg failed:', err)
    matplotlib.use('Agg')

d2l_save = sys.modules[__name__]

def _infer_project_root() -> Path:
    if '__file__' in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()

def clean_pycache():
    """Clean the __pycache__ folder when the program exits."""
    root = _infer_project_root()
    for folder in root.rglob('__pycache__'):
        shutil.rmtree(folder, ignore_errors=True)

atexit.register(clean_pycache)  # clean the __pycache__ folder when the program exits

base_path = _infer_project_root()

def reset_dir(path: str) -> None:
    """
    Remove a file/dir at `path` if it exists, then (re)create it as a directory.

    This is useful for experiment output folders (ensure a clean run).
    """
    if not path:
        raise ValueError("reset_dir: path is empty.")

    abs_path = os.path.abspath(path)
    if abs_path == os.path.abspath(os.path.sep):
        raise ValueError(f"reset_dir: refuse to delete root directory: {path!r}")

    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    os.makedirs(path, exist_ok=True)

def vision_loaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    resize: int | tuple[int, int] | None = None,
) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    A unified torchvision vision dataset loader (train/test).

    Supported datasets:
      - "mnist"
      - "fashion_mnist" (also accepts "fashionmnist", "fmnist")
      - "cifar10"

    Notes:
      - We normalize to [0,1] via ToTensor(), then binarize (if needed) in the training loop.
      - Resize (if provided) is applied BEFORE ToTensor() (i.e. on PIL images).
    """
    os.makedirs(data_dir, exist_ok=True)

    # Basic image transform: optional resize (on PIL) -> ToTensor (float32 in [0,1])
    transform_list = [transforms.ToTensor()]
    if resize is not None:
        transform_list.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform_list)

    key = dataset.strip().lower().replace("-", "").replace("_", "")
    if key == "mnist":
        ds_cls = torchvision.datasets.MNIST
    elif key in {"fashionmnist", "fmnist"}:
        ds_cls = torchvision.datasets.FashionMNIST
    elif key == "cifar10":
        ds_cls = torchvision.datasets.CIFAR10
    else:
        raise ValueError(
            f"Unknown dataset={dataset!r}. Supported: 'mnist', 'fashion_mnist' (or 'fmnist'), 'cifar10'."
        )

    train_ds = ds_cls(root=data_dir, train=True, transform=transform, download=True)
    test_ds = ds_cls(root=data_dir, train=False, transform=transform, download=True)

    pin = pin_memory and torch.cuda.is_available()
    persistent = num_workers > 0
    train_iter = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=persistent,
    )
    test_iter = data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=persistent,
    )
    return train_iter, test_iter


# -----------------------------------------
# Device
# -----------------------------------------
def set_seed(seed: int | None = None) -> None:
    # Use a high-entropy / high-resolution seed when not provided.
    # NOTE: int(time.time()) has only 1s resolution and can collide easily.
    if seed is None:
        seed = int(torch.seed())
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is not strictly required; RBM uses randomness anyway.
    # If you want more determinism, uncomment:
    # torch.use_deterministic_algorithms(True)

def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the device to use. By default, automatically selects CUDA or CPU based on
    the current environment, or you can explicitly specify it via the argument.

    Args:
        device: Optional device specifier (e.g., "cuda", "cpu", or torch.device).

    Returns:
        torch.device: The selected device.
    """
    resolved = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {resolved} (CUDA available: {torch.cuda.is_available()})")
    # Speedups for modern NVIDIA GPUs (Ampere+ / Ada like RTX 4070 Ti):
    # Enable TF32 (keeps float32 API, uses TF32 internally for matmul/conv where appropriate).
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Let PyTorch choose higher-performance matmul kernels (often TF32 on CUDA).
        torch.set_float32_matmul_precision("high")
    return resolved


# -----------------------------------------
# Visualization
# -----------------------------------------
def use_svg_display():
    """Use SVG display in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size in Matplotlib."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes in Matplotlib."""
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
    """Plot the data in Matplotlib."""
    if legend is None:
        legend = []
    
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    
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


# -----------------------------------------
# Timer
# -----------------------------------------
class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self) -> float:
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self) -> float:
        """Return the sum of time."""
        return sum(self.times)

    def format_time(self, seconds: float | None = None, precision: int = 1) -> str:
        """Format given seconds or the accumulated time using ``_time_str``."""
        total = self.sum() if seconds is None else seconds
        return _time_str(total, precision=precision)

def _time_str(seconds: float, precision: int = 1) -> str:
    """Return a formatted string given seconds in non-zero units format (d, h, min and sec)."""
    total = seconds

    # Decompose total seconds into days, hours, minutes and seconds
    remainder = total
    days = int(remainder // 86400)
    remainder -= days * 86400
    hours = int(remainder // 3600)
    remainder -= hours * 3600
    minutes = int(remainder // 60)
    secs = remainder - minutes * 60

    def _fmt_secs(value: float) -> str:
        return f"{value:.{precision}f}"

    # Build parts starting from the highest non-zero unit
    parts: list[str] = []
    if days > 0:
        parts.append(f"{days} d")
        parts.append(f"{hours} h")
        parts.append(f"{minutes} min")
        parts.append(f"{_fmt_secs(secs)} sec")
    elif hours > 0:
        parts.append(f"{hours} h")
        parts.append(f"{minutes} min")
        parts.append(f"{_fmt_secs(secs)} sec")
    elif minutes > 0:
        parts.append(f"{minutes} min")
        parts.append(f"{_fmt_secs(secs)} sec")
    else:
        # All higher units are zero, only show seconds
        parts.append(f"{_fmt_secs(secs)} sec")

    return " ".join(parts)


# -----------------------------------------
# Machine Learning Basics
# -----------------------------------------
def synthetic_data(w, b, num_examples):
    """
    Generate y = Xw + b + noise (gaussian noise, N(0, 0.01)).

    Args:
        w: the weights
        b: the bias
        num_examples: the number of examples
    Returns:
        A tuple of input features and the true values: (X, y)
        - X: the input features (num_examples, len(w))
        - y: the true values (num_examples, 1)
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def linreg(X, w, b):
    """
    The linear regression model.
    
    Args:
        X: the input features
        w: the weights
        b: the bias
    Returns:
        the predicted value (num_examples, 1)
    """
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """
    Squared loss.
    
    Args:
        y_hat: the predicted value
        y: the true value
    Returns:
        the squared loss (multiply by 0.5 for convenience)
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr):
    """
    Batch stochastic gradient descent.
    
    Args:
        params: the parameters
        lr: the learning rate
    """
    with torch.no_grad():  # no need to track gradient, just update the parameters
        for param in params:  # traverse the parameters
            param -= lr * param.grad  # update the parameter
            param.grad.zero_()        # reset the gradient


# -----------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------
def load_array(data_arrays, batch_size, is_train=True):
    """
    Construct a PyTorch data iterator.
    
    Args:
        data_arrays: a tuple of tensors
        batch_size: the batch size
        is_train: whether to shuffle the data
    Returns:
        the data iterator (DataLoader)
    """
    dataset = data.TensorDataset(*data_arrays)  # input: a tuple of tensors
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_fashion_mnist_labels(labels):
    """
    Return text labels for the Fashion-MNIST dataset.
    
    Args:
        labels: the labels
    Returns:
        the text labels (1D list [text_label])
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
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
    """
    Use process_count processes to read the data (for multithreading).
    
    Args:
        process_count: the number of processes to use (Default: 16)
    Returns:
        the number of processes to use (Default: 16)
    """
    return process_count

def load_data_fashion_mnist(batch_size, resize=None, process_count=16):
    """
    Download the Fashion-MNIST dataset and then load it into memory (for multithreading).
    
    Args:
        batch_size: the batch size
        resize: the resize (Default: None)
        process_count: the number of processes to use (Default: 16)
    Returns:
        A tuple of training data iterator and test data iterator: (train_iter, test_iter)
        - train_iter: the training data iterator (DataLoader)
        - test_iter: the test data iterator (DataLoader)
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
        root=f"{base_path}/data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=f"{base_path}/data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers(process_count)),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers(process_count)))


# -----------------------------------------
# Model Evaluation
# -----------------------------------------
def accuracy(y_hat, y):
    """
    Compute the number of correct predictions.

    Args:
        y_hat: the predicted value (batch_size, num_outputs) or (batch_size,)
        y: the true value (batch_size,)
    Returns:
        the number of correct predictions
    """
    # len(y_hat.shape) > 1 and y_hat.shape[1] > 1: means y_hat is a matrix
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat_idx = y_hat.argmax(axis=1)  # y_hat_idx: (batch_size,), obtain the index of the maximum probability
        cmp = y_hat_idx.type(y.dtype) == y  # cmp: (batch_size,), True or False
        return float(cmp.type(y.dtype).sum())  # sum the True values, convert to float
    return 0.0  # return 0.0 if y_hat is not a matrix

def evaluate_accuracy(net, data_iter):
    """
    Compute the accuracy for a model on a dataset.

    Args:
        net: the network
        data_iter: the data iterator
    Returns:
        the accuracy of the model
    """
    if isinstance(net, torch.nn.Module): # Determine whether net is an instance of torch.nn.Module
        net.eval()  # set the model to evaluation mode
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # metric.add(correct predictions, total predictions)
    return metric[0] / metric[1]  # return the accuracy

class Accumulator:
    """
    Accumulate sum over n variables (for multiple metrics).
    
    Args:
        n: the number of variables, initialize the data with n zeros
    """
    def __init__(self, n):
        self.data = [0.0] * n  # initialize the data with n zeros

    def add(self, *args):
        """Add the arguments to the data."""
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
        """Reset the data to n zeros."""
        self.data = [0.0] * len(self.data)  # reset the data to n zeros

    def __getitem__(self, idx):  # double underscores getitem: get the data at the index
        """Get the data at the index."""
        return self.data[idx]  # return the data at the index

class Animator:
    """Animate data curves using Matplotlib interactive mode."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.ion()
        self.fig.show()
        self._closed = False

    def add(self, x, y):
        """Add the data to the animator."""
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
        plt.pause(0.001)


# -----------------------------------------
# Model Training and Prediction
# -----------------------------------------
def train_epoch_ch3(net, train_iter, loss, updater): 
    """
    Train the model for one epoch.

    Args:
        net: the network
        train_iter: the training data iterator
        loss: the loss function
        updater: the optimizer
    Returns:
        A tuple: (average_loss, average_accuracy)
    """
    if isinstance(net, torch.nn.Module): # Determine whether net is an instance of torch.nn.Module
        net.train()  # set the model to training mode
    metric = Accumulator(3)  # l.sum(), correct predictions, num_samples
    for X, y in train_iter:
        # compute the gradient and update the parameters
        y_hat = net(X)      # y_hat: the predicted value (batch_size, num_outputs) or (batch_size,)
        l = loss(y_hat, y)  # l: the loss (batch_size,); y: the true value (batch_size,)
        if isinstance(updater, torch.optim.Optimizer):  # Determine whether updater is an instance of torch.optim.Optimizer
            # use the built-in optimizer and loss function in PyTorch
            updater.zero_grad()  # reset the gradient
            l.mean().backward()  # compute the gradient
            updater.step()       # update the parameters
        else:
            # use the custom optimizer and loss function
            l.mean().backward()  # compute the gradient
            updater()  # update the parameters
        metric.add(l.detach().sum().item(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    Train a model with multiple epochs.
    
    Args:
        net: the network
        train_iter: the training data iterator
        test_iter: the test data iterator
        loss: the loss function
        num_epochs: the number of epochs
        updater: the optimizer
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
    """
    Predict labels.
    
    Args:
        net: the network
        test_iter: the test data iterator
        n: the number of images to predict (Default: 6)
    """
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

def evaluate_loss(net, data_iter, loss):
    """
    Evaluate the model's loss on the given dataset.
    
    Args:
        net: the model
        data_iter: the data iterator
        loss: the loss function
    Returns:
        the average loss
    """
    metric = Accumulator(2)  # loss_sum, num_samples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


# -----------------------------------------
# Data Download and Extraction
# -----------------------------------------
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=None):
    """
    Download a file inserted into DATA_HUB, and return the local filename.
    
    Args:
        name: the name of the file to download
        cache_dir: the folder to cache the file (Default: None)
    Returns:
        the path to the downloaded file
    """
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
    """
    Download and extract a zip/tar file.
    
    Args:
        name: the name of the file to download
        folder: the folder to extract the file to (Default: None)
    Returns:
        the path to the extracted file
    """
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
    """Download all files in DATA_HUB"""
    for name in DATA_HUB:
        download(name)

DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

def try_gpu(i=0):
    """
    Return gpu(i) if exists, otherwise return cpu.
    
    Args:
        i: the index of the GPU (Default: 0)
    Returns:
        The GPU(i) if exists, otherwise return cpu
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """
    Return all available GPUs, or [cpu,] if no GPU exists.
    
    Returns:
        A list of all available GPUs, or [cpu,] if no GPU exists
    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# -----------------------------------------
# Convolutional Neural Network
# -----------------------------------------
def corr2d(X, K):
    """
    Calculate 2D cross-correlation.
    
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
    """
    Evaluate the accuracy of the model on the given dataset using GPU.
    
    Args:
        net: the model
        data_iter: the data iterator
        device: the device to use (Default: None)
    Returns:
        The accuracy of the model on the given dataset using GPU
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            # Get the device of the first parameter of the net
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT fine-tuning (to be introduced later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]  # Return the accuracy

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device, net_name=None):
    """
    Train the model using GPU.
    
    Args:
        net: the network
        train_iter: the training data iterator
        test_iter: the testing data iterator
        num_epochs: the number of epochs
        lr: the learning rate
        device: the device to use
        net_name: the name of the network (Default: None)
    """
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], 
                                 legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)

    if net_name is not None:
        print(f'\n{net_name} is training on {device} ...')
    else:
        print(f'\nTraining on {device} ...')

    for epoch in range(num_epochs):
        # (training) loss_sum, total number of correct predictions, total number of samples
        metric = Accumulator(3)
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
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec, '
          f'total time: {timer.format_time()}')

class Residual(nn.Module):
    """
    Residual block.
    
    Args:
        input_channels: the number of input channels
        num_channels: the number of output channels
        use_1x1conv: whether to use 1x1 convolution (Default: False)
        strides: the stride of the convolution (Default: 1)
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        """
        Forward pass.
        
        Args:
            X: the input tensor (batch_size, input_channels, height, width)
        Returns:
            The output tensor (batch_size, num_channels, height, width)
        """
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


# -----------------------------------------
# Text Preprocessing
# -----------------------------------------
DATA_HUB['time_machine'] = (
    DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """
    Load the Time Machine dataset into a list of text lines.
    
    Returns:
        A list of text lines (1D list [text line])
    """
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    """
    Split text lines into word or character tokens.
    
    Args:
        lines: a list of text lines
        token: the type of token (Default: 'word')
    Returns:
        A list of tokens (2D list [([token])])
        - If token is 'word', return a list of words
        - If token is 'char', return a list of characters
    """
    if token == 'word':
        return [line.split() for line in lines]     # 2D list [([word])]
    elif token == 'char':
        return [list(line) for line in lines]  # 2D list [([char])]
    else:
        print('Error: unknown token type: ' + token)

class Vocab:
    """
    Text vocabulary.
    
    Args:
        tokens: a list of tokens (1D list [token] or 2D list [[token]]) (Default: None)
        min_freq: the minimum frequency of the token (less than min_freq will be ignored) (Default: 0)
        reserved_tokens: a list of reserved tokens (Default: None)
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort by frequency
        counter = count_corpus(tokens)
        # Sorted by frequency in descending order, type: list of tuples [(token, frequency)]
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index of the unknown token is 0, and the reserved tokens are prepended
        # 1D list [token], the index is the position of the token in the list
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 2D dictionary {token: index}
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """Return the number of tokens."""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        Get the index of the token (if it exists, otherwise return the index of the unknown token (0)).

        Args:
            tokens: a single token or a list of tokens (1D list [token])
        Returns:
            - If tokens is a single token, return the index of the token
            - If tokens is a list of tokens, return a list of indices of the tokens
        """
        if not isinstance(tokens, (list, tuple)):  # if tokens is a single token
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        Convert indices to tokens.
        
        Args:
            indices: a single index or a list of indices (1D list [index])
        Returns:
            - If indices is a single index, return the token
            - If indices is a list of indices, return a list of tokens
        """
        if not isinstance(indices, (list, tuple)):  # if indices is a single index
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """(@property) The index of the unknown token is 0."""
        return 0

    @property
    def token_freqs(self):
        """(@property) Return a list of tuples sorted by token frequencies: [(token, frequency)]."""
        return self._token_freqs

def count_corpus(tokens):
    """
    Count token frequencies.
    
    Args:
        tokens: a list of tokens (1D list [token] or 2D list [[token]])
    Returns:
        A counter of tokens (collections.Counter)
    """
    # Here tokens is a 1D list or a 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a 2D list of tokens into a single list
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
    """
    Load the Time Machine dataset into a list of corpus indices and a vocabulary object.
    
    Args:
        max_tokens: the maximum number of tokens to load (Default: -1, unlimited)
    Returns:
        A tuple of corpus indices and vocabulary: (corpus, vocab)
        - corpus: a list of corpus indices corresponding to the tokens (1D list [index])
        - vocab: a vocabulary object (Vocab)
    """
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Because each line in the Time Machine dataset is not necessarily a sentence or a paragraph,
    # we flatten all text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]  # hidden flatten: 1D list [index]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    Generate a batch of subsequences using random sampling.
    
    Args:
        corpus: a list of corpus indices corresponding to the tokens (1D list [index])
        batch_size: the batch size
        num_steps: the number of steps
    Returns:
        A batch of subsequences as tensors: (X, Y)
        - X: a tensor of shape (batch_size, num_steps)
        - Y: a tensor of shape (batch_size, num_steps)
    """
    # Partition the sequence starting from a random offset, whose range includes num_steps - 1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 because we need to account for the labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # Initial indices of subsequences with length num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # During iteration with random sampling, subsequences from two adjacent random batches 
    # are not necessarily adjacent in the original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return the subsequence of length num_steps starting from index pos
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, initial_indices contains the random initial indices of subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # Yield the subsequences as tensors: (batch_size, num_steps)
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    Generate a batch of subsequences using sequential partitioning.
    
    Args:
        corpus: a list of corpus indices corresponding to the tokens (1D list [index])
        batch_size: the batch size
        num_steps: the number of steps
    Returns:
        A batch of subsequences as tensors: (X, Y)
        - X: a tensor of shape (batch_size, num_steps)
        - Y: a tensor of shape (batch_size, num_steps)
    """
    # Split the sequence starting from a random offset
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        # Yield the subsequences as tensors: (batch_size, num_steps)
        yield X, Y


# -----------------------------------------
# Language Models and Dataset
# -----------------------------------------
class SeqDataLoader:
    """
    Iterator for loading sequence data.
    
    Args:
        batch_size: the batch size
        num_steps: the number of steps
        use_random_iter: whether to use random sampling
        max_tokens: the maximum number of tokens to load
    """
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        """Return the iterator of the data."""
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """
    Load the Time Machine dataset into an iterator and a vocabulary object.
    
    Args:
        batch_size: the batch size
        num_steps: the number of steps
        use_random_iter: whether to use random sampling (Default: False)
        max_tokens: the maximum number of tokens to load (Default: 10000)
    Returns:
        A tuple of an iterator and a vocabulary object: (data_iter, data_iter.vocab)
        - data_iter: an iterator for loading sequence data (SeqDataLoader)
        - data_iter.vocab: a vocabulary object (Vocab)
    """
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# -----------------------------------------
# Recurrent Neural Network
# -----------------------------------------
class RNNModelScratch:
    """
    Recurrent neural network model implemented from scratch (for Chapter 8).
    
    Args:
        vocab_size: the size of the vocabulary
        num_hiddens: the number of hidden units
        device: the device to use
        get_params: a function to get the parameters
        init_state: a function to initialize the state
        forward_fn: a function to forward the model
    """
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """
        Forward pass (allow the object to be called like a function).
        
        Args:
            X: the input features (batch_size, num_steps)
            state: the state (num_layers, batch_size, num_hiddens) if LSTM, otherwise (batch_size, num_hiddens)
        """
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # X: (num_steps, batch_size, vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """
        Initialize the state.
        
        Args:
            batch_size: the batch size
            device: the device to use
        """
        return self.init_state(batch_size, self.num_hiddens, device)

def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    Generate new characters following the given prefix.
    
    Args:
        prefix: a string of the prefix
        num_preds: the number of predictions
        net: the network
        vocab: a vocabulary object
        device: the device to use
    Returns:
        A string of the generated characters
    """
    state = net.begin_state(batch_size=1, device=device)  # (1, num_hiddens)
    outputs = [vocab[prefix[0]]]  # [index] of the prefix and generated characters
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # [last index] -> (1, 1): (num_steps, batch_size)
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)  # warm-up state only
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict for num_preds steps
        y, state = net(get_input(), state)  # (1 * 1, vocab_size), (1, num_hiddens)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """
    Clip gradients (global norm clipping).
    
    Args:
        net: the network
        theta: the threshold of the gradient
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter, timer):
    """
    Train the network for one epoch (see Chapter 8 for the definition).
    
    Args:
        net: the network
        train_iter: the training data iterator
        loss: the loss function
        updater: the optimizer
        device: the device to use
        use_random_iter: whether to use random sampling
        timer: the timer instance (Timer)
    Returns:
        A tuple of perplexity and the number of tokens: (ppl, num_tokens)
        - ppl: the perplexity
        - num_tokens: the number of tokens (num_steps * batch_size)
    """
    state = None
    metric = Accumulator(2)  # l.sum(), num_tokens (num_steps * batch_size)
    for X, Y in train_iter:
        timer.start()
        if state is None or use_random_iter:
            # Initialize state during the first iteration or when using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # detach the state from the computation graph, avoid gradient explosion
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # For nn.GRU, state is a tensor
                state.detach_()
            else:
                # For nn.LSTM or for our scratch implementation, state is a tuple of tensors
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)  # Flatten over num_steps (num_steps * batch_size)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)  # (num_steps * batch_size, vocab_size), (batch_size, num_hiddens)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater()
        metric.add(l * y.numel(), y.numel())
        timer.stop()
    return math.exp(metric[0] / metric[1]), metric[1]  # perplexity, num_tokens

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False, net_name=None):
    """
    Train the model (see Chapter 8 for the definition).
    
    Args:
        net: the network
        train_iter: the training data iterator
        vocab: a vocabulary object
        lr: the learning rate
        num_epochs: the number of epochs
        device: the device to use
        use_random_iter: whether to use random sampling (Default: False)
        net_name: the name of the network (Default: None)
    Returns:
        None: prints the perplexity, speed, and total time, and the generated characters
    """
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialization
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda: sgd(net.params, lr)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    if net_name is not None:
        print(f'\n{net_name} is training on {str(device)} ...')
    else:
        print(f'\nTraining on {str(device)} ...')
    
    timer, total_tokens = Timer(), 0.0
    # Training and prediction
    for epoch in range(num_epochs):
        ppl, num_tokens = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter, timer)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
        total_tokens += num_tokens
    print(f'Perplexity {ppl:.1f}, {total_tokens / timer.sum():.1f} tokens/sec, total time: {timer.format_time()}')
    print(predict('time traveller'))
    print(predict('traveller'))

class RNNModel(nn.Module):
    """
    Recurrent neural network model (for Chapter 8).
    
    Args:
        rnn_layer: the RNN layer
        vocab_size: the size of the vocabulary
        **kwargs: additional arguments
    """
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (introduced later), 
        # num_directions should be 2; otherwise it should be 1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        """
        Forward pass.
        
        Args:
            inputs: the input features (batch_size, num_steps)
            state: the state (num_directions * num_layers, batch_size, num_hiddens)
        Returns:
            A tuple of the output and the state: (output, state)
            - output: the output (num_steps * batch_size, vocab_size)
            - state: the state (num_directions * num_layers, batch_size, num_hiddens)
        """
        X = F.one_hot(inputs.T.long(), self.vocab_size)  # X: (num_steps, batch_size, vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)  # Y: (num_steps, batch_size, num_hiddens * num_directions)
        # First reshape Y to (num_steps * batch_size, num_hiddens * num_directions)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        """
        Initialize the state.
        
        Args:
            device: the device to use
            batch_size: the batch size (Default: 1)
        Returns:
            A tuple of the state: (H, C) if LSTM, otherwise (H,)
            - H: (num_directions * num_layers, batch_size, num_hiddens)
            - C: (num_directions * num_layers, batch_size, num_hiddens) if LSTM, otherwise None
        """
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU uses a tensor (H,) as its hidden state
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM uses a tuple of tensors (H, C) as its hidden state
            return (torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


# -----------------------------------------
# Language Models and Dataset
# -----------------------------------------
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():  # neural machine translation dataset
    """
    Load the 'English-French' dataset into a string of text.
        - The first column is the source language (English)
        - The second column is the target language (French)

    Returns:
        text (str): the raw text of the dataset
    """
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    """Preprocess the 'English-French' dataset"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking spaces with spaces
    # Convert uppercase letters to lowercase letters
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert spaces between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """
    Tokenize the 'English-French' dataset.
    
    Args:
        text (str): the raw text of the dataset
        num_examples (int): the number of examples (lines) to tokenize (Default: None)
    Returns:
        source: the source language tokens (2D list [[token]])
        target: the target language tokens (2D list [[token]])
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot a histogram of list length pairs"""
    set_figsize()
    _, _, patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)

def truncate_pad(line, num_steps, padding_token):
    """
    Truncate or pad text sequences.
    
    Args:
        line: the text sequence
        num_steps: the number of steps (tokens)
        padding_token: the token to pad
    Returns:
        The truncated or padded text sequence with length num_steps
    """
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def build_array_nmt(lines, vocab, num_steps):
    """
    Convert machine translation text sequences into batches.
    
    Args:
        lines: the text sequences (2D list [[token]])
        vocab: the vocabulary (Vocab)
        num_steps: the number of steps (tokens)
    Returns:
        A tuple of tensors: (array, valid_len)
        - array: the indices array of tokenized text sequences (2D tensor [[index]])  (num_lines, num_steps)
        - valid_len: the valid length of each text sequence (1D tensor [length]) (num_lines)
    """
    lines = [vocab[l] for l in lines]              # convert to 2D list of indices [[index]]
    lines = [l + [vocab['<eos>']] for l in lines]  # add <eos> token index to the end of each line
    array = torch.tensor([truncate_pad(            # truncate or pad each line to num_steps
        l, num_steps, vocab['<pad>']) for l in lines])   # return a 2D tensor [[index]] (num_lines, num_steps)
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)  # the valid length (1D tensor [length]) (num_lines)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """
    Return the iterator and vocabularies for the translation dataset.
    
    Args:
        batch_size: the batch size
        num_steps: the number of steps (tokens)
        num_examples: the number of examples (lines) to load (Default: 600)
    Returns:
        A tuple of: (data_iter, src_vocab, tgt_vocab)
        - data_iter: the data iterator (DataLoader)
        - src_vocab: the source vocabulary (Vocab)
        - tgt_vocab: the target vocabulary (Vocab)
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    reserved_tokens=['<pad>', '<bos>', '<eos>']
    # Build source and target vocabularies
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=reserved_tokens)  # source vocabulary
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=reserved_tokens)  # target vocabulary
    # Build source and target arrays by truncating or padding
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # Load data into data loader
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# -----------------------------------------
# Encoder-Decoder Architecture
# -----------------------------------------
class Encoder(nn.Module):
    """Basic encoder interface for the encoder-decoder architecture"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):  # encode the input X
        raise NotImplementedError  # to be implemented by subclass

class Decoder(nn.Module):
    """Basic decoder interface for the encoder-decoder architecture"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):  # initialize the decoder state based on the encoder outputs
        raise NotImplementedError  # to be implemented by subclass

    def forward(self, X, state):  # decode the decoder input X based on the decoder state
        raise NotImplementedError  # to be implemented by subclass

class EncoderDecoder(nn.Module):
    """
    Base class for the encoder-decoder architecture.
    
    Args:
        encoder: a encoder instance (Encoder)
        decoder: a decoder instance (Decoder)
        **kwargs: additional arguments
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """
        Forward pass for the encoder-decoder architecture.
        
        Args:
            enc_X: the encoder input
            dec_X: the decoder input
            *args: additional arguments
        Returns:
            The decoder output
        """
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# -----------------------------------------
# Sequence-to-Sequence Learning
# -----------------------------------------
class Seq2SeqEncoder(Encoder):
    """Recurrent neural network encoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # Input 'X' shape: (batch_size, num_steps)
        X = self.embedding(X)   # Output 'X' shape: (batch_size, num_steps, embed_size)
        # In recurrent neural network models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)  # Permute the dimensions of 'X' to (num_steps, batch_size, embed_size)
        output, state = self.rnn(X)
        # if state is not specified, it defaults to zeros
        # Shape of output: (num_steps, batch_size, num_hiddens)
        # Shape of state: (num_layers, batch_size, num_hiddens)
        return output, state

def sequence_mask(X, valid_len, value=0):
    """
    Mask irrelevant items in sequences.

    Args:
        X: the input sequence (batch_size, num_steps)
        valid_len: the valid length of the input sequence (batch_size,)
        value: the value to fill the irrelevant items with (Default: 0)
    Returns:
        The masked input sequence (batch_size, num_steps),
        where the irrelevant items are filled with the value.
    """
    maxlen = X.size(1)  # num_steps
    # Shape of mask: (1, num_steps) < (batch_size, 1), 
    # resulting in a boolean tensor of shape (batch_size, num_steps)
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value  # Fill the irrelevant items with the value, ~mask is the negation of mask
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """Softmax cross-entropy loss with masking."""
    # Shape of pred: (batch_size, num_steps, vocab_size)
    # Shape of label: (batch_size, num_steps)
    # Shape of valid_len: (batch_size,)
    def forward(self, pred, label, valid_len):  # Rewrite the forward method of the base class nn.CrossEntropyLoss
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'  # Set the reduction method to 'none' to return the loss for each item
        # Permute the dimensions of 'pred' to (batch_size, vocab_size, num_steps) 
        # to meet the needs of the base class nn.CrossEntropyLoss (input shape: (N, C, ...))
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)  # Call the forward method of the base class nn.CrossEntropyLoss
        # Shape of unweighted_loss: (batch_size, num_steps)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)  # Shape of weighted_loss: (batch_size,)
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device, net_name=None):
    """Train a sequence-to-sequence model."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    if net_name is not None:
        print(f'\n{net_name} is training on {device} ...')
    else:
        print(f'\nTraining on {device} ...')
    
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    timer = Timer()
    total_tokens = 0.0
    for epoch in range(num_epochs):
        metric = Accumulator(2)  # l.sum(), num_tokens
        for batch in data_iter:
            timer.start()
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)  # Shape of bos: (batch_size, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing (Shape of dec_input: (batch_size, num_steps + 1))
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # Perform backpropagation using the scalar loss
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
            timer.stop()
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
        total_tokens += metric[1]
    tokens_per_sec = total_tokens / timer.sum()
    print(f'loss {metric[0] / metric[1]:.3f}, {tokens_per_sec:.1f} '
          f'tokens/sec, total time: {timer.format_time()}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Prediction for sequence-to-sequence model."""
    # Set net to evaluation mode during prediction
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add a batch dimension (dim=0) using unsqueeze to make the shape of enc_X: (1, num_steps)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add a batch dimension (dim=0) using unsqueeze to make the shape of dec_X: (1, 1)
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)  # Shape of Y: (1, 1, vocab_size)
        # Use the token with the highest predicted probability as the decoder input at the next time step
        dec_X = Y.argmax(dim=2)  # Shape of dec_X: (1, 1) (greedy search)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()  # .item() to get the scalar value
        # Save attention weights (to be discussed later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, stop generating the output sequence
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):
    """Compute BLEU (Bilingual Evaluation Understudy)."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                # Prevent the same label n-gram sub-sequence from being matched multiple times
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# -----------------------------------------
# Attention Weights Visualization
# -----------------------------------------
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """
    Show heatmaps of matrices.
    Args:
        matrices: (number of rows for display, number of columns for display, number of queries, number of keys)
        xlabel: x-axis label
        ylabel: y-axis label
        titles: titles for each subplot
        figsize: figure size
        cmap: color map
    """
    use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

# -----------------------------------------
# Attention Scoring Function
# -----------------------------------------
def masked_softmax(X, valid_lens):
    """Perform softmax by masking elements on the last axis."""
    # X: 3D tensorv (B, N, H), valid_lens: 1D or 2D tensor (B,) or (B, N)
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Replace masked elements with a very large negative value in the last axis, 
        # so their softmax outputs are 0.
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # keys shape: (batch_size, num_key_value_pairs, key_size)
        # queries shape: (batch_size, num_queries, query_size)
        # key_size: the dimension of the key vectors
        # query_size: the dimension of the query vectors
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion:
        # queries shape: (batch_size, num_queries, 1, num_hiddens)
        # keys shape: (batch_size, 1, num_key_value_pairs, num_hiddens)
        # Use broadcasting for summation. 
        # features shape: (batch_size, num_queries, num_key_value_pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v has a single output, so remove the last dimension.
        # scores shape: (batch_size, num_queries, num_key_value_pairs)
        scores = self.w_v(features).squeeze(-1)
        # attention_weights shape: (batch_size, num_queries, num_key_value_pairs)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values shape: (batch_size, num_key_value_pairs, value_size)
        # output shape: (batch_size, num_queries, value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    """Scaled dot-product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries shape: (batch_size, num_queries, d)
        # keys shape: (batch_size, num_key_value_pairs, d)
        # values shape: (batch_size, num_key_value_pairs, value_size)
        # valid_lens shape: (batch_size,) or (batch_size, num_queries)
        d = queries.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of keys.
        # scores shape: (batch_size, num_queries, num_key_value_pairs)
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        # attention_weights shape: (batch_size, num_queries, num_key_value_pairs)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # output shape: (batch_size, num_queries, value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)

class AttentionDecoder(Decoder):
    """Basic interface for an attention-based decoder."""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        # p_q = p_k = p_v = num_hiddens/num_heads in one head
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, values shape:
        # (batch_size, num_queries or num_key_value_pairs, num_hiddens)
        # valid_lens shape:
        # (batch_size,) or (batch_size, num_queries)
        # After projection, queries/keys/values shape:
        # (batch_size*num_heads, num_queries or num_key_value_pairs,
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, repeat each item (scalar or vector) num_heads times.
            # And then copy the second item, etc.
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output shape: (batch_size*num_heads, num_queries,
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat shape: (batch_size, num_queries, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    """Transpose for parallel multi-head attention computation."""
    # Input X shape: (batch_size, num_queries or num_key_value_pairs, num_hiddens)
    # Output X shape: (batch_size, num_queries or num_key_value_pairs, num_heads,
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Output X shape: (batch_size, num_heads, num_queries or num_key_value_pairs,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # Final output shape: (batch_size*num_heads, num_queries or num_key_value_pairs,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """Reverse the transpose_qkv operation."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P.
        self.P = torch.zeros((1, max_len, num_hiddens))
        # X = position / 10000^(2j/d)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # sin(X) and cos(X) are alternating
        # '0::2' is the even indices, i.e., start from 0 and increment by 2
        # '1::2' is the odd indices, i.e., start from 1 and increment by 2
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """Layer normalization after residual connection."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        # normalized_shape defines the feature dimensions to normalize.
        # It must match the last N dimensions of the input tensor.
        # int -> normalize last dimension; list/tuple -> normalize last N dimensions.
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # X: residual connection input
        # Y: output of the layer before the residual connection
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # The output of each layer in the Transformer encoder has the same shape as its input.
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are in [-1, 1],
        # scale embeddings by sqrt(embedding_dim) and then add them.
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            # Forward-propagate the input X through the current EncoderBlock.
            # Since each EncoderBlock has exactly the same input and output shape—
            # both (batch, num_steps, num_hiddens)—we can overwrite X directly, 
            # enabling layer-by-layer stacking.
            X = blk(X, valid_lens)
            # Store the attention weights for visualization.
            # Shape of attention weights: 
            # (batch_size * num_heads, num_queries, num_key_value_pairs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
