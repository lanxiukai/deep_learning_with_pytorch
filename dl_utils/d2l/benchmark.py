from torch import nn

from dl_utils.d2l.cnn import Residual
from dl_utils.training.timing import Timer

class Benchmark:
    """For measuring running time"""
    def __init__(self, description='Done'):
        self.description = description
        self.timer = None

    def __enter__(self):
        # __enter__ is a dunder method that implements the context manager protocol;
        # it is automatically called when execution enters a `with` block,
        # allowing setup logic (here: starting the timer) to run before the block body
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        # __exit__ is a dunder method that implements the context manager protocol;
        # it is automatically called when execution exits a `with` block,
        # allowing cleanup logic (here: stopping the timer and printing the result) to run after the block body
        assert self.timer is not None
        print(f'{self.description}: {self.timer.stop():.4f} sec')


def split_batch(X, y, devices):
    """Split X and y across multiple devices"""
    # Ensure X and y have the same number of samples.
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))


def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # This model uses smaller kernel size, stride and padding, and removes the max pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
